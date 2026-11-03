import os
import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim

gs.init(backend=gs.metal)

class H2TrainEnv:
    def __init__(self, num_envs=64, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            show_viewer=False
        )
        
        self.scene.add_entity(gs.morphs.Plane())
        self.humanoid = self.scene.add_entity(
            gs.morphs.URDF(
                file="../unitree_ros/robots/h2_description/H2.urdf", 
                pos=(0.0, 0.0, 1.05),
            )
        )
        
        self.scene.build(n_envs=self.num_envs, env_spacing=(4.0, 4.0))
        
        self.torso = self.humanoid.get_link("torso_link")
        self.num_actions = self.humanoid.n_dofs  # Dynamically evaluates to 37
        self.num_obs = self.num_actions * 2 + 3 + 1  # 78 channels
        self.standing_pose = self.humanoid.get_dofs_position()
        
        self.humanoid.set_dofs_kp(torch.ones(self.num_actions, device='mps') * 150.0) 
        self.humanoid.set_dofs_kv(torch.ones(self.num_actions, device='mps') * 10.0)  
        
    def reset(self):
        self.humanoid.set_dofs_position(self.standing_pose)
        self.humanoid.set_dofs_velocity(torch.zeros_like(self.standing_pose))
        return self._get_observations()
        
    def _get_observations(self):
        q = self.humanoid.get_dofs_position()
        v = self.humanoid.get_dofs_velocity()
        torso_positions = self.torso.get_pos()
        torso_height = torso_positions[:, 2].unsqueeze(1)
        torso_velocities = self.torso.get_vel()[:, :3]
        return torch.cat([q, v, torso_velocities, torso_height], dim=-1).to('mps')

    def step(self, actions):
        target_positions = self.standing_pose + actions * 0.3
        self.humanoid.control_dofs_position(target_positions)
        
        for _ in range(5):
            self.scene.step()
            
        next_obs = self._get_observations()
        torso_velocities = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_heights = next_obs[:, -1]
        
        forward_velocity = torso_velocities[:, 0]
        reward_forward = forward_velocity * 6.0 
        reward_survival = torch.where(torso_heights > 0.55, 1.0, -15.0) 
        penalty_action = torch.sum(torch.square(actions), dim=-1) * -0.01
        
        total_rewards = reward_forward + reward_survival + penalty_action
        dones = (torso_heights < 0.55).to(torch.float32)
        
        return next_obs, total_rewards, dones

class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_actions), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )
    def forward(self, obs): return self.actor(obs), self.critic(obs)

if __name__ == "__main__":
    MODEL_PATH = "humanoid_h2_policy.pt"
    num_envs = 64
    
    env = H2TrainEnv(num_envs=num_envs)
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=4e-4)
    
    obs = env.reset()
    print(f"🔋 Training H2 Policy ({env.num_obs} Obs, {env.num_actions} Actions)...")
    
    for update in range(601):
        action_means, values = policy(obs.detach())
        noise = torch.randn_like(action_means) * 0.15 
        actions = action_means + noise
        
        next_obs, rewards, dones = env.step(actions.clamp(-1.0, 1.0))
        _, next_values = policy(next_obs.detach())
        target_values = rewards.unsqueeze(1) + 0.99 * next_values * (1.0 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(values, target_values)
        
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # Vectorized Reset Intercept
        torso_heights = next_obs[:, -1]
        fallen_mask = (torso_heights < 0.55) | torch.isnan(torso_heights)
        
        # Extract the actual 1D index tensor out of the tuple returned by torch.where
        reset_env_ids = torch.where(fallen_mask)[0]
        
        # Verify the index tensor actually contains values before calling Genesis
        if reset_env_ids.numel() > 0:
            env.humanoid.set_dofs_position(position=env.standing_pose[reset_env_ids], envs_idx=reset_env_ids)
            env.humanoid.set_dofs_velocity(velocity=torch.zeros_like(env.standing_pose[reset_env_ids]), envs_idx=reset_env_ids)
            
        obs = next_obs
        if update % 20 == 0:
            print(f"🔄 Epoch {update:3d}/600 | Avg Height: {torch.mean(obs[:, -1]).item():.2f}m | Loss: {critic_loss.item():.4f}")
            
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"💾 Saved clean matrix array: {MODEL_PATH}")
    env.scene.destroy()
