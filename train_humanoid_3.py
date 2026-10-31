import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Initialize Genesis targeting your Mac's Metal GPU backend
gs.init(backend=gs.metal)

# --- THE ENVIRONMENT WORKSPACE ---
class HumanoidTrainEnv:
    def __init__(self, num_envs=64, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            show_viewer=False # Running headlessly optimizes M5 Max training speeds
        )
        
        self.scene.add_entity(gs.morphs.Plane())
        self.humanoid = self.scene.add_entity(
            gs.morphs.URDF(
                file="../unitree_ros/robots/h1_description/urdf/h1.urdf",
                pos=(0.0, 0.0, 1.05),
            )
        )
        
        self.scene.build(n_envs=self.num_envs, env_spacing=(4.0, 4.0))
        self.torso = self.humanoid.get_link("torso_link")
        self.num_actions = self.humanoid.n_dofs
        self.num_obs = self.num_actions * 2 + 3 + 1 
        self.standing_pose = self.humanoid.get_dofs_position()
        
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
        return torch.cat([q, v, torso_velocities, torso_height], dim=-1)

    def step(self, actions):
        target_positions = self.standing_pose + actions * 0.2
        self.humanoid.set_dofs_position(target_positions)
        
        # Sub-stepping physics inside the GPU core cluster
        for _ in range(5):
            self.scene.step()
            
        next_obs = self._get_observations()
        torso_velocities = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_heights = next_obs[:, -1]
        
        # PPO Reward Matrix
        forward_velocity = torso_velocities[:, 0]
        reward_forward = forward_velocity * 5.0 # High motivation to walk forward
        reward_survival = torch.where(torso_heights > 0.82, 1.0, -10.0) # Massive penalty for falling
        penalty_action = torch.sum(torch.square(actions), dim=-1) * -0.01
        
        total_rewards = reward_forward + reward_survival + penalty_action
        dones = (torso_heights < 0.55).to(torch.float32)
        
        return next_obs, total_rewards, dones

# --- THE PPO NEURAL NETWORK POLICY ---
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 256), # Switched to 256 neurons for better capacity
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

# --- EXECUTE POLICY OPTIMIZATION ---
if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    num_envs = 64 # Bumped to 64 parallel environments for better sample size
    
    env = HumanoidTrainEnv(num_envs=num_envs)
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    
    obs = env.reset()
    print(f"🔋 M5 Max Core Training Online. Optimizing 64 Humanoids across 500 epochs...")
    
    # Running for 500 epochs will give the network enough data to improve balance
    for update in range(501):
        action_means, values = policy(obs.detach())
        noise = torch.randn_like(action_means) * 0.15 # Dynamic noise for exploration
        actions = action_means + noise
        
        next_obs, rewards, dones = env.step(actions.clamp(-1.0, 1.0))
        
        _, next_values = policy(next_obs.detach())
        target_values = rewards.unsqueeze(1) + 0.99 * next_values * (1.0 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(values, target_values)
        
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # Reset any humanoids that have fallen over so they can try again
        torso_heights = next_obs[:, -1]
        fallen_mask = torso_heights < 0.65
        if torch.any(fallen_mask):
            # If a robot falls, reset its joint parameters back to standing
            env.humanoid.set_dofs_position(env.standing_pose)
            
        obs = next_obs
        
        if update % 50 == 0:
            avg_height = torch.mean(obs[:, -1]).item()
            print(f"🔄 Epoch {update:3d}/500 | Average Torso Height: {avg_height:.2f}m | Loss: {critic_loss.item():.4f}")
            
    # Save the finalized model file to disk
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"\n💾 SUCCESS: Humanoid brain saved to disk at -> {MODEL_PATH}")
    
    env.scene.destroy()
