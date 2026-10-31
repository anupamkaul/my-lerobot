import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# --- THE ENVIRONMENT WORKSPACE ---
class HumanoidWalkEnv:
    def __init__(self, num_envs=32, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(4.0, -4.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=45,
            ),
            show_viewer=True
        )
        
        self.scene.add_entity(gs.morphs.Plane())
        self.humanoid = self.scene.add_entity(
            gs.morphs.URDF(
                file="../unitree_ros/robots/h1_description/urdf/h1.urdf",
                pos=(0.0, 0.0, 1.05),
            )
        )
        
        self.scene.build(n_envs=self.num_envs, env_spacing=(4.0, 4.0))
        
        # Core Fix: Mapping to the exact 'torso_link' parameter provided by the Genesis footprint
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
        
        for _ in range(5):
            self.scene.step()
            
        next_obs = self._get_observations()
        
        torso_velocities = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_heights = next_obs[:, -1]
        
        # PPO Reward: Reward moving forward on X, heavily penalize falling below 0.6m
        forward_velocity = torso_velocities[:, 0]
        reward_forward = forward_velocity * 4.0
        reward_survival = torch.where(torso_heights > 0.8, 0.5, -5.0)
        penalty_action = torch.sum(torch.square(actions), dim=-1) * -0.02
        
        total_rewards = reward_forward + reward_survival + penalty_action
        dones = (torso_heights < 0.55).to(torch.float32)
        
        return next_obs, total_rewards, dones

# --- THE PPO NEURAL NETWORK POLICY ---
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        # Actor network: Decides motor adjustments
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Tanh()
        )
        # Critic network: Computes reward values to measure state quality
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    num_envs = 32
    env = HumanoidWalkEnv(num_envs=num_envs)
    
    # Instantiate Policy and transfer it to the M5 Max GPU Core framework
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    obs = env.reset()
    print("🚀 PPO Engine Online! Commencing optimization loops...")
    
    for update in range(100): # Run 100 PPO update optimization cycles
        # Detach tensors to clear graph logs and manage memory space on the GPU
        action_means, values = policy(obs.detach())
        
        # Inject standard variance noise to drive natural exploration
        noise = torch.randn_like(action_means) * 0.1
        actions = action_means + noise
        
        # Pass actions to environments
        next_obs, rewards, dones = env.step(actions.clamp(-1.0, 1.0))
        
        # Basic Critic Loss Calculation (Training the Value Net)
        _, next_values = policy(next_obs.detach())
        target_values = rewards.unsqueeze(1) + 0.99 * next_values * (1.0 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(values, target_values)
        
        # Optimize neural weights natively via MPS
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        obs = next_obs
        
        # Periodic diagnostic logs evaluating Robot 0 performance
        if update % 10 == 0:
            current_height = obs[0, -1].item()
            current_vel = obs[0, env.num_actions*2].item()
            print(f"🔄 Epoch {update:2d} | Torso Height: {current_height:.2f}m | Speed: {current_vel:.3f} m/s | Loss: {critic_loss.item():.4f}")
