import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Initialize Genesis targeting your Mac's Metal GPU Performance backend
gs.init(backend=gs.metal)

class HumanoidTrainEnv:
    def __init__(self, num_envs=64, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            show_viewer=False # Running headlessly keeps training iterations as fast as possible
        )
        
        # Drop in the flat floor grid
        self.scene.add_entity(gs.morphs.Plane())
        
        # 🚨 THE CRITICAL CORE RESOLUTION:
        # We switch completely to Genesis's native, pre-verified MuJoCo XML humanoid layout.
        # This completely avoids the buggy external Unitree legacy text URDF parsing layers!
        self.humanoid = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/humanoid.xml",
                pos=(0.0, 0.0, 1.3), # Positions the clean humanoid cleanly above the plane
            )
        )
        
        # Build out the 64 parallel environments cleanly on your GPU cores
        self.scene.build(n_envs=self.num_envs, env_spacing=(4.0, 4.0))
        
        # Gather system degrees of freedom variables straight from the native asset parameters
        self.num_actions = self.humanoid.n_dofs
        self.num_obs = self.num_actions * 2 + 3 + 1 # Position metrics + Velocities + Torso state channels
        self.standing_pose = self.humanoid.get_dofs_position()
        
    def reset(self):
        self.humanoid.set_dofs_position(self.standing_pose)
        self.humanoid.set_dofs_velocity(torch.zeros_like(self.standing_pose))
        return self._get_observations()
        
    def _get_observations(self):
        q = self.humanoid.get_dofs_position()
        v = self.humanoid.get_dofs_velocity()
        
        # Natively extract position and tracking frames off the root base of the creature assembly
        torso_positions = self.humanoid.get_pos() # Shape: (num_envs, 3)
        torso_height = torso_positions[:, 2].unsqueeze(1)
        torso_velocities = self.humanoid.get_vel()[:, :3]
        
        return torch.cat([q, v, torso_velocities, torso_height], dim=-1)

    def step(self, actions):
        # Neural choices scale target continuous torque position settings
        target_positions = self.standing_pose + actions * 0.4
        self.humanoid.set_dofs_position(target_positions)
        
        # Execute sub-stepping to ensure kinematic continuity (5 steps @ 0.002 = 100Hz control loop)
        for _ in range(5):
            self.scene.step()
            
        next_obs = self._get_observations()
        torso_velocities = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_heights = next_obs[:, -1]
        
        # --- THE PPO REWARD ALIGNMENT MATRIX ---
        forward_velocity = torso_velocities[:, 0]
        reward_forward = forward_velocity * 6.0 # Strongly incentivize walking forward along X axis
        reward_survival = torch.where(torso_heights > 0.8, 1.0, -15.0) # Massive penalty if it drops
        penalty_action = torch.sum(torch.square(actions), dim=-1) * -0.01
        
        total_rewards = reward_forward + reward_survival + penalty_action
        dones = (torso_heights < 0.65).to(torch.float32)
        
        return next_obs, total_rewards, dones

# --- THE PPO REINFORCEMENT LEARNING NEURAL NETWORK ---
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 256),
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

if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    num_envs = 64
    
    env = HumanoidTrainEnv(num_envs=num_envs)
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=4e-4)
    
    obs = env.reset()
    # -- print("🔋 M5 Max Core Optimization Engine Online. Running 200 epochs on Native XML Humanoid...")
    print("🔋 M5 Max Core Optimization Engine Online. Running 600 epochs on Native XML Humanoid...")
    
    # -- for update in range(201):
    for update in range(601):
        action_means, values = policy(obs.detach())
        noise = torch.randn_like(action_means) * 0.15 # Exploratory variance
        actions = action_means + noise
        
        next_obs, rewards, dones = env.step(actions.clamp(-1.0, 1.0))
        
        _, next_values = policy(next_obs.detach())
        target_values = rewards.unsqueeze(1) + 0.99 * next_values * (1.0 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(values, target_values)
        
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # Automated Training Reset Intercept
        torso_heights = next_obs[:, -1]
        fallen_mask = torso_heights < 0.70
        if torch.any(fallen_mask):
            env.humanoid.set_dofs_position(env.standing_pose)
            
        obs = next_obs
        
        if update % 20 == 0:
            avg_height = torch.mean(obs[:, -1]).item()
            print(f"🔄 Epoch {update:3d}/200 | Average Torso Height: {avg_height:.2f}m | Loss: {critic_loss.item():.4f}")
            
    # Compile weight state matrices and write directly to disk storage
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"\n💾 SUCCESS: Optimized matrix file compiled cleanly at -> {MODEL_PATH}")
    env.scene.destroy()
