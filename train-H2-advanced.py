import os
import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# 1. Initialize Genesis targeting Mac's Metal Performance Shaders
gs.init(backend=gs.metal)

class H2AdvancedTrainEnv:
    def __init__(self, num_envs=64, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            show_viewer=False  # Headless execution for maximum optimization performance
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
        self.num_actions = self.humanoid.n_dofs  # 37 joints
        
        # Obs space layout: q (37) + v (37) + torso_lin_vel (3) + torso_ang_vel (3) + projected_gravity (3) + torso_height (1)
        self.num_obs = self.num_actions * 2 + 3 + 3 + 3 + 1  # 84 channels
        self.standing_pose = self.humanoid.get_dofs_position()
        
        # Configure joint metrics
        self.humanoid.set_dofs_kp(torch.ones(self.num_actions, device='mps') * 150.0) 
        self.humanoid.set_dofs_kv(torch.ones(self.num_actions, device='mps') * 10.0)  
        
    def reset(self, env_ids=None):
        if env_ids is None:
            # Full system manual reset initialization with slight noise for Domain Randomization
            init_pos = self.standing_pose.clone() + torch.randn((self.num_envs, self.num_actions), device='mps') * 0.02
            init_vel = torch.randn((self.num_envs, self.num_actions), device='mps') * 0.05
            self.humanoid.set_dofs_position(init_pos)
            self.humanoid.set_dofs_velocity(init_vel)
        else:
            # Target selective subset reset initialization
            init_pos = self.standing_pose[env_ids] + torch.randn((len(env_ids), self.num_actions), device='mps') * 0.02
            init_vel = torch.randn((len(env_ids), self.num_actions), device='mps') * 0.05
            self.humanoid.set_dofs_position(position=init_pos, envs_idx=env_ids)
            self.humanoid.set_dofs_velocity(velocity=init_vel, envs_idx=env_ids)
            
        return self._get_observations()
        
    """
    # the get_projected_gravity is not available here (only in OmniIsaacGym or certain custom RL wrappers) 

    # instead, we can compute the projected gravity vector manually by rotating the world vertical axis [0, 0, -1] 
    # into the robot's local coordinate frame using its torso orientation quaternion.

    def _get_observations(self):
        q = self.humanoid.get_dofs_position()
        v = self.humanoid.get_dofs_velocity()
        
        torso_positions = self.torso.get_pos()
        torso_height = torso_positions[:, 2].unsqueeze(1)
        torso_lin_vel = self.torso.get_vel()[:, :3]
        torso_ang_vel = self.torso.get_vel()[:, 3:6]
        
        # FIX 3: Extracted real local frame tilt variables to cure side-falling
        projected_gravity = self.humanoid.get_projected_gravity()
        
        return torch.cat([q, v, torso_lin_vel, torso_ang_vel, projected_gravity, torso_height], dim=-1).to('mps')
    """

    def _get_observations(self):
        q = self.humanoid.get_dofs_position()
        v = self.humanoid.get_dofs_velocity()
        
        torso_positions = self.torso.get_pos()
        torso_height = torso_positions[:, 2].unsqueeze(1)
        
        # --- THE REAL, NON-HALLUCINATED GENESIS API METHODS ---
        torso_lin_vel = self.torso.get_vel()  # Natively returns a (64, 3) tracking tensor
        torso_ang_vel = self.torso.get_ang()  # Natively returns a (64, 3) tracking tensor
        
        # --- MANUAL PROJECTED GRAVITY COMPOSITION ---
        torso_quat = self.torso.get_quat()
        w, x, y, z = torso_quat[:, 0], torso_quat[:, 1], torso_quat[:, 2], torso_quat[:, 3]
        
        gravity_x = 2 * (x * z - w * y)
        gravity_y = 2 * (y * z + w * x)
        gravity_z = 1.0 - 2 * (x**2 + y**2)
        
        projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1).to('mps')
        # --------------------------------------------
        
        # Exactly 84 elements: 37 + 37 + 3 + 3 + 3 + 1 = 84 channels cleanly fed into shared_trunk
        return torch.cat([q, v, torso_lin_vel, torso_ang_vel, projected_gravity, torso_height], dim=-1).to('mps')


    def step(self, actions):
        target_positions = self.standing_pose + actions * 0.3
        self.humanoid.control_dofs_position(target_positions)
        
        for _ in range(5):
            self.scene.step()
            
        next_obs = self._get_observations()
        
        # Slice states cleanly for reward computations
        torso_lin_vel = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_ang_vel = next_obs[:, self.num_actions * 2 + 3 : self.num_actions * 2 + 6]
        torso_heights = next_obs[:, -1]
        
        # --- ADVANCED STABILIZATION REWARD BALANCING ---
        # 1. Forward linear velocity along X while penalizing sideways drift
        reward_lin_forward = torch.exp(-torch.square(1.0 - torso_lin_vel[:, 0]) / 0.25)
        reward_lin_lateral = torch.exp(-torch.square(torso_lin_vel[:, 1]) / 0.1)
        
        # 2. Kill spinning / twisting roll, pitch, yaw motions
        reward_ang_stabilize = torch.exp(-torch.sum(torch.square(torso_ang_vel), dim=-1) / 0.2)
        
        # 3. Upright target tracking survival metric
        reward_upright = torch.exp(-torch.square(1.05 - torso_heights) / 0.05)
        
        # 4. Action smoothness regularization to stop high-frequency muscle flailing
        penalty_torque = torch.sum(torch.square(actions), dim=-1) * -0.01
        
        total_rewards = (reward_lin_forward * 0.4 + reward_lin_lateral * 0.2 + 
                         reward_ang_stabilize * 0.2 + reward_upright * 0.2) + penalty_torque
                         
        dones = (torso_heights < 0.55).to(torch.float32)
        
        return next_obs, total_rewards, dones

# --- THE ADVANCED STOCHASTIC ACTOR-CRITIC NETWORK PIPELINE ---
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        
        # Shared feature trunk layout allows Critic structures to speed up Actor policies
        self.shared_trunk = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
            nn.Tanh()
        )
        
        # Trainable log standard deviation array parameter for exploratory action scaling
        self.log_std = nn.Parameter(torch.ones(num_actions) * -1.0)
        
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        features = self.shared_trunk(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        value = self.critic_head(features)
        return action_mean, action_std, value

if __name__ == "__main__":
    MODEL_PATH = "humanoid_h2_policy_adv.pt"
    num_envs = 64
    num_steps = 128      # Length of policy rollout collection phase
    ppo_epochs = 4       # Mini-batch optimization epochs per loop iteration
    batch_size = 512     # Trajectory block size slicing parameters
    lr = 3e-4
    
    env = H2AdvancedTrainEnv(num_envs=num_envs)
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    obs = env.reset()
    print(f"🚀 Production H2 Balancing Pipeline Engaged | Tuning {env.num_actions} Joints simultaneously on Apple Silicon Core...")
    
    for update in range(1001):
        # 1. BUFFER ROLLOUT COLLECTION LAYOUT
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_values = []
        rollout_dones = []
        
        for _ in range(num_steps):
            with torch.no_grad():
                action_mean, action_std, value = policy(obs)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
            next_obs, rewards, dones = env.step(action.clamp(-1.0, 1.0))
            
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(rewards)
            rollout_values.append(value.squeeze(-1))
            rollout_dones.append(dones)
            
            # FIX 2: Corrected raw 1D tuple unpacking for the vectorized intercept
            fallen_mask = (next_obs[:, -1] < 0.55) | torch.isnan(next_obs[:, -1])
            reset_env_ids = torch.where(fallen_mask)[0]
            if reset_env_ids.numel() > 0:
                env.reset(env_ids=reset_env_ids)
                next_obs = env._get_observations()
                
            obs = next_obs
            
        # Convert lists to continuous tensor structures
        b_obs = torch.stack(rollout_obs).view(-1, env.num_obs)
        b_actions = torch.stack(rollout_actions).view(-1, env.num_actions)
        b_log_probs = torch.stack(rollout_log_probs).view(-1)
        b_rewards = torch.stack(rollout_rewards).view(-1)
        b_values = torch.stack(rollout_values).view(-1)
        b_dones = torch.stack(rollout_dones).view(-1)
        
        # 2. COMPUTE GENERALIZED ADVANTAGE ESTIMATION (GAE)
        with torch.no_grad():
            _, _, next_value = policy(obs)
            next_value = next_value.squeeze(-1)
            
        b_returns = torch.zeros_like(b_rewards)
        last_gae_lam = 0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones
                next_values = next_value
            else:
                next_non_terminal = 1.0 - b_dones[(t + 1) * num_envs : (t + 2) * num_envs]
                next_values = b_values[(t + 1) * num_envs : (t + 2) * num_envs]
                
            rewards_slice = b_rewards[t * num_envs : (t + 1) * num_envs]
            values_slice = b_values[t * num_envs : (t + 1) * num_envs]
            
            delta = rewards_slice + 0.99 * next_values * next_non_terminal - values_slice
            last_gae_lam = delta + 0.99 * 0.95 * next_non_terminal * last_gae_lam
            b_returns[t * num_envs : (t + 1) * num_envs] = last_gae_lam + values_slice
            
        b_advantages = b_returns - b_values
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # 3. MINI-BATCH BACKPROPAGATION PROXIMAL POLICY OPTIMIZATION EPOCHS (FIXED INDENTATION)
        num_data_points = b_obs.shape[0]
        for epoch in range(ppo_epochs):
            permutation = torch.randperm(num_data_points, device='mps')
            for start in range(0, num_data_points, batch_size):
                end = start + batch_size
                mb_indices = permutation[start:end]
                
                new_mean, new_std, new_value = policy(b_obs[mb_indices])
                new_dist = Normal(new_mean, new_std)
                new_log_probs = new_dist.log_prob(b_actions[mb_indices]).sum(dim=-1)
                entropy = new_dist.entropy().sum(dim=-1).mean()
                
                # PPO Clip Objective Calculations
                ratios = torch.exp(new_log_probs - b_log_probs[mb_indices])
                surr1 = ratios * b_advantages[mb_indices]
                surr2 = torch.clamp(ratios, 0.8, 1.2) * b_advantages[mb_indices]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(new_value.squeeze(-1), b_returns[mb_indices])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
        if update % 10 == 0:
            avg_height = torch.mean(b_obs[:, -1]).item()
            print(f"🔄 Update {update:4d}/1000 | Average Torso Height: {avg_height:.2f}m | Combined Loss: {loss.item():.4f}")
            
    # Save optimized states natively
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"\n💾 SUCCESS: Production PPO H2 configuration compiled cleanly at -> {MODEL_PATH}")
    env.scene.destroy()
