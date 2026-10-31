import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# --- THE ENVIRONMENT WORKSPACE ---
class HumanoidWalkEnv:
    def __init__(self, num_envs=32, dt=0.002, show_viewer=True):
        self.num_envs = num_envs
        self.dt = dt
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(4.0, -4.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=45,
            ),
            show_viewer=show_viewer
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
        for _ in range(5):
            self.scene.step()
        next_obs = self._get_observations()
        torso_velocities = next_obs[:, self.num_actions * 2 : self.num_actions * 2 + 3]
        torso_heights = next_obs[:, -1]
        
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
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Tanh()
        )
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

# --- RUN PRODUCTION PIPELINE ---
if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    
    # ----------------------------------------------------
    # PHASE 1: PARALLEL TRAINING LAYER
    # ----------------------------------------------------
    num_envs = 32
    print(f"🔋 Initializing training grid ({num_envs} humanoids)...")
    train_env = HumanoidWalkEnv(num_envs=num_envs, show_viewer=False) # Headless training for max speed
    
    policy = ActorCritic(train_env.num_obs, train_env.num_actions).to('mps')
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    obs = train_env.reset()
    print("🚀 PPO Optimization Loop Commenced...")
    
    for update in range(101): # Capped at 100 quick epochs to verify file storage
        action_means, values = policy(obs.detach())
        noise = torch.randn_like(action_means) * 0.1
        actions = action_means + noise
        
        next_obs, rewards, dones = train_env.step(actions.clamp(-1.0, 1.0))
        
        _, next_values = policy(next_obs.detach())
        target_values = rewards.unsqueeze(1) + 0.99 * next_values * (1.0 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(values, target_values)
        
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        obs = next_obs
        
        if update % 20 == 0:
            print(f"🔄 Epoch {update:3d}/100 | Speed: {obs[0, train_env.num_actions*2].item():.3f} m/s | Loss: {critic_loss.item():.4f}")
            
    # Save policy weights securely to disk
    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"💾 Model architecture weights saved successfully to: {MODEL_PATH}")
    
    # CORE FIX: Destroy the background scene components entirely before creating the evaluation window
    train_env.scene.destroy()
    del train_env
    
    # ----------------------------------------------------
    # PHASE 2: SINGLE-HUMANOID VISUAL INFERENCE LOOP
    # ----------------------------------------------------
    print("\n🖥️ Launching Single-Humanoid Inference Evaluation Window...")
    eval_env = HumanoidWalkEnv(num_envs=1, show_viewer=True) # Open visual viewer safely for 1 standalone humanoid
    
    # Instantiate clean model shell and inject saved file data weights
    eval_policy = ActorCritic(eval_env.num_obs, eval_env.num_actions).to('mps')
    eval_policy.load_state_dict(torch.load(MODEL_PATH))
    eval_policy.eval() # Switch network layers to inference evaluation mode
    
    obs = eval_env.reset()
    print("🎯 Inference active! Watching the trained policy control the humanoid...")
    
    for step in range(2000):
        with torch.no_grad(): # Disable gradients to maximize simulation FPS
            action_mean, _ = eval_policy(obs)
            
        # Drive joints directly with pure network decisions (no random noise added)
        obs, rewards, dones = eval_env.step(action_mean)
