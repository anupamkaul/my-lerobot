import os
import time
import genesis as gs
import torch
import torch.nn as nn

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# --- THE ENVIRONMENT WORKSPACE ---
class HumanoidEvalEnv:
    def __init__(self, num_envs=1, dt=0.002):
        self.num_envs = num_envs
        self.dt = dt
        
        # Configure the simulation scene parameters
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0.0, 0.0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -3.0, 1.8),
                camera_lookat=(0.0, 0.0, 0.7),
                camera_fov=45,
            ),
            show_viewer=True # Wakes up the visualization window
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
        # Apply incoming policy choices directly onto motor joint parameters
        target_positions = self.standing_pose + actions * 0.2
        self.humanoid.set_dofs_position(target_positions)
        
        # Step physics forward.
        for _ in range(5):
            self.scene.step()
            
        return self._get_observations()

# --- THE POLICY NEURAL NETWORK ---
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

# --- EXECUTE POLICY EVALUATION ---
if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    
    # Instantiate environment container for 1 standalone robot focus
    env = HumanoidEvalEnv(num_envs=1)
    
    # Load neural layout matching our baseline parameters
    policy = ActorCritic(env.num_obs, env.num_actions).to('mps')
    
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH))
        policy.eval()
        print(f"🎯 SUCCESS: Weights loaded from '{MODEL_PATH}'.")
    else:
        print(f"⚠️ WARNING: '{MODEL_PATH}' missing. Driving model with random weights instead.")

    obs = env.reset()
    print("🎬 Monitoring execution. Close the canvas window to close out.")
    
    while True:
        with torch.no_grad():
            action_means, _ = policy(obs)
            
        # Push the next operational step state to your environment
        obs = env.step(action_means)
        
        # Extract torso position markers directly out of the state vector array
        torso_height = obs[0, -1].item()
        
        # Check fall metric bounds. If the robot collapses, reset its values immediately
        if torso_height < 0.60:
            print("🔄 Falling threshold reached! Resetting humanoid stance...")
            obs = env.reset()
            
        # CRITICAL REFRESH REPAIR: Force-refresh the active UI renderer canvas
        # This resolves the freeze issue completely by manually updating the display layout
        if env.scene.visualizer is not None:
            env.scene.visualizer.update()
            
        # Throttle loop execution to real-world pacing (5 steps * dt 0.002 = 0.01s)
        time.sleep(0.01)
