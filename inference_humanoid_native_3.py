import os
import time
import genesis as gs
import torch
import torch.nn as nn

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

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

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    
    # 2. Configure the simulation scene natively
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 1.8),
            camera_lookat=(0.0, 0.0, 0.7),
            camera_fov=45,
        ),
        show_viewer=True # Wakes up the visualization window fluidly
    )
    
    scene.add_entity(gs.morphs.Plane())
    humanoid = scene.add_entity(
        gs.morphs.URDF(
            file="../unitree_ros/robots/h1_description/urdf/h1.urdf",
            pos=(0.0, 0.0, 1.05),
        )
    )
    
    # Build exactly 1 standalone humanoid environment
    scene.build(n_envs=1)
    
    # Extract reference tracking handles
    torso = humanoid.get_link("torso_link")
    num_actions = humanoid.n_dofs
    num_obs = num_actions * 2 + 3 + 1 
    standing_pose = humanoid.get_dofs_position()
    
    # Rebuild neural layer mapping
    policy = ActorCritic(num_obs, num_actions).to('mps')
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH))
        policy.eval()
        print(f"🎯 SUCCESS: Weights loaded from '{MODEL_PATH}'.")
    else:
        print(f"⚠️ WARNING: Model file missing. Driving with random weights.")

    # Reset parameters to clean stance boundaries
    humanoid.set_dofs_position(standing_pose)
    humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
    
    print("🎬 Monitoring execution. Close the canvas window to exit.")
    
    # 3. THE AUTHORITATIVE NATIVE REFRESH LOOP
    while scene.viewer.is_alive:
        
        # Gather sensory data straight out of the active engine frame
        q = humanoid.get_dofs_position()
        v = humanoid.get_dofs_velocity()
        torso_pos = torso.get_pos()
        torso_height = torso_pos[:, 2].unsqueeze(1)
        torso_vel = torso.get_vel()[:, :3]
        
        obs = torch.cat([q, v, torso_vel, torso_height], dim=-1)
        
        # Query neural model decisions
        with torch.no_grad():
            action_means, _ = policy(obs)
            
        # Map actions to joint motor outputs
        target_positions = standing_pose + action_means * 0.2
        humanoid.set_dofs_position(target_positions)
        
        # Step physics forward.
        scene.step()
        
        # 🚨 CRITICAL FIX: Explicitly tell the Genesis viewer to refresh the UI window
        # This breaks the frame-buffer lock and animates the robot on screen fluidly!
        scene.viewer.update()
        
        # Robust Fall Detection Check
        current_height = torso_height.item()
        
        if current_height < 0.60 or current_height != current_height:
            print("🔄 Falling threshold reached! Resetting humanoid stance parameters...")
            humanoid.set_dofs_position(standing_pose)
            humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
            
        # Pacing throttle to match the 0.002s delta time step and prevent CPU flooding
        time.sleep(0.002)
