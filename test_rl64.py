import os
import time
import genesis as gs
import torch
import torch.nn as nn

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# --- THE EXACT NEURAL NETWORK ARCHITECTURE ---
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
    
    # 2. Configure the simulation scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 1.8),
            camera_lookat=(0.0, 0.0, 0.7),
            camera_fov=45,
        ),
        show_viewer=True
    )
    
    scene.add_entity(gs.morphs.Plane())
    humanoid = scene.add_entity(
        gs.morphs.URDF(
            file="../unitree_ros/robots/h1_description/urdf/h1.urdf", 
            pos=(0.0, 0.0, 1.05), # Perfectly flush foot boundaries
        )
    )
    
    # from inference of 1 humanoid, move to 64 humanoids 
    num_envs = 64
    scene.build(n_envs=num_envs, env_spacing=(3.0, 3.0))

    # in this version, I will have some of the humanoids act on policy
    # and have some not and fall down completely. See if that works.
    
    # Secure tracking links
    torso = humanoid.get_link("torso_link")
    num_actions = humanoid.n_dofs
    standing_pose = humanoid.get_dofs_position()
    
    # Force the network topology shapes to strictly match checkpoint matrices
    CHECKPOINT_OBS = 58
    CHECKPOINT_ACTIONS = 27
    
    # Load weights safely using the forced layout dims
    policy = ActorCritic(CHECKPOINT_OBS, CHECKPOINT_ACTIONS).to('mps')
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location='mps'))
        policy.eval()
        print(f"🎯 SUCCESS: Weights loaded smoothly from '{MODEL_PATH}'.")
    else:
        print(f"⚠️ WARNING: '{MODEL_PATH}' missing. Running random initial baseline policy.")

    # Post-build gain setup: configuring joint spring metrics
    humanoid.set_dofs_kp(torch.ones(num_actions, device='mps') * 150.0) 
    humanoid.set_dofs_kv(torch.ones(num_actions, device='mps') * 10.0)  
    
    # Apply standard initial manual reset coordinates to drop into scene
    humanoid.set_dofs_position(standing_pose)
    humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
    
    print("🎬 Monitoring inference execution. Close the visualizer canvas window to close out.")
    
    # 4. PACED REAL-TIME LOOP
    while scene.viewer.is_alive:
        
        # Pull clean sensor tensors and explicitly decouple them from tracking memory
        q = humanoid.get_dofs_position().clone()
        v = humanoid.get_dofs_velocity().clone()
        torso_positions = torso.get_pos().clone()
        torso_height = torso_positions[:, 2].unsqueeze(1)
        torso_velocities = torso.get_vel()[:, :3].clone()
        
        # Construct isolated observation tensor matrix explicitly targeted to mps device
        obs_base = torch.cat([q, v, torso_velocities, torso_height], dim=-1).to('mps')
        
        # Pad or slice to align observations cleanly to 58 dimensions
        if obs_base.shape[-1] < CHECKPOINT_OBS:

	    # make sure I account for all humanoids
            batch_size = obs_base.shape[0]
            padding = torch.zeros((batch_size, CHECKPOINT_OBS - obs_base.shape[-1]), device='mps')
            obs = torch.cat([obs_base, padding], dim=-1)

        else:
            obs = obs_base[:, :CHECKPOINT_OBS]
            
        with torch.no_grad():
            action_means, _ = policy(obs)
            
        # Extract actions and match them to the current robot's degrees of freedom
        active_actions = action_means[:, :num_actions]
        target_positions = standing_pose + active_actions * 0.3
        
        # Pass position targets directly down into control pipeline
        humanoid.control_dofs_position(target_positions)
        
        # Advance physics ticks (Sub-step to match real time pacing)
        for _ in range(5):
            scene.step()
            
        # 🚨 THE PERMANENT FIX: Extract batch index 0, spatial index 2
        live_height = torso.get_pos()[0, 2].item()
        
        # Robust fall checking

        # First, instead of biasing everything on the first humanoid, we vectorize the
        # obs across all 4 and watch this tensor independently for fall analytics and take
        # actions only on the ones that are falling down

        # 1. Pull the height of the torso link across all 64 parallel environments 
        # torso.get_pos() shape is (64, 3) -> we grab index 2 for the Z-axis (height)

        live_heights = torso.get_pos()[:, 2]  # this becomes a 64 vector of heights..

        # 2. Construct a mask of all 64 heights
        fallen_mask = (live_heights < 0.55 ) | torch.isnan(live_heights)

        # 3. Find the exact indices of the environments that need a reset
        reset_env_ids = torch.where(fallen_mask)[0] 

        # 4. If any enviroments have fallen, apply reset targets
        if len(reset_env_ids) > 0:
            print(f"🔄 Resetting {len(reset_env_ids)} fallen humanoids at indices: {reset_env_ids.tolist()}")

            # Genesis allows you to pass env_ids to restrict state-setting to specific sub-scenes

            
            humanoid.set_dofs_position(position = standing_pose[reset_env_ids], envs_idx=reset_env_ids)
            humanoid.set_dofs_velocity(velocity = torch.zeros_like(standing_pose[reset_env_ids]), envs_idx=reset_env_ids)

        '''
        if live_height < 0.55 or live_height != live_height:
            print(f"🔄 Falling threshold reached ({live_height:.2f}m)! Resetting stance parameters...")
            humanoid.set_dofs_position(standing_pose)
            humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
        '''
            
        # Enforce execution time break step to let macOS handle window event frames fluidly
        #time.sleep(0.01)

