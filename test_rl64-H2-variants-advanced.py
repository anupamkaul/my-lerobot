import os
import time
import genesis as gs
import torch
import torch.nn as nn

# 1. Initialize Genesis targeting Mac's Metal Performance Shaders
gs.init(backend=gs.metal)

# --- ADVANCED STOCHASTIC ACTOR-CRITIC ARCHITECTURE MATCHING TRAINING ---
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        
        # Shared feature trunk layout matching train-H2-advanced.py
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
        
        # Trainable log standard deviation array parameter
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
    # Updated path tracking the newly compiled advanced weights file name
    MODEL_PATH = "humanoid_h2_policy_adv.pt"
    num_envs = 64
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 1.8), camera_lookat=(0.0, 0.0, 0.7), camera_fov=45,
        ),
        show_viewer=True
    )
    
    scene.add_entity(gs.morphs.Plane())
    humanoid = scene.add_entity(
        gs.morphs.URDF(
            file="../unitree_ros/robots/h2_description/H2.urdf", 
            pos=(0.0, 0.0, 1.05),
        )
    )
    
    scene.build(n_envs=num_envs, env_spacing=(3.0, 3.0))
    
    torso = humanoid.get_link("torso_link")
    num_actions = humanoid.n_dofs           # 37 joints
    num_obs = num_actions * 2 + 3 + 3 + 3 + 1  # Exactly 84 advanced channels
    standing_pose = humanoid.get_dofs_position()
    
    policy = ActorCritic(num_obs, num_actions).to('mps')
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location='mps'))
        policy.eval()
        print(f"🎯 SUCCESS: Natively loaded H2 Advanced shared-trunk weights from '{MODEL_PATH}'.")
    else:
        print(f"⚠️ WARNING: '{MODEL_PATH}' missing. Running random initial baseline policy.")

    humanoid.set_dofs_kp(torch.ones(num_actions, device='mps') * 150.0) 
    humanoid.set_dofs_kv(torch.ones(num_actions, device='mps') * 10.0)  
    
    humanoid.set_dofs_position(standing_pose)
    humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
    
    print("🎬 Monitoring inference split execution. Alternate humanoids hold a static standing posture baseline.")
    
    while scene.viewer.is_alive:

        # Pull clean sensor tensors and explicitly map them to match training observations
        q = humanoid.get_dofs_position().clone()
        v = humanoid.get_dofs_velocity().clone()
        torso_positions = torso.get_pos().clone()
        torso_height = torso_positions[:, 2].unsqueeze(1)
        
        # Pull the complete combined velocity matrices using verified non-hallucinated methods
        torso_lin_vel = torso.get_vel().clone()
        torso_ang_vel = torso.get_ang().clone()
        
        # --- FIXED MANUAL PROJECTED GRAVITY COMPOSITION ---
        torso_quat = torso.get_quat()
        w, x, y, z = torso_quat[:, 0], torso_quat[:, 1], torso_quat[:, 2], torso_quat[:, 3]
        
        gravity_x = 2 * (x * z - w * y)
        gravity_y = 2 * (y * z + w * x)
        gravity_z = 1.0 - 2 * (x**2 + y**2)
        
        projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1).to('mps')
        # --------------------------------------------------
        
        obs = torch.cat([q, v, torso_lin_vel, torso_ang_vel, projected_gravity, torso_height], dim=-1).to('mps')
        
        with torch.no_grad():
            # Reading advanced output layout: we discard noise parameters during tracking evaluation loops
            action_means, _, _ = policy(obs)
            
        env_indices = torch.arange(num_envs, device='mps')
        alternate_mask = (env_indices % 2 == 1)
        alternate_env_ids = torch.where(alternate_mask)[0]
        
        # Apply active network actions solely to even-numbered robots
        target_positions = standing_pose.clone()
        target_positions[~alternate_mask] = standing_pose[~alternate_mask] + action_means[~alternate_mask] * 0.3
        humanoid.control_dofs_position(target_positions)
        
        # Hard-snap non-policy variants back to static starting pose frames to freeze them completely
        humanoid.set_dofs_position(position=standing_pose[alternate_env_ids], envs_idx=alternate_env_ids)
        humanoid.set_dofs_velocity(velocity=torch.zeros_like(standing_pose[alternate_env_ids]), envs_idx=alternate_env_ids)
        
        for _ in range(5):
            scene.step()
            
        # Fall tracking reset logic exclusively evaluated on active agents
        live_heights = torso.get_pos()[:, 2]
        fallen_mask = (live_heights < 0.55) | torch.isnan(live_heights)
        
        allowed_to_reset_mask = (env_indices % 2 == 0)
        fallen_mask = fallen_mask & allowed_to_reset_mask
        reset_env_ids = torch.where(fallen_mask)[0]
        
        if reset_env_ids.numel() > 0:
            print(f"🔄 Resetting {reset_env_ids.numel()} active humanoids at indices: {reset_env_ids.tolist()}")
            humanoid.set_dofs_position(position=standing_pose[reset_env_ids], envs_idx=reset_env_ids)
            humanoid.set_dofs_velocity(velocity=torch.zeros_like(standing_pose[reset_env_ids]), envs_idx=reset_env_ids)
