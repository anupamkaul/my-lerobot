import os
import time
import genesis as gs
import torch
import torch.nn as nn

# Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

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

from time import sleep

if __name__ == "__main__":
    MODEL_PATH = "humanoid_ppo_policy.pt"
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.5, 2.0),
            camera_lookat=(0.0, 0.0, 0.9),
            camera_fov=45,
        ),
        show_viewer=True # Opens up the desktop interactive panel cleanly
    )
    
    scene.add_entity(gs.morphs.Plane())
    
    # 🚨 REWIRING FILE RESOLUTION: Loading the stable, native XML asset layout directly
    humanoid = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/humanoid.xml", 
            pos=(0.0, 0.0, 1.3),
        )
    )
    
    scene.build(n_envs=1)
    
    num_actions = humanoid.n_dofs
    num_obs = num_actions * 2 + 3 + 1 
    standing_pose = humanoid.get_dofs_position()
    
    policy = ActorCritic(num_obs, num_actions).to('mps')
    if os.path.exists(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH))
        policy.eval()
        print(f"🎯 SUCCESS: Evaluator weights loaded from '{MODEL_PATH}'.")

    humanoid.set_dofs_position(standing_pose)
    humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
    
    print("🎬 Monitoring execution loop. Close the 3D window window canvas to exit.")
    obs = torch.cat([humanoid.get_dofs_position(), humanoid.get_dofs_velocity(), humanoid.get_vel()[:, :3], humanoid.get_pos()[:, 2].unsqueeze(1)], dim=-1)
    
    # Run the standard visual tracking monitor using the native viewer framework
    while scene.viewer.is_alive:
        
        with torch.no_grad():
            action_means, _ = policy(obs)
            
        target_positions = standing_pose + action_means * 0.4
        humanoid.set_dofs_position(target_positions)
        sleep(1.5)
        
        # Step physics forward. Genesis refreshes frames automatically here
        print("debug: step start ")
        scene.step()
        print("debug: step end ")
        sleep(1.5)
        
        # Refresh current sensory data vectors from the engine layer
        q = humanoid.get_dofs_position()
        v = humanoid.get_dofs_velocity()
        torso_pos = humanoid.get_pos()
        torso_height = torso_pos[:, 2].unsqueeze(1)
        torso_vel = humanoid.get_vel()[:, :3]
        obs = torch.cat([q, v, torso_vel, torso_height], dim=-1)
        
        current_height = torso_height.item()
        
        # Fall detection reset bounds check
        if current_height < 0.65 or current_height != current_height:
            print("🔄 Falling threshold reached! Resetting humanoid stance parameters...")
            humanoid.set_dofs_position(standing_pose)
            humanoid.set_dofs_velocity(torch.zeros_like(standing_pose))
            obs = torch.cat([humanoid.get_dofs_position(), humanoid.get_dofs_velocity(), humanoid.get_vel()[:, :3], humanoid.get_pos()[:, 2].unsqueeze(1)], dim=-1)
            
        # Pacing throttle
        time.sleep(0.002)
