import genesis as gs
import torch

# 1. Initialize Genesis with native Apple Silicon GPU acceleration
gs.init(backend=gs.metal)

# 2. Configure the simulation workspace
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.002, # Humanoids require a faster physics step (500Hz) to remain stable
        gravity=(0.0, 0.0, -9.81)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(4.0, -4.0, 3.0),
        camera_lookat=(0.0, 0.0, 0.8),
        camera_fov=45,
    ),
    show_viewer=True
)

# 3. Add a ground plane
scene.add_entity(gs.morphs.Plane())

# 4. Load the standing Unitree H1 Humanoid from your local directory
humanoid = scene.add_entity(
    gs.morphs.URDF(
        file="../unitree_ros/robots/h1_description/urdf/h1.urdf", # Clean relative path
        pos=(0.0, 0.0, 1.05),
    )
)

# 5. Spin up 64 parallel humanoids across a spacious grid
num_envs = 64
scene.build(n_envs=num_envs, env_spacing=(3.0, 3.0))

print(f"🤖 Humanoid initialized with {humanoid.n_dofs} Motorized Degrees of Freedom!")
print("🚀 64 Parallel Humanoid Environments Running Natively on M5 Max GPU!")

# 6. Extract the default factory standing posture coordinates
standing_pose = humanoid.get_dofs_position()

# 7. Execute the control simulation loop
for step in range(2000):
    # Actively command the motors to fight gravity and hold the standing position
    humanoid.set_dofs_position(standing_pose)
    
    # Advance the 64 parallel worlds forward by 1 tick
    scene.step()
