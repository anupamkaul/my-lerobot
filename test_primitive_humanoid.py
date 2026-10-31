import time
import genesis as gs

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# 2. Configure the simulation scene scene dimensions
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.002, gravity=(0.0, 0.0, -9.81)), # Humanoids require a tighter dt
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -3.0, 1.8),
        camera_lookat=(0.0, 0.0, 0.7),
        camera_fov=45,
    ),
    show_viewer=True
)

# 3. Add flat ground floor plane
scene.add_entity(gs.morphs.Plane())

# 4. Load the Unitree H1 Humanoid using your local folder path
humanoid = scene.add_entity(
    gs.morphs.URDF(
        file="../unitree_ros/robots/h1_description/urdf/h1.urdf",
        pos=(0.0, 0.0, 1.2), # Float it 1.2 meters in the air
    )
)

# 5. Build exactly 1 standalone environment focus
scene.build(n_envs=1)

print("🎬 Starting Articulated Humanoid isolation test loop...")

# 6. Execute the physics step sequence
for step in range(500):
    # Advance the mathematical world equations forward by 1 tick
    scene.step()
    
    # Track the humanoid's pelvis/base position array in memory
    if step % 50 == 0:
        humanoid_pos = humanoid.get_pos() 
        z_height = humanoid_pos.flatten()[2].item() # Safely pull the 3rd index (Z coordinate)
        print(f"📋 Step: {step:3d} | Humanoid Base Z-Height in Memory: {z_height:.3f}m")
        
    # Maintain pacing to allow macOS to refresh the window canvas frames fluidly
    # time.sleep(0.01)
