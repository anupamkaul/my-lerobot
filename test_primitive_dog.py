import time
import genesis as gs

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# 2. Configure the simulation scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -3.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=45,
    ),
    show_viewer=True
)

# 3. Add floor grid
scene.add_entity(gs.morphs.Plane())

# 4. Load the internal Unitree Go2 robot dog asset
robot_dog = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=(0.0, 0.0, 1.0), # Float it 1 meter in the air
    )
)

# 5. Build exactly 1 standalone environment copy
scene.build(n_envs=1)

print("🎬 Starting Articulated Robot Dog isolation test loop...")

# 6. Execute the physics step sequence
for step in range(500):
    # Step the global physics simulation forward by 1 tick
    scene.step()
    
    if step % 50 == 0:
        # Genesis Core Fix: To read actual world-frame positions, you query the entity's global transform tensor
        # This returns a clean (1, 3) matrix containing the exact [X, Y, Z] world values
        current_coordinates = robot_dog.get_pos() # Shape: (1, 3) or link transform
        
        # Safely pull the Z coordinate using flat array parsing to handle any dimension shape
        z_height = current_coordinates.flatten()[2].item()
        print(f"📋 Step: {step:3d} | Dog Base Z-Height in Memory: {z_height:.3f}m")
        
    time.sleep(0.01)
