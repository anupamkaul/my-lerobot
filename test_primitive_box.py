import time
import genesis as gs

# 1. Initialize Genesis targeting your Mac's Metal backend
gs.init(backend=gs.metal)

# 2. Configure the scene with an active viewer window
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -3.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
    ),
    show_viewer=True # Instantiates the native visual window container
)

# 3. Add a flat ground floor
scene.add_entity(gs.morphs.Plane())

# 4. Add exactly ONE simple primitive box floating 2 meters in the air
box = scene.add_entity(
    gs.morphs.Box(
        pos=(0.0, 0.0, 2.0),
        size=(0.4, 0.4, 0.4)
    )
)

# 5. Build the simple standalone workspace
scene.build(n_envs=1)

print("🎬 Starting isolated primitive test loop...")

# 6. Run a clean, structured physics loop
for step in range(500):
    # Advance the mathematical world equations forward by 1 tick
    scene.step()
    
    # Check if the box position vector is actually tracking and updating in memory
    if step % 50 == 0:
        box_pos = box.get_pos() # Pulls Cartesian coordinate arrays
        print(f"📋 Step: {step:3d} | Box Z-Height in Memory: {box_pos[0, 2].item():.3f}m")
        
    # Pacing break to let macOS handle window manager events and redraw the canvas
    time.sleep(0.01)
