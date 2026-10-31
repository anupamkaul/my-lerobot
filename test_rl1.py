import genesis as gs
import torch

# 1. Initialize Genesis targeting your Mac's Metal Performance Shaders backend
gs.init(backend=gs.metal)

# 2. Configure the simulation scene parameters utilizing the options namespace
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(8.0, -8.0, 5.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
    ),
    show_viewer=True  # Wakes up the native desktop UI loop automatically
)

# 3. Drop in a flat structural ground plane
scene.add_entity(gs.morphs.Plane())

# 4. Load the internal Unitree Go2 robot asset using the proper Genesis path mapping
robot_dog = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf",
        pos=(0.0, 0.0, 0.4),
    )
)

# 5. Spin up 64 parallel clones running simultaneously on your M5 Max GPU
num_envs = 64
scene.build(n_envs=num_envs, env_spacing=(2.0, 2.0))

print("🚀 64 Parallel Robot Dog Environments Initialized Successfully on M5 Max GPU Backend!")

# 6. Execute a clean, automated physics loop execution
# Genesis handles real-time visual syncing automatically behind scene.step()
for step in range(1000):
    # Setup control inputs mapped to the 18 degree-of-freedom tracking channels
    actions = torch.zeros((num_envs, robot_dog.n_dofs), device='mps')
    
    # Send position controls down to the structural links
    robot_dog.set_dofs_position(actions)
    
    # Step the entire physics canvas forward. The viewer refreshes automatically here.
    scene.step()
