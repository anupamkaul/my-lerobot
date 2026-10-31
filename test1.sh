python -c "import mujoco; print('🎉 MuJuCo Engine: READY'); import torch; print('🍏 Apple Silicon GPU Acceleration (MPS):', 'ACTIVE' if torch.backends.mps.is_available() else 'INACTIVE')"
