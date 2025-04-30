# 1D Wave Simulation with CUDA

GPU-accelerated finite difference solver for 1D wave propagation...

## Features
- CUDA parallelization
- Periodic boundary conditions
- Gaussian initial conditions

## Requirements
- NVIDIA GPU with Compute Capability 3.5+
- CUDA Toolkit 11.0+
- Linux/Windows with build tools

## Installation
```bash
git clone https://github.com/your/repo
cd repo
nvcc main.cu kernels.cu -o wave_sim -arch=sm_70
```
## Basic execution
./wave_sim
## Save results 
./wave_sim > wave_data.txt
## for visualization
!python3 plot_height_multipanel.py output.txt
