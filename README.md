# GPU-Accelerated Trajectory Optimization for Multi-Body Gravity-Assist Maneuvers

This repository contains the source code and numerical simulation framework for the comparative astrodynamical analysis of multi-body gravity-assist (MGA) trajectories: **Internal Serpentining vs. Outer Hyperbolic Escapes**. 

This code supports the findings presented in the associated research article and provides a fully reproducible environment for N-body trajectory optimization using GPU-accelerated tensor integrators.

## 📌 Overview

Designing complex MGA trajectories in a restricted N-body environment is a highly non-linear problem that suffers from the curse of dimensionality. To robustly explore the search space and prevent numerical integration artifacts, this framework employs a **dual-tier computational architecture**:

1. **Massive GPU Exploration (JAX/CUDA):** An explicit 4th-order Runge-Kutta (RK4) integrator is compiled via Just-In-Time (JIT) processing and parallelized using tensor vectorization (`jax.vmap`). This is guided by a Differential Evolution algorithm to evaluate thousands of initial conditions simultaneously.
2. **High-Fidelity CPU Cross-Validation (SciPy):** The optimal basins of attraction discovered by the GPU are rigorously propagated using an 8th-order adaptive ordinary differential equation solver (`DOP853`) with strict error tolerances (`RTOL = 1e-10`, `ATOL = 1e-12`).

## ⚙️ Features

- **JAX-Accelerated Dynamics:** Vectorized N-body gravitational equations for multi-moon systems.
- **Differential Evolution Optimization:** Stochastic search implementation to minimize phasing miss distance and required mid-course $\Delta v$ corrections.
- **Automated Validation:** Built-in routine to compute numerical error percentages between the fixed-step GPU tensor and the adaptive CPU solver.
- **Publication-Ready Exports:** The script automatically generates high-quality `.pdf` figures (using `matplotlib` and `seaborn`) and formatted `.tex` tables ready for LaTeX integration.

## 📂 Repository Structure

- `mga_optimization.ipynb`: The main Jupyter Notebook containing the full experimental pipeline, from dynamics formulation to visualization.
- `figures/`: Output directory for generated orbital plots, cross-validation graphs, and energy evolution charts.
- `tables/`: Output directory for generated LaTeX tabular data.

## 🛠️ Dependencies

To run the notebook, ensure you have a Python 3 environment with the following packages installed. (Note: Running this on Google Colab is highly recommended to easily access NVIDIA T4 GPUs).

```bash
pip install jax jaxlib scipy numpy matplotlib seaborn
```

## 🚀 UsageClone the repository

Open the mga_optimization.ipynb notebook.Run all cells. The script will automatically:Create data/, figures/, and tables/ directories if they do not exist.

Execute the Differential Evolution algorithm on the GPU.Run the DOP853 cross-validation on the CPU.
Plot and save the trajectory topologies, energy evolution, and trade-off comparisons into the figures/ folder.
Export the astrodynamical metrics to LaTeX files in the tables/ folder.

## 📊 OutputsThe framework generates several analytical visualizations:Cross-Validation Plot: Proves the physical robustness of the GPU solutions against integration artifacts.Energy Evolution: Demonstrates the successive specific energy depletion (pump-down) during internal serpentining.Propulsive Trade-offs: Compares phasing correction costs ($\Delta v_{corr}$) versus arrival excess velocity ($v_\infty$).Topological Maps: 2D scatter plots of the spacecraft traversing the multi-moon system.

## 📄 LicenseThis project is licensed under the Apache License 2.0 - see the LICENSE file for details.
