# JAX Swarm ES

An ultra-fast, hardware-accelerated Evolutionary Strategies (ES) framework powered by **Google JAX** for swarm-based continuous control. Train thousands of agents in parallel directly on GPU or CPU with minimal overhead.

## 📊 Performance & Scaling Graphs

### 1. Vectorization Throughput Scaling Curve
By leveraging Google JAX's compiler (`jit`) and automatic vectorization wrapper (`vmap`), we avoid Python interpreter loops during parallel swarm rollouts:

![Throughput Scaling Curve](results/throughput_scaling.png)

### 2. Training Curves & Convergence
Below is the comparative performance graph showing maximum fitness, population mean, and average policy return over iterations:

![Training Comparison](results/training_comparison.png)

---

## 🎬 Environment Simulations

### Drone Control
This environment trains a neural network controller to stabilize or guide a drone. Below are animations showing the control strategy in action:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Drone Ep 0](results/drone/episode_0.gif) | ![Drone Ep 1](results/drone/episode_1.gif) | ![Drone Ep 2](results/drone/episode_2.gif) | ![Drone Ep 3](results/drone/episode_3.gif) | ![Drone Ep 4](results/drone/episode_4.gif) |

### Pendulum Balance
Classical continuous control task utilizing evolutionary parameter adjustments to balance an inverted pendulum:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Pendulum Ep 0](results/pendulum/episode_0.gif) | ![Pendulum Ep 1](results/pendulum/episode_1.gif) | ![Pendulum Ep 2](results/pendulum/episode_2.gif) | ![Pendulum Ep 3](results/pendulum/episode_3.gif) | ![Pendulum Ep 4](results/pendulum/episode_4.gif) |

---

## 🧠 Evolutionary Strategy & Architecture Overview

This repository implements **OpenAI-ES (Evolutionary Strategies)** optimized entirely to run within native JAX arrays and compiled functions. 

### Vectorized Population Rollouts
Traditional reinforcement learning or ES loops evaluate policies sequentially or use heavy multi-processing libraries, adding significant overhead from IPC (inter-process communication).

```
               +-----------------------+
               |   Central ES Policy   |
               |      Parameters       |
               +-----------+-----------+
                           |  Perturb via Mutation Noise
                           v
         +-------------------------------------+
         |  Vectorized Swarm / Population (W)  |
         |   [W_1, W_2, ... , W_PopSize]       |
         +-----------------+-------------------+
                           |  vmap(Rollout)
                           v
     +---------------------------------------------+
     | Vectorized JAX Environments (vmap + jit)    |
     | Runs whole episode steps in parallel on GPU  |
     +---------------------+-----------------------+
                           | Gather Total Rewards / Fitness
                           v
         +-------------------------------------+
         |  Fitness Evaluation Array (F)       |
         +-----------------+-------------------+
                           | Compute Gradient Estimate
                           v
               +-----------------------+
               |      Adam/SGD         |
               |  Parameter Update     |
               +-----------------------+
```

Instead of launching sub-processes, `jax.vmap` scales execution at the hardware execution layer (CUDA warps or SIMD registers). By grouping the environment step rules into pure JAX functions, we can step a population of **65,536 environments** in a fraction of a millisecond.

---

## 🛠️ Detailed Installation & Setup

### Prerequisites
- Python 3.9+
- JAX (Compatible with CPU, GPU, or TPU backends)

Clone the repository and install in editable mode:
```bash
git clone https://github.com/shiroonigami23-ui/ax-swarm-es.git
cd ax-swarm-es
pip install -e .
```

### Configuring GPU/TPU Support
Ensure that you install the JAX version matching your hardware's CUDA platform to unlock massive acceleration:
```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

---

## 💻 CLI Usage Guide

### 1. Benchmarking Environment Speeds
Evaluate raw step throughput (environment steps per second) over different vectorized batch sizes:
```bash
python -m swarm_es.bench --env [drone|pendulum] --sizes 1 64 1024 8192 32768
```

### 2. Training Agents
To train an agent, launch the training script. Performance checkpoints, parameters (`.npz`), history (`.csv`), and loss curves are automatically saved under the directory designated by `--env`.
```bash
# Train Drone stabilizer
python -m swarm_es.train --env drone --gens 500 --log-every 10

# Train Pendulum swing-up
python -m swarm_es.train --env pendulum --pop 1024 --gens 150 --log-every 5
```

### 3. Visualizing Renders
Use the renderer to generate evaluation GIFs demonstrating how the best policy performs:
```bash
python -m swarm_es.render --run runs/drone --episodes 5
```

---

## 🎛️ Hyperparameters Directory

Below is a list of parameters adjustable via source code customization or configuration edits:

| Parameter | Description | Drone Default | Pendulum Default |
| :--- | :--- | :---: | :---: |
| `pop` / Population Size | Number of perturbed policy mutations evaluated per generation. | 2048 | 1024 |
| `gens` / Generations | Maximum evolutionary generations to train. | 500 | 150 |
| `learning_rate` | Step size adjustment rate for the parameter optimizer (Adam). | 0.01 | 0.02 |
| `l2_decay` | Weight decay penalty coefficient to prevent exploding weights. | 0.005 | 0.005 |
| `sigma` | Standard deviation of the Gaussian mutation noise. | 0.05 | 0.1 |
| `episode_len` | Maximum step limits allowed before termination. | 500 | 200 |

---

## 🚀 JAX Performance Tuning & Tips

For peak efficiency on your local workstation, virtual machines, or GCP environments, make use of these helpful parameters:

1. **Prevent Out Of Memory (OOM) Errors**: JAX by default pre-allocates 90% of GPU memory at startup. Disable this or adjust allocation fractions using environment variables if running in parallel with other models:
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```
2. **Float 32 vs Float 64**: JAX operates on single-precision float32 by default. Stick to float32 to utilize GPU tensor cores and increase execution speed by up to 4x over float64.
3. **Compilation Caching**: If your run times are extended, enable the compilation cache to avoid recompiling XLA modules across multiple execution runs:
   ```bash
   export JAX_COMPILER_CACHE_DIR=/tmp/jax_cache
   ```

---

## ❓ FAQ & Troubleshooting

#### Q: I receive a `ConcretizationTypeError` or JAX Tracer Error.
* **Reason**: JAX requires loop conditions and array dimensions to be static/constant at compile-time. Avoid standard python conditions inside the jit-compiled loop that depend on dynamic array values. Use `jax.lax.cond` or `jax.lax.select` instead.

#### Q: The performance is slow during the first generation, then dramatically speeds up. Why?
* **Reason**: This is normal! The first run triggers JAX compilation. XLA compiles the mathematical graphs into highly optimized assembly instruction sets for your CPU/GPU hardware. Subsequent steps run at bare-metal execution speeds.

#### Q: How do I load a trained model?
* **Reason**: The model parameters are saved inside `runs/{env}/params.npz`. You can load them back in python using:
  ```python
  import numpy as np
  params = np.load('runs/drone/params.npz')
  ```
