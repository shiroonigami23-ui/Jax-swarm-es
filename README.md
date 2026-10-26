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
This environment trains a policy to stabilize or guide a drone. Below are animations showing the control strategy in action:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Drone Ep 0](results/drone/episode_0.gif) | ![Drone Ep 1](results/drone/episode_1.gif) | ![Drone Ep 2](results/drone/episode_2.gif) | ![Drone Ep 3](results/drone/episode_3.gif) | ![Drone Ep 4](results/drone/episode_4.gif) |

### Pendulum Balance
Classical continuous control task utilizing evolutionary parameter adjustments to balance a pendulum:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Pendulum Ep 0](results/pendulum/episode_0.gif) | ![Pendulum Ep 1](results/pendulum/episode_1.gif) | ![Pendulum Ep 2](results/pendulum/episode_2.gif) | ![Pendulum Ep 3](results/pendulum/episode_3.gif) | ![Pendulum Ep 4](results/pendulum/episode_4.gif) |

---

## 🛠️ Installation & Setup

Clone the repository and install in editable mode:
```bash
git clone https://github.com/shiroonigami23-ui/ax-swarm-es.git
cd ax-swarm-es
pip install -e .
```

## 💻 Usage

### 1. Run Benchmarks
To evaluate raw step throughput across different vectorized swarm sizes:
```bash
python -m swarm_es.bench --env drone --sizes 1 64 1024 8192
```

### 2. Train an Agent
To train an agent using Evolutionary Strategies:
```bash
# Train Drone controller
python -m swarm_es.train --env drone --gens 500 --log-every 10

# Train Pendulum swing-up
python -m swarm_es.train --env pendulum --pop 1024 --gens 150
```
