# JAX Swarm ES

An ultra-fast, hardware-accelerated Evolutionary Strategies (ES) framework powered by **Google JAX** for swarm-based continuous control. Train thousands of agents in parallel directly on GPU or CPU with minimal overhead.

## 🚀 Features
- **100% JAX-Native**: Parallelized environment steps using `jax.vmap` and jit-compiled loops.
- **Scalable Evolutionary Strategies**: Optimized population evaluation scaling seamlessly up to 65k+ environment instances.
- **Built-in Environments**:
  - **Drone**: 2D drone stabilization and tracking task.
  - **Pendulum**: Classical continuous-state inverted pendulum swing-up task.
- **Visualization Suite**: Automated rendering of high-quality comparative tracking graphs and agent simulation GIFs.

## 📊 Performance & Results

We provide a comprehensive performance analysis, complete with training metrics and simulated episodes. See our detailed results overview:

👉 **[Results and Simulations Overview](results/results_overview.md)**

### Quick Summary
- **Massive Parallelism**: Capable of processing over **700,000 environment steps per second** using vectorized CPU/GPU execution.
- **Sample Trajectory Visuals**:
  
  ![Drone Control](results/drone/episode_0.gif)

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

### 3. Render Agent Simulations
Export animated GIFs of your trained policy over multiple evaluation episodes:
```bash
python -m swarm_es.render --run runs/drone --episodes 5
```
