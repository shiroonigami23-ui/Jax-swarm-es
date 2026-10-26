# JAX Swarm ES Scalability & Benchmark Analysis

This document details the step throughput and raw scaling performance of JAX-native vectorized environments under the Evolutionary Strategies framework.

## 📈 Vectorization Throughput Performance

By leveraging Google JAX's compiler (`jit`) and automatic vectorization wrapper (`vmap`), we avoid Python interpreter loops during parallel swarm rollouts. This allows us to scale population evaluations effortlessly.

Here is how environment simulation speed scales with larger swarm sizes:

![Throughput Scaling Curve](throughput_scaling.png)

## What the CPU numbers show

These runs were on a Colab **CPU** runtime (see [`bench/drone_cpu.csv`](bench/drone_cpu.csv)).

- Batching with `jax.vmap` lifts drone throughput from about 94k env-steps/s (1 env) to a peak of about **708k env-steps/s at 1,024 envs**, roughly 7.5x.
- Past that, CPU throughput **falls**: about 519k at 8,192 envs and about 440k at 32,768 to 65,536 envs, as the batch outgrows the CPU caches. The pendulum curve shows the same rise and fall.
- `vmap` itself adds no Python loops or multiprocessing, but on CPU there is a clear sweet spot in batch size. Large populations are meant for GPU/TPU, where these numbers still need to be measured.
