# JAX Swarm ES Scalability & Benchmark Analysis

This document details the step throughput and raw scaling performance of JAX-native vectorized environments under the Evolutionary Strategies framework.

## 📈 Vectorization Throughput Performance

By leveraging Google JAX's compiler (`jit`) and automatic vectorization wrapper (`vmap`), we avoid Python interpreter loops during parallel swarm rollouts. This allows us to scale population evaluations effortlessly.

Here is how environment simulation speed scales with larger swarm sizes:

![Throughput Scaling Curve](throughput_scaling.png)

## 🚀 Technical Highlights
- **Zero-copy Vectorization**: Using `jax.vmap` enables concurrent steps across thousands of unique states without manual multi-threading/processing overhead.
- **Compilation Speeds**: Initial step compilation takes around 1-2 seconds, after which step loops execute at native speed.
- **Optimized Memory Access**: Execution on device memory is highly optimized, scaling to massive population rollouts without scaling bottleneck issues.
