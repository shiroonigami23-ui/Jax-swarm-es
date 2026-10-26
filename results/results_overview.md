# Swarm ES Performance & Training Report

This document provides an overview of the training results and hardware benchmarks obtained for the Evolutionary Strategies (ES) swarm framework.

## 1. Environment Performance Summary

### Drone Control
This environment trains a policy to stabilize or guide a drone. 
Below are animations showing the control strategy in action across 5 generated test episodes:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Ep 0](results/drone/episode_0.gif) | ![Ep 1](results/drone/episode_1.gif) | ![Ep 2](results/drone/episode_2.gif) | ![Ep 3](results/drone/episode_3.gif) | ![Ep 4](results/drone/episode_4.gif) |

### Pendulum Balance
Classical continuous control task utilizing evolutionary parameter adjustments:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Ep 0](results/pendulum/episode_0.gif) | ![Ep 1](results/pendulum/episode_1.gif) | ![Ep 2](results/pendulum/episode_2.gif) | ![Ep 3](results/pendulum/episode_3.gif) | ![Ep 4](results/pendulum/episode_4.gif) |

## 2. Comparative Training Analysis

Below is the performance improvement comparison graph over iterations:

![Training Comparison](training_comparison.png)
