# Swarm ES Performance & Training Report

This document provides an overview of the training results and hardware benchmarks obtained for the Evolutionary Strategies (ES) swarm framework.

## 1. Environment Performance Summary

### Drone Control
The policy learns to chase waypoints through gusty wind, reaching about 9 targets per 10-second episode by generation 500. 
Below are animations showing the control strategy in action across 5 generated test episodes:

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Ep 0](drone/episode_0.gif) | ![Ep 1](drone/episode_1.gif) | ![Ep 2](drone/episode_2.gif) | ![Ep 3](drone/episode_3.gif) | ![Ep 4](drone/episode_4.gif) |

### Pendulum Swing-up (not solved yet)
Same dynamics as Gym Pendulum-v1. Current policy reaches about -770 (solved is about -150), so these GIFs show an unsolved policy.

| Episode 0 | Episode 1 | Episode 2 | Episode 3 | Episode 4 |
| :---: | :---: | :---: | :---: | :---: |
| ![Ep 0](pendulum/episode_0.gif) | ![Ep 1](pendulum/episode_1.gif) | ![Ep 2](pendulum/episode_2.gif) | ![Ep 3](pendulum/episode_3.gif) | ![Ep 4](pendulum/episode_4.gif) |

## 2. Comparative Training Analysis

Below is the performance improvement comparison graph over iterations:

![Training Comparison](training_comparison.png)
