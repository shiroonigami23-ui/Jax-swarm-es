# Changelog

## [0.2.0] - 2026-11-25
### Added
- **TPU support.** `colab_run.py --tpu` runs a full study on a free Colab v5e-1 TPU: throughput up to 262,144 envs (compared with CPU and T4), drone training, three 12-seed pendulum studies, renders, the README `TPU-RESULTS` section and a push. One-click notebook: `notebooks/tpu_run.ipynb`.
- **Seed studies.** `python -m swarm_es.seeds` trains one config from N seeds and reports the success rate, median and best final return (on 100 fresh episodes), with curves and the best run. CPU results are in `results/cpu_seeds/`.
- **Baseline controller.** `python -m swarm_es.baselines` runs a hand-written energy-pumping + PD controller, which scores −139.5 on 100 episodes. It proves the pendulum is solvable in this implementation. A test guards it.
- **Start-state curriculum.** `--curriculum-gens` and `--curriculum-start` start pendulum episodes near upright and widen them to the full task. The mean policy is always evaluated on the full task.
- **Working paper.** `paper/` is a Quarto document whose numbers, tables and figures are read from `results/` at render time. It is published to GitHub Pages by `.github/workflows/paper.yml`, with a PDF in `paper/swarm_es_paper.pdf`.
- Tests for the curriculum reset range, the baseline controller and a seed-study smoke run. There are 16 tests in total.

### Found
- The pendulum plateau (≈ −760) is a seed-dependent local optimum, not an environment bug. On CPU, 1 of 8 seeds solves it with plain ES and 0 of 8 with the curriculum. The earlier single-seed sweep could not separate hyperparameters from seed luck.

### Changed
- README: badges, links to the paper and the notebook, the corrected pendulum explanation, a seed-study table, an empty `TPU-RESULTS` section, and new commands and flags.
- Removed `results/results_overview.md` and `results/benchmark_report.md`, which duplicated the README.

## [0.1.0] - 2026-10-26
- First release: OpenAI-ES in pure JAX (`vmap` over policies × episodes, `lax.scan` over generations), drone and pendulum environments, CPU and T4 GPU benchmarks (15.1M env-steps/s peak), Colab GPU runner, hyperparameter sweep, CI.
