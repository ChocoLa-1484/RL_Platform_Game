# Procedural platformer experiment

Mean ± sample standard deviation across training seeds. Each stage uses the same held-out layouts and starts for every model.

| Variant | Stage | Completion | Death rate | Progress | Platform coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ppo_shaped | 1 | 67.3% ± 28.3% | 32.7% | 92.4% | 89.1% |
| ppo_shaped | 2 | 96.3% ± 6.4% | 3.7% | 99.1% | 98.2% |
| ppo_shaped | 3 | 78.7% ± 25.8% | 21.3% | 93.8% | 84.3% |
| ppo_auto_jump | 1 | 51.0% ± 0.0% | 49.0% | 88.4% | 83.7% |
| ppo_auto_jump | 2 | 96.3% ± 6.4% | 3.7% | 99.1% | 98.2% |
| ppo_auto_jump | 3 | 85.0% ± 26.0% | 15.0% | 95.9% | 87.0% |
| ppo_sparse | 1 | 33.3% ± 57.7% | 37.7% | 51.7% | 55.8% |
| ppo_sparse | 2 | 30.0% ± 51.1% | 32.7% | 54.2% | 56.0% |
| ppo_sparse | 3 | 15.0% ± 26.0% | 42.0% | 37.4% | 32.5% |

Sparse reward is goal +20 / death -5 only; shaped return values are not directly comparable to sparse returns.
All variants receive the same interaction budget, curriculum, rehearsal probability, and starting-position distribution.
Auto-jump has three learned directional actions and automatically jumps whenever grounded.
No training-only intermediate-platform starts, demonstration trajectories, or movement controllers are used.
Generalization is measured within this generator's parameter ranges; it does not establish generalization to arbitrary games.

See manifest.json for parameters and layout hashes, learning_curve.csv for learning efficiency and PPO diagnostics,
and each run's test_episodes.csv for individual outcomes. Missing 80% milestones mean the threshold was not reached.
