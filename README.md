# Strategic Cyber Defense Simulation

This project models cyber attack and defense interactions across IT and OT layers over discrete timesteps.

## Documentation

- [System Design](docs/SYSTEM_DESIGN.md)

The system design document explains:
- How the simulation loop runs at each timestep
- What state variables and actions drive outcomes
- How attacker, defender, and recovery dynamics are modeled
- How `threshold_v1` and `qlearn_v1` defender policies work
- What logs and metrics are produced for analysis

## Quick Run

Train and evaluate the tabular Q-learning policy:

```bash
PYTHONPATH=src python scripts/train_qlearn.py
```

