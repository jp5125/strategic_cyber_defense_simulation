# System Design: Strategic Cyber Defense Simulation

## 1. Purpose

This model simulates repeated cyber conflict between:
- A defender managing IT/OT risk with one action per timestep
- An attacker choosing whether to attack, where to attack (IT or OT), and attack intensity

The simulator is intended for:
- Policy comparison (`always_passive`, `random`, `threshold_v1`, `qlearn_v1`)
- Learning adaptive defense behavior with tabular Q-learning
- Measuring security and resilience outcomes over time

---

## 2. High-Level Architecture

Core modules in `src/cyber_sim/`:
- `parameters.py`: Default configuration and policy/RL defaults
- `state.py`: State initialization and state snapshots
- `sim.py`: Main timestep loop and run orchestration
- `attacker.py`: Attack generation and attack resolution
- `defender.py`: Defender action logic and policy selection
- `dynamics.py`: Detection/containment, damage, downtime, outage, recovery
- `rl.py`: State discretization, reward, Q-learning update, Q-table agent
- `metrics.py`: Run summaries and action-frequency diagnostics

Training/evaluation entrypoint:
- `scripts/train_qlearn.py`

---

## 3. Model State

Continuous state variables:
- `it_vuln`: IT vulnerability level, clipped to `[0, 1]`
- `ot_vuln`: OT vulnerability level, clipped to `[0, 1]`
- `id_cap`: Identification capability, clipped to `[0, 1]`
- `downtime`: Service downtime stock
- `phys_damage`: Physical OT damage stock
- `outage`: Outage level, clipped to `[0, 1]`

Binary compromise flags:
- `it_comp`: IT compromised (`0`/`1`)
- `ot_comp`: OT compromised (`0`/`1`)

Governance modifier:
- `G` controls a multiplier `gov_mult = 0.5 + 0.5*G`, amplifying defender action effects.

---

## 4. Defender Action Space

Available actions (`Action` enum):
- `PASSIVE`: Invest in baseline hardening (lower vulnerabilities, raise identification capability)
- `ACTIVE`: Improve detect/contain probability and reduce damage during compromise
- `RECOVER`: Attempt compromise clearance and reduce downtime/damage

Action choice is policy-driven in `defender.py`.

---

## 5. Attacker Model

Each timestep in `attacker.py`:
1. Attack occurrence sampled from `p_attack`
2. Attack target (`IT` vs `OT`) sampled from `p_ot_given_attack_base`
   - Bonus toward OT targeting if IT is already compromised
   - Bonus toward OT targeting if OT vulnerability exceeds a threshold
3. Attack intensity (`LOW`/`HIGH`) sampled from:
   - `p_high = p_high_base * exp(-k_deterrence * id_cap)`
   - Higher `id_cap` deters high-intensity attacks
4. Attack success probability:
   - Based on target vulnerability (`it_vuln` or `ot_vuln`)
   - Plus `high_success_bonus` for high-intensity attacks
   - Clipped to `[0, 1]`
5. Successful attacks set `it_comp` or `ot_comp` to `1`

---

## 6. Timestep Execution Order

The model loop (`sim_step`) runs this sequence each timestep:
1. Snapshot pre-action state for logging
2. Defender selects action from configured policy
3. Defender action applies immediate state/boost changes
4. Attacker event sampled (target + intensity)
5. Attack resolved and compromise flags updated
6. Detection and containment phase attempts to clear compromise
7. OT physical damage update if OT remains compromised
8. Downtime update from compromise + damage
9. Recovery step (if `RECOVER`) may clear compromise and reduce damage
10. Damage persistence/decay applied
11. Outage state updated
12. RL reward/update step (only for `qlearn_v1`)
13. Full row appended to run log

This ordering ensures policy decisions occur before threat realization and that response/recovery effects are reflected in the same timestep.

---

## 7. Policy Modes

Implemented defender policies:
- `always_passive`: Always uses `PASSIVE`
- `random`: Uniform random action selection
- `threshold_v1`: Rule-based heuristic prioritizing:
   - Recover if OT compromised
   - Recover if damage/outage exceeds thresholds
   - Active if IT compromised
   - Passive if identification capability is below threshold
   - Otherwise passive investment
- `qlearn_v1`: Tabular Q-learning policy (`epsilon`-greedy during training, greedy eval)

---

## 8. Q-Learning Design

### 8.1 Discretized State

To support tabular RL, continuous variables are binned:
- `id_cap`: 3 bins
- `phys_damage`: 3 bins
- `outage`: 3 bins

Combined with binary compromise flags:
- `(it_comp, ot_comp, id_cap_bin, damage_bin, outage_bin)`
- Total state space: `2 * 2 * 3 * 3 * 3 = 108` states

### 8.2 Reward

Step reward is negative weighted loss:
- Damage this step
- Current outage
- IT compromise indicator
- OT compromise indicator
- Accumulated physical damage
- Optional action costs (`ACTIVE`, `RECOVER`)

### 8.3 Learning Rule

Q-table update uses standard temporal-difference target:
- `Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

Learning can be toggled with `rl_learn`:
- `1`: training mode (update table)
- `0`: evaluation mode (freeze table)

---

## 9. Output and Observability

Each run returns a pandas DataFrame with per-timestep observability, including:
- Action and policy effects
- Attack details (target, intensity, probabilities, success)
- Detection/containment outcomes
- Damage, downtime, outage progression
- Pre/post compromise status
- RL reward and Q-table size (for RL runs)

`metrics.py` provides:
- Aggregate run summary (means, compromise duration, action frequencies)
- Rolling action-frequency diagnostics for behavior analysis over time

---

## 10. Configuration Surface

Primary configuration groups in `parameters.py`:
- Simulation control: horizon (`T`), seed
- Governance effect: `G`
- Initial state values
- Attack process probabilities
- Detection/containment and recovery probabilities
- Damage/downtime/outage dynamics
- Policy thresholds
- RL hyperparameters, discretization bins, reward weights, action costs

Defaults are created by:
- `default_parameters()`
- `apply_defaults()` (adds outage, policy, and RL default sets)

---

## 11. Execution Entry Point

Use `scripts/train_qlearn.py` to:
1. Train `qlearn_v1` for `train_steps`
2. Evaluate trained policy greedily
3. Compare against `threshold_v1`, `always_passive`, and `random`
4. Run low/high threat checks by changing `p_attack`

Example:

```bash
PYTHONPATH=src python scripts/train_qlearn.py --train_steps 100000 --eval_steps 25000
```

---

## 12. Current Model Scope

Important current assumptions:
- Single attacker and single defender actor
- Fixed action set of size 3
- Compromise modeled as binary per layer (no partial compromise)
- Tabular RL with hand-crafted discretization
- Placeholder-calibrated parameters in several areas (not yet empirically fitted)

This scope makes the system transparent and easy to audit while supporting controlled policy and reward experiments.
