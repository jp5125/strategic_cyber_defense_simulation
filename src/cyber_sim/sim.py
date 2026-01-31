from .state import snapshot_state, make_initial_state
from .rl import discretize_state, qlearn_update_step
from .defender import apply_defender_action, choose_action
from .attacker import sample_attacker_event, resolve_attack, p_high_given_idcap 
from .dynamics import ot_physical_damage_step, downtime_update_step, recovery_resolution_step, outage_update_step, detection_and_containment_step


import pandas as pd
import numpy as np


def sim_step(Parameters, State, rng, t, rows, agent = None):
  """
  Simulation Loop event ordering is as follows:
  1. First, the defender chooses an action (PASSIVE, ACTIVE, or RECOVER) to play for the current timestep
  2. The attacker than chooses whether or not it will attack and at what intensity
  3. We resolve the attack phase and determine if there is any damage to the defenders IT or OT layers
  4. Defender Detection/Containment procedures attempt to detect damage and if detected, mitigate damage
  5. OT damage accumulates if ot_comp remains
  6. downtime of defender infrastructure (represented by ot_layer) updates
  7. If in RECOVER, an additional step may clear compromise and reduce damage missed by the detection/containment step
  8. if policy = qlearn, run update_step method for q-learning
  """
  pre = snapshot_state(Parameters, State, t)

  policy = Parameters.get('defender_policy', 'always_passive')
  s_pre = discretize_state(Parameters, State) if policy == 'qlearn_v1' else None

  #defender action decision
  action = choose_action(Parameters, State, rng, t, agent = agent)

  #Defender action effects
  B = apply_defender_action(Parameters, State, action)

  #Attacker strategy determination
  attack_target, intensity = sample_attacker_event(Parameters, State, rng)

  #Attack resolution
  p_success, attack_success = resolve_attack(Parameters, State, rng, attack_target, intensity)

  #Defender detection and containment step
  dc = detection_and_containment_step(Parameters, State, rng, B)

  #compromise status after detection/containment, but before recovery
  it_comp_post_dc = int(State['it_comp'])
  ot_comp_post_dc = int(State['ot_comp'])

  #damage step
  damage_step = ot_physical_damage_step(Parameters, State, intensity, B)

  #downtime
  downtime_step = downtime_update_step(Parameters, State, B, action)

  #recovery action step (if recovery action is chosen)
  recovery = recovery_resolution_step(Parameters, State, rng, B, action)

  #damage recovery over time (helps Qlearner reach more states, can be rationalized as 'normal maintinence operations')
  State['phys_damage'] = max(0.0, float(State['phys_damage']) * float(Parameters.get('damage_persistence', 1.0)))

  #compromise state after recovery step
  it_comp_end = int(State['it_comp'])
  ot_comp_end = int(State['ot_comp'])

  #system outage state at end of timestep
  outage_status = outage_update_step(Parameters, State)
  outage_end = float(State['outage'])

  #additional learning step for Qlearn policy
  rl_reward = 0.0
  if policy == 'qlearn_v1':
    rl_reward = qlearn_update_step(Parameters, State, agent, s_pre = s_pre, action = action, damage_step = damage_step, it_comp_end = it_comp_end, ot_comp_end = ot_comp_end)


  #Log row for simulation data collection
  row = dict(pre)
  row.update({
     'action' : int(action),
     'action_name' : action.name,

     #log boosts to active defense values
     'detect_boost': float(B["detect_boost"]),
     'contain_boost': float(B["contain_boost"]),
     'recover_clear_boost': float(B['recover_clear_boost']),
     'downtime_reduction_boost': float(B['downtime_reduction_boost']),
     'active_damage_reduction': float(B['active_damage_reduction']),

     #attacker event
     'attack': int(attack_target),
     'attack_name': attack_target.name,
     'intensity': int(intensity),
     'intensity_name': intensity.name,
     'p_high': float(p_high_given_idcap(Parameters, State)),

     #attack outcome values
     "p_success": float(p_success),
     "attack_success": int(attack_success),

      # compromise status after detect/containment (pre-recovery)
     "it_comp_post_dc": it_comp_post_dc,
     "ot_comp_post_dc": ot_comp_post_dc,

     # compromise status at end of timestep (post-recovery)
     "it_comp_end": it_comp_end,
     "ot_comp_end": ot_comp_end,

     #Next state values
     'it_vuln_next' : float(State['it_vuln']),
     'ot_vuln_next' : float(State['ot_vuln']),
     'id_cap_next' : float(State['id_cap']),

     #damage and recovery values
     'damage_step': float(damage_step),
     'phys_damage_next': float(State['phys_damage']),
     'downtime_step': float(downtime_step),
     'downtime_next': float(State['downtime']),
     'outage_status': outage_status,
     'outage_next': outage_end,

     #rl values
     'rl_reward' : rl_reward,
     'q_size': len(agent.Q) if policy == 'qlearn_v1' else np.nan

    })

  row.update(dc)
  row.update(recovery)

  rows.append(row)
  return t + 1 #advance time

def run_sim(Parameters, State, rng, agent = None):
  rows_local = []
  t_local = 0

  for _ in range(int(Parameters['T'])):
    t_local = sim_step(Parameters, State, rng, t_local, rows_local, agent = agent)

  return pd.DataFrame(rows_local)


def run_one(Parameters, seed, policy, T, agent, learn=None, epsilon=None):
  P = Parameters.copy()
  P['Seed'] = int(seed)
  P['T'] = int(T)
  P['defender_policy'] = policy
  if learn is not None:
    P['rl_learn'] = int(learn)
  if epsilon is not None:
    P['rl_epsilon'] = float(epsilon)

  local_rng = np.random.default_rng(int(P['Seed']))
  S0 = make_initial_state(P)
  return run_sim(P, S0, local_rng, agent = agent)