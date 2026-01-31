from .enums import Action
from .state import gov_mult
from .utils import clip01
from .rl import discretize_state

import pandas as pd

def init_boosts():
    return pd.Series({
        "detect_boost": 0.0,
        "contain_boost": 0.0,
        "recover_clear_boost": 0.0,
        "downtime_reduction_boost": 0.0,
        "active_damage_reduction": 0.0
    })

def apply_defender_action(Parameters, State, action):
    gm = gov_mult(Parameters)
    B = init_boosts()

    if action == Action.PASSIVE:
        State["it_vuln"] = clip01(State["it_vuln"] - gm * Parameters["delta_it_vuln"])
        State["ot_vuln"] = clip01(State["ot_vuln"] - gm * Parameters["delta_ot_vuln"])
        State["id_cap"]  = clip01(State["id_cap"]  + gm * Parameters["delta_id_cap"])

    elif action == Action.ACTIVE:
        B["detect_boost"] = gm * Parameters["delta_detect"]
        B["contain_boost"] = gm * Parameters["delta_contain"]
        B["active_damage_reduction"] = clip01(gm * Parameters["active_damage_reduction"])

    elif action == Action.RECOVER:
        B["recover_clear_boost"] = gm * Parameters["delta_recover_clear"]
        B["downtime_reduction_boost"] = clip01(gm * Parameters["delta_downtime_reduction"])

    else:
        raise ValueError(f"Invalid action: {action}")

    return B

def choose_action(Parameters, State, rng, t, agent = None):
  """
  Function that uses defender policy to determine which action the defender will choose each time step. There are currently three policies we can have the defender implement:
  1. always_passive: the current baseline/placeholder policy in which the defender just plays PASSIVE no matter what
  2. random: a policy in which the defender uses a uniform, random dist. to pick the three actions (PASSIVE, ACTIVE, RECOVER) at each time step.
  3. threshold_v1: policy which uses a simple heuristic to determine action selection based on parameter thresholds.
  """

  policy = Parameters.get('defender_policy', 'always_passive')

  if policy == 'always_passive':
    return Action.PASSIVE

  if policy == 'random':
    return Action(int(rng.integers(0, 3))) #if 0: Aaction.PASSIVE, if 1: Action.ACTIVE, if 2: Action.RECOVER

  if policy == 'threshold_v1':
    it_comp = int(State['it_comp'])
    ot_comp = int(State['ot_comp'])
    id_cap = float(State['id_cap'])
    phys_damage = float(State['phys_damage'])
    outage = float(State['outage'])

    id_low = float(Parameters.get('id_cap_min_threshold', 0.30))
    dmg_high = float(Parameters.get("phys_damage_threshold", 0.50))
    outage_high = float(Parameters.get('outage_high_threshold', 0.60))

    #Priority 1: if OT is compromised from previous timestep attacks, RECOVER
    if ot_comp == 1:
      return Action.RECOVER

    # Priority 2, if infrastructure has high physical damage or experiences significant downtime, RECOVER
    if phys_damage >= dmg_high or outage >= outage_high:
      return Action.RECOVER

    #Priority 3: if IT layer is compromised, implement ACTIVE response action
    if it_comp == 1:
      return Action.ACTIVE

    #Priority 4: If ability to identify attacker is low, use PASSIVE action to improve capabilities
    if id_cap < id_low:
      return Action.PASSIVE

    # Otherwise, invest in long-term defensive assets
    return Action.PASSIVE

  if policy == 'qlearn_v1':
    if agent is None:
            raise ValueError("Q-learning policy requires an agent instance")

    s = discretize_state(Parameters, State)
    action = agent.select_action(s, float(Parameters['rl_epsilon']), rng)
    return Action(action)

  raise ValueError(f"Unknown defender_policy: {policy}")