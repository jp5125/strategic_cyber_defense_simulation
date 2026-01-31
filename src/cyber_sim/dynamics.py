from .utils import clip01
from .enums import Action, Intensity

def detect_and_contain_one(Parameters, State, rng, comp_key, detect_boost, contain_boost):
  """If the state of the defender's infrastructure this time step is compromised we do the following:
     1. the defender detects the compromise with p_detect = p_detect_base + detect_boost
     2. if detection is successful, we then determine if the damage can be contained with P-contain = p_contain_base + contain_boost
     3. if contained, set State[comp_key] == 0
     *** comp key can be either it_comp or ot_comp

     Returns: (detected_flag, contained_flag)
  """

  if int(State[comp_key]) != 1:
    return 0,0

  p_detect = clip01(Parameters['p_detect_base'] + float(detect_boost))
  detected = 1 if rng.random() < p_detect else 0

  if detected == 0:
    return 0, 0

  p_contain = clip01(Parameters['p_contain_base'] + float(contain_boost))
  contained = 1 if rng.random() < p_contain else 0

  if contained == 1:
    State[comp_key] = 0

  return detected, contained

def detection_and_containment_step(Parameters, State, rng, B):
  """This function applies detection and containment logic for IT and OT compromises.
    It returns a dict of outcomes for logging and data analysis
  """

  it_detected, it_contained = detect_and_contain_one(Parameters, State, rng, comp_key = 'it_comp', detect_boost = B['detect_boost'], contain_boost = B['contain_boost'])

  ot_detected, ot_contained = detect_and_contain_one(Parameters, State, rng, comp_key = 'ot_comp', detect_boost = B['detect_boost'], contain_boost = B['contain_boost'])

  return{
      "it_detected" : it_detected,
      "it_contained": it_contained,
      "ot_detected": ot_detected,
      "ot_contained": ot_contained,
      "it_comp_post": int(State['it_comp']), #it_compromise status after detection and containment is run
      "ot_comp_post": int(State['ot_comp']) #ot_compromise status after detetcion and containment is run this time step
  }

def ot_physical_damage_step(Parameters, State, intensity, B):
  """If the OT layer is compromised, we add physical damage to the defender infrastructure
  1. start with base damage per step for low intensity atacks
  2. add multiplication factor for high intensity attacks
  3. reduce damage dealt based on the defenders' ACTIVE damage reduction boost
  *** returns damage for logging
  """

  if int(State['ot_comp']) != 1:
    return 0.0

  damage = float(Parameters['base_damage'])
  if intensity == Intensity.HIGH:
    damage *= float(Parameters['high_damage_multiplier'])

  #active defense boosts reduce the damage done to defender
  damage *= (1.0 - float(B['active_damage_reduction']))

  #apply damage to the defender's systems and ensure this value is not negative
  State["phys_damage"] = max(0.0, float(State['phys_damage']) + damage)
  return float(damage)

def downtime_update_step(Parameters, State, B, action):
  """Updates the downtime the defender has experienced thus far in the sim
  1. increases downtime value if a compromise is present
  2. further increases with accumulated physical damage
  3. RECOVER reduces downtime via boosts which are applied in this function
  """

  comp_present = int(int(State['it_comp']) == 1 or int(State['ot_comp']) == 1)

  dt_counter = 0.0
  dt_counter += float(Parameters['downtime_comp_cost'] * comp_present)
  dt_counter += float(Parameters['downtime_damage_cost']) * float(State['phys_damage'])

  #optional natural decay of downtime (currently set to 0)
  dt = float(State['downtime']) + dt_counter
  dt = max(0.0, dt - float(Parameters.get('downtime_decay', 0.0)))

  #If recover is the defenders chosen action this turn, apply a downtime reduction boost
  if action == Action.RECOVER:
    dt = max(0.0, dt * (1.0 - float(B["downtime_reduction_boost"])))

  State['downtime'] = float(dt)
  return float(dt_counter)

def outage_update_step(Parameters, State):
  comp_present = int(int(State['it_comp']) == 1 or int(State['ot_comp']) == 1)

  out = 0.0
  out += float(Parameters.get('outage_comp_cost', 0.40)) * float(comp_present)
  out += float(Parameters.get('outage_damage_cost', 0.20)) * float(State['phys_damage'])

  decay = float(Parameters.get('outage_decay', 0.60))
  prev = float(State['outage'])

  new_outage = (1.0 - decay) * prev + out
  State['outage'] = float(clip01(new_outage))

  return float(out)

def recovery_resolution_step(Parameters, State, rng, B, action):
  """If RECOVER is the chosen action of the defender:
  1. probabilistically clear IT/OT compromise even if undetected
  2. optionally, reduce accumulated damage a bit
  """

  out = {
      "recovery_it_cleared": 0,
      'recovery_ot_cleared': 0,
      'damage_reduction': 0.0
  }

  if action != Action.RECOVER:
    return out

  p_clear = clip01(float(Parameters['p_recover_clear_base']) + float(B['recover_clear_boost']))

  if int(State['it_comp']) == 1 and (rng.random() < p_clear):
    State['it_comp'] = 0
    out['recovery_it_cleared'] = 1

  if int(State['ot_comp']) == 1 and (rng.random() < p_clear):
    State['ot_comp'] = 0
    out['recovery_ot_cleared'] = 1

  #logic for implementing an optional modest damage reduction under the RECOVER action
  frac = clip01(float(Parameters.get('damage_recover_decay', 0.0)))
  if frac > 0:
    before = float(State['phys_damage'])
    after = max(0.0, before * (1.0 - frac))
    State['phys_damage'] = after
    out['damage_reduction'] = float(before - after)

  return out