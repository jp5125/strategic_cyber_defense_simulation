from .utils import clip01
from .enums import AttackTarget, Intensity
import numpy as np



def p_high_given_idcap(Parameters, State):
  """calculates the probability that the attacker will use a high intensity attack as a function of the identifying capabilities of the defender entity"""
  # P(HIGH) = p_high_base * exp(-k_deterrence * id_cap)

  return clip01(Parameters['p_high_base'] * np.exp(-Parameters['k_deterrence'] * State['id_cap']))



def sample_attacker_event(Parameters, State, rng):
  """
  After calculating the probability the attacker uses a high-intensity attack this timestep,
  We determine based on this probability if the attacker attacks this turn and if so,
  1. Is the target IT or OT infrastructure layer?
  2. Is the attack high or low intensity?
  """

  if rng.random() > Parameters['p_attack']:
    return(AttackTarget.NONE, Intensity.NONE)

  p_ot = Parameters['p_ot_given_attack_base']
  it_comp_bonus = Parameters['p_ot_bonus_if_it_comp']
  ot_vuln_bonus = Parameters['p_ot_bonus_if_ot_high_vuln']

  if State['it_comp'] == 1:
    if State['ot_vuln'] >= Parameters['ot_high_vuln_threshold']:
      p_ot = p_ot + it_comp_bonus + ot_vuln_bonus
    else:
      p_ot = p_ot + it_comp_bonus
  else:
    if State['ot_vuln'] >= Parameters['ot_high_vuln_threshold']:
      p_ot = p_ot + ot_vuln_bonus
    else:
      p_ot = p_ot

  p_ot = clip01(p_ot)
  target = AttackTarget.OT if rng.random() < p_ot else AttackTarget.IT


  p_high = p_high_given_idcap(Parameters, State)
  intensity = Intensity.HIGH if rng.random() < p_high else Intensity.LOW

  return target, intensity



def attack_success_probability(Parameters, State, attack_target, intensity):
  """
  Once all of the details regarding attacker target and intensity are determined for the current timestep,
  we use this function to determine if the attack is successful.
  """

  #Determine if the defender's relevant vulnerability level for success calculation uses it_vuln or ot_vuln
  if attack_target == AttackTarget.IT:
    vuln = State['it_vuln']
  elif attack_target == AttackTarget.OT:
    vuln = State['ot_vuln']
  else:
    return 0.0

  #Now, we need to calculate the success probability scaled by vulnerability level
  p_success = Parameters['base_success_mult'] * vuln

  #add intensity effect if HIGH intensity attack is happening
  if intensity == Intensity.HIGH:
    p_success += Parameters["high_success_bonus"]

  return float(clip01(p_success))



def resolve_attack(Parameters, State, rng, attack_target, intensity):
  """
  Now we compute the resolution of the attack phase using the
  attack success probability function. We will return the calculated
  probability of success as well as the attack outcome to be logged.
  """

  if attack_target == AttackTarget.NONE:
    return 0.0 , 0

  #save the calculated probability of success as a variable
  p_success = attack_success_probability(Parameters, State, attack_target, intensity)
  success = 1 if rng.random() < p_success else 0

  if success == 1:
    if attack_target == AttackTarget.IT:
      State["it_comp"] = 1
    elif attack_target == AttackTarget.OT:
      State["ot_comp"] = 1

  return p_success, success