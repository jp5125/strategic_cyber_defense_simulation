from .utils import clip01
import pandas as pd

def make_initial_state(Parameters):
  return pd.Series({
    'it_vuln' : clip01(Parameters['it_vuln_init']),
    'ot_vuln' : clip01(Parameters['ot_vuln_init']),
    'id_cap' : clip01(Parameters['id_cap_init']),

    #defender compromised treated as boolean, flags [0,1] indicate whether defender IT or OT is compromised
    "it_comp" : int(Parameters['it_comp_init']),
    "ot_comp" : int(Parameters['ot_comp_init']),
    "downtime" : float(Parameters['downtime_init']),
    "phys_damage" : float(Parameters['phys_damage_init']),
    "outage" : float(Parameters['outage_init']),
    })

def gov_mult(Parameters):
  #baseline government multiplier = 0.5 + 0.5 * G
  return 0.5 + 0.5 * clip01(Parameters['G'])

def snapshot_state(Parameters, State, t):
  #returns a dictionary of the pre-action state we wish to record in the log
  return{
      't' : t,
      'G' : float(clip01(Parameters['G'])),
      'gov_mult': float(gov_mult(Parameters)),

      'it_vuln' : float(State['it_vuln']),
      'ot_vuln' : float(State['ot_vuln']),
      'id_cap' : float(State['id_cap']),

      'it_comp' : int(State['it_comp']),
      'ot_comp' : int(State['ot_comp']),
      'downtime' : float(State['downtime']),
      'phys_damage' : float(State['phys_damage']),
      'outage' : float(State['outage'])
  }
