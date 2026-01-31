import pandas as pd
from .utils import add_kv_pairs

# Model Parameters

def default_parameters() -> pd.Series:
    return pd.Series({
        #Simulation Control
        "T" : 500,
        'Seed': 1,

        #Governance
        "G" : 0.6,

        #Initial Defender State variables
        "it_vuln_init" : 0.6,
        "ot_vuln_init" : 0.7,
        "id_cap_init" : 0.2,
        'it_comp_init' : 0,
        'ot_comp_init' : 0,
        'downtime_init' : 0.0,
        'phys_damage_init' : 0.0,
        'outage_init': 0.0,

        #Attacker event process
        'p_attack': 0.35,
        'p_ot_given_attack_base' : 0.35,
        'p_ot_bonus_if_it_comp' : 0.20,
        'p_ot_bonus_if_ot_high_vuln' : 0.20,
        'ot_high_vuln_threshold': 0.7,

        #Attacker success parameters, as of 1/16/26 these are placeholder values
        'base_success_mult': 1.0,
        'high_success_bonus': 0.25, #high intensity attack have an additive bonus

        #Attack intensity deterrence via identification
        'p_high_base' : 0.50,
        'k_deterrence' : 2.0,

        # Delta parameters for defender actions
        "delta_it_vuln": 0.04,
        "delta_ot_vuln": 0.02,
        "delta_id_cap": 0.03,

        #defender detection and containment parameters (placeholders for now, 1/20/26)
        "p_detect_base": 0.10, #baseline detection probability
        "p_contain_base": 0.20, #baseline containment probability

        #Changes in defender stats based on previous turn actions (PASSIVE, ACTIVE, RECOVER)
        "delta_detect": 0.25,
        "delta_contain": 0.25,
        "active_damage_reduction": 0.35,

        "delta_recover_clear": 0.30,
        "delta_downtime_reduction": 0.40,

        #Damage and downtime dynamics (placeholders until we tune parameter values)
        'base_damage' : 0.02, #per timestep damage done to defender while OT is compromised
        'high_damage_multiplier' : 3.0, #multiplication factor for when attacks are high intensity
        'damage_persistence': 0.95, #1.0 means no decay, if we decrease below 1.0 systems damage will decay over time

        'downtime_comp_cost': 0.05, #cost of downtime increases the longer a system remains compromised
        'downtime_damage_cost' : 0.02, #additional downtime increases per step unit damage
        'downtime_decay': 0.0, #can add a natural recovery function by making downtime_decay > 0.0

        #Recovery probabilities
        'p_recover_clear_base' : 0.10, #chance recovery clears compromise status
        'damage_recover_decay' : 0.05 #fraction of damage removed under RECOVER action
    })

def apply_defaults(P: pd.Series) -> pd.Series:
    P = P.copy()
    outage_defaults = {
        'outage_decay': 0.60,
        'outage_comp_cost': 0.4,
        'outage_damage_cost': 0.2,
    }
    add_kv_pairs(P, outage_defaults)

    policy_defaults = {
        'defender_policy': 'threshold_v1',
        'id_cap_min_threshold': 0.30,
        'phys_damage_threshold': 0.50,
        'outage_high_threshold': 0.60,
    }
    add_kv_pairs(P, policy_defaults)

    rl_defaults = {
    #hyperparameters for agent learning, adjustable during parameter sweeps
    'rl_alpha': 0.15,
    'rl_gamma': 0.95,
    'rl_epsilon': 0.20,

    #discretization values, this step allows for us to use the tabular q-learning technique with continuous data
    'rl_id_cap_lo' : 0.33,
    'rl_id_cap_high': 0.66,
    'rl_damage_lo': 0.25,
    'rl_damage_high': 0.75,
    'rl_outage_lo': 0.25,
    'rl_outage_high': 0.60,

    #reward weights, specifically measured in an actions ability to minimize loss per/step per action (reward = -loss)
    'rl_w_damage_step': 5.0,  # physical damage to infrastructure highly penalizes reward return
    'rl_w_outage': 2.0,       # penalty for interruption to infrastructure services
    'rl_w_ot_comp': 2.0,      # penalty for OT compromise
    'rl_w_it_comp': 0.5,     # small IT compromise penalty, since this does not directly influence infrastructure only allows access for further damage
    #I realized damage per step is penalized but accumulated damage is not penalized, this needs to be accounted for in reward weighting
    'rl_w_phys_damage': 2.0,  #penalize the infrasturctures accumulated damage stock

    'rl_learn' : 1,     # 1 = training mode, 0 = evaluation mode (stops updating)

    #added small costs to certain action because I found under certain parameter values the agent would continuously recover
    'rl_cost_active': 0.05,
    'rl_cost_recover': 0.10,
}
    add_kv_pairs(P, rl_defaults)

    return P




    