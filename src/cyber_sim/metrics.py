import numpy as np
from .sim import run_sim
from .state import make_initial_state


def summarize_run(df):
  # general summaries
  out = {}
  out['mean_reward'] = float(df['rl_reward'].mean()) if 'rl_reward' in df.columns else np.nan
  out['mean_outage'] = float(df['outage_next'].mean())
  out['mean_damage_step'] = float(df['damage_step'].mean())
  out['time_it_comp'] = float(df['it_comp_end'].mean())
  out['time_ot_comp'] = float(df['ot_comp_end'].mean())
  out['action_freq'] = (df['action_name'].value_counts(normalize=True)).to_dict()
  out['q_size_end'] = float(df['q_size'].iloc[-1]) if 'q_size' in df.columns and not df['q_size'].isna().all() else np.nan
  return out


def rolling_action_freq(df, window=500):
  # log for frequency of actions over time
  a = df['action_name']
  idx = np.arange(len(df))
  buckets = (idx // window)
  return (df.assign(bucket=buckets)
            .groupby(['bucket','action_name'])
            .size()
            .groupby(level=0)
            .apply(lambda s: (s / s.sum()))
            .unstack(fill_value=0.0))


#evaluate qlearning effectiveness under high vs low threat (different than attack intensity, basically just hard coding a probability of an attack occuring to examine 'high' and 'low' attack threat conditions)
def eval_high_vs_low_threat(P, q_agent, p_attack, seed=123, T=25000):
    P2 = P.copy()
    P2['defender_policy'] = 'qlearn_v1'
    P2['rl_learn'] = 0
    P2['rl_epsilon'] = 0.0
    P2['p_attack'] = p_attack
    P2['Seed'] = seed
    rng = np.random.default_rng(int(P2['Seed']))
    df = run_sim(P2, make_initial_state(P2), rng)
    return summarize_run(df), df['action_name'].value_counts(normalize=True).to_dict()

#compare threshold and random policy performance to qlearning performance in high and low threat conditions
def eval_policy_under(P, policy, p_attack, seed=123, T=25000):
    P2 = P.copy()
    P2['defender_policy'] = policy
    P2['p_attack'] = p_attack
    P2['Seed'] = seed
    rng = np.random.default_rng(int(P2['Seed']))
    df = run_sim(P2, make_initial_state(P2), rng)
    return summarize_run(df)