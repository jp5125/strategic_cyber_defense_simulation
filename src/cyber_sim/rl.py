import numpy as np
from .enums import Action


def rl_bin(x, lo, high):
  """
  Function that helps discretize state values based on discretization thresholds established in rl_defaults
  0 = little to no damage/compromise to the defender systems
  1 = signifcant damage/compromise to the defender systems
  2 = catastrophic damage/compromise to the defender systems
  """
  if x < lo : return 0
  if x < high: return 1
  return 2

def discretize_state(Parameters, State):
  """
  Discretized State Tuple:
  (it_comp = 2, ot_comp = 2, id_cap_bin = 3, damage_bin = 3, outage_bin =3)
  2 x 2 x 3 x 3 x 3 = 108 possible states in which our defender agent needs learn to make action decisions in
  """

  it_c = int(State['it_comp'])
  ot_c = int(State['ot_comp'])

  id_c_discrete = rl_bin(float(State['id_cap']), float(Parameters['rl_id_cap_lo']), float(Parameters['rl_id_cap_high']))
  damage_discrete = rl_bin(float(State['phys_damage']), float(Parameters['rl_damage_lo']), float(Parameters['rl_damage_high']))
  outage_discrete = rl_bin(float(State['outage']), float(Parameters['rl_outage_lo']), float(Parameters['rl_outage_high']))

  return(it_c, ot_c, id_c_discrete, damage_discrete, outage_discrete)

def rl_step_reward(Parameters, damage_step, phys_damage_next, outage_next, it_comp_end, ot_comp_end, action):
  loss = 0.0
  loss += float(Parameters['rl_w_damage_step']) * float(damage_step)
  loss += float(Parameters['rl_w_outage']) * float(outage_next)
  loss += float(Parameters['rl_w_it_comp']) * float(it_comp_end)
  loss += float(Parameters['rl_w_ot_comp']) * float(ot_comp_end)
  loss += float(Parameters['rl_w_phys_damage']) * float(phys_damage_next) #account for accumulated damage in the reward calculation

  #reduce reward for step by action cost parameter
  cost = 0.0
  if action == Action.ACTIVE:
    cost += float(Parameters.get('rl_cost_active', 0.0))
  if action == Action.RECOVER:
    cost += float(Parameters.get('rl_cost_recover', 0.0))

  return -(float(loss) + float(cost))

def qlearn_update_step(Parameters, State, agent, s_pre, action, damage_step, it_comp_end, ot_comp_end):
  outage_next = float(State.get('outage', 0.0))
  phys_damage_next = float(State.get('phys_damage', 0.0))
  r = rl_step_reward(Parameters, damage_step, phys_damage_next, outage_next, it_comp_end, ot_comp_end, action)

  s_post = discretize_state(Parameters, State)

  #ensure learning only occurs during training runs
  if int(Parameters.get('rl_learn', 1)) == 1:
    agent.update(s_pre, int(action), r, s_post, alpha = float(Parameters['rl_alpha']), gamma = float(Parameters['rl_gamma']))

  return float(r)


class QLearner:
  """
  Class blueprint for an agent who implements Qlearn policy
  """
  def __init__(self, n_actions = 3):
    self.n_actions = n_actions
    self.Q = {} #array that tracks the defender states for the Q-table

  #if a one of the 108 state tuples isnt in the q-table yet, create a new row for that particular permutation of state variable values and add an array of 0s into the row
  def row(self, s):
    if s not in self.Q:
      self.Q[s] = np.zeros(self.n_actions, dtype = float)
    return self.Q[s]

  #needed to add this since all states are not being touched during training, thus when select_action calls row(), it creates additional states-value pairs during evaluation
  def qvals(self,s):
    return self.Q.get(s, np.zeros(self.n_actions, dtype = float))

  #function to choose whether agent will either explore by randomly selecting a strategy with p = epsilon, or exploit the current best action choice with p = 1 - epsilon
  def select_action(self, s, epsilon, rng):
    if rng.random() < epsilon:
      return int(rng.integers(0, self.n_actions))

    q = self.qvals(s)
    best_actions = np.flatnonzero(q == q.max())
    return int(rng.choice(best_actions))

  #update the q-value of action under specific state tuple, essentially the Bellman equation
  def update(self, s, a, r, s_next, alpha, gamma):
    q = self.row(s)
    q_next = self.row(s_next)
    td_target = float(r) + float(gamma) * float(np.max(q_next)) # td_target = reward value at the current step plus discounted reward value at next step
    q[a] = q[a] + float(alpha) * (td_target - q[a])