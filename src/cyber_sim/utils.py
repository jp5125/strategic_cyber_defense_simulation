import numpy as np

#clip01 keeps all continuous variables in the simulation bounded between [0,1]
def clip01(x):
  return np.clip(x, 0.0, 1.0)

def add_kv_pairs(P, kvs):

  items = kvs.items() if hasattr(kvs, 'items') else kvs
  for k, v in items:
        if k not in P.index:
            P[k] = v