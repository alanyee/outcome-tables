import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(torch.__version__, device, dtype)

import numpy as np
import functools
import itertools
import math
import statistics

# fast exponentiation and binary search
@functools.lru_cache(maxsize=20)
def matrix_pow(a, p):
  """Matrix exponentation a^p in O(d^3 log n)"""

  assert p > 0
  if p == 1: return a

  u = matrix_pow(a, p>>1)
  u @= u
  if p&1:
    u @= a
  return u

def binary_search(f, start, stop):
  """Binary search for the point at which f() becomes true. O(log n)"""

  l, r = start, stop
  while l < r:
    m = (l + r)//2
    if f(m):
      r = m
    else:
      l = m+1

  return r

# markov chain
def markov(
        t, *,
        p_base, pity_timer, p_category, p_card, desired=1, multiple_cards=1,
        _matrix_cache={}
):
  """Returns the probability of having `desired` number of cards after `t` pulls
  If `t` is None, instead return the mean number of pulls to obtain `desired` cards

  p_base := p of rarity (0.075 for SR, 0.025 for UR)
  p_category := p of featured card vs master pack card
                (Only matters for secret pack. Make this 0.5 for Secret Pack otherwise 1)
  p_card := p of card from within rarity pool (e.g. Maxx C = 1/964 UR in Master Pack,
            Nadir Servant = 1/13 UR in Beginning of the Next Journey Selection Pack)
  pity_timer := 160 for UR and 80 for SR

  multiple_cards := number of different cards desired (looking for `desired` copies each)
  """

  # here the Markov state vector is a flattened [x, y, z] array, where
  # x := copies of desired card received
  # y := "pity" counter: pulls since last card of matching rarity
  #
  # if multiple_cards > 1, then x is further sub-arrayed as [x_1, x_2, ..., x_n] where n = multiple_cards

  d1d = desired + 1
  d1m = multiple_cards
  d1 = d1d ** d1m
  if pity_timer != 160:
    d3 = 159 # constant since SR on its own is not guaranteed
  else:
    d3 = pity_timer
  d = d1 * d3
  shape = (*(d1d,)*d1m, d3)
  stride = d3

  matrix_cache_key = (p_base, pity_timer, p_category, p_card,
                      desired, multiple_cards)

  if matrix_cache_key in _matrix_cache:
    transition = _matrix_cache[matrix_cache_key]
  else:
    _matrix_cache.clear()

    transition = torch.zeros((d, d), device=device, dtype=dtype)
    transition[-stride:, -stride:] = torch.eye(stride, device=device, dtype=dtype)

    for i in range(d - stride):
      # i_tuple := i in index-tuple form
      i_tuple = np.array(np.unravel_index(i, shape))
      i_tuple.flags.writeable = False

      # let the vector u be the i-th column of the transition matrix
      # we construct u as a tensor and then flatten it into a vector to copy into the matrix
      u = torch.zeros(shape, device=device, dtype=dtype)


      # let p110 denote "rarity hit, category hit, card miss"
      # similar for p0, p1, p10, p11, etc

      # p1 := rarity hit
      pity_function = {
        80: {80: 0.8},
        160: {160: 1.0, 80: 0.2},
      }

      y = pity_function[pity_timer].get(i_tuple[-1] + 1)
      if y is not None:
        p1 = y
      else:
        p1 = p_base

      p0 = 1.0 - p1

      # p11 := category hit
      p11 = p1 * p_category

      p10 = p1 - p11

      # p111 := card hit
      p111 = p11 * p_card * multiple_cards
      p110 = p11 - p111

      # p111x := p of each card if multiple cards
      p111x = p111 / multiple_cards


      # let j0 through j110 be destination Markov state index-tuples, corresponding to above p0 through p110
      # let j111x[0], j111x[1], j111x[2], ... be destinations for each different card if multiple cards

      prev_d1 = tuple(i_tuple[:-1]) + (0,)

      j0 = tuple(i_tuple + [*(0,)*d1m, +1])
      j111x = []

      j10 = prev_d1
      j110 = prev_d1
      for k in range(d1m):
        j = list(prev_d1)
        j[k] = min(j[k] + 1, desired)
        j111x.append(tuple(j))

      def u_add(j, p):
        if p == 0:
          return None
        try:
          u[j] += p
        except IndexError:
          pass
      u_add(j0   , p0   )
      u_add(j10  , p10  )
      u_add(j110 , p110 )
      for j in j111x:
        u_add(j, p111x)

      # verify each state's outgoing edges sum to 1
      # assert math.isclose(s := u.sum(), 1.0, rel_tol=1e-5), f'column {i} sum of p is not 1 = {s:.6f}\n{u}'

      transition[:, i] = u.flatten()


    _matrix_cache[matrix_cache_key] = transition


  state = torch.zeros(d, device=device, dtype=dtype)
  state[0] = 1.0

  if t is not None:
    state = matrix_pow(transition, t) @ state
    return state[-stride:].sum().item()
  else:
    mean = 0.0
    pf = 0.0
    for i in itertools.count(start=1):
      state = transition @ state
      delta_pf = state[-stride:].sum().item() - pf
      mean += delta_pf * i
      pf += delta_pf
      if pf > 0.50 and math.isclose(delta_pf, 0, abs_tol=1e-7):
        return mean / pf

# mean and percentile
def percentile(pctl, **kwarg):
  return binary_search(lambda t: markov(t, **kwarg) >= pctl, 1, 1<<20)

def percentiles(pctls, **kwarg):
  return [percentile(pctl, **kwarg) for pctl in pctls]

def ci_90(**kwarg):
  """Return [t5, t95], the 5th and 95th percentile finish t's for the given parameters"""
  return percentiles([0.05, 0.95], **kwarg)

def ci_90_98(**kwarg):
  return percentiles([0.01, 0.05, 0.95, 0.99], **kwarg)

def mean(**kwarg):
  """Return the mean finish t for the given parameters"""
  return markov(t=None, **kwarg)

# Example: Distribution of # of pulls to obtain 3 copies of a specific SR card in Selection Pack
percentiles([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999],\
                  p_base=0.075, pity_timer=80, p_category=1, p_card=1/20,\
                  desired=3)

# Example: Distribution of # of pulls to obtain 3 copies of a specific SR card in Selection Pack
percentiles([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999],\
                  p_base=0.075, pity_timer=80, p_category=1, p_card=1/20,\
                  desired=3)

# Example: Distribution of # of pulls to obtain 3 copies of specific UR card in Selection Pack
percentiles([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999],\
                  p_base=0.025, pity_timer=160, p_category=1, p_card=1/11,\
                  desired=3)

# Example: Distribution of # of pulls to obtain 1 copy of all UR cards in Selection Pack
percentiles([0.01, 0.05, 0.50, 0.95, 0.99],\
                  p_base=0.025, pity_timer=160, p_category=1, p_card=1/13,\
                  desired=1, multiple_cards=13)

# Example: Distribution of # of pulls to obtain 3 copies of 3 UR cards in Selection Pack
percentiles([0.01, 0.05, 0.50, 0.95, 0.99],\
                  p_base=0.025, pity_timer=160, p_category=1, p_card=1/13,\
                  desired=3, multiple_cards=3)

# Example: Distribution of # of pulls to obtain a copy of all UR cards in Secret Pack with only 5 URs (e.g. Bystial Secret Pack)
percentiles([0.01, 0.05, 0.50, 0.95, 0.99],\
                  p_base=0.025, pity_timer=160, p_category=0.5, p_card=1/5,\
                  desired=1, multiple_cards=5)

# Example: Mean and distribution of # of pulls to obtain 3 copies of a specific UR card in Selection Pack
kwarg = dict(p_base=0.025, pity_timer=160, p_category=1, p_card=1/13, desired=3)

time m = mean(**kwarg)
print(kwarg, end='\n\n')
print(f'mean = {m}', end='\n\n')

pctls = [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
print(*pctls, sep='\t')
print(*percentiles(pctls, **kwarg), sep='\t')

def ten_pack_conversion(dividend, divisor=80):
  return math.ceil(dividend / divisor)

# generate CSV table
def table_row(banner, rarity, p_category, p_card, multiple_cards, up_to, label):
  partial_kwarg = dict(
    p_base={"SR": 0.075, "UR": 0.025}[rarity],
    pity_timer={"SR": 80, "UR": 160}[rarity],
    p_category=p_category,
    p_card=p_card,
    multiple_cards=multiple_cards
  )

  means = []
  ci_s = []
  means_10pack = []
  ci_s_10pack = []
  mean_intermediary = None
  ci_s_intermediary = None
  for c in range(1, up_to + 1):
    mean_intermediary = round(mean(**partial_kwarg, desired=c))
    ci_s_intermediary = ci_90_98(**partial_kwarg, desired=c)
    means += ['{:,}'.format(mean_intermediary)]
    ci_s += ['{:,} - {:,} ... {:,} - {:,}'.format(*ci_s_intermediary)]
    means_10pack += ['{:,}'.format(ten_pack_conversion(round(mean(**partial_kwarg, desired=c))))]
    ci_s_10pack += ['{:,} - {:,} ... {:,} - {:,}'.format(*[ten_pack_conversion(ci) for ci in ci_s_intermediary])]
  pad = 3 - up_to
  means += [''] * pad
  ci_s += [''] * pad
  print(banner, label, *means, sep=';')
  print('', '', *ci_s, sep=';')
  print(banner, label + " (in 10 pack pulls)", *means_10pack, sep=';')
  print('', '', *ci_s_10pack, sep=';')

table_row('select' , "SR", 1, 20/20, 1, 1, 'any SR from Selection Pack')
table_row('select' , "SR", 1, 1/20 , 1, 3, 'specific SR from Selection Pack')
table_row('select' , "SR", 1, 1/20 , 2, 3, 'specific two SRs from Selection Pack')
table_row('select' , "SR", 1, 1/20 , 3, 3, 'specific three SRs from Selection Pack')

print(';' * 8)
table_row('select' , "UR", 1, 13/13 , 1, 1, 'any UR from Selection Pack')
table_row('select' , "UR", 1, 1/13 , 1, 3, 'specific UR from Selection Pack')
table_row('select' , "UR", 1, 1/13 , 2, 3, 'specific two URs from Selection Pack')
table_row('select' , "UR", 1, 1/13 , 3, 3, 'specific three URs from Selection Pack')
print(';' * 8)
table_row('secret', "SR", 0.5, 10/10, 1, 1, 'any featured SR from Secret Pack')
table_row('secret', "SR", 0.5, 1/10 , 1, 3, 'specific featured SR from Secret Pack')
table_row('secret', "UR", 0.5, 8/8  , 1, 1, 'any featured UR from Secret Pack')
table_row('secret', "UR", 0.5, 1/8  , 1, 3, 'specific featured UR from Secret Pack')
table_row('secret', "UR", 0.5,  1/976, 1, 3, 'specific non-featured UR from Secret Pack')
print(';' * 8)
table_row('master', "SR", 1, 1831/1831, 1, 1, 'any SR from Master Pack')
table_row('master', "SR", 1, 1/1831 , 1, 3, 'specific SR from Master Pack')
print(';' * 8)
table_row('master', "UR", 1, 976/976, 1, 1, 'any UR from Master Pack')
table_row('master', "UR", 1, 1/976  , 1, 3, 'specific UR from Master Pack')
print()

# debug info
if device.type == 'cuda':
  print(f'peak GPU RAM usage: {torch.cuda.max_memory_allocated():,}')
  torch.cuda.reset_peak_memory_stats()
