import pickle
import numpy as np
with open('fluid_tables/water_nitrogen_mixture_120x120_mix_94.48.pkl', 'rb') as f:
    table = pickle.load(f)
if 'metadata' in table:
    print('metadata:', table['metadata'])
if 'enthalpy' in table:
    h = np.array(table['enthalpy']['value'])
    print(f'h range in table: [{h.min()}, {h.max()}]')
    print(f'h shape: {h.shape}')
if 'pressure' in table:
    p = np.array(table['pressure']['value'])
    print(f'p range in table: [{p.min()}, {p.max()}]')
    print(f'p shape: {p.shape}')
if 'temperature' in table:
    T = np.array(table['temperature']['value'])
    print(f'T range in table: [{T.min()}, {T.max()}]')
print('top-level keys:', [k for k in table.keys()])
