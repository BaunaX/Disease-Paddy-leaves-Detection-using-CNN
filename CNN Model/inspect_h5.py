import h5py
import os
p='rice_leaf_disease_model.h5'
if not os.path.exists(p):
    print('HDF5 not found:', p)
    raise SystemExit(1)
with h5py.File(p,'r') as f:
    if 'model_weights' in f:
        print('model_weights keys:')
        for k in f['model_weights'].keys():
            print(' -', k)
            g = f['model_weights'][k]
            for ds in g.keys():
                try:
                    shape = g[ds].shape
                except Exception:
                    shape = None
                print('    ', ds, 'shape=', shape)
    else:
        print('No model_weights group. top keys:', list(f.keys()))
