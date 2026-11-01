"""e4_ensemble.py -- average predictions across training seeds (same held-out landmarks) -> ensemble run dir."""
import glob, os, sys
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
seeds = sorted(glob.glob(os.path.join(root, 'seed*')))
os.makedirs(out, exist_ok=True)
tags = sorted({os.path.basename(f) for s in seeds for f in glob.glob(os.path.join(s, 'pred_*.parquet'))})
key = ['NACCID', 'NACCVNUM', 'model']
for t in tags:
    parts = [pd.read_parquet(os.path.join(s, t)) for s in seeds if os.path.exists(os.path.join(s, t))]
    if len(parts) < len(seeds):
        print('incomplete', t, len(parts), flush=True)
    P = pd.concat(parts, ignore_index=True)
    fc = [c for c in P.columns if c.startswith('F1_') or c.startswith('F2_')]
    other = [c for c in P.columns if c not in fc + ['seed']]
    meanp = P.groupby(key)[fc].mean().reset_index()
    first = P.drop_duplicates(key)[other]
    E = first.merge(meanp, on=key, how='left')
    E['seed'] = -1
    E['n_seeds'] = len(parts)
    E.to_parquet(os.path.join(out, t), index=False)
print('done', len(tags))
