"""e4_combine.py -- concatenate aggregate eval CSVs from E4 eval dirs, tagging kind/seed/variant from path."""
import glob, os, sys
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for k in ['metrics', 'pooled', 'center', 'subgroups', 'calib_deciles', 'calib_curves', 'center_calib', 'decision']:
    fs = sorted(glob.glob(os.path.join(root, '**', f'{k}_*.csv'), recursive=True))
    if k == 'center':
        fs = [f for f in fs if not os.path.basename(f).startswith('center_calib')]
    dfs = []
    for f in fs:
        if os.path.getsize(f) <= 1:
            continue
        d = pd.read_csv(f)
        rel = os.path.relpath(os.path.dirname(f), root).split(os.sep)
        d['kind'] = rel[0]
        d['run'] = rel[1] if len(rel) > 1 else ''
        dfs.append(d)
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(os.path.join(out, f'{k}_e4.csv'), index=False)
        print(k, len(dfs))
