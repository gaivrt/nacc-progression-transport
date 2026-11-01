"""features.py -- visit-level predictor matrix (Methods 2.3) and sequence tensors.

Recoding rules
- -4 (not collected in this form version / packet) -> structurally missing (NaN, mask 0).
- 8/9/88/99/888/999 etc. outside each variable's valid range -> missing (NaN).
- Neuropsychological tests: 95-98 (995-998 for timed tests) 'not completed' -> NaN value plus two binary flags
  per test: <test>_nc_cog (96, cognitive/behavioural reason) and <test>_nc_oth (95 physical, 97 other, 98 refusal).
- One source per condition: subject health history (A5/A5D2) only; clinician-assessed D2 conditions not used.
  NACC-derived medication classes (NACC*). Health-history items are carried forward (LOCF) from the most recent
  visit at which the form was collected, because UDS v3 follow-up packets mostly do not re-collect them.
- Excluded from predictors: race/ethnicity, clinician etiologic diagnoses (NACCETPR, NACCALZP, ...), COGSTAT.
- Versions of tests that changed between UDS versions are kept as separate variables (MMSE vs MoCA etc.).
"""
import numpy as np
import pandas as pd

# (lo, hi) valid range; block
NP12 = {'NACCMMSE': (0, 30), 'LOGIMEM': (0, 25), 'MEMUNITS': (0, 25), 'DIGIF': (0, 12), 'DIGIFLEN': (0, 8),
        'DIGIB': (0, 12), 'DIGIBLEN': (0, 8), 'WAIS': (0, 93), 'BOSTON': (0, 30)}
NP3 = {'MOCATOTS': (0, 30), 'CRAFTVRS': (0, 44), 'CRAFTURS': (0, 25), 'CRAFTDVR': (0, 44), 'CRAFTDRE': (0, 25),
       'UDSBENTC': (0, 17), 'UDSBENTD': (0, 17), 'UDSBENRS': (0, 1), 'DIGFORCT': (0, 14), 'DIGFORSL': (0, 9),
       'DIGBACCT': (0, 14), 'DIGBACLS': (0, 8), 'MINTTOTS': (0, 32), 'UDSVERFC': (0, 40), 'UDSVERLC': (0, 40),
       'UDSVERTN': (0, 80)}
NPSH = {'ANIMALS': (0, 77), 'VEG': (0, 77), 'TRAILA': (0, 150), 'TRAILARR': (0, 40), 'TRAILALI': (0, 24),
        'TRAILB': (0, 300), 'TRAILBRR': (0, 40), 'TRAILBLI': (0, 24)}
NP4 = {'REY1REC': (0, 15), 'REY2REC': (0, 15), 'REY3REC': (0, 15), 'REY4REC': (0, 15), 'REY5REC': (0, 15),
       'REY6REC': (0, 15), 'REYDREC': (0, 15), 'REYTCOR': (0, 15), 'REYFPOS': (0, 15), 'OTRAILA': (0, 100),
       'OTRLARR': (0, 40), 'OTRAILB': (0, 300), 'OTRLBRR': (0, 40), 'MOCBTOTS': (0, 22), 'CERAD1REC': (0, 10),
       'CERAD2REC': (0, 10), 'CERAD3REC': (0, 10), 'CERADJ6REC': (0, 10), 'VNTTOTW': (0, 50)}
TIMED = {'TRAILA', 'TRAILB', 'OTRAILA', 'OTRAILB'}
CDR = {k: (0, 3) for k in ['MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'HOMEHOBB', 'PERSCARE', 'CDRGLOB', 'COMPORT', 'CDRLANG']}
CDR['CDRSUM'] = (0, 18)
FAQ = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
NPI = ['DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'ELAT', 'APA', 'DISN', 'IRR', 'MOT', 'NITE', 'APP']
HIST_EVER = ['CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACDEF', 'CVPACE', 'CVCHF', 'CVANGINA', 'CVHVALVE', 'CVOTHR',
             'CBSTROKE', 'CBTIA', 'PD', 'PDOTHR', 'SEIZURES', 'TBI', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'NCOTHR',
             'DIABETES', 'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'ARTHRIT', 'INCONTU', 'INCONTF', 'APNEA', 'RBD',
             'INSOMN', 'OTHSLEEP', 'ALCOHOL', 'ABUSOTHR', 'PTSD', 'BIPOLAR', 'SCHIZ', 'ANXIETY', 'OCD', 'NPSYDEV',
             'PSYCDIS', 'TOBAC30', 'TOBAC100']
HIST_BIN = ['DEP2YRS', 'DEPOTHR', 'ALCOCCAS']
MEDS = ['NACCAHTN', 'NACCHTNC', 'NACCACEI', 'NACCAAAS', 'NACCBETA', 'NACCCCBS', 'NACCDIUR', 'NACCVASD', 'NACCANGI',
        'NACCLIPL', 'NACCNSD', 'NACCAC', 'NACCADEP', 'NACCAPSY', 'NACCADMD', 'NACCPDMD', 'NACCEMD', 'NACCEPMD',
        'NACCDBMD', 'NACCAANX']
VITALS = {'NACCBMI': (10, 80), 'BPSYS': (70, 230), 'BPDIAS': (30, 140), 'HRATE': (33, 160)}
SENS = ['HEARING', 'HEARAID', 'VISION', 'VISCORR']
B9 = ['DECSUB', 'DECIN', 'COGMEM', 'COGJUDG', 'COGLANG', 'COGVIS', 'COGATTN', 'COGOTHR', 'BEAPATHY', 'BEDEP',
      'BEDISIN', 'BEIRRIT', 'BEAGIT', 'MOGAIT', 'MOFALLS', 'MOTREM', 'MOSLOW']
OTHER = ['NACCAGE', 'NACCSEX', 'EDUC', 'NACCNE4S', 'MARISTAT', 'NACCLIVS', 'INDEPEND', 'RESIDENC', 'NACCGDS',
         'NACCAMD', 'SMOKYRS', 'PACKSPER', 'PACKET', 'FORMVER', 'NACCUDSD', 'NACCID', 'NACCVNUM', 'VISITDATE',
         'NACCADC']
ETIO = ['NACCALZD', 'NACCALZP', 'NACCLBDE', 'NACCFTD', 'CVD', 'VASC', 'PSP', 'CORT', 'DEP']
RAW_COLS = sorted(set(list(NP12) + list(NP3) + list(NPSH) + list(NP4) + list(CDR) + FAQ + NPI + HIST_EVER + HIST_BIN
                      + MEDS + list(VITALS) + SENS + B9 + OTHER))
BLOCKS_NP = {'np12': list(NP12), 'np3': list(NP3), 'npsh': list(NPSH), 'np4': list(NP4)}


def _rng(s, lo, hi):
    s = pd.to_numeric(s, errors='coerce').astype(float)
    return s.where((s >= lo) & (s <= hi))


def build_visit_features(raw, etio=False):
    """raw: visit-level DataFrame (RAW_COLS). Returns (meta, F, blocks) with F float32 DataFrame (NaN = missing)."""
    r = raw.copy()
    r['date'] = pd.to_datetime(r['VISITDATE'])
    r = r.sort_values(['NACCID', 'date', 'NACCVNUM']).reset_index(drop=True)
    F = {}
    blocks = {}

    def put(name, s, block):
        F[name] = s.astype('float32').values
        blocks[name] = block
    # demographics / static
    put('age', _rng(r.NACCAGE, 18, 120), 'demo')
    put('female', (r.NACCSEX == 2).astype(float).where(r.NACCSEX.isin([1, 2])), 'demo')
    put('educ', _rng(r.EDUC, 0, 36), 'demo')
    put('apoe4', _rng(r.NACCNE4S, 0, 2), 'demo')
    put('married', (r.MARISTAT == 1).astype(float).where(r.MARISTAT.between(1, 6)), 'demo')
    put('lives_alone', (r.NACCLIVS == 1).astype(float).where(r.NACCLIVS.between(1, 5)), 'demo')
    put('independ', _rng(r.INDEPEND, 1, 4), 'demo')
    put('residenc', _rng(r.RESIDENC, 1, 4), 'demo')
    put('telephone', r.PACKET.isin(['T', 'IT']).astype(float), 'visit')
    for k, lab in [(1, 'nc'), (2, 'impnomci'), (3, 'mci'), (4, 'dem')]:
        put(f'status_{lab}', (r.NACCUDSD == k).astype(float), 'status')
    for v, (lo, hi) in CDR.items():
        put(v.lower(), _rng(r[v], lo, hi), 'cdr')
    faq = pd.concat([_rng(r[v], 0, 3) for v in FAQ], axis=1)
    for v in FAQ:
        put('faq_' + v.lower(), _rng(r[v], 0, 3), 'faq')
    nval = faq.notna().sum(1)
    put('faq_total', (faq.sum(1) * 10 / nval.replace(0, np.nan)).where(nval >= 7), 'faq')
    for v in NPI:
        put('npi_' + v.lower(), _rng(r[v], 0, 1), 'npi')
    put('gds', _rng(r.NACCGDS, 0, 15), 'gds')
    for v in B9:
        put('b9_' + v.lower(), _rng(r[v], 0, 1), 'b9')
    # health history with LOCF within participant
    hist = {}
    for v in HIST_EVER:
        s = pd.to_numeric(r[v], errors='coerce')
        hist['hx_' + v.lower()] = (s >= 1).astype(float).where(s.isin([0, 1, 2]))
    for v in HIST_BIN:
        hist['hx_' + v.lower()] = _rng(r[v], 0, 1)
    sm = pd.to_numeric(r.SMOKYRS, errors='coerce')
    hist['hx_smokyrs'] = sm.where(sm.between(0, 87)).where(sm != 88, 0.0).where(sm.between(0, 88))
    pk = pd.to_numeric(r.PACKSPER, errors='coerce')
    hist['hx_packsper'] = pk.where(pk.between(0, 5)).where(pk != 8, 0.0).where(pk.isin([0, 1, 2, 3, 4, 5, 8]))
    H = pd.DataFrame(hist)
    H = H.groupby(r['NACCID']).ffill()
    for k in H.columns:
        put(k, H[k], 'hist')
    for v in MEDS:
        put('med_' + v.lower(), _rng(r[v], 0, 1), 'med')
    put('med_count', _rng(r.NACCAMD, 0, 60), 'med')
    for v, (lo, hi) in VITALS.items():
        put('vit_' + v.lower(), _rng(r[v], lo, hi), 'vitals')
    for v in SENS:
        put('sen_' + v.lower(), _rng(r[v], 0, 1), 'sensory')
    # neuropsychology
    for blk, spec in [('np12', NP12), ('np3', NP3), ('npsh', NPSH), ('np4', NP4)]:
        for v, (lo, hi) in spec.items():
            s = pd.to_numeric(r[v], errors='coerce')
            base = 990 if v in TIMED else 90
            put('np_' + v.lower(), s.where((s >= lo) & (s <= hi)), blk)
            put('np_' + v.lower() + '_nc_cog', (s == base + 6).astype(float).where(s != -4), blk)
            put('np_' + v.lower() + '_nc_oth', s.isin([base + 5, base + 7, base + 8]).astype(float).where(s != -4), blk)
    if etio:   # sensitivity (Methods 2.8): clinician presumptive etiologic diagnoses at the visit
        for v in ETIO:
            s = pd.to_numeric(r[v], errors='coerce')
            if v == 'NACCALZP':
                x = (s == 1).astype(float).where(s.isin([1, 2, 3, 7, 8]))      # AD = primary etiology
            else:
                x = s.where(s.isin([0, 1])).where(s != 8, 0.0)
                x = x.where(~s.isin([8]), 0.0)                                  # 8 = no impairment -> 0
            put('etio_' + v.lower(), x, 'etio')
    Fd = pd.DataFrame(F)
    meta = r[['NACCID', 'NACCVNUM', 'NACCADC', 'date', 'FORMVER', 'PACKET', 'NACCUDSD']].copy()
    meta['vidx'] = np.arange(len(meta))
    # version index 0..3 for v1, v2, v3.x, v4 ; years since first visit; gap
    meta['ver'] = np.select([meta.FORMVER < 1.5, meta.FORMVER < 2.5, meta.FORMVER < 3.9], [0, 1, 2], 3)
    first = meta.groupby('NACCID')['date'].transform('min')
    meta['t_since_first'] = (meta['date'] - first).dt.days / 365.25
    meta['gap'] = meta.groupby('NACCID')['date'].diff().dt.days.fillna(0) / 365.25
    meta['pos'] = meta.groupby('NACCID').cumcount()
    return meta, Fd, blocks


PARSIMONIOUS = ['age', 'female', 'educ', 'apoe4', 'cdrsum', 'faq_total', 'gds', 'np_naccmmse', 'np_mocatots']


def attach_landmarks(lm, meta):
    """Map landmark rows (NACCID, NACCVNUM) to visit index."""
    key = meta[['NACCID', 'NACCVNUM', 'vidx', 'pos']]
    out = lm.merge(key, on=['NACCID', 'NACCVNUM'], how='left', validate='many_to_one')
    assert out.vidx.notna().all(), 'landmark without visit row'
    return out
