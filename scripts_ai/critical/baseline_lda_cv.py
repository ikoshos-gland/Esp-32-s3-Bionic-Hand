"""
Leakage-free baseline: leave-one-repetition-out cross-validation
================================================================

Runs classical EMG baselines on a TD4 feature NPZ (needs the `reps` array):

  - LDA (the historical reference classifier for TD4 features)
  - Logistic regression (standardised features)
  - sklearn MLP mirroring the deployed architecture (256-128-64, ReLU)

and reports, per classifier:
  - leave-one-rep-out (LORO) accuracy: mean +- std over folds
  - the inflated number a random 80/20 split of overlapping windows gives
  - per-class recall (LORO, pooled predictions)

Also runs the training-data sanity check used by train_test_model.py, so a
corrupted NPZ (e.g. uint16 overflow) is reported instead of silently scored.

Usage:
  python baseline_lda_cv.py path/to/features.npz [--no-mlp]
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Windows console safety


import argparse
import sys
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

MAV_MAX_PHYSICAL = 0.5   # |centered| <= 2047.5 -> MAV/4095 <= 0.5
WL_MAX_PHYSICAL = 0.5


def sanity_check(features, feature_names):
    """Return list of problems (empty when the NPZ looks physically plausible)."""
    problems = []
    for j, name in enumerate(feature_names):
        col = features[:, j]
        if name.startswith('mav_') and col.max() > MAV_MAX_PHYSICAL:
            problems.append(f"{name}: max {col.max():.3f} > {MAV_MAX_PHYSICAL} (impossible for 12-bit ADC, overflow?)")
        if name.startswith('wl_') and col.max() > WL_MAX_PHYSICAL:
            problems.append(f"{name}: max {col.max():.3f} > {WL_MAX_PHYSICAL}")
        if col.std() == 0:
            problems.append(f"{name}: constant column")
    return problems


def make_models(include_mlp=True, seed=42):
    models = {
        'LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    }
    if include_mlp:
        models['MLP-256-128-64'] = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=1e-3, max_iter=400,
                          early_stopping=True, random_state=seed))
    return models


def loro_cv(model_factory, X, y, reps):
    """Leave-one-rep-out; returns (fold accuracies, pooled y_true, pooled y_pred)."""
    accs, yt, yp = [], [], []
    for r in sorted(np.unique(reps)):
        test = reps == r
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        m = model_factory()
        m.fit(X[~test], y[~test])
        pred = m.predict(X[test])
        accs.append(accuracy_score(y[test], pred))
        yt.append(y[test]); yp.append(pred)
    return np.array(accs), np.concatenate(yt), np.concatenate(yp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz')
    ap.add_argument('--no-mlp', action='store_true')
    ap.add_argument('--ignore-sanity', action='store_true', help='score even if the sanity check fails')
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    X = d['features'].astype(np.float64)
    y_str = d['labels'].astype(str)
    class_names, y = np.unique(y_str, return_inverse=True)   # integer labels (sklearn MLP early stopping needs numeric y)
    names = [str(n) for n in d['feature_names']]

    print("=" * 72)
    print(f"LEAVE-ONE-REP-OUT BASELINES  ({a.npz})")
    print("=" * 72)
    print(f"windows: {len(y)}   features: {X.shape[1]}   classes: {len(np.unique(y))}")

    problems = sanity_check(X, names)
    if problems:
        print("\nSANITY CHECK FAILED, this NPZ is not physically plausible training data:")
        for p in problems[:12]:
            print("   - " + p)
        if len(problems) > 12:
            print(f"   ... {len(problems) - 12} more")
        if not a.ignore_sanity:
            print("\nRefusing to score corrupted data (use --ignore-sanity to override).")
            return 1
    else:
        print("\nsanity check: OK (all MAV/WL within physical range, no constant columns)")

    if 'reps' not in d.files:
        print("ERROR: NPZ has no 'reps' array; re-run feature_extraction.py (v2+) to get rep numbers")
        return 2
    reps = d['reps'].astype(int)
    print(f"reps: {sorted(np.unique(reps).tolist())}")

    # Rep 0 = preparation/baseline rows; keep them but they cannot be a test fold on their own
    usable = reps > 0
    if usable.sum() < len(reps):
        print(f"note: {int((~usable).sum())} windows with Rep=0 (baseline/preparation) are used for training only")

    models = make_models(include_mlp=not a.no_mlp)
    print("\n%-16s  %-22s  %-18s" % ("classifier", "LORO accuracy", "random 80/20 (leaky)"))
    print("-" * 62)
    reports = {}
    for name, proto in models.items():
        def factory(p=proto):
            from sklearn.base import clone
            return clone(p)
        accs, yt, yp = loro_cv(factory, X, y, np.where(usable, reps, -1))
        # leaky reference: random split of overlapping windows
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        leaky = factory().fit(Xtr, ytr).score(Xte, yte)
        print("%-16s  %.3f +- %.3f (%d folds)   %.3f" % (name, accs.mean(), accs.std(), len(accs), leaky))
        reports[name] = (yt, yp)

    chance = 1.0 / len(np.unique(y))
    print(f"\nchance level: {chance:.3f}")
    best = max(reports, key=lambda k: accuracy_score(*reports[k]))
    print(f"\nper-class recall, LORO pooled, best classifier = {best}:")
    print(classification_report(*reports[best], target_names=list(class_names), zero_division=0, digits=3))
    return 0


if __name__ == '__main__':
    sys.exit(main())
