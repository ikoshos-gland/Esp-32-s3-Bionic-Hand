"""
Python <-> C++ filter equivalence test
======================================

What is actually tested (no hardware needed):

1. Coefficients: the numbers in include/filter_coefficients_50hz.h equal the
   scipy design (butter 4th-order 20 Hz HPF, 450 Hz LPF, iirnotch 50 Hz Q=12.5)
   at 1000 Hz, to 1e-6.
2. Sample-level equivalence: a float32 transliteration of the C++ biquad
   cascade (Direct Form II Transposed) run with the HEADER coefficients gives
   the same output as scipy.signal.sosfilt with the DESIGNED coefficients, on
   a realistic synthetic EMG signal (tolerance 0.05 ADC counts, i.e. below
   ADC resolution).
3. Stability: every biquad has poles strictly inside the unit circle.
4. Frequency response sanity: DC and 50 Hz rejected, 100 Hz passed.
5. TD4 equivalence: the Python TD4FeatureExtractor and a transliteration of
   the C++ extract_features_from_raw() loop agree on random windows.

Exit code 0 on success, 1 on any failure (usable in CI).
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Windows console safety


import os
import re
import sys

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'critical'))

from filters.dsp_filters import EMGFilterBank, biquad_cascade_reference, sos_to_stages  # noqa: E402
from feature_extraction import TD4FeatureExtractor  # noqa: E402

HEADER = os.path.join(HERE, '..', '..', 'include', 'filter_coefficients_50hz.h')
FS = 1000
FAILURES = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def parse_header(path=HEADER, block='1000'):
    """Return dict name -> float for the 1000 Hz (default) block of the header."""
    text = open(path, encoding='utf-8').read()
    # default block is between '#ifndef FILTER_SAMPLING_2000HZ' and '#else'
    m = re.search(r'#ifndef FILTER_SAMPLING_2000HZ(.*?)#else', text, re.S)
    if not m:
        raise RuntimeError('header layout not recognised (expected #ifndef FILTER_SAMPLING_2000HZ ... #else)')
    body = m.group(1)
    coeffs = {}
    for name, val in re.findall(r'const float (\w+) = ([-0-9.]+)f;', body):
        coeffs[name] = float(val)
    return coeffs


def header_stages(coeffs):
    def st(prefix):
        return (coeffs[f'{prefix}_B0'], coeffs[f'{prefix}_B1'], coeffs[f'{prefix}_B2'],
                coeffs[f'{prefix}_A1'], coeffs[f'{prefix}_A2'])
    return [st('HPF_STAGE1'), st('HPF_STAGE2'), st('LPF_STAGE1'), st('LPF_STAGE2'), st('NOTCH')]


def synthetic_emg(seconds=2.0, fs=FS, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    emg = 250 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t))
    return 1850 + 120 * np.sin(2 * np.pi * 3 * t) + 150 * np.sin(2 * np.pi * 50 * t) + emg


def test_coefficients_match_design():
    print("\nTEST 1: header coefficients == scipy design")
    coeffs = parse_header()
    fb = EMGFilterBank(sampling_rate=FS, powerline_freq=50, verbose=False)
    designed = sos_to_stages(fb.cascade_sos())
    header = header_stages(coeffs)
    worst = max(abs(np.array(h) - np.array(d)).max() for h, d in zip(header, designed))
    check(worst < 1e-6, f"max coefficient difference {worst:.2e} (< 1e-6)")
    return header, designed


def test_sample_equivalence(header):
    print("\nTEST 2: C++ biquad cascade (float32, header coeffs) == scipy sosfilt (causal)")
    x = synthetic_emg()
    fb = EMGFilterBank(sampling_rate=FS, powerline_freq=50, verbose=False)
    y_py = fb.filter_signal(x)
    y_cpp = biquad_cascade_reference(x, header)
    err = np.abs(y_py - y_cpp)
    check(err.max() < 0.05, f"max |python - cpp| = {err.max():.4f} ADC counts over {len(x)} samples (< 0.05)")
    check(np.abs(y_py[500:].mean()) < 1.0, f"DC removed: mean after settling {y_py[500:].mean():.3f}")


def test_stability(header):
    print("\nTEST 3: every biquad stable (poles inside unit circle)")
    for name, (_, _, _, a1, a2) in zip(['HPF1', 'HPF2', 'LPF1', 'LPF2', 'NOTCH'], header):
        poles = np.roots([1.0, a1, a2])
        r = np.abs(poles).max()
        check(r < 1.0, f"{name}: max |pole| = {r:.4f}")


def test_frequency_response():
    print("\nTEST 4: frequency response sanity at 1000 Hz")
    fb = EMGFilterBank(sampling_rate=FS, powerline_freq=50, verbose=False)
    w, mag = fb.get_frequency_response('cascade')

    def at(f):
        return mag[np.argmin(np.abs(w - f))]
    check(at(0.5) < -40, f"0.5 Hz: {at(0.5):.1f} dB (< -40)")
    check(at(50) < -20, f"50 Hz: {at(50):.1f} dB (< -20)")
    check(at(100) > -3, f"100 Hz: {at(100):.2f} dB (> -3)")
    check(at(499) < -10, f"499 Hz: {at(499):.1f} dB (< -10)")


def cpp_td4_reference(window: np.ndarray, n=250, zc_thr=15.0, ssc_thr=15.0, adc_max=4095.0):
    """Transliteration of src/functions.cpp::extract_features_from_raw (float32)."""
    f32 = np.float32
    out = []
    for s in range(window.shape[1]):
        data = window[:, s].astype(np.float32)
        mean = f32(0)
        for v in data:
            mean = f32(mean + v)
        mean = f32(mean / n)
        centered = np.array([f32(v - mean) for v in data], dtype=np.float32)
        mav = f32(0)
        for v in centered:
            mav = f32(mav + abs(v))
        mav = f32(mav / n)
        wl = f32(0)
        for i in range(1, n):
            wl = f32(wl + abs(f32(data[i] - data[i - 1])))
        zc = 0
        for i in range(n - 1):
            if centered[i] * centered[i + 1] < 0 and abs(f32(data[i] - data[i + 1])) >= zc_thr:
                zc += 1
        ssc = 0
        for i in range(1, n - 1):
            left = f32(data[i] - data[i - 1])
            right = f32(data[i] - data[i + 1])
            if f32(left * right) >= ssc_thr:
                ssc += 1
        out += [mav / adc_max, wl / (adc_max * n), zc / n, ssc / n]
    return np.array(out, dtype=np.float64)


def test_td4_equivalence():
    print("\nTEST 5: Python TD4FeatureExtractor == C++ extract_features_from_raw")
    rng = np.random.default_rng(1)
    ext = TD4FeatureExtractor(window_size_ms=250, overlap_ms=125, sampling_rate=FS, verbose=False)
    names = [f'EMG{i}' for i in range(1, 7)]

    def run(windows):
        worst = 0.0
        for window in windows:
            py = ext.extract_window_features(window, names)
            py_vec = np.array([py[f'{k}_{s}'] for s in names for k in ('mav', 'wl', 'zc', 'ssc')])
            worst = max(worst, np.abs(py_vec - cpp_td4_reference(window)).max())
        return worst

    # (a) device-like data: filtered floats, no exact threshold ties -> must match to float precision
    cont = [rng.standard_normal((250, 6)) * rng.choice([5, 40, 300]) + rng.uniform(-50, 50, size=6) for _ in range(20)]
    worst = run(cont)
    check(worst < 1e-5, f"continuous data: max |python - cpp| over 20 windows = {worst:.2e} (< 1e-5)")

    # (b) integer data creates exact ties at the 15-count thresholds. Differences are taken on
    #     raw samples in both implementations, so ties are decided identically.
    ints = [np.round(rng.standard_normal((250, 6)) * rng.choice([5, 40, 300])) for _ in range(20)]
    worst_i = run(ints)
    check(worst_i < 1e-5, f"integer data (threshold ties): max |python - cpp| = {worst_i:.2e} (< 1e-5)")


def run_all_tests():
    print("=" * 70)
    print("FILTER + TD4 EQUIVALENCE TESTS (Python vs C++ firmware)")
    print("=" * 70)
    header, _ = test_coefficients_match_design()
    test_sample_equivalence(header)
    test_stability(header)
    test_frequency_response()
    test_td4_equivalence()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
