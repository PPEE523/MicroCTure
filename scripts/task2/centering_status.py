"""Explicit missing-evidence semantics for centering experiments and legacy tables.

None is an unassessed decision, never a negative. CSV writes it as an empty
cell; JSON writes null. This module does not read matrices or change thresholds.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import math


def as_bool(value):
    if value in (None, '', 'NA', 'None', 'null'):
        return None
    if value in (True, 'True', 'true', 1, '1'):
        return True
    if value in (False, 'False', 'false', 0, '0'):
        return False
    raise ValueError(f'Invalid decision: {value!r}')


def finite(values):
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def assess(distances, cutoffs, quality, calibration_windows, minimum):
    distances, cutoffs = list(distances), list(cutoffs)
    reason = None
    if len(distances) != 2 or len(cutoffs) != 2:
        reason = 'missing_replicate_evidence'
    elif calibration_windows < minimum:
        reason = 'insufficient_calibration'
    elif not finite(cutoffs):
        reason = 'nonfinite_threshold'
    elif not quality:
        reason = 'quality_failed'
    elif not finite(distances):
        reason = 'nonfinite_distance'
    return dict(assessment_status='not_evaluable' if reason else 'assessed',
                assessment_reason=reason or '',
                outside_both=None if reason else all(float(d) > float(c) for d, c in zip(distances, cutoffs)))


def decision_from_row(row, field='outside_both', cutoff_fields=('rep1_cutoff', 'rep2_cutoff'),
                      calibration_windows=None, minimum=10):
    """Read current nullable decisions and old False-with-NaN output safely."""
    if row.get('assessment_status') == 'not_evaluable':
        return None
    if calibration_windows is not None and calibration_windows < minimum:
        return None
    if not finite([row.get(k) for k in cutoff_fields]):
        return None
    distance_fields = ('rep1_distance', 'rep2_distance') if field == 'outside_both' else ('rep1_change', 'rep2_change')
    if not finite([row.get(k) for k in distance_fields]):
        return None
    if as_bool(row.get('quality_pass', True)) is not True:
        return None
    return as_bool(row.get(field))


def both(a, b):
    return None if a is None or b is None else bool(a and b)


def metric_count(a, b):
    return float('nan') if a is None or b is None else int(a) + int(b)


def complete_count(values):
    values = list(values)
    return sum(values) if values and all(v is not None for v in values) else None


def summarize_candidate(settings, flags, boundary_reproducible, all_quality, calibration_sufficient,
                        robust_fraction, anchors=('original', 'boundary'), widths=(64, 128, 256)):
    """Keep a complete-grid endpoint NA whenever evidence is missing from it."""
    row = dict(calibration_sufficient=bool(calibration_sufficient))
    for anchor in anchors:
        values = [both(flags[vi, 'pca'], flags[vi, 'shape']) for vi, v in enumerate(settings) if v['anchor'] == anchor]
        count = complete_count(values)
        row[anchor + '_assessed_variants'] = sum(v is not None for v in values)
        row[anchor + '_outside_variants'] = count
        row[anchor + '_outside_fraction'] = count / len(values) if count is not None else None
    fractions = []
    for width in widths:
        values = [both(flags[vi, 'pca'], flags[vi, 'shape']) for vi, v in enumerate(settings)
                  if v['anchor'] == 'boundary' and int(v['width_bins']) == width]
        count = complete_count(values)
        fraction = count / len(values) if count is not None else None
        row[f'boundary_w{width}_outside_fraction'] = fraction
        fractions.append(fraction)
    reason = ('insufficient_calibration' if not calibration_sufficient else
              'quality_failed' if not all_quality else
              'incomplete_grid' if any(f is None for f in fractions) or
              any(row[a + '_outside_variants'] is None for a in anchors) else '')
    row['assessment_status'] = 'not_evaluable' if reason else 'assessed'
    row['assessment_reason'] = reason
    row['robust_recentered_anomaly'] = (None if reason else
                                      bool(boundary_reproducible and min(fractions) >= robust_fraction))
    return row


def normalize_summary(row, calibration_windows, minimum):
    """Compatibility view: never rewrite an archived CSV to repair its semantics."""
    out = dict(row)
    unavailable = (calibration_windows < minimum or as_bool(row.get('calibration_sufficient', True)) is False
                   or as_bool(row.get('all_variants_quality_pass', True)) is False
                   or row.get('assessment_status') == 'not_evaluable')
    if unavailable:
        for key in list(out):
            if key.endswith(('_outside_variants', '_outside_fraction')) or key == 'robust_recentered_anomaly':
                out[key] = None
        out.update(assessment_status='not_evaluable', assessment_reason=row.get('assessment_reason') or
                   ('insufficient_calibration' if calibration_windows < minimum else 'unavailable_evidence'))
    else:
        for key, value in list(out.items()):
            if key.endswith('_outside_variants'):
                out[key] = int(value) if value not in ('', None) else None
        out['robust_recentered_anomaly'] = as_bool(out['robust_recentered_anomaly'])
        out.setdefault('assessment_status', 'assessed')
    return out


def aggregate(summaries):
    assessed = [r for r in summaries if r.get('assessment_status') == 'assessed']
    complete = len(assessed) == len(summaries)
    return dict(assessment_status='assessed' if complete else 'not_evaluable',
                assessed_candidates=len(assessed), not_evaluable_candidates=len(summaries)-len(assessed),
                candidates_with_any_two_metric_anomaly=(sum(r['original_outside_variants'] + r['boundary_outside_variants'] > 0 for r in assessed) if complete else None),
                candidates_with_recentered_two_metric_anomaly=(sum(r['boundary_outside_variants'] > 0 for r in assessed) if complete else None),
                robust_recentered_anomalies=(sum(r['robust_recentered_anomaly'] for r in assessed) if complete else None))


def display(value):
    return 'NA（未评估）' if value is None or value == '' else str(value)


def require_new_output(path):
    """Scientific outputs use a fresh namespace outside retained evidence."""
    from optional_common import fresh_report_path
    fresh_report_path(path)
