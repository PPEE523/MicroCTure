"""Training-only normalization, spatial roles and paired AE/DAE experiments."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
import torch
from discover_task2 import AE, losses


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _partition_config(cfg):
    _positive_integer(cfg['spatial_block_bins'], 'spatial_block_bins')
    modulus = _positive_integer(cfg['split_block_modulus'], 'split_block_modulus')
    seen = set()
    for name in ['train', 'validation', 'test']:
        values = cfg[f'{name}_block_remainders']
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < modulus for v in values) or
                len(values) != len(set(values)) or seen.intersection(values)):
            raise ValueError('Partition remainders must be nonempty, unique, disjoint and within the modulus')
        seen.update(values)
    if seen != set(range(modulus)):
        raise ValueError('Partition remainders must cover every block remainder')


def validate_config(cfg):
    """Reject unsupported or ambiguous protocols before any experiment output."""
    _partition_config(cfg)
    if cfg['resolution_bp'] != 100:
        raise ValueError('This contact-band protocol supports resolution_bp=100 only')
    widths = cfg['window_bins']
    if (not isinstance(widths, list) or not widths or len(widths) != len(set(widths)) or
            any(isinstance(w, bool) or not isinstance(w, int) or w < 64 or w % 64 for w in widths)):
        raise ValueError('window_bins must contain unique positive multiples of 64')
    if max(widths) > cfg['spatial_block_bins']:
        raise ValueError('A window must fit within one spatial block')
    for name in ['stride_bins', 'ae_max_epochs', 'ae_patience', 'ae_max_train_windows',
                 'ae_batch_size', 'latent_dim', 'random_references', 'kmeans_initializations',
                 'controls_per_candidate', 'min_controls', 'min_new_cluster_members']:
        _positive_integer(cfg[name], name)
    for name, minimum in [('seeds', 1), ('clustering_seeds', 2), ('clusters', 1)]:
        values = cfg[name]
        if (not isinstance(values, list) or len(values) < minimum or len(values) != len(set(values)) or
                any(isinstance(v, bool) or not isinstance(v, int) or v < (2 if name == 'clusters' else 0) for v in values)):
            raise ValueError(f'{name} must contain unique valid integers (at least {minimum})')
    for name in ['min_valid_fraction', 'max_zero_fraction', 'mask_probability', 'cluster_min_jaccard', 'control_quantile']:
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and within [0, 1]')
    if not np.isfinite(cfg['ae_learning_rate']) or cfg['ae_learning_rate'] <= 0:
        raise ValueError('ae_learning_rate must be finite and positive')
    if not np.isfinite(cfg['replicate_min_correlation']) or not -1 <= cfg['replicate_min_correlation'] <= 1:
        raise ValueError('replicate_min_correlation must be finite and within [-1, 1]')
    for name in ['full_candidate_quotas', 'test_candidate_quotas']:
        quotas = cfg[name]
        if not isinstance(quotas, dict) or set(quotas) != {str(w) for w in widths}:
            raise ValueError(f'{name} must specify every window scale exactly once')
        for width, quota in quotas.items():
            _positive_integer(quota, f'{name}[{width}]')
    if any(cfg['test_candidate_quotas'][str(w)] > cfg['full_candidate_quotas'][str(w)] for w in widths):
        raise ValueError('Test scale quotas cannot exceed full scale quotas')
    if cfg['min_controls'] > cfg['controls_per_candidate']:
        raise ValueError('min_controls cannot exceed controls_per_candidate')
    methods = {'shape', 'density'} | {f'{family}{fusion}_{seed}' for family in ['ae', 'dae']
                for fusion in ['', '_fusion'] for seed in cfg['seeds']}
    if (cfg['primary_method'] not in methods or cfg['primary_k'] not in cfg['clusters'] or
            cfg['primary_cluster_seed'] not in cfg['clustering_seeds']):
        raise ValueError('Primary method, K and clustering seed must belong to the registered comparisons')
    json.dumps(cfg, allow_nan=False)
    return cfg


def bin_roles(n, cfg):
    _partition_config(cfg)
    block = np.arange(n) // cfg['spatial_block_bins'] % cfg['split_block_modulus']
    roles = np.full(n, -1, int)
    for role, name in enumerate(['train', 'validation', 'test']):
        roles[np.isin(block, cfg[f'{name}_block_remainders'])] = role
    return roles


def window_role(start, width, roles, block_size):
    end = start + width
    if start < 0 or end > len(roles) or start // block_size != (end - 1) // block_size:
        return -1
    return int(roles[start])


def train_background(band, roles):
    """Fit QC threshold and distance expected values on training endpoints only."""
    b = dict(band)
    train_bins = roles == 0
    # Local coverage uses only pairs whose endpoints have the SAME partition.
    # Raw genome-wide coverage includes contacts to held-out regions and must
    # not be used to fit training QC or affect validation masks.
    coverage = np.zeros(len(roles), dtype=np.float64)
    for distance in range(2, b['band'].shape[1]):
        same_partition = roles == np.roll(roles, -distance)
        counts = np.where(same_partition, b['band'][:, distance], 0)
        coverage += counts + np.roll(counts, distance)
    positive = coverage[train_bins & (coverage > 0)]
    if not len(positive):
        raise ValueError('No positive training coverage')
    threshold = float(np.median(positive) * .1)
    valid = coverage >= threshold
    # Retain the incomplete terminal-bin exclusion from the raw cache.
    valid[-1] = valid[-1] and bool(band['valid'][-1])
    expected = np.zeros(b['band'].shape[1])
    opportunities = np.zeros(len(expected), int)
    fit = train_bins & valid
    for distance in range(2, len(expected)):
        use = fit & np.roll(fit, -distance)
        opportunities[distance] = use.sum()
        expected[distance] = b['band'][use, distance].sum(dtype=np.float64) / max(use.sum(), 1)
    if (expected[2:] <= 0).any():
        raise ValueError('Insufficient training-only distance background')
    b.update(valid=valid, expected=expected, opportunities=opportunities, threshold=threshold, coverage=coverage)
    return b


def symmetric_mask(x, probability, generator):
    draw = torch.rand((len(x), x.shape[-1], x.shape[-1]), generator=generator)
    upper = torch.triu(draw < probability, diagonal=1)
    hide = upper | upper.transpose(1, 2)
    corrupted = x.clone()
    corrupted[:, 0] = corrupted[:, 0].masked_fill(hide, 0)
    corrupted[:, 1] = corrupted[:, 1].masked_fill(hide, 0)
    return corrupted


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def model_identity(x, roles, eligible, cfg, family, seed, train, validation):
    return dict(schema_version=1, family=family, seed=seed, config_sha256=_json_sha256(cfg),
                x=_array_identity(x), roles=_array_identity(roles), eligible=_array_identity(eligible),
                train_indices=train.tolist(), validation_indices=validation.tolist(),
                sources={Path(path).name: file_sha256(path) for path in [__file__, inspect.getfile(AE)]})


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def publish_model_artifacts(out, name, identity):
    paths = [out / f'{name}.pt', out / f'{name}_history.json']
    metadata = dict(schema_version=1, status='completed', identity=identity,
                    identity_sha256=_json_sha256(identity),
                    files={p.name: dict(sha256=file_sha256(p), size_bytes=p.stat().st_size) for p in paths})
    _atomic_json(out / f'{name}_artifacts.json', metadata)
    return metadata


def load_model_artifacts(out, name, identity):
    checkpoint, history_path, metadata_path = [out / f'{name}{suffix}' for suffix in ['.pt', '_history.json', '_artifacts.json']]
    if not all(path.is_file() and not path.is_symlink() for path in [checkpoint, history_path, metadata_path]):
        raise ValueError(f'Incomplete model artifacts for {name}; refusing to retrain or overwrite')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('schema_version') != 1 or metadata.get('status') != 'completed' or
            metadata.get('identity') != identity or metadata.get('identity_sha256') != _json_sha256(identity)):
        raise ValueError(f'Model identity mismatch for {name}; refusing to reuse or overwrite')
    expected_files = {checkpoint.name, history_path.name}
    if set(metadata.get('files', {})) != expected_files:
        raise ValueError(f'Incomplete model artifact index for {name}')
    for path in [checkpoint, history_path]:
        expected = metadata['files'][path.name]
        if expected != dict(sha256=file_sha256(path), size_bytes=path.stat().st_size):
            raise ValueError(f'Model artifact bytes mismatch for {name}: {path.name}')
    history = json.loads(history_path.read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('identity_sha256') != _json_sha256(identity) or saved.get('family') != identity['family'] or
            saved.get('seed') != identity['seed'] or saved.get('train_indices') != identity['train_indices'] or
            saved.get('validation_indices') != identity['validation_indices']):
        raise ValueError(f'Checkpoint identity mismatch for {name}')
    if (not history or [h.get('epoch') for h in history] != list(range(1, len(history) + 1)) or
            not 1 <= saved.get('epoch', 0) <= len(history) or
            history[saved['epoch'] - 1]['validation_loss'] != saved.get('validation_loss') or
            not np.isfinite(saved['validation_loss'])):
        raise ValueError(f'Checkpoint/history mismatch for {name}')
    return saved, history, metadata


def fit_model(x, roles, eligible, cfg, out, family, seed):
    """Fit a fresh model or read a complete identity-bound model; never resume."""
    if family not in {'ae', 'dae'} or seed not in cfg['seeds']:
        raise ValueError('Unregistered model family or seed')
    if x.ndim != 4 or len(roles) != len(x) or len(eligible) != len(x):
        raise ValueError('Mismatched model input, roles or eligibility')
    name = f'{family}_{seed}'
    rng = np.random.default_rng(seed)
    train = np.flatnonzero((roles == 0) & eligible)
    train = rng.choice(train, min(len(train), cfg['ae_max_train_windows']), replace=False)
    validation = np.flatnonzero((roles == 1) & eligible)
    if not len(train) or not len(validation):
        raise ValueError('Empty train or validation partition')
    identity = model_identity(x, roles, eligible, cfg, family, seed, train, validation)
    checkpoint = out / f'{name}.pt'
    history_path = out / f'{name}_history.json'
    metadata_path = out / f'{name}_artifacts.json'
    paths = [checkpoint, history_path, metadata_path]
    partial_paths = paths + [path.with_name(path.name + '.tmp') for path in paths]
    if any(path.exists() or path.is_symlink() for path in partial_paths):
        # This check occurs before model construction or optimizer creation.
        if any(path.exists() or path.is_symlink() for path in partial_paths[len(paths):]):
            raise ValueError(f'Incomplete temporary model artifacts for {name}; refusing to overwrite')
        saved, history, metadata = load_model_artifacts(out, name, identity)
    else:
        saved = history = metadata = None
    torch.manual_seed(seed)
    model = AE(cfg['latent_dim'])
    xt = torch.from_numpy(x)
    if saved is None:
        opt = torch.optim.Adam(model.parameters(), lr=cfg['ae_learning_rate'])
        generator = torch.Generator().manual_seed(seed + 1000)
        history, best, stale = [], float('inf'), 0
        for epoch in range(1, cfg['ae_max_epochs'] + 1):
            model.train()
            running = []
            order = rng.permutation(train)
            for offset in range(0, len(order), cfg['ae_batch_size']):
                ids = order[offset:offset + cfg['ae_batch_size']]
                target = xt[ids]
                inputs = symmetric_mask(target, cfg['mask_probability'], generator) if family == 'dae' else target
                prediction, _ = model(inputs)
                loss = losses(prediction, target).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite training loss for {name}')
                opt.zero_grad()
                loss.backward()
                opt.step()
                running.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_loss = float(np.concatenate([losses(model(xt[validation[j:j + 128]])[0], xt[validation[j:j + 128]]).numpy()
                                                  for j in range(0, len(validation), 128)]).mean())
            if not np.isfinite(val_loss):
                raise ValueError(f'Nonfinite validation loss for {name}')
            history.append(dict(epoch=epoch, train_loss=float(np.mean(running)), validation_loss=val_loss))
            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                temporary = checkpoint.with_name(checkpoint.name + '.tmp')
                torch.save(dict(state_dict=model.state_dict(), family=family, seed=seed, epoch=epoch,
                                identity_sha256=_json_sha256(identity), validation_loss=val_loss,
                                train_indices=train.tolist(), validation_indices=validation.tolist()), temporary)
                os.replace(temporary, checkpoint)
            else:
                stale += 1
            if epoch % 5 == 0:
                print(name, 'epoch', epoch, 'validation', round(val_loss, 6), flush=True)
            if stale >= cfg['ae_patience']:
                break
        _atomic_json(history_path, history)
        publish_model_artifacts(out, name, identity)
        saved, history, metadata = load_model_artifacts(out, name, identity)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    scores, features = [], []
    with torch.no_grad():
        for offset in range(0, len(x), 128):
            batch = xt[offset:offset + 128]
            prediction, z = model(batch)
            scores.extend(losses(prediction, batch).tolist())
            features.extend(z.numpy())
    scores, features = np.array(scores), np.array(features)
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError(f'Nonfinite model outputs for {name}')
    test = (roles == 2) & eligible
    summary = dict(model=name, family=family, seed=seed, epochs=len(history), best_epoch=saved['epoch'],
                   train_windows=len(train), validation_windows=len(validation),
                   validation_loss=saved['validation_loss'], test_loss=float(scores[test].mean()) if test.any() else None,
                   checkpoint=checkpoint.name, checkpoint_sha256=metadata['files'][checkpoint.name]['sha256'],
                   identity_sha256=metadata['identity_sha256'], artifact_index=metadata_path.name)
    return scores, features, summary


def training_percentiles(values, widths, train, eligible):
    scores = np.zeros(len(values))
    for width in np.unique(widths):
        reference = np.sort(values[train & eligible & (widths == width)])
        if not len(reference):
            raise ValueError('Missing training scale')
        ids = widths == width
        scores[ids] = np.searchsorted(reference, values[ids], side='right') / len(reference)
    return scores


def quota_select(scores, rows, eligible, quotas, n):
    """Identical scale counts and coverage; long windows allocated first."""
    occupied = np.zeros(n, bool)
    selected = []
    for width, quota in sorted(((int(k), v) for k, v in quotas.items()), reverse=True):
        ids = [i for i, r in enumerate(rows) if eligible[i] and r['width_bins'] == width]
        ids.sort(key=lambda i: (-scores[i], i))
        count = 0
        for i in ids:
            bins = (rows[i]['start_bin'] + np.arange(width)) % n
            if not occupied[bins].any():
                occupied[bins] = True
                selected.append(i)
                count += 1
                if count == quota:
                    break
        if count != quota:
            raise ValueError(f'Cannot allocate fixed quota {width}: {count}/{quota}; do not silently change budget')
    return selected
