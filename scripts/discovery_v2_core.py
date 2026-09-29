"""Training-only normalization, spatial roles and paired AE/DAE experiments."""
import json
import numpy as np
import torch
from discover_task2 import AE, losses


def bin_roles(n, cfg):
    block = np.arange(n) // cfg['spatial_block_bins'] % cfg['split_block_modulus']
    roles = np.full(n, -1, int)
    for role, name in enumerate(['train', 'validation', 'test']):
        roles[np.isin(block, cfg[f'{name}_block_remainders'])] = role
    assert (roles >= 0).all()
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


def fit_model(x, roles, eligible, cfg, out, family, seed):
    name = f'{family}_{seed}'
    rng = np.random.default_rng(seed)
    train = np.flatnonzero((roles == 0) & eligible)
    train = rng.choice(train, min(len(train), cfg['ae_max_train_windows']), replace=False)
    validation = np.flatnonzero((roles == 1) & eligible)
    if not len(train) or not len(validation):
        raise ValueError('Empty train or validation partition')
    torch.manual_seed(seed)
    model = AE(cfg['latent_dim'])
    checkpoint = out / f'{name}.pt'
    history_path = out / f'{name}_history.json'
    xt = torch.from_numpy(x)
    if checkpoint.exists() and history_path.exists():
        history = json.loads(history_path.read_text())
    else:
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
                opt.zero_grad()
                loss.backward()
                opt.step()
                running.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_loss = float(np.concatenate([losses(model(xt[validation[j:j + 128]])[0], xt[validation[j:j + 128]]).numpy()
                                                  for j in range(0, len(validation), 128)]).mean())
            history.append(dict(epoch=epoch, train_loss=float(np.mean(running)), validation_loss=val_loss))
            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                torch.save(dict(state_dict=model.state_dict(), family=family, seed=seed, epoch=epoch,
                                validation_loss=val_loss, train_indices=train.tolist(), validation_indices=validation.tolist()), checkpoint)
            else:
                stale += 1
            if epoch % 5 == 0:
                print(name, 'epoch', epoch, 'validation', round(val_loss, 6), flush=True)
            if stale >= cfg['ae_patience']:
                break
        history_path.write_text(json.dumps(history, indent=2))
    saved = torch.load(checkpoint, weights_only=False)
    np.testing.assert_array_equal(saved['train_indices'], train)
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
    summary = dict(model=name, family=family, seed=seed, epochs=len(history), best_epoch=saved['epoch'],
                   train_windows=len(train), validation_windows=len(validation),
                   validation_loss=saved['validation_loss'], test_loss=float(scores[(roles == 2) & eligible].mean()))
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
