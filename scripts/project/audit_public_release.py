"""Plan a public snapshot and inspect reachable Git history, without staging.

Findings contain locations and counts, never matched credentials or raw emails.
This is a heuristic review aid, not a guarantee that a repository is secret-free.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'microcture/layout.py').is_file())
sys.path.insert(0, str(ROOT))
from microcture.layout import bootstrap
bootstrap()
from check_repository import public_files

PATTERNS = {
    'private_key_header': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{32,}|github_pat_[A-Za-z0-9_]{40,})'),
    'openai_style_token': re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}'),
    'aws_access_key': re.compile(rb'AKIA[0-9A-Z]{16}'),
    'credential_url': re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@'),
    'plain_credential_assignment': re.compile(rb'''(?i)\b(?:password|api[_-]?key|access[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
    'student_id': re.compile(rb'(?<![A-Za-z0-9])3024[0-9]{6}(?![A-Za-z0-9])'),
}
LOCAL_PATH = re.compile(rb'/home/[^\s/"\']+')
EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
BLOCKING_FINDINGS = set(PATTERNS) | {'forbidden_public_file', 'restricted_history_format'}
PRIVATE_SUFFIXES = {'.docx', '.pptx', '.pdf', '.xlsx', '.cool', '.mcool',
                    '.hic', '.bam', '.zip', '.pem', '.key'}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def scan(data):
    """Return redacted finding counts, including useful provenance warnings."""
    hits = {name: len(pattern.findall(data)) for name, pattern in PATTERNS.items()
            if pattern.search(data)}
    if LOCAL_PATH.search(data):
        hits['local_path_provenance'] = len(LOCAL_PATH.findall(data))
    # Do not interpret compressed PNG bytes or Python matrix multiplication as emails.
    if b'\x00' not in data:
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            addresses = [v for v in EMAIL.findall(data)
                         if not v.endswith((b'.pinv', b'.T', b'.dot'))]
            if addresses:
                hits['email_review'] = len(addresses)
    return hits


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe publication path')
    return path


def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root)


def tracked_entries(root=ROOT):
    result = {}
    for row in git('ls-files', '--stage', '-z', root=root).split(b'\0'):
        if not row:
            continue
        header, name = row.split(b'\t', 1)
        mode, blob, stage = header.decode().split()
        if stage != '0':
            raise ValueError('Resolve the conflicted index before planning publication')
        result[name.decode()] = {'mode': mode, 'blob': blob}
    return result


def plan(root=ROOT, candidates=None):
    tracked = tracked_entries(root)
    rows, findings = [], []
    for path in sorted(public_files() if candidates is None else candidates):
        name = path.relative_to(root).as_posix()
        safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Publication forbids symlinks: ' + name)
        data = path.read_bytes()
        old = tracked.get(name)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        state = 'new' if old is None else ('unchanged' if blob == old['blob'] else 'modified')
        rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                     'change': state})
        hits = scan(data)
        if path.suffix.lower() in PRIVATE_SUFFIXES:
            hits['forbidden_public_file'] = 1
        if hits:
            findings.append({'path': name, 'findings': hits})
    names = {row['path'] for row in rows}
    remove = [{'path': name, 'exists_locally': (root / name).is_file(),
               'reason': 'local-only; remove from index, retain local file' if (root / name).is_file()
               else 'old path already relocated/deleted in worktree'}
              for name in sorted(set(tracked) - names)]
    return rows, remove, findings


def history(root=ROOT):
    names = {}
    for line in git('rev-list', '--objects', '--all', root=root).decode().splitlines():
        oid, _, name = line.partition(' ')
        names[oid] = name
    head = {line.split(' ', 1)[0] for line in
            git('rev-list', '--objects', 'HEAD', root=root).decode().splitlines()}
    records, blob_count, blob_bytes, large = [], 0, 0, []
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for oid, name in names.items():
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            object_id, kind, size = process.stdout.readline().decode().split()
            data = process.stdout.read(int(size))
            if process.stdout.read(1) != b'\n' or len(data) != int(size):
                raise ValueError('Incomplete Git object stream')
            if kind != 'blob':
                continue
            blob_count += 1
            blob_bytes += len(data)
            if len(data) > 5 * 1024**2:
                large.append({'object': object_id, 'path': name, 'bytes': len(data)})
            hits = scan(data)
            if Path(name).suffix.lower() in PRIVATE_SUFFIXES:
                hits['restricted_history_format'] = 1
            if hits:
                records.append({'object': object_id, 'path': name,
                                'reachable_from_head': oid in head, 'findings': hits})
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise ValueError('Git object inspection failed')
    emails = sorted(set(git('log', '--all', '--format=%ae%n%ce', root=root).decode().splitlines()))
    return {'blobs_scanned': blob_count, 'blob_bytes': blob_bytes, 'over_5MiB': large,
            'findings': records,
            'commit_emails': [{'masked': re.sub(r'(^.).*(@.*$)', r'\1***\2', value),
                               'requires_review': not value.endswith(('@users.noreply.github.com', '@github.com'))}
                              for value in emails]}


def audit(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'reports') or output.exists():
        raise ValueError('Choose a fresh local reports/ directory')
    index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = ROOT / index
    index_before = sha256(index)
    rows, remove, findings = plan()
    old = history()
    blockers = [row for row in findings if set(row['findings']) & BLOCKING_FINDINGS]
    history_blockers = [row for row in old['findings']
                        if row['reachable_from_head'] and set(row['findings']) & BLOCKING_FINDINGS]
    if sha256(index) != index_before:
        raise ValueError('Git index changed during the read-only audit; inspect before publishing')
    summary = {'schema_version': 1, 'status': 'blocked_for_push' if blockers or history_blockers else 'requires_owner_review',
               'root': str(ROOT), 'branch': git('branch', '--show-current').decode().strip(),
               'head': git('rev-parse', 'HEAD').decode().strip(),
               'remotes': git('remote', '-v').decode().splitlines(),
               'public_files': len(rows), 'public_bytes': sum(row['bytes'] for row in rows),
               'new_files': sum(row['change'] == 'new' for row in rows),
               'new_file_bytes': sum(row['bytes'] for row in rows if row['change'] == 'new'),
               'change_counts': dict(Counter(row['change'] for row in rows)),
               'tracked_paths_to_remove': len(remove),
               'locally_preserved_tracked_paths_to_untrack': sum(row['exists_locally'] for row in remove),
               'largest_files': sorted(rows, key=lambda row: row['bytes'], reverse=True)[:12],
               'files_over_50MiB': [row['path'] for row in rows if row['bytes'] > 50 * 1024**2],
               'files_over_100MiB': [row['path'] for row in rows if row['bytes'] > 100 * 1024**2],
               'current_snapshot_blockers': blockers, 'head_history_blockers': history_blockers,
               'git_index_unchanged': True, 'commit_performed': False, 'push_performed': False,
               'warning': 'Heuristic scan; matched values are never printed. Review provenance paths, source metadata and commit identities separately.'}
    output.mkdir(parents=True)
    for name, value in [('summary.json', summary), ('inventory.json', rows),
                        ('remove_from_index_plan.json', remove),
                        ('privacy.json', {'prospective': findings, 'history': old})]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    (output / 'public_paths.nul').write_bytes(b''.join(row['path'].encode() + b'\0' for row in rows))
    (output / 'remove_from_index_paths.nul').write_bytes(b''.join(row['path'].encode() + b'\0' for row in remove))
    print(json.dumps({key: summary[key] for key in ('status', 'public_files', 'public_bytes',
                     'new_files', 'new_file_bytes', 'tracked_paths_to_remove', 'git_index_unchanged')}, indent=2))
    return 2 if blockers or history_blockers else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh local reports/ directory; never stages or commits')
    args = parser.parse_args(argv)
    return audit(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
