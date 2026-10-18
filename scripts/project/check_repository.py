"""Check prospective public files without modifying the Git index."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import re
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = _layout_root
EXCLUDED = {'.git', '.venv', 'venv', 'env', '__pycache__', '.pytest_cache',
            '.ruff_cache', '.mypy_cache', '.agents', '.codex'}
MAX_SOURCE_BYTES = 32 * 1024**2  # Retain both frozen scientific source and current runtime adapters.
MAX_TRACK_ATLAS_BYTES = 96 * 1024**2  # Explicit complete 465-page digital supplement, checked separately.
TRACK_ATLAS = 'submission/task3/atlas/v1/'


def local_history(path):
    relative = path.relative_to(ROOT)
    public_reports = {"reports/README.md", "reports/shared/data_audit.json",
                      "reports/task1/diagnostics/task1_results.json",
                      "reports/shared/conditions/task34_verification.json",
                      "reports/shared/tasks56/task56_results.json",
                      "reports/shared/tasks56/task56_verification.json"}
    return (relative.as_posix() == 'AGENTS.md'
            or relative.parts[0] in {'results', 'checkpoints', 'micro-c数据', 'local_archive',
                                  'report_materials', 'coursework'}
            or (relative.parts[0] == 'reports' and relative.as_posix() not in public_reports)
            or relative.as_posix().startswith(('data/raw/', 'data/task1/cache/', 'data/shared/cache/',
                                               'data/task6/cache/', 'data/task1/ablation/'))
            or relative.as_posix().startswith(('data/cache/', 'data/legacy_experiments/',
                                               'data/task1_native_roi_v2/'))
            or (relative.parts[0] == 'evidence' and relative.name != 'freeze_manifest.json')
            or path.suffix.lower() == '.zip'
            or path.name.endswith(('.zip.sha256', ':Zone.Identifier')))


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in {'href', 'src'} and value)


def public_files():
    # A source archive has no .git; ignore interpreter caches created after extraction.
    if (ROOT/'.git').exists():
        output = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
        return sorted({ROOT/name.decode() for name in output.split(b'\0')
                       if name and (ROOT/name.decode()).is_file()
                       and not local_history(ROOT/name.decode())})
    local_only = {'results', 'checkpoints', 'micro-c数据', 'local_archive', '.git'}
    files = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if (not path.is_file() or EXCLUDED.intersection(relative.parts)
                or relative.parts[0] in local_only or local_history(path)):
            continue
        if relative.as_posix().startswith('data/cache/'):
            continue
        files.append(path)
    return sorted(files)


def main():
    files = public_files()
    public = {p.resolve() for p in files}
    errors, total, links, local_scientific_links = [], 0, 0, 0
    atlas_total = 0
    if (ROOT / TRACK_ATLAS).exists():
        # The extra budget applies only to a complete signed atlas, never arbitrary outputs.
        from microcture.tasks.task3.atlas import verify as verify_atlas
        verify_atlas(ROOT / TRACK_ATLAS)
    selected_artifacts = set()
    # Only the selected, checksummed inference artifacts are part of the hand-in.
    manifest_path = ROOT/'submission/manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        selected_artifacts = {'submission/'+row['path'] for row in manifest.get('files', [])}
    for path in files:
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        total += size
        if relative.as_posix().startswith(TRACK_ATLAS):
            atlas_total += size
        if not path.resolve().is_relative_to(ROOT):
            errors.append(f'External symlink: {relative}')
        if size > 5*1024**2:
            errors.append(f'File exceeds 5 MiB: {relative}')
        if (relative.parts[0] in {'results', 'checkpoints', 'micro-c数据', '.agents', '.codex'}
                or relative.as_posix().startswith('data/cache/')
                or EXCLUDED.intersection(relative.parts)
                or path.suffix.lower() in {'.cool', '.mcool', '.hic', '.bam', '.bai', '.xlsx', '.pdf', '.pth', '.pkl', '.npy'}
                or (path.suffix.lower() in {'.pt', '.npz'} and
                    (relative.as_posix() not in selected_artifacts or relative.parts[0] != 'submission'))
                or (path.name.startswith('.env') and path.name != '.env.example')):
            errors.append(f'Local-only file in public inventory: {relative}')
        if path.suffix not in {'.md', '.html'}:
            continue
        if relative.as_posix().startswith('microcture/history/'):
            # Byte-exact historical protocols retain their original links.
            continue
        content = path.read_text(encoding='utf-8')
        if path.suffix == '.md':
            content = re.sub(r'```.*?```', '', content, flags=re.S)
            targets = re.findall(r'\[[^\]]*\]\(([^\n)]+)\)', content)
            parser = HTMLLinks()
            parser.feed(content)
            targets.extend(parser.links)
        else:
            parser = HTMLLinks()
            parser.feed(content)
            targets = parser.links
        for target in targets:
            target = target.strip().strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            links += 1
            dest = _layout_resolve(path.parent/unquote(parsed.path)).resolve()
            if dest not in public and not (dest.is_dir() and any(p.is_relative_to(dest) for p in public)):
                historical_note = (relative.parts[0] == 'docs' and
                                   (path.name == 'WORK_LOG.md' or path.name.startswith('TASK1_')))
                if historical_note and dest.is_relative_to(ROOT) and dest.exists():
                    local_scientific_links += 1
                else:
                    errors.append(f'Unpublished link: {relative} -> {target}')
    if total - atlas_total > MAX_SOURCE_BYTES:
        errors.append(f'Source inventory excluding verified track atlas exceeds 32 MiB: {total-atlas_total:,} bytes')
    if atlas_total > MAX_TRACK_ATLAS_BYTES:
        errors.append(f'Complete track atlas exceeds its explicit 96 MiB budget: {atlas_total:,} bytes')
    if errors:
        raise SystemExit('\n'.join(errors))
    subprocess.run([sys.executable, '-B', str(_layout_resolve(ROOT/'scripts/export_showcase.py')), '--check'], check=True)
    if manifest_path.exists():
        subprocess.run([sys.executable, '-B', str(_layout_resolve(ROOT/'scripts/check_submission_evidence.py'))], check=True)
    if (ROOT/'submission/results_delivery_manifest.json').exists():
        subprocess.run([sys.executable, '-B', str(_layout_resolve(ROOT/'scripts/build_results_delivery.py')), 'verify'], check=True)
    print(f'Workspace verified: {len(files)} source/evidence files, {total:,} bytes, '
          f'{atlas_total:,} bytes in the verified 465-page atlas; '
          f'{links} local Markdown/HTML links; {local_scientific_links} historical scientific links retained locally')


if __name__ == '__main__':
    main()
