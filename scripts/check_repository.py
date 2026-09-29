"""Check prospective public files without modifying the Git index."""
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', 'env', '__pycache__', '.pytest_cache',
            '.ruff_cache', '.mypy_cache', '.agents', '.codex'}


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
                       if name and (ROOT/name.decode()).is_file()})
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not EXCLUDED.intersection(p.relative_to(ROOT).parts))


def main():
    files = public_files()
    public = {p.resolve() for p in files}
    errors, total, links = [], 0, 0
    for path in files:
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        total += size
        if not path.resolve().is_relative_to(ROOT):
            errors.append(f'External symlink: {relative}')
        if size > 5*1024**2:
            errors.append(f'File exceeds 5 MiB: {relative}')
        if (relative.parts[0] in {'results', 'checkpoints', 'micro-c数据', '.agents', '.codex'}
                or relative.as_posix().startswith('data/cache/')
                or EXCLUDED.intersection(relative.parts)
                or path.suffix.lower() in {'.cool', '.mcool', '.hic', '.bam', '.bai', '.xlsx',
                                           '.pdf', '.pt', '.pth', '.pkl', '.npz', '.npy'}
                or (path.name.startswith('.env') and path.name != '.env.example')):
            errors.append(f'Local-only file in public inventory: {relative}')
        if path.suffix not in {'.md', '.html'}:
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
            dest = (path.parent/unquote(parsed.path)).resolve()
            if dest not in public and not (dest.is_dir() and any(p.is_relative_to(dest) for p in public)):
                errors.append(f'Unpublished link: {relative} -> {target}')
    if total > 20*1024**2:
        errors.append(f'Public inventory exceeds 20 MiB: {total:,} bytes')
    if errors:
        raise SystemExit('\n'.join(errors))
    subprocess.run([sys.executable, str(ROOT/'scripts/export_showcase.py'), '--check'], check=True)
    print(f'Public repository verified: {len(files)} files, {total:,} bytes, {links} local Markdown/HTML links')


if __name__ == '__main__':
    main()
