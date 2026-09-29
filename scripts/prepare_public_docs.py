"""Map local research-output links to public snapshots or explicit local-path text."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(!?)\[([^\]]+)\]\(([^)]+)\)')


def main():
    changed = []
    for path in (ROOT/'docs').glob('*.md'):
        if path.name in {'README.md', 'REPOSITORY.md', 'REPRODUCIBILITY.md'}:
            continue
        def replace(match):
            image, label, target = match.groups()
            if target.startswith('../results/'):
                mapped = '../examples/results/'+target[len('../results/'):]
                if (path.parent/mapped).is_file():
                    return f'{image}[{label}]({mapped})'
                return f'{label}（本地生成：`{target.removeprefix("../")}`）'
            if target.lower().endswith('.pdf'):
                return f'{label}（课程材料，本地提供）'
            # Absolute workspace links should never appear in the public documentation.
            if target.startswith('/home/'):
                return f'{label}（本地路径）'
            return match.group(0)
        text = path.read_text()
        updated = LINK.sub(replace, text)
        if updated != text:
            path.write_text(updated)
            changed.append(path.relative_to(ROOT).as_posix())
    print(f'Prepared public links in {len(changed)} documents')


if __name__ == '__main__':
    main()
