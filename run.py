"""MicroCTure coursework entry point; no computation is started without a command."""
import argparse
from pathlib import Path
import subprocess
import sys
from microcture.layout import bootstrap, resolve

bootstrap()

ROOT = Path(__file__).resolve().parent
ENTRIES = {
    'task1': 'run_task1_local.py',
    'task2': 'discover_task2_v2.py',
    'task2-v1': 'discover_task2.py',
    'task34': 'run_conditions_local.py',
    'task3-atlas': 'task3/build_atlas.py',
    'task5': 'train_task5.py',
    'task5-analysis': 'task5/analyze_saved.py',
    'task6': 'train_task6.py',
    'prepare-task6': 'prepare_task6.py',
    'prepare-bands': 'prepare_band_cache.py',
    'review-task2': 'review_saved_centering.py',
    'audit-task2': 'task2/review_task2.py',
    'validate-task2': 'task2/validate_candidates.py',
    'publish-task2-validation': 'task2/publish_validation.py',
    'review-task56': 'review_saved_task56.py',
    'report-results': 'build_results_delivery.py',
    'package': 'build_submission.py',
    'verify-evidence': 'shared/verify_evidence.py',
}


def call(script, args):
    return subprocess.call([sys.executable, '-B', str(resolve(ROOT/'scripts'/script)), *args], cwd=ROOT)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'demo':
        return call('predict_task1_final.py', argv[1:] or ['--verify-demo'])
    if argv and argv[0] in ENTRIES:
        return call(ENTRIES[argv[0]], argv[1:] or ['--help'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['check', 'demo', 'serve', *ENTRIES],
                        help='check: portable validation; demo: bundled M2+ inference; taskN: explicit computations')
    parser.add_argument('--skip-tests', action='store_true', help='check: verify saved evidence and links only')
    parser.add_argument('--port', type=int, default=8000, help='serve: local HTTP port')
    args, rest = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'check':
        if rest:
            parser.error('Unknown check arguments: '+' '.join(rest))
        code = call('check_repository.py', [])
        if code or args.skip_tests:
            return code
        import os
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(sys.path))
        return subprocess.call([sys.executable, '-B', '-m', 'scripts.shared.legacy_tests'], cwd=ROOT, env=environment)
    if args.command == 'demo':
        return call('predict_task1_final.py', rest or ['--help'])
    if args.command == 'serve':
        if rest:
            parser.error('Unknown serve arguments: '+' '.join(rest))
        print(f'Open http://localhost:{args.port}/site/', flush=True)
        return subprocess.call([sys.executable, '-B', '-m', 'http.server', str(args.port), '--bind', '127.0.0.1'], cwd=ROOT)
    parser.error('Use an explicit task command')


if __name__ == '__main__':
    raise SystemExit(main())
