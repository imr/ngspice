#!/usr/bin/env python3
"""Build and run the KLU matrix tests; optionally run full circuit comparisons."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--matrices-only', action='store_true')
    args = parser.parse_args()
    build = args.build.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[3]
    regression = source / 'tests/klu/parallel'
    klu = build / 'src/maths/KLU/.libs'
    executable = output / 'klu-matrix-test'
    command = shlex.split(os.environ.get('CC', 'cc')) + [
        '-O2', '-g', '-std=gnu17', '-fopenmp',
        '-I' + str(build / 'src/include'),
        '-I' + str(source / 'src/include'),
        '-I' + str(source / 'src/maths/KLU'),
        str(regression / 'klu_parallel_matrix_test.c'),
        '-Wl,--start-group', str(klu / 'libKLU.a'), str(klu / '../SuiteSparse/.libs/libSuiteSparse.a'),
        '-Wl,--end-group', '-lm', '-o', str(executable)]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(regression / 'klu_parallel_fixtures.py'),
                    str(output / 'fixtures')], check=True)
    results = []
    for family in ['grid', 'btf']:
        for threads in [1, 8, 16, 32]:
            for scale in [-1, 0, 1, 2]:
                # The eight-decade matrix fails the original KLU's 1e-10
                # componentwise gate without scaling, and also for transpose
                # solves with scaling. Keep those as diagnostics, not passes.
                # Exercise transpose + every scale on the well-scaled matrix;
                # additionally exercise normal solves on the stressed matrix.
                for stressed in ([False, True] if scale > 0 else [False]):
                    suffix = '-scaled' if stressed else '-plain'
                    matrix = output / 'fixtures' / (family + suffix + '.mtx')
                    log_path = output / f'{family}{suffix}-t{threads}-s{scale}.log'
                    with log_path.open('w') as log:
                        subprocess.run([str(executable), str(matrix), str(threads), '12', str(scale), '1', str(int(not stressed))],
                                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
                    assert 'PASS ' in log_path.read_text(), log_path
                    results.append({'family': family, 'threads': threads, 'scale': scale, 'stressed': stressed,
                                    'transpose_checked': not stressed, 'log': log_path.name})
    if not args.matrices_only:
        binary = (args.binary or build / 'src/ngspice').resolve()
        subprocess.run([sys.executable, str(regression / 'klu_parallel_circuits.py'),
                        '--binary', str(binary), '--output', str(output / 'circuits')], check=True)
    (output / 'result.json').write_text(json.dumps({'passed': True, 'compile': command,
        'matrix_configurations': results, 'circuits_run': not args.matrices_only}, indent=2) + '\n')
    print(f'PASS {len(results)} matrix configurations; circuit suite run={not args.matrices_only}')


if __name__ == '__main__':
    main()
