import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(root))
from solvers.external.fivi import export_model, solve_fivi  # noqa: E402
from solvers.problem import POMDP  # noqa: E402


def exhaustive_value(data):
    transition, observation, reward = (np.asarray(data[key]) for key in ['transition', 'observation', 'reward'])
    vectors = [np.zeros(len(reward))]
    for step in range(data['horizon']):
        updated = []
        for action in range(len(transition)):
            for continuations in itertools.product(vectors, repeat=observation.shape[2]):
                continuation = np.sum(observation[action] * np.asarray(continuations).T, axis=1)
                updated.append(reward[:, action] + transition[action] @ continuation)
        vectors = updated
    return max(float(np.asarray(data['initial_belief']) @ vector) for vector in vectors)


def fixture(seed=0, terminal=False, negative=False):
    random = np.random.default_rng(seed)
    transition = random.dirichlet([1, 1], size=(2, 2))
    observation = random.dirichlet([1, 1], size=(2, 2))
    reward = -1 - random.random((2, 2)) if negative else random.normal(size=(2, 2))
    return POMDP([0, 1], [0, 1], [0, 1], transition, observation, reward, np.array([0.3, 0.7]), (1,) if terminal else ())


def verify(result, time_limit_seconds):
    """Bounds at the initial belief: the lower bound never decreases and stays below the upper bound."""
    previous = -float('inf')
    for row in result.trace:
        assert row['lower_bound'] >= previous - 1e-12
        assert row['lower_bound'] <= row['upper_bound'] + 1e-6
        assert 0 <= row['charged_cpu'] <= time_limit_seconds
        previous = row['lower_bound']
    return result.trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--benchmarks', action='store_true')
    arguments = parser.parse_args()
    destination = Path(arguments.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {'tiny_instances': [], 'reference': {}, 'benchmarks': {}}
    for horizon in [0, 1, 2, 3]:
        for terminal, negative in [(False, False), (True, False), (False, True)]:
            model = fixture(horizon, terminal, negative)
            result = solve_fivi(model, horizon, 10, output_dir=destination / f'tiny-{horizon}-{terminal}-{negative}', max_iterations=12)
            assert result.status == 'iterations', result
            records = verify(result, 10)
            assert len(records) == 12
            optimum = exhaustive_value(export_model(model, horizon))
            assert result.lower_bound <= optimum + 1e-8 <= result.upper_bound + 2e-8, (result.lower_bound, optimum, result.upper_bound)
            np.testing.assert_allclose(result.lower_bound, optimum, atol=1e-8, rtol=1e-9)
            report['tiny_instances'].append({'horizon': horizon, 'terminal': terminal, 'negative': negative, 'lower_bound': result.lower_bound, 'optimum': optimum})
    model = fixture()
    for label, reward, terminal, expected in [('zero', 0.0, False, 0.0), ('negative', -2.0, False, -6.0), ('terminal', 5.0, True, 5.0)]:
        constant = POMDP([0], [0], [0], np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.array([[reward]]), np.ones(1), (0,) if terminal else ())
        result = solve_fivi(constant, 3, 5, output_dir=destination / label)
        np.testing.assert_allclose(result.lower_bound, expected, atol=1e-10)
        verify(result, 5)
    for kwargs in [{'horizon': -1}, {'horizon': 1.5}, {'time_limit_seconds': 0}, {'discount': 0.95}, {'terminal_values': np.zeros(2)}, {'max_iterations': 0}]:
        options = dict(horizon=2, time_limit_seconds=1, output_dir=destination / 'invalid')
        options.update(kwargs)
        try:
            solve_fivi(model, **options)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    try:
        solve_fivi(model, 2, 1, build_dir=destination / 'missing')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('missing build accepted')
    invalid = fixture()
    invalid.reward[0, 0] = np.nan
    try:
        solve_fivi(invalid, 2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite reward accepted')
    result = solve_fivi(model, 3, 1e-9, output_dir=destination / 'budget-before-first-sweep')
    assert result.status == 'budget' and result.lower_bound is None
    from experiments.registry.benchmarks import load_benchmark
    benchmark = load_benchmark('tiger')
    result = solve_fivi(benchmark.model, benchmark.horizon, 10, seed=19, output_dir=destination / 'reference-instrumented', max_iterations=4)
    expected = [row['lower_bound'] for row in verify(result, 10)]
    build = root / 'build/fivi-sparse-paper'
    output = destination / 'fork-reference.json'
    dense_model = destination / 'reference-dense-model.json'
    dense_model.write_text(json.dumps(export_model(benchmark.model, benchmark.horizon)))
    classpath = os.pathsep.join(map(str, [build / 'source/target/test-classes', build / 'fivi.jar']))
    subprocess.run([str(build / 'jdk-21.0.5+11/bin/java'), '-ea', '-cp', classpath, 'executables.FiviReference', str(dense_model), '4', '19', str(output)], stdout=subprocess.DEVNULL, check=True)
    report['reference']['fork'] = json.loads(output.read_text())
    np.testing.assert_allclose(report['reference']['fork'], expected, atol=1e-10)
    report['reference']['runner'] = expected
    budget = solve_fivi(benchmark.model, benchmark.horizon, 0.4, output_dir=destination / 'watchdog', max_iterations=1000000)
    assert budget.status in ('budget', 'converged')
    verify(budget, 0.4)
    report['watchdog'] = {'status': budget.status, 'sweeps': len(budget.trace), 'charged_seconds': budget.charged_seconds}
    if arguments.benchmarks:
        for key in ['tiger', 'tmaze', 'robot_nav']:
            benchmark = load_benchmark(key)
            directory = destination / key
            result = solve_fivi(benchmark.model, benchmark.horizon, 5, output_dir=directory, max_iterations=8)
            records = verify(result, 5)
            assert records, (key, result.status)
            report['benchmarks'][key] = {'status': result.status, 'lower_bound': result.lower_bound, 'upper_bound': result.upper_bound, 'vectors_per_stage': result.vectors_per_stage, 'sweeps': len(records)}
    (destination / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
