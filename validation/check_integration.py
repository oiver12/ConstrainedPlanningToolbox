import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(root))
from solvers.external.fivi import evaluate_graph, export_model, recover_checkpoints, solve_fivi  # noqa: E402
from solvers.problem import POMDP  # noqa: E402


def replay_fivi(output_dir):
    directory = Path(output_dir)
    encoded = (directory / 'model.json').read_text().rstrip('\n')
    settings = json.loads((directory / 'settings.json').read_text())
    if hashlib.sha256(encoded.encode()).hexdigest() != settings['model_sha256']:
        raise ValueError('Saved model differs from the recorded model hash')
    model_data = json.loads(encoded)
    records = recover_checkpoints(directory)
    return [{**record, 'replayed_value': evaluate_graph(model_data, json.loads((directory / record['policy']).read_text()))} for record in records]


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


def verify(directory):
    records = replay_fivi(directory)
    previous = -float('inf')
    settings = json.loads((Path(directory) / 'settings.json').read_text())
    for record in records:
        np.testing.assert_allclose(record['policy_value'], record['replayed_value'], atol=1e-9, rtol=1e-10)
        assert record['best_policy_value'] >= previous
        assert 0 <= record['elapsed_seconds'] <= settings['time_limit_seconds']
        previous = record['best_policy_value']
    return records


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
            assert result.status == 'completed', result
            records = verify(result.output_dir)
            assert len(records) == 12
            optimum = exhaustive_value(export_model(model, horizon))
            assert result.value <= optimum + 1e-8, (result.value, optimum)
            np.testing.assert_allclose(result.value, optimum, atol=1e-8, rtol=1e-9)
            report['tiny_instances'].append({'horizon': horizon, 'terminal': terminal, 'negative': negative, 'value': result.value, 'optimum': optimum})
    model = fixture()
    for label, reward, terminal, expected in [('zero', 0.0, False, 0.0), ('negative', -2.0, False, -6.0), ('terminal', 5.0, True, 5.0)]:
        constant = POMDP([0], [0], [0], np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.array([[reward]]), np.ones(1), (0,) if terminal else ())
        result = solve_fivi(constant, 3, 5, output_dir=destination / label)
        np.testing.assert_allclose(result.value, expected, atol=1e-10)
        verify(result.output_dir)
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
    result = solve_fivi(model, 3, 1e-9, output_dir=destination / 'timeout-before-start')
    assert result.status == 'no_completed_policy' and result.value is None
    from experiments.registry.benchmarks import load_benchmark
    benchmark = load_benchmark('tiger')
    result = solve_fivi(benchmark.model, benchmark.horizon, 10, seed=19, output_dir=destination / 'reference-instrumented', max_iterations=4)
    expected = [record['policy_value'] for record in verify(result.output_dir)]
    build = root / 'build/fivi'
    output = destination / 'fork-reference.json'
    classpath = os.pathsep.join(map(str, [build / 'reference', build / 'fivi.jar']))
    subprocess.run([str(build / 'jdk-21.0.5+11/bin/java'), '-ea', '-cp', classpath, 'executables.FiviReference', str(result.output_dir / 'model.json'), '4', '19', str(output)], stdout=subprocess.DEVNULL, check=True)
    report['reference']['fork'] = json.loads(output.read_text())
    np.testing.assert_allclose(report['reference']['fork'], expected, atol=1e-10)
    report['reference']['instrumented'] = expected
    # Kill the Java worker after it publishes a checkpoint; the Python harness must retain it.
    killed = destination / 'forced-kill'
    code = 'from experiments.registry.benchmarks import load_benchmark; from solvers.external.fivi import solve_fivi; b=load_benchmark("tmaze"); solve_fivi(b.model,b.horizon,20,output_dir=' + repr(str(killed)) + ',max_iterations=1000000)'
    worker = subprocess.Popen([sys.executable, '-c', code], cwd=root)
    import psutil
    try:
        deadline = time.monotonic() + 15
        while not recover_checkpoints(killed) and worker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert recover_checkpoints(killed), 'No checkpoint before forced kill'
        for child in psutil.Process(worker.pid).children(recursive=True):
            if child.name() == 'java':
                child.kill()
        worker.wait(timeout=5)
        assert json.loads((killed / 'result.json').read_text())['status'] == 'error'
        records = verify(killed)
        with (killed / 'trace.jsonl').open('a') as stream:
            stream.write('{"partial":')
        assert len(recover_checkpoints(killed)) == len(records)
    finally:
        if worker.poll() is None:
            for child in psutil.Process(worker.pid).children(recursive=True):
                child.kill()
            worker.kill()
            worker.wait()
    budget = solve_fivi(benchmark.model, benchmark.horizon, 0.4, output_dir=destination / 'watchdog', max_iterations=1000000)
    assert budget.status in ('timeout', 'no_completed_policy')
    verify(budget.output_dir)
    report['watchdog'] = {'status': budget.status, 'checkpoints': len(budget.checkpoints), 'elapsed': budget.elapsed_seconds}
    if arguments.benchmarks:
        for key in ['tiger', 'tmaze', 'robot_nav']:
            benchmark = load_benchmark(key)
            directory = destination / key
            result = solve_fivi(benchmark.model, benchmark.horizon, 5, output_dir=directory, max_iterations=8)
            records = verify(directory)
            assert records, (key, result.status)
            report['benchmarks'][key] = {'status': result.status, 'value': result.value, 'checkpoints': len(records)}
    (destination / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
