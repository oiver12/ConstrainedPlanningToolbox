# Finite-horizon FiVI

Fork of [ConstrainedPlanningToolbox](https://github.com/AlgTUDelft/ConstrainedPlanningToolbox) for standalone, undiscounted finite-horizon POMDP planning.

```sh
mvn package
java -jar target/FiVI.jar --model model.json --seed 0 --budget 60
```

FiVI reports its bounds at the initial belief after every backward sweep, the solution quality
reported by Walraven and Spaan (2019): the lower bound is the value of its alpha-vector policy. No
policy graph is extracted. Options: `--budget SECONDS`, `--clock cpu|wall`, `--iterations N` (fixed
sweeps instead of convergence stopping), `--parallelism N`, and for resumable runs `--checkpoint PATH
--stop PATH --identity TEXT [--checkpoint-interval SECONDS]`.

The JSON model contains `transition[A][S][S]`, `observation[A][S][O]`, `reward[S][A]`, `initial_belief[S]`, and `horizon`.
For sparse input, replace `transition` with `transition_destinations[S][A][]`
and `transition_probabilities[S][A][]`. This avoids dense transition allocation
and scanning during JSON loading. Both formats build the same sparse successor
lists. Dense alpha vectors and observation tables remain part of FiVI.
Only zero terminal value is supported; represent early termination by an absorbing zero-reward state.

Every stdout line is one JSON message; the solver's console log goes to stderr. `bound` messages carry
the iteration, both bounds, the charged CPU and wall seconds, the number of alpha vectors per stage,
and the first-stage vector attaining the lower bound, so callers can recompute it. With a checkpoint,
the runner announces each snapshot with a `checkpoint` message and waits for `{"chunks": n}` on stdin,
so the caller can persist its rows first.

The budget charges only the solver's own work: initialization, backups, upper-bound updates and
belief expansion. The ledger is paused while the runner parses the model, restores or writes
checkpoints and reports. Work after the last checkpoint of an interrupted run is discarded and charged
again only when it is redone. A daemon samples the budget, and the numerical loops read only a flag, so
a sweep may end up to about 0.1 s past the budget. Parallel back-projections and first-pass backups are
optional and preserve vector order. `-Dfivi.profile.phases=true` enables CPU timing per phase for audits.

Changes from upstream: complete initial-belief construction, stopping thresholds for nonpositive returns, progress callbacks, fixed-iteration control, deadlines checked outside the inner loops over states and successors, a JSON runner that passes only nonzero transitions, checkpointing, parallel backups, and pinned Maven build plugins. The algorithm is the PBS variant of the paper; dependency-based bound updates (DBBU) are not implemented. JAR timestamps are fixed for reproducible builds. The fixed-iteration reference driver is compiled under `target/test-classes` for validation only.

Integration checks for the companion Python project live in `validation/`:

```sh
/path/to/project/.venv/bin/python -m pip install -r validation/requirements.txt
/path/to/project/.venv/bin/python validation/check_integration.py /path/to/project --output /path/to/project/outputs/fivi/validation --benchmarks
```

The upstream toolbox and these changes are GPL-3.0-or-later; see [LICENSE](LICENSE).
