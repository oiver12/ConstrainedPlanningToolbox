# Finite-horizon FiVI

Fork of [ConstrainedPlanningToolbox](https://github.com/AlgTUDelft/ConstrainedPlanningToolbox) for standalone, undiscounted finite-horizon POMDP planning.

```sh
mvn package
java -jar target/FiVI.jar model.json output-directory seconds seed iterations
```

Create the output directory first. Use `iterations=0` for normal convergence stopping.
The JSON model contains `transition[A][S][S]`, `observation[A][S][O]`, `reward[S][A]`, `initial_belief[S]`, and `horizon`.
Only zero terminal value is supported; represent early termination by an absorbing zero-reward state.

Each completed sweep saves an evaluated policy graph and a flushed `java-trace.jsonl` record. A cooperative deadline preserves earlier checkpoints. The caller must enforce an external process limit and account for startup and export time.

Changes from upstream: complete initial-belief construction, stopping thresholds for nonpositive returns, progress callbacks, fixed-iteration control, deadlines, a JSON runner, and pinned Maven build plugins. JAR timestamps are fixed for reproducible builds. The fixed-iteration reference driver is compiled under `target/test-classes` for validation only.

The upstream toolbox and these changes are GPL-3.0-or-later; see [LICENSE](LICENSE).
