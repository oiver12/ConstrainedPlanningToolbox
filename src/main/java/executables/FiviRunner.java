// SPDX-License-Identifier: GPL-3.0-or-later
package executables;

import algorithms.pomdp.cgcp.FiniteVI;
import com.google.gson.*;
import model.*;
import java.io.*;
import java.lang.management.ManagementFactory;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.*;

/**
 * Runs FiVI and reports its bounds at the initial belief, the solution quality reported by
 * Walraven and Spaan (2019). No policy graph is extracted.
 *
 * Options: --model PATH --seed N [--budget SECONDS] [--clock cpu|wall] [--iterations N] [--parallelism N]
 *          [--checkpoint PATH --stop PATH --identity TEXT [--checkpoint-interval SECONDS]]
 *
 * Every stdout line is one JSON message {"kind": ..., "value": {...}}:
 *   start       model dimensions, and the iteration and charge restored from a checkpoint
 *   bound       iteration, lower_bound, upper_bound, charged_cpu, charged_wall, vectors_per_stage, initial_vector
 *   checkpoint  the caller persists its rows and answers {"chunks": n} on stdin before the snapshot is written
 *   saved       the snapshot now holds this iteration
 *   done        status converged, iterations, budget or interrupted
 *
 * The budget charges only the solver's own work. The ledger is paused while the runner parses the model,
 * restores or writes checkpoints and reports, so restarts and bookkeeping cost nothing. Work after the last
 * checkpoint of an interrupted run is discarded and charged again only when it is redone.
 */
public class FiviRunner {
    static final Gson gson = new Gson();
    // Protocol messages own stdout; the solver's console log goes to stderr.
    static final PrintStream protocol = System.out;

    static class Input {
        double[][][] transition, observation;
        int[][][] transition_destinations;
        double[][][] transition_probabilities;
        double[][] reward;
        double[] initial_belief;
        int horizon;
    }

    static class Snapshot implements Serializable {
        private static final long serialVersionUID = 2L;
        String identity;
        FiniteVI solver;
        POMDP model;
        int horizon, chunks;
        double chargedCpu, chargedWall;
    }

    /** Charged solver time; paused segments are never counted. */
    static final class Ledger {
        static final com.sun.management.OperatingSystemMXBean bean = (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        final boolean wallClock;
        final double budget;
        volatile double chargedCpu, chargedWall;
        volatile double segmentCpu = Double.NaN, segmentWall;

        Ledger(boolean wallClock, double budget, double chargedCpu, double chargedWall) {
            this.wallClock = wallClock; this.budget = budget; this.chargedCpu = chargedCpu; this.chargedWall = chargedWall;
        }
        static double cpu() { return bean.getProcessCpuTime() * 1e-9; }
        static double wall() { return System.nanoTime() * 1e-9; }
        void resume() { segmentWall = wall(); segmentCpu = cpu(); }
        void pause() {
            double cpu = cpu(), wall = wall(), startCpu = segmentCpu;
            // Mark the segment closed before adding it, so a concurrent reader can only undercount.
            segmentCpu = Double.NaN;
            chargedCpu += cpu - startCpu; chargedWall += wall - segmentWall;
        }
        boolean expired() {
            double startCpu = segmentCpu;
            boolean running = !Double.isNaN(startCpu);
            double used = wallClock ? chargedWall + (running ? wall() - segmentWall : 0) : chargedCpu + (running ? cpu() - startCpu : 0);
            return used >= budget;
        }
    }

    static void emit(String kind, Object value) {
        Map<String,Object> message = new LinkedHashMap<>();
        message.put("kind", kind); message.put("value", value);
        protocol.println(gson.toJson(message)); protocol.flush();
    }

    static Map<String,String> options(String[] arguments) {
        Map<String,String> options = new HashMap<>();
        for (int i = 0; i < arguments.length; i += 2) {
            if (!arguments[i].startsWith("--") || i+1 >= arguments.length) throw new IllegalArgumentException("Expected --name value pairs");
            options.put(arguments[i].substring(2), arguments[i+1]);
        }
        return options;
    }

    static POMDP readModel(Path path, int[] horizon) throws IOException {
        Input input = gson.fromJson(new String(Files.readAllBytes(path), StandardCharsets.UTF_8), Input.class);
        int states = input.initial_belief.length, actions = input.observation.length;
        POMDP model = new POMDP(states, actions, input.observation[0][0].length, input.observation, new BeliefPoint(input.initial_belief), input.horizon);
        model.setRewardFunction(input.reward);
        // Sparse successor lists: the solver loops only over listed destinations.
        int[][][] destinations = input.transition_destinations;
        double[][][] probabilities = input.transition_probabilities;
        if (destinations == null) {
            destinations = new int[states][actions][];
            probabilities = new double[states][actions][];
            for (int s=0; s<states; s++) for (int a=0; a<actions; a++) {
                double[] row = input.transition[a][s];
                int count = 0;
                for (double p : row) if (p != 0.0) count++;
                destinations[s][a] = new int[count];
                probabilities[s][a] = new double[count];
                for (int next=0, i=0; next<states; next++) if (row[next] != 0.0) {
                    destinations[s][a][i] = next;
                    probabilities[s][a][i++] = row[next];
                }
            }
        }
        model.setTransitionFunction(destinations, probabilities);
        horizon[0] = input.horizon;
        return model;
    }

    static class FastGzip extends GZIPOutputStream {
        FastGzip(OutputStream output) throws IOException { super(output); def.setLevel(Deflater.BEST_SPEED); }
    }

    static void save(Path path, Snapshot snapshot) throws IOException {
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
        try (FileOutputStream file = new FileOutputStream(temporary.toFile()); GZIPOutputStream zip = new FastGzip(file); ObjectOutputStream stream = new ObjectOutputStream(zip)) {
            stream.writeObject(snapshot); stream.flush(); zip.finish(); file.getFD().sync();
        }
        Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        try (FileChannel directory = FileChannel.open(path.toAbsolutePath().getParent(), StandardOpenOption.READ)) { directory.force(true); }
    }

    static Snapshot load(Path path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream stream = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(path)))) { return (Snapshot) stream.readObject(); }
    }

    public static void main(String[] arguments) throws Exception {
        System.setOut(System.err);
        Map<String,String> options = options(arguments);
        Path checkpoint = options.containsKey("checkpoint") ? Paths.get(options.get("checkpoint")) : null;
        Path stopRequest = options.containsKey("stop") ? Paths.get(options.get("stop")) : null;
        String identity = options.getOrDefault("identity", "");
        double budget = Double.parseDouble(options.getOrDefault("budget", "Infinity"));
        boolean wallClock = options.getOrDefault("clock", "cpu").equals("wall");
        int maxIterations = Integer.parseInt(options.getOrDefault("iterations", "0"));
        double checkpointInterval = Double.parseDouble(options.getOrDefault("checkpoint-interval", "60"));
        BufferedReader replies = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        Snapshot snapshot;
        if (checkpoint != null && Files.exists(checkpoint)) {
            snapshot = load(checkpoint);
            if (!snapshot.identity.equals(identity)) throw new IOException("Checkpoint identity mismatch");
        } else {
            snapshot = new Snapshot();
            snapshot.identity = identity;
            int[] horizon = new int[1];
            snapshot.model = readModel(Paths.get(options.get("model")), horizon);
            snapshot.horizon = horizon[0];
            snapshot.solver = new FiniteVI(new Random(Long.parseLong(options.get("seed"))));
        }
        FiniteVI solver = snapshot.solver;
        solver.setTerminateTime(Double.POSITIVE_INFINITY);
        solver.setParallelism(Integer.parseInt(options.getOrDefault("parallelism", "1")));
        if (maxIterations > 0) { solver.setMaxIterations(maxIterations); solver.fixedIterations = true; }
        Map<String,Object> start = new LinkedHashMap<>();
        start.put("states", snapshot.model.getNumStates()); start.put("actions", snapshot.model.getNumActions()); start.put("observations", snapshot.model.getNumObservations());
        start.put("horizon", snapshot.horizon); start.put("iteration", solver.completedIterations); start.put("chunks", snapshot.chunks);
        start.put("charged_cpu", snapshot.chargedCpu); start.put("charged_wall", snapshot.chargedWall);
        emit("start", start);

        Ledger ledger = new Ledger(wallClock, budget, snapshot.chargedCpu, snapshot.chargedWall);
        AtomicBoolean expired = new AtomicBoolean();
        // Clock reads are system calls, so a daemon samples the budget and the solver only reads a flag.
        Thread monitor = new Thread(() -> {
            while (!expired.get()) {
                if (ledger.expired()) { expired.set(true); return; }
                try { Thread.sleep(100); } catch (InterruptedException stop) { return; }
            }
        }, "fivi-budget-monitor");
        monitor.setDaemon(true);
        solver.deadlineCheck = () -> { if (expired.get()) throw new CancellationException("budget"); };
        double[] lastSaved = {snapshot.chargedCpu};
        solver.progress = (iteration, lower, upper) -> {
            ledger.pause();
            double[][] vectors = solver.getInitialStageVectors();
            double[] belief = snapshot.model.getInitialBelief().getBelief();
            double[] best = vectors[0];
            for (double[] vector : vectors) if (dot(vector, belief) > dot(best, belief)) best = vector;
            Map<String,Object> row = new LinkedHashMap<>();
            row.put("iteration", iteration); row.put("lower_bound", lower); row.put("upper_bound", upper);
            row.put("charged_cpu", ledger.chargedCpu); row.put("charged_wall", ledger.chargedWall);
            row.put("vectors_per_stage", solver.getVectorCounts()); row.put("initial_vector", best);
            row.put("phase_cpu", solver.getPhaseCpu());
            emit("bound", row);
            ledger.resume();
        };
        solver.boundary = () -> {
            ledger.pause();
            boolean stopping = stopRequest != null && Files.exists(stopRequest);
            if (checkpoint != null && (stopping || solver.finished || ledger.chargedCpu - lastSaved[0] >= checkpointInterval)) {
                try {
                    emit("checkpoint", Map.of("iteration", solver.completedIterations));
                    String reply = replies.readLine();
                    // The caller is gone, so nothing could record this checkpoint's rows.
                    if (reply == null) throw new CancellationException("interrupted");
                    snapshot.chunks = JsonParser.parseString(reply).getAsJsonObject().get("chunks").getAsInt();
                    snapshot.chargedCpu = ledger.chargedCpu; snapshot.chargedWall = ledger.chargedWall;
                    save(checkpoint, snapshot);
                    lastSaved[0] = ledger.chargedCpu;
                    emit("saved", Map.of("iteration", solver.completedIterations, "chunks", snapshot.chunks, "charged_cpu", snapshot.chargedCpu, "charged_wall", snapshot.chargedWall));
                } catch (IOException error) { throw new UncheckedIOException(error); }
            }
            if (stopping) throw new CancellationException("interrupted");
            ledger.resume();
        };
        String status;
        monitor.start();
        ledger.resume();
        try {
            solver.solve(snapshot.model, snapshot.horizon);
            status = maxIterations > 0 && solver.completedIterations >= maxIterations ? "iterations" : "converged";
        } catch (CancellationException stopped) {
            status = stopped.getMessage();
        }
        if (!Double.isNaN(ledger.segmentCpu)) ledger.pause();
        Map<String,Object> done = new LinkedHashMap<>();
        done.put("status", status); done.put("iteration", solver.completedIterations);
        done.put("charged_cpu", ledger.chargedCpu); done.put("charged_wall", ledger.chargedWall);
        emit("done", done);
        System.exit(0);
    }

    static double dot(double[] vector, double[] belief) {
        double total = 0;
        for (int s=0; s<belief.length; s++) total += vector[s] * belief[s];
        return total;
    }
}
