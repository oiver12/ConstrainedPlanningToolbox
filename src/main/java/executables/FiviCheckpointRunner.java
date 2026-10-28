// SPDX-License-Identifier: GPL-3.0-or-later
package executables;

import algorithms.pomdp.cgcp.FiniteVI;
import com.google.gson.*;
import model.*;
import java.nio.file.*;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.zip.*;
import java.io.*;
import java.lang.management.ManagementFactory;

public class FiviCheckpointRunner {
    static final Gson gson = new Gson();
    static final BufferedReader controls = new BufferedReader(new InputStreamReader(System.in));
    static class Snapshot implements Serializable {
        private static final long serialVersionUID = 1L;
        FiniteVI solver;
        double charged, best = -Double.MAX_VALUE;
        double wall;
        double checkpointCpu;
        int chunks;
        String identity;
    }
    static void emit(String kind, Object value) {
        Map<String,Object> message = new LinkedHashMap<>();
        message.put("kind", kind); message.put("value", value);
        System.out.println(gson.toJson(message)); System.out.flush();
    }
    static class FastGzip extends GZIPOutputStream {
        FastGzip(OutputStream output) throws IOException {
            super(output);
            def.setLevel(Deflater.BEST_SPEED);
        }
    }
    static void save(Path path, Snapshot snapshot) throws IOException {
        Path temporary = path.resolveSibling("checkpoint.tmp");
        try (FileOutputStream file = new FileOutputStream(temporary.toFile()); GZIPOutputStream zip = new FastGzip(file); ObjectOutputStream stream = new ObjectOutputStream(zip)) {
            stream.writeObject(snapshot); stream.flush(); zip.finish(); file.getFD().sync();
        }
        Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        try (FileChannel directory = FileChannel.open(path.getParent(), StandardOpenOption.READ)) { directory.force(true); }
    }
    static final com.sun.management.OperatingSystemMXBean cpuBean = (com.sun.management.OperatingSystemMXBean)ManagementFactory.getOperatingSystemMXBean();
    static double cpu() { return cpuBean.getProcessCpuTime() * 1e-9; }
    static volatile boolean expired;
    public static void main(String[] args) throws Exception {
        final long startNanos = System.nanoTime();
        Path checkpoint = Paths.get(args[1]), request = Paths.get(args[2]);
        String identity = args[3];
        Snapshot state;
        if (Files.exists(checkpoint)) {
            try (ObjectInputStream stream = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(checkpoint)))) { state = (Snapshot) stream.readObject(); }
            if (!state.identity.equals(identity)) throw new IOException("Checkpoint identity mismatch");
        } else {
            state = new Snapshot(); state.identity = identity;
            FiviRunner.Input input = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[0]))), FiviRunner.Input.class);
            int states = input.initial_belief.length, actions = input.observation.length;
            POMDP model = new POMDP(states, actions, input.observation[0][0].length, input.observation, new BeliefPoint(input.initial_belief), input.horizon);
            model.setRewardFunction(input.reward);
            int[][][] destinations = input.transition_destinations; double[][][] probabilities = input.transition_probabilities;
            if (destinations == null) {
            destinations = new int[states][actions][]; probabilities = new double[states][actions][];
            for (int s=0; s<states; s++) for (int a=0; a<actions; a++) {
                double[] row = input.transition[a][s]; int count = 0;
                for (double p : row) if (p != 0.0) count++;
                destinations[s][a] = new int[count]; probabilities[s][a] = new double[count];
                for (int next=0, i=0; next<states; next++) if (row[next] != 0.0) { destinations[s][a][i] = next; probabilities[s][a][i++] = row[next]; }
            }
            }
            model.setTransitionFunction(destinations, probabilities);
            long successors = 0;
            for (int s=0; s<states; s++) for (int a=0; a<actions; a++) successors += destinations[s][a].length;
            emit("loaded", Map.of("states", states, "actions", actions, "successors", successors, "transition_storage", "sparse successor lists"));
            state.solver = new FiniteVI(new Random(Long.parseLong(args[4])));
            // Initialize through solve below, retaining the exact sparse model in the snapshot.
            initialModel = model;
        }
        final Snapshot snapshot = state;
        double restoredCpu = state.charged;
        emit("restore", Map.of("cpu", state.charged, "wall", state.wall, "chunks", state.chunks, "iteration", state.solver.completedIterations));
        JsonObject start = JsonParser.parseString(controls.readLine()).getAsJsonObject();
        double prefix = start.get("prefix_cpu").getAsDouble();
        double prefixWall = start.has("prefix_wall") ? start.get("prefix_wall").getAsDouble() : 0.0;
        double restoredWall = state.wall;
        boolean wallBudget = args.length > 8 && args[8].equals("wall");
        double budget = Double.parseDouble(args[5]);
        int maxIterations = Integer.parseInt(args[6]);
        state.solver.setTerminateTime(Double.POSITIVE_INFINITY);
        state.solver.setParallelism(Integer.getInteger("fivi.parallelism", 1));
        if (maxIterations > 0) { state.solver.setMaxIterations(maxIterations); state.solver.fixedIterations = true; }
        final double[] last = {state.charged};
        // CPU-clock reads are system calls. Never perform one per vector/observation
        // inside the numerical loops: a daemon samples the limit instead. The parent
        // independently enforces the cutoff and validates controller eligibility.
        Thread budgetMonitor = new Thread(() -> {
            while (!expired) {
                double used = wallBudget ? restoredWall + prefixWall + (System.nanoTime()-startNanos)*1e-9 : restoredCpu + prefix + cpu();
                if (used >= budget) { expired = true; return; }
                try { Thread.sleep(100); } catch (InterruptedException stop) { return; }
            }
        }, "fivi-budget-monitor");
        budgetMonitor.setDaemon(true); budgetMonitor.start();
        state.solver.deadlineCheck = () -> { if (expired) throw new CancellationException("budget"); };
        state.solver.progress = (iteration, lower, upper, graph) -> {
            Map<String,Object> row = new LinkedHashMap<>();
            row.put("iteration", iteration); row.put("lower_bound", lower); row.put("upper_bound", upper);
            row.put("value", graph.getExpectedReward()); row.put("java_cpu", cpu());
            row.put("phase_cpu", snapshot.solver.getPhaseCpu());
            if (graph.getExpectedReward() > snapshot.best) {
                snapshot.best = graph.getExpectedReward();
                row.put("graph", Map.of("actions", graph.getLayerNodeActions(), "transitions", graph.getLayerTransitions(), "start_node", graph.getStartNode(), "horizon", graph.getLayerTransitions().size()-1));
            }
            emit("bound", row);
        };
        state.solver.boundary = () -> {
            boolean stopping = Files.exists(request);
            double charged = restoredCpu + prefix + cpu();
            if (stopping || snapshot.solver.finished || charged-last[0] >= 60 || (maxIterations > 0 && snapshot.solver.completedIterations == 1)) {
                try {
                    emit("checkpoint", Map.of("java_cpu", cpu(), "iteration", snapshot.solver.completedIterations));
                    JsonObject control = JsonParser.parseString(controls.readLine()).getAsJsonObject();
                    snapshot.charged = control.get("cpu").getAsDouble(); snapshot.chunks = control.get("chunks").getAsInt();
                    snapshot.wall = control.has("wall") ? control.get("wall").getAsDouble() : 0.0;
                    double saveStart = cpu();
                    save(checkpoint, snapshot); snapshot.checkpointCpu += cpu()-saveStart; last[0] = snapshot.charged;
                    emit("saved", Map.of("cpu", snapshot.charged, "wall", snapshot.wall, "chunks", snapshot.chunks));
                    if (stopping) throw new CancellationException("interruption");
                } catch (IOException error) { throw new UncheckedIOException(error); }
            }
        };
        try {
            state.solver.solve(initialModel, Integer.parseInt(args[7]));
            emit("done", Map.of("status", "converged", "phase_cpu", snapshot.solver.getPhaseCpu(), "checkpoint_cpu", snapshot.checkpointCpu));
        } catch (CancellationException stopped) { emit("done", Map.of("status", stopped.getMessage(), "phase_cpu", snapshot.solver.getPhaseCpu(), "checkpoint_cpu", snapshot.checkpointCpu)); }
    }
    static POMDP initialModel;
}
