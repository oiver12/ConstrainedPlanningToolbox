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
        int chunks;
        String identity;
    }
    static void emit(String kind, Object value) {
        Map<String,Object> message = new LinkedHashMap<>();
        message.put("kind", kind); message.put("value", value);
        System.out.println(gson.toJson(message)); System.out.flush();
    }
    static void save(Path path, Snapshot snapshot) throws IOException {
        Path temporary = path.resolveSibling("checkpoint.tmp");
        try (FileOutputStream file = new FileOutputStream(temporary.toFile()); GZIPOutputStream zip = new GZIPOutputStream(file); ObjectOutputStream stream = new ObjectOutputStream(zip)) {
            stream.writeObject(snapshot); stream.flush(); zip.finish(); file.getFD().sync();
        }
        Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        try (FileChannel directory = FileChannel.open(path.getParent(), StandardOpenOption.READ)) { directory.force(true); }
    }
    static double cpu() { return ((com.sun.management.OperatingSystemMXBean)ManagementFactory.getOperatingSystemMXBean()).getProcessCpuTime() * 1e-9; }
    public static void main(String[] args) throws Exception {
        Path checkpoint = Paths.get(args[1]), request = Paths.get(args[2]);
        String identity = args[3];
        Snapshot state;
        if (Files.exists(checkpoint)) {
            try (ObjectInputStream stream = new ObjectInputStream(new GZIPInputStream(Files.newInputStream(checkpoint)))) { state = (Snapshot) stream.readObject(); }
            if (!state.identity.equals(identity)) throw new IOException("Checkpoint identity mismatch");
        } else {
            state = new Snapshot(); state.identity = identity;
            FiviRunner.Input input = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[0]))), FiviRunner.Input.class);
            int states = input.initial_belief.length, actions = input.transition.length;
            POMDP model = new POMDP(states, actions, input.observation[0][0].length, input.observation, new BeliefPoint(input.initial_belief), input.horizon);
            model.setRewardFunction(input.reward);
            int[][][] destinations = new int[states][actions][]; double[][][] probabilities = new double[states][actions][];
            for (int s=0; s<states; s++) for (int a=0; a<actions; a++) {
                double[] row = input.transition[a][s]; int count = 0;
                for (double p : row) if (p != 0.0) count++;
                destinations[s][a] = new int[count]; probabilities[s][a] = new double[count];
                for (int next=0, i=0; next<states; next++) if (row[next] != 0.0) { destinations[s][a][i] = next; probabilities[s][a][i++] = row[next]; }
            }
            model.setTransitionFunction(destinations, probabilities);
            state.solver = new FiniteVI(new Random(Long.parseLong(args[4])));
            // Initialize through solve below, retaining the exact sparse model in the snapshot.
            initialModel = model;
        }
        final Snapshot snapshot = state;
        double restoredCpu = state.charged;
        emit("restore", Map.of("cpu", state.charged, "chunks", state.chunks, "iteration", state.solver.completedIterations));
        JsonObject start = JsonParser.parseString(controls.readLine()).getAsJsonObject();
        double prefix = start.get("prefix_cpu").getAsDouble();
        double budget = Double.parseDouble(args[5]);
        int maxIterations = Integer.parseInt(args[6]);
        state.solver.setTerminateTime(Double.POSITIVE_INFINITY);
        if (maxIterations > 0) { state.solver.setMaxIterations(maxIterations); state.solver.fixedIterations = true; }
        final double[] last = {state.charged};
        state.solver.deadlineCheck = () -> { if (restoredCpu + prefix + cpu() >= budget) throw new CancellationException("budget"); };
        state.solver.progress = (iteration, lower, upper, graph) -> {
            Map<String,Object> row = new LinkedHashMap<>();
            row.put("iteration", iteration); row.put("lower_bound", lower); row.put("upper_bound", upper);
            row.put("value", graph.getExpectedReward()); row.put("java_cpu", cpu());
            if (graph.getExpectedReward() > snapshot.best) {
                snapshot.best = graph.getExpectedReward();
                row.put("graph", Map.of("actions", graph.getLayerNodeActions(), "transitions", graph.getLayerTransitions(), "start_node", graph.getStartNode(), "horizon", graph.getLayerTransitions().length));
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
                    save(checkpoint, snapshot); last[0] = snapshot.charged;
                    emit("saved", Map.of("cpu", snapshot.charged, "chunks", snapshot.chunks));
                    if (stopping) throw new CancellationException("interruption");
                } catch (IOException error) { throw new UncheckedIOException(error); }
            }
        };
        try {
            state.solver.solve(initialModel, Integer.parseInt(args[7]));
            emit("done", Map.of("status", "converged"));
        } catch (CancellationException stopped) { emit("done", Map.of("status", stopped.getMessage())); }
    }
    static POMDP initialModel;
}
