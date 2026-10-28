// SPDX-License-Identifier: GPL-3.0-or-later
// Integration runner; upstream solver and graph evaluator remain GPL-3.0-or-later.
package executables;

import algorithms.pomdp.cgcp.FiniteVI;
import com.google.gson.*;
import model.*;
import solutions.pomdp.POMDPPolicyGraph;
import java.nio.file.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.io.*;

public class FiviRunner {
    static final Gson gson = new Gson();
    static class Input {
        double[][][] transition, observation;
        int[][][] transition_destinations;
        double[][][] transition_probabilities;
        double[][] reward;
        double[] initial_belief;
        int horizon;
    }
    static void save(Path destination, Object value) throws IOException {
        Path temporary = destination.resolveSibling(destination.getFileName() + ".tmp");
        Files.write(temporary, gson.toJson(value).getBytes(StandardCharsets.UTF_8));
        try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) { channel.force(true); }
        Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }
    public static void main(String[] args) throws Exception {
        long start = System.nanoTime();
        Path output = Paths.get(args[1]);
        double budget = Double.parseDouble(args[2]);
        int iterations = Integer.parseInt(args[4]);
        Input input = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[0])), StandardCharsets.UTF_8), Input.class);
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
        FiniteVI solver = new FiniteVI(new Random(Long.parseLong(args[3])));
        solver.setTerminateTime(Double.POSITIVE_INFINITY);
        if (iterations > 0) { solver.setMaxIterations(iterations); solver.fixedIterations = true; }
        solver.deadlineCheck = () -> { if ((System.nanoTime()-start)*1e-9 >= budget) throw new CancellationException(); };
        final double[] best = {Double.NEGATIVE_INFINITY};
        try (FileOutputStream stream = new FileOutputStream(output.resolve("java-trace.jsonl").toFile())) {
            solver.progress = (iteration, lower, upper, graph) -> {
                solver.deadlineCheck.run();
                try {
                    Map<String,Object> policy = new LinkedHashMap<>();
                    policy.put("actions", graph.getLayerNodeActions());
                    policy.put("transitions", graph.getLayerTransitions());
                    policy.put("start_node", graph.getStartNode());
                    policy.put("horizon", input.horizon);
                    String filename = String.format(Locale.ROOT, "policy-%06d.json", iteration);
                    save(output.resolve(filename), policy);
                    solver.deadlineCheck.run();
                    best[0] = Math.max(best[0], graph.getExpectedReward());
                    Map<String,Object> record = new LinkedHashMap<>();
                    record.put("iteration", iteration);
                    record.put("solver_seconds", (System.nanoTime()-start)*1e-9);
                    record.put("lower_bound", lower);
                    record.put("upper_bound", upper);
                    record.put("policy_value", graph.getExpectedReward());
                    record.put("best_policy_value", best[0]);
                    record.put("policy", filename);
                    record.put("policy_size", Arrays.stream(graph.getLayerNodeCount()).sum());
                    stream.write((gson.toJson(record)+"\n").getBytes(StandardCharsets.UTF_8));
                    stream.flush(); stream.getFD().sync();
                } catch (IOException error) { throw new UncheckedIOException(error); }
            };
            try {
                solver.solve(model, input.horizon);
                save(output.resolve("java-status.json"), Collections.singletonMap("status", "completed"));
            } catch (CancellationException timeout) {
                save(output.resolve("java-status.json"), Collections.singletonMap("status", "timeout"));
            }
        }
    }
}
