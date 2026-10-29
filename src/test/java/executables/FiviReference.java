// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed-iteration reference driver: replays the solver phases through reflection and records the lower bound
// at the initial belief after each sweep.
package executables;
import algorithms.pomdp.cgcp.FiniteVI;
import com.google.gson.*;
import model.*;
import java.nio.file.*;
import java.util.*;
import java.lang.reflect.*;

public class FiviReference {
    static class Input { double[][][] transition, observation; double[][] reward; double[] initial_belief; int horizon; }
    static void set(Object object, String name, Object value) throws Exception {
        Field field = FiniteVI.class.getDeclaredField(name); field.setAccessible(true); field.set(object, value);
    }
    static Object call(Object object, String name, Class<?>[] types, Object... values) throws Exception {
        Method method = FiniteVI.class.getDeclaredMethod(name, types); method.setAccessible(true); return method.invoke(object, values);
    }
    @SuppressWarnings("unchecked")
    static ArrayList<ArrayList<AlphaVector>> vectors(FiniteVI solver) throws Exception {
        Field field = FiniteVI.class.getDeclaredField("vectorSets"); field.setAccessible(true);
        return (ArrayList<ArrayList<AlphaVector>>) field.get(solver);
    }
    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        Gson gson = new Gson();
        Input input = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[0])), "UTF-8"), Input.class);
        int states = input.initial_belief.length, actions = input.transition.length;
        POMDP model = new POMDP(states, actions, input.observation[0][0].length, input.observation, new BeliefPoint(input.initial_belief), input.horizon);
        model.setRewardFunction(input.reward);
        int[][][] destinations = new int[states][actions][states];
        double[][][] probabilities = new double[states][actions][states];
        for (int s=0; s<states; s++) for (int a=0; a<actions; a++) for (int next=0; next<states; next++) {
            destinations[s][a][next] = next; probabilities[s][a][next] = input.transition[a][s][next];
        }
        model.setTransitionFunction(destinations, probabilities);
        FiniteVI solver = new FiniteVI(new Random(Long.parseLong(args[2])));
        set(solver, "pomdp", model); set(solver, "T", input.horizon); set(solver, "lambda", 0.0);
        call(solver, "initBeliefSets", new Class<?>[]{});
        ArrayList<ArrayList<AlphaVector>> vectors = new ArrayList<>();
        for (int t=0; t<=input.horizon; t++) vectors.add(new ArrayList<>());
        for (int a=0; a<actions; a++) { AlphaVector vector = new AlphaVector(new double[states]); vector.setAction(a); vectors.get(input.horizon).add(vector); }
        set(solver, "vectorSets", vectors);
        List<AlphaVector> rewards = new ArrayList<>();
        for (int a=0; a<actions; a++) { double[] entries=new double[states]; for(int s=0;s<states;s++) entries[s]=input.reward[s][a]; AlphaVector vector=new AlphaVector(entries); vector.setAction(a); rewards.add(vector); }
        set(solver, "immediateRewards", rewards);
        Field field=FiniteVI.class.getDeclaredField("beliefLists"); field.setAccessible(true);
        ArrayList<ArrayList<BeliefPoint>> beliefs=(ArrayList<ArrayList<BeliefPoint>>)field.get(solver);
        for (BeliefPoint point: beliefs.get(input.horizon)) point.setUpperBound(0);
        ArrayList<Double> result=new ArrayList<>();
        int iterations=Integer.parseInt(args[1]);
        for(int i=0;i<iterations;i++) {
            for(int t=input.horizon-1;t>=0;t--) {
                call(solver,"backupStagePerseus",new Class<?>[]{int.class,int.class},t,i);
                call(solver,"upperBoundUpdateSkip",new Class<?>[]{int.class,ArrayList.class},t,beliefs.get(t));
            }
            result.add(AlphaVector.getValue(model.getInitialBelief().getBelief(), vectors(solver).get(0)));
            if(i+1<iterations) call(solver,"findNewBeliefs",new Class<?>[]{});
        }
        Files.write(Paths.get(args[3]),gson.toJson(result).getBytes("UTF-8"));
    }
}
