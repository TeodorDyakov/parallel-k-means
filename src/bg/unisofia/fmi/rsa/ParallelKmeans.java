package bg.unisofia.fmi.rsa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ParallelKmeans {

    private static CountDownLatch countDownLatch;
    private final int n;
    private final int k;
    private static final int MAX_ITERATIONS = 50;
    public int numThreads = 1;
    List<Node> observations = new ArrayList<>();
    float[][] clusters;

    public ParallelKmeans(int n, int k) {
        this.n = n;
        this.k = k;
        clusters = new float[k][n];
    }

    public void assignStep(ExecutorService executorService) throws InterruptedException {
        Runnable[] assignWorkers = new AssignWorker[numThreads];
        final int chunk = observations.size() / assignWorkers.length;
        countDownLatch = new CountDownLatch(numThreads);
        for (int j = 0; j < assignWorkers.length; j++) {
            assignWorkers[j] = new AssignWorker(j * chunk, (j + 1) * chunk);
            executorService.execute(assignWorkers[j]);
        }
        countDownLatch.await();

    }

    public void updateStep(ExecutorService executorService) throws InterruptedException {

        countDownLatch = new CountDownLatch(numThreads);

        UpdateWorker[] updateWorkers = new UpdateWorker[numThreads];
        final int chunk = observations.size() / updateWorkers.length;
        for (int j = 0; j < updateWorkers.length; j++) {
            updateWorkers[j] = new UpdateWorker(j * chunk, (j + 1) * chunk);
            executorService.execute(updateWorkers[j]);
        }
        countDownLatch.await();
        clusters = new float[k][n];
        int[] counts = new int[k];

        for (UpdateWorker u : updateWorkers) {
            VectorMath.add(counts, u.getCounts());
            for (int j = 0; j < k; j++) {
                VectorMath.add(clusters[j], u.getClusters()[j]);
            }
        }

        for (int j = 0; j < clusters.length; j++) {
            if (counts[j] != 0) {
                VectorMath.divide(clusters[j], counts[j]);
            }
        }
    }

    void cluster() throws InterruptedException {
        Random rng = new Random();
        //initialize the cluster at random
        for(int i = 0; i < clusters.length; i++){
            float[] vec = observations.get(rng.nextInt(observations.size())).vec;
            clusters[i] = Arrays.copyOf(vec, vec.length);
        }

        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            assignStep(executorService);
            updateStep(executorService);
        }
        executorService.shutdown();
    }

    public static class Node {
        float[] vec;
        int cluster;
    }

    class AssignWorker implements Runnable {
        int l, r;

        public AssignWorker(int l, int r) {
            this.l = l;
            this.r = r;
        }

        @Override
        public void run() {
            List<Node> chunk = observations.subList(l, r);
            for (Node ob : chunk) {
                float minDist = Float.POSITIVE_INFINITY;
                int idx = 0;
                for (int i = 0; i < clusters.length; i++) {
                    if (minDist > VectorMath.dist(ob.vec, clusters[i])) {
                        minDist = VectorMath.dist(ob.vec, clusters[i]);
                        idx = i;
                    }
                }
                ob.cluster = idx;
            }
            countDownLatch.countDown();
        }
    }

    class UpdateWorker implements Runnable {
        int[] counts;
        int l, r;
        float[][] clusters;

        UpdateWorker(int l, int r) {
            this.l = l;
            this.r = r;
        }

        int[] getCounts() {
            return counts;
        }

        public float[][] getClusters() {
            return clusters;
        }

        @Override
        public void run() {
            this.counts = new int[k];
            this.clusters = new float[k][n];
            for (Node ob : observations.subList(l, r)) {
                VectorMath.add(this.clusters[ob.cluster], ob.vec);
                this.counts[ob.cluster]++;
            }
            countDownLatch.countDown();
        }
    }

}