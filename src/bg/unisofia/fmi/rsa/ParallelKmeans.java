package bg.unisofia.fmi.rsa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ParallelKmeans {

    private static final int MAX_ITERATIONS = 10;
    private static CountDownLatch countDownLatch;
    private final int n;
    private final int k;
    public int numThreads = 1;
    List<Node> observations = new ArrayList<>();
    float[][] clusters;

    public ParallelKmeans(int n, int k) {
        this.n = n;
        this.k = k;
        clusters = new float[k][n];
    }

    public void assignStep(ExecutorService executorService, AssignWorker[] assignWorkers) throws InterruptedException {
        countDownLatch = new CountDownLatch(numThreads);
        Arrays.stream(assignWorkers).forEach(executorService::execute);
        countDownLatch.await();
    }

    public void updateStep(ExecutorService executorService, UpdateWorker[] updateWorkers) throws InterruptedException {
        countDownLatch = new CountDownLatch(numThreads);
        Arrays.stream(updateWorkers).forEach(executorService::execute);
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
            VectorMath.divide(clusters[j], counts[j]);
        }
    }

    private void initializeClusters() {
        Random rng = new Random();
        //initialize the cluster at random
        for (int i = 0; i < clusters.length; i++) {
            float[] vec = observations.get(rng.nextInt(observations.size())).vec;
            clusters[i] = Arrays.copyOf(vec, vec.length);
        }
    }

    void cluster() throws InterruptedException {
        initializeClusters();

        final int maxNumberOfThreads = Runtime.getRuntime().availableProcessors() * 2;
        ExecutorService executorService = Executors.newFixedThreadPool(maxNumberOfThreads);

        AssignWorker[] assignWorkers = new AssignWorker[numThreads];
        UpdateWorker[] updateWorkers = new UpdateWorker[numThreads];
        final int chunkSz = observations.size() / numThreads;

        for (int i = 0; i < numThreads; i++) {
            int leftIdx = i * chunkSz;
            int rightIdx = (i + 1) * chunkSz;
            //handle the corner case for the last chunk
            if (i == numThreads - 1) {
                rightIdx = observations.size();
            }
            List<Node> chunk = observations.subList(leftIdx, rightIdx);
            assignWorkers[i] = new AssignWorker(chunk);
            updateWorkers[i] = new UpdateWorker(chunk);
        }
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            assignStep(executorService, assignWorkers);
            updateStep(executorService, updateWorkers);
        }
        executorService.shutdown();
    }

    public static class Node {
        float[] vec;
        int cluster;
    }

    class AssignWorker implements Runnable {
        List<Node> chunk;

        public AssignWorker(List<Node> chunk) {
            this.chunk = chunk;
        }

        @Override
        public void run() {
            for (Node ob : chunk) {
                float minDist = Float.POSITIVE_INFINITY;
                int idx = 0;
                for (int i = 0; i < clusters.length; i++) {
                    float dist = VectorMath.dist(ob.vec, clusters[i]);
                    if (minDist > dist) {
                        minDist = dist;
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
        float[][] clusters;
        List<Node> chunk;

        public UpdateWorker(List<Node> chunk) {
            this.chunk = chunk;
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
            for (Node ob : chunk) {
                VectorMath.add(this.clusters[ob.cluster], ob.vec);
                this.counts[ob.cluster]++;
            }
            countDownLatch.countDown();
        }
    }

}
