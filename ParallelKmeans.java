import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ParallelKmeans {

    private static final int MAX_ITERATIONS = 100;
    private static CountDownLatch countDownLatch;
    private final int n;
    private final int k;
    public int numThreads = 1;
    List<Observation> observations = new ArrayList<>();
    float[][] clusterCenters;

    public ParallelKmeans(int n, int k) {
        this.n = n;
        this.k = k;
        clusterCenters = new float[k][n];
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

        clusterCenters = new float[k][n];
        int[] counts = new int[k];

        for (UpdateWorker u : updateWorkers) {
            VectorMath.add(counts, u.getCounts());
            for (int j = 0; j < k; j++) {
                VectorMath.add(clusterCenters[j], u.getClusterCenters()[j]);
            }
        }

        for (int j = 0; j < clusterCenters.length; j++) {
            VectorMath.divide(clusterCenters[j], counts[j]);
        }
    }

    private void initializeClusters() {
        Random rng = new Random();
        //initialize the cluster at random
        for (int i = 0; i < clusterCenters.length; i++) {
            float[] vec = observations.get(rng.nextInt(observations.size())).vec;
            clusterCenters[i] = Arrays.copyOf(vec, vec.length);
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
            List<Observation> chunk = observations.subList(leftIdx, rightIdx);
            assignWorkers[i] = new AssignWorker(chunk);
            updateWorkers[i] = new UpdateWorker(chunk);
        }
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            assignStep(executorService, assignWorkers);
            updateStep(executorService, updateWorkers);
        }
        executorService.shutdown();
    }

    public static class Observation {
        float[] vec;
        volatile int cluster;
    }

    class AssignWorker implements Runnable {
        private final List<Observation> chunk;

        public AssignWorker(List<Observation> chunk) {
            this.chunk = chunk;
        }

        @Override
        public void run() {
            for (Observation ob : chunk) {
                float minDist = Float.POSITIVE_INFINITY;
                int idx = 0;
                for (int i = 0; i < clusterCenters.length; i++) {
                    float dist = VectorMath.dist(ob.vec, clusterCenters[i]);
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
        private volatile int[] counts;
        private volatile float[][] clusterCenters;
        private final List<Observation> chunk;

        UpdateWorker(List<Observation> chunk) {
            this.chunk = chunk;
        }

        int[] getCounts() {
            return counts;
        }

        float[][] getClusterCenters() {
            return clusterCenters;
        }

        @Override
        public void run() {
            this.counts = new int[k];
            this.clusterCenters = new float[k][n];
            for (Observation ob : chunk) {
                VectorMath.add(this.clusterCenters[ob.cluster], ob.vec);
                this.counts[ob.cluster]++;
            }
            countDownLatch.countDown();
        }
    }

}
