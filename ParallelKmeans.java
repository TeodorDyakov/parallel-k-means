package com.company;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ParallelKmeans {

    static final int numThreads = 4;

    int n = 3;
    int k = 32;
    float[][] clusters = new float[k][n];
    List<Node> observations = new ArrayList<>();

    {
        for (float[] cluster : clusters) {
            for (int i = 0; i < cluster.length; i++) {
                cluster[i] = (float) Math.random();
            }
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        ParallelKmeans k = new ParallelKmeans();
        BufferedImage image = ImageIO.read(new File("owl.jpg"));

        int width = image.getWidth();
        int height = image.getHeight();

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {

                Color color = new Color(image.getRGB(j, i));
                Node node = new Node();
                node.vec = color.getColorComponents(node.vec);
                k.observations.add(node);

            }
        }
        long tic = System.currentTimeMillis();
        k.cluster();
        System.out.println(System.currentTimeMillis() - tic);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                Node n = k.observations.get(idx);
                float[] vec = k.clusters[n.cluster];
                Color c = new Color(vec[0], vec[1], vec[2]);
                image.setRGB(j, i, c.getRGB());
            }
        }

        File output = new File("result.jpg");
        ImageIO.write(image, "jpg", output);

    }

    public float dist(float[] vec1, float[] vec2) {
        float res = 0;
        for (int i = 0; i < vec1.length; i++) {
            float diff = vec1[i] - vec2[i];
            res += diff * diff;
        }
        return res;
    }

    public void add(float[] vec1, float[] vec2) {
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] += vec2[i];
        }
    }

    public void divide(float[] vec, float n) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] /= n;
        }
    }

    void cluster() throws InterruptedException {
        for (int i = 0; i < 100; i++) {
            Thread[] threads = new AssignWorker[numThreads];
            final int chunk = observations.size() / threads.length;

            for (int j = 0; j < threads.length; j++) {
                threads[j] = new AssignWorker(j * chunk, (j + 1) * chunk);
                threads[j].start();
            }
            for (Thread t : threads) {
                t.join();
            }

            UpdateWorker[] updateWorkers = new UpdateWorker[numThreads];
            for (int j = 0; j < updateWorkers.length; j++) {
                updateWorkers[j] = new UpdateWorker(j * chunk, (j + 1) * chunk);
                updateWorkers[j].start();
            }
            for (Thread t : updateWorkers) {
                t.join();
            }

            clusters = new float[k][n];
            int[] counts = new int[k];

            for (UpdateWorker u : updateWorkers) {
                add(counts, u.getCounts());
                for (int j = 0; j < k; j++) {
                    add(clusters[j], u.getClusters()[j]);
                }
            }

            for (int j = 0; j < clusters.length; j++) {
                if (counts[j] != 0) {
                    divide(clusters[j], counts[j]);
                }
            }
        }
    }

    private void add(int[] counts, int[] counts1) {
        for (int i = 0; i < counts.length; i++) {
            counts[i] += counts1[i];
        }
    }

    public static class Node {
        float[] vec;
        int cluster;
    }

    class AssignWorker extends Thread {
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
                    if (minDist > dist(ob.vec, clusters[i])) {
                        minDist = dist(ob.vec, clusters[i]);
                        idx = i;
                    }
                }
                ob.cluster = idx;
            }
        }
    }

    class UpdateWorker extends Thread {
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
                add(this.clusters[ob.cluster], ob.vec);
                this.counts[ob.cluster]++;
            }
        }
    }

}