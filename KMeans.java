package com.company;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

public class KMeans {

    public static class Node {
        float[] vec;
        int cluster;
    }

    int n = 3;
    int k = 32;

    float[][] clusters = new float[k][n];
    {
        for (float[] cluster : clusters) {
            for (int i = 0; i < cluster.length; i++) {
                cluster[i] = (float) Math.random();
            }
        }
    }
    List<Node> observations = new ArrayList<>();

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

    public void update() {
        int[] counts = new int[k];
        for (var c : clusters) {
            Arrays.fill(c, 0);
        }
        for (Node ob : observations) {
            add(clusters[ob.cluster], ob.vec);
            counts[ob.cluster]++;
        }

        for (int i = 0; i < clusters.length; i++) {
            if (counts[i] != 0) {
                divide(clusters[i], counts[i]);
            }
        }
    }

     class AssignWorker extends Thread{
        int l, r;

         public AssignWorker(int l, int r) {
             this.l  = l;
             this.r = r;
         }

         @Override
         public void run(){
            List<Node> chunk = observations.subList(l, r);
            for(Node ob : chunk) {
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

    class UpdateWorker extends Thread{
        int[] counts;
        int l, r;

        UpdateWorker(int l, int r){
            this.l = l;
            this.r = r;
        }

        int[] getCounts(){
            return counts;
        }

        @Override
        public void run() {
            counts = new int[k];
            for (Node ob : observations) {
                add(clusters[ob.cluster], ob.vec);
                counts[ob.cluster]++;
            }
        }
    }

    void cluster() throws InterruptedException {

        for (int i = 0; i < 100; i++) {
            Thread[] threads = new AssignWorker[4];
            int chunk = observations.size()/threads.length;
            for(int j = 0; j < threads.length; j++){
                threads[j] = new AssignWorker(j * chunk, (j + 1)*chunk);
                threads[j].start();
            }
            for(Thread t : threads){
                t.join();
            }
            for (var c : clusters) {
                Arrays.fill(c, 0);
            }
            Thread[] updateWorkers = new UpdateWorker[4];
            int chunk = observations.size()/threads.length;
            for(int j = 0; j < updateWorkers.length; j++){
                threads[j] = new UpdateWorker(j * chunk, (j + 1)*chunk);
                threads[j].start();
            }
            for(Thread t : updateWorkers){
                t.join();
            }
            int[] counts = new int[k];

            for (int i = 0; i < clusters.length; i++) {
                if (counts[i] != 0) {
                    divide(clusters[i], counts[i]);
                }
            }
            update();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        KMeans k = new KMeans();
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
//        for(float[] vec : k.clusters){
//            System.out.println(Arrays.toString(vec));
//        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                Node n = k.observations.get(idx);
                float[] vec = k.clusters[n.cluster];
                Color c = new Color(vec[0], vec[1], vec[2]);
                image.setRGB(j, i, c.getRGB());
            }
        }

        File ouptut = new File("result.jpg");
        ImageIO.write(image, "jpg", ouptut);

    }

}