package bg.unisofia.fmi.rsa;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Demo {
    public static void main(String[] args) throws IOException, InterruptedException {
        ParallelKmeans k = new ParallelKmeans(3, 32);
        BufferedImage image = ImageIO.read(new File("owl.jpg"));

        int width = image.getWidth();
        int height = image.getHeight();

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Color color = new Color(image.getRGB(j, i));
                ParallelKmeans.Node node = new ParallelKmeans.Node();
                node.vec = color.getColorComponents(node.vec);
                k.observations.add(node);
            }
        }
        final int maxThreads = Runtime.getRuntime().availableProcessors() * 2;
        System.out.println("Threads   time");
        for (int i = 1; i <= maxThreads; i++) {
            k.numThreads = i;
            long tic = System.currentTimeMillis();
            k.cluster();
            System.out.println(i + "         " + (System.currentTimeMillis() - tic));
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                ParallelKmeans.Node n = k.observations.get(idx);
                float[] vec = k.clusters[n.cluster];
                Color c = new Color(vec[0], vec[1], vec[2]);
                image.setRGB(j, i, c.getRGB());
            }
        }
        File output = new File("result.jpg");
        ImageIO.write(image, "jpg", output);
    }
}
