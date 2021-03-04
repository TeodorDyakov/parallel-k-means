public class VectorMath {

    public static float dist(float[] vec1, float[] vec2) {
        float res = 0;
        for (int i = 0; i < vec1.length; i++) {
            float diff = vec1[i] - vec2[i];
            res += diff * diff;
        }
        return res;
    }

    public static void add(float[] vec1, float[] vec2) {
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] += vec2[i];
        }
    }

    public static void divide(float[] vec, float n) {
        if (n == 0) {
            return;
        }
        for (int i = 0; i < vec.length; i++) {
            vec[i] /= n;
        }
    }

    public static void add(int[] counts, int[] counts1) {
        for (int i = 0; i < counts.length; i++) {
            counts[i] += counts1[i];
        }
    }
}
