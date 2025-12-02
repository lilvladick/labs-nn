namespace LabsAlgorithms.Gradients;

public class GradientDescent
{
    public static double[] Optimize(Func<double[], double[]> grad, double[] x0, double learningRate = 0.01, double tolerance = 1e-6, int maxIter = 10000)
    {
        var x = (double[])x0.Clone();
        var n = x.Length;

        for (var iter = 0; iter < maxIter; iter++)
        {
            var g = grad(x);
            var norm = Norm(g);

            if (norm < tolerance)
                return x;

            for (var i = 0; i < n; i++)
                x[i] -= learningRate * g[i];
        }

        return x;
    }

    private static double Norm(double[] v)
    {
        var s = v.Sum(t => t * t);
        return Math.Sqrt(s);
    }
}