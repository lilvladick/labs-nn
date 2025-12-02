namespace LabsAlgorithms.Gradients;

public class ConjugateGradient // он нелинейный
{
    public static double[] Minimize( Func<double[], double> f, Func<double[], double[]> grad, double[] x0, int maxIters = 1000, double tol = 1e-6)
    {
        var n = x0.Length;
        var x = (double[])x0.Clone();
        var g = grad(x);
        var d = Scale(g, -1.0);
        var gnorm = Norm(g);

        if (gnorm < tol) return x;

        for (var k = 0; k < maxIters; k++)
        {
            var alpha = BacktrackingLineSearch(f, x, d, g);

            AddScaledInPlace(x, d, alpha);

            var gNew = grad(x);
            var gnewNorm = Norm(gNew);
            if (gnewNorm < tol) return x;

            var y = Subtract(gNew, g);
            var betaNum = Dot(gNew, y);
            var betaDen = Math.Max(1e-30, Dot(g, g));
            var beta = Math.Max(0.0, betaNum / betaDen);

            for (var i = 0; i < n; i++) d[i] = -gNew[i] + beta * d[i];

            g = gNew;
        }

        return x;
    }

    private static double BacktrackingLineSearch( Func<double[], double> f, double[] x, double[] d, double[] g, double alpha0 = 1.0, double rho = 0.5, double c = 1e-4)
    {
        var alpha = alpha0;
        var f0 = f(x);
        var gd = Dot(g, d);

        var xTrial = new double[x.Length];

        while (true)
        {
            for (var i = 0; i < x.Length; i++) xTrial[i] = x[i] + alpha * d[i];
            var fTrial = f(xTrial);

            if (fTrial <= f0 + c * alpha * gd)
                return alpha;

            alpha *= rho;
            if (alpha < 1e-16) return alpha;
        }
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        for (var i = a.Length - 1; i >= 0; i--) s += a[i] * b[i];
        return s;
    }

    private static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    private static double[] Scale(double[] a, double s)
    {
        var r = new double[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i] * s;
        return r;
    }

    private static void AddScaledInPlace(double[] a, double[] b, double scale)
    {
        for (var i = 0; i < a.Length; i++) a[i] += scale * b[i];
    }

    private static double[] Subtract(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i] - b[i];
        return r;
    }
}