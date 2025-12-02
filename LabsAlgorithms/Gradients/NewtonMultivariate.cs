namespace LabsAlgorithms.Gradients;

public class NewtonMultivariate
{
    public static double[] Optimize( Func<double[], double> f, Func<double[], double[]> grad, Func<double[], double[,]> hess, double[] x0, double tolerance = 1e-6, int maxIter = 100)
    {
        var x = (double[])x0.Clone();
        var n = x.Length;

        for (var iter = 0; iter < maxIter; iter++)
        {
            var g = grad(x);
            var norm = Norm(g);
            if (norm < tolerance)
                return x;

            var h = hess(x);

            var d = SolveLinearSystem(h, Negate(g));

            var alpha = BacktrackingLineSearch(f, x, g, d);

            for (var i = 0; i < n; i++)
                x[i] += alpha * d[i];
        }

        return x;
    }

    private static double[] Negate(double[] g)
    {
        var r = new double[g.Length];
        for (var i = 0; i < g.Length; i++)
            r[i] = -g[i];
        return r;
    }

    private static double BacktrackingLineSearch(Func<double[], double> f, double[] x, double[] g, double[] d, double alpha0 = 1.0, double rho = 0.5, double c = 1e-4)
    {
        var alpha = alpha0;
        var f0 = f(x);
        var gd = Dot(g, d);
        var xTrial = new double[x.Length];

        while (true)
        {
            for (var i = 0; i < x.Length; i++)
                xTrial[i] = x[i] + alpha * d[i];

            if (f(xTrial) <= f0 + c * alpha * gd)
                return alpha;

            alpha *= rho;
            if (alpha < 1e-16)
                return alpha;
        }
    }

    private static double[] SolveLinearSystem(double[,] a, double[] b)
    {
        var n = b.Length;
        var m = new double[n, n + 1];

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
                m[i, j] = a[i, j];
            m[i, n] = b[i];
        }

        for (var k = 0; k < n; k++)
        {
            var maxRow = k;
            for (var i = k + 1; i < n; i++)
                if (Math.Abs(m[i, k]) > Math.Abs(m[maxRow, k]))
                    maxRow = i;

            if (Math.Abs(m[maxRow, k]) < 1e-15)
                throw new Exception("Гессиан вырожден (необратим)");

            for (var j = k; j <= n; j++)
                (m[k, j], m[maxRow, j]) = (m[maxRow, j], m[k, j]);

            var pivot = m[k, k];
            for (var j = k; j <= n; j++)
                m[k, j] /= pivot;

            for (var i = k + 1; i < n; i++)
            {
                var factor = m[i, k];
                for (var j = k; j <= n; j++)
                    m[i, j] -= factor * m[k, j];
            }
        }

        var x = new double[n];
        for (var i = n - 1; i >= 0; i--)
        {
            x[i] = m[i, n];
            for (var j = i + 1; j < n; j++)
                x[i] -= m[i, j] * x[j];
        }

        return x;
    }

    private static double Dot(double[] a, double[] b)
    {
        return a.Select((t, i) => t * b[i]).Sum();
    }

    private static double Norm(double[] v)
    {
        var s = v.Sum(t => t * t);
        return Math.Sqrt(s);
    }
}