namespace LabsAlgorithm.CoordinateReduction;

public class CoordinateReductionClass
{
    private static double Function(double x, double y) => Math.Pow(x - 5, 2) + Math.Pow(y - 2, 2);

    public static (double x, double y) CoordinateReduction(double x0, double y0, double searchRadius = 5.0, double eps = 1e-6, int maxIter = 100)
    {
        var x = x0;
        var y = y0;
        var prevF = Function(x, y);

        for (var iter = 0; iter < maxIter; iter++)
        {
            var changed = false;

            var y1 = y;
            var newX = GoldenSection(t => Function(t, y1), x - searchRadius, x + searchRadius, eps / 10);
            if (Math.Abs(newX-x) > eps)
            {
                x = newX;
                changed = true;
            }

            var x1 = x;
            var newY = GoldenSection(t => Function(x1, t), y - searchRadius, y + searchRadius, eps / 10);
            if (Math.Abs(newY - y) > eps)
            {
                y = newY;
                changed = true;
            }

            var currentF = Function(x, y);

            if (!changed || Math.Abs(currentF - prevF) < eps) break;
        }

        return (x,y);
    }

    private static double GoldenSection(Func<double, double> f, double a, double b, double e = 1e-8)
    {
        var phi = (1 + Math.Sqrt(5)) / 2;
        var resphi = 2 - phi;

        var x1 = a + resphi * (b - a);
        var x2 = b - resphi * (b - a);
        var f1 = f(x1);
        var f2 = f(x2);

        while ((b - a) >= 2 * e)
        {
            if (f1 < f2)
            {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + resphi * (b - a);
                f1 = f(x1);
            }
            else
            {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = b - resphi * (b - a);
                f2 = f(x2);
            }
        }

        return (a + b) / 2;
    }
}