using System;

public class CoordinateReductionClass
{
    public static double Function(double x, double y) => Math.Pow(x - 5, 2) + Math.Pow(y - 2, 2);

    public static (double x, double y) CoordinateReduction(double x0, double y0, double searchRadius = 5.0, double eps = 1e-6, int maxIter = 100)
    {
        double x = x0, y = y0;
        double prevF = Function(x, y);

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;

            double newX = GoldenSection(t => Function(t, y), x - searchRadius, x + searchRadius, eps / 10);
            if (Math.Abs(newX-x) > eps)
            {
                x = newX;
                changed = true;
            }

            double newY = GoldenSection(t => Function(x, t), y - searchRadius, y + searchRadius, eps / 10);
            if (Math.Abs(newY - y) > eps)
            {
                y = newY;
                changed = true;
            }

            double currentF = Function(x, y);

            if (!changed || Math.Abs(currentF - prevF) < eps) break;
        }

        return (x,y);
    }

    private static double GoldenSection(Func<double, double> f, double a, double b, double e = 1e-8)
    {
        double phi = (1 + Math.Sqrt(5)) / 2;
        double resphi = 2 - phi;

        double x1 = a + resphi * (b - a);
        double x2 = b - resphi * (b - a);
        double f1 = f(x1);
        double f2 = f(x2);

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
class Program
{
    static void Main()
    {
        Console.WriteLine("Введите x0: ");
        double startX = Convert.ToDouble(Console.ReadLine());
        Console.WriteLine("Введите y0: ");
        double startY = Convert.ToDouble(Console.ReadLine());

        Console.WriteLine();
        Console.WriteLine($"Начальная точка: ({startX:F2}, {startY:F2})");
        Console.WriteLine($"f = {CoordinateReductionClass.Function(startX, startY):F6}\n");

        var (xMin, yMin) = CoordinateReductionClass.CoordinateReduction(startX, startY);

        Console.WriteLine();
        Console.WriteLine($"Минимум в: ({xMin:F8}, {yMin:F8})");
        Console.WriteLine($"f(x, y) = {CoordinateReductionClass.Function(xMin, yMin):F8}");
    }
}