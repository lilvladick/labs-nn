using System;
using System.Diagnostics;

class OptimizationAlgorithms
{
    public static double Function(double x)
    {
        return Math.Pow(x, 3) - x - 2;
    }
    // для ньютона производная
    static double DFunction(double x)
    {
        return 3 * Math.Pow(x, 2) - 1;
    }
    //дихотомия
    public static double Dichotomy(double a, double b, double e = 1e-8)
    {
        double c = (a + b) / 2;

        while ((b - a) >= 2 * e)
        {
            if (Function(a) * Function(c) < 0) b = c;
            else a = c;

            c = (a + b) / 2;
        }

        return c;
    }

    // золотое
    public static double GoldenSection(double a, double b, double e = 1e-8)
    {
        double phi = (1 + Math.Sqrt(5)) / 2;
        double resphi = 2 - phi;

        double x1 = a + resphi * (b - a);
        double x2 = b - resphi * (b - a);
        double f1 = Function(x1);
        double f2 = Function(x2);

        while ((b - a) >= 2 * e)
        {
            if (f1 < f2)
            {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + resphi * (b - a);
                f1 = Function(x1);
            }
            else
            {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = b - resphi * (b - a);
                f2 = Function(x2);
            }
        }

        return (a + b) / 2;
    }

    // фибоначи
    public static double Fibonacci(double a, double b, double e = 1e-8)
    {
        var fibonachi = new List<int> { 1, 1 };

        while ((b - a) / e > fibonachi[^1])
        {
            fibonachi.Add(fibonachi[^1] + fibonachi[^2]);
        }

        var n = fibonachi.Count - 1;
        double x1 = a + (double)fibonachi[n - 2] / fibonachi[n] * (b - a);
        double x2 = a + (double)fibonachi[n - 1] / fibonachi[n] * (b - a);

        double f1 = Function(x1);
        double f2 = Function(x2);

        for (var k = 1; k <= n - 2; k++)
        {
            if (f1 > f2)
            {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + (double)fibonachi[n - k - 1] / fibonachi[n - k] * (b - a);
                f2 = Function(x2);
            }
            else
            {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + (double)fibonachi[n - k - 2] / fibonachi[n - k] * (b - a);
                f1 = Function(x1);
            }
        }
        return (x1 + x2) / 2;
    }

    // ньютон
    public static double Newton(double x0, double e = 1e-8, int maxIter = 100)
    {
        double x = x0;
        for (int iter = 0; iter < maxIter; iter++)
        {
            double f = Function(x);
            double df = DFunction(x);

            if (Math.Abs(df) < 1e-12) break;

            double next = x - f / df;

            if (Math.Abs(Function(next)) < e || Math.Abs(next - x) < e) return next;

            x = next;
        }
        
        return x;
    }
}

class Program
{
    static void Main()
    {
        Console.WriteLine("Введите а: ");
        var a = Convert.ToDouble(Console.ReadLine());
        Console.WriteLine("Введите b: ");
        var b = Convert.ToDouble(Console.ReadLine());

        var dichotomyRes = OptimizationAlgorithms.Dichotomy(a, b);
        Console.WriteLine($"\nКорень уравнения на отрезке [{a}, {b}] равен {dichotomyRes:F8}");
        Console.WriteLine($"f(x) = {OptimizationAlgorithms.Function(dichotomyRes):F8}");
        Console.WriteLine();

        double goldRes = OptimizationAlgorithms.GoldenSection(a, b);

        Console.WriteLine($"\n(Gold) Минимум функции на отрезке [{a}, {b}] ≈ x = {goldRes:F6}");
        Console.WriteLine($"f(x) = {OptimizationAlgorithms.Function(goldRes):F8}");
        Console.WriteLine();
        
        double fibRes = OptimizationAlgorithms.Fibonacci(a, b);

        Console.WriteLine($"(Fib) Минимум функции на отрезке [{a}, {b}] x = {fibRes:F8}");
        Console.WriteLine($"f(x) = {OptimizationAlgorithms.Function(fibRes):F8}");
        Console.WriteLine();

        Console.WriteLine("Введите начальное приближение: ");
        double x0 = Convert.ToDouble(Console.ReadLine());

        double newtonRes = OptimizationAlgorithms.Newton(x0);
        Console.WriteLine($"\n(Newton) Корень x = {newtonRes:F8}");
        Console.WriteLine($"f(x) = {OptimizationAlgorithms.Function(newtonRes):F8}");
    }
}