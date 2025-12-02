namespace LabsAlgorithm.MinimizationAlgorithms;

public class MinimizationAlgorithms
{
    private static double Function(double x)
    {
        return Math.Pow(x, 3) - x - 2;
    }
    // для ньютона производная
    private static double DFunction(double x)
    {
        return 3 * Math.Pow(x, 2) - 1;
    }
    // вторая производная (вообще у нас должен быть double x, но 2-я производная будет просто цифрой)
    private static double D2Function() => 2.0;
    
    //дихотомия
    public static double Dichotomy(double a, double b, double e = 1e-8)
    {
        if (DFunction(a) * DFunction(b) > 0) throw new ArgumentException("Производная не меняет знак на отрезке — нет гарантии минимума.");


        while (b - a >= 2 * e)
        {
            var c = (a + b) / 2;
            if (DFunction(a) * DFunction(c) < 0)
                b = c;
            else
                a = c;
        }

        return (a + b) / 2;
    }

    // золотое
    public static double GoldenSection(double a, double b, double e = 1e-8)
    {
        var phi = (1 + Math.Sqrt(5)) / 2;
        var resphi = 2 - phi;

        var x1 = a + resphi * (b - a);
        var x2 = b - resphi * (b - a);
        var f1 = Function(x1);
        var f2 = Function(x2);

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
        var x1 = a + (double)fibonachi[n - 2] / fibonachi[n] * (b - a);
        var x2 = a + (double)fibonachi[n - 1] / fibonachi[n] * (b - a);

        var f1 = Function(x1);
        var f2 = Function(x2);

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
        var x = x0;
        for (var iter = 0; iter < maxIter; iter++)
        {
            var f1 = DFunction(x);
            var f2 = D2Function(); // ну тут тогда никакого x в аргументах не будет

            if (Math.Abs(f2) < 1e-12) break;

            var next = x - f1 / f2;

            if (Math.Abs(Function(next)) < e || Math.Abs(next - x) < e) return next;

            x = next;
        }
        
        return x;
    }
}