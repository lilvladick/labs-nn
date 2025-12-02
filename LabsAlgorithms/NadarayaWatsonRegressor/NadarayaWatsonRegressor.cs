namespace LabsAlgorithms.NadarayaWatsonRegressor;

public class NadarayaWatsonRegressor
{
    private readonly List<(double[] X, double Y)> _data = [];
    private readonly double _bandwidth;

    public NadarayaWatsonRegressor(double bandwidth)
    {
        if (bandwidth <= 0)
            throw new ArgumentException("bandwidth должен быть > 0");

        _bandwidth = bandwidth;
    }

    public void AddSample(double[] x, double y)
    {
        _data.Add((x, y));
    }

    public double Predict(double[] x)
    {
        if (_data.Count == 0)
            throw new InvalidOperationException("Нет обучающих данных");

        double numerator = 0;
        double denominator = 0;

        foreach (var (xi, yi) in _data)
        {
            var d = EuclideanDistance(x, xi);
            var u = d / _bandwidth;

            var w = GaussianKernel(u);

            numerator += w * yi;
            denominator += w;
        }

        return numerator / denominator;
    }

    private static double GaussianKernel(double u)
    {
        return Math.Exp(-(u * u) / 2.0);
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        var sum = a.Select((t, i) => t - b[i]).Sum(diff => diff * diff);
        return Math.Sqrt(sum);
    }
}