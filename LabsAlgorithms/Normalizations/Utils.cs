namespace LabsAlgorithms.Normalizations;

public class Utils
{
    public static double[] NormalizeUserInputMinMax(double[] x, double[] min, double[] max)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            var range = max[i] - min[i];
            r[i] = range == 0 ? 0 : (x[i] - min[i]) / range;
        }
        return r;
    }

    public static double[] NormalizeUserInputZScore(double[] x, double[] mean, double[] std)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
            r[i] = (x[i] - mean[i]) / std[i];
        return r;
    }

    public static double[] NormalizeUserInputLog(double[] x)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            if (x[i] < 0)
                throw new ArgumentException("Для лог-нормализации значения должны быть ≥ 0");

            r[i] = Math.Log(1 + x[i]);
        }
        return r;
    }

    public static double[] NormalizeUserInputMaxAbs(double[] x, double[] maxAbs)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
            r[i] = x[i] / maxAbs[i];
        return r;
    }

}