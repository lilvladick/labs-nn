namespace LabsAlgorithms.Normalizations;

using System;
using System.Collections.Generic;
using System.Linq;

public class Normalizer
{
    public static (double[] Min, double[] Max) FitMinMax(List<(double[] Features, string Label)> data)
    {
        var dim = data[0].Features.Length;
        var minVal = new double[dim];
        var maxVal = new double[dim];

        for (var i = 0; i < dim; i++)
        {
            minVal[i] = data.Min(d => d.Features[i]);
            maxVal[i] = data.Max(d => d.Features[i]);
        }

        return (minVal, maxVal);
    }

    public static List<(double[] Features, string Label)> LinearNormalization(
        List<(double[] Features, string Label)> data,
        double[] minValues,
        double[] maxValues)
    {
        var dim = minValues.Length;
        var result = new List<(double[] Features, string Label)>(data.Count);

        foreach (var (features, label) in data)
        {
            var norm = new double[dim];
            for (var i = 0; i < dim; i++)
            {
                var range = maxValues[i] - minValues[i];
                norm[i] = range == 0 ? 0 : (features[i] - minValues[i]) / range;
            }
            result.Add((norm, label));
        }

        return result;
    }

    public static (double[] Mean, double[] Std) FitZScore(List<(double[] Features, string Label)> data)
    {
        var dim = data[0].Features.Length;
        var mean = new double[dim];
        var std = new double[dim];

        for (var j = 0; j < dim; j++)
        {
            mean[j] = data.Average(d => d.Features[j]);
            std[j] = Math.Sqrt(data.Average(d => Math.Pow(d.Features[j] - mean[j], 2)));
            if (std[j] == 0) std[j] = 1e-9;
        }

        return (mean, std);
    }

    public static List<(double[] Features, string Label)> ZScoreNormalization(
        List<(double[] Features, string Label)> data,
        double[] mean,
        double[] std)
    {
        var dim = mean.Length;
        var result = new List<(double[] Features, string Label)>(data.Count);

        foreach (var (features, label) in data)
        {
            var norm = new double[dim];
            for (var j = 0; j < dim; j++)
                norm[j] = (features[j] - mean[j]) / std[j];

            result.Add((norm, label));
        }

        return result;
    }

    public static List<(double[] Features, string Label)> LogNormalization(
        List<(double[] Features, string Label)> data)
    {
        var result = new List<(double[] Features, string Label)>(data.Count);
        var dim = data[0].Features.Length;

        foreach (var (features, label) in data)
        {
            var norm = new double[dim];
            for (var i = 0; i < dim; i++)
            {
                var v = features[i];
                if (v < 0) 
                    throw new ArgumentException("Логарифмическая нормализация требует значения >= 0");

                norm[i] = Math.Log(1 + v);
            }

            result.Add((norm, label));
        }

        return result;
    }

    public static double[] FitMaxAbs(List<(double[] Features, string Label)> data)
    {
        var dim = data[0].Features.Length;
        var maxAbs = new double[dim];

        for (var j = 0; j < dim; j++)
        {
            maxAbs[j] = data.Max(d => Math.Abs(d.Features[j]));
            if (maxAbs[j] == 0) maxAbs[j] = 1e-9;
        }

        return maxAbs;
    }

    public static List<(double[] Features, string Label)> MaxAbsNormalization(
        List<(double[] Features, string Label)> data,
        double[] maxAbs)
    {
        var dim = maxAbs.Length;
        var result = new List<(double[] Features, string Label)>(data.Count);

        foreach (var (f, label) in data)
        {
            var norm = new double[dim];
            for (var j = 0; j < dim; j++)
                norm[j] = f[j] / maxAbs[j];

            result.Add((norm, label));
        }

        return result;
    }
}
