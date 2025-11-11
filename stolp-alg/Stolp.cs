using System;

public static class Stolp
{
    public static List<(double[] Features, string Label)> Reduce(List<(double[] Features, string Label)> data, double minMargin = 0.0)
    {
        var reduced = new List<(double[] Features, string Label)>();
        var classes = data.Select(d => d.Label).Distinct().ToList();

        foreach (var cls in classes)
        {
            var sameClass = data.Where(d => d.Label == cls).ToList();

            double bestMargin = double.NegativeInfinity;
            (double[] Features, string Label) best = sameClass.First();

            foreach (var obj in sameClass)
            {
                double margin = CalculateMargin(obj, data);
                if (margin > bestMargin)
                {
                    bestMargin = margin;
                    best = obj;
                }
            }

            reduced.Add(best);
        }

        bool changed;
        do
        {
            changed = false;
            var knn = new KnnClassifier(k: 1);
            foreach (var (f, l) in reduced)
                knn.Train(f, l);

            foreach (var obj in data)
            {
                string predicted = knn.Classify(obj.Features);
                if (predicted != obj.Label)
                {
                    reduced.Add(obj);
                    changed = true;
                }
            }

        } while (changed);

        reduced = [.. reduced.Where(obj => CalculateMargin(obj, reduced) >= minMargin)];

        return reduced;
    }

    private static double CalculateMargin((double[] Features, string Label) obj, List<(double[] Features, string Label)> data)
    {
        var same = data.Where(d => d.Label == obj.Label && !ReferenceEquals(d.Features, obj.Features))
                       .Select(d => Distance(obj.Features, d.Features))
                       .DefaultIfEmpty(double.MaxValue).Min();

        var diff = data.Where(d => d.Label != obj.Label)
                       .Select(d => Distance(obj.Features, d.Features))
                       .DefaultIfEmpty(double.MaxValue).Min();

        return diff - same;
    }

    private static double Distance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}