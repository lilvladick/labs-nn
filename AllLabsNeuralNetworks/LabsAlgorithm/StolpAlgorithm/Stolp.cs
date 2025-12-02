namespace LabsAlgorithm.StolpAlgorithm;

public static class Stolp
{
    public static List<(double[] Features, string Label)> Reduce(List<(double[] Features, string Label)> data, double minMargin = 0.0)
    {
        var reduced = new List<(double[] Features, string Label)>();
        var classes = data.Select(d => d.Label).Distinct().ToList();

        foreach (var cls in classes)
        {
            var sameClass = data.Where(d => d.Label == cls).ToList();

            var bestMargin = double.NegativeInfinity;
            var best = sameClass.First();

            foreach (var obj in sameClass)
            {
                var margin = CalculateMargin(obj, data);
                if (!(margin > bestMargin)) continue;
                bestMargin = margin;
                best = obj;
            }

            reduced.Add(best);
        }

        bool changed;
        do
        {
            changed = false;
            var knn = new LabsAlgorithm.KnnClassifier.KnnClassifier(k: 1);
            foreach (var (f, l) in reduced)
                knn.Train(f, l);

            var toAdd = new List<(double[] Features, string Label)>();

            var reduced2 = reduced;
            foreach (var obj in from obj in data let predicted = knn.Classify(obj.Features) where predicted != obj.Label && !reduced2.Contains(obj) && !toAdd.Contains(obj) select obj)
            {
                toAdd.Add(obj);
                changed = true;
            }

            reduced.AddRange(toAdd);

        } while (changed);

        var reduced1 = reduced;
        reduced = [.. reduced.Where(obj => CalculateMargin(obj, reduced1) >= minMargin)];

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
        var sum = a.Select((t, i) => t - b[i]).Sum(diff => diff * diff);
        return Math.Sqrt(sum);
    }
}