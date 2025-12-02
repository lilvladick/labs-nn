namespace LabsAlgorithm.KnnClassifier;

public class WeightedKnnClassifier
{
    private readonly int _k;
    private readonly List<(double[] Features, string Label)> _trainingData = [];
    private readonly Func<double, double> _weightFunc;

    public WeightedKnnClassifier(int k, Func<double, double>? weightFunc = null)
    {
        if (k <= 0)
            throw new ArgumentException("k должно быть больше 0", nameof(k));

        _k = k;
        _weightFunc = weightFunc ?? (d => 1.0 / (d + 1e-9));
    }

    public void Train(double[] features, string label)
    {
        _trainingData.Add((features, label));
    }

    public string Classify(double[] features)
    {
        if (_trainingData.Count == 0)
            throw new InvalidOperationException("Нет данных для обучения");

        var neighbors = _trainingData
            .Select(t => new
            {
                Distance = CalculateDistance(features, t.Features),
                t.Label
            })
            .OrderBy(t => t.Distance)
            .Take(_k)
            .ToList();

        if (neighbors[0].Distance == 0)
            return neighbors[0].Label;

        var labelScores = neighbors
            .GroupBy(n => n.Label)
            .Select(g => new
            {
                Label = g.Key,
                Score = g.Sum(x => _weightFunc(x.Distance))
            })
            .OrderByDescending(g => g.Score)
            .First();

        return labelScores.Label;
    }
    private static double CalculateDistance(double[] a, double[] b)
    {
        var sum = a.Select((t, i) => t - b[i]).Sum(diff => diff * diff);
        return Math.Sqrt(sum);
    }

    public double Evaluate(List<(double[] Features, string Label)> testData)
    {
        var correct = 0;

        foreach (var (features, label) in testData)
        {
            var predicted = Classify(features);
            if (predicted == label) correct++;
        }

        return (double)correct / testData.Count;
    }
}