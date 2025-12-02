namespace LabsAlgorithms.KnnClassifier;

public class KnnClassifier
{
    private readonly int _k;
    private readonly List<(double[] Features, string Label)> _trainingData = [];

    public KnnClassifier(int k)
    {
        if (k <= 0) throw new ArgumentException("k должно быть больше 0", nameof(k));
        this._k = k;
    }

    public void Train(double[] features, string label)
    {
        _trainingData.Add((features, label));
    }

    public string Classify(double[] features)
    {
        if (_trainingData.Count == 0) throw new InvalidOperationException("Нет данных для обучения");


        var neighbors = _trainingData
            .Select(t =>
            {
                var d = CalculateDistance(features, t.Features);
                return new { Distance = d, t.Label };
            })
            .OrderBy(t => t.Distance)
            .Take(_k);

        var labelScores = neighbors
            .GroupBy(n => n.Label)
            .Select(g => new
            {
                Label = g.Key,
                Score = g.Sum(x => 1.0 / (x.Distance + 1e-9)) // вот этот 1е-9 нужен чтобы на 0 случайно не поделить
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