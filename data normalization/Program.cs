using System;
using System.Collections.Generic;
using System.Linq;

public class Normalizer
{
    public static (double[] Min, double[] Max) FitData(List<(double[] Features, string Label)> data)
    {
        if (data.Count == 0)
            throw new ArgumentException("Пустые данные");

        int dim = data[0].Features.Length;
        double[] minValues = new double[dim];
        double[] maxValues = new double[dim];

        for (int i = 0; i < dim; i++)
        {
            minValues[i] = data.Min(d => d.Features[i]);
            maxValues[i] = data.Max(d => d.Features[i]);
        }

        return (minValues, maxValues);
    }

    public static List<(double[] Features, string Label)> LinearNormalization(List<(double[] Features, string Label)> data,double[] minValues,double[] maxValues)
    {
        int dim = minValues.Length;
        var result = new List<(double[] Features, string Label)>(data.Count);

        foreach (var (features, label) in data)
        {
            var norm = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                double range = maxValues[i] - minValues[i];
                norm[i] = range == 0 ? 0 : (features[i] - minValues[i]) / range;
            }
            result.Add((norm, label));
        }

        return result;
    }

    public static double[] NormalizeUserInput(double[] x, double[] min, double[] max)
    {
        var norm = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double range = max[i] - min[i];
            norm[i] = range == 0 ? 0 : (x[i] - min[i]) / range;
        }
        return norm;
    }

}

public class KnnClassifier
{
    private readonly int k;
    private readonly List<(double[] Features, string Label)> trainingData = new();

    public KnnClassifier(int k)
    {
        if (k <= 0) throw new ArgumentException("k должно быть больше 0", nameof(k));
        this.k = k;
    }

    public void Train(double[] features, string label)
    {
        trainingData.Add((features, label));
    }

    public string Classify(double[] features)
    {
        if (trainingData.Count == 0) throw new InvalidOperationException("Нет данных для обучения");


        var neighbors = trainingData
            .Select(t =>
            {
                var d = CalculateDistance(features, t.Features);
                return new { Distance = d, Label = t.Label };
            })
            .OrderBy(t => t.Distance)
            .Take(k);

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
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    public double Evaluate(List<(double[] Features, string Label)> testData)
    {
        int correct = 0;
        foreach (var (features, label) in testData)
        {
            var predicted = Classify(features);
            if (predicted == label) correct++;
        }
        return (double)correct / testData.Count;
    }

    // targetEncoding для препоратов, чтобы модель не путалась а понимала риски тех или иных препоратов
    public static Dictionary<int, double> ComputeTargetEncoding(List<(double[] Features, string Label)> trainData)
    {
        var map = new Dictionary<int, List<double>>();

        foreach (var (features, label) in trainData)
        {
            int drugId = (int)features[2];
            double value = LabelToValue(label);

            if (!map.ContainsKey(drugId))
                map[drugId] = new List<double>();

            map[drugId].Add(value);
        }

        return map.ToDictionary(
            kv => kv.Key,
            kv => kv.Value.Average()
        );
    }

    private static double LabelToValue(string label)
    {
        return label switch
        {
            "low" => 0.0,
            "medium" => 1.0,
            "high" => 2.0,
            _ => 1.0 // если пошло все не по плану то мы тут сделаем медиану )))
        };
    }


    public static double[] MakeNewDrugFeatures(int year, int month, int drugId, Dictionary<int, double> targetEncoding)
    {
        double monthSin = Math.Sin(2 * Math.PI * month / 12.0);
        double monthCos = Math.Cos(2 * Math.PI * month / 12.0);

        double drugEncoded = targetEncoding.ContainsKey(drugId)? targetEncoding[drugId]: 1.0;
        return[year, monthSin, monthCos, drugEncoded];
    }   
}


class Program
{
    private static string GetRiskLevel(int deaths)
    {
        if (deaths < 2000) return "низкий";
        if (deaths < 8000) return "средний";
        return "высокий";
    }

   public static List<(double[] Features, string Label)> LoadDrugData(string csvPath, out Dictionary<string, int> drugToId)
    {
        var lines = File.ReadAllLines(csvPath);
        if (lines.Length <= 1)
        {
            drugToId = [];
            return [];
        }

        drugToId = [];
        int nextId = 0;
        var data = new List<(double[] Features, string Label)>();

        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrEmpty(line)) continue;

            var parts = line.Split(',');
            if (parts.Length < 8) continue;

            if (!int.TryParse(parts[1], out int year)) continue;
            if (!int.TryParse(parts[2], out int month)) continue;
            string drug = parts[4].Trim();
            if (!int.TryParse(parts[7], out int deaths)) continue;

            if (!drugToId.ContainsKey(drug)) drugToId[drug] = nextId++;

            int drugId = drugToId[drug];
            string label = GetRiskLevel(deaths);
            double[] features = [year, month, drugId];
            data.Add((features, label));
        }

        return data;
    }

    static void Main()
    {
        var allData = LoadDrugData("drugs_death.csv", out var drugToId);
        var (minVals, maxVals) = Normalizer.FitData(allData);
        allData = Normalizer.LinearNormalization(allData, minVals, maxVals);

        if (allData.Count == 0)
        {
            Console.WriteLine("Нет данных для обработки");
            return;
        }

        var rnd = new Random(42);
        var shuffled = allData.OrderBy(_ => rnd.Next()).ToList();
        int trainCount = (int)(shuffled.Count * 0.8);
        var trainData = shuffled.Take(trainCount).ToList();
        var testData = shuffled.Skip(trainCount).ToList();

        var targetEncoding = KnnClassifier.ComputeTargetEncoding(trainData);

        var featuredTrainData = trainData.Select(t => (KnnClassifier.MakeNewDrugFeatures((int)t.Features[0], (int)t.Features[1], (int)t.Features[2], targetEncoding), t.Label)).ToList();

        var featuredTestData = testData.Select(t => (KnnClassifier.MakeNewDrugFeatures((int)t.Features[0], (int)t.Features[1], (int)t.Features[2], targetEncoding), t.Label)).ToList();

        var knn = new KnnClassifier(k: 3);
        foreach (var (features, label) in featuredTrainData)
            knn.Train(features, label);

        double accuracy = knn.Evaluate(featuredTestData);
        Console.WriteLine($"Точность: {accuracy:P2}");

        while(true)
        {
            Console.WriteLine("Введите год:");
            var year = Convert.ToInt32(Console.ReadLine());
            Console.WriteLine("Введите месяц (от 1 до 12):");
            var month = Convert.ToInt32(Console.ReadLine());
            if (month < 1 || month > 12)
            {
                Console.WriteLine("Неверный месяц");
                return;
            }
            Console.WriteLine("Айди наркотиков:");
            foreach (var kvp in drugToId.OrderBy(x => x.Value))
            {
                Console.WriteLine($"  {kvp.Value}: {kvp.Key}");
            }
            Console.WriteLine();
            Console.WriteLine("Введите айди наркотика:");
            var drugId = Convert.ToInt32(Console.ReadLine());

            double[] input = new double[] { year, month, drugId };
            double[] norm = Normalizer.NormalizeUserInput(input, minVals, maxVals);
            double[] featuredInput = KnnClassifier.MakeNewDrugFeatures((int)norm[0], (int)norm[1], (int)norm[2], targetEncoding);

            // год месяц и айди наркотика
            string pred = knn.Classify(featuredInput);
            Console.WriteLine($"Прогноз: {pred}");
        }
    }
}