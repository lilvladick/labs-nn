using LabsAlgorithm.KnnClassifier;
using LabsAlgorithm.Normalizations;
using LabsAlgorithm.StolpAlgorithm;
using Utils = LabsAlgorithm.Normalizations.Utils;

namespace AllLabsNeuralNetworks;

class Program
{
    private static string GetRiskLevel(int deaths)
    {
        return deaths switch
        {
            < 2000 => "низкий",
            < 8000 => "средний",
            _ => "высокий"
        };
    }


    private static List<(double[] Features, string Label)> LoadDrugData( string csvPath, out Dictionary<string, int> drugToId)
    {
        var lines = File.ReadAllLines(csvPath);
        drugToId = new Dictionary<string, int>();

        if (lines.Length <= 1)
            return [];

        var data = new List<(double[] Features, string Label)>();
        var nextId = 0;

        for (var i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 8) continue;

            if (!int.TryParse(parts[1], out var year)) continue;
            if (!int.TryParse(parts[2], out var month)) continue;
            var drug = parts[4];
            if (!int.TryParse(parts[7], out var deaths)) continue;

            if (!drugToId.ContainsKey(drug))
                drugToId[drug] = nextId++;

            double[] features = [year, month, drugToId[drug]];
            var label = GetRiskLevel(deaths);

            data.Add((features, label));
        }

        return data;
    }
    
    private static Func<List<(double[], string)>, List<(double[], string)>> SelectNormalization(
        List<(double[], string)> data,
        out object? fitParams)
    {
        Console.WriteLine("Выберите метод нормализации:");
        Console.WriteLine("1 — Linear (MinMax)");
        Console.WriteLine("2 — Z-Score");
        Console.WriteLine("3 — Logarithmic");
        Console.WriteLine("4 — MaxAbs");
        Console.Write("Ваш выбор: ");

        var choice = Console.ReadLine();

        switch (choice)
        {
            case "1":
                var (min, max) = Normalizer.FitMinMax(data);
                fitParams = (min, max);
                return d => Normalizer.LinearNormalization(d, min, max);

            case "2":
                var (mean, std) = Normalizer.FitZScore(data);
                fitParams = (mean, std);
                return d => Normalizer.ZScoreNormalization(d, mean, std);

            case "3":
                fitParams = null;
                return Normalizer.LogNormalization;

            case "4":
                var maxAbs = Normalizer.FitMaxAbs(data);
                fitParams = maxAbs;
                return d => Normalizer.MaxAbsNormalization(d, maxAbs);

            default:
                Console.WriteLine("Выбран MinMax по умолчанию.");
                var (mn, mx) = Normalizer.FitMinMax(data);
                fitParams = (mn, mx);
                return d => Normalizer.LinearNormalization(d, mn, mx);
        }
    }
    
    private static void MainFake()
    {
        var allData = LoadDrugData("drugs_death.csv", out var drugToId);
        if (allData.Count == 0)
        {
            Console.WriteLine("Нет данных.");
            return;
        }
        var normalizer = SelectNormalization(allData, out var fitParams);

        allData = normalizer(allData);

        var rnd = new Random(42);
        var shuffled = allData.OrderBy(_ => rnd.Next()).ToList();
        var trainCount = (int)(shuffled.Count * 0.8);
        var trainData = shuffled.Take(trainCount).ToList();
        var testData = shuffled.Skip(trainCount).ToList();

        var targetEncoding = LabsAlgorithm.KnnClassifier.Utils.ComputeTargetEncoding(trainData);

        var featuredTrainData = trainData.Select(t =>
            (LabsAlgorithm.KnnClassifier.Utils.MakeNewDrugFeatures(
                (int)t.Features[0], (int)t.Features[1], (int)t.Features[2], targetEncoding), t.Label)
        ).ToList();

        var featuredTestData = testData.Select(t =>
            (LabsAlgorithm.KnnClassifier.Utils.MakeNewDrugFeatures(
                (int)t.Features[0], (int)t.Features[1], (int)t.Features[2], targetEncoding), t.Label)
        ).ToList();
        
        var reducedTrainData = Stolp.Reduce(featuredTrainData, minMargin: 0.0);

        var knn = new WeightedKnnClassifier(k: 5);
        foreach (var (f, label) in reducedTrainData)
            knn.Train(f, label);

        var acc = knn.Evaluate(featuredTestData);
        Console.WriteLine($"Точность: {acc:P2}");
        Console.WriteLine();

        while (true)
        {
            Console.WriteLine("Введите год:");
            var year = int.Parse(Console.ReadLine() ?? string.Empty);
            Console.WriteLine("Введите месяц (1-12):");
            var month = int.Parse(Console.ReadLine() ?? string.Empty);

            Console.WriteLine("Айди наркотиков:");
            foreach (var kv in drugToId.OrderBy(x => x.Value))
                Console.WriteLine($"{kv.Value}: {kv.Key}");

            Console.WriteLine("Введите айди:");
            var drugId = int.Parse(Console.ReadLine() ?? string.Empty);

            double[] input = [year, month, drugId];
            if (fitParams == null) continue;
            var normalized = NormalizeUserInput(input, fitParams);

            var feat = LabsAlgorithm.KnnClassifier.Utils.MakeNewDrugFeatures(
                (int)normalized[0], (int)normalized[1], (int)normalized[2], targetEncoding);

            var pred = knn.Classify(feat);
            Console.WriteLine($"Прогноз: {pred}");
        }
    }
    
    private static double[] NormalizeUserInput(double[] input, object fitParams)
    {
        return fitParams switch
        {
            null => Utils.NormalizeUserInputLog(input),
            (double[] min, double[] max) => Utils.NormalizeUserInputMinMax(input, min,
                max),
            _ => fitParams switch
            {
                (double[] mean, double[] std) => Utils.NormalizeUserInputZScore(input,
                    mean, std),
                double[] maxAbs => Utils.NormalizeUserInputMaxAbs(input, maxAbs),
                _ => throw new Exception("Неизвестная нормализация")
            }
        };
    }

    static void Main()
    {
        TestProgram.Run();
    }

}