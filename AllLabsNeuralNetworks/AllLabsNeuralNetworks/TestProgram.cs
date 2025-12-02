using LabsAlgorithm.KnnClassifier;
using LabsAlgorithm.Normalizations;
using LabsAlgorithm.StolpAlgorithm;
using LabsAlgorithm.SVMClassifier;

namespace AllLabsNeuralNetworks;

public static class TestProgram
{
    private static string GetRiskLevel(int deaths) =>
        deaths switch
        {
            < 2000 => "низкий",
            < 8000 => "средний",
            _ => "высокий"
        };

    private static List<(double[] Features, string Label)> LoadDrugData(string csvPath, out Dictionary<string, int> drugToId)
    {
        var lines = File.ReadAllLines(csvPath);
        drugToId = new Dictionary<string, int>();

        if (lines.Length <= 1)
            return new();

        var data = new List<(double[], string)>();
        var nextId = 0;

        for (var i = 1; i < lines.Length; i++)
        {
            var p = lines[i].Split(',');
            if (p.Length < 8) continue;

            if (!int.TryParse(p[1], out var year)) continue;
            if (!int.TryParse(p[2], out var month)) continue;
            var drug = p[4];
            if (!int.TryParse(p[7], out var deaths)) continue;

            if (!drugToId.ContainsKey(drug))
                drugToId[drug] = nextId++;

            double[] features = { year, month, drugToId[drug] };
            var label = GetRiskLevel(deaths);

            data.Add((features, label));
        }

        return data;
    }

    private static List<(double[], string)> ApplyNormalization(List<(double[], string)> data, string type)
    {
        switch (type)
        {
            case "none":
                return data;

            case "minmax":
                var (mn, mx) = Normalizer.FitMinMax(data);
                return Normalizer.LinearNormalization(data, mn, mx);

            case "zscore":
                var (mean, std) = Normalizer.FitZScore(data);
                return Normalizer.ZScoreNormalization(data, mean, std);

            case "log":
                return Normalizer.LogNormalization(data);

            case "maxabs":
                var maxAbs = Normalizer.FitMaxAbs(data);
                return Normalizer.MaxAbsNormalization(data, maxAbs);

            default:
                throw new Exception($"Unknown normalization:{type}");
        }
    }
    
    private static double PredictRaw(LinearSvmClassifier svm, double[] x)
    {
        var method = typeof(LinearSvmClassifier)
            .GetMethod("PredictRaw", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        return (double)method.Invoke(svm, new object[] { x });
    }

    private static Dictionary<string, int> BuildClassIndex(List<(double[] f, string label)> data)
    {
        return data.Select(d => d.label).Distinct()
            .Select((label, index) => (label, index))
            .ToDictionary(x => x.label, x => x.index);
    }

    private static (LinearSvmClassifier[] Svms, Dictionary<string,int> ClassToIndex)
        TrainSvmMulticlass(List<(double[] f, string label)> train)
    {
        var classToIndex = BuildClassIndex(train);
        var numClasses = classToIndex.Count;
        var featureCount = train[0].f.Length;

        var svms = new LinearSvmClassifier[numClasses];
        for (var c = 0; c < numClasses; c++)
            svms[c] = new LinearSvmClassifier(featureCount);

        var x = train.Select(t => t.f).ToArray();

        foreach (var (key, classIdx) in classToIndex)
        {
            var y = train.Select(t =>
                t.label == key ? +1 : -1
            ).ToArray();

            svms[classIdx].Train(x, y);
        }

        return (svms, classToIndex);
    }

    private static string? PredictSvm(LinearSvmClassifier[] svms, Dictionary<string, int> classToIndex, double[] f)
    {
        var bestScore = double.NegativeInfinity;
        string? bestClass = null;

        foreach (var (key, idx) in classToIndex)
        {
            var score = PredictRaw(svms[idx], f);

            if (!(score > bestScore)) continue;
            bestScore = score;
            bestClass = key;
        }

        return bestClass;
    }

    private static double EvaluateKnn(KnnClassifier knn, List<(double[], string)> test)
    {
        int correct = 0;
        foreach (var (f, label) in test)
        {
            var pred = knn.Classify(f);
            if (pred == label) correct++;
        }
        return (double)correct / test.Count;
    }

    private static double EvaluateWeighted(WeightedKnnClassifier knn, List<(double[], string)> test)
    {
        int correct = 0;
        foreach (var (f, label) in test)
        {
            var pred = knn.Classify(f);
            if (pred == label) correct++;
        }
        return (double)correct / test.Count;
    }

    private static double EvaluateSvm(LinearSvmClassifier[] svms, Dictionary<string, int> classToIndex, List<(double[], string)> test)
    {
        int correct = 0;

        foreach (var (f, label) in test)
        {
            var pred = PredictSvm(svms, classToIndex, f);
            if (pred == label) correct++;
        }

        return (double)correct / test.Count;
    }

    public static void Run()
    {
        Console.WriteLine("EXPERIMENT START");

        var all = LoadDrugData("drugs_death.csv", out _);

        var rnd = new Random(42);
        var shuffled = all.OrderBy(_ => rnd.Next()).ToList();
        int trainCount = (int)(shuffled.Count * 0.8);

        var rawTrain = shuffled.Take(trainCount).ToList();
        var rawTest = shuffled.Skip(trainCount).ToList();

        var normalizations = new[]
        {
            "none", "minmax", "zscore", "log", "maxabs"
        };

        foreach (var norm in normalizations)
        {
            Console.WriteLine();
            Console.WriteLine("Normalization: " + norm + "");

            var train = ApplyNormalization(rawTrain, norm);
            var test = ApplyNormalization(rawTest, norm);

            var te = LabsAlgorithm.KnnClassifier.Utils.ComputeTargetEncoding(train);

            List<(double[], string)> MakeFeatures(List<(double[], string)> src) =>
                src.Select(t =>
                (
                    LabsAlgorithm.KnnClassifier.Utils.MakeNewDrugFeatures(
                        (int)t.Item1[0], (int)t.Item1[1], (int)t.Item1[2], te),
                    t.Item2
                )).ToList();

            var featTrain = MakeFeatures(train);
            var featTest = MakeFeatures(test);

            foreach (bool useStolp in new[] { false, true })
            {
                var reducedTrain = useStolp
                    ? Stolp.Reduce(featTrain, 0.0)
                    : featTrain;

                Console.WriteLine();
                Console.WriteLine($"STOLP = {useStolp}");

                // 1) KNN
                var knn = new KnnClassifier(k: 5);
                foreach (var (f, label) in reducedTrain)
                    knn.Train(f, label);
                Console.WriteLine($"knn: {EvaluateKnn(knn, featTest):P2}");

                // 2) Weighted KNN
                var wknn = new WeightedKnnClassifier(k: 5);
                foreach (var (f, label) in reducedTrain)
                    wknn.Train(f, label);
                Console.WriteLine($"weighted_knn: {EvaluateWeighted(wknn, featTest):P2}");

                // 3) SVM
                var (svms, classToIndex) = TrainSvmMulticlass(reducedTrain);
                var svmAcc = EvaluateSvm(svms, classToIndex, featTest);

                Console.WriteLine($"svm: {svmAcc:P2}");
            }
        }

        Console.WriteLine("EXPERIMENT END");
    }
}