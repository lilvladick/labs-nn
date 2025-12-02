namespace LabsAlgorithm.KnnClassifier;

public static class Utils
{
    public static Dictionary<int, double> ComputeTargetEncoding(List<(double[] Features, string Label)> trainData)
    {
        var map = new Dictionary<int, List<double>>();

        foreach (var (features, label) in trainData)
        {
            var drugId = (int)features[2];
            var value = LabelToValue(label);

            if (!map.ContainsKey(drugId))
                map[drugId] = [];

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
            "низкий" => 0.0,
            "средний" => 1.0,
            "высокий" => 2.0,
            _ => 1.0 
        };
    }


    public static double[] MakeNewDrugFeatures(int year, int month, int drugId, Dictionary<int, double> targetEncoding)
    {
        var monthSin = Math.Sin(2 * Math.PI * month / 12.0);
        var monthCos = Math.Cos(2 * Math.PI * month / 12.0);

        var drugEncoded = targetEncoding.GetValueOrDefault(drugId, 1.0);
        return[year, monthSin, monthCos, drugEncoded];
    }  
}