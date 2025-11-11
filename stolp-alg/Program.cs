using System;

public class Program
{
    static void Main()
    {
        var trainingData = StudentDataGenerator.Generate(100);
        Console.WriteLine($"Исходных данных: {trainingData.Count}");

        var reducedData = StolpReducer.Reduce(trainingData, minMargin: 0.0);
        Console.WriteLine($"После STOLP осталось: {reducedData.Count}");

        var knn = new KnnClassifier(k: 5);
        foreach (var (features, label) in reducedData)
            knn.Train(features, label);

        var testData = StudentDataGenerator.Generate(30);
        double accuracy = knn.Evaluate(testData);

        Console.WriteLine($"Точность после STOLP: {accuracy:P2}");

        double[] student = [4.2, 8];
        string predicted = knn.Classify(student);

        Console.WriteLine($"Студент → {predicted}");
    }    
}
