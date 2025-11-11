using System;
using System.Collections.Generic;
using System.Linq;


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
}

public static class StudentDataGenerator
{
    private static readonly Random rnd = new();

    public static List<(double[] Features, string Label)> Generate(int count)
    {
        var data = new List<(double[], string)>();

        for (int i = 0; i < count; i++)
        {
            double grade = Math.Round(rnd.NextDouble() * 3 + 2, 2);
            double absences = rnd.Next(0, 30);

            string label = ClassifyStudent(grade, absences);
            data.Add((new double[] { grade, absences }, label));
        }

        return data;
    }

    private static string ClassifyStudent(double grade, double absences)
    {
         // средний балл, количество пропущенных занятий
        if (grade >= 4.5 && absences < 5) return "Отл";
        if (grade >= 3.5 && absences < 10) return "Хор";
        if (grade >= 3.0 && absences < 15) return "удовл";
        return "неуд";
    }
}

class Program
{
    static void Main()
    {
        var knn = new KnnClassifier(k: 5);

        var trainingData = StudentDataGenerator.Generate(100);
        foreach (var (features, label) in trainingData)
            knn.Train(features, label);

        var testData = StudentDataGenerator.Generate(30);

        double accuracy = knn.Evaluate(testData);
        Console.WriteLine($"Точность на тестовых данных: {accuracy:P2}");

        double[] student = [4.2, 8];
        string predicted = knn.Classify(student);

        Console.WriteLine($"Студент → {predicted}");
    }
}
