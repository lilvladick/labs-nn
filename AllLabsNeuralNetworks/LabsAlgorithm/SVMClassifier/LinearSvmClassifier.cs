namespace LabsAlgorithm.SVMClassifier;

public class LinearSvmClassifier(int featureCount, double lambda = 0.01, double learningRate = 0.1)
{
    private readonly double[] _w = new double[featureCount];
    private double _b;

    public void Train(double[][] x, int[] y, int epochs = 100)
    {
        var n = x.Length;

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            for (var i = 0; i < n; i++)
            {
                var margin = y[i] * PredictRaw(x[i]);

                if (margin >= 1)
                {
                    for (var j = 0; j < _w.Length; j++)
                        _w[j] -= learningRate * (2 * lambda * _w[j]);
                }
                else
                {
                    for (var j = 0; j < _w.Length; j++)
                        _w[j] += learningRate * (y[i] * x[i][j] - 2 * lambda * _w[j]);

                    _b += learningRate * y[i];
                }
            }
        }
    }

    private double PredictRaw(double[] x)
    {
        return _b + _w.Select((t, i) => t * x[i]).Sum();
    }

    public int Predict(double[] x)
    {
        return PredictRaw(x) >= 0 ? +1 : -1;
    }
}