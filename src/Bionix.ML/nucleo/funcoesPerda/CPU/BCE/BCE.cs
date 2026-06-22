using System;

namespace Bionix.ML.nucleo.funcoesPerda.BCE
{
    public static class BCE
    {
        public static double Loss(double[] predicted, double[] target)
        {
            if (predicted == null) throw new ArgumentNullException(nameof(predicted));
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (predicted.Length != target.Length) throw new ArgumentException("predicted and target must have same length");
            double sum = 0.0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double p = Math.Max(1e-12, Math.Min(1.0 - 1e-12, predicted[i]));
                sum += -(target[i] * Math.Log(p) + (1 - target[i]) * Math.Log(1 - p));
            }
            return sum / predicted.Length;
        }
    }
}
