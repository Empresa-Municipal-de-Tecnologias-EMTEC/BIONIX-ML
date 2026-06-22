using System;

namespace Bionix.ML.nucleo.funcoesPerda.MSE
{
    public static class MSE
    {
        public static double Loss(double[] predicted, double[] target)
        {
            if (predicted == null) throw new ArgumentNullException(nameof(predicted));
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (predicted.Length != target.Length) throw new ArgumentException("predicted and target must have same length");
            double sum = 0.0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double d = predicted[i] - target[i];
                sum += d * d;
            }
            return sum / predicted.Length;
        }
    }
}
