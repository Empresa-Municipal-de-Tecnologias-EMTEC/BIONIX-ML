using System;

namespace Bionix.ML.dados.normalizacao
{
    public static class Normalizacao
    {
        public static double[] Unit01(double[] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            double[] r = new double[data.Length];
            for (int i = 0; i < data.Length; i++) r[i] = data[i] / 255.0;
            return r;
        }

        public static double[] MeanStd(double[] data, double mean, double std)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            double[] r = new double[data.Length];
            for (int i = 0; i < data.Length; i++) r[i] = (data[i] - mean) / (std == 0 ? 1.0 : std);
            return r;
        }
    }
}
