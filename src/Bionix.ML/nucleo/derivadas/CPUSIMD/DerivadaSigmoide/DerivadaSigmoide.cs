using System.Numerics;

namespace Bionix.ML.nucleo.derivadas.CPUSIMD.DerivadaSigmoide
{
    public static class DerivadaSigmoide
    {
        public static double FromActivated(double activated) => activated * (1.0 - activated);

        public static void FromActivated(double[] activated, int activatedOffset, double[] output, int outputOffset, int length)
        {
            if (activated == null) return;
            if (output == null) return;

            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                int limit = length - (length % vecSize);
                for (; i < limit; i += vecSize)
                {
                    var v = new Vector<double>(activated, activatedOffset + i);
                    var one = new Vector<double>(1.0);
                    var res = v * (one - v);
                    res.CopyTo(output, outputOffset + i);
                }
            }

            for (; i < length; ++i)
            {
                double a = activated[activatedOffset + i];
                output[outputOffset + i] = a * (1.0 - a);
            }
        }
    }
}
