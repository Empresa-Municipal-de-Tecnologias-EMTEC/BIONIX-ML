using System;
using System.Numerics;

namespace Bionix.ML.nucleo.funcoesAtivacao.CPUSIMD.ReLU
{
    public static class ReLU
    {
        public static double Forward(double x) => x > 0.0 ? x : 0.0;

        public static void Forward(double[] input, int inputOffset, double[] output, int outputOffset, int length)
        {
            if (input == null || output == null) return;
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                var zero = new Vector<double>(0.0);
                for (; i <= length - vecSize; i += vecSize)
                {
                    var v = new Vector<double>(input, inputOffset + i);
                    var r = Vector.Max(v, zero);
                    r.CopyTo(output, outputOffset + i);
                }
            }
            for (; i < length; ++i)
            {
                double x = input[inputOffset + i];
                output[outputOffset + i] = x > 0.0 ? x : 0.0;
            }
        }
    }
}
