using System;
using System.Threading.Tasks;

namespace Bionix.ML.nucleo.funcoesAtivacao.CPUSIMD.Sigmoid
{
    public static class Sigmoid
    {
        public static double Forward(double x) => 1.0 / (1.0 + Math.Exp(-x));

        public static void Forward(double[] input, int inputOffset, double[] output, int outputOffset, int length)
        {
            if (input == null || output == null) return;
            Parallel.For(0, length, i =>
            {
                double x = input[inputOffset + i];
                output[outputOffset + i] = 1.0 / (1.0 + Math.Exp(-x));
            });
        }
    }
}
