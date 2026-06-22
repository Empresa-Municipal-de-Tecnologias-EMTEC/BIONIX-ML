using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Exemplo0001RegressaoLinear
{
    internal static class Program
    {
        static void Main()
        {
            var ctx = new ComputacaoCPUContexto();

            // Create synthetic data y = 3 + 2*x + noise
            int n = 100;
            var xs = new double[n];
            var ys = new double[n];
            var rand = new Random(42);
            for (int i = 0; i < n; i++)
            {
                xs[i] = i / 10.0;
                ys[i] = 3.0 + 2.0 * xs[i] + (rand.NextDouble() - 0.5) * 1.0;
            }

            // Build design matrix X (n x 2), column0 = 1 for bias, column1 = x
            double[] Xdata = new double[n * 2];
            for (int i = 0; i < n; i++)
            {
                Xdata[i * 2 + 0] = 1.0;
                Xdata[i * 2 + 1] = xs[i];
            }
            var fabrica = new FabricaTensor(ctx);
            var X = fabrica.FromArray(new[] { n, 2 }, Xdata);
            var y = fabrica.FromArray(new[] { n, 1 }, ys);

            // Initialize weights w (2 x 1) to zeros
            var w = fabrica.Criar(2, 1);

            double lr = 0.0005;
            int iters = 5000;

            for (int it = 0; it < iters; it++)
            {
                // pred = X * w  -> (n x 2) * (2 x 1) => (n x 1)
                var pred = X.MatMul(w);
                var error = pred.Sub(y); // (n x 1)
                // grad = (2/n) * X^T * error  -> (2 x n) * (n x 1) => (2 x 1)
                var grad = X.Transpose().MatMul(error).MulScalar(2.0 / n);
                // update w = w - lr * grad
                w = w.Sub(grad.MulScalar(lr));
            }

            Console.WriteLine("Learned weights:");
            var wArr = w.ToArray();
            Console.WriteLine($"bias={wArr[0]:F4}, slope={wArr[1]:F4}");
        }
    }
}
