using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class BCEFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU preds;
        private readonly TensorCPU targets;
        private readonly TensorCPU outT;
        private readonly int N;
        private readonly double[] pvals;

        public BCEFunction(Tensor preds_, Tensor targets_, Tensor out_)
        {
            preds = preds_ as TensorCPU ?? throw new ArgumentException();
            targets = targets_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
            N = preds.Size;
            pvals = new double[N];
            for (int i = 0; i < N; i++)
            {
                double pv = preds[i];
                // clamp
                if (pv < 1e-12) pv = 1e-12;
                else if (pv > 1.0 - 1e-12) pv = 1.0 - 1e-12;
                pvals[i] = pv;
            }
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double eps = 1e-12;
            if (!preds.RequiresGrad) return;
            var dst = preds.Grad;
            for (int i = 0; i < N; i++)
            {
                double p = pvals[i];
                double t = targets[i];
                double grad = g * (-(t / p) + (1.0 - t) / (1.0 - p)) / N;
                dst[i] += grad;
            }
            preds.GradFn?.Backward(preds.Grad);
        }
    }
}
