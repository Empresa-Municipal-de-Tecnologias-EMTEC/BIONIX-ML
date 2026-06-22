using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class FocalLossFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD logits;
        private readonly TensorCPUSIMD targets;
        private readonly TensorCPUSIMD outT;
        private readonly double alpha;
        private readonly double gamma;
        private readonly double[] probs;
        private readonly int N;

        public FocalLossFunction(Tensor logits_, Tensor targets_, Tensor out_, double alpha_, double gamma_)
        {
            logits = logits_ as TensorCPUSIMD ?? throw new ArgumentException();
            targets = targets_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
            alpha = alpha_;
            gamma = gamma_;
            N = logits.Size;
            probs = new double[N];

            // Parallelize sigmoid computation for probs (exp is scalar per element)
            System.Threading.Tasks.Parallel.For(0, N, i =>
            {
                double z = logits[i];
                probs[i] = 1.0 / (1.0 + Math.Exp(-z));
            });
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double eps = 1e-12;
            if (!logits.RequiresGrad) return;

            // Compute per-element gradients into a temp buffer in parallel,
            // then add into logits.Grad with a SIMD-accelerated helper.
            var tmp = new double[N];
            System.Threading.Tasks.Parallel.For(0, N, i =>
            {
                double p = probs[i];
                double t = targets[i];
                double dLdP = 0.0;
                if (t >= 0.5)
                {
                    double one_minus_p = 1.0 - p;
                    dLdP = alpha * gamma * Math.Pow(one_minus_p, gamma - 1.0) * Math.Log(p + eps)
                            - alpha * Math.Pow(one_minus_p, gamma) / (p + eps);
                }
                else
                {
                    double ppow = Math.Pow(p, gamma);
                    dLdP = -(1.0 - alpha) * gamma * Math.Pow(p, gamma - 1.0) * Math.Log(1.0 - p + eps)
                            + (1.0 - alpha) * ppow / (1.0 - p + eps);
                }

                double dLdZ = dLdP * p * (1.0 - p);
                tmp[i] = g * dLdZ / N;
            });

            // Add tmp into logits.Grad using SIMD via context helper
            logits.Context.AddInto(tmp, 0, logits.Grad, 0, N);
            logits.GradFn?.Backward(logits.Grad);
        }
    }
}
