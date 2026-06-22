using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class FocalLossFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU logits;
        private readonly TensorCPU targets;
        private readonly TensorCPU outT;
        private readonly double alpha;
        private readonly double gamma;
        private readonly double[] probs;
        private readonly int N;

        public FocalLossFunction(Tensor logits_, Tensor targets_, Tensor out_, double alpha_, double gamma_)
        {
            logits = logits_ as TensorCPU ?? throw new ArgumentException();
            targets = targets_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
            alpha = alpha_;
            gamma = gamma_;
            N = logits.Size;
            probs = new double[N];
            for (int i = 0; i < N; i++)
            {
                double z = logits[i];
                double p = 1.0 / (1.0 + Math.Exp(-z));
                probs[i] = p;
            }
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double eps = 1e-12;
            for (int i = 0; i < N; i++)
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

                // d p / d z = p * (1 - p)
                double dLdZ = dLdP * p * (1.0 - p);
                // average over N
                double grad = g * dLdZ / N;
                if (logits.RequiresGrad)
                {
                    logits.Grad[i] += grad;
                }
            }
            if (logits.RequiresGrad) logits.GradFn?.Backward(logits.Grad);
        }
    }
}
