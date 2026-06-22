using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    // Elementwise sigmoid autograd function for TensorCPU
    public class SigmoidFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src;
        private readonly TensorCPU outT;
        private readonly int N;

        public SigmoidFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPU ?? throw new ArgumentException("src must be TensorCPU");
            outT = out_ as TensorCPU ?? throw new ArgumentException("out must be TensorCPU");
            N = src.Size;
        }

        public void Backward(double[] gradOutput)
        {
            if (!src.RequiresGrad) return;
            var s = src;
            var o = outT;
            var dst = s.Grad;
            for (int i = 0; i < N; i++)
            {
                double pi = o[i]; // sigmoid(src)
                double grad = gradOutput[i] * pi * (1.0 - pi);
                dst[i] += grad;
            }
            s.GradFn?.Backward(s.Grad);
        }
    }
}
