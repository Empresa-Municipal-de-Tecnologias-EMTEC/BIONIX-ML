using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class SmoothL1Function : IFuncaoRetropropagacao
    {
        private readonly TensorCPU preds;
        private readonly TensorCPU targets;
        private readonly TensorCPU outT;
        private readonly int N;

        public SmoothL1Function(Tensor preds_, Tensor targets_, Tensor out_)
        {
            preds = preds_ as TensorCPU ?? throw new ArgumentException();
            targets = targets_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
            N = preds.Size;
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            for (int i = 0; i < N; i++)
            {
                double diff = preds[i] - targets[i];
                double abs = Math.Abs(diff);
                double grad = abs < 1.0 ? diff : Math.Sign(diff);
                double finalGrad = g * grad / N;
                if (preds.RequiresGrad)
                {
                    preds.Grad[i] += finalGrad;
                }
            }
            if (preds.RequiresGrad) preds.GradFn?.Backward(preds.Grad);
        }
    }
}
