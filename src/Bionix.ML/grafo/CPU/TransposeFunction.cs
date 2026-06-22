using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class TransposeFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src, outT;
        public TransposeFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            int m = src.Shape[0];
            int n = src.Shape[1];
            if (src.RequiresGrad)
            {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        src.Grad[j * n + i] += gradOutput[i * m + j];
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
