using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class IdentityFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src, outT;
        public IdentityFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                for (int i = 0; i < src.Size; i++) src.Grad[i] += gradOutput[i];
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
