using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class SubFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU a, b, outT;
        public SubFunction(Tensor a_, Tensor b_, Tensor out_)
        {
            a = a_ as TensorCPU ?? throw new ArgumentException();
            b = b_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            if (a.RequiresGrad)
            {
                for (int i = 0; i < a.Size; i++) a.Grad[i] += gradOutput[i];
                a.GradFn?.Backward(a.Grad);
            }
            if (b.RequiresGrad)
            {
                for (int i = 0; i < b.Size; i++) b.Grad[i] -= gradOutput[i];
                b.GradFn?.Backward(b.Grad);
            }
        }
    }
}
