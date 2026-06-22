using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class MulScalarFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU a, outT;
        private readonly double scalar;
        public MulScalarFunction(Tensor a_, double s, Tensor out_)
        {
            a = a_ as TensorCPU ?? throw new ArgumentException();
            scalar = s;
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (a.RequiresGrad)
            {
                for (int i = 0; i < a.Size; i++) a.Grad[i] += gradOutput[i] * scalar;
                a.GradFn?.Backward(a.Grad);
            }
        }
    }
}
