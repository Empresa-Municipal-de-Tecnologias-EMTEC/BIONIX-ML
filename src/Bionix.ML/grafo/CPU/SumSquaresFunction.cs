using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class SumSquaresFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src;
        private readonly TensorCPU outT;
        public SumSquaresFunction(TensorCPU src_, TensorCPU out_)
        {
            src = src_ ?? throw new ArgumentException();
            outT = out_ ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            // gradOutput is scalar length-1
            double g = gradOutput[0];
            if (src.RequiresGrad)
            {
                for (int i = 0; i < src.Size; i++) src.Grad[i] += 2.0 * src[i] * g;
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
