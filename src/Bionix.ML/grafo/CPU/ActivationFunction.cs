using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class ActivationFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src, outT;
        public ActivationFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                for (int i = 0; i < src.Size; i++)
                {
                    double grad = src[i] > 0 ? gradOutput[i] : 0.0;
                    src.Grad[i] += grad;
                }
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
