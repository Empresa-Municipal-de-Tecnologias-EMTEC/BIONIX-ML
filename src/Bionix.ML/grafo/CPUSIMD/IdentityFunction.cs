using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class IdentityFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src, outT;
        public IdentityFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                src.Context.AddInto(gradOutput, 0, src.Grad, 0, outT.Size);
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
