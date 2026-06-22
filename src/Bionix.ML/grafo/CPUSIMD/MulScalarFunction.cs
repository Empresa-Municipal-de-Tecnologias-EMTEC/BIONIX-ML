using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class MulScalarFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src, outT;
        private readonly double scalar;

        public MulScalarFunction(Tensor src_, double scalar_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            scalar = scalar_;
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                // src.Grad += gradOutput * scalar (vectorized, avoid temp allocation)
                src.Context.AddScaled(gradOutput, scalar, src.Grad);
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
