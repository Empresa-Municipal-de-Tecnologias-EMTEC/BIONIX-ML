using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class SumSquaresFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src;
        private readonly TensorCPUSIMD outT;
        public SumSquaresFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }
        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            if (src.RequiresGrad)
            {
                // dst += src * (2*g)  -- use AddScaled to avoid temporary allocation
                src.Context.AddScaled(src.Data, 2.0 * g, src.Grad);
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
