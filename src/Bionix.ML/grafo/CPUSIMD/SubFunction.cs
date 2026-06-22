using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class SubFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD a, b, outT;
        public SubFunction(Tensor a_, Tensor b_, Tensor out_)
        {
            a = a_ as TensorCPUSIMD ?? throw new ArgumentException();
            b = b_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (a.RequiresGrad)
            {
                a.Context.AddDouble(a.Grad, gradOutput, a.Grad);
                a.GradFn?.Backward(a.Grad);
            }
            if (b.RequiresGrad)
            {
                // b receives negative gradient: use AddScaled to avoid temp allocation
                b.Context.AddScaled(gradOutput, -1.0, b.Grad);
                b.GradFn?.Backward(b.Grad);
            }
        }
    }
}
