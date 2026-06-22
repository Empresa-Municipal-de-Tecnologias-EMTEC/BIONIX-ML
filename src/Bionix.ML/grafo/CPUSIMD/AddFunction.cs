using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class AddFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD a, b, outT;
        public AddFunction(Tensor a_, Tensor b_, Tensor out_)
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
                b.Context.AddDouble(b.Grad, gradOutput, b.Grad);
                b.GradFn?.Backward(b.Grad);
            }
        }
    }
}
