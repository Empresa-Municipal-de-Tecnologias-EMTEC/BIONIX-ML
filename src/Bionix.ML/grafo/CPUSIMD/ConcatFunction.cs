using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class ConcatFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD[] parts;
        private readonly TensorCPUSIMD output;
        public ConcatFunction(Tensor[] parts_, Tensor output_)
        {
            parts = new TensorCPUSIMD[parts_.Length];
            for (int i = 0; i < parts_.Length; i++) parts[i] = parts_[i] as TensorCPUSIMD ?? throw new ArgumentException();
            output = output_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            int idx = 0;
            foreach (var p in parts)
            {
                // vectorized segment add: p.Grad += gradOutput[idx .. idx+p.Size]
                p.Context.AddInto(gradOutput, idx, p.Grad, 0, p.Size);
                idx += p.Size;
                p.GradFn?.Backward(p.Grad);
            }
        }
    }
}
