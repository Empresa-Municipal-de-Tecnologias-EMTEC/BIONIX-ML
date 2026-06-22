using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class ConcatFunction : IFuncaoRetropropagacao
    {
        private readonly Bionix.ML.nucleo.tensor.Tensor[] parts;
        private readonly Bionix.ML.nucleo.tensor.Tensor output;
        public ConcatFunction(Bionix.ML.nucleo.tensor.Tensor[] parts_, Bionix.ML.nucleo.tensor.Tensor output_)
        {
            parts = new Bionix.ML.nucleo.tensor.Tensor[parts_.Length];
            for (int i = 0; i < parts_.Length; i++) parts[i] = parts_[i] ?? throw new ArgumentException();
            output = output_ ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            int idx = 0;
            foreach (var p in parts)
            {
                for (int i = 0; i < p.Size; i++)
                {
                    p.Grad[i] += gradOutput[idx++];
                }
                p.GradFn?.Backward(p.Grad);
            }
        }
    }
}
