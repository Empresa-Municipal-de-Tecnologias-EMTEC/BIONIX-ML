using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class DownsampleFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD input, output;
        public DownsampleFunction(Tensor input_, Tensor output_)
        {
            input = input_ as TensorCPUSIMD ?? throw new ArgumentException();
            output = output_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            int inH = input.Shape[0], inW = input.Shape[1], inC = input.Shape[2];
            int outH = output.Shape[0], outW = output.Shape[1];
            // Parallelize outer spatial loops; each index written by single thread
            System.Threading.Tasks.Parallel.For(0, outH, y =>
            {
                for (int x = 0; x < outW; x++)
                {
                    int sy = Math.Min(inH - 1, y * 2);
                    int sx = Math.Min(inW - 1, x * 2);
                    int baseIn = (sy * inW + sx) * inC;
                    int baseOut = (y * outW + x) * inC;
                    // add across channels (contiguous)
                    input.Context.AddInto(gradOutput, baseOut, input.Grad, baseIn, inC);
                }
            });
            input.GradFn?.Backward(input.Grad);
        }
    }
}
