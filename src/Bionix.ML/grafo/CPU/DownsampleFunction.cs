using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class DownsampleFunction : IFuncaoRetropropagacao
    {
        private readonly Bionix.ML.nucleo.tensor.Tensor input, output;
        public DownsampleFunction(Bionix.ML.nucleo.tensor.Tensor input_, Bionix.ML.nucleo.tensor.Tensor output_)
        {
            input = input_ ?? throw new ArgumentException();
            output = output_ ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            int inH = input.Shape[0], inW = input.Shape[1], inC = input.Shape[2];
            int outH = output.Shape[0], outW = output.Shape[1];
            for (int y = 0; y < outH; y++)
            for (int x = 0; x < outW; x++)
            for (int ch = 0; ch < inC; ch++)
            {
                int sy = Math.Min(inH - 1, y * 2);
                int sx = Math.Min(inW - 1, x * 2);
                int inIndex = (sy * inW + sx) * inC + ch;
                int outIndex = (y * outW + x) * inC + ch;
                input.Grad[inIndex] += gradOutput[outIndex];
            }
            input.GradFn?.Backward(input.Grad);
        }
    }
}
