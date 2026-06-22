using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class UpsampleFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU input, output;
        public UpsampleFunction(Tensor input_, Tensor output_)
        {
            input = input_ as TensorCPU ?? throw new ArgumentException();
            output = output_ as TensorCPU ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            // output is 2x upsample of input: each input pixel contributed to 4 outputs
            int inH = input.Shape[0], inW = input.Shape[1], inC = input.Shape[2];
            int outH = output.Shape[0], outW = output.Shape[1];
            for (int y = 0; y < inH; y++)
            for (int x = 0; x < inW; x++)
            for (int ch = 0; ch < inC; ch++)
            {
                int outY = y * 2; int outX = x * 2;
                double sum = 0.0;
                int baseOut = (outY * outW + outX) * inC + ch;
                // accumulate 2x2 block
                sum += gradOutput[baseOut];
                if (outX + 1 < outW) sum += gradOutput[baseOut + inC];
                if (outY + 1 < outH) sum += gradOutput[baseOut + outW * inC];
                if (outY + 1 < outH && outX + 1 < outW) sum += gradOutput[baseOut + outW * inC + inC];
                input.Grad[(y * inW + x) * inC + ch] += sum;
            }
            input.GradFn?.Backward(input.Grad);
        }
    }
}
