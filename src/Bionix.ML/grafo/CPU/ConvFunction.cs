using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class ConvFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU input, weight, bias, output;
        private readonly int inH, inW, inC, outC, k;
        public ConvFunction(Tensor input_, Tensor weight_, Tensor bias_, Tensor output_, int ksize)
        {
            input = input_ as TensorCPU ?? throw new ArgumentException();
            weight = weight_ as TensorCPU ?? throw new ArgumentException();
            bias = bias_ as TensorCPU ?? throw new ArgumentException();
            output = output_ as TensorCPU ?? throw new ArgumentException();
            inH = input.Shape[0]; inW = input.Shape[1]; inC = input.Shape[2];
            outC = output.Shape[2]; k = ksize;
        }

        public void Backward(double[] gradOutput)
        {
            int pad = k / 2;
            // grad w.r.t input
            if (input.RequiresGrad)
            {
                for (int y = 0; y < inH; y++)
                    for (int x = 0; x < inW; x++)
                        for (int ic = 0; ic < inC; ic++)
                        {
                            double sum = 0.0;
                            for (int oc = 0; oc < outC; oc++)
                            {
                                for (int ky = 0; ky < k; ky++)
                                for (int kx = 0; kx < k; kx++)
                                {
                                    int oy = y + ky - pad;
                                    int ox = x + kx - pad;
                                    if (oy < 0 || oy >= inH || ox < 0 || ox >= inW) continue;
                                    int outIndex = (oy * inW + ox) * outC + oc;
                                    int wIndex = ((oc * inC + ic) * k + ky) * k + kx;
                                    sum += gradOutput[outIndex] * weight[wIndex];
                                }
                            }
                            input.Grad[(y * inW + x) * inC + ic] += sum;
                        }
                input.GradFn?.Backward(input.Grad);
            }

            // grad w.r.t weight
            if (weight.RequiresGrad)
            {
                for (int oc = 0; oc < outC; oc++)
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                double sum = 0.0;
                                for (int y = 0; y < inH; y++)
                                    for (int x = 0; x < inW; x++)
                                    {
                                        int oy = y + ky - (k/2);
                                        int ox = x + kx - (k/2);
                                        if (oy < 0 || oy >= inH || ox < 0 || ox >= inW) continue;
                                        int outIndex = (oy * inW + ox) * outC + oc;
                                        int inIndex = (y * inW + x) * inC + ic;
                                        sum += gradOutput[outIndex] * input[inIndex];
                                    }
                                int wIndex = ((oc * inC + ic) * k + ky) * k + kx;
                                weight.Grad[wIndex] += sum;
                            }
                weight.GradFn?.Backward(weight.Grad);
            }

            // bias grad
            if (bias != null && bias.RequiresGrad)
            {
                for (int oc = 0; oc < outC; oc++)
                {
                    double sum = 0.0;
                    for (int y = 0; y < inH; y++) for (int x = 0; x < inW; x++) sum += gradOutput[(y * inW + x) * outC + oc];
                    bias.Grad[oc] += sum;
                }
                bias.GradFn?.Backward(bias.Grad);
            }
        }
    }
}
