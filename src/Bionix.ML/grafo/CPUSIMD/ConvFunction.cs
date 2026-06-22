using System;
using System.Numerics;
using System.Threading.Tasks;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class ConvFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD input, weight, bias, output;
        private readonly int inH, inW, inC, outC, k;

        public ConvFunction(Tensor input_, Tensor weight_, Tensor bias_, Tensor output_, int ksize)
        {
            input = input_ as TensorCPUSIMD ?? throw new ArgumentException();
            weight = weight_ as TensorCPUSIMD ?? throw new ArgumentException();
            bias = bias_ as TensorCPUSIMD ?? throw new ArgumentException();
            output = output_ as TensorCPUSIMD ?? throw new ArgumentException();
            inH = input.Shape[0]; inW = input.Shape[1]; inC = input.Shape[2];
            outC = output.Shape[2]; k = ksize;
        }

        public void Backward(double[] gradOutput)
        {
            int pad = k / 2;
            // grad w.r.t input
            if (input.RequiresGrad)
            {
                var inGrad = input.GradArray;
                var wData = weight.Data ?? weight.Grad; // weight may be TensorCPUSIMD or similar
                var outData = gradOutput;
                int vecSizeGlobal = Vector<double>.Count;
                var tmpLocal = new System.Threading.ThreadLocal<double[]>(() => new double[vecSizeGlobal]);
                try
                {
                    Parallel.For(0, inH, y =>
                    {
                        // per-thread buffer for gathers to avoid per-iteration allocations
                        double[] tmp = tmpLocal.Value;

                        for (int x = 0; x < inW; x++)
                        {
                            int baseIdx = (y * inW + x) * inC;
                            for (int ic = 0; ic < inC; ic++)
                            {
                                double sum = 0.0;
                                // accumulate over output channels and kernel window
                                for (int ky = 0; ky < k; ky++)
                                for (int kx = 0; kx < k; kx++)
                                {
                                    int oy = y + ky - pad;
                                    int ox = x + kx - pad;
                                    if (oy < 0 || oy >= inH || ox < 0 || ox >= inW) continue;
                                    int outBase = (oy * inW + ox) * outC;
                                    // weight stride between consecutive output channels
                                    int block = inC * k * k;
                                    int innerOffset = (ic * k + ky) * k + kx;

                                    // vectorized pass across oc in blocks of vecSizeGlobal
                                    int oc = 0;
                                    var vAcc = Vector<double>.Zero;
                                    for (; oc <= outC - vecSizeGlobal; oc += vecSizeGlobal)
                                    {
                                        int wBase = oc * block + innerOffset;
                                        // gather weight values for this oc block into tmp
                                        for (int t = 0; t < vecSizeGlobal; t++) tmp[t] = wData[wBase + t * block];
                                        var vW = new Vector<double>(tmp);
                                        var vOut = new Vector<double>(outData, outBase + oc);
                                        vAcc += vOut * vW;
                                    }
                                    // horizontal sum of vector accumulator
                                    for (int t = 0; t < vecSizeGlobal; t++) sum += vAcc[t];
                                    // scalar tail
                                    for (; oc < outC; oc++)
                                    {
                                        int outIndex = outBase + oc;
                                        int wIndex = oc * block + innerOffset;
                                        sum += outData[outIndex] * wData[wIndex];
                                    }
                                }
                                inGrad[baseIdx + ic] += sum;
                            }
                        }
                    });
                }
                finally
                {
                    tmpLocal.Dispose();
                }

                input.GradFn?.Backward(inGrad);
            }

            // grad w.r.t weight (vectorized across output-channel blocks)
            if (weight.RequiresGrad)
            {
                var wGrad = weight.GradArray;
                var inData = input.Data;
                var outData = gradOutput;
                int vecSize = Vector<double>.Count;
                int ocBlocks = outC / vecSize;
                int blockStride = inC * k * k;

                // Full SIMD blocks: each block handles `vecSize` output channels at once
                Parallel.For(0, ocBlocks, b =>
                {
                    int ocStart = b * vecSize;
                    for (int ic = 0; ic < inC; ic++)
                    {
                        for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++)
                        {
                            var vAcc = Vector<double>.Zero;
                            for (int y = 0; y < inH; y++)
                            {
                                for (int x = 0; x < inW; x++)
                                {
                                    int oy = y + ky - pad;
                                    int ox = x + kx - pad;
                                    if (oy < 0 || oy >= inH || ox < 0 || ox >= inW) continue;
                                    int outBase = (oy * inW + ox) * outC;
                                    int inIndex = (y * inW + x) * inC + ic;
                                    // load vecSize output channel values starting at ocStart
                                    var vOut = new Vector<double>(outData, outBase + ocStart);
                                    // multiply by input scalar and accumulate
                                    vAcc += vOut * new Vector<double>(inData[inIndex]);
                                }
                            }
                            // write accumulated vector to wGrad (non-contiguous by blockStride)
                            int wBase = ((ocStart * inC + ic) * k + ky) * k + kx;
                            for (int t = 0; t < vecSize; t++)
                            {
                                wGrad[wBase + t * blockStride] += vAcc[t];
                            }
                        }
                    }
                });

                // Remainder output channels (scalar)
                int ocRemStart = ocBlocks * vecSize;
                if (ocRemStart < outC)
                {
                    Parallel.For(ocRemStart, outC, oc =>
                    {
                        for (int ic = 0; ic < inC; ic++)
                        {
                            for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                double sum = 0.0;
                                for (int y = 0; y < inH; y++)
                                    for (int x = 0; x < inW; x++)
                                    {
                                        int oy = y + ky - pad;
                                        int ox = x + kx - pad;
                                        if (oy < 0 || oy >= inH || ox < 0 || ox >= inW) continue;
                                        int outIndex = (oy * inW + ox) * outC + oc;
                                        int inIndex = (y * inW + x) * inC + ic;
                                        sum += outData[outIndex] * inData[inIndex];
                                    }
                                int wIndex = ((oc * inC + ic) * k + ky) * k + kx;
                                wGrad[wIndex] += sum;
                            }
                        }
                    });
                }

                weight.GradFn?.Backward(wGrad);
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
