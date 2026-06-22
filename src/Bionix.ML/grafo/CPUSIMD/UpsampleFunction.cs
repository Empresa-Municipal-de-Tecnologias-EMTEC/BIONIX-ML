using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class UpsampleFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD input, output;
        public UpsampleFunction(Tensor input_, Tensor output_)
        {
            input = input_ as TensorCPUSIMD ?? throw new ArgumentException();
            output = output_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            int inH = input.Shape[0], inW = input.Shape[1], inC = input.Shape[2];
            int outH = output.Shape[0], outW = output.Shape[1];
            // Parallelize across rows for throughput; inner channel loop is contiguous
            System.Threading.Tasks.Parallel.For(0, inH, y =>
            {
                for (int x = 0; x < inW; x++)
                {
                    int outY = y * 2; int outX = x * 2;
                    int baseOut = (outY * outW + outX) * inC;
                    int baseIn = (y * inW + x) * inC;
                    // accumulate up to 4 neighboring outputs for each channel
                    // Vectorize channel accumulation when the full 2x2 block exists
                    bool fullBlock = (outX + 1 < outW) && (outY + 1 < outH);
                    var dst = input.Grad;
                    if (fullBlock)
                    {
                        int vecSize = System.Numerics.Vector<double>.Count;
                        int ch = 0;
                        int offA = baseOut; // top-left
                        int offB = baseOut + inC; // top-right
                        int offC = baseOut + outW * inC; // bottom-left
                        int offD = offC + inC; // bottom-right
                        for (; ch <= inC - vecSize; ch += vecSize)
                        {
                            var va = new System.Numerics.Vector<double>(gradOutput, offA + ch);
                            var vb = new System.Numerics.Vector<double>(gradOutput, offB + ch);
                            var vc = new System.Numerics.Vector<double>(gradOutput, offC + ch);
                            var vd = new System.Numerics.Vector<double>(gradOutput, offD + ch);
                            var sum = va + vb + vc + vd;
                            var vDst = new System.Numerics.Vector<double>(dst, baseIn + ch);
                            vDst += sum;
                            vDst.CopyTo(dst, baseIn + ch);
                        }
                        for (; ch < inC; ch++)
                        {
                            double s = gradOutput[offA + ch] + gradOutput[offB + ch] + gradOutput[offC + ch] + gradOutput[offD + ch];
                            dst[baseIn + ch] += s;
                        }
                    }
                    else
                    {
                        for (int ch = 0; ch < inC; ch++)
                        {
                            double sum = gradOutput[baseOut + ch];
                            if (outX + 1 < outW) sum += gradOutput[baseOut + inC + ch];
                            if (outY + 1 < outH) sum += gradOutput[baseOut + outW * inC + ch];
                            if (outY + 1 < outH && outX + 1 < outW) sum += gradOutput[baseOut + outW * inC + inC + ch];
                            dst[baseIn + ch] += sum;
                        }
                    }
                }
            });
            input.GradFn?.Backward(input.Grad);
        }
    }
}
