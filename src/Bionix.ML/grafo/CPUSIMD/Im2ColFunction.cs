using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    // Backprop for im2col copy: maps gradients from the unfolded Xcol back to the
    // source spatial tensor. This is a lightweight GradFn attached to the
    // Xcol tensor created in the model forward pass so that MatMul->... backprop
    // reaches the original input tensor.
    public class Im2ColFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src;
        private readonly TensorCPUSIMD outT;
        private readonly int inW, inC, kh, kw, outH, outW, patchSize;

        public Im2ColFunction(Tensor src_, Tensor out_, int inWidth, int inChannels, int kernelH, int kernelW, int outH_, int outW_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException("src must be TensorCPUSIMD");
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException("out must be TensorCPUSIMD");
            inW = inWidth; inC = inChannels; kh = kernelH; kw = kernelW; outH = outH_; outW = outW_; patchSize = kh * kw * inC;
        }

        public void Backward(double[] gradOutput)
        {
            // gradOutput corresponds to Xcol.Grad (shape [outH*outW, patchSize]) flattened
            var dst = src.Grad;
            // iterate over output spatial locations and scatter back into src
            for (int oy = 0; oy < outH; oy++)
            {
                for (int ox = 0; ox < outW; ox++)
                {
                    int row = oy * outW + ox;
                    int p = 0;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        for (int kx = 0; kx < kw; kx++)
                        {
                            for (int ic = 0; ic < inC; ic++)
                            {
                                int inIdx = ((oy + ky) * inW + (ox + kx)) * inC + ic;
                                dst[inIdx] += gradOutput[row * patchSize + p];
                                p++;
                            }
                        }
                    }
                }
            }
            src.GradFn?.Backward(src.Grad);
        }
    }
}
