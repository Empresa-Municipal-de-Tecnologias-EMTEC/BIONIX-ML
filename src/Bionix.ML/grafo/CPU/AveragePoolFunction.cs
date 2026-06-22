using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class AveragePoolFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU src;
        private readonly TensorCPU outT;
        private readonly int outH, outW, pH, pW, pooledH, pooledW;

        public AveragePoolFunction(Tensor src_, Tensor out_, int outH_, int outW_, int pH_, int pW_)
        {
            src = src_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
            outH = outH_; outW = outW_; pH = pH_; pW = pW_;
            pooledH = outH / pH; pooledW = outW / pW;
        }

        public void Backward(double[] gradOutput)
        {
            // src: [outH*outW, outC] flattened row-major with channel stride
            int pooledN = pooledH * pooledW;
            int outC = src.Shape[1];
            var srcGrad = src.Grad;

            double scale = 1.0 / (pH * pW);
            for (int prow = 0; prow < pooledN; prow++)
            {
                int py = prow / pooledW;
                int px = prow % pooledW;
                for (int dy = 0; dy < pH; dy++)
                {
                    for (int dx = 0; dx < pW; dx++)
                    {
                        int oy = py * pH + dy;
                        int ox = px * pW + dx;
                        int srcIdx = oy * outW + ox;
                        int srcBase = srcIdx * outC;
                        int gBase = prow * outC;
                        for (int c = 0; c < outC; c++)
                        {
                            srcGrad[srcBase + c] += gradOutput[gBase + c] * scale;
                        }
                    }
                }
            }
            src.GradFn?.Backward(srcGrad);
        }
    }
}
