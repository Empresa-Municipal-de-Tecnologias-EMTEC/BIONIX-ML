using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class L2NormFunction : IFuncaoRetropropagacao
    {
        private readonly Tensor src;
        private readonly Tensor outT;

        public L2NormFunction(Tensor src_, Tensor out_)
        {
            src = src_ ?? throw new ArgumentNullException(nameof(src_));
            outT = out_ ?? throw new ArgumentNullException(nameof(out_));
        }

        public void Backward(double[] gradOutput)
        {
            if (!src.RequiresGrad) return;
            int rows = src.Shape != null && src.Shape.Length >= 2 ? src.Shape[0] : 1;
            int cols = src.Size / Math.Max(1, rows);
            const double eps = 1e-12;

            // Ensure gradient buffer exists
            if (src.Grad == null) src.SetGrad(new double[src.Size]);

            for (int r = 0; r < rows; r++)
            {
                int baseIdx = r * cols;
                double sumSq = 0.0;
                for (int j = 0; j < cols; j++) sumSq += src[baseIdx + j] * src[baseIdx + j];
                double denom = sumSq + eps;
                double norm = Math.Sqrt(denom);

                double dot = 0.0;
                for (int j = 0; j < cols; j++) dot += src[baseIdx + j] * gradOutput[baseIdx + j];

                for (int j = 0; j < cols; j++)
                {
                    double g = (gradOutput[baseIdx + j] - src[baseIdx + j] * dot / denom) / norm;
                    src.Grad[baseIdx + j] += g;
                }
            }

            src.GradFn?.Backward(src.Grad);
        }
    }
}
