using System;
using System.Numerics;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class MSEFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD preds;
        private readonly TensorCPUSIMD targets;
        private readonly TensorCPUSIMD outT;
        private readonly int N;

        public MSEFunction(Tensor preds_, Tensor targets_, Tensor out_)
        {
            preds = preds_ as TensorCPUSIMD ?? throw new ArgumentException();
            targets = targets_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
            N = preds.Size;
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            if (preds.RequiresGrad)
            {
                double scale = 2.0 * g / N;
                var pData = preds.Data;
                var tData = targets.Data;
                var dst = preds.GradArray;
                int n = N;
                if (Vector.IsHardwareAccelerated)
                {
                    int vecSize = Vector<double>.Count;
                    var vScale = new Vector<double>(scale);
                    int i = 0;
                    for (; i <= n - vecSize; i += vecSize)
                    {
                        var vP = new Vector<double>(pData, i);
                        var vT = new Vector<double>(tData, i);
                        var vDiff = vP - vT;
                        var vUpdate = vDiff * vScale;
                        var vDst = new Vector<double>(dst, i);
                        vDst += vUpdate;
                        vDst.CopyTo(dst, i);
                    }
                    for (; i < n; i++) dst[i] += (pData[i] - tData[i]) * scale;
                }
                else
                {
                    for (int i = 0; i < n; i++) dst[i] += (preds[i] - targets[i]) * scale;
                }
                preds.GradFn?.Backward(preds.Grad);
            }
        }
    }
}
