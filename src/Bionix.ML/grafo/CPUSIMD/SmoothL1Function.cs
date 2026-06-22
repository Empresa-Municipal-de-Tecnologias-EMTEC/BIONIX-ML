using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class SmoothL1Function : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD preds;
        private readonly TensorCPUSIMD targets;
        private readonly TensorCPUSIMD outT;

        public SmoothL1Function(Tensor preds_, Tensor targets_, Tensor out_)
        {
            preds = preds_ as TensorCPUSIMD ?? throw new ArgumentException();
            targets = targets_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            int N = preds.Size;
            if (!preds.RequiresGrad) return;
            var src = preds.Data;
            var tgt = targets.Data;
            var dst = preds.Grad;
            int i = 0;
            double scale = g / (double)N;
            if (System.Numerics.Vector.IsHardwareAccelerated)
            {
                int vecSize = System.Numerics.Vector<double>.Count;
                var vScale = new System.Numerics.Vector<double>(scale);
                var vOne = new System.Numerics.Vector<double>(1.0);
                for (; i <= N - vecSize; i += vecSize)
                {
                    var vPred = new System.Numerics.Vector<double>(src, i);
                    var vTgt = new System.Numerics.Vector<double>(tgt, i);
                    var vDiff = vPred - vTgt;
                    var vAbs = System.Numerics.Vector.Abs(vDiff);
                    var vDenom = System.Numerics.Vector.Max(vAbs, vOne);
                    var vGrad = vDiff / vDenom; // diff / max(abs,1) -> gives diff when abs<1, sign(diff) when abs>=1
                    vGrad *= vScale;
                    var vDst = new System.Numerics.Vector<double>(dst, i);
                    vDst += vGrad;
                    vDst.CopyTo(dst, i);
                }
            }
            for (; i < N; i++)
            {
                double diff = src[i] - tgt[i];
                double abs = Math.Abs(diff);
                double grad = (abs < 1.0) ? diff : Math.Sign(diff);
                grad = scale * grad;
                dst[i] += grad;
            }
            preds.GradFn?.Backward(preds.Grad);
        }
    }
}
