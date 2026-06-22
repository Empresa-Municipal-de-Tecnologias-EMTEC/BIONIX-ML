using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class BCEFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD preds;
        private readonly TensorCPUSIMD targets;
        private readonly TensorCPUSIMD outT;
        private readonly int N;

        public BCEFunction(Tensor preds_, Tensor targets_, Tensor out_)
        {
            preds = preds_ as TensorCPUSIMD ?? throw new ArgumentException();
            targets = targets_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
            N = preds.Size;
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double eps = 1e-12;
            if (!preds.RequiresGrad) return;
            var src = preds.Data;
            var tgt = targets.Data;
            var dst = preds.Grad;
            int i = 0;
            if (System.Numerics.Vector.IsHardwareAccelerated)
            {
                int vecSize = System.Numerics.Vector<double>.Count;
                var vEps = new System.Numerics.Vector<double>(eps);
                var vOne = new System.Numerics.Vector<double>(1.0);
                var vG = new System.Numerics.Vector<double>(g / (double)N);
                for (; i <= N - vecSize; i += vecSize)
                {
                    var vp = new System.Numerics.Vector<double>(src, i);
                    // clamp p to [eps, 1-eps]
                    vp = System.Numerics.Vector.Min(System.Numerics.Vector.Max(vp, vEps), vOne - vEps);
                    var vt = new System.Numerics.Vector<double>(tgt, i);
                    var oneMinusT = vOne - vt;
                    var oneMinusP = vOne - vp;
                    var term = (-(vt / vp) + (oneMinusT / oneMinusP));
                    var vGrad = vG * term;
                    var vDst = new System.Numerics.Vector<double>(dst, i);
                    vDst += vGrad;
                    vDst.CopyTo(dst, i);
                }
            }
            for (; i < N; i++)
            {
                double p = Math.Max(eps, Math.Min(1.0 - eps, src[i]));
                double t = tgt[i];
                double grad = g * (-(t / p) + (1.0 - t) / (1.0 - p)) / N;
                dst[i] += grad;
            }
            // optional detailed debug: print sum of abs(pred grads)
            var dbg = Environment.GetEnvironmentVariable("DEBUG_GRADS_DETAIL");
            if (!string.IsNullOrEmpty(dbg) && (dbg == "1" || dbg.Equals("true", StringComparison.OrdinalIgnoreCase)))
            {
                double s = 0.0; for (int ii = 0; ii < dst.Length; ii++) s += Math.Abs(dst[ii]);
                Console.WriteLine($"[BCEFunction] preds.Grad abs-sum={s:E6}");
            }
            preds.GradFn?.Backward(preds.Grad);
        }
    }
}
