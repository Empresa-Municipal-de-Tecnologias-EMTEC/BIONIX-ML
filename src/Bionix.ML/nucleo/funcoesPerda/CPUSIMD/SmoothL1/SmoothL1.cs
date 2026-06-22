using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.CPUSIMD.SmoothL1
{
    public static class SmoothL1
    {
        public static Tensor Loss(ComputacaoContexto ctx, Tensor preds, Tensor targets)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (preds == null) throw new ArgumentNullException(nameof(preds));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            if (ctx is ComputacaoCPUSIMDContexto simd)
            {
                var p = preds as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD preds for CPUSIMD context");
                var t = targets as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD targets for CPUSIMD context");
                if (p.Size != t.Size) throw new ArgumentException("Shape mismatch");

                var fabrica = new FabricaTensor(simd);
                var outT = fabrica.Criar(1) as TensorCPUSIMD;
                double sum = 0.0;
                int N = p.Size;
                if (System.Numerics.Vector.IsHardwareAccelerated)
                {
                    int vecSize = System.Numerics.Vector<double>.Count;
                    int i = 0;
                    var vHalf = new System.Numerics.Vector<double>(0.5);
                    var vOne = new System.Numerics.Vector<double>(1.0);
                    var vSum = System.Numerics.Vector<double>.Zero;
                    for (; i <= N - vecSize; i += vecSize)
                    {
                        var vP = new System.Numerics.Vector<double>(p.Data, i);
                        var vT = new System.Numerics.Vector<double>(t.Data, i);
                        var vD = vP - vT;
                        var vA = System.Numerics.Vector.Abs(vD);
                        var mask = System.Numerics.Vector.LessThan(vA, vOne);
                        var vSq = vD * vD * vHalf; // 0.5 * d * d
                        var vAlt = vA - vHalf;
                        var vRes = System.Numerics.Vector.ConditionalSelect(mask, vSq, vAlt);
                        vSum += vRes;
                    }
                    for (int k = 0; k < System.Numerics.Vector<double>.Count; k++) sum += vSum[k];
                    for (int r = (N / vecSize) * vecSize; r < N; r++)
                    {
                        double d = p[r] - t[r];
                        double a = Math.Abs(d);
                        if (a < 1.0) sum += 0.5 * d * d;
                        else sum += a - 0.5;
                    }
                }
                else
                {
                    for (int i = 0; i < p.Size; i++)
                    {
                        double d = p[i] - t[i];
                        double a = Math.Abs(d);
                        if (a < 1.0) sum += 0.5 * d * d;
                        else sum += a - 0.5;
                    }
                }
                outT[0] = sum / p.Size;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.SmoothL1Function(p, t, outT);
                return outT;
            }
            throw new NotImplementedException("SmoothL1Loss not implemented for this ComputacaoContexto");
        }
    }
}
