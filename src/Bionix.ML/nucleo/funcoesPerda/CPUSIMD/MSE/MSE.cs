using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.CPUSIMD.MSE
{
    public static class MSE
    {
        public static Tensor Loss(ComputacaoContexto ctx, Tensor predicted, Tensor target)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (predicted == null) throw new ArgumentNullException(nameof(predicted));
            if (target == null) throw new ArgumentNullException(nameof(target));

            if (ctx is ComputacaoCPUSIMDContexto simd)
            {
                var p = predicted as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD predicted for CPUSIMD context");
                var t = target as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD target for CPUSIMD context");
                if (p.Size != t.Size) throw new ArgumentException("Shape mismatch");

                var fabrica = new FabricaTensor(simd);
                var outT = fabrica.Criar(1) as TensorCPUSIMD;
                double sum = 0.0;
                var pd = p.Data;
                var td = t.Data;
                int n = p.Size;
                if (System.Numerics.Vector.IsHardwareAccelerated)
                {
                    int vecSize = System.Numerics.Vector<double>.Count;
                    var acc = new System.Numerics.Vector<double>(0.0);
                    int i = 0;
                    for (; i <= n - vecSize; i += vecSize)
                    {
                        var vp = new System.Numerics.Vector<double>(pd, i);
                        var vt = new System.Numerics.Vector<double>(td, i);
                        var d = vp - vt;
                        acc += d * d;
                    }
                    var tmp = new double[System.Numerics.Vector<double>.Count];
                    acc.CopyTo(tmp);
                    for (int k = 0; k < tmp.Length; k++) sum += tmp[k];
                    for (; i < n; i++) { double d = pd[i] - td[i]; sum += d * d; }
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        double d = pd[i] - td[i];
                        sum += d * d;
                    }
                }
                outT[0] = sum / n;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.MSEFunction(p, t, outT);
                return outT;
            }
            throw new NotImplementedException("MSE not implemented for this ComputacaoContexto");
        }
    }
}
