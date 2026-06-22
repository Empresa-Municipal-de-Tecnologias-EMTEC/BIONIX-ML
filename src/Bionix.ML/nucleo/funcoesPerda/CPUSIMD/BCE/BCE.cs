using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.CPUSIMD.BCE
{
    public static class BCE
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
                double eps = 1e-12;
                double sum = 0.0;
                var pData = p.Data;
                var tData = t.Data;
                object sumLock = new object();

                System.Threading.Tasks.Parallel.For(0, p.Size,
                    () => 0.0,
                    (i, state, local) =>
                    {
                        double prob = pData[i];
                        if (prob < eps) prob = eps;
                        else if (prob > 1.0 - eps) prob = 1.0 - eps;
                        double ti = tData[i];
                        local += -(ti * Math.Log(prob) + (1.0 - ti) * Math.Log(1.0 - prob));
                        return local;
                    },
                    local => { lock (sumLock) sum += local; }
                );

                outT[0] = sum / p.Size;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.BCEFunction(p, t, outT);
                return outT;
            }
            throw new NotImplementedException("BCE not implemented for this ComputacaoContexto");
        }
    }
}
