using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.CPUSIMD.Focal
{
    public static class Focal
    {
        public static Tensor Loss(ComputacaoContexto ctx, Tensor logits, Tensor targets, double alpha = 0.25, double gamma = 2.0)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            if (ctx is ComputacaoCPUSIMDContexto simd)
            {
                var l = logits as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD logits for CPUSIMD context");
                var t = targets as TensorCPUSIMD ?? throw new ArgumentException("Expected TensorCPUSIMD targets for CPUSIMD context");
                if (l.Size != t.Size) throw new ArgumentException("Shape mismatch");

                var fabrica = new FabricaTensor(simd);
                var outT = fabrica.Criar(1) as TensorCPUSIMD;
                double eps = 1e-12;
                double sum = 0.0;
                // Parallelize element work (math ops remain scalar per element)
                object sumLock = new object();
                System.Threading.Tasks.Parallel.For(0, l.Size,
                    () => 0.0,
                    (i, state, local) =>
                    {
                        double z = l[i];
                        double p = 1.0 / (1.0 + Math.Exp(-z));
                        double ti = t[i] >= 0.5 ? 1.0 : 0.0;
                        if (ti >= 0.5)
                        {
                            double one_minus_p = 1.0 - p;
                            local += -alpha * Math.Pow(one_minus_p, gamma) * Math.Log(p + eps);
                        }
                        else
                        {
                            local += -(1.0 - alpha) * Math.Pow(p, gamma) * Math.Log(1.0 - p + eps);
                        }
                        return local;
                    },
                    local => { lock (sumLock) sum += local; }
                );
                outT[0] = sum / l.Size;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.FocalLossFunction(l, t, outT, alpha, gamma);
                return outT;
            }
            throw new NotImplementedException("FocalLoss not implemented for this ComputacaoContexto");
        }
    }
}
