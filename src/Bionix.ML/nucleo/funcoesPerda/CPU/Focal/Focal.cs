using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.Focal
{
    public static class Focal
    {
        // Tensor-level focal loss: delegates to grafo.Perdas for context-aware implementation
        public static Tensor Loss(ComputacaoContexto ctx, Tensor logits, Tensor targets, double alpha = 0.25, double gamma = 2.0)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            // Generic implementation using base Tensor API so it works for CPU and CPUSIMD tensors.
            if (logits.Size != targets.Size) throw new ArgumentException("Shape mismatch");
            var fabricaOut = new FabricaTensor(ctx);
            var outT = fabricaOut.Criar(1);
            double eps = 1e-12;
            double sum = 0.0;
            for (int i = 0; i < logits.Size; i++)
            {
                double z = logits[i];
                double p = 1.0 / (1.0 + Math.Exp(-z));
                double ti = targets[i] >= 0.5 ? 1.0 : 0.0;
                if (ti >= 0.5)
                {
                    double one_minus_p = 1.0 - p;
                    double val = -alpha * Math.Pow(one_minus_p, gamma) * Math.Log(p + eps);
                    sum += val;
                }
                else
                {
                    double val = -(1.0 - alpha) * Math.Pow(p, gamma) * Math.Log(1.0 - p + eps);
                    sum += val;
                }
            }
            outT[0] = sum / logits.Size;
            outT.RequiresGrad = true;
            outT.GradFn = Bionix.ML.grafo.FabricaFuncoesRetropropagacao.CriarFocalLoss(ctx, logits, targets, outT, alpha, gamma);
            return outT;
        }
    }
}
