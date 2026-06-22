using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda.SmoothL1
{
    public static class SmoothL1
    {
        // Tensor-level Smooth L1 loss: delegates to grafo.Perdas
        public static Tensor Loss(ComputacaoContexto ctx, Tensor preds, Tensor targets)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (preds == null) throw new ArgumentNullException(nameof(preds));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            if (ctx is ComputacaoCPUContexto)
            {
                var p = preds as TensorCPU ?? throw new ArgumentException("Expected TensorCPU preds for CPU context");
                var t = targets as TensorCPU ?? throw new ArgumentException("Expected TensorCPU targets for CPU context");
                if (p.Size != t.Size) throw new ArgumentException("Shape mismatch");

                var fabrica = new FabricaTensor(new ComputacaoCPUContexto());
                var outT = fabrica.Criar(1) as TensorCPU;
                double sum = 0.0;
                for (int i = 0; i < p.Size; i++)
                {
                    double diff = p[i] - t[i];
                    double abs = Math.Abs(diff);
                    if (abs < 1.0) sum += 0.5 * diff * diff;
                    else sum += abs - 0.5;
                }
                outT[0] = sum / p.Size;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPU.SmoothL1Function(p, t, outT);
                return outT;
            }
            throw new NotImplementedException("SmoothL1Loss not implemented for this ComputacaoContexto");
        }
    }
}
