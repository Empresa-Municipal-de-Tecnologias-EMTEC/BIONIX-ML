using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda
{
    public static class FabricaFuncoesPerdaCPU
    {
        // Each method returns a Func that consumes (predTensor, targetTensor) -> scalar Tensor
        public static Func<Tensor, Tensor, Tensor> CriarMSE(ComputacaoContexto ctx)
        {
            return (pred, target) =>
            {
                if (!(pred is TensorCPU p) || !(target is TensorCPU t)) throw new ArgumentException("MSE wrapper expects TensorCPU tensors");
                var pa = p.ToArray();
                var ta = t.ToArray();
                double val = Bionix.ML.nucleo.funcoesPerda.MSE.MSE.Loss(pa, ta);
                var fabrica = new FabricaTensor(ctx);
                var outT = fabrica.Criar(1);
                outT[0] = val;
                return outT;
            };
        }

        public static Func<Tensor, Tensor, Tensor> CriarFocal(ComputacaoContexto ctx)
        {
            return (pred, target) => Bionix.ML.nucleo.funcoesPerda.Focal.Focal.Loss(ctx, pred, target);
        }

        public static Func<Tensor, Tensor, Tensor> CriarSmoothL1(ComputacaoContexto ctx)
        {
            return (pred, target) => Bionix.ML.nucleo.funcoesPerda.SmoothL1.SmoothL1.Loss(ctx, pred, target);
        }

        public static Func<Tensor, Tensor, Tensor> CriarBCE(ComputacaoContexto ctx)
        {
            return (pred, target) =>
            {
                if (!(pred is TensorCPU p) || !(target is TensorCPU t)) throw new ArgumentException("BCE wrapper expects TensorCPU tensors");
                var fabrica = new FabricaTensor(ctx);
                var outT = fabrica.Criar(1) as TensorCPU;
                // compute scalar loss value for reporting
                var pa = p.ToArray();
                var ta = t.ToArray();
                double val = Bionix.ML.nucleo.funcoesPerda.BCE.BCE.Loss(pa, ta);
                outT[0] = val;
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPU.BCEFunction(p, t, outT);
                return outT;
            };
        }
    }
}
