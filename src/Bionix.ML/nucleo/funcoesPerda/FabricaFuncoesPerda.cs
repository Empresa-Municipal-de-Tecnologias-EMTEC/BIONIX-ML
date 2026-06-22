using System;
using Bionix.ML.nucleo.funcoesPerda.MSE;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda
{
    public static class FabricaFuncoesPerda
    {
        // Tensor-level creators that take only the ComputacaoContexto and return a (pred,target)->Tensor function
        public static Func<Tensor, Tensor, Tensor> CriarMSE(ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return (pred, target) => Bionix.ML.nucleo.funcoesPerda.CPUSIMD.MSE.MSE.Loss(ctx, pred, target);
            if (ctx is ComputacaoCPUContexto) return FabricaFuncoesPerdaCPU.CriarMSE(ctx);
            throw new NotImplementedException("MSE not implemented for this ComputacaoContexto");
        }

        public static Func<Tensor, Tensor, Tensor> CriarFocal(ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return (pred, target) => Bionix.ML.nucleo.funcoesPerda.CPUSIMD.Focal.Focal.Loss(ctx, pred, target);
            if (ctx is ComputacaoCPUContexto) return FabricaFuncoesPerdaCPU.CriarFocal(ctx);
            throw new NotImplementedException("Focal not implemented for this ComputacaoContexto");
        }

        public static Func<Tensor, Tensor, Tensor> CriarSmoothL1(ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return (pred, target) => Bionix.ML.nucleo.funcoesPerda.CPUSIMD.SmoothL1.SmoothL1.Loss(ctx, pred, target);
            if (ctx is ComputacaoCPUContexto) return FabricaFuncoesPerdaCPU.CriarSmoothL1(ctx);
            throw new NotImplementedException("SmoothL1 not implemented for this ComputacaoContexto");
        }

        public static Func<Tensor, Tensor, Tensor> CriarBCE(ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return (pred, target) => Bionix.ML.nucleo.funcoesPerda.CPUSIMD.BCE.BCE.Loss(ctx, pred, target);
            if (ctx is ComputacaoCPUContexto) return FabricaFuncoesPerdaCPU.CriarBCE(ctx);
            throw new NotImplementedException("BCE not implemented for this ComputacaoContexto");
        }

        // Legacy array-based factory remains for compatibility: these will convert arrays to tensors and call the tensor creators when possible
        public static Func<double[], double[], double> Criar(ComputacaoContexto ctx)
        {
            return Criar("mse", ctx);
        }

        public static Func<double[], double[], double> Criar(string nome, ComputacaoContexto ctx)
        {
            if (string.IsNullOrWhiteSpace(nome)) return Criar(ctx);
            if (ctx is ComputacaoCPUContexto)
            {
                if (string.Equals(nome, "focal", StringComparison.OrdinalIgnoreCase))
                {
                    var fn = CriarFocal(ctx);
                    return (pred, target) =>
                    {
                        var fabrica = new FabricaTensor(ctx);
                        var p = fabrica.FromArray(new int[] { pred.Length }, pred);
                        var t = fabrica.FromArray(new int[] { target.Length }, target);
                        var outT = fn(p, t);
                        return outT[0];
                    };
                }
                if (string.Equals(nome, "smoothl1", StringComparison.OrdinalIgnoreCase) || string.Equals(nome, "smooth_l1", StringComparison.OrdinalIgnoreCase))
                {
                    var fn = CriarSmoothL1(ctx);
                    return (pred, target) =>
                    {
                        var fabrica = new FabricaTensor(ctx);
                        var p = fabrica.FromArray(new int[] { pred.Length }, pred);
                        var t = fabrica.FromArray(new int[] { target.Length }, target);
                        var outT = fn(p, t);
                        return outT[0];
                    };
                }
                if (string.Equals(nome, "bce", StringComparison.OrdinalIgnoreCase))
                {
                    var fn = CriarBCE(ctx);
                    return (pred, target) =>
                    {
                        var fabrica = new FabricaTensor(ctx);
                        var p = fabrica.FromArray(new int[] { pred.Length }, pred);
                        var t = fabrica.FromArray(new int[] { target.Length }, target);
                        var outT = fn(p, t);
                        return outT[0];
                    };
                }
            }
            // fallback array implementations
            if (string.Equals(nome, "bce", StringComparison.OrdinalIgnoreCase)) return BCE.BCE.Loss;
            return MSE.MSE.Loss;
        }
    }
}
