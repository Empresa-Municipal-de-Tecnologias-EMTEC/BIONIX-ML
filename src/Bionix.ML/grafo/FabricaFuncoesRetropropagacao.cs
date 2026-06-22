using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo
{
    public static class FabricaFuncoesRetropropagacao
    {
        public static IFuncaoRetropropagacao CriarAdd(ComputacaoContexto ctx, Tensor a, Tensor b, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.AddFunction(a, b, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.AddFunction(a, b, outT);
            throw new NotImplementedException($"AddFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarSub(ComputacaoContexto ctx, Tensor a, Tensor b, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.SubFunction(a, b, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.SubFunction(a, b, outT);
            throw new NotImplementedException($"SubFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarMulScalar(ComputacaoContexto ctx, Tensor a, double s, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.MulScalarFunction(a, s, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.MulScalarFunction(a, s, outT);
            throw new NotImplementedException($"MulScalarFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarMatMul(ComputacaoContexto ctx, Tensor A, Tensor B, Tensor Out)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.MatMulFunction(A, B, Out);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.MatMulFunction(A, B, Out);
            throw new NotImplementedException($"MatMulFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarTranspose(ComputacaoContexto ctx, Tensor src, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.TransposeFunction(src, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.TransposeFunction(src, outT);
            throw new NotImplementedException($"TransposeFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarIdentity(ComputacaoContexto ctx, Tensor src, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.IdentityFunction(src, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.IdentityFunction(src, outT);
            throw new NotImplementedException($"IdentityFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarSumSquares(ComputacaoContexto ctx, Tensor src, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.SumSquaresFunction(src as TensorCPU, outT as TensorCPU);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.SumSquaresFunction(src as TensorCPUSIMD, outT as TensorCPUSIMD );
            throw new NotImplementedException($"SumSquaresFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarConv(ComputacaoContexto ctx, Tensor input, Tensor weight, Tensor bias, Tensor output, int ksize)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.ConvFunction(input, weight, bias, output, ksize);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.ConvFunction(input, weight, bias, output, ksize);
            throw new NotImplementedException($"ConvFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarBN(ComputacaoContexto ctx, Tensor input, Tensor gamma, Tensor beta, int h, int w, int c, double eps = 1e-5)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.BNFunction(input, gamma, beta, h, w, c, eps);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.BNFunction(input, gamma, beta, h, w, c, eps);
            throw new NotImplementedException($"BNFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarFocalLoss(ComputacaoContexto ctx, Tensor logits, Tensor targets, Tensor outT, double alpha, double gamma)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.FocalLossFunction(logits, targets, outT, alpha, gamma);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.FocalLossFunction(logits, targets, outT, alpha, gamma);
            throw new NotImplementedException($"FocalLossFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarSmoothL1(ComputacaoContexto ctx, Tensor preds, Tensor targets, Tensor outT)
        {
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.SmoothL1Function(preds, targets, outT);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.SmoothL1Function(preds, targets, outT);
            throw new NotImplementedException($"SmoothL1Function not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarDownsample(ComputacaoContexto ctx, Tensor input, Tensor output)
        {
            // Currently no CPUSIMD specialized Downsample; fall back to CPU implementation
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.DownsampleFunction(input, output);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.DownsampleFunction(input, output);
            throw new NotImplementedException($"DownsampleFunction not implemented for context {ctx.GetType().Name}");
        }

        public static IFuncaoRetropropagacao CriarConcat(ComputacaoContexto ctx, Tensor[] parts, Tensor output)
        {
            // Currently no CPUSIMD specialized Concat; fall back to CPU implementation
            if (ctx is ComputacaoCPUContexto) return new Bionix.ML.grafo.CPU.ConcatFunction(parts, output);
            if (ctx is ComputacaoCPUSIMDContexto) return new Bionix.ML.grafo.CPUSIMD.ConcatFunction(parts, output);
            throw new NotImplementedException($"ConcatFunction not implemented for context {ctx.GetType().Name}");
        }
    }
}
