using System;
using Bionix.ML.nucleo.funcoesAtivacao.Sigmoid;
using Bionix.ML.computacao;

namespace Bionix.ML.nucleo.funcoesAtivacao
{
    public static class FabricaFuncoesAtivacao
    {
        public static Func<double,double> Criar(ComputacaoContexto ctx)
        {
            // Prefer CPUSIMD implementations when available
            if (ctx is ComputacaoCPUSIMDContexto) return Bionix.ML.nucleo.funcoesAtivacao.CPUSIMD.Sigmoid.Sigmoid.Forward;
            return Bionix.ML.nucleo.funcoesAtivacao.Sigmoid.Sigmoid.Forward;
        }

        public static Func<double,double> Criar(string nome, ComputacaoContexto ctx)
        {
            if (string.Equals(nome, "relu", StringComparison.OrdinalIgnoreCase))
            {
                if (ctx is ComputacaoCPUSIMDContexto) return Bionix.ML.nucleo.funcoesAtivacao.CPUSIMD.ReLU.ReLU.Forward;
                return ReLU.ReLU.Forward;
            }
            return Criar(ctx);
        }
    }
}
