using System;
using Bionix.ML.nucleo.derivadas.DerivadaSigmoide;
using Bionix.ML.computacao;

namespace Bionix.ML.nucleo.derivadas
{
    public static class FabricaDerivadas
    {
        public static Func<double,double> Criar(ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return Bionix.ML.nucleo.derivadas.DerivadaSigmoide.DerivadaSigmoide.FromActivated;
            return DerivadaSigmoide.DerivadaSigmoide.FromActivated;
        }
    }
}
