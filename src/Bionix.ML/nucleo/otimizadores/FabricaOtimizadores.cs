using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.otimizadores.CPUSIMD;

namespace Bionix.ML.nucleo.otimizadores
{
    public static class FabricaOtimizadores
    {
        public static IStatefulOptimizer CriarStatefulSGD(IEnumerable<Tensor> parameters, ComputacaoContexto ctx, double lr = 1e-3, double momentum = 0.9)
        {
            if (ctx is ComputacaoCPUSIMDContexto simdCtx && simdCtx.IsAvailable)
                return new CPUSIMD.StatefulSGDCPUSIMD(parameters, simdCtx, lr, momentum);
            return new StatefulSGD(parameters, lr, momentum);
        }
    }
}
