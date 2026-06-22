using System;

namespace Bionix.ML.nucleo.funcoesAtivacao.ReLU
{
    public static class ReLU
    {
        public static double Forward(double x) => x > 0.0 ? x : 0.0;
    }
}
