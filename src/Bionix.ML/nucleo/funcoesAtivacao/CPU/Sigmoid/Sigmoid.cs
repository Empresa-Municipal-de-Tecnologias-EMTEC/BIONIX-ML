using System;

namespace Bionix.ML.nucleo.funcoesAtivacao.Sigmoid
{
    public static class Sigmoid
    {
        public static double Forward(double x) => 1.0 / (1.0 + Math.Exp(-x));
    }
}
