using System;
using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.otimizadores
{
    public static class SGD
    {
        public static void Step(IEnumerable<Tensor> parameters, double lr)
        {
            foreach (var p in parameters)
            {
                if (p == null) continue;
                if (p.Grad == null) continue;
                for (int i = 0; i < p.Size; i++)
                {
                    p[i] = p[i] - lr * p.Grad[i];
                }
                // zero grads after step
                p.ZeroGrad();
            }
        }
    }
}
