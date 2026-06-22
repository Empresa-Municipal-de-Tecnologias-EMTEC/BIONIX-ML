using System;
using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.nucleo.otimizadores.CPUSIMD
{
    public static class SGD
    {
        public static void Step(IEnumerable<Tensor> parameters, double lr)
        {
            if (parameters == null) return;
            foreach (var p in parameters)
            {
                if (p == null) continue;
                if (p.Grad == null) continue;
                if (p is TensorCPUSIMD tp)
                {
                    var data = tp.Data;
                    var grad = tp.GradArray;
                    int n = tp.Size;
                    if (System.Numerics.Vector.IsHardwareAccelerated)
                    {
                        int vecSize = System.Numerics.Vector<double>.Count;
                        int i = 0;
                        var vLr = new System.Numerics.Vector<double>(lr);
                        for (; i <= n - vecSize; i += vecSize)
                        {
                            var vData = new System.Numerics.Vector<double>(data, i);
                            var vGrad = new System.Numerics.Vector<double>(grad, i);
                            var vRes = vData - vGrad * vLr;
                            vRes.CopyTo(data, i);
                        }
                        for (; i < n; i++) data[i] = data[i] - lr * grad[i];
                    }
                    else
                    {
                        for (int i = 0; i < n; i++) data[i] = data[i] - lr * grad[i];
                    }
                    tp.ZeroGrad();
                }
                else
                {
                    throw new ArgumentException("SGD CPUSIMD expects TensorCPUSIMD parameters when running in CPUSIMD context");
                }
            }
        }
    }
}
