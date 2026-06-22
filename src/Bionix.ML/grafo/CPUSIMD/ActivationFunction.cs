using System;
using System.Numerics;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class ActivationFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src, outT;
        public ActivationFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                int n = src.Size;
                var srcData = src.Data;
                var srcGrad = src.GradArray;
                if (Vector.IsHardwareAccelerated)
                {
                    int vecSize = Vector<double>.Count;
                    int i = 0;
                    var vZero = Vector<double>.Zero;
                    for (; i <= n - vecSize; i += vecSize)
                    {
                        var vSrc = new Vector<double>(srcData, i);
                        var vGradOut = new Vector<double>(gradOutput, i);
                        var mask = Vector.GreaterThan(vSrc, vZero);
                        var vUpdate = Vector.ConditionalSelect(mask, vGradOut, vZero);
                        var vDst = new Vector<double>(srcGrad, i);
                        vDst += vUpdate;
                        vDst.CopyTo(srcGrad, i);
                    }
                    for (; i < n; i++)
                    {
                        double grad = srcData[i] > 0 ? gradOutput[i] : 0.0;
                        srcGrad[i] += grad;
                    }
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        double grad = src[i] > 0 ? gradOutput[i] : 0.0;
                        src.Grad[i] += grad;
                    }
                }
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
