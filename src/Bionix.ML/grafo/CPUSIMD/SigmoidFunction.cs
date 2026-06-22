using System;
using System.Numerics;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class SigmoidFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src;
        private readonly TensorCPUSIMD outT;
        private readonly int N;

        public SigmoidFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
            N = src.Size;
        }

        public void Backward(double[] gradOutput)
        {
            if (!src.RequiresGrad) return;
            var dbg = Environment.GetEnvironmentVariable("DEBUG_GRADS_DETAIL");
            if (!string.IsNullOrEmpty(dbg) && (dbg == "1" || dbg.Equals("true", StringComparison.OrdinalIgnoreCase)))
            {
                double sumIn = 0.0; for (int ii = 0; ii < gradOutput.Length; ii++) sumIn += Math.Abs(gradOutput[ii]);
                Console.WriteLine($"[SigmoidFunction] called src.RequiresGrad={src.RequiresGrad} gradOut-abs={sumIn:E6}");
            }
            var srcGrad = src.GradArray;
            var outData = outT.Data; // sigmoid outputs
            int n = N;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                int i = 0;
                for (; i <= n - vecSize; i += vecSize)
                {
                    var vOut = new Vector<double>(outData, i);
                    var vOne = new Vector<double>(1.0);
                    var vOneMinusOut = vOne - vOut;
                    var vProd = vOut * vOneMinusOut; // p*(1-p)
                    var vGradOut = new Vector<double>(gradOutput, i);
                    var vUpdate = vGradOut * vProd;
                    var vDst = new Vector<double>(srcGrad, i);
                    vDst += vUpdate;
                    vDst.CopyTo(srcGrad, i);
                }
                for (int j = n - (n % vecSize); j < n; j++)
                {
                    double p = outData[j];
                    srcGrad[j] += gradOutput[j] * p * (1.0 - p);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    double p = outData[i];
                    srcGrad[i] += gradOutput[i] * p * (1.0 - p);
                }
            }
            src.GradFn?.Backward(src.Grad);
        }
    }
}
