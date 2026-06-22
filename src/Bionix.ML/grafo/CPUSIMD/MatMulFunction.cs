using System;
using System.Numerics;
using System.Threading.Tasks;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class MatMulFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD a, b, outT;
        public MatMulFunction(Tensor a_, Tensor b_, Tensor out_)
        {
            a = a_ as TensorCPUSIMD ?? throw new ArgumentException();
            b = b_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            // a: (m x n), b: (n x p), outT: (m x p)
            int m = a.Shape[0];
            int n = a.Shape[1];
            int p = b.Shape[1];

            if (a.RequiresGrad)
            {
                // compute a.Grad = gradOutput (m x p) * b^T (p x n)
                var aGrad = a.GradArray;
                var bData = b.Data;
                for (int i = 0; i < m; i++)
                {
                    int rowA = i * n;
                    int rowG = i * p;
                    for (int k = 0; k < p; k++)
                    {
                        double g = gradOutput[rowG + k];
                        if (g == 0.0) continue;
                        int bRow = k * n;
                        int j = 0;
                        if (Vector.IsHardwareAccelerated)
                        {
                            int vecSize = Vector<double>.Count;
                            var vg = new Vector<double>(g);
                            for (; j <= n - vecSize; j += vecSize)
                            {
                                var vB = new Vector<double>(bData, bRow + j);
                                var vA = new Vector<double>(aGrad, rowA + j);
                                vA += vB * vg;
                                vA.CopyTo(aGrad, rowA + j);
                            }
                        }
                        for (; j < n; j++)
                        {
                            aGrad[rowA + j] += g * bData[bRow + j];
                        }
                    }
                }
                a.GradFn?.Backward(aGrad);
            }

            if (b.RequiresGrad)
            {
                var dbg = Environment.GetEnvironmentVariable("DEBUG_GRADS_DETAIL");
                double beforeSum = 0.0; if (!string.IsNullOrEmpty(dbg) && (dbg == "1" || dbg.Equals("true", StringComparison.OrdinalIgnoreCase))) { for (int ii = 0; ii < b.GradArray.Length; ii++) beforeSum += Math.Abs(b.GradArray[ii]); }
                // compute b.Grad = a^T (n x m) * gradOutput (m x p) => (n x p)
                var bGrad = b.GradArray;
                var aData = a.Data;
                for (int k = 0; k < m; k++)
                {
                    int rowA = k * n;
                    int rowG = k * p;
                    for (int i = 0; i < n; i++)
                    {
                        double aik = aData[rowA + i];
                        if (aik == 0.0) continue;
                        int j = 0;
                        if (Vector.IsHardwareAccelerated)
                        {
                            int vecSize = Vector<double>.Count;
                            var vAik = new Vector<double>(aik);
                            for (; j <= p - vecSize; j += vecSize)
                            {
                                var vG = new Vector<double>(gradOutput, rowG + j);
                                var vB = new Vector<double>(bGrad, i * p + j);
                                vB += vG * vAik;
                                vB.CopyTo(bGrad, i * p + j);
                            }
                        }
                        for (; j < p; j++)
                        {
                            bGrad[i * p + j] += aik * gradOutput[rowG + j];
                        }
                    }
                }
                if (!string.IsNullOrEmpty(dbg) && (dbg == "1" || dbg.Equals("true", StringComparison.OrdinalIgnoreCase)))
                {
                    double afterSum = 0.0; for (int ii = 0; ii < bGrad.Length; ii++) afterSum += Math.Abs(bGrad[ii]);
                    Console.WriteLine($"[MatMulFunction] B.grad abs-sum before={beforeSum:E6} after={afterSum:E6} (m={m},n={n},p={p})");
                    if (afterSum == 0.0 && beforeSum == 0.0)
                    {
                        // dump small sample of aData and gradOutput to help debugging
                        int sampleA = Math.Min(8, aData.Length);
                        int sampleG = Math.Min(8, gradOutput.Length);
                        Console.Write("[MatMulFunction] aData sample="); for (int s=0; s<sampleA; s++) Console.Write(aData[s].ToString("E6") + ","); Console.WriteLine();
                        Console.Write("[MatMulFunction] gradOutput sample="); for (int s=0; s<sampleG; s++) Console.Write(gradOutput[s].ToString("E6") + ","); Console.WriteLine();
                    }

                    // if gradOutput is non-zero but B.grad stays zero, log B identity and flags
                    double gradOutSum = 0.0; for (int ii = 0; ii < gradOutput.Length; ii++) gradOutSum += Math.Abs(gradOutput[ii]);
                    if (gradOutSum > 0.0 && afterSum == 0.0)
                    {
                        int bid = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(b);
                        bool bReq = b.RequiresGrad;
                        int bGradLen = bGrad == null ? 0 : bGrad.Length;
                        Console.WriteLine($"[MatMulFunction] OBSERVE: gradOut-abs={gradOutSum:E6} but B.grad stayed zero => B.id={bid} B.RequiresGrad={bReq} B.gradLen={bGradLen} (m={m},n={n},p={p})");
                    }
                }
                b.GradFn?.Backward(bGrad);
            }
        }
    }
}
