using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class TransposeFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD src, outT;
        public TransposeFunction(Tensor src_, Tensor out_)
        {
            src = src_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            if (src.RequiresGrad)
            {
                // Use blocked transpose to improve cache locality when writing
                int m = src.Shape[0];
                int n = src.Shape[1];
                int B = 64; // block size (tunable)
                var dst = src.Grad;
                bool hasVec = System.Numerics.Vector.IsHardwareAccelerated;
                int vecSize = hasVec ? System.Numerics.Vector<double>.Count : 1;

                double[] gather = hasVec ? new double[vecSize] : null;

                for (int i0 = 0; i0 < m; i0 += B)
                {
                    int iMax = Math.Min(m, i0 + B);
                    for (int j0 = 0; j0 < n; j0 += B)
                    {
                        int jMax = Math.Min(n, j0 + B);
                        for (int i = i0; i < iMax; i++)
                        {
                            int baseI = i * n;
                            int j = j0;
                            if (hasVec)
                            {
                                for (; j <= jMax - vecSize; j += vecSize)
                                {
                                    for (int k = 0; k < vecSize; k++)
                                        gather[k] = gradOutput[(j + k) * m + i];
                                    var gv = new System.Numerics.Vector<double>(gather);
                                    var dv = new System.Numerics.Vector<double>(dst, baseI + j);
                                    dv += gv;
                                    dv.CopyTo(dst, baseI + j);
                                }
                            }
                            for (; j < jMax; j++)
                            {
                                dst[baseI + j] += gradOutput[j * m + i];
                            }
                        }
                    }
                }
                src.GradFn?.Backward(src.Grad);
            }
        }
    }
}
