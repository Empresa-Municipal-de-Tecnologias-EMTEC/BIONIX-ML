using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class MatMulFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU A, B, Out;
        public MatMulFunction(Tensor A_, Tensor B_, Tensor Out_)
        {
            A = A_ as TensorCPU ?? throw new ArgumentException();
            B = B_ as TensorCPU ?? throw new ArgumentException();
            Out = Out_ as TensorCPU ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            // shapes: A [m,n], B [n,p], Out [m,p]
            int m = A.Shape[0];
            int n = A.Shape[1];
            int p = B.Shape[1];
            if (A.RequiresGrad)
            {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < p; k++) sum += gradOutput[i * p + k] * B[k * p + j];
                        A.Grad[i * n + j] += sum;
                    }
                A.GradFn?.Backward(A.Grad);
            }
            if (B.RequiresGrad)
            {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < p; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < m; k++) sum += A[k * n + i] * gradOutput[k * p + j];
                        B.Grad[i * p + j] += sum;
                    }
                B.GradFn?.Backward(B.Grad);
            }
        }
    }
}
