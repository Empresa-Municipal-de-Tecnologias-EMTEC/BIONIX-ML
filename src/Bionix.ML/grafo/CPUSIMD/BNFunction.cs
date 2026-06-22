using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class BNFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD input;
        private readonly TensorCPUSIMD gamma;
        private readonly TensorCPUSIMD beta;
        private readonly int h, w, c;
        private readonly double eps;

        public BNFunction(Tensor input_, Tensor gamma_, Tensor beta_, int h_, int w_, int c_, double eps_ = 1e-5)
        {
            input = input_ as TensorCPUSIMD ?? throw new ArgumentException();
            gamma = gamma_ as TensorCPUSIMD ?? throw new ArgumentException();
            beta = beta_ as TensorCPUSIMD ?? throw new ArgumentException();
            h = h_; w = w_; c = c_; eps = eps_;
        }

        public void Backward(double[] gradOutput)
        {
            int n = h * w;
            for (int ch = 0; ch < c; ch++)
            {
                double mean = 0.0;
                double var = 0.0;
                // Fast path when channel stride is 1 (c == 1): contiguous across spatial dim
                if (c == 1)
                {
                    int i = 0;
                    var src = input.Data;
                    if (System.Numerics.Vector.IsHardwareAccelerated)
                    {
                        int vecSize = System.Numerics.Vector<double>.Count;
                        var vSum = new System.Numerics.Vector<double>(0.0);
                        for (; i <= n - vecSize; i += vecSize)
                        {
                            var v = new System.Numerics.Vector<double>(src, i);
                            vSum += v;
                        }
                        double tail = 0.0;
                        for (int k = i; k < n; k++) tail += src[k];
                        double blockSum = 0.0;
                        for (int k = 0; k < System.Numerics.Vector<double>.Count; k++) blockSum += vSum[k];
                        mean = (blockSum + tail) / n;
                    }
                    else
                    {
                        for (int k = 0; k < n; k++) mean += src[k];
                        mean /= n;
                    }
                    if (System.Numerics.Vector.IsHardwareAccelerated)
                    {
                        int i2 = 0;
                        var vInv = new System.Numerics.Vector<double>(0.0);
                        for (; i2 <= n - System.Numerics.Vector<double>.Count; i2 += System.Numerics.Vector<double>.Count)
                        {
                            var v = new System.Numerics.Vector<double>(src, i2);
                            var d = v - new System.Numerics.Vector<double>(mean);
                            vInv += d * d;
                        }
                        double tailv = 0.0;
                        for (int k = i2; k < n; k++) { double d = src[k] - mean; tailv += d * d; }
                        double blockVar = 0.0;
                        for (int k = 0; k < System.Numerics.Vector<double>.Count; k++) blockVar += vInv[k];
                        var = (blockVar + tailv) / n;
                    }
                    else
                    {
                        for (int k = 0; k < n; k++) { double d = src[k] - mean; var += d * d; }
                        var /= n;
                    }
                }
                else
                {
                    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mean += input[(y * w + x) * c + ch];
                    mean /= n;
                    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                    {
                        double d = input[(y * w + x) * c + ch] - mean;
                        var += d * d;
                    }
                    var /= n;
                }
                double inv = 1.0 / Math.Sqrt(var + eps);

                double sum_dy = 0.0;
                double sum_dy_xhat = 0.0;
                if (c == 1)
                {
                    var src = input.Data;
                    var outArr = gradOutput;
                    for (int k = 0; k < n; k++)
                    {
                        double xhat = (src[k] - mean) * inv;
                        double dy = outArr[k];
                        sum_dy += dy;
                        sum_dy_xhat += dy * xhat;
                    }
                }
                else
                {
                    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                    {
                        int idx = (y * w + x) * c + ch;
                        double xhat = (input[idx] - mean) * inv;
                        double dy = gradOutput[idx];
                        sum_dy += dy;
                        sum_dy_xhat += dy * xhat;
                    }
                }

                if (gamma.RequiresGrad)
                {
                    gamma.Grad[ch] += sum_dy_xhat;
                    gamma.GradFn?.Backward(gamma.Grad);
                }
                if (beta.RequiresGrad)
                {
                    beta.Grad[ch] += sum_dy;
                    beta.GradFn?.Backward(beta.Grad);
                }

                if (input.RequiresGrad)
                {
                    if (c == 1)
                    {
                        var inData = input.Data;
                        var gOut = gradOutput;
                        for (int k = 0; k < n; k++)
                        {
                            double xhat = (inData[k] - mean) * inv;
                            double dy = gOut[k];
                            double gx = (1.0 / n) * gamma[ch] * inv * (n * dy - sum_dy - xhat * sum_dy_xhat);
                            input.Grad[k] += gx;
                        }
                    }
                    else
                    {
                        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                        {
                            int idx = (y * w + x) * c + ch;
                            double xhat = (input[idx] - mean) * inv;
                            double dy = gradOutput[idx];
                            double gx = (1.0 / n) * gamma[ch] * inv * (n * dy - sum_dy - xhat * sum_dy_xhat);
                            input.Grad[idx] += gx;
                        }
                    }
                    input.GradFn?.Backward(input.Grad);
                }
            }
        }
    }
}
