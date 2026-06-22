using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class BNFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU input;
        private readonly TensorCPU gamma;
        private readonly TensorCPU beta;
        private readonly int h, w, c;
        private readonly double eps;

        public BNFunction(Tensor input_, Tensor gamma_, Tensor beta_, int h_, int w_, int c_, double eps_ = 1e-5)
        {
            input = input_ as TensorCPU ?? throw new ArgumentException();
            gamma = gamma_ as TensorCPU ?? throw new ArgumentException();
            beta = beta_ as TensorCPU ?? throw new ArgumentException();
            h = h_; w = w_; c = c_; eps = eps_;
        }

        public void Backward(double[] gradOutput)
        {
            int n = h * w;
            // gradOutput is shape [h,w,c] linearized as (y*w+x)*c + ch
            for (int ch = 0; ch < c; ch++)
            {
                // compute mean and variance for this channel
                double mean = 0.0;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mean += input[(y * w + x) * c + ch];
                mean /= n;
                double var = 0.0;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                {
                    double d = input[(y * w + x) * c + ch] - mean;
                    var += d * d;
                }
                var /= n;
                double inv = 1.0 / Math.Sqrt(var + eps);

                // compute sums
                double sum_dy = 0.0;
                double sum_dy_xhat = 0.0;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * c + ch;
                    double xhat = (input[idx] - mean) * inv;
                    double dy = gradOutput[idx];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                }

                // gamma and beta grads
                if (gamma.RequiresGrad)
                {
                    for (int i = 0; i < 1; i++) { /* placeholder */ }
                    gamma.Grad[ch] += sum_dy_xhat;
                    gamma.GradFn?.Backward(gamma.Grad);
                }
                if (beta.RequiresGrad)
                {
                    beta.Grad[ch] += sum_dy;
                    beta.GradFn?.Backward(beta.Grad);
                }

                // input grads
                if (input.RequiresGrad)
                {
                    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                    {
                        int idx = (y * w + x) * c + ch;
                        double xhat = (input[idx] - mean) * inv;
                        double dy = gradOutput[idx];
                        double gx = (1.0 / n) * gamma[ch] * inv * (n * dy - sum_dy - xhat * sum_dy_xhat);
                        input.Grad[idx] += gx;
                    }
                    input.GradFn?.Backward(input.Grad);
                }
            }
        }
    }
}
