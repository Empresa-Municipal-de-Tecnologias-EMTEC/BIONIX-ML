using System;
using System.Numerics;
using System.Threading.Tasks;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.CPUSIMD
{
    public class BatchNormLayer : Interfaces.IBatchNormLayer
    {
        private Tensor _gamma;
        private Tensor _beta;
        private int _channels;

        public BatchNormLayer(int channels)
        {
            _channels = channels;
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            _gamma = fabrica.Criar(_channels);
            _beta = fabrica.Criar(_channels);
            for (int i = 0; i < _channels; i++) { _gamma[i] = 1.0; _beta[i] = 0.0; }
            _gamma.RequiresGrad = true;
            _beta.RequiresGrad = true;
        }

        public Tensor Gamma => _gamma;
        public Tensor Beta => _beta;

        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            var shape = input.Shape;
            if (shape.Length != 3) throw new ArgumentException("Input must be [h,w,c]");
            int h = shape[0], w = shape[1], c = shape[2];
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var outT = fabrica.Criar(h, w, c);

            int n = h * w;
            Parallel.For(0, c, ch =>
            {
                double mean = 0.0;
                for (int p = 0; p < n; ++p) mean += input[p * c + ch];
                mean /= n;

                double var = 0.0;
                for (int q = 0; q < n; ++q)
                {
                    double dv = input[q * c + ch] - mean;
                    var += dv * dv;
                }
                var /= n;
                double inv = 1.0 / Math.Sqrt(var + 1e-5);

                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int vecSize = Vector<double>.Count;
                    var meanV = new Vector<double>(mean);
                    var invV = new Vector<double>(inv);
                    var gammaV = new Vector<double>(_gamma[ch]);
                    var betaV = new Vector<double>(_beta[ch]);
                    double[] gather = new double[vecSize];
                    for (; i <= n - vecSize; i += vecSize)
                    {
                        for (int j = 0; j < vecSize; ++j) gather[j] = input[(i + j) * c + ch];
                        var v = new Vector<double>(gather);
                        var norm = (v - meanV) * invV;
                        var res = norm * gammaV + betaV;
                        res.CopyTo(gather);
                        for (int j = 0; j < vecSize; ++j) outT[(i + j) * c + ch] = gather[j];
                    }
                }
                for (; i < n; ++i)
                {
                    int idx = i * c + ch;
                    double norm = (input[idx] - mean) * inv;
                    outT[idx] = _gamma[ch] * norm + _beta[ch];
                }
            });
            outT.RequiresGrad = true;
            outT.GradFn = new Bionix.ML.grafo.CPUSIMD.BNFunction(input, _gamma, _beta, h, w, c, 1e-5);
            return outT;
        }
    }
}
