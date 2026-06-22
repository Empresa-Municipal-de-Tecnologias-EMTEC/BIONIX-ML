using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.grafo.CPU;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    public class BatchNormLayer : IBatchNormLayer
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

            for (int ch = 0; ch < c; ch++)
            {
                double mean = 0.0;
                int n = h * w;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) mean += input[(y * w + x) * c + ch];
                mean /= n;
                double var = 0.0;
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                {
                    var doubleVal = input[(y * w + x) * c + ch] - mean;
                    var += doubleVal * doubleVal;
                }
                var /= n;
                double inv = 1.0 / Math.Sqrt(var + 1e-5);
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * c + ch;
                    double norm = (input[idx] - mean) * inv;
                    outT[idx] = _gamma[ch] * norm + _beta[ch];
                }
            }
            outT.RequiresGrad = true;
            outT.GradFn = new Bionix.ML.grafo.CPU.BNFunction(input as Bionix.ML.nucleo.tensor.TensorCPU, _gamma, _beta, h, w, c, 1e-5);
            return outT;
        }
    }
}
