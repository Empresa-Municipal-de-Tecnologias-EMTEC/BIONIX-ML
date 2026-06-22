using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.grafo.CPU;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    public class ConvLayer : IConvLayer
    {
        public int InChannels { get; }
        public int OutChannels { get; }
        public int KernelSize { get; }

        private Tensor _weight; // shape [out, in, k, k]
        private Tensor _bias;   // shape [out]

        public ConvLayer(int inChannels, int outChannels, int kernelSize)
        {
            InChannels = inChannels;
            OutChannels = outChannels;
            KernelSize = kernelSize;
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            _weight = fabrica.Criar(OutChannels, InChannels, KernelSize, KernelSize);
            _bias = fabrica.Criar(OutChannels);
            var rnd = new Random(1234);
            var wdata = _weight.ToArray();
            for (int i = 0; i < wdata.Length; i++) wdata[i] = (rnd.NextDouble() - 0.5) * 0.1;
            for (int i = 0; i < _bias.Size; i++) _bias[i] = 0.0;
            _weight.RequiresGrad = true;
            _bias.RequiresGrad = true;
            // copy values into tensor storage
            int idx = 0;
            for (int oc = 0; oc < OutChannels; oc++)
            for (int ic = 0; ic < InChannels; ic++)
            for (int ky = 0; ky < KernelSize; ky++)
            for (int kx = 0; kx < KernelSize; kx++)
            {
                _weight[idx] = wdata[idx];
                idx++;
            }
        }

        public Tensor Weight => _weight;
        public Tensor Bias => _bias;

        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var shape = input.Shape;
            if (shape.Length != 3) throw new ArgumentException("Input tensor must have shape [h,w,c]");
            int h = shape[0];
            int w = shape[1];
            int c = shape[2];
            if (c != InChannels) throw new ArgumentException("Input channels mismatch");
            var outTensor = fabrica.Criar(h, w, OutChannels);
            int pad = KernelSize / 2;
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            for (int oc = 0; oc < OutChannels; oc++)
            {
                double sum = 0.0;
                for (int ic = 0; ic < InChannels; ic++)
                for (int ky = 0; ky < KernelSize; ky++)
                for (int kx = 0; kx < KernelSize; kx++)
                {
                    int iy = y + ky - pad;
                    int ix = x + kx - pad;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                    int inIndex = (iy * w + ix) * InChannels + ic;
                    int weightIndex = ((oc * InChannels + ic) * KernelSize + ky) * KernelSize + kx;
                    sum += input[inIndex] * _weight[weightIndex];
                }
                double b = _bias != null && _bias.Size > oc ? _bias[oc] : 0.0;
                int outIndex = (y * w + x) * OutChannels + oc;
                outTensor[outIndex] = sum + b;
            }
            outTensor.RequiresGrad = true;
            outTensor.GradFn = new Bionix.ML.grafo.CPU.ConvFunction(input, _weight, _bias, outTensor, KernelSize);
            return outTensor;
        }
    }
}
