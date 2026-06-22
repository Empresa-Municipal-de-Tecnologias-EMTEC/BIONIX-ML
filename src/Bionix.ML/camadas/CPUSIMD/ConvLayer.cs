using System;
using Bionix.ML.nucleo.tensor;
using System.Numerics;
using System.Threading.Tasks;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPUSIMD
{
    // Minimal CPUSIMD ConvLayer: uses same algorithm as CPU Conv for now.
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
            // SIMD path when running under CPUSIMD context and tensors are CPUSIMD-backed
            var simdCtx = ctx as ComputacaoCPUSIMDContexto;
            var inT = input as TensorCPUSIMD;
            var wT = _weight as TensorCPUSIMD;
            var bT = _bias as TensorCPUSIMD;
            var outT = outTensor as TensorCPUSIMD;
            if (simdCtx != null && inT != null && wT != null && outT != null)
            {
                var inData = inT.Data;
                var wData = wT.Data;
                var outData = outT.Data;
                var biasData = bT?.Data;
                int vecSize = Vector<double>.Count;
                int block = InChannels * KernelSize * KernelSize;
                var tmpLocal = new ThreadLocal<double[]>(() => new double[vecSize]);
                try
                {
                    Parallel.For(0, h, y =>
                    {
                        double[] tmp = tmpLocal.Value;
                        for (int x = 0; x < w; x++)
                        {
                            int baseOut = (y * w + x) * OutChannels;
                            // process full SIMD output-channel blocks
                            for (int ocStart = 0; ocStart <= OutChannels - vecSize; ocStart += vecSize)
                            {
                                var vAcc = Vector<double>.Zero;
                                for (int ic = 0; ic < InChannels; ic++)
                                for (int ky = 0; ky < KernelSize; ky++)
                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int iy = y + ky - pad;
                                    int ix = x + kx - pad;
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                                    int inIndex = (iy * w + ix) * InChannels + ic;
                                    double inVal = inData[inIndex];
                                    int wBase = ((ocStart * InChannels + ic) * KernelSize + ky) * KernelSize + kx;
                                    for (int t = 0; t < vecSize; t++) tmp[t] = wData[wBase + t * block];
                                    var vW = new Vector<double>(tmp);
                                    vAcc += vW * new Vector<double>(inVal);
                                }
                                if (biasData != null) vAcc += new Vector<double>(biasData, ocStart);
                                vAcc.CopyTo(outData, baseOut + ocStart);
                            }
                            // scalar remainder
                            for (int oc = (OutChannels / vecSize) * vecSize; oc < OutChannels; oc++)
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
                                    sum += inData[inIndex] * wData[weightIndex];
                                }
                                double b = biasData != null && biasData.Length > oc ? biasData[oc] : 0.0;
                                outData[baseOut + oc] = sum + b;
                            }
                        }
                    });
                }
                finally { tmpLocal.Dispose(); }

                outT.RequiresGrad = true;
                // GradFn will be set by factory
                return outTensor;
            }

            // Fallback scalar path
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
            // GradFn will be set by factory to appropriate grafo implementation
            return outTensor;
        }
    }
}
