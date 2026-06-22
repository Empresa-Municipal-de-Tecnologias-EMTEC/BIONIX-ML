using System;
using Bionix.ML.nucleo.tensor;
using System.Numerics;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPUSIMD
{
    // Minimal CPUSIMD activation layer: currently reuses CPU-style loop but placed under CPUSIMD namespace
    public class ActivationLayer : IActivationLayer
    {
        private readonly Func<double, double> _func;

        public ActivationLayer(Func<double, double> func)
        {
            _func = func ?? throw new ArgumentNullException(nameof(func));
        }

        public void Initialize(ComputacaoContexto ctx) { }

        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var shape = input.Shape;
            var outT = fabrica.Criar(shape[0], shape[1], shape[2]);
            int size = input.Size;
            // Attempt SIMD vectorization for common activations (ReLU). For other activations, fall back to scalar.
            var simdCtx = ctx as ComputacaoCPUSIMDContexto;
            bool didSimd = false;
            try
            {
                // Detect known CPUSIMD ReLU implementation delegate when running under CPUSIMD context
                if (simdCtx != null && _func?.Method?.DeclaringType != null && _func.Method.DeclaringType.FullName.Contains("CPUSIMD.ReLU"))
                {
                    int vecSize = Vector<double>.Count;
                    int i = 0;
                    var vZero = Vector<double>.Zero;
                    var srcData = (input as TensorCPUSIMD)?.Data;
                    var dstData = (outT as TensorCPUSIMD)?.Data;
                    if (srcData != null && dstData != null)
                    {
                        for (; i <= size - vecSize; i += vecSize)
                        {
                            var v = new Vector<double>(srcData, i);
                            var vOut = Vector.Max(v, vZero);
                            vOut.CopyTo(dstData, i);
                        }
                        for (; i < size; i++) dstData[i] = Math.Max(0.0, srcData[i]);
                        didSimd = true;
                    }
                }
            }
            catch { didSimd = false; }

            if (!didSimd)
            {
                for (int i = 0; i < size; i++) outT[i] = _func(input[i]);
            }
            outT.RequiresGrad = true;
            outT.GradFn = new Bionix.ML.grafo.CPUSIMD.ActivationFunction(input, outT);
            return outT;
        }
    }
}
