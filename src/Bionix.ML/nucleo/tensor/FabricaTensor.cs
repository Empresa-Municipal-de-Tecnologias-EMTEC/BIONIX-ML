using System;
using Bionix.ML.computacao;

namespace Bionix.ML.nucleo.tensor
{
    public class FabricaTensor
    {
        private readonly ComputacaoContexto _ctx;

        public FabricaTensor(ComputacaoContexto ctx)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
        }

        public Tensor Criar(params int[] shape)
        {
            // Select implementation based on computation context
            if (_ctx is ComputacaoCPUSIMDContexto simdCtx && simdCtx.IsAvailable)
            {
                return new TensorCPUSIMD(simdCtx, shape);
            }

            // Fallback to classic CPU tensor
            return new TensorCPU(shape);
        }

        public Tensor FromArray(int[] shape, double[] data)
        {
            if (shape == null) throw new ArgumentNullException(nameof(shape));
            if (data == null) throw new ArgumentNullException(nameof(data));
            // If context supports SIMD CPU tensors, create that implementation
            if (_ctx is ComputacaoCPUSIMDContexto simdCtx && simdCtx.IsAvailable)
            {
                if (data.Length != shape.Aggregate(1, (acc, s) => acc * Math.Max(1, s))) throw new ArgumentException("Shape and data length mismatch");
                return new TensorCPUSIMD(simdCtx, shape, data);
            }

            var impl = new TensorCPU(shape);
            if (data.Length != impl.Size) throw new ArgumentException("Shape and data length mismatch");
            for (int i = 0; i < data.Length; i++) impl[i] = data[i];
            return impl;
        }
    }
}
