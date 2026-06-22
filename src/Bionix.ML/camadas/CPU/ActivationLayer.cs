using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.grafo.CPU;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    public class ActivationLayer : IActivationLayer
    {
        private readonly Func<double, double> _func;

        public ActivationLayer(Func<double, double> func)
        {
            _func = func ?? throw new ArgumentNullException(nameof(func));
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            // no resources to initialize for stateless activation layer
        }

        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            var shape = input.Shape;
            var fabrica = new FabricaTensor(ctx);
            var outT = fabrica.Criar(shape[0], shape[1], shape[2]);
            int size = input.Size;
            for (int i = 0; i < size; i++) outT[i] = _func(input[i]);
            outT.RequiresGrad = true;
            outT.GradFn = new ActivationFunction(input as TensorCPU, outT as TensorCPU);
            return outT;
        }
    }
}
