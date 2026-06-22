using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.camadas.CPU;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    public class ResidualBlock : IResidualBlock
    {
        public ConvLayer Conv1 { get; }
        public BatchNormLayer Bn1 { get; }
        public ActivationLayer Act1 { get; }
        public ConvLayer Conv2 { get; }
        public BatchNormLayer Bn2 { get; }

        public ResidualBlock(int channels, int kernel = 3)
        {
            Conv1 = new ConvLayer(inChannels: channels, outChannels: channels, kernelSize: kernel);
            Bn1 = new BatchNormLayer(channels);
            Act1 = new ActivationLayer(Bionix.ML.nucleo.funcoesAtivacao.ReLU.ReLU.Forward);
            Conv2 = new ConvLayer(inChannels: channels, outChannels: channels, kernelSize: kernel);
            Bn2 = new BatchNormLayer(channels);
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            Conv1.Initialize(ctx);
            Bn1.Initialize(ctx);
            Conv2.Initialize(ctx);
            Bn2.Initialize(ctx);
        }

        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            var x = Conv1.Forward(input, ctx);
            x = Bn1.Forward(x, ctx);
            x = Act1.Forward(x, ctx);
            x = Conv2.Forward(x, ctx);
            x = Bn2.Forward(x, ctx);
            // residual add (assume same shape)
            if (input.Shape.Length == x.Shape.Length && input.Size == x.Size)
            {
                var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
                var outT = fabrica.Criar(input.Shape[0], input.Shape[1], input.Shape[2]);
                for (int i = 0; i < input.Size; i++) outT[i] = input[i] + x[i];
                outT.RequiresGrad = true;
                return outT;
            }
            return x;
        }
    }
}
