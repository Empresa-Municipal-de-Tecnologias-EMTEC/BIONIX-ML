using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas
{
    public static class FabricaCamadas
    {
        public static IConvLayer CriarConvLayer(int inChannels, int outChannels, int kernelSize, ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return new CPUSIMD.ConvLayer(inChannels, outChannels, kernelSize);
            return new CPU.ConvLayer(inChannels, outChannels, kernelSize);
        }

        public static IActivationLayer CriarActivationLayer(System.Func<double,double> func, ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return new CPUSIMD.ActivationLayer(func);
            return new CPU.ActivationLayer(func);
        }

        public static IResidualBlock CriarResidualBlock(int channels, ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return new CPUSIMD.ResidualBlock(channels);
            return new CPU.ResidualBlock(channels);
        }

        public static IFPN CriarFPN(int inChannelsC5, int inChannelsC4, int inChannelsC3, int outChannels, ComputacaoContexto ctx)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return new CPUSIMD.FPN(inChannelsC5, inChannelsC4, inChannelsC3, outChannels);
            return new CPU.FPN(inChannelsC5, inChannelsC4, inChannelsC3, outChannels);
        }

        public static IDetectionHead CriarDetectionHead(int inChannels, int interChannels, ComputacaoContexto ctx, int anchorsPerLocation = 9)
        {
            if (ctx is ComputacaoCPUSIMDContexto) return new CPUSIMD.DetectionHead(inChannels, interChannels, anchorsPerLocation);
            return new CPU.DetectionHead(inChannels, interChannels, anchorsPerLocation);
        }
    }
}
