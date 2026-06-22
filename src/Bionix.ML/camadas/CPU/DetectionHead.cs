using System;
using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.camadas.CPU;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    // Per-scale detection head producing cls and reg tensors
    public class DetectionHead : IDetectionHead
    {
        public ConvLayer HeadConv { get; }
        public ConvLayer ClsConv { get; }
        public ConvLayer RegConv { get; }
        public ConvLayer LandmarksConv { get; }

        public DetectionHead(int inChannels, int interChannels = 32, int anchorsPerLocation = 9)
        {
            HeadConv = new ConvLayer(inChannels: inChannels, outChannels: interChannels, kernelSize: 3);
            ClsConv = new ConvLayer(inChannels: interChannels, outChannels: anchorsPerLocation * 1, kernelSize: 1);
            RegConv = new ConvLayer(inChannels: interChannels, outChannels: anchorsPerLocation * 4, kernelSize: 1);
            LandmarksConv = new ConvLayer(inChannels: interChannels, outChannels: anchorsPerLocation * 10, kernelSize: 1);
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            HeadConv.Initialize(ctx);
            ClsConv.Initialize(ctx);
            RegConv.Initialize(ctx);
            LandmarksConv.Initialize(ctx);
        }

        // Forward returns tuple (cls, reg, landmarks) for this scale
        public (Tensor cls, Tensor reg, Tensor lmk) Forward(Tensor x, ComputacaoContexto ctx)
        {
            var h = HeadConv.Forward(x, ctx);
            h = new ActivationLayer(Bionix.ML.nucleo.funcoesAtivacao.ReLU.ReLU.Forward).Forward(h, ctx);
            var cls = ClsConv.Forward(h, ctx);
            var reg = RegConv.Forward(h, ctx);
            var lmk = LandmarksConv.Forward(h, ctx);
            return (cls, reg, lmk);
        }
    }
}
