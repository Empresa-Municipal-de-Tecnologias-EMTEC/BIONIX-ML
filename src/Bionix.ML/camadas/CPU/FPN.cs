using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.grafo.CPU;
using Bionix.ML.computacao;
using Bionix.ML.camadas.Interfaces;

namespace Bionix.ML.camadas.CPU
{
    // Very small FPN: lateral 1x1 convs and nearest-neighbour upsample/add
    public class FPN : IFPN
    {
        public ConvLayer LatC5 { get; }
        public ConvLayer LatC4 { get; }
        public ConvLayer LatC3 { get; }

        public FPN(int inChannelsC5, int inChannelsC4, int inChannelsC3, int outChannels)
        {
            LatC5 = new ConvLayer(inChannels: inChannelsC5, outChannels: outChannels, kernelSize: 1);
            LatC4 = new ConvLayer(inChannels: inChannelsC4, outChannels: outChannels, kernelSize: 1);
            LatC3 = new ConvLayer(inChannels: inChannelsC3, outChannels: outChannels, kernelSize: 1);
        }

        public void Initialize(ComputacaoContexto ctx)
        {
            LatC5.Initialize(ctx);
            LatC4.Initialize(ctx);
            LatC3.Initialize(ctx);
        }

        private Tensor Upsample2x(Tensor src, ComputacaoContexto ctx)
        {
            var s = src.Shape; // [h,w,c]
            int h = s[0], w = s[1], c = s[2];
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var dst = fabrica.Criar(h * 2, w * 2, c);
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            for (int ch = 0; ch < c; ch++)
            {
                double v = src[(y * w + x) * c + ch];
                int yy = y * 2; int xx = x * 2;
                dst[(yy * (w * 2) + xx) * c + ch] = v;
                dst[(yy * (w * 2) + (xx + 1)) * c + ch] = v;
                dst[((yy + 1) * (w * 2) + xx) * c + ch] = v;
                dst[((yy + 1) * (w * 2) + (xx + 1)) * c + ch] = v;
            }
            dst.RequiresGrad = true;
            dst.GradFn = new UpsampleFunction(src, dst);
            return dst;
        }

        // Input c3,c4,c5 tensors; returns p3,p4,p5
        public (Tensor p3, Tensor p4, Tensor p5) Forward(Tensor c3, Tensor c4, Tensor c5, ComputacaoContexto ctx)
        {
            var t5 = LatC5.Forward(c5, ctx);
            var t4 = LatC4.Forward(c4, ctx);
            var t3 = LatC3.Forward(c3, ctx);

            var up5 = Upsample2x(t5, ctx);
            // align shapes for add: assume up5 matches t4
            var p4 = AddTensors(up5, t4, ctx);
            var up4 = Upsample2x(p4, ctx);
            var p3 = AddTensors(up4, t3, ctx);

            // return p3,p4,p5 (p5 is t5)
            return (p3, p4, t5);
        }

        private Tensor AddTensors(Tensor a, Tensor b, ComputacaoContexto ctx)
        {
            // assume same shape; if not, try to crop/pad b to a
            var shapeA = a.Shape; var shapeB = b.Shape;
            if (shapeA.Length != shapeB.Length || shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1] || shapeA[2] != shapeB[2])
            {
                // naive: if b smaller, paste b into center of a
                var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
                var outT = fabrica.Criar(shapeA[0], shapeA[1], shapeA[2]);
                for (int i = 0; i < outT.Size; i++) outT[i] = a[i];
                int minh = Math.Min(shapeA[0], shapeB[0]);
                int minw = Math.Min(shapeA[1], shapeB[1]);
                int minc = Math.Min(shapeA[2], shapeB[2]);
                for (int y = 0; y < minh; y++)
                for (int x = 0; x < minw; x++)
                for (int ch = 0; ch < minc; ch++)
                {
                    int ia = (y * shapeA[1] + x) * shapeA[2] + ch;
                    int ib = (y * shapeB[1] + x) * shapeB[2] + ch;
                    outT[ia] = a[ia] + b[ib];
                }
                outT.RequiresGrad = true;
                // mismatched shapes: no specialized GradFn implemented; gradients to overlapping region will not flow back to inputs automatically
                return outT;
            }
            var fabr = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var outSame = fabr.Criar(shapeA[0], shapeA[1], shapeA[2]);
            for (int i = 0; i < outSame.Size; i++) outSame[i] = a[i] + b[i];
            outSame.RequiresGrad = true;
            outSame.GradFn = new AddFunction(a, b, outSame);
            return outSame;
        }
    }
}
