using System;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.grafo.CPUSIMD
{
    public class TripletFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPUSIMD anchor, positive, negative, outT;
        public TripletFunction(Tensor anchor_, Tensor positive_, Tensor negative_, Tensor out_)
        {
            anchor = anchor_ as TensorCPUSIMD ?? throw new ArgumentException();
            positive = positive_ as TensorCPUSIMD ?? throw new ArgumentException();
            negative = negative_ as TensorCPUSIMD ?? throw new ArgumentException();
            outT = out_ as TensorCPUSIMD ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double lossVal = outT[0];
            if (lossVal <= 0.0) return;

            int D = anchor.Size;
            var aData = anchor.Data;
            var pData = positive.Data;
            var nData = negative.Data;

            if (anchor.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    anchor.Grad[i] += g * (2.0 * (aData[i] - pData[i]) - 2.0 * (aData[i] - nData[i]));
                anchor.GradFn?.Backward(anchor.Grad);
            }
            if (positive.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    positive.Grad[i] += g * (-2.0 * (aData[i] - pData[i]));
                positive.GradFn?.Backward(positive.Grad);
            }
            if (negative.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    negative.Grad[i] += g * (2.0 * (aData[i] - nData[i]));
                negative.GradFn?.Backward(negative.Grad);
            }
        }
    }
}
