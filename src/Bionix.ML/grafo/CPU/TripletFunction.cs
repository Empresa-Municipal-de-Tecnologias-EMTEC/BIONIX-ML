using System;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.grafo.CPU
{
    public class TripletFunction : IFuncaoRetropropagacao
    {
        private readonly TensorCPU anchor, positive, negative, outT;
        public TripletFunction(Tensor anchor_, Tensor positive_, Tensor negative_, Tensor out_)
        {
            anchor = anchor_ as TensorCPU ?? throw new ArgumentException();
            positive = positive_ as TensorCPU ?? throw new ArgumentException();
            negative = negative_ as TensorCPU ?? throw new ArgumentException();
            outT = out_ as TensorCPU ?? throw new ArgumentException();
        }

        public void Backward(double[] gradOutput)
        {
            double g = gradOutput[0];
            double lossVal = outT[0];
            if (lossVal <= 0.0) return; // hinge inactive -> no grads

            int D = anchor.Size;
            if (anchor.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    anchor.Grad[i] += g * (2.0 * (anchor[i] - positive[i]) - 2.0 * (anchor[i] - negative[i]));
                anchor.GradFn?.Backward(anchor.Grad);
            }
            if (positive.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    positive.Grad[i] += g * (-2.0 * (anchor[i] - positive[i]));
                positive.GradFn?.Backward(positive.Grad);
            }
            if (negative.RequiresGrad)
            {
                for (int i = 0; i < D; i++)
                    negative.Grad[i] += g * (2.0 * (anchor[i] - negative[i]));
                negative.GradFn?.Backward(negative.Grad);
            }
        }
    }
}
