using System;
using System.Linq;
using Bionix.ML.computacao;
using Bionix.ML.grafo;

namespace Bionix.ML.nucleo.tensor
{
    public abstract class Tensor
    {
        public int[] Shape { get; protected set; }

        // Autograd fields
        public double[] Grad { get; protected set; }
        public bool RequiresGrad { get; set; } = false;
        public IFuncaoRetropropagacao? GradFn { get; set; }

        public abstract int Size { get; }
        public abstract double this[int linearIndex] { get; set; }
        public abstract double[] ToArray();

        public abstract Tensor Add(Tensor other);
        public abstract Tensor Sub(Tensor other);
        public abstract Tensor MulScalar(double scalar);
        public abstract Tensor MatMul(Tensor other);
        public abstract Tensor Transpose();

        public override string ToString()
        {
            return $"Tensor(shape=[{string.Join(',', Shape)}], size={Size})";
        }

        public virtual void ZeroGrad()
        {
            if (Grad != null) Array.Clear(Grad, 0, Grad.Length);
        }

        // Allow controlled setting of gradient arrays from outside assemblies
        public void SetGrad(double[] grad)
        {
            Grad = grad;
        }

        public void Backward()
        {
            if (Size != 1) throw new InvalidOperationException("Backward currently supports scalar output tensors only");
            if (Grad == null) Grad = new double[Size];
            Grad[0] = 1.0;
            GradFn?.Backward(Grad);
        }
    }

    

    public static class TensorExtensions
    {
        // Sum of squares reduction producing scalar tensor
        public static Tensor SumSquares(this Tensor t)
        {
            if (t == null) throw new ArgumentNullException(nameof(t));
            var fabrica = new FabricaTensor(new ComputacaoCPUContexto());
            var outT = fabrica.Criar(1);
            double s = 0.0;
            for (int i = 0; i < t.Size; i++) s += t[i] * t[i];
            outT[0] = s;
            outT.RequiresGrad = true;
            outT.GradFn = new Bionix.ML.grafo.CPU.SumSquaresFunction(t as TensorCPU, outT as TensorCPU);
            return outT;
        }
    }
}
