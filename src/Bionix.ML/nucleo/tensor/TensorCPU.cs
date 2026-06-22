using System;
using System.Linq;
using Bionix.ML.grafo.CPU;

namespace Bionix.ML.nucleo.tensor
{
    public class TensorCPU : Tensor
    {
        private readonly double[] _data;
        private double[] _grad;

        public TensorCPU(params int[] shape)
        {
            Shape = shape ?? Array.Empty<int>();
            SizeField = Shape.Aggregate(1, (acc, s) => acc * Math.Max(1, s));
            _data = new double[SizeField];
            _grad = new double[SizeField];
            Grad = _grad;
        }

        internal TensorCPU(int[] shape, double[] data)
        {
            Shape = shape;
            SizeField = data.Length;
            _data = (double[])data.Clone();
            _grad = new double[SizeField];
            Grad = _grad;
        }

        private int SizeField;
        public override int Size => SizeField;

        public override double this[int linearIndex]
        {
            get => _data[linearIndex];
            set => _data[linearIndex] = value;
        }

        // allow setting/getting raw grad values
        public double GetGrad(int i) => _grad[i];
        public void SetGrad(int i, double v) => _grad[i] = v;

        public override double[] ToArray() => (double[])_data.Clone();

        public override Tensor Add(Tensor other)
        {
            if (other is not TensorCPU o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (!Shape.SequenceEqual(o.Shape)) throw new ArgumentException("Shape mismatch");
            var r = new TensorCPU(Shape);
            for (int i = 0; i < Size; i++) r._data[i] = _data[i] + o._data[i];
            // setup autograd
            r.RequiresGrad = true;
            r.GradFn = new AddFunction(this, o, r);
            return r;
        }

        public override Tensor Sub(Tensor other)
        {
            if (other is not TensorCPU o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (!Shape.SequenceEqual(o.Shape)) throw new ArgumentException("Shape mismatch");
            var r = new TensorCPU(Shape);
            for (int i = 0; i < Size; i++) r._data[i] = _data[i] - o._data[i];
            r.RequiresGrad = true;
            r.GradFn = new SubFunction(this, o, r);
            return r;
        }

        public override Tensor MulScalar(double scalar)
        {
            var r = new TensorCPU(Shape);
            for (int i = 0; i < Size; i++) r._data[i] = _data[i] * scalar;
            r.RequiresGrad = true;
            r.GradFn = new MulScalarFunction(this, scalar, r);
            return r;
        }

        public override Tensor MatMul(Tensor other)
        {
            if (other is not TensorCPU o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (Shape.Length != 2 || o.Shape.Length != 2) throw new ArgumentException("MatMul expects 2D tensors");
            int m = Shape[0];
            int n = Shape[1];
            int n2 = o.Shape[0];
            int p = o.Shape[1];
            if (n != n2) throw new ArgumentException("Inner dimensions must match");
            var result = new TensorCPU(m, p);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += _data[i * n + k] * o._data[k * p + j];
                    }
                    result._data[i * p + j] = sum;
                }
            }
            result.RequiresGrad = true;
            result.GradFn = new MatMulFunction(this, o, result);
            return result;
        }

        public override Tensor Transpose()
        {
            if (Shape.Length != 2) throw new ArgumentException("Transpose expects 2D tensor");
            int m = Shape[0];
            int n = Shape[1];
            var t = new TensorCPU(n, m);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    t._data[j * m + i] = _data[i * n + j];
            t.RequiresGrad = true;
            t.GradFn = new TransposeFunction(this, t);
            return t;
        }
    }
}
