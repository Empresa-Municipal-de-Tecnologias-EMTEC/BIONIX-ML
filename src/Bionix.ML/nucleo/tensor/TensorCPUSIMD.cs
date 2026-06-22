using System;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Bionix.ML.computacao;
using Bionix.ML.grafo.CPUSIMD;

namespace Bionix.ML.nucleo.tensor
{
    // Tensor implementation using managed SIMD helpers (System.Numerics.Vector<T>)
    // Provides SIMD-accelerated element-wise kernels where possible.
    public class TensorCPUSIMD : Tensor, IDisposable
    {
        private readonly double[] _data;
        private double[] _grad;
        private readonly ComputacaoCPUSIMDContexto _ctx;

        // Expose internal buffers to CPUSIMD implementations within the assembly
        internal double[] Data => _data;
        internal double[] GradArray => _grad;

        public TensorCPUSIMD(ComputacaoCPUSIMDContexto ctx, params int[] shape)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            Shape = shape ?? Array.Empty<int>();
            SizeField = Shape.Aggregate(1, (acc, s) => acc * Math.Max(1, s));
            _data = new double[SizeField];
            _grad = new double[SizeField];
            Grad = _grad;
        }

        // Expose computation context used by this tensor
        public ComputacaoCPUSIMDContexto Context => _ctx;

        internal TensorCPUSIMD(ComputacaoCPUSIMDContexto ctx, int[] shape, double[] data)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
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

        public double GetGrad(int i) => _grad[i];
        public void SetGrad(int i, double v) => _grad[i] = v;

        public override double[] ToArray() => (double[])_data.Clone();

        public void Dispose()
        {
            // nothing to dispose here; accelerator lifecycle owned by context
        }

        public override Tensor Add(Tensor other)
        {
            if (other is not TensorCPUSIMD o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (!Shape.SequenceEqual(o.Shape)) throw new ArgumentException("Shape mismatch");

            // Fast path when both tensors share the same context instance
            if (ReferenceEquals(_ctx, o._ctx))
            {
                var r = new TensorCPUSIMD(_ctx, Shape, new double[Size]);
                _ctx.AddDouble(_data, o.Data, r.Data);
                r.RequiresGrad = true;
                r.GradFn = new AddFunction(this, o, r);
                return r;
            }

            // Fallback: contexts differ (e.g., different CPUSIMD context instances in WASM).
            // Perform element-wise addition using safe array access to avoid ReferenceEquals requirement.
            var outArr = new double[Size];
            var otherArr = o.Data;
            for (int i = 0; i < Size; i++) outArr[i] = _data[i] + otherArr[i];
            var rf = new TensorCPUSIMD(_ctx, Shape, outArr);
            rf.RequiresGrad = true;
            rf.GradFn = new AddFunction(this, o, rf);
            return rf;
        }

        public override Tensor Sub(Tensor other)
        {
            if (other is not TensorCPUSIMD o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (!Shape.SequenceEqual(o.Shape)) throw new ArgumentException("Shape mismatch");

            if (ReferenceEquals(_ctx, o._ctx))
            {
                var r = new TensorCPUSIMD(_ctx, Shape, new double[Size]);
                Array.Copy(_data, r.Data, Size);
                _ctx.AddScaled(o.Data, -1.0, r.Data);
                r.RequiresGrad = true;
                r.GradFn = new SubFunction(this, o, r);
                return r;
            }

            var outArr = new double[Size];
            var otherArr = o.Data;
            for (int i = 0; i < Size; i++) outArr[i] = _data[i] - otherArr[i];
            var rf = new TensorCPUSIMD(_ctx, Shape, outArr);
            rf.RequiresGrad = true;
            rf.GradFn = new SubFunction(this, o, rf);
            return rf;
        }

        public override Tensor MulScalar(double scalar)
        {
            var r = new TensorCPUSIMD(_ctx, Shape, new double[Size]);
            int n = Size;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                var vScalar = new Vector<double>(scalar);
                int i = 0;
                for (; i <= n - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(_data, i);
                    (va * vScalar).CopyTo(r._data, i);
                }
                for (; i < n; i++) r._data[i] = _data[i] * scalar;
            }
            else
            {
                for (int i = 0; i < n; i++) r._data[i] = _data[i] * scalar;
            }
            r.RequiresGrad = true;
            r.GradFn = new MulScalarFunction(this, scalar, r);
            return r;
        }

        public override Tensor MatMul(Tensor other)
        {
            if (other is not TensorCPUSIMD o) throw new ArgumentException("Type mismatch or unsupported platform");
            if (Shape.Length != 2 || o.Shape.Length != 2) throw new ArgumentException("MatMul expects 2D tensors");
            int m = Shape[0];
            int n = Shape[1];
            int n2 = o.Shape[0];
            int p = o.Shape[1];
            if (n != n2) throw new ArgumentException("Inner dimensions must match");

            var result = new TensorCPUSIMD(_ctx, m, p);
            Array.Clear(result.Data, 0, result.Data.Length);

            // Fast path when contexts match
            if (ReferenceEquals(_ctx, o._ctx))
            {
                Parallel.For(0, m, i =>
                {
                    int rowA = i * n;
                    int rowR = i * p;
                    for (int k = 0; k < n; k++)
                    {
                        double aik = _data[rowA + k];
                        int j = 0;
                        if (Vector.IsHardwareAccelerated)
                        {
                            int vecSize = Vector<double>.Count;
                            var vAik = new Vector<double>(aik);
                            for (; j <= p - vecSize; j += vecSize)
                            {
                                var vB = new Vector<double>(o.Data, k * p + j);
                                var vR = new Vector<double>(result.Data, rowR + j);
                                vR += vB * vAik;
                                vR.CopyTo(result.Data, rowR + j);
                            }
                        }
                        for (; j < p; j++)
                        {
                            result.Data[rowR + j] += aik * o.Data[k * p + j];
                        }
                    }
                });
            }
            else
            {
                // Fallback: other tensor uses a different CPUSIMD context instance.
                // Use a safer non-vectorized accumulation to avoid cross-context assumptions.
                var otherArr = o.Data;
                for (int i = 0; i < m; i++)
                {
                    int rowA = i * n;
                    int rowR = i * p;
                    for (int k = 0; k < n; k++)
                    {
                        double aik = _data[rowA + k];
                        for (int j = 0; j < p; j++)
                        {
                            result.Data[rowR + j] += aik * otherArr[k * p + j];
                        }
                    }
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
            var t = new TensorCPUSIMD(_ctx, n, m);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    t._data[j * m + i] = _data[i * n + j];
            t.RequiresGrad = true;
            t.GradFn = new TransposeFunction(this, t);
            return t;
        }
    }
}
