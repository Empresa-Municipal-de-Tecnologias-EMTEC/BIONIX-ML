using System;
using System;
using System.Numerics;

namespace Bionix.ML.computacao
{
    // Managed CPUSIMD computation context: provides SIMD-accelerated helpers
    // implemented with System.Numerics.Vector<T> as a fallback where ILGPU
    // integration is not available or not yet configured.
    public class ComputacaoCPUSIMDContexto : ComputacaoContexto, IDisposable
    {
        public ComputacaoCPUSIMDContexto() : base(TipoComputacao.CPUSIMD)
        {
        }

        public bool IsAvailable => true;

        public void Dispose()
        {
        }

        // Element-wise add for double arrays using Vector<double> when possible
        public void AddDouble(double[] a, double[] b, double[] dst)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (a.Length != b.Length || a.Length != dst.Length) throw new ArgumentException("Array lengths must match.");

            int i = 0;
            int n = a.Length;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                for (; i <= n - vecSize; i += vecSize)
                {
                    var va = new Vector<double>(a, i);
                    var vb = new Vector<double>(b, i);
                    (va + vb).CopyTo(dst, i);
                }
            }
            for (; i < n; i++) dst[i] = a[i] + b[i];
        }

        // Element-wise add for float arrays using Vector<float>
        public void Add(float[] a, float[] b, float[] dst)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (a.Length != b.Length || a.Length != dst.Length) throw new ArgumentException("Array lengths must match.");

            int i = 0;
            int n = a.Length;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<float>.Count;
                for (; i <= n - vecSize; i += vecSize)
                {
                    var va = new Vector<float>(a, i);
                    var vb = new Vector<float>(b, i);
                    (va + vb).CopyTo(dst, i);
                }
            }
            for (; i < n; i++) dst[i] = a[i] + b[i];
        }

        // Add scaled: dst += src * scalar
        public void AddScaled(double[] src, double scalar, double[] dst)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (src.Length != dst.Length) throw new ArgumentException("Array lengths must match.");

            int i = 0;
            int n = src.Length;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                var vScalar = new Vector<double>(scalar);
                for (; i <= n - vecSize; i += vecSize)
                {
                    var vs = new Vector<double>(src, i);
                    var vd = new Vector<double>(dst, i);
                    vd += vs * vScalar;
                    vd.CopyTo(dst, i);
                }
            }
            for (; i < n; i++) dst[i] += src[i] * scalar;
        }

        // Add a segment: dst[dstOffset..dstOffset+length) += src[srcOffset..srcOffset+length)
        public void AddInto(double[] src, int srcOffset, double[] dst, int dstOffset, int length)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (srcOffset < 0 || dstOffset < 0 || length < 0) throw new ArgumentOutOfRangeException("Offsets and length must be non-negative.");
            if (srcOffset + length > src.Length || dstOffset + length > dst.Length) throw new ArgumentException("Segment out of range.");

            int i = 0;
            int n = length;
            if (Vector.IsHardwareAccelerated)
            {
                int vecSize = Vector<double>.Count;
                for (; i <= n - vecSize; i += vecSize)
                {
                    var vs = new Vector<double>(src, srcOffset + i);
                    var vd = new Vector<double>(dst, dstOffset + i);
                    vd += vs;
                    vd.CopyTo(dst, dstOffset + i);
                }
            }
            for (; i < n; i++) dst[dstOffset + i] += src[srcOffset + i];
        }
    }
}
