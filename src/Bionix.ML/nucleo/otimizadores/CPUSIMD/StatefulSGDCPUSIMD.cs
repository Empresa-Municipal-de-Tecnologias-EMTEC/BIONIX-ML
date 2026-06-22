using System;
using System.Collections.Generic;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.nucleo.otimizadores.CPUSIMD
{
    // CPUSIMD implementation of a simple stateful SGD with momentum
    public class StatefulSGDCPUSIMD : IStatefulOptimizer
    {
        private readonly List<Tensor> _parameters;
        private readonly List<Tensor> _velocities;
        public double Lr { get; set; }
        public double Momentum { get; set; }
        private readonly ComputacaoCPUSIMDContexto _ctx;

        public StatefulSGDCPUSIMD(IEnumerable<Tensor> parameters, ComputacaoCPUSIMDContexto ctx, double lr = 1e-3, double momentum = 0.9)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _parameters = new List<Tensor>(parameters);
            Lr = lr;
            Momentum = momentum;
            _velocities = new List<Tensor>();
            foreach (var p in _parameters)
            {
                if (p == null) { _velocities.Add(null); continue; }
                var shape = p.Shape;
                var v = new TensorCPUSIMD(_ctx, shape);
                Array.Clear(v.Data, 0, v.Size);
                _velocities.Add(v);
            }
        }

        public void Step()
        {
            for (int idx = 0; idx < _parameters.Count; idx++)
            {
                var p = _parameters[idx];
                var v = _velocities[idx] as TensorCPUSIMD;
                if (p == null || p.Grad == null || v == null) continue;
                // operate element-wise; use SIMD where available
                if (p is TensorCPUSIMD tp && v is TensorCPUSIMD tv)
                {
                    var pData = tp.Data;
                    var pGrad = tp.GradArray;
                    var vData = tv.Data;
                    int n = tp.Size;
                    if (System.Numerics.Vector.IsHardwareAccelerated)
                    {
                        int vecSize = System.Numerics.Vector<double>.Count;
                        int i = 0;
                        var vMom = new System.Numerics.Vector<double>(Momentum);
                        var vLr = new System.Numerics.Vector<double>(Lr);
                        for (; i <= n - vecSize; i += vecSize)
                        {
                            var vv = new System.Numerics.Vector<double>(vData, i);
                            var vg = new System.Numerics.Vector<double>(pGrad, i);
                            var vNew = vv * vMom + vg * vLr;
                            vNew.CopyTo(vData, i);

                            var vp = new System.Numerics.Vector<double>(pData, i);
                            var vpNew = vp - vNew;
                            vpNew.CopyTo(pData, i);
                        }
                        for (; i < n; i++)
                        {
                            double newv = Momentum * vData[i] + Lr * pGrad[i];
                            vData[i] = newv;
                            pData[i] = pData[i] - newv;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double newv = Momentum * vData[i] + Lr * pGrad[i];
                            vData[i] = newv;
                            pData[i] = pData[i] - newv;
                        }
                    }
                    tp.ZeroGrad();
                }
            }
        }

        public void SaveState(string dir)
        {
            try
            {
                if (!System.IO.Directory.Exists(dir)) System.IO.Directory.CreateDirectory(dir);
                for (int i = 0; i < _velocities.Count; i++)
                {
                    var v = _velocities[i];
                    if (v == null) continue;
                    var path = System.IO.Path.Combine(dir, $"opt_slot_{i}.bin");
                    Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(path, v);
                }
                var meta = System.Text.Json.JsonSerializer.Serialize(new { lr = Lr, momentum = Momentum, slots = _velocities.Count, timestamp = DateTime.UtcNow });
                System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "opt_meta.json"), meta);
            }
            catch { }
        }

        public void LoadState(string dir)
        {
            try
            {
                if (!System.IO.Directory.Exists(dir)) return;
                var metaPath = System.IO.Path.Combine(dir, "opt_meta.json");
                int expectedSlots = -1;
                if (System.IO.File.Exists(metaPath))
                {
                    try
                    {
                        var meta = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(System.IO.File.ReadAllText(metaPath));
                        if (meta != null && meta.ContainsKey("slots")) expectedSlots = Convert.ToInt32(meta["slots"]);
                    }
                    catch { }
                }

                for (int i = 0; i < _velocities.Count; i++)
                {
                    var path = System.IO.Path.Combine(dir, $"opt_slot_{i}.bin");
                    if (!System.IO.File.Exists(path)) continue;
                    var t = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(path);
                    var v = _velocities[i] as TensorCPUSIMD;
                    if (v == null || t == null) continue;
                    if (v.Size != t.Size) continue;
                    var arr = t.ToArray();
                    Array.Copy(arr, 0, v.Data, 0, v.Size);
                }
            }
            catch { }
        }
    }
}
