using System;
using System.Collections.Generic;
using System.IO;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;

namespace Bionix.ML.nucleo.otimizadores
{
    // Simple Adam optimizer implementation for CPU tensors
    public class Adam : IStatefulOptimizer
    {
        private readonly List<Tensor> _parameters;
        private readonly List<Tensor> _m;
        private readonly List<Tensor> _v;
        private int _t;
        public double Lr { get; set; }
        public double Beta1 { get; set; }
        public double Beta2 { get; set; }
        public double Eps { get; set; }

        public Adam(IEnumerable<Tensor> parameters, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            _parameters = new List<Tensor>(parameters);
            Lr = lr; Beta1 = beta1; Beta2 = beta2; Eps = eps;
            _m = new List<Tensor>(); _v = new List<Tensor>();
            foreach (var p in _parameters)
            {
                if (p == null) { _m.Add(null); _v.Add(null); continue; }
                var shape = p.Shape;
                var m = new TensorCPU(shape);
                var v = new TensorCPU(shape);
                for (int i = 0; i < m.Size; i++) { m[i] = 0.0; v[i] = 0.0; }
                _m.Add(m); _v.Add(v);
            }
            _t = 0;
        }

        public void Step()
        {
            _t++;
            double biasCorr1 = 1.0 - Math.Pow(Beta1, _t);
            double biasCorr2 = 1.0 - Math.Pow(Beta2, _t);
            for (int idx = 0; idx < _parameters.Count; idx++)
            {
                var p = _parameters[idx];
                var m = _m[idx];
                var v = _v[idx];
                if (p == null || p.Grad == null || m == null || v == null) continue;
                for (int i = 0; i < p.Size; i++)
                {
                    double g = p.Grad[i];
                    m[i] = Beta1 * m[i] + (1 - Beta1) * g;
                    v[i] = Beta2 * v[i] + (1 - Beta2) * g * g;
                    double mHat = m[i] / biasCorr1;
                    double vHat = v[i] / biasCorr2;
                    double update = Lr * mHat / (Math.Sqrt(vHat) + Eps);
                    p[i] = p[i] - update;
                }
                p.ZeroGrad();
            }
        }

        public void SaveState(string dir)
        {
            try
            {
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                for (int i = 0; i < _m.Count; i++)
                {
                    var mi = _m[i]; if (mi == null) continue;
                    SerializadorTensor.SaveBinary(Path.Combine(dir, $"opt_m_{i}.bin"), mi);
                }
                for (int i = 0; i < _v.Count; i++)
                {
                    var vi = _v[i]; if (vi == null) continue;
                    SerializadorTensor.SaveBinary(Path.Combine(dir, $"opt_v_{i}.bin"), vi);
                }
                var meta = System.Text.Json.JsonSerializer.Serialize(new { lr = Lr, beta1 = Beta1, beta2 = Beta2, eps = Eps, t = _t, slots = _m.Count, type = "adam", timestamp = DateTime.UtcNow });
                File.WriteAllText(Path.Combine(dir, "opt_meta.json"), meta);
            }
            catch { }
        }

        public void LoadState(string dir)
        {
            try
            {
                if (!Directory.Exists(dir)) return;
                for (int i = 0; i < _m.Count; i++)
                {
                    var path = Path.Combine(dir, $"opt_m_{i}.bin");
                    if (!File.Exists(path)) continue;
                    var t = SerializadorTensor.LoadBinary(path);
                    var slot = _m[i]; if (slot == null || t == null) continue;
                    if (slot.Size != t.Size) continue;
                    var arr = t.ToArray(); for (int k = 0; k < slot.Size; k++) slot[k] = arr[k];
                }
                for (int i = 0; i < _v.Count; i++)
                {
                    var path = Path.Combine(dir, $"opt_v_{i}.bin");
                    if (!File.Exists(path)) continue;
                    var t = SerializadorTensor.LoadBinary(path);
                    var slot = _v[i]; if (slot == null || t == null) continue;
                    if (slot.Size != t.Size) continue;
                    var arr = t.ToArray(); for (int k = 0; k < slot.Size; k++) slot[k] = arr[k];
                }
                // try to read metadata (t, lr, beta1, beta2, eps) from meta JSON
                var metaPath = Path.Combine(dir, "opt_meta.json");
                if (File.Exists(metaPath))
                {
                    try
                    {
                        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath));
                        var root = doc.RootElement;
                        if (root.TryGetProperty("t", out var tEl) && tEl.ValueKind == System.Text.Json.JsonValueKind.Number)
                        {
                            try { _t = tEl.GetInt32(); } catch { _t = Convert.ToInt32(tEl.GetDouble()); }
                        }
                        if (root.TryGetProperty("lr", out var lrEl) && (lrEl.ValueKind == System.Text.Json.JsonValueKind.Number))
                        {
                            try { Lr = lrEl.GetDouble(); } catch { }
                        }
                        if (root.TryGetProperty("beta1", out var b1) && b1.ValueKind == System.Text.Json.JsonValueKind.Number)
                        {
                            try { Beta1 = b1.GetDouble(); } catch { }
                        }
                        if (root.TryGetProperty("beta2", out var b2) && b2.ValueKind == System.Text.Json.JsonValueKind.Number)
                        {
                            try { Beta2 = b2.GetDouble(); } catch { }
                        }
                        if (root.TryGetProperty("eps", out var eEl) && eEl.ValueKind == System.Text.Json.JsonValueKind.Number)
                        {
                            try { Eps = eEl.GetDouble(); } catch { }
                        }
                    }
                    catch { }
                }
            }
            catch { }
        }

        // Expose diagnostic helpers for external inspection
        public int Steps => _t;
        public int SlotCount => _m.Count;

        public (double meanAbs, double maxAbs) GetMStats(int index)
        {
            if (index < 0 || index >= _m.Count) return (0.0, 0.0);
            var t = _m[index]; if (t == null) return (0.0, 0.0);
            double sumAbs = 0.0; double max = 0.0; int cnt = t.Size;
            for (int i = 0; i < cnt; i++) { var a = Math.Abs(t[i]); sumAbs += a; if (a > max) max = a; }
            return (cnt>0 ? sumAbs/cnt : 0.0, max);
        }

        public (double meanAbs, double maxAbs) GetVStats(int index)
        {
            if (index < 0 || index >= _v.Count) return (0.0, 0.0);
            var t = _v[index]; if (t == null) return (0.0, 0.0);
            double sumAbs = 0.0; double max = 0.0; int cnt = t.Size;
            for (int i = 0; i < cnt; i++) { var a = Math.Abs(t[i]); sumAbs += a; if (a > max) max = a; }
            return (cnt>0 ? sumAbs/cnt : 0.0, max);
        }
    }
}
