using System;
using System.Collections.Generic;
using System.IO;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;

namespace Bionix.ML.nucleo.otimizadores
{
    // Simple stateful SGD with momentum that can save/load velocity slots per-parameter
    public class StatefulSGD : IStatefulOptimizer
    {
        private readonly List<Tensor> _parameters;
        private readonly List<Tensor> _velocities;
        public double Lr { get; set; }
        public double Momentum { get; set; }

        public StatefulSGD(IEnumerable<Tensor> parameters, double lr = 1e-3, double momentum = 0.9)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            _parameters = new List<Tensor>(parameters);
            Lr = lr;
            Momentum = momentum;
            _velocities = new List<Tensor>();
            foreach (var p in _parameters)
            {
                if (p == null) { _velocities.Add(null); continue; }
                var shape = p.Shape;
                var v = new TensorCPU(shape);
                // initialize zeros
                for (int i = 0; i < v.Size; i++) v[i] = 0.0;
                _velocities.Add(v);
            }
        }

        public void Step()
        {
            for (int idx = 0; idx < _parameters.Count; idx++)
            {
                var p = _parameters[idx];
                var v = _velocities[idx];
                if (p == null || p.Grad == null || v == null) continue;
                for (int i = 0; i < p.Size; i++)
                {
                    v[i] = Momentum * v[i] + Lr * p.Grad[i];
                    p[i] = p[i] - v[i];
                }
                p.ZeroGrad();
            }
        }

        // Save velocities to directory as opt_slot_0.bin, opt_slot_1.bin ... and meta.json
        public void SaveState(string dir)
        {
            try
            {
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                for (int i = 0; i < _velocities.Count; i++)
                {
                    var v = _velocities[i];
                    if (v == null) continue;
                    var path = Path.Combine(dir, $"opt_slot_{i}.bin");
                    SerializadorTensor.SaveBinary(path, v);
                }
                var meta = System.Text.Json.JsonSerializer.Serialize(new { lr = Lr, momentum = Momentum, slots = _velocities.Count, timestamp = DateTime.UtcNow });
                File.WriteAllText(Path.Combine(dir, "opt_meta.json"), meta);
            }
            catch { }
        }

        // Load velocities if present (match by slot index)
        public void LoadState(string dir)
        {
            try
            {
                if (!Directory.Exists(dir)) return;
                // read opt_meta.json if present to validate expected slots
                var metaPath = Path.Combine(dir, "opt_meta.json");
                int expectedSlots = -1;
                if (File.Exists(metaPath))
                {
                    try
                    {
                        var meta = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(File.ReadAllText(metaPath));
                        if (meta != null && meta.ContainsKey("slots")) expectedSlots = Convert.ToInt32(meta["slots"]);
                    }
                    catch { }
                }

                if (expectedSlots >= 0 && expectedSlots != _velocities.Count)
                {
                    Console.WriteLine($"StatefulSGD.LoadState: slot count mismatch (meta={expectedSlots} vs local={_velocities.Count}), attempting best-effort load.");
                }

                for (int i = 0; i < _velocities.Count; i++)
                {
                    var path = Path.Combine(dir, $"opt_slot_{i}.bin");
                    if (!File.Exists(path))
                    {
                        // missing slot file
                        continue;
                    }
                    var t = SerializadorTensor.LoadBinary(path);
                    var v = _velocities[i];
                    if (v == null)
                    {
                        Console.WriteLine($"StatefulSGD.LoadState: local velocity slot {i} is null, skipping.");
                        continue;
                    }
                    if (t == null)
                    {
                        Console.WriteLine($"StatefulSGD.LoadState: loaded tensor for slot {i} is null, skipping.");
                        continue;
                    }
                    if (v.Size != t.Size)
                    {
                        Console.WriteLine($"StatefulSGD.LoadState: size mismatch for slot {i} (disk={t.Size} vs local={v.Size}), skipping this slot.");
                        continue;
                    }
                    var arr = t.ToArray();
                    for (int k = 0; k < v.Size; k++) v[k] = arr[k];
                }
            }
            catch { }
        }
    }
}
