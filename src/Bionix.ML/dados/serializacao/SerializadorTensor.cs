using System;
using System.IO;
using System.Linq;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.dados.serializacao
{
    public static class SerializadorTensor
    {
        public static void SaveBinary(string path, Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));
            var dir = Path.GetDirectoryName(path);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            var shape = tensor.Shape;
            bw.Write(shape.Length);
            foreach (var s in shape) bw.Write(s);
            var data = tensor.ToArray();
            bw.Write(data.Length);
            foreach (var v in data) bw.Write(v);
        }

        // Save raw shape and data without requiring a Tensor object (useful when only arrays are available)
        public static void SaveBinary(string path, int[] shape, double[] data)
        {
            if (shape == null) throw new ArgumentNullException(nameof(shape));
            if (data == null) throw new ArgumentNullException(nameof(data));
            var dir = Path.GetDirectoryName(path);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            bw.Write(shape.Length);
            foreach (var s in shape) bw.Write(s);
            bw.Write(data.Length);
            foreach (var v in data) bw.Write(v);
        }

        public static Tensor LoadBinary(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);
            int dims = br.ReadInt32();
            var shape = new int[dims];
            for (int i = 0; i < dims; i++) shape[i] = br.ReadInt32();
            int len = br.ReadInt32();
            var data = new double[len];
            for (int i = 0; i < len; i++) data[i] = br.ReadDouble();
            return new TensorCPU(shape, data);
        }

        // Load and create a Tensor using the provided computation context so the
        // returned implementation matches the caller's context (e.g., TensorCPUSIMD).
        public static Tensor LoadBinary(string path, Bionix.ML.computacao.ComputacaoContexto ctx)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);
            int dims = br.ReadInt32();
            var shape = new int[dims];
            for (int i = 0; i < dims; i++) shape[i] = br.ReadInt32();
            int len = br.ReadInt32();
            var data = new double[len];
            for (int i = 0; i < len; i++) data[i] = br.ReadDouble();
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx ?? new Bionix.ML.computacao.ComputacaoCPUContexto());
            return fabrica.FromArray(shape, data);
        }

        public static void SaveText(string path, Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));
            var dir = Path.GetDirectoryName(path);
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            using var sw = new StreamWriter(path);
            sw.WriteLine(string.Join(',', tensor.Shape));
            var data = tensor.ToArray();
            foreach (var v in data) sw.WriteLine(v.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        public static Tensor LoadText(string path)
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length == 0) throw new InvalidDataException("Empty file");
            var shape = lines[0].Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
            var values = lines.Skip(1).Select(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            return new TensorCPU(shape, values);
        }
    }
}
