using System;
using System.Diagnostics;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.nucleo.funcoesPerda.CPUSIMD.MSE;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("CPUSIMD microbenchmark (small, quick runs)");

        // MatMul sizes
        int dim = 256;
        int matRuns = 3;

        // MSE size
        int mseN = 200_000;
        int mseRuns = 5;

        RunMatMulBenchmark(dim, matRuns);
        RunMSEBenchmark(mseN, mseRuns);
    }

    static void RunMatMulBenchmark(int dim, int runs)
    {
        Console.WriteLine($"MatMul {dim}x{dim} x {dim}x{dim}, runs={runs}");
        var rnd = new Random(123);

        // CPU baseline
        var A_cpu = new TensorCPU(dim, dim);
        var B_cpu = new TensorCPU(dim, dim);
        for (int i = 0; i < A_cpu.Size; i++) A_cpu[i] = rnd.NextDouble();
        for (int i = 0; i < B_cpu.Size; i++) B_cpu[i] = rnd.NextDouble();

        // CPUSIMD tensors
        var simdCtx = new ComputacaoCPUSIMDContexto();
        var A_simd = new TensorCPUSIMD(simdCtx, dim, dim);
        var B_simd = new TensorCPUSIMD(simdCtx, dim, dim);
        for (int i = 0; i < A_simd.Size; i++) A_simd[i] = rnd.NextDouble();
        for (int i = 0; i < B_simd.Size; i++) B_simd[i] = rnd.NextDouble();

        // Warmup
        Console.WriteLine("Warmup CPU...");
        var sw = Stopwatch.StartNew();
        var r0 = A_cpu.MatMul(B_cpu);
        sw.Stop();
        Console.WriteLine($"CPU warmup time: {sw.ElapsedMilliseconds} ms");

        Console.WriteLine("CPU runs...");
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            var r = A_cpu.MatMul(B_cpu);
        }
        sw.Stop();
        Console.WriteLine($"CPU avg time: {sw.Elapsed.TotalMilliseconds / runs} ms");

        // SIMD warmup
        Console.WriteLine("Warmup CPUSIMD...");
        sw.Restart();
        var r1 = A_simd.MatMul(B_simd);
        sw.Stop();
        Console.WriteLine($"CPUSIMD warmup time: {sw.ElapsedMilliseconds} ms");

        Console.WriteLine("CPUSIMD runs...");
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            var r = A_simd.MatMul(B_simd);
        }
        sw.Stop();
        Console.WriteLine($"CPUSIMD avg time: {sw.Elapsed.TotalMilliseconds / runs} ms");
    }

    static void RunMSEBenchmark(int n, int runs)
    {
        Console.WriteLine($"MSE size={n}, runs={runs}");
        var rnd = new Random(321);

        var simdCtx = new ComputacaoCPUSIMDContexto();
        var p = new TensorCPUSIMD(simdCtx, n);
        var t = new TensorCPUSIMD(simdCtx, n);
        for (int i = 0; i < n; i++) { p[i] = rnd.NextDouble(); t[i] = rnd.NextDouble(); }

        // Warmup
        var outT = MSE.Loss(simdCtx, p, t);
        var sw = Stopwatch.StartNew();
        outT.GradFn?.Backward(new double[] { 1.0 });
        sw.Stop();
        Console.WriteLine($"MSE CPUSIMD warmup backward: {sw.ElapsedMilliseconds} ms");

        // Runs
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            var outTi = MSE.Loss(simdCtx, p, t);
            outTi.GradFn?.Backward(new double[] { 1.0 });
        }
        sw.Stop();
        Console.WriteLine($"MSE CPUSIMD avg time (create+backward): {sw.Elapsed.TotalMilliseconds / runs} ms");
    }
}
