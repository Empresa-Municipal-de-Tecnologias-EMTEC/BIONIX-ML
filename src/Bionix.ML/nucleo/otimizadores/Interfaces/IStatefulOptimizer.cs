namespace Bionix.ML.nucleo.otimizadores.Interfaces
{
    public interface IStatefulOptimizer
    {
        double Lr { get; set; }
        void Step();
        void SaveState(string dir);
        void LoadState(string dir);
    }
}
