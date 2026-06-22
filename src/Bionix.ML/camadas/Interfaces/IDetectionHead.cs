using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.Interfaces
{
    public interface IDetectionHead : ILayer
    {
        (Tensor cls, Tensor reg, Tensor lmk) Forward(Tensor x, ComputacaoContexto ctx);
    }
}
