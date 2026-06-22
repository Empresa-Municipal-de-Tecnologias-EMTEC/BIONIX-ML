using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.Interfaces
{
    public interface IActivationLayer : ILayer
    {
        Tensor Forward(Tensor input, ComputacaoContexto ctx);
    }
}
