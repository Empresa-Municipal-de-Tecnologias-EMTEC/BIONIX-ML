using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.Interfaces
{
    public interface IResidualBlock : ILayer
    {
        Tensor Forward(Tensor input, ComputacaoContexto ctx);
    }
}
