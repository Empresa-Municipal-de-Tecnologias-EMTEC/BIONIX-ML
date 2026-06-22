using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.Interfaces
{
    public interface IBatchNormLayer : ILayer
    {
        Tensor Forward(Tensor input, ComputacaoContexto ctx);
    }
}
