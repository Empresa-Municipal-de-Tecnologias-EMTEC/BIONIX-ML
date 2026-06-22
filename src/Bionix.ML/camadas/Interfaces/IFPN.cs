using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace Bionix.ML.camadas.Interfaces
{
    public interface IFPN : ILayer
    {
        (Tensor p3, Tensor p4, Tensor p5) Forward(Tensor c3, Tensor c4, Tensor c5, ComputacaoContexto ctx);
    }
}
