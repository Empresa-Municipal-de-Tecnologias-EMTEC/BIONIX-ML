using System;

namespace Bionix.ML.grafo.Interfaces
{
    public interface IFuncaoRetropropagacao
    {
        void Backward(double[] gradOutput);
    }
}
