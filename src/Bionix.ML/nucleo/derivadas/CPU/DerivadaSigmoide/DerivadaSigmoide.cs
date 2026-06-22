namespace Bionix.ML.nucleo.derivadas.DerivadaSigmoide
{
    public static class DerivadaSigmoide
    {
        // Compute derivative given the activated value (sigmoid(x))
        public static double FromActivated(double activated) => activated * (1.0 - activated);
    }
}
