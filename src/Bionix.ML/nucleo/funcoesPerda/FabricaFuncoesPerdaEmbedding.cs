using System;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;

namespace Bionix.ML.nucleo.funcoesPerda
{
    /// <summary>
    /// Fábrica para criar funções de perda para treinamento de embeddings faciais.
    /// Inclui Triplet Loss e Contrastive Loss.
    /// </summary>
    public static class FabricaFuncoesPerdaEmbedding
    {
        /// <summary>
        /// Cria Triplet Loss com margin.
        /// Input: anchor [1, D], positive [1, D], negative [1, D] (todos normalizados L2)
        /// Output: scalar loss = max(0, d(a,p) - d(a,n) + margin)
        /// </summary>
        public static Func<Tensor, Tensor, Tensor, Tensor> CriarTripletLoss(ComputacaoContexto ctx, double margin = 0.3)
        {
            if (ctx is ComputacaoCPUSIMDContexto)
                return (anchor, positive, negative) => TripletLossCPUSIMD(ctx, anchor, positive, negative, margin);
            if (ctx is ComputacaoCPUContexto)
                return (anchor, positive, negative) => TripletLossCPU(ctx, anchor, positive, negative, margin);
            throw new NotImplementedException("TripletLoss not implemented for this ComputacaoContexto");
        }

        private static Tensor TripletLossCPU(ComputacaoContexto ctx, Tensor anchor, Tensor positive, Tensor negative, double margin)
        {
            var fabrica = new FabricaTensor(ctx);
            
            // Calcular distâncias euclidianas ao quadrado
            // d(a,p)^2 = ||a - p||^2 = ||a||^2 + ||p||^2 - 2*a.p
            // Como estão normalizados L2: ||a||^2 = ||p||^2 = ||n||^2 = 1
            // Então: d(a,p)^2 = 2 - 2*cos(a,p)
            
            var ap_diff = anchor.Sub(positive);
            var an_diff = anchor.Sub(negative);

            // ||diff||^2 = sum(diff^2)
            double ap_dist_sq = 0.0;
            double an_dist_sq = 0.0;
            for (int i = 0; i < ap_diff.Size; i++)
            {
                ap_dist_sq += ap_diff[i] * ap_diff[i];
                an_dist_sq += an_diff[i] * an_diff[i];
            }

            // Triplet loss (squared): max(0, ap_dist_sq - an_dist_sq + margin)
            double loss_val = Math.Max(0.0, ap_dist_sq - an_dist_sq + margin);

            // Criar tensor de saída scalar [1] (compatível com outras perdas)
            var lossTensor = fabrica.Criar(1);
            lossTensor[0] = loss_val;
            lossTensor.RequiresGrad = true;
            // attach GradFn to propagate gradients to anchor/positive/negative
            lossTensor.GradFn = new Bionix.ML.grafo.CPU.TripletFunction(anchor, positive, negative, lossTensor);
            return lossTensor;
        }

        private static Tensor TripletLossCPUSIMD(ComputacaoContexto ctx, Tensor anchor, Tensor positive, Tensor negative, double margin)
        {
            // Implementação SIMD da Triplet Loss com GradFn CPUSIMD
            var fabrica = new FabricaTensor(ctx);

            var ap_diff = anchor.Sub(positive);
            var an_diff = anchor.Sub(negative);

            double ap_dist_sq = 0.0;
            double an_dist_sq = 0.0;
            for (int i = 0; i < ap_diff.Size; i++)
            {
                ap_dist_sq += ap_diff[i] * ap_diff[i];
                an_dist_sq += an_diff[i] * an_diff[i];
            }

            double loss_val = Math.Max(0.0, ap_dist_sq - an_dist_sq + margin);
            var lossTensor = fabrica.Criar(1);
            lossTensor[0] = loss_val;
            lossTensor.RequiresGrad = true;
            lossTensor.GradFn = new Bionix.ML.grafo.CPUSIMD.TripletFunction(anchor, positive, negative, lossTensor);
            return lossTensor;
        }

        /// <summary>
        /// Cria Contrastive Loss.
        /// Input: embedding1 [1, D], embedding2 [1, D], label (0=diferente, 1=igual)
        /// Output: scalar loss
        /// </summary>
        public static Func<Tensor, Tensor, Tensor, Tensor> CriarContrastiveLoss(ComputacaoContexto ctx, double margin = 1.0)
        {
            if (ctx is ComputacaoCPUSIMDContexto)
                return (emb1, emb2, label) => ContrastiveLossCPUSIMD(ctx, emb1, emb2, label, margin);
            if (ctx is ComputacaoCPUContexto)
                return (emb1, emb2, label) => ContrastiveLossCPU(ctx, emb1, emb2, label, margin);
            throw new NotImplementedException("ContrastiveLoss not implemented for this ComputacaoContexto");
        }

        private static Tensor ContrastiveLossCPU(ComputacaoContexto ctx, Tensor emb1, Tensor emb2, Tensor label, double margin)
        {
            var fabrica = new FabricaTensor(ctx);
            
            // Distância euclidiana
            var diff = emb1.Sub(emb2);
            double dist_sq = 0.0;
            for (int i = 0; i < diff.Size; i++)
                dist_sq += diff[i] * diff[i];
            double dist = Math.Sqrt(dist_sq);
            
            double y = label[0]; // 1 se iguais, 0 se diferentes
            double loss;
            
            if (y > 0.5) // Par positivo: minimizar distância
                loss = dist_sq;
            else // Par negativo: maximizar distância até margin
                loss = Math.Max(0.0, margin - dist);
            
            var lossTensor = fabrica.Criar(1, 1);
            lossTensor[0] = loss;
            
            return lossTensor;
        }

        private static Tensor ContrastiveLossCPUSIMD(ComputacaoContexto ctx, Tensor emb1, Tensor emb2, Tensor label, double margin)
        {
            return ContrastiveLossCPU(ctx, emb1, emb2, label, margin);
        }

        /// <summary>
        /// Calcula similaridade de cosseno entre dois embeddings normalizados L2.
        /// Retorna produto escalar (já que são normalizados).
        /// </summary>
        public static double CosineSimilarity(Tensor emb1, Tensor emb2)
        {
            if (emb1.Size != emb2.Size)
                throw new ArgumentException("Embeddings devem ter o mesmo tamanho");
            
            double dot = 0.0;
            for (int i = 0; i < emb1.Size; i++)
                dot += emb1[i] * emb2[i];
            
            return dot; // Já é a similaridade de cosseno para vetores L2-normalizados
        }
    }
}
