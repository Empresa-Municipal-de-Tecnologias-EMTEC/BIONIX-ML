using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Bionix.ML.dados.imagem.bmp
{
    public class BMP
    {
        public int Width { get; }
        public int Height { get; }
        public int QuantidadeCanais { get; }
        // armazenamento: buffer plano intercalado por pixel: [r,g,b,(a)]
        public byte[] Armazenamento { get; }

        public BMP(int width, int height, int quantidadeCanais, byte[] armazenamento)
        {
            Width = width;
            Height = height;
            QuantidadeCanais = quantidadeCanais;
            Armazenamento = armazenamento ?? throw new ArgumentNullException(nameof(armazenamento));
            if (Armazenamento.Length != Width * Height * QuantidadeCanais)
                throw new ArgumentException("Tamanho do buffer não corresponde às dimensões e canais");
        }

        public static BMP FromImage(Image<Rgba32> img)
        {
            int width = img.Width;
            int height = img.Height;
            int channels = 3; // R,G,B
            var buffer = new byte[width * height * channels];
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        var px = row[x];
                        int dstIndex = (y * width + x) * channels;
                        buffer[dstIndex + 0] = px.R;
                        buffer[dstIndex + 1] = px.G;
                        buffer[dstIndex + 2] = px.B;
                    }
                }
            });
            return new BMP(width, height, channels, buffer);
        }
    }
}
