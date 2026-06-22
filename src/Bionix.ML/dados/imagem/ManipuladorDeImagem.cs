using System;
using System.IO;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.dados.normalizacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Bionix.ML.dados.imagem
{
    public static class ManipuladorDeImagem
    {
        public static BMP carregarBmpDeJPEG(string caminho)
        {
            using var img = Image.Load<Rgba32>(caminho);
            var bmp = BMP.FromImage(img);
            // Garantir que seja RGB (3 canais)
            if (bmp.QuantidadeCanais == 1)
            {
                return ConvertGrayscaleToRGB(bmp);
            }
            return bmp;
        }

        public static BMP carregarBMPDePNG(string caminho)
        {
            using var img = Image.Load<Rgba32>(caminho);
            var bmp = BMP.FromImage(img);
            // Garantir que seja RGB (3 canais)
            if (bmp.QuantidadeCanais == 1)
            {
                return ConvertGrayscaleToRGB(bmp);
            }
            return bmp;
        }

        public static BMP redimensionar(BMP origem, int novaLargura, int novaAltura)
        {
            using var img = new Image<Rgba32>(origem.Width, origem.Height);
            // fill image from origem
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < origem.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < origem.Width; x++)
                    {
                        int srcIndex = (y * origem.Width + x) * origem.QuantidadeCanais;
                        byte r = origem.Armazenamento[srcIndex + 0];
                        byte g = origem.Armazenamento[srcIndex + 1];
                        byte b = origem.Armazenamento[srcIndex + 2];
                        row[x] = new Rgba32(r, g, b, 255);
                    }
                }
            });
            img.Mutate(x => x.Resize(novaLargura, novaAltura));
            return BMP.FromImage(img);
        }

        public static BMP cortar(BMP origem, int x, int y, int largura, int altura)
        {
            using var img = new Image<Rgba32>(origem.Width, origem.Height);
            img.ProcessPixelRows(accessor =>
            {
                for (int yy = 0; yy < origem.Height; yy++)
                {
                    var row = accessor.GetRowSpan(yy);
                    for (int xx = 0; xx < origem.Width; xx++)
                    {
                        int srcIndex = (yy * origem.Width + xx) * origem.QuantidadeCanais;
                        byte r = origem.Armazenamento[srcIndex + 0];
                        byte g = origem.Armazenamento[srcIndex + 1];
                        byte b = origem.Armazenamento[srcIndex + 2];
                        row[xx] = new Rgba32(r, g, b, 255);
                    }
                }
            });
            var crop = img.Clone(ctx => ctx.Crop(new SixLabors.ImageSharp.Rectangle(x, y, largura, altura)));
            return BMP.FromImage(crop);
        }

        public static double[] normalizar(BMP bmp, string tipo)
        {
            // retorna vetor double plano normalizado conforme tipo
            var bytes = bmp.Armazenamento;
            double[] arr = new double[bytes.Length];
            for (int i = 0; i < bytes.Length; i++) arr[i] = bytes[i];
            return tipo switch
            {
                "unit01" => Normalizacao.Unit01(arr),
                _ => Normalizacao.Unit01(arr)
            };
        }

        public static Tensor transformarEmTensor(BMP bmp, ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx);
            int h = bmp.Height;
            int w = bmp.Width;
            int c = bmp.QuantidadeCanais;
            // shape: [h, w, c]
            int[] shape = new[] { h, w, c };
            // convert bytes to double [h*w*c] and normalize 0..1
            var data = new double[h * w * c];
            for (int i = 0; i < data.Length; i++) data[i] = bmp.Armazenamento[i] / 255.0;
            return fabrica.FromArray(shape, data);
        }

        // Crop a square region centered on the provided rectangle (x,y,w,h).
        // Ensures the square lies within image bounds by clamping.
        public static BMP CropSquare(BMP origem, int x, int y, int largura, int altura)
        {
            if (origem == null) throw new ArgumentNullException(nameof(origem));
            int side = Math.Max(largura, altura);
            // center of box
            int cx = x + largura / 2;
            int cy = y + altura / 2;
            int startX = cx - side / 2;
            int startY = cy - side / 2;
            if (startX < 0) startX = 0;
            if (startY < 0) startY = 0;
            if (startX + side > origem.Width) startX = Math.Max(0, origem.Width - side);
            if (startY + side > origem.Height) startY = Math.Max(0, origem.Height - side);
            // if side larger than image, fallback to full image
            if (side > origem.Width || side > origem.Height)
            {
                return cortar(origem, 0, 0, origem.Width, origem.Height);
            }
            return cortar(origem, startX, startY, side, side);
        }

        // Convert an RGB BMP to a single-channel grayscale normalized tensor of size targetSize x targetSize x 1
        public static Tensor TransformarCropParaTensorGrayscale(BMP origemCrop, int targetSize, ComputacaoContexto ctx)
        {
            if (origemCrop == null) throw new ArgumentNullException(nameof(origemCrop));
            if (targetSize <= 0) throw new ArgumentOutOfRangeException(nameof(targetSize));
            // resize using existing redimensionar (returns RGB BMP)
            var resized = redimensionar(origemCrop, targetSize, targetSize);
            // ensure grayscale conversion: create a single-channel BMP from the RGB resized image
            var gray = CriarBMPEscalaCinza(resized);
            var fabrica = new FabricaTensor(ctx);
            // create tensor shape [h,w,1]
            var t = fabrica.Criar(targetSize, targetSize, 1);
            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    int idx = (y * targetSize + x) * gray.QuantidadeCanais;
                    byte v = gray.Armazenamento[idx + 0];
                    double lum = v / 255.0;
                    t[(y * targetSize + x) * 1 + 0] = lum;
                }
            }
            t.RequiresGrad = false;
            return t;
        }

        // Convert an RGB BMP to a 3-channel RGB normalized tensor of size targetSize x targetSize x 3
        public static Tensor TransformarCropParaTensorRGB(BMP origemCrop, int targetSize, ComputacaoContexto ctx)
        {
            if (origemCrop == null) throw new ArgumentNullException(nameof(origemCrop));
            if (targetSize <= 0) throw new ArgumentOutOfRangeException(nameof(targetSize));
            var resized = redimensionar(origemCrop, targetSize, targetSize);
            var fabrica = new FabricaTensor(ctx);
            // create tensor shape [h,w,3]
            var t = fabrica.Criar(targetSize, targetSize, 3);
            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    int idx = (y * targetSize + x) * resized.QuantidadeCanais;
                    byte r = resized.Armazenamento[idx + 0];
                    byte g = resized.Armazenamento[idx + 1];
                    byte b = resized.Armazenamento[idx + 2];
                    t[(y * targetSize + x) * 3 + 0] = r / 255.0;
                    t[(y * targetSize + x) * 3 + 1] = g / 255.0;
                    t[(y * targetSize + x) * 3 + 2] = b / 255.0;
                }
            }
            t.RequiresGrad = false;
            return t;
        }

        // Create a single-channel grayscale BMP from an RGB BMP. Resulting BMP has QuantidadeCanais == 1.
        public static BMP CriarBMPEscalaCinza(BMP rgbBmp)
        {
            if (rgbBmp == null) throw new ArgumentNullException(nameof(rgbBmp));
            int w = rgbBmp.Width;
            int h = rgbBmp.Height;
            var buf = new byte[w * h * 1];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int srcIdx = (y * w + x) * rgbBmp.QuantidadeCanais;
                    byte r = rgbBmp.Armazenamento[srcIdx + 0];
                    byte g = rgbBmp.Armazenamento[srcIdx + 1];
                    byte b = rgbBmp.Armazenamento[srcIdx + 2];
                    byte lum = (byte)Math.Round(0.299 * r + 0.587 * g + 0.114 * b);
                    buf[(y * w + x) * 1 + 0] = lum;
                }
            }
            return new BMP(w, h, 1, buf);
        }

        // Convert a single-channel grayscale BMP to a 3-channel RGB BMP by replicating the luminance channel
        public static BMP ConvertGrayscaleToRGB(BMP grayBmp)
        {
            if (grayBmp == null) throw new ArgumentNullException(nameof(grayBmp));
            if (grayBmp.QuantidadeCanais != 1) return grayBmp; // already RGB or invalid
            int w = grayBmp.Width;
            int h = grayBmp.Height;
            var buf = new byte[w * h * 3];
            for (int i = 0; i < w * h; i++)
            {
                byte lum = grayBmp.Armazenamento[i];
                buf[i * 3 + 0] = lum; // R
                buf[i * 3 + 1] = lum; // G
                buf[i * 3 + 2] = lum; // B
            }
            return new BMP(w, h, 3, buf);
        }

        private static Image<Rgba32> ToImage(BMP bmp)
        {
            var img = new Image<Rgba32>(bmp.Width, bmp.Height);
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < bmp.Width; x++)
                    {
                        int srcIndex = (y * bmp.Width + x) * bmp.QuantidadeCanais;
                        byte r = bmp.Armazenamento[srcIndex + 0];
                        byte g = bmp.Armazenamento[srcIndex + 1];
                        byte b = bmp.Armazenamento[srcIndex + 2];
                        row[x] = new Rgba32(r, g, b, 255);
                    }
                }
            });
            return img;
        }
    }
}
