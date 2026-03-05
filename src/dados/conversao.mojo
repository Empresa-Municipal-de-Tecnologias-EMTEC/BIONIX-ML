// Conversões de BMP/WAV para Tensor do núcleo
from ..nucleo.Tensor import Tensor

struct BMPInfo(Movable, Copyable):
    var largura: Int
    var altura: Int
    var canais: Int
    var pixels: List[List[List[Float32]]] // [row][col][c]

struct WAVInfo(Movable, Copyable):
    var sample_rate: Int
    var canais: Int
    var amostras: List[List[Float32]] // [frame][channel]

fn bmp_to_tensor(bmp: BMPInfo) -> Tensor:
    var h = bmp.altura
    var w = bmp.largura
    var c = bmp.canais
    var shape = List[Int](3)
    shape[0] = h
    shape[1] = w
    shape[2] = c
    var total = h * w * c
    var dados = List[Float32](total)
    var idx = 0
    for i in range(h):
        for j in range(w):
            for k in range(c):
                dados[idx] = bmp.pixels[i][j][k]
                idx = idx + 1

    return Tensor(shape, dados)

fn wav_to_tensor(wav: WAVInfo, mixdown: Bool = true) -> Tensor:
    var n_frames = len(wav.amostras)
    var n_ch = wav.canais
    if mixdown and n_ch > 1:
        var shape = List[Int](2)
        shape[0] = n_frames
        shape[1] = 1
        var dados = List[Float32](n_frames)
        for i in range(n_frames):
            var s: Float32 = 0.0
            for ch in range(n_ch):
                s = s + wav.amostras[i][ch]
            dados[i] = s / Float32(n_ch)
        return Tensor(shape, dados)
    else:
        var shape = List[Int](2)
        shape[0] = n_frames
        shape[1] = n_ch
        var dados = List[Float32](n_frames * n_ch)
        var idx = 0
        for i in range(n_frames):
            for ch in range(n_ch):
                dados[idx] = wav.amostras[i][ch]
                idx = idx + 1
        return Tensor(shape, dados)
import src.nucleo.nucleo as nucleo
import src.dados.bmp as bmpmod
import src.dados.wav as wavmod

# Converte BMPInfo -> Tensor (shape: [height, width, channels])
fn bmp_to_tensor(var bmp_info: bmpmod.BMPInfo) -> nucleo.Tensor:
    var h = bmp_info.height
    var w = bmp_info.width
    var c = 3
    var formato = List[Int](3)
    formato.append(h)
    formato.append(w)
    formato.append(c)
    var t = nucleo.Tensor(formato^)
    for y in range(h):
        for x in range(w):
            var pixel = bmp_info.pixels[y][x]
            for ch in range(c):
                t.dados[(y * w + x) * c + ch] = pixel[ch]
    return t^

# Converte WAVInfo -> Tensor
# modo: "mono" (mixagem por média), "all" (todos os canais)
fn wav_to_tensor(var wav_info: wavmod.WAVInfo, var modo: String = "mono") -> nucleo.Tensor:
    var n_frames = len(wav_info.samples)
    var n_ch = wav_info.num_channels
    if modo == "all":
        var formato = List[Int](2)
        formato.append(n_frames)
        formato.append(n_ch)
        var t = nucleo.Tensor(formato^)
        for i in range(n_frames):
            var frame = wav_info.samples[i]
            for ch in range(n_ch):
                t.dados[i * n_ch + ch] = frame[ch]
        return t^
    else:
        # mixar para mono por média simples
        var formato = List[Int](2)
        formato.append(n_frames)
        formato.append(1)
        var t = nucleo.Tensor(formato^)
        for i in range(n_frames):
            var frame = wav_info.samples[i]
            var acc: Float32 = 0.0
            for ch in range(n_ch):
                acc = acc + frame[ch]
            var avg: Float32 = acc / Float32(n_ch)
            t.dados[i] = avg
        return t^
