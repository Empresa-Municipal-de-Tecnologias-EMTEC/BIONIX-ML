import src.dados.bmp as bmpmod
import src.dados.wav as wavmod
import src.nucleo.Tensor as tensor_defs


# Converte BMPInfo -> Tensor (shape: [height, width, channels])
fn bmp_to_tensor(var bmp_info: bmpmod.BMPInfo) -> tensor_defs.Tensor:
    var h = bmp_info.height
    var w = bmp_info.width
    var c = 3
    var formato = List[Int]()
    formato.append(h)
    formato.append(w)
    formato.append(c)

    var t = tensor_defs.Tensor(formato^)
    for y in range(h):
        for x in range(w):
            var pixel = bmp_info.pixels[y][x]
            for ch in range(c):
                t.dados[(y * w + x) * c + ch] = pixel[ch]
    return t^


# Converte WAVInfo -> Tensor
# modo: "mono" (mixagem por média), "all" (todos os canais)
fn wav_to_tensor(var wav_info: wavmod.WAVInfo, var modo: String = "mono") -> tensor_defs.Tensor:
    var n_frames = len(wav_info.samples)
    var n_ch = wav_info.num_channels

    if modo == "all":
        var formato = List[Int]()
        formato.append(n_frames)
        formato.append(n_ch)
        var t = tensor_defs.Tensor(formato^)
        for i in range(n_frames):
            var frame = wav_info.samples[i]
            for ch in range(n_ch):
                t.dados[i * n_ch + ch] = frame[ch]
        return t^

    var formato = List[Int]()
    formato.append(n_frames)
    formato.append(1)
    var t = tensor_defs.Tensor(formato^)
    for i in range(n_frames):
        var frame = wav_info.samples[i]
        var acc: Float32 = 0.0
        for ch in range(n_ch):
            acc = acc + frame[ch]
        var avg: Float32 = acc / Float32(n_ch) if n_ch > 0 else 0.0
        t.dados[i] = avg
    return t^
