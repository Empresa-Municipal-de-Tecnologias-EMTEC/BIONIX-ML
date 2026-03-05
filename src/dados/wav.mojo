import src.dados.arquivo as io

struct WAVInfo(Movable, Copyable):
    var sample_rate: Int
    var num_channels: Int
    var bits_per_sample: Int
    var data_offset: Int
    var samples: List[List[Float32]]

    fn __init__(out self, var sr: Int, var nch: Int, var bps: Int, var off: Int, var amostras: List[List[Float32]]):
        self.sample_rate = sr
        self.num_channels = nch
        self.bits_per_sample = bps
        self.data_offset = off
        self.samples = amostras^

fn _bytes_to_uint32_le(var b: List[Int], var offset: Int) -> Int:
    var v: Int = 0
    for i in range(4):
        v = v | (b[offset + i] << (8 * i))
    return v

fn _bytes_to_uint16_le(var b: List[Int], var offset: Int) -> Int:
    return (b[offset] | (b[offset+1] << 8))

fn _validar_wav_bytes(var b: List[Int], var log: Bool = False) -> Bool:
    if len(b) < 44:
        if log:
            print("[diag.wav] inválido: menos de 44 bytes (", len(b), ")")
        return False
    if not (b[0] == 82 and b[1] == 73 and b[2] == 70 and b[3] == 70):
        if log:
            print("[diag.wav] inválido: assinatura não é RIFF")
        return False

    var wave_pos = -1
    var max_scan = 16
    if len(b) < max_scan:
        max_scan = len(b)
    for i in range(max_scan - 3):
        if b[i] == 87 and b[i+1] == 65 and b[i+2] == 86 and b[i+3] == 69:
            wave_pos = i
            break
    if wave_pos == -1:
        if log:
            print("[diag.wav] inválido: marcador WAVE não encontrado")
        return False

    var offset = wave_pos + 4
    var sample_rate = 0
    var num_channels = 0
    var bits_per_sample = 0
    var data_offset = -1
    var encontrou_fmt = False

    while offset + 8 <= len(b):
        var chunk_id0 = b[offset]
        var chunk_id1 = b[offset+1]
        var chunk_id2 = b[offset+2]
        var chunk_id3 = b[offset+3]
        var chunk_size = _bytes_to_uint32_le(b, offset+4)
        if chunk_id0 == 102 and chunk_id1 == 109 and chunk_id2 == 116 and chunk_id3 == 32:
            encontrou_fmt = True
            if offset + 24 <= len(b):
                num_channels = _bytes_to_uint16_le(b, offset+10)
                sample_rate = _bytes_to_uint32_le(b, offset+12)
                bits_per_sample = _bytes_to_uint16_le(b, offset+22)
        elif chunk_id0 == 100 and chunk_id1 == 97 and chunk_id2 == 116 and chunk_id3 == 97:
            data_offset = offset + 8
            break

        offset = offset + 8 + chunk_size
        if offset < 0 or offset > len(b):
            break

    if data_offset == -1:
        if len(b) > 44:
            data_offset = 44
        else:
            if log:
                print("[diag.wav] inválido: chunk data não encontrado")
            return False

    if num_channels <= 0:
        num_channels = 1
    if bits_per_sample <= 0:
        bits_per_sample = 16
    if sample_rate <= 0:
        sample_rate = 22050

    if bits_per_sample != 8 and bits_per_sample != 16 and bits_per_sample != 24 and bits_per_sample != 32:
        if log:
            print("[diag.wav] inválido: bits_per_sample não suportado (", bits_per_sample, ")")
        return False

    var bytes_per_sample = bits_per_sample // 8
    var frame_bytes = bytes_per_sample * num_channels
    if frame_bytes <= 0:
        if log:
            print("[diag.wav] inválido: frame_bytes <= 0")
        return False
    if data_offset < 0 or data_offset >= len(b):
        if log:
            print("[diag.wav] inválido: data_offset fora do arquivo (", data_offset, ")")
        return False

    var n_frames = (len(b) - data_offset) // frame_bytes
    if n_frames <= 0:
        if log:
            print("[diag.wav] inválido: sem frames decodificáveis")
        return False

    if log:
        print("[diag.wav] válido: sr=", sample_rate, " ch=", num_channels, " bps=", bits_per_sample, " off=", data_offset, " frames=", n_frames, " fmt=", encontrou_fmt)
    return True

fn diagnosticar_wav(var caminho: String) -> Bool:
    var b = io.ler_arquivo_binario(caminho)
    print("[diag.wav] arquivo=", caminho, " bytes=", len(b))
    return _validar_wav_bytes(b, True)

fn parse_wav(var caminho: String) -> WAVInfo:
    var b = io.ler_arquivo_binario(caminho)
    if not _validar_wav_bytes(b, False):
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^

    var wave_pos = -1
    var max_scan = 16
    if len(b) < max_scan:
        max_scan = len(b)
    for i in range(max_scan - 3):
        if b[i] == 87 and b[i+1] == 65 and b[i+2] == 86 and b[i+3] == 69:
            wave_pos = i
            break
    if wave_pos == -1:
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    # Procurar chunk fmt (posição 12 em diante)
    var offset = wave_pos + 4
    var sample_rate = 0
    var num_channels = 0
    var bits_per_sample = 0
    var data_offset = -1
    while offset + 8 <= len(b):
        var chunk_id0 = b[offset]
        var chunk_id1 = b[offset+1]
        var chunk_id2 = b[offset+2]
        var chunk_id3 = b[offset+3]
        var chunk_size = _bytes_to_uint32_le(b, offset+4)
        if chunk_id0 == 102 and chunk_id1 == 109 and chunk_id2 == 116 and chunk_id3 == 32:  # 'fmt '
            num_channels = _bytes_to_uint16_le(b, offset+10)
            sample_rate = _bytes_to_uint32_le(b, offset+12)
            bits_per_sample = _bytes_to_uint16_le(b, offset+22)
        elif chunk_id0 == 100 and chunk_id1 == 97 and chunk_id2 == 116 and chunk_id3 == 97:  # 'data'
            data_offset = offset + 8
            break
        offset = offset + 8 + chunk_size
        if offset < 0 or offset > len(b):
            break
    if data_offset == -1:
        if len(b) > 44:
            data_offset = 44
        else:
            return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    if num_channels <= 0 or bits_per_sample <= 0:
        num_channels = 1
        bits_per_sample = 16
    if sample_rate <= 0:
        sample_rate = 22050

    var bytes_per_sample = bits_per_sample // 8
    var frame_bytes = bytes_per_sample * num_channels
    if frame_bytes == 0:
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    if data_offset < 0 or data_offset >= len(b):
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    var n_frames = (len(b) - data_offset) // frame_bytes

    var samples = List[List[Float32]]()
    for f_idx in range(n_frames):
        var frame = List[Float32](num_channels)
        for ch in range(num_channels):
            var s_offset = data_offset + f_idx * frame_bytes + ch * bytes_per_sample
            if s_offset < 0 or s_offset + bytes_per_sample > len(b):
                break
            var value_f: Float32 = 0.0
            if bits_per_sample == 8:
                var uv = b[s_offset]
                value_f = (Float32(uv) - 128.0) / 128.0
            elif bits_per_sample == 16:
                var lo = b[s_offset]
                var hi = b[s_offset + 1]
                var iv = lo | (hi << 8)
                if iv >= 32768:
                    iv = iv - 65536
                value_f = Float32(iv) / 32768.0
            elif bits_per_sample == 24:
                var lo = b[s_offset]
                var mid = b[s_offset + 1]
                var hi = b[s_offset + 2]
                var iv = lo | (mid << 8) | (hi << 16)
                if (iv & (1 << 23)) != 0:
                    iv = iv - (1 << 24)
                value_f = Float32(iv) / 8388608.0
            elif bits_per_sample == 32:
                var iv = _bytes_to_uint32_le(b, s_offset)
                if (iv & (1 << 31)) != 0:
                    iv = iv - (1 << 32)
                value_f = Float32(iv) / 2147483648.0
            else:
                return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
            frame[ch] = value_f
        samples.append(frame)

    return WAVInfo(sample_rate, num_channels, bits_per_sample, data_offset, samples)^
