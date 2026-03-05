import src.dados.csv as io

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

fn parse_wav(var caminho: String) -> WAVInfo:
    var b = io.ler_arquivo_binario(caminho)
    if len(b) < 44:
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    # Verifica 'RIFF' e 'WAVE'
    if not (b[0] == 82 and b[1] == 73 and b[2] == 70 and b[3] == 70):
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    if not (b[8] == 87 and b[9] == 65 and b[10] == 86 and b[11] == 69):
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    # Procurar chunk fmt (posição 12 em diante)
    var offset = 12
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
    if data_offset == -1:
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^

    var bytes_per_sample = bits_per_sample // 8
    var frame_bytes = bytes_per_sample * num_channels
    if frame_bytes == 0:
        return WAVInfo(0, 0, 0, -1, List[List[Float32]]())^
    var n_frames = (len(b) - data_offset) // frame_bytes

    var samples = List[List[Float32]]()
    for f_idx in range(n_frames):
        var frame = List[Float32](num_channels)
        for ch in range(num_channels):
            var s_offset = data_offset + f_idx * frame_bytes + ch * bytes_per_sample
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
