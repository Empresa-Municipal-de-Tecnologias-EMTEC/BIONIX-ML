import src.dados.arquivo as io

fn _bytes_to_uint32_le(var b: List[Int], var offset: Int) -> Int:
    var v: Int = 0
    for i in range(4):
        v = v | (b[offset + i] << (8 * i))
    return v

fn _bytes_to_int32_le(var b: List[Int], var offset: Int) -> Int:
    var uv = _bytes_to_uint32_le(b, offset)
    if uv >= (1 << 31):
        return uv - (1 << 32)
    return uv

fn _bytes_to_uint16_le(var b: List[Int], var offset: Int) -> Int:
    return (b[offset] | (b[offset+1] << 8))

struct BMPInfo(Movable, Copyable):
    var width: Int
    var height: Int
    var bits_per_pixel: Int
    var pixel_array_offset: Int
    var pixels: List[List[List[Float32]]]  # rows x cols x channels (r,g,b) normalized 0..1

    fn __init__(out self, var w: Int, var h: Int, var bpp: Int, var off: Int, var px: List[List[List[Float32]]]):
        self.width = w
        self.height = h
        self.bits_per_pixel = bpp
        self.pixel_array_offset = off
        self.pixels = px^

fn _validar_bmp_bytes(var b: List[Int], var log: Bool = False) -> Bool:
    if len(b) < 54:
        if log:
            print("[diag.bmp] inválido: menos de 54 bytes (", len(b), ")")
        return False
    if not (b[0] == 0x42 and b[1] == 0x4D):
        if log:
            print("[diag.bmp] inválido: assinatura não é BM")
        return False

    var pixel_array_offset = _bytes_to_uint32_le(b, 10)
    var dib_header_size = _bytes_to_uint32_le(b, 14)
    var width = _bytes_to_int32_le(b, 18)
    var height = _bytes_to_int32_le(b, 22)
    var bits_per_pixel = _bytes_to_uint16_le(b, 28)

    if dib_header_size < 40:
        if log:
            print("[diag.bmp] inválido: DIB header < 40 (", dib_header_size, ")")
        return False
    if width <= 0 or height == 0:
        if log:
            print("[diag.bmp] inválido: dimensões inválidas w=", width, " h=", height)
        return False
    if bits_per_pixel != 8 and bits_per_pixel != 24 and bits_per_pixel != 32:
        if log:
            print("[diag.bmp] inválido: bpp não suportado (", bits_per_pixel, ")")
        return False
    if pixel_array_offset < 0 or pixel_array_offset >= len(b):
        if log:
            print("[diag.bmp] inválido: offset de pixels fora do arquivo (", pixel_array_offset, ")")
        return False

    var bytes_per_pixel = bits_per_pixel // 8
    if bytes_per_pixel <= 0:
        if log:
            print("[diag.bmp] inválido: bytes_per_pixel <= 0")
        return False

    var abs_height = height
    if abs_height < 0:
        abs_height = -abs_height

    var row_raw = width * bytes_per_pixel
    var row_stride = ((row_raw + 3) // 4) * 4
    if row_stride <= 0:
        if log:
            print("[diag.bmp] inválido: row_stride <= 0")
        return False

    var available_bytes = len(b) - pixel_array_offset
    if available_bytes <= 0:
        if log:
            print("[diag.bmp] inválido: sem bytes de pixel disponíveis")
        return False

    var max_rows = available_bytes // row_stride
    if max_rows <= 0:
        if log:
            print("[diag.bmp] inválido: arquivo não contém linha completa de pixels")
        return False

    if log:
        print("[diag.bmp] válido: w=", width, " h=", height, " bpp=", bits_per_pixel, " off=", pixel_array_offset, " rows_disponiveis=", max_rows, "/", abs_height)
    return True

fn diagnosticar_bmp(var caminho: String) -> Bool:
    var b = io.ler_arquivo_binario(caminho)
    print("[diag.bmp] arquivo=", caminho, " bytes=", len(b))
    return _validar_bmp_bytes(b, True)

fn parse_bmp(var caminho: String) -> BMPInfo:
    var b = io.ler_arquivo_binario(caminho)
    if not _validar_bmp_bytes(b, False):
        return BMPInfo(0, 0, 0, 0, List[List[List[Float32]]]())^
    var pixel_array_offset = _bytes_to_uint32_le(b, 10)
    var width = _bytes_to_int32_le(b, 18)
    var height = _bytes_to_int32_le(b, 22)
    var bits_per_pixel = _bytes_to_uint16_le(b, 28)

    var bytes_per_pixel = bits_per_pixel // 8
    var row_raw = width * bytes_per_pixel
    var row_stride = ((row_raw + 3) // 4) * 4  # padded to 4 bytes

    var bottom_up = True
    if height < 0:
        bottom_up = False
        height = -height

    var available_bytes = len(b) - pixel_array_offset
    if available_bytes <= 0:
        return BMPInfo(0, 0, 0, 0, List[List[List[Float32]]]())^
    var max_rows = available_bytes // row_stride
    if max_rows <= 0:
        return BMPInfo(0, 0, 0, 0, List[List[List[Float32]]]())^
    if max_rows < height:
        height = max_rows

    var pixels = List[List[List[Float32]]]()
    for y in range(height):
        var row = List[List[Float32]]()
        var src_y = (height - 1 - y) if bottom_up else y
        var row_start = pixel_array_offset + src_y * row_stride
        for x in range(width):
            var off = row_start + x * bytes_per_pixel
            if off + bytes_per_pixel > len(b):
                break
            var b0 = b[off]
            var b1 = 0
            var b2 = 0
            var b3 = 255
            if bytes_per_pixel > 1:
                b1 = b[off + 1]
            if bytes_per_pixel > 2:
                b2 = b[off + 2]
            if bytes_per_pixel > 3:
                b3 = b[off + 3]
            # BMP stores in B,G,R,(A) order
            var blue = Float32(b0) / 255.0
            var green = Float32(b1) / 255.0
            var red = Float32(b2) / 255.0
            if bytes_per_pixel == 1:
                red = blue
                green = blue
            var pixel = List[Float32]()
            pixel.append(red)
            pixel.append(green)
            pixel.append(blue)
            row.append(pixel)
        pixels.append(row)

    return BMPInfo(width, height, bits_per_pixel, pixel_array_offset, pixels)^
