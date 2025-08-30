import colorsys

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#' + ''.join(f'{int(c*255):02x}' for c in rgb)

def rotate_hue(rgb, angle):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + angle) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)

def adjust_sv(rgb, s_factor=1.0, v_factor=1.0):
    """Adjust saturation and value"""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    s = max(0, min(1, s * s_factor))
    v = max(0, min(1, v * v_factor))
    return colorsys.hsv_to_rgb(h, s, v)

def get_harmony(hex_color):
    rgb = hex_to_rgb(hex_color)

    return {
        # 1 warna, beda value & saturation
        'monochromatic': [
            rgb_to_hex(adjust_sv(rgb, 0.5, 1.0)),
            rgb_to_hex(adjust_sv(rgb, 1.0, 0.7)),
            rgb_to_hex(adjust_sv(rgb, 1.2, 1.2))
        ],

        # warna berdekatan (sudah ada)
        'analogous': [
            rgb_to_hex(rotate_hue(rgb, 0.08)),
            rgb_to_hex(rotate_hue(rgb, -0.08))
        ],

        # warna lawan (sudah ada)
        'complementary': rgb_to_hex(rotate_hue(rgb, 0.5)),

        # split complementary: lawan ±30°
        'split_complementary': [
            rgb_to_hex(rotate_hue(rgb, 0.5 + 0.08)),
            rgb_to_hex(rotate_hue(rgb, 0.5 - 0.08))
        ],

        # triadic: 120° dan 240° (sudah ada, aku rapihin)
        'triadic': [
            rgb_to_hex(rotate_hue(rgb, 1/3)),
            rgb_to_hex(rotate_hue(rgb, 2/3))
        ],

        # tetradic: 4 warna (90°)
        'tetradic': [
            rgb_to_hex(rotate_hue(rgb, 0.25)),
            rgb_to_hex(rotate_hue(rgb, 0.5)),
            rgb_to_hex(rotate_hue(rgb, 0.75))
        ]
    }
