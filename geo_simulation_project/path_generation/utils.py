def color2RGB(color):
    if not isinstance(color, str):
        return color
    
    color = color.lower()
    color_dict = {
        'k': [0, 0, 0],
        'black': [0, 0, 0],
        'b': [0, 0, 1],
        'blue': [0, 0, 1],
        'g': [0, 1, 0],
        'green': [0, 1, 0],
        'c': [0, 1, 1],
        'cyan': [0, 1, 1],
        'r': [1, 0, 0],
        'red': [1, 0, 0],
        'm': [1, 0, 1],
        'magenta': [1, 0, 1],
        'y': [1, 1, 0],
        'yellow': [1, 1, 0],
        'w': [1, 1, 1],
        'white': [1, 1, 1]
    }
    
    return color_dict.get(color, None)