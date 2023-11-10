import numpy as np
import scipy
import colorsys

def dom_rgb_mine(ar):
    shape=ar.shape
    ar = ar.reshape(np.product(shape[:len(shape)-1]), shape[len(shape)-1]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, 1)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)       
    counts, bins = np.histogram(vecs, len(codes)) 
    idx_max=np.argmax(counts)
    peak=codes[idx_max]
    return peak[0],peak[1],peak[2]

def rgb_to_color(r, g, b):
    (hue, lgt, sat) = colorsys.rgb_to_hls(r/255,g/255,b/255)
    hue=hue*360
    if lgt < 0.2:
        return "Black"
    elif lgt > 0.8:
        return "White"
    elif sat < 0.25:
        return "Gray"
    elif hue < 30:
        return "Red"
    elif hue < 90:
        return "Yellow"
    elif hue < 150:
        return "Green"
    elif hue < 210:
        return "Cyan"
    elif hue < 270:
        return "Blue"
    elif hue < 330:
        return "Magenta"
    else:
        return "Red"

