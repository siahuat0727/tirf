from PIL import Image
import numpy as np
import sys

for path in sys.argv[1:]:
    print(path)
    im = Image.open(path)
    im = np.array(im)
    im = im - im.min()
    im = im.clip(0, 1000) * 60
    Image.fromarray(im).save(path)
