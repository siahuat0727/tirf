def crop(image, x_lo, x_hi, y_lo, y_hi):
    """
    Crop an image to a given region.
    """
    return image[y_lo:y_hi, x_lo:x_hi]
