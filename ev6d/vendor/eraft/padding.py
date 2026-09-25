"""Top/left zero padding from upstream ImagePadder (MIT).

At least 128 pixels per axis keep all four correlation levels >= 2,
avoiding undefined align_corners=True sampling on a singleton dimension.
Normal DSEC 480x640 padding is unchanged. No resizing is performed.
"""
import torch.nn.functional as F


class ImagePadder:
    def __init__(self, min_size=32):
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        h, w = image.shape[-2:]
        ph = max(128, ((h + self.min_size - 1) // self.min_size) * self.min_size) - h
        pw = max(128, ((w + self.min_size - 1) // self.min_size) * self.min_size) - w
        if self.pad_height is not None and (ph, pw) != (self.pad_height, self.pad_width):
            raise ValueError('Both event volumes must have identical geometry')
        self.pad_height, self.pad_width = ph, pw
        return F.pad(image, (pw, 0, ph, 0), mode='constant', value=0)

    def unpad(self, image):
        return image[..., self.pad_height:, self.pad_width:]
