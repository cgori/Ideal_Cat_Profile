class Image_File:

    def __init__(self, name, original):
        self.name = name
        self.coef = None
        self.curve = None
        self.mask_crop = None
        self.mask_full = None
        self.original = original
        self.pr = None
        self.predict = None