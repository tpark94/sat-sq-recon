'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

class PrimitiveParameters(object):
    """ All predicted primitives into a single python object
    """
    def __init__(self, shape, size, translation, rotation, prob, taper=None):
        self._size        = size
        self._shape       = shape
        self._translation = translation
        self._rotation    = rotation
        self._prob        = prob
        self._taper       = taper


    def __getattr__(self, name):
        try:
            return getattr(self, name)
        except:
            raise AttributeError(f'{name} is not a valid attribute.')


    def __str__(self):
        return "".join(f"{k} ({v.shape})\n" for k, v in self.__dict__.items())


    @property
    def params(self):
        return (
            self._size,
            self._shape,
            self._translation,
            self._rotation,
            self._prob,
            self._taper
        )

    @property
    def batch_size(self):
        return self._shape.shape[0]

    @property
    def n_primitives(self):
        return self._shape.shape[1]

    def detach(self):
        return PrimitiveParameters(
            self._shape.detach(),
            self._size.detach(),
            self._translation.detach(),
            self._rotation.detach(),
            self._prob.detach(),
            self._taper.detach() if self._taper is not None else None,
        )
