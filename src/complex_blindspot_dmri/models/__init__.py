from .complex_cnn import GuidedComplexConvNet
from .complex_nafnet import GuidedComplexNAFNet

ComplexNAFNet = GuidedComplexNAFNet
ComplexCNN = GuidedComplexConvNet

__all__ = [
    "ComplexNAFNet",
    "ComplexCNN",
    "GuidedComplexNAFNet",
    "GuidedComplexConvNet",
]
