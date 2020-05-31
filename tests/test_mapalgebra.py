from mapalgebra import __version__
from mapalgebra import get_bands_of_month

def test_version():
    assert __version__ == '0.1.0'

def test_get_bands_of_month():
    print(get_bands_of_month(1, 432, 1981, 1982))
    print(get_bands_of_month(2, 432, 1981, 1981))
    print(get_bands_of_month(None, 432, 1981, 2016))
    print(get_bands_of_month(None, 432, 1981, 2016))
