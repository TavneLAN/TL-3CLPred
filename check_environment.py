
from __future__ import print_function

import sys


EXPECTED = {
    "keras": "2.1.6",
    "tensorflow": "1.13.1",
    "numpy": "1.19.2",
    "pandas": "0.25.1",
}


def main():
    import keras
    import matplotlib
    import numpy
    import pandas
    import tensorflow as tf
    from keras import backend as K

    actual = {
        "keras": keras.__version__,
        "tensorflow": tf.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }
    print("Python       :", sys.version.replace("\n", " "))
    for name in ("keras", "tensorflow", "numpy", "pandas"):
        status = "OK" if actual[name] == EXPECTED[name] else "EXPECTED " + EXPECTED[name]
        print("{:<12} : {:<12} {}".format(name, actual[name], status))
    print("matplotlib   :", matplotlib.__version__)
    print("Keras backend:", K.backend())
    print("GPU available:", bool(tf.test.is_gpu_available(cuda_only=True)))

    valid = all(actual[name] == version for name, version in EXPECTED.items())
    valid = valid and K.backend() == "tensorflow"
    if not valid:
        print("\nEnvironment versions do not fully match; please fix the EXPECTED items above.")
        return 1
    print("\nCore package versions match. If GPU available=False, check CUDA/cuDNN and the GPU driver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

