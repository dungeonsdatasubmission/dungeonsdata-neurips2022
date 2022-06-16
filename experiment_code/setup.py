import time

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="hackrl",
        version="0.1.%s" % time.strftime("%Y%m%d.%H%M%S"),
        description="hackrl.",
        python_requires=">=3.7",
        packages=["hackrl"],
    )
