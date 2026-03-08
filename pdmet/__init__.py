import os
import sys

_w90dir = os.environ.get("W90DIR")
if _w90dir is None:
    raise EnvironmentError(
        "\n\nW90DIR environment variable is not set."
        "\nPlease add to your ~/.zshrc or ~/.bashrc:"
        "\n\n    export W90DIR=/path/to/wannier90"
    )


# This for using the wannier90 wrapper
sys.path.insert(0, f"{_w90dir}/wrap")
os.environ["LD_LIBRARY_PATH"] = (
    os.environ.get("LD_LIBRARY_PATH", "") + f":{_w90dir}/wrap"
)

__version__ = "1.3.0"

from . import (  # noqa: E402
    localbasis,
    schmidtbasis,
    dmet,
    qcsolvers,
    helper,
    tools,
    lib,
)

__all__ = [
    "localbasis",
    "schmidtbasis",
    "dmet",
    "qcsolvers",
    "helper",
    "tools",
    "lib",
]
