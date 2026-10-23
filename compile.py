# reads one Kernel IR json path as argv[1] and writes PTX to stdout.
import sys

from thompiler import prologue
from thompiler.emitter import Emitter
from thompiler.load import load
from thompiler.walk import Walker


def main():
    kernel = load(sys.argv[1])
    emitter = Emitter()

    prologue.emit_param_loads(emitter, kernel)
    Walker(emitter, kernel).body(kernel.body)
    emitter.emit("ret")

    print(emitter.assemble(prologue.header(), prologue.signature(kernel)), end="")


if __name__ == "__main__":
    main()
