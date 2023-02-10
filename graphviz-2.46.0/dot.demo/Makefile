CFLAGS=`pkg-config libgvc --cflags` -Wall -g -O2
LDFLAGS=`pkg-config libgvc --libs`

.PHONY: all
all: dot demo simple example neatopack

dot: dot.o
demo: demo.o
simple: simple.o
example: example.o
neatopack: neatopack.o

.PHONY: clean
clean:
	rm -rf *.o dot demo simple example neatopack *.png
