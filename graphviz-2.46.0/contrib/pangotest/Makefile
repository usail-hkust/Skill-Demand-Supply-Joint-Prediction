CFLAGS = `pkg-config --cflags pangocairo` -Wall -O0 -g
LDFLAGS = `pkg-config --libs pangocairo`

pangotest: pangotest.c

.PHONY: test
test: pangotest
	./pangotest Ay Times 64 >`uname -s`.png

.PHONY: clean
clean:
	rm -f pangotest *.png
