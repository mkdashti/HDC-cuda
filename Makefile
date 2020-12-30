CC = nvcc
CFLAGS = -lm -std=c++11 -O3 -g -arch=sm_75
TARGET = pim_hdc

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif
# define TEST in the source if we are debugging
ifeq ($(TEST), 1)
	CFLAGS+=-DTEST
endif

# Default

MAX_BIT_DIM ?= 313
MAX_CHANNELS ?= 4
MAX_N ?= 5
MAX_IM_LENGTH ?= 22
HDC_MAX_INPUT ?= 384

CFLAGS+=-DHOST=1
CFLAGS+=-DMAX_BIT_DIM=$(MAX_BIT_DIM) -DMAX_CHANNELS=$(MAX_CHANNELS)
CFLAGS+=-DMAX_N=$(MAX_N) -DMAX_IM_LENGTH=$(MAX_IM_LENGTH)
CFLAGS+=-DHDC_MAX_INPUT=$(HDC_MAX_INPUT)

.PHONY: default all clean

default: $(TARGET)
all: default

SOURCES = $(wildcard src/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(SOURCES))
HEADERS = $(wildcard include/*.h)
INC=-I. -Iinclude

%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) -o $@

clean:
	find . -type f -name '*.o' -delete
	-rm -f $(TARGET)
