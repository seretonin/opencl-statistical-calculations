CC = gcc
AR = ar
RANLIB = ranlib

CFLAGS = -g -Wall -std=c99
DEBUG = -DDEBUG

OPENCL_INC_PATH = -DOpenCL_INCLUDE_DIR=/usr/include/CL
OPENCL_LIB_PATH = /usr/lib/x86_64-linux-gnu/libOpenCL.so

OUTPUT_FILE = a
SOURCES = bsort.c util.c
OBJS = bsort.o util.o

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(OBJS)
	$(CC) $(OPENCL_LIB_PATH) -o $@ $^ -lOpenCL -lm

%.o:%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@ $(OPENCL_INC_PATH) $(DEBUG)

.PHONY: clean
clean:
	-rm -f *.o $(OUTPUT_FILE) *.d *.d.*
