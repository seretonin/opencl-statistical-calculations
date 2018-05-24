CC = gcc
AR = ar
RANLIB = ranlib

CFLAGS = -g -std=c99
DEBUG = -DDEBUG

OPENCL_INC_PATH = -I/opt/AMDAPPSDK-2.9-1/include
OPENCL_LIB_PATH = -L/opt/AMDAPPSDK-2.9-1/lib/x86_64

OUTPUT_FILE = b
SOURCES = bsort_seq.c
OBJS = bsort_seq.o

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(OBJS)
	$(CC) $(OPENCL_LIB_PATH) -o $@ $^ -lOpenCL

%.o:%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@ $(OPENCL_INC_PATH) $(DEBUG)

.PHONY: clean
clean:
	-rm -f *.o $(OUTPUT_FILE) *.d *.d.*
