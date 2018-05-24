CC = gcc
AR = ar
RANLIB = ranlib

CFLAGS = -g -Wall -std=c99
DEBUG = -DDEBUG

OPENCL_INC_PATH = -I/opt/AMDAPPSDK-2.9-1/include
OPENCL_LIB_PATH = -L/opt/AMDAPPSDK-2.9-1/lib/x86_64

EXE_FILE = bsort
SOURCES = bsort.c util.c clutil.c
OBJS = bsort.o util.o clutil.o

all: $(EXE_FILE)

$(EXE_FILE): $(OBJS)
	$(CC) $(OPENCL_LIB_PATH) -o $@ $^ -lOpenCL

%.o:%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@ $(OPENCL_INC_PATH) $(DEBUG)

# ignore the warn message "XXX.d: No such file or directory"
-include $(SOURCES:.c=.d)
%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$


install: all
	@if [ -d $(INSTDIR) ]; \
	then \
	cp myapp $(INSTDIR);\
	chmod a+x $(INSTDIR)/$(EXE_FILE);\
	chmod og-w $(INSTDIR)/$(EXE_FILE);\
	echo “Installed in $(INSTDIR)“;\
	else \
	echo “Sorry, $(INSTDIR) does not exist”;\
	fi

.PHONY: clean
clean:
	-rm -f *.o $(EXE_FILE) *.d *.d.*
