OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: main.c
	gcc -I. -g main.c -o main.exe -lm -lOpenCL

clean:
	rm -rf main