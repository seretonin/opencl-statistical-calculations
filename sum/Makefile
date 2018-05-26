OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: reduceSum.c
	gcc -I. reduceSum.c C:\Windows\System32\OpenCL.dll -o main.exe

clean:
	rm -rf main