OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: main.c
	gcc -I. main.c C:\Windows\System32\OpenCL.dll -o main.exe

clean:
	rm -rf main