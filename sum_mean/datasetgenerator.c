#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char** argv)
{
	srand(time(NULL));
	for(int i = 0; i < 1000; i++)
	{
		printf("%d", rand());
	}
}