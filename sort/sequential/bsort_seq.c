#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define INPUT_LENGTH 1000000
#define SWAP(x,y) t = x; x = y; y = t;

double getTime();
void populateArray(int);
void compare();
void bitonicmerge(int, int, int);
void recbitonic(int, int, int);
void sort();

int data[INPUT_LENGTH];
int up = 1;
int down = 0;

int main()
{
	double t1 = 0.0;
	double t2 = 0.0; 	
	populateArray(INPUT_LENGTH);
	t1 = getTime();
	sort();
	int median = data[INPUT_LENGTH/2];
	int min = data[0];
	int max = data[INPUT_LENGTH-1];
	t2 = getTime(); 
	
	for (int i = 0;i < INPUT_LENGTH;i++)
	{
		printf("%d \t", data[i]);
		if (i%5 == 0) {
			printf("\n");
		}
	}
	
	printf("\n-----------------------------------------\n");
	printf("INPUT_LENGTH : %d\n",INPUT_LENGTH);
	printf("median       : %d\n",median);
	printf("min          : %d\n",min);
	printf("max          : %d\n",max);
	printf("time taken   : %6.5f secs\n",(t2 - t1));		
	printf("-------------------------------------------\n");	
}


double getTime(){
  struct timeval t;
  double sec, msec;
  
	while (gettimeofday(&t, NULL) != 0);
	sec = t.tv_sec;
	msec = t.tv_usec;
	sec = sec + msec/1000000.0;

	return sec;
}

void populateArray(int array_length){
    srand((unsigned)time(NULL));
    for(int i = 0; i < array_length; i++){
        data[i] = rand();
    }
}

void compare(int i, int j, int dir)
{
	int t;

	if (dir == (data[i] > data[j]))
	{
		SWAP(data[i], data[j]);
	}
}

/*
 * Sorts a bitonic sequence in ascending order if dir=1
 * otherwise in descending order
 */
void bitonicmerge(int low, int c, int dir)
{
	int k, i;

	if (c > 1)
	{
		k = c / 2;
		for (i = low;i < low+k ;i++)
			compare(i, i+k, dir);    
			bitonicmerge(low, k, dir);
			bitonicmerge(low+k, k, dir);    
	}
}

/*
 * Generates bitonic sequence by sorting recursively
 * two halves of the array in opposite sorting orders
 * bitonicmerge will merge the resultant data
 */
void recbitonic(int low, int c, int dir)
{
	int k;
	if (c > 1)
	{
		k = c / 2;
		recbitonic(low, k, up);
		recbitonic(low + k, k, down);
		bitonicmerge(low, c, dir);
	}
}

/* 
 * Sorts the entire array
 */
void sort()
{
	recbitonic(0, INPUT_LENGTH, up);
}
