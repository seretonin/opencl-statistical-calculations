#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MAX 1000000
#define SWAP(x,y) t = x; x = y; y = t;

double getTime();
void populateArray(int);
void compare();
void bitonicmerge(int, int, int);
void recbitonic(int, int, int);
void sort();

int data[MAX];
int up = 1;
int down = 0;

int main()
{
	double t1 = 0.0;
	double t2 = 0.0; 	
	populateArray(MAX);
	t1 = getTime();
	sort();
	t2 = getTime(); 
	
	for (int i = 0;i < MAX;i++)
	{
		printf("%d \n", data[i]);
	}
	printf("time: %6.2f secs\n",(t2 - t1));
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
	recbitonic(0, MAX, up);
}
