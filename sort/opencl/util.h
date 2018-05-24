#ifndef _UTIL_H_
#define _UTIL_H_ 1

#define TRUE 1
#define FALSE 0

#define Free(ptr)            \
do{                          \
  if (ptr != NULL){          \
    free(ptr);               \
    ptr = NULL;              \
  }                          \
}while(0)

void *Malloc(size_t size);
void *Realloc(void *ptr, size_t size);

char *readFile(const char *fname);

/*
 * string utilities
 */
bool strstartswith(const char *str, const char *prefix);
bool strendswith(const char *str, const char *suffix);
int strcount(char *ss, char c);

char *strlstrip(char *str, char *d_chars);
char *strrstrip(char *str, char *d_chars);
char *strstrip(char *str, char *d_chars);

#ifdef DEBUG
    #define DEBUG_PRINT(fmt, args...)    fprintf(stderr, fmt, ## args)
#else
    #define DEBUG_PRINT(fmt, args...)    /* do nothing in release build*/
#endif

#endif /* _UTIL_H_ */

