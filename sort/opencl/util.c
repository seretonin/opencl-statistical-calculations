#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>

#include <string.h>
#include <assert.h>

#include "util.h"

/* int debug(const char *format, ...) */
/* { */
/*     va_list args; */
/*     va_start(args, format); */
/*     vprintf(format, args); */
/*     va_end(args); */
/*     return 0; */
/* } */

/*
 * return true if the string contain the character otherwith return false
*/
static bool contain_char(char *s, char c)
{
    for (; *s != '\0'; ++s)
    {
        if (*s == c)
        {
            return true;
        }
    }
    return false;
}

void *Malloc(size_t size)
{
	char *ptr = calloc(1, size);
	if (ptr == NULL)
	{
		perror("malloc error:");
		exit(1);
	}
	return ptr;
}

void *Realloc(void *ptr, size_t size)
{
	assert(ptr);
	char *ret = realloc(ptr, size);
	if (!ret)
	{
		perror("realloc error");
		exit(1);
	}
	return ret;
}

char *readFile(const char *fname)
{
    FILE *fp = fopen(fname, "rb");

	assert(fp != NULL);
	fseek(fp, 0, SEEK_END);
	int length = ftell(fp);
	char *data = (char *)malloc((length + 1) * sizeof(char));
	rewind(fp);
	fread(data, 1, length, fp);
	data[length] = '\0';
	fclose(fp);
	return data;
}


/*
 * string utilities
 */

//check if the ``ss'' starts with ``prefix''.
bool strstartswith(const char *str, const char *prefix)
{
    if (strncmp(str, prefix, strlen(prefix)))
    {
        return false;
    }
    else
    {
        return true;
    }
}

//check if the ``ss'' ends with ``prefix''.
bool strendswith(const char *str, const char *suffix)
{
    if(strlen(str) < strlen(suffix))
        return false;
    const char *p = str + (strlen(str) - strlen(suffix));
    return strstartswith(p, suffix);
}

int strcount(char *ss, char c)
{
    int count = 0;
    for(; *ss != '\0'; ss++)
    {
        if(*ss == c)
        {
            count++;
        }
    }
    return count;
}

/*
 * strip the characters contained in d_chars at the beginning of the string
 */
char *strlstrip(char *str, char *d_chars)
{
    for (; *str != '\0'; ++str)
    {
        char c = *str;
        if (!contain_char(d_chars, c))
        {
            break;
        }
    }
    return str;
}

char *strrstrip(char *str, char *d_chars)
{
    char *end = str + strlen(str) - 1;
    for (; end >= str; --end)
    {
        char c = *end;
        if (!contain_char(d_chars, c))
        {
            break;
        }
    }
    *(++end) = '\0';
    return str;
}

char *strstrip(char *str, char *d_chars)
{
    char *start = strlstrip(str, d_chars);
    return strrstrip(start, d_chars);
}
