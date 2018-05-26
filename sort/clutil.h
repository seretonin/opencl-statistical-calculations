#ifndef _CLUTIL_H_
#define _CLUTIL_H_ 1

const char *clGetErrorString(cl_int error);
void clCheckEqWithMsg(int status, int expect, char *msg);
void clCheckNeqWithMsg(int status, int expect, char *msg);
void clCheckLtWithMsg(int status, int expect, char *msg);
void clCheckGtWithMsg(int status, int expect, char *msg);

cl_context CreateContext();

#endif /* _CLUTIL_H_ */

