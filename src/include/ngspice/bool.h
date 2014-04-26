#ifndef ngspice_BOOL_H
#define ngspice_BOOL_H

//typedef unsigned char bool;
#ifndef COMPILED_BY_NVCC
typedef int bool;
#endif

typedef int BOOL ;

#define BOOLEAN int
#define TRUE  1
#define FALSE 0
#define NO    0
#define YES   1


#endif
