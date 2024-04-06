#ifndef ngspice_BOOL_H
#define ngspice_BOOL_H

#if defined (__MINGW32__) || defined (_MSC_VER)
#ifndef __cplusplus
typedef int bool;
#endif
#else
#include <stdbool.h>
#endif

typedef int BOOL;

#define BOOLEAN int
#define TRUE  1
#define FALSE 0
#define NO    0
#define YES   1

#endif
