#ifndef _COMPATMODE_H
#define _COMPATMODE_H

#include <config.h>

typedef enum {
  COMPATMODE_NATIVE = 0,
  COMPATMODE_HSPICE = 1,
  COMPATMODE_SPICE3 = 2,
  COMPATMODE_ALL = 3,
} COMPATMODE_T ;

#endif
