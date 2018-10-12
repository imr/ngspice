#ifndef ngspice_COMPATMODE_H
#define ngspice_COMPATMODE_H

#include "ngspice/config.h"

typedef enum {
  COMPATMODE_NATIVE = 0,
  COMPATMODE_HS = 1,
  COMPATMODE_SPICE3 = 2,
  COMPATMODE_ALL = 3,
  COMPATMODE_PS = 4,
  COMPATMODE_PSA = 5,
  COMPATMODE_LT = 6,
  COMPATMODE_LTA = 7,
  COMPATMODE_LTPS = 8,
  COMPATMODE_LTPSA = 9
} COMPATMODE_T ;

extern COMPATMODE_T inp_compat_mode;

#endif
