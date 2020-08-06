#ifndef NG_SIMD_NIINTEG_H
#define NG_SIMD_NIINTEG_H

#include "ngspice/SIMD/simdvector.h"

int
vecN_NIintegrate(CKTcircuit *ckt, double *geq, double *ceq, double cap, VecNm qcap);

#endif
