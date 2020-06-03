#ifndef hicumL2_temp
#define hicumL2_temp
#include "hicum2defs.h"

#ifdef __cplusplus
extern "C" {
#endif
    int hicum_thermal_update(HICUMmodel *, HICUMinstance *, double Temp);
    int HICUMtemp(GENmodel *inModel, CKTcircuit *ckt);
#ifdef __cplusplus
}
#endif

#endif /* hicumL2_temp */