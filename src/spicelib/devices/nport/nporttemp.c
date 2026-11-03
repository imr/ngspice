/**********
Enhancement-242: native n-port device -- temperature.
All fit data is frequency/temperature-independent and loaded in NPORTsetup, so
this is a no-op.  (Kept as a hook for future temperature-scaled fits.)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    NG_IGNORE(inModel);
    NG_IGNORE(ckt);
    return OK;
}
