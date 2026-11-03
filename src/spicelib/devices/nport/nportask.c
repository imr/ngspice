/**********
Enhancement-242: native n-port device -- instance query.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTask(CKTcircuit *ckt, GENinstance *inst, int which,
         IFvalue *value, IFvalue *select)
{
    NPORTinstance *here = (NPORTinstance *)inst;
    NG_IGNORE(ckt);
    NG_IGNORE(select);

    switch (which) {
    case NPORT_NPORTS:
        value->iValue = here->NPORTn;
        return OK;
    default:
        return E_BADPARM;
    }
}
