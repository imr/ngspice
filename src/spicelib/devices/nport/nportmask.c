/**********
Enhancement-242: native n-port device -- model query.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTmAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NG_IGNORE(ckt);

    switch (which) {
    case NPORT_MOD_FILE:
        value->sValue = model->NPORTfile;
        return OK;
    case NPORT_NPORTS:
        value->iValue = model->NPORTnPorts;
        return OK;
    case NPORT_NPOLES:
        value->iValue = model->NPORTnPoles;
        return OK;
    default:
        return E_BADPARM;
    }
}
