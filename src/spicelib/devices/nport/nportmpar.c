/**********
Enhancement-242: native n-port device -- model parameter parsing.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTmParam(int param, IFvalue *value, GENmodel *inModel)
{
    NPORTmodel *model = (NPORTmodel *)inModel;

    switch (param) {
    case NPORT_MOD_NPORT:
        /* the bare `nport` type keyword -- nothing to store */
        break;
    case NPORT_MOD_FILE:
        model->NPORTfile = strdup(value->sValue);
        model->NPORTfileGiven = TRUE;
        break;
    default:
        return E_BADPARM;
    }
    return OK;
}
