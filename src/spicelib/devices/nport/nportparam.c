/**********
Enhancement-242: native n-port device -- instance parameter parsing.
The n-port has no instance parameters (all data comes from the .model fit file),
so this only exists to satisfy the PARSECALL path for `N` instances.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    NG_IGNORE(value);
    NG_IGNORE(inst);
    NG_IGNORE(select);

    switch (param) {
    default:
        return E_BADPARM;
    }
}
