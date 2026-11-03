/**********
Enhancement-242: native n-port device -- instance teardown.
Frees the heap arrays allocated in NPORTsetup.
**********/

#include "ngspice/ngspice.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

int
NPORTdelete(GENinstance *inst)
{
    NPORTinstance *here = (NPORTinstance *)inst;

    if (here->NPORTallocated) {
        tfree(here->NPORTyPtr);
        tfree(here->NPORTyColPtr);
        tfree(here->NPORTyRowPtr);
        tfree(here->NPORTyBind);
        tfree(here->NPORTyColBind);
        tfree(here->NPORTyRowBind);
        here->NPORTallocated = 0;
    }
    return OK;
}
