/**********
Enhancement-242: native n-port device -- AC load.

Stamps the complex admittance Y_ij(jw) directly (multi-terminal conductance)
into the (real, imag) matrix slots -- the (ptr, ptr+1) convention used by the
RLC / OSDI devices.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"

extern void NPORTadmittance(NPORTmodel *, int, int, double, double,
                            double *, double *);

int
NPORTacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    int i, j, N;
    double w = ckt->CKTomega;

    for (; model; model = NPORTnextModel(model)) {
        N = model->NPORTnPorts;
        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    double yr, yi;
                    NPORTadmittance(model, i, j, 0.0, w, &yr, &yi);
                    *(here->NPORTyPtr[i * N + j])     += yr;
                    *(here->NPORTyPtr[i * N + j] + 1) += yi;
                    *(here->NPORTyColPtr[i])          += -yr;
                    *(here->NPORTyColPtr[i] + 1)      += -yi;
                    *(here->NPORTyRowPtr[j])          += -yr;
                    *(here->NPORTyRowPtr[j] + 1)      += -yi;
                    *(here->NPORTyRefPtr)             += yr;
                    *(here->NPORTyRefPtr + 1)         += yi;
                }
            }
        }
    }
    return OK;
}
