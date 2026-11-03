/**********
Enhancement-242: native n-port device -- KLU CSC binding.

After the KLU reorder, re-point each stamped matrix element from its Sparse (COO)
location to the CSC slot, and support the complex<->real toggling used by AC.  The
built-in devices do this with the named-field CREATE_KLU_BINDING_TABLE macros; this
device stamps through pointer ARRAYS, so the same bsearch/replace logic is applied
element-by-element here.  A ground row/col (node index 0) is left unbound -- its
pointer stays the valid Sparse trash location and its stamp is harmlessly ignored,
exactly as for a grounded RLC terminal.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

/* Bind one COO pointer to its CSC slot.  row/col are the 1-based node numbers of
 * this element; a 0 (ground) is skipped.  Returns the matched BindElement (NULL if
 * skipped or not found), and rewrites *pptr to the CSC location on success. */
static BindElement *
nport_bind(BindElement *BindStruct, size_t nz, double **pptr, int row, int col)
{
    BindElement key, *matched;
    if (row <= 0 || col <= 0 || *pptr == NULL)
        return NULL;                       /* ground element: keep Sparse pointer */
    key.COO = *pptr; key.CSC = NULL; key.CSC_Complex = NULL;
    matched = (BindElement *) bsearch(&key, BindStruct, nz, sizeof(BindElement), BindCompare);
    if (matched == NULL) {
        printf("nport: Ptr %p not found in KLU bind table\n", (void *) *pptr);
        return NULL;
    }
    *pptr = matched->CSC;
    return matched;
}

int
NPORTbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    BindElement *BindStruct;
    size_t nz;
    int i, j, N, ref;
    int *node;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO;
    nz = (size_t) ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ;

    for (; model; model = NPORTnextModel(model)) {
        N = model->NPORTnPorts;
        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {
            node = GENnode(&here->gen);
            ref  = here->NPORTrefNode;
            for (i = 0; i < N; i++) {
                here->NPORTyColBind[i] =
                    nport_bind(BindStruct, nz, &here->NPORTyColPtr[i], node[i], ref);
                here->NPORTyRowBind[i] =
                    nport_bind(BindStruct, nz, &here->NPORTyRowPtr[i], ref, node[i]);
                for (j = 0; j < N; j++)
                    here->NPORTyBind[i * N + j] =
                        nport_bind(BindStruct, nz, &here->NPORTyPtr[i * N + j], node[i], node[j]);
            }
            here->NPORTyRefBind =
                nport_bind(BindStruct, nz, &here->NPORTyRefPtr, ref, ref);
        }
    }
    return OK;
}

int
NPORTbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    int i, j, N;

    NG_IGNORE(ckt);

    for (; model; model = NPORTnextModel(model)) {
        N = model->NPORTnPorts;
        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {
            for (i = 0; i < N; i++) {
                if (here->NPORTyColBind[i])
                    here->NPORTyColPtr[i] = here->NPORTyColBind[i]->CSC_Complex;
                if (here->NPORTyRowBind[i])
                    here->NPORTyRowPtr[i] = here->NPORTyRowBind[i]->CSC_Complex;
                for (j = 0; j < N; j++)
                    if (here->NPORTyBind[i * N + j])
                        here->NPORTyPtr[i * N + j] = here->NPORTyBind[i * N + j]->CSC_Complex;
            }
            if (here->NPORTyRefBind)
                here->NPORTyRefPtr = here->NPORTyRefBind->CSC_Complex;
        }
    }
    return OK;
}

int
NPORTbindCSCComplexToReal(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    int i, j, N;

    NG_IGNORE(ckt);

    for (; model; model = NPORTnextModel(model)) {
        N = model->NPORTnPorts;
        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {
            for (i = 0; i < N; i++) {
                if (here->NPORTyColBind[i])
                    here->NPORTyColPtr[i] = here->NPORTyColBind[i]->CSC;
                if (here->NPORTyRowBind[i])
                    here->NPORTyRowPtr[i] = here->NPORTyRowBind[i]->CSC;
                for (j = 0; j < N; j++)
                    if (here->NPORTyBind[i * N + j])
                        here->NPORTyPtr[i * N + j] = here->NPORTyBind[i * N + j]->CSC;
            }
            if (here->NPORTyRefBind)
                here->NPORTyRefPtr = here->NPORTyRefBind->CSC;
        }
    }
    return OK;
}
