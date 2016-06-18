/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

#include <stdlib.h>


static
int
BindCompare (const void *a, const void *b)
{
    BindElement *A, *B ;
    A = (BindElement *)a ;
    B = (BindElement *)b ;

    return ((int)(A->Sparse - B->Sparse)) ;
}

int
CPLbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CPLmodel *model = (CPLmodel *)inModel ;
    CPLinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;
    int m, p ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CPL models */
    for ( ; model != NULL ; model = model->CPLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CPLinstances ; here != NULL ; here = here->CPLnextInstance)
        {
            for (m = 0; m < here->dimension; m++) {
                CREATE_KLU_BINDING_TABLE(CPLibr1Ibr1Ptr[m], CPLibr1Ibr1Binding[m], CPLibr1[m], CPLibr1[m]);
                CREATE_KLU_BINDING_TABLE(CPLibr2Ibr2Ptr[m], CPLibr2Ibr2Binding[m], CPLibr2[m], CPLibr2[m]);
                CREATE_KLU_BINDING_TABLE(CPLposIbr1Ptr[m], CPLposIbr1Binding[m], CPLposNodes[m], CPLibr1[m]);
                CREATE_KLU_BINDING_TABLE(CPLnegIbr2Ptr[m], CPLnegIbr2Binding[m], CPLnegNodes[m], CPLibr2[m]);
                CREATE_KLU_BINDING_TABLE(CPLposPosPtr[m], CPLposPosBinding[m], CPLposNodes[m], CPLposNodes[m]);
                CREATE_KLU_BINDING_TABLE(CPLnegNegPtr[m], CPLnegNegBinding[m], CPLnegNodes[m], CPLnegNodes[m]);
                CREATE_KLU_BINDING_TABLE(CPLnegPosPtr[m], CPLnegPosBinding[m], CPLnegNodes[m], CPLposNodes[m]);
                CREATE_KLU_BINDING_TABLE(CPLposNegPtr[m], CPLposNegBinding[m], CPLposNodes[m], CPLnegNodes[m]);

                for (p = 0; p < here->dimension; p++) {
                    CREATE_KLU_BINDING_TABLE(CPLibr1PosPtr[m][p], CPLibr1PosBinding[m][p], CPLibr1[m], CPLposNodes[p]);
                    CREATE_KLU_BINDING_TABLE(CPLibr2NegPtr[m][p], CPLibr2NegBinding[m][p], CPLibr2[m], CPLnegNodes[p]);
                    CREATE_KLU_BINDING_TABLE(CPLibr1NegPtr[m][p], CPLibr1NegBinding[m][p], CPLibr1[m], CPLnegNodes[p]);
                    CREATE_KLU_BINDING_TABLE(CPLibr2PosPtr[m][p], CPLibr2PosBinding[m][p], CPLibr2[m], CPLposNodes[p]);
                    CREATE_KLU_BINDING_TABLE(CPLibr1Ibr2Ptr[m][p], CPLibr1Ibr2Binding[m][p], CPLibr1[m], CPLibr2[p]);
                    CREATE_KLU_BINDING_TABLE(CPLibr2Ibr1Ptr[m][p], CPLibr2Ibr1Binding[m][p], CPLibr2[m], CPLibr1[p]);
                }
            }
        }
    }

    return (OK) ;
}

int
CPLbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CPLmodel *model = (CPLmodel *)inModel ;
    CPLinstance *here ;
    int m, p;

    NG_IGNORE (ckt) ;

    /* loop through all the CPL models */
    for ( ; model != NULL ; model = model->CPLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CPLinstances ; here != NULL ; here = here->CPLnextInstance)
        {
            for (m = 0; m < here->dimension; m++) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr1Ibr1Ptr[m], CPLibr1Ibr1Binding[m], CPLibr1[m], CPLibr1[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr2Ibr2Ptr[m], CPLibr2Ibr2Binding[m], CPLibr2[m], CPLibr2[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLposIbr1Ptr[m], CPLposIbr1Binding[m], CPLposNodes[m], CPLibr1[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLnegIbr2Ptr[m], CPLnegIbr2Binding[m], CPLnegNodes[m], CPLibr2[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLposPosPtr[m], CPLposPosBinding[m], CPLposNodes[m], CPLposNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLnegNegPtr[m], CPLnegNegBinding[m], CPLnegNodes[m], CPLnegNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLnegPosPtr[m], CPLnegPosBinding[m], CPLnegNodes[m], CPLposNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLposNegPtr[m], CPLposNegBinding[m], CPLposNodes[m], CPLnegNodes[m]);

                for (p = 0; p < here->dimension; p++) {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr1PosPtr[m][p], CPLibr1PosBinding[m][p], CPLibr1[m], CPLposNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr2NegPtr[m][p], CPLibr2NegBinding[m][p], CPLibr2[m], CPLnegNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr1NegPtr[m][p], CPLibr1NegBinding[m][p], CPLibr1[m], CPLnegNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr2PosPtr[m][p], CPLibr2PosBinding[m][p], CPLibr2[m], CPLposNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr1Ibr2Ptr[m][p], CPLibr1Ibr2Binding[m][p], CPLibr1[m], CPLibr2[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CPLibr2Ibr1Ptr[m][p], CPLibr2Ibr1Binding[m][p], CPLibr2[m], CPLibr1[p]);
                }
            }
        }
    }

    return (OK) ;
}

int
CPLbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CPLmodel *model = (CPLmodel *)inModel ;
    CPLinstance *here ;
    int m, p;

    NG_IGNORE (ckt) ;

    /* loop through all the CPL models */
    for ( ; model != NULL ; model = model->CPLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CPLinstances ; here != NULL ; here = here->CPLnextInstance)
        {
            for (m = 0; m < here->dimension; m++) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr1Ibr1Ptr[m], CPLibr1Ibr1Binding[m], CPLibr1[m], CPLibr1[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr2Ibr2Ptr[m], CPLibr2Ibr2Binding[m], CPLibr2[m], CPLibr2[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLposIbr1Ptr[m], CPLposIbr1Binding[m], CPLposNodes[m], CPLibr1[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLnegIbr2Ptr[m], CPLnegIbr2Binding[m], CPLnegNodes[m], CPLibr2[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLposPosPtr[m], CPLposPosBinding[m], CPLposNodes[m], CPLposNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLnegNegPtr[m], CPLnegNegBinding[m], CPLnegNodes[m], CPLnegNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLnegPosPtr[m], CPLnegPosBinding[m], CPLnegNodes[m], CPLposNodes[m]);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLposNegPtr[m], CPLposNegBinding[m], CPLposNodes[m], CPLnegNodes[m]);

                for (p = 0; p < here->dimension; p++) {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr1PosPtr[m][p], CPLibr1PosBinding[m][p], CPLibr1[m], CPLposNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr2NegPtr[m][p], CPLibr2NegBinding[m][p], CPLibr2[m], CPLnegNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr1NegPtr[m][p], CPLibr1NegBinding[m][p], CPLibr1[m], CPLnegNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr2PosPtr[m][p], CPLibr2PosBinding[m][p], CPLibr2[m], CPLposNodes[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr1Ibr2Ptr[m][p], CPLibr1Ibr2Binding[m][p], CPLibr1[m], CPLibr2[p]);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(CPLibr2Ibr1Ptr[m][p], CPLibr2Ibr1Binding[m][p], CPLibr2[m], CPLibr1[p]);
                }
            }
        }
    }

    return (OK) ;
}
