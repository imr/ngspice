/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "relmodeldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
RELMODELmAsk (CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    RELMODELmodel *model = (RELMODELmodel *)inst ;

    NG_IGNORE (ckt) ;

    switch (which)
    {
        case RELMODEL_MOD_T0:
            value->rValue = model->RELMODELt0 ;
            return (OK) ;

        case RELMODEL_MOD_EA:
            value->rValue = model->RELMODELea ;
            return (OK) ;

        case RELMODEL_MOD_K1_2:
            value->rValue = model->RELMODELk1_2 ;
            return (OK) ;

        case RELMODEL_MOD_E_01:
            value->rValue = model->RELMODELe_01 ;
            return (OK) ;

        case RELMODEL_MOD_X1:
            value->rValue = model->RELMODELx1 ;
            return (OK) ;

        case RELMODEL_MOD_X2:
            value->rValue = model->RELMODELx2 ;
            return (OK) ;

        case RELMODEL_MOD_T_CLK:
            value->rValue = model->RELMODELt_clk ;
            return (OK) ;

        case RELMODEL_MOD_ALFA:
            value->rValue = model->RELMODELalfa ;
            return (OK) ;

        case RELMODEL_MOD_RELMODEL:
            value->iValue = model->RELMODELrelmodel ;
            return (OK) ;

        default:
            return (E_BADPARM) ;
    }
}
