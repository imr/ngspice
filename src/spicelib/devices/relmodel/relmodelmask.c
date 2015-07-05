/**********
Author: Francesco Lannutti - July 2015
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
        case RELMODEL_MOD_KB:
            value->rValue = model->RELMODELk_b ;
            return (OK) ;

        case RELMODEL_MOD_HCUT:
            value->rValue = model->RELMODELh_cut ;
            return (OK) ;

        case RELMODEL_MOD_NTS:
            value->rValue = model->RELMODELnts ;
            return (OK) ;

        case RELMODEL_MOD_EPSHK:
            value->rValue = model->RELMODELeps_hk ;
            return (OK) ;

        case RELMODEL_MOD_EPSSIO2:
            value->rValue = model->RELMODELeps_SiO2 ;
            return (OK) ;

        case RELMODEL_MOD_MSTAR:
            value->rValue = model->RELMODELm_star ;
            return (OK) ;

        case RELMODEL_MOD_W:
            value->rValue = model->RELMODELw ;
            return (OK) ;

        case RELMODEL_MOD_TAU0:
            value->rValue = model->RELMODELtau_0 ;
            return (OK) ;

        case RELMODEL_MOD_BETA:
            value->rValue = model->RELMODELbeta ;
            return (OK) ;

        case RELMODEL_MOD_TAUE:
            value->rValue = model->RELMODELtau_e ;
            return (OK) ;

        case RELMODEL_MOD_BETA1:
            value->rValue = model->RELMODELbeta1 ;
            return (OK) ;

        case RELMODEL_MOD_RELMODEL:
            value->iValue = model->RELMODELrelmodel ;
            return (OK) ;

        default:
            return (E_BADPARM) ;
    }
}
