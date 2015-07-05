/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "relmodeldefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"

int
RELMODELmParam (int param, IFvalue *value, GENmodel *inMod)
{
    RELMODELmodel *model = (RELMODELmodel*)inMod ;

    switch (param)
    {
        case RELMODEL_MOD_KB:
            model->RELMODELk_b = value->rValue ;
            model->RELMODELk_bGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_HCUT:
            model->RELMODELh_cut = value->rValue ;
            model->RELMODELh_cutGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_NTS:
            model->RELMODELnts = value->rValue ;
            model->RELMODELntsGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_EPSHK:
            model->RELMODELeps_hk = value->rValue ;
            model->RELMODELeps_hkGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_EPSSIO2:
            model->RELMODELeps_SiO2 = value->rValue ;
            model->RELMODELeps_SiO2Given = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_MSTAR:
            model->RELMODELm_star = value->rValue ;
            model->RELMODELm_starGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_W:
            model->RELMODELw = value->rValue ;
            model->RELMODELwGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_TAU0:
            model->RELMODELtau_0 = value->rValue ;
            model->RELMODELtau_0Given = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_BETA:
            model->RELMODELbeta = value->rValue ;
            model->RELMODELbetaGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_TAUE:
            model->RELMODELtau_e = value->rValue ;
            model->RELMODELtau_eGiven = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_BETA1:
            model->RELMODELbeta1 = value->rValue ;
            model->RELMODELbeta1Given = TRUE ;
            return (OK) ;

        case RELMODEL_MOD_RELMODEL:
            model->RELMODELrelmodel = 1 ;
            model->RELMODELrelmodelGiven = TRUE ;
            break ;

        default:
            return (E_BADPARM) ;
    }

    return (OK) ;
}
