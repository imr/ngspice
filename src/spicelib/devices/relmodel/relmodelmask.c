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
        // RELMODEL type
        case RELMODEL_MOD_TYPE:
            value->iValue = model->RELMODELtype ;
            return (OK) ;

        // RELMODEL flag
        case RELMODEL_MOD_RELMODEL:
            value->iValue = model->RELMODELrelmodel ;
            return (OK) ;

        // 1: Giorgio Liatis (Sapienza University - DIET - Ultra Thin Oxide)
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

        // 2: 65nm found over Internet
        case RELMODEL_MOD_ALPHA_NEW:
            value->rValue = model->RELMODELalpha_new ;
            return (OK) ;

        case RELMODEL_MOD_B_NEW:
            value->rValue = model->RELMODELb_new ;
            return (OK) ;

        case RELMODEL_MOD_BETA_NEW:
            value->rValue = model->RELMODELbeta_new ;
            return (OK) ;

        case RELMODEL_MOD_E0_NEW:
            value->rValue = model->RELMODELe0_new ;
            return (OK) ;

        case RELMODEL_MOD_K_NEW:
            value->rValue = model->RELMODELk_new ;
            return (OK) ;

        case RELMODEL_MOD_KB_NEW:
            value->rValue = model->RELMODELkb_new ;
            return (OK) ;

        case RELMODEL_MOD_TAU_C_FAST_NEW:
            value->rValue = model->RELMODELtau_c_fast_new ;
            return (OK) ;

        case RELMODEL_MOD_TAU_C_SLOW_NEW:
            value->rValue = model->RELMODELtau_c_slow_new ;
            return (OK) ;

        case RELMODEL_MOD_TAU_E_FAST_NEW:
            value->rValue = model->RELMODELtau_e_fast_new ;
            return (OK) ;

        case RELMODEL_MOD_TAU_E_SLOW_NEW:
            value->rValue = model->RELMODELtau_e_slow_new ;
            return (OK) ;

        case RELMODEL_MOD_T_R_NEW:
            value->rValue = model->RELMODELt_r_new ;
            return (OK) ;

        case RELMODEL_MOD_T_S_NEW:
            value->rValue = model->RELMODELt_s_new ;
            return (OK) ;

        default:
            return (E_BADPARM) ;
    }
}
