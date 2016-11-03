/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "relmodeldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
RELMODELsetup (SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    RELMODELmodel *model = (RELMODELmodel *)inModel ;
    NG_IGNORE (matrix) ;
    NG_IGNORE (ckt) ;
    NG_IGNORE (states) ;

    /*  loop through all the RELMODEL device models */
    for ( ; model != NULL ; model = model->RELMODELnextModel)
    {
        // RELMODEL type
        // 1: Giorgio Liatis (Sapienza University - DIET - Ultra Thin Oxide)
        // 2: 65nm found over Internet

        if (!model->RELMODELtypeGiven)
        {
            model->RELMODELtype = 2 ;
        }

        // 1: Giorgio Liatis (Sapienza University - DIET - Ultra Thin Oxide)
        if (!model->RELMODELk_bGiven)
        {
            model->RELMODELk_b = 8.6e-5 ;
        }

        if (!model->RELMODELh_cutGiven)
        {
            model->RELMODELh_cut = 1.05e-34 ;
        }

        if (!model->RELMODELntsGiven)
        {
            model->RELMODELnts = 2e13 ;
        }

        if (!model->RELMODELeps_hkGiven)
        {
            model->RELMODELeps_hk = 25 ;
        }

        if (!model->RELMODELeps_SiO2Given)
        {
            model->RELMODELeps_SiO2 = 3.9 ;
        }

        if (!model->RELMODELm_starGiven)
        {
            model->RELMODELm_star = 0.1 * 9.11e-31 ;
        }

        if (!model->RELMODELwGiven)
        {
            model->RELMODELw = 1.5 * 1.6e-19 ;
        }

        if (!model->RELMODELtau_0Given)
        {
            model->RELMODELtau_0 = 1e-11 ;
        }

        if (!model->RELMODELbetaGiven)
        {
            model->RELMODELbeta = 0.373 ;
        }

        if (!model->RELMODELtau_eGiven)
        {
            model->RELMODELtau_e = 0.85e-9 ;
        }

        if (!model->RELMODELbeta1Given)
        {
            model->RELMODELbeta1 = 0.112 ;
        }

        // 2: 65nm found over Internet
        if (!model->RELMODELalpha_newGiven)
        {
            model->RELMODELalpha_new = 0.3 ;
        }

        if (!model->RELMODELb_newGiven)
        {
            model->RELMODELb_new = 0.075 ;
        }

        if (!model->RELMODELbeta_newGiven)
        {
            model->RELMODELbeta_new = 0.5 ;
        }

        if (!model->RELMODELe0_newGiven)
        {
            model->RELMODELe0_new = 0.1897 ;
        }

        if (!model->RELMODELk_newGiven)
        {
            model->RELMODELk_new = 420 ;
        }

        if (!model->RELMODELkb_newGiven)
        {
            model->RELMODELkb_new = 8.6e-5 ;
        }

        if (!model->RELMODELtau_c_fast_newGiven)
        {
            model->RELMODELtau_c_fast_new = 1e-3 ;
        }

        if (!model->RELMODELtau_c_slow_newGiven)
        {
            model->RELMODELtau_c_slow_new = 1e3 ;
        }

        if (!model->RELMODELtau_e_fast_newGiven)
        {
            model->RELMODELtau_e_fast_new = 20e-6 ;
        }

        if (!model->RELMODELtau_e_slow_newGiven)
        {
            model->RELMODELtau_e_slow_new = 1e6 ;
        }

        if (!model->RELMODELt_r_newGiven)
        {
            model->RELMODELt_r_new = 0.0 ;
        }

        if (!model->RELMODELt_s_newGiven)
        {
            model->RELMODELt_s_new = 0.0 ;
        }
    }

    return (OK) ;
}
