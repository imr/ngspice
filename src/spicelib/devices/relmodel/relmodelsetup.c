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
    }

    return (OK) ;
}
