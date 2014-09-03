/**********
Author: Francesco Lannutti - August 2014
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
        case RELMODEL_MOD_T0:
            model->RELMODELt0 = value->rValue ;
            model->RELMODELt0Given = TRUE ;
            break ;

        case RELMODEL_MOD_EA:
            model->RELMODELea = value->rValue ;
            model->RELMODELeaGiven = TRUE ;
            break ;

        case RELMODEL_MOD_K1_2:
            model->RELMODELk1_2 = value->rValue ;
            model->RELMODELk1_2Given = TRUE ;
            break ;

        case RELMODEL_MOD_E_01:
            model->RELMODELe_01 = value->rValue ;
            model->RELMODELe_01Given = TRUE ;
            break ;

        case RELMODEL_MOD_X1:
            model->RELMODELx1 = value->rValue ;
            model->RELMODELx1Given = TRUE ;
            break ;

        case RELMODEL_MOD_X2:
            model->RELMODELx2 = value->rValue ;
            model->RELMODELx2Given = TRUE ;
            break ;

        case RELMODEL_MOD_T_CLK:
            model->RELMODELt_clk = value->rValue ;
            model->RELMODELt_clkGiven = TRUE ;
            break ;

        case RELMODEL_MOD_ALFA:
            model->RELMODELalfa = value->rValue ;
            model->RELMODELalfaGiven = TRUE ;
            break ;

        case RELMODEL_MOD_RELMODEL:
            model->RELMODELrelmodel = 1 ;
            model->RELMODELrelmodelGiven = TRUE ;
            break ;

        default:
            return (E_BADPARM) ;
    }

    return (OK) ;
}
