/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"

int
VSRCbindklu(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->VSRCnextModel ) {
	VSRCinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
	    here = here->VSRCnextInstance) {

		i = 0 ;
		if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCposIbrptr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->VSRCposIbrptr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCnegIbrptr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->VSRCnegIbrptr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0)) {
			while (here->VSRCibrNegptr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->VSRCibrNegptr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0)) {
			while (here->VSRCibrPosptr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->VSRCibrPosptr = ckt->CKTkluBind_KLU [i] ;
		}
	}
    }
    return(OK);
}

int
VSRCbindkluComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->VSRCnextModel ) {
	VSRCinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
	    here = here->VSRCnextInstance) {

		i = 0 ;
		if ((here->VSRCposNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCposIbrptr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->VSRCposIbrptr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCnegNode != 0) && (here->VSRCbranch != 0)) {
			while (here->VSRCnegIbrptr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->VSRCnegIbrptr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCnegNode != 0)) {
			while (here->VSRCibrNegptr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->VSRCibrNegptr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->VSRCbranch != 0) && (here->VSRCposNode != 0)) {
			while (here->VSRCibrPosptr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->VSRCibrPosptr = ckt->CKTkluBind_KLU_Complex [i] ;
		}
	}
    }
    return(OK);
}
