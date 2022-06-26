/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define PI 3.141592654

static double Lundin(double l, double csec);

int
INDsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
   /* load the inductor structure with those pointers needed later 
   * for fast matrix loading 
   */
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {
 
   /* Default Value Processing for Model Parameters */
        if (!model->INDmIndGiven) {
             model->INDmInd = 0.0;
        }
        if (!model->INDtnomGiven) {
             model->INDtnom = ckt->CKTnomTemp;
        }
        if (!model->INDtc1Given) {
             model->INDtempCoeff1 = 0.0;
        }
        if (!model->INDtc2Given) {
             model->INDtempCoeff2 = 0.0;
        }
        if (!model->INDcsectGiven){
             model->INDcsect = 0.0;
        }
        if (!model->INDdiaGiven){
             model->INDdia = 0.0;
        }
        if (!model->INDlengthGiven) {
             model->INDlength = 0.0;
        }
        if (!model->INDmodNtGiven) {
             model->INDmodNt = 0.0;
        }
        if (!model->INDmuGiven) {
             model->INDmu = 1.0;
        }

        /* diameter takes preference over cross section */
        if (model->INDdiaGiven) {
            model->INDcsect = PI * model->INDdia * model->INDdia / 4.;
        }
          
        /* precompute specific inductance (one turn) */
        if((model->INDlengthGiven) && (model->INDlength > 0.0)) {
             model->INDspecInd = (model->INDmu * CONSTmuZero
		         * model->INDcsect) / model->INDlength;
        } else  {
            model->INDspecInd = 0.0;
        }

        /* Lundin's geometry correction factor */
        if(model->INDlengthGiven && (model->INDdiaGiven || model->INDcsectGiven))
            model->INDspecInd *= Lundin(model->INDlength, model->INDcsect);

/*
        double kl = Lundin(model->INDlength, model->INDcsect);
        double Dl = model->INDlength / sqrt(model->INDcsect / PI) / 2.;
        fprintf(stdout, "Lundin's correction factor %f at l/D %f\n", kl, Dl);
*/

        /* How many turns ? */
        if (!model->INDmIndGiven) 
            model->INDmInd = model->INDmodNt * model->INDmodNt * model->INDspecInd;
	
        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            here->INDflux = *states;
            *states += INDnumStates ;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += INDnumSenStates * (ckt->CKTsenInfo->SENparms);
            }

            if(here->INDbrEq == 0) {
                error = CKTmkCur(ckt,&tmp,here->INDname,"branch");
                if(error) return(error);
                here->INDbrEq = tmp->number;
            }

            here->system = NULL;
            here->system_next_ind = NULL;

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(INDposIbrPtr,INDposNode,INDbrEq);
            TSTALLOC(INDnegIbrPtr,INDnegNode,INDbrEq);
            TSTALLOC(INDibrNegPtr,INDbrEq,INDnegNode);
            TSTALLOC(INDibrPosPtr,INDbrEq,INDposNode);
            TSTALLOC(INDibrIbrPtr,INDbrEq,INDbrEq);
        }
    }
    return(OK);
}

int
INDunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model;
    INDinstance *here;

    for (model = (INDmodel *)inModel; model != NULL;
	    model = INDnextModel(model))
    {
        for (here = INDinstances(model); here != NULL;
                here=INDnextInstance(here))
	{
	    if (here->INDbrEq > 0)
		CKTdltNNum(ckt, here->INDbrEq);
            here->INDbrEq = 0;
	}
    }
    return OK;
}

/* Calculates Nagaoka's coeff. using Lundin's handbook formula.
   D: W. Knight, https://g3ynh.info/zdocs/magnetics/Solenoids.pdf, p. 36 */
static double Lundin(double l, double csec)
{
    /* x = solenoid diam. / length */
    double num, den, kk, x, xx, xxxx;



    if (csec < 1e-12 || l < 1e-6) {
        fprintf(stderr, "Warning: coil geometries too small (< 1um length dimensions),\n");
        fprintf(stderr, "    Lundin's correction factor will not be calculated\n");
        return 1;
    }

    x = sqrt(csec / PI) * 2. / l;

    xx = x * x;
    xxxx = xx * xx;

    if (x < 1) {
        num = 1 + 0.383901 * xx + 0.017108 * xxxx;
        den = 1 + 0.258952 * xx;
        return num / den - 4 * x / (3 * PI);
    }
    else {
        num = (log(4 * x) - 0.5) * (1 + 0.383901 / xx + 0.017108 / xxxx);
        den = 1 + 0.258952 / xx;
        kk = 0.093842 / xx + 0.002029 / xxxx - 0.000801 / (xx * xxxx);
        return 2 * (num / den + kk) / (PI * x);
    }
}

