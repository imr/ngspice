/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CAPsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the capacitor structure with those pointers needed later
         * for fast matrix loading
         */

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

        /*Default Value Processing for Model Parameters */
        if (!model->CAPmCapGiven) {
            model->CAPmCap = 0.0;
        }
        if (!model->CAPcjswGiven){
             model->CAPcjsw = 0.0;
        }
        if (!model->CAPdefWidthGiven) {
            model->CAPdefWidth = 10.e-6;
        }
        if (!model->CAPdefLengthGiven) {
            model->CAPdefLength = 0.0;
        }
        if (!model->CAPnarrowGiven) {
            model->CAPnarrow = 0.0;
        }
        if (!model->CAPshortGiven) {
            model->CAPshort = 0.0;
        }
        if (!model->CAPdelGiven) {
            model->CAPdel = 0.0;
        }
        if (!model->CAPtc1Given) {
            model->CAPtempCoeff1 = 0.0;
        }
        if (!model->CAPtc2Given) {
            model->CAPtempCoeff2 = 0.0;
        }
        if (!model->CAPtnomGiven) {
            model->CAPtnom = ckt->CKTnomTemp;
        }
        if (!model->CAPdiGiven) {
            model->CAPdi = 0.0;
        }
        if (!model->CAPthickGiven) {
            model->CAPthick = 0.0;
        }
        if (!model->CAPbv_maxGiven) {
            model->CAPbv_max = 1e99;
        }

        if (!model->CAPcjGiven) {
            if((model->CAPthickGiven)
               && (model->CAPthick > 0.0)) {
               if (model->CAPdiGiven)
                 model->CAPcj = (model->CAPdi * CONSTepsZero) / model->CAPthick;
               else
                 model->CAPcj = CONSTepsSiO2 / model->CAPthick;
            } else {
               model->CAPcj = 0.0;
            }
        }

        if (model->CAPdelGiven) {
            if (!model->CAPnarrowGiven)
                model->CAPnarrow = 2 * model->CAPdel;
            if (!model->CAPshortGiven)
                model->CAPshort = 2 * model->CAPdel;
        }

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ;
                here=CAPnextInstance(here)) {

            /* Default Value Processing for Capacitor Instance */
            if (!here->CAPlengthGiven) {
                here->CAPlength = 0;
            }
            if (!here->CAPbv_maxGiven) {
                here->CAPbv_max = model->CAPbv_max;
            }

            here->CAPqcap = *states;
            *states += CAPnumStates;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += CAPnumSenStates * (ckt->CKTsenInfo->SENparms);
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(CAPposPosPtr,CAPposNode,CAPposNode);
            TSTALLOC(CAPnegNegPtr,CAPnegNode,CAPnegNode);
            TSTALLOC(CAPposNegPtr,CAPposNode,CAPnegNode);
            TSTALLOC(CAPnegPosPtr,CAPnegNode,CAPposNode);
        }
    }
    return(OK);
}

