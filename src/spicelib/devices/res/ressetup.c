/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


int
RESsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit*ckt, int *state)
        /* load the resistor structure with those pointers needed later
         * for fast matrix loading
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;

    NG_IGNORE(state);
    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {

        /* Default Value Processing for Resistor Models */
        if(!model->REStnomGiven) model->REStnom         = ckt->CKTnomTemp;
        if(!model->RESsheetResGiven) model->RESsheetRes = 0.0;
        if(!model->RESdefWidthGiven) model->RESdefWidth = 10e-6; /*M*/
        if(!model->RESdefLengthGiven) model->RESdefLength = 10e-6;
        if(!model->REStc1Given) model->REStempCoeff1    = 0.0;
        if(!model->REStc2Given) model->REStempCoeff2    = 0.0;
        if(!model->REStceGiven) model->REStempCoeffe    = 0.0;
        if(!model->RESnarrowGiven) model->RESnarrow     = 0.0;
        if(!model->RESshortGiven) model->RESshort       = 0.0;
        if(!model->RESfNcoefGiven) model->RESfNcoef     = 0.0;
        if(!model->RESfNexpGiven) model->RESfNexp       = 1.0;
        if(!model->RESlfGiven) model->RESlf             = 1.0;
        if(!model->RESwfGiven) model->RESwf             = 1.0;
        if(!model->RESefGiven) model->RESef             = 1.0;

        if(!model->RESbv_maxGiven)
            model->RESbv_max = 1e99;

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
                here=RESnextInstance(here)) {

            if(!here->RESwidthGiven)   here->RESwidth  = model->RESdefWidth;
            if(!here->RESlengthGiven)  here->RESlength = model->RESdefLength;
            if(!here->RESscaleGiven)   here->RESscale  = 1.0;
            if(!here->RESmGiven)       here->RESm      = 1.0;
            if(!here->RESnoisyGiven)   here->RESnoisy  = 1;

            if(!here->RESbv_maxGiven)
                here->RESbv_max = model->RESbv_max;

            if((here->RESwidthGiven)||(here->RESlengthGiven))
                here->RESeffNoiseArea = pow(here->RESlength - 2 * model->RESshort, model->RESlf)
                                       *pow(here->RESwidth - 2 * model->RESnarrow, model->RESwf);
            else
                here->RESeffNoiseArea = 1.0;

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(RESposPosPtr, RESposNode, RESposNode);
            TSTALLOC(RESnegNegPtr, RESnegNode, RESnegNode);
            TSTALLOC(RESposNegPtr, RESposNode, RESnegNode);
            TSTALLOC(RESnegPosPtr, RESnegNode, RESposNode);
        }
    }
    return(OK);
}
