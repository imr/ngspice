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
    for( ; model != NULL; model = model->RESnextModel ) {

        /* Default Value Processing for Resistor Models */
        if(!model->REStnomGiven) model->REStnom         = ckt->CKTnomTemp;
        if(!model->RESsheetResGiven) model->RESsheetRes = 0.0;
        if(!model->RESdefWidthGiven) model->RESdefWidth = 10e-6; /*M*/
        if(!model->RESdefLengthGiven) model->RESdefLength = 10e-6;
        if(!model->REStc1Given) model->REStempCoeff1    = 0.0;
        if(!model->REStc2Given) model->REStempCoeff2    = 0.0;
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
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

            if(!here->RESwidthGiven)   here->RESwidth  = model->RESdefWidth;
            if(!here->RESlengthGiven)  here->RESlength = model->RESdefLength;
            if(!here->RESscaleGiven)   here->RESscale  = 1.0;
            if(!here->RESmGiven)       here->RESm      = 1.0;
            if(!here->RESnoisyGiven)   here->RESnoisy  = 1;
            if(!here->RESresGiven)  {
                if(here->RESlength * here->RESwidth * model->RESsheetRes > 0.0) {
                    here->RESresist = model->RESsheetRes * (here->RESlength -
                                                            model->RESshort) / (here->RESwidth - model->RESnarrow);
                } else {
                    if(model->RESresGiven) {
                        here->RESresist = model->RESres;
                    } else {
                        SPfrontEnd->IFerrorf (ERR_WARNING,
                                             "%s: resistance to low, set to 1 mOhm", here->RESname);
                        here->RESresist = 1e-03;
                    }
                }
            }

            if(!here->RESbv_maxGiven)
                here->RESbv_max = model->RESbv_max;

            if((here->RESwidthGiven)||(here->RESlengthGiven))
                here->RESeffNoiseArea = pow((here->RESlength-model->RESshort),model->RESlf)
                                       *pow((here->RESwidth-model->RESnarrow),model->RESwf);
            else
                here->RESeffNoiseArea = 1.0;

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(RESposPosptr, RESposNode, RESposNode);
            TSTALLOC(RESnegNegptr, RESnegNode, RESnegNode);
            TSTALLOC(RESposNegptr, RESposNode, RESnegNode);
            TSTALLOC(RESnegPosptr, RESnegNode, RESposNode);
            if (here->RESbrEq & (here->RESbrptr == NULL))
                TSTALLOC(RESbrptr, RESbrEq, RESbrEq);
        }
    }
    return(OK);
}
