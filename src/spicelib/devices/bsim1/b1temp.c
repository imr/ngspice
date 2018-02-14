/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
B1temp(GENmodel *inModel, CKTcircuit *ckt)
        /* load the B1 device structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    B1model *model = (B1model*) inModel;
    B1instance *here;
    double  EffChanLength;
    double EffChanWidth;
    double Cox;
    double CoxWoverL ;
    double Leff;    /* effective channel length im micron */
    double Weff;    /* effective channel width in micron */

    NG_IGNORE(ckt);

    /*  loop through all the B1 device models */
    for( ; model != NULL; model = B1nextModel(model)) {
    
/* Default value Processing for B1 MOSFET Models */
        /* Some Limiting for Model Parameters */
        if( model->B1bulkJctPotential < 0.1)  {
            model->B1bulkJctPotential = 0.1;
        }
        if( model->B1sidewallJctPotential < 0.1)  {
            model->B1sidewallJctPotential = 0.1;
        }

        Cox = 3.453e-13/(model->B1oxideThickness * 1.0e-4);/*in F/cm**2 */
        model->B1Cox = Cox;     /* unit:  F/cm**2  */

        /* loop through all the instances of the model */
        for (here = B1instances(model); here != NULL ;
                here=B1nextInstance(here)) {

            if( (EffChanLength = here->B1l - model->B1deltaL *1e-6 )<=0) { 
                SPfrontEnd->IFerrorf (ERR_FATAL,
                    "B1: mosfet %s, model %s: Effective channel length <=0",
                    model->B1modName, here->B1name);
                return(E_BADPARM);
            }
            if( (EffChanWidth = here->B1w - model->B1deltaW *1e-6 ) <= 0 ) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                    "B1: mosfet %s, model %s: Effective channel width <=0",
                    model->B1modName, here->B1name);
                return(E_BADPARM);
            }
            here->B1GDoverlapCap=EffChanWidth *model->B1gateDrainOverlapCap;
            here->B1GSoverlapCap=EffChanWidth*model->B1gateSourceOverlapCap;
            here->B1GBoverlapCap=here->B1l * model->B1gateBulkOverlapCap;

            /* process drain series resistance */
            if( (here->B1drainConductance=model->B1sheetResistance *
                    here->B1drainSquares) != 0.0 ) {   
                here->B1drainConductance = 1. / here->B1drainConductance ;
            } 
                   
            /* process source series resistance */
            if( (here->B1sourceConductance=model->B1sheetResistance *
                    here->B1sourceSquares) != 0.0 ) { 
                here->B1sourceConductance = 1. / here->B1sourceConductance ;
            }
                   
            Leff = EffChanLength * 1.e6; /* convert into micron */
            Weff = EffChanWidth * 1.e6; /* convert into micron */
            CoxWoverL = Cox * Weff / Leff ; /* F/cm**2 */

            here->B1vfb = model->B1vfb0 + 
                model->B1vfbL / Leff + model->B1vfbW / Weff;
            here->B1phi = model->B1phi0 +
                model->B1phiL / Leff + model->B1phiW / Weff;
            here->B1K1 = model->B1K10 +
                model->B1K1L / Leff + model->B1K1W / Weff;
            here->B1K2 = model->B1K20 +
                model->B1K2L / Leff + model->B1K2W / Weff;
            here->B1eta = model->B1eta0 +
                model->B1etaL / Leff + model->B1etaW / Weff;
            here->B1etaB = model->B1etaB0 +
                model->B1etaBl / Leff + model->B1etaBw / Weff;
            here->B1etaD = model->B1etaD0 +
                model->B1etaDl / Leff + model->B1etaDw / Weff;
            here->B1betaZero = model->B1mobZero;
            here->B1betaZeroB = model->B1mobZeroB0 + 
                model->B1mobZeroBl / Leff + model->B1mobZeroBw / Weff;
            here->B1ugs = model->B1ugs0 +
                model->B1ugsL / Leff + model->B1ugsW / Weff;
            here->B1ugsB = model->B1ugsB0 +
                model->B1ugsBL / Leff + model->B1ugsBW / Weff;
            here->B1uds = model->B1uds0 +
                model->B1udsL / Leff + model->B1udsW / Weff;
            here->B1udsB = model->B1udsB0 +
                model->B1udsBL / Leff + model->B1udsBW / Weff;
            here->B1udsD = model->B1udsD0 +
                model->B1udsDL / Leff + model->B1udsDW / Weff;
            here->B1betaVdd = model->B1mobVdd0 +
                model->B1mobVddl / Leff + model->B1mobVddw / Weff;
            here->B1betaVddB = model->B1mobVddB0 + 
                model->B1mobVddBl / Leff + model->B1mobVddBw / Weff;
            here->B1betaVddD = model->B1mobVddD0 +
                model->B1mobVddDl / Leff + model->B1mobVddDw / Weff;
            here->B1subthSlope = model->B1subthSlope0 + 
                model->B1subthSlopeL / Leff + model->B1subthSlopeW / Weff;
            here->B1subthSlopeB = model->B1subthSlopeB0 +
                model->B1subthSlopeBL / Leff + model->B1subthSlopeBW / Weff;
            here->B1subthSlopeD = model->B1subthSlopeD0 + 
                model->B1subthSlopeDL / Leff + model->B1subthSlopeDW / Weff;

            if(here->B1phi < 0.1 ) here->B1phi = 0.1;
            if(here->B1K1 < 0.0) here->B1K1 = 0.0;
            if(here->B1K2 < 0.0) here->B1K2 = 0.0;

            here->B1vt0 = here->B1vfb + here->B1phi + here->B1K1 * 
                sqrt(here->B1phi) - here->B1K2 * here->B1phi;

            here->B1von = here->B1vt0;  /* added for initialization*/

                /* process Beta Parameters (unit: A/V**2) */

            here->B1betaZero = here->B1betaZero * CoxWoverL;
            here->B1betaZeroB = here->B1betaZeroB * CoxWoverL;
            here->B1betaVdd = here->B1betaVdd * CoxWoverL;
            here->B1betaVddB = here->B1betaVddB * CoxWoverL;
            here->B1betaVddD = MAX(here->B1betaVddD * CoxWoverL,0.0);

        }
    }
    return(OK);
}  


