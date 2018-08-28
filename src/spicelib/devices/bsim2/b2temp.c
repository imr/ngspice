/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
B2temp(GENmodel *inModel, CKTcircuit *ckt)
        /* load the B2 device structure with those pointers needed later
         * for fast matrix loading
         */

{
    B2model *model = (B2model*) inModel;
    B2instance *here;
    struct bsim2SizeDependParam *pSizeDependParamKnot, *pLastKnot;
    double  EffectiveLength;
    double EffectiveWidth;
    double CoxWoverL, Inv_L, Inv_W, tmp;
    int Size_Not_Found;

    NG_IGNORE(ckt);

    /*  loop through all the B2 device models */
    for( ; model != NULL; model = B2nextModel(model)) {

/* Default value Processing for B2 MOSFET Models */
        /* Some Limiting for Model Parameters */
        if( model->B2bulkJctPotential < 0.1)  {
            model->B2bulkJctPotential = 0.1;
        }
        if( model->B2sidewallJctPotential < 0.1)  {
            model->B2sidewallJctPotential = 0.1;
        }

        model->B2Cox = 3.453e-13/(model->B2tox * 1.0e-4);/*in F/cm**2 */
        model->B2vdd2 = 2.0 * model->B2vdd;
        model->B2vgg2 = 2.0 * model->B2vgg;
        model->B2vbb2 = 2.0 * model->B2vbb;
        model->B2Vtm = 8.625e-5 * (model->B2temp + 273.0);

        struct bsim2SizeDependParam *p = model->pSizeDependParamKnot;
        while (p) {
            struct bsim2SizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
        model->pSizeDependParamKnot = NULL;
        pLastKnot = NULL;

        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ;
                here=B2nextInstance(here)) {

            pSizeDependParamKnot = model->pSizeDependParamKnot;
            Size_Not_Found = 1;

            while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
            {   if ((here->B2l == pSizeDependParamKnot->Length)
                    && (here->B2w == pSizeDependParamKnot->Width))
                {   Size_Not_Found = 0;
                    here->pParam = pSizeDependParamKnot;
                }
                else
                {   pLastKnot = pSizeDependParamKnot;
                    pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                }
            }

            if (Size_Not_Found)
            {   here->pParam = TMALLOC(struct bsim2SizeDependParam, 1);
                if (pLastKnot == NULL)
                    model->pSizeDependParamKnot = here->pParam;
                else
                    pLastKnot->pNext = here->pParam;
                here->pParam->pNext = NULL;

                EffectiveLength = here->B2l - model->B2deltaL * 1.0e-6;
                EffectiveWidth = here->B2w - model->B2deltaW * 1.0e-6;

                if(EffectiveLength<=0)
                {
                   SPfrontEnd->IFerrorf (ERR_FATAL,
                    "B2: mosfet %s, model %s: Effective channel length <=0",
                    model->B2modName, here->B2name);
                   return(E_BADPARM);
                }

                if(EffectiveWidth <= 0)
                {
                   SPfrontEnd->IFerrorf (ERR_FATAL,
                    "B2: mosfet %s, model %s: Effective channel width <=0",
                    model->B2modName, here->B2name);
                   return(E_BADPARM);
                }

                Inv_L = 1.0e-6 / EffectiveLength;
                Inv_W = 1.0e-6 / EffectiveWidth;
                here->pParam->Width = here->B2w;
                here->pParam->Length = here->B2l;
                here->pParam->B2vfb = model->B2vfb0 + model->B2vfbW * Inv_W
                               + model->B2vfbL * Inv_L;
                here->pParam->B2phi = model->B2phi0 + model->B2phiW * Inv_W
                               + model->B2phiL * Inv_L;
                here->pParam->B2k1 = model->B2k10 + model->B2k1W * Inv_W
                               + model->B2k1L * Inv_L;
                here->pParam->B2k2 = model->B2k20 + model->B2k2W * Inv_W
                               + model->B2k2L * Inv_L;
                here->pParam->B2eta0 = model->B2eta00
                               + model->B2eta0W * Inv_W
                               + model->B2eta0L * Inv_L;
                here->pParam->B2etaB = model->B2etaB0 + model->B2etaBW
                               * Inv_W + model->B2etaBL * Inv_L;
                here->pParam->B2beta0 = model->B2mob00;
                here->pParam->B2beta0B = model->B2mob0B0
                               + model->B2mob0BW * Inv_W
                               + model->B2mob0BL * Inv_L;
                here->pParam->B2betas0 = model->B2mobs00
                               + model->B2mobs0W * Inv_W
                               + model->B2mobs0L * Inv_L;
                if (here->pParam->B2betas0 < 1.01 * here->pParam->B2beta0)
                    here->pParam->B2betas0 = 1.01 * here->pParam->B2beta0;
                here->pParam->B2betasB = model->B2mobsB0
                               + model->B2mobsBW * Inv_W
                               + model->B2mobsBL * Inv_L;
                tmp = (here->pParam->B2betas0 - here->pParam->B2beta0
                               - here->pParam->B2beta0B * model->B2vbb);
                if ((-here->pParam->B2betasB * model->B2vbb) > tmp)
                    here->pParam->B2betasB = -tmp / model->B2vbb;
                here->pParam->B2beta20 = model->B2mob200
                              + model->B2mob20W * Inv_W
                              + model->B2mob20L * Inv_L;
                here->pParam->B2beta2B = model->B2mob2B0
                              + model->B2mob2BW * Inv_W
                              + model->B2mob2BL * Inv_L;
                here->pParam->B2beta2G = model->B2mob2G0
                              + model->B2mob2GW * Inv_W
                              + model->B2mob2GL * Inv_L;
                here->pParam->B2beta30 = model->B2mob300
                              + model->B2mob30W * Inv_W
                              + model->B2mob30L * Inv_L;
                here->pParam->B2beta3B = model->B2mob3B0
                              + model->B2mob3BW * Inv_W
                              + model->B2mob3BL * Inv_L;
                here->pParam->B2beta3G = model->B2mob3G0
                              + model->B2mob3GW * Inv_W
                              + model->B2mob3GL * Inv_L;
                here->pParam->B2beta40 = model->B2mob400
                              + model->B2mob40W * Inv_W
                              + model->B2mob40L * Inv_L;
                here->pParam->B2beta4B = model->B2mob4B0
                              + model->B2mob4BW * Inv_W
                              + model->B2mob4BL * Inv_L;
                here->pParam->B2beta4G = model->B2mob4G0
                              + model->B2mob4GW * Inv_W
                              + model->B2mob4GL * Inv_L;

                CoxWoverL = model->B2Cox * EffectiveWidth / EffectiveLength;

                here->pParam->B2beta0 *= CoxWoverL;
                here->pParam->B2beta0B *= CoxWoverL;
                here->pParam->B2betas0 *= CoxWoverL;
                here->pParam->B2betasB *= CoxWoverL;
                here->pParam->B2beta30 *= CoxWoverL;
                here->pParam->B2beta3B *= CoxWoverL;
                here->pParam->B2beta3G *= CoxWoverL;
                here->pParam->B2beta40 *= CoxWoverL;
                here->pParam->B2beta4B *= CoxWoverL;
                here->pParam->B2beta4G *= CoxWoverL;

                here->pParam->B2ua0 = model->B2ua00 + model->B2ua0W * Inv_W
                               + model->B2ua0L * Inv_L;
                here->pParam->B2uaB = model->B2uaB0 + model->B2uaBW * Inv_W
                               + model->B2uaBL * Inv_L;
                here->pParam->B2ub0 = model->B2ub00 + model->B2ub0W * Inv_W
                               + model->B2ub0L * Inv_L;
                here->pParam->B2ubB = model->B2ubB0 + model->B2ubBW * Inv_W
                               + model->B2ubBL * Inv_L;
                here->pParam->B2u10 = model->B2u100 + model->B2u10W * Inv_W
                               + model->B2u10L * Inv_L;
                here->pParam->B2u1B = model->B2u1B0 + model->B2u1BW * Inv_W
                               + model->B2u1BL * Inv_L;
                here->pParam->B2u1D = model->B2u1D0 + model->B2u1DW * Inv_W
                               + model->B2u1DL * Inv_L;
                here->pParam->B2n0 = model->B2n00 + model->B2n0W * Inv_W
                               + model->B2n0L * Inv_L;
                here->pParam->B2nB = model->B2nB0 + model->B2nBW * Inv_W
                               + model->B2nBL * Inv_L;
                here->pParam->B2nD = model->B2nD0 + model->B2nDW * Inv_W
                               + model->B2nDL * Inv_L;
                if (here->pParam->B2n0 < 0.0)
                    here->pParam->B2n0 = 0.0;

                here->pParam->B2vof0 = model->B2vof00
                               + model->B2vof0W * Inv_W
                               + model->B2vof0L * Inv_L;
                here->pParam->B2vofB = model->B2vofB0
                               + model->B2vofBW * Inv_W
                               + model->B2vofBL * Inv_L;
                here->pParam->B2vofD = model->B2vofD0
                               + model->B2vofDW * Inv_W
                               + model->B2vofDL * Inv_L;
                here->pParam->B2ai0 = model->B2ai00 + model->B2ai0W * Inv_W
                               + model->B2ai0L * Inv_L;
                here->pParam->B2aiB = model->B2aiB0 + model->B2aiBW * Inv_W
                               + model->B2aiBL * Inv_L;
                here->pParam->B2bi0 = model->B2bi00 + model->B2bi0W * Inv_W
                               + model->B2bi0L * Inv_L;
                here->pParam->B2biB = model->B2biB0 + model->B2biBW * Inv_W
                               + model->B2biBL * Inv_L;
                here->pParam->B2vghigh = model->B2vghigh0
                               + model->B2vghighW * Inv_W
                               + model->B2vghighL * Inv_L;
                here->pParam->B2vglow = model->B2vglow0
                               + model->B2vglowW * Inv_W
                               + model->B2vglowL * Inv_L;

                here->pParam->CoxWL = model->B2Cox * EffectiveLength
                               * EffectiveWidth * 1.0e4;
                here->pParam->One_Third_CoxWL = here->pParam->CoxWL / 3.0;
                here->pParam->Two_Third_CoxWL = 2.0
                               * here->pParam->One_Third_CoxWL;
                here->pParam->B2GSoverlapCap = model->B2gateSourceOverlapCap
                               * EffectiveWidth;
                here->pParam->B2GDoverlapCap = model->B2gateDrainOverlapCap
                               * EffectiveWidth;
                here->pParam->B2GBoverlapCap = model->B2gateBulkOverlapCap
                               * EffectiveLength;
                here->pParam->SqrtPhi = sqrt(here->pParam->B2phi);
                here->pParam->Phis3 = here->pParam->SqrtPhi
                               * here->pParam->B2phi;
                here->pParam->Arg = here->pParam->B2betasB
                               - here->pParam->B2beta0B - model->B2vdd
                               * (here->pParam->B2beta3B - model->B2vdd
                               * here->pParam->B2beta4B);


             }


            /* process drain series resistance */
            if( (here->B2drainConductance=model->B2sheetResistance *
                    here->B2drainSquares) != 0.0 ) {
                here->B2drainConductance = 1. / here->B2drainConductance ;
            }

            /* process source series resistance */
            if( (here->B2sourceConductance=model->B2sheetResistance *
                    here->B2sourceSquares) != 0.0 ) {
                here->B2sourceConductance = 1. / here->B2sourceConductance ;
            }


            here->pParam->B2vt0 = here->pParam->B2vfb
                          + here->pParam->B2phi
                          + here->pParam->B2k1 * here->pParam->SqrtPhi
                          - here->pParam->B2k2 * here->pParam->B2phi;
            here->B2von = here->pParam->B2vt0; /* added for initialization*/
        }
    }
    return(OK);
}



