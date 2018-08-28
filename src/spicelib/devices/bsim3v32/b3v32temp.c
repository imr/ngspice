/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3temp.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5  /* Kb / q  where q = 1.60219e-19 */
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define Charge_q 1.60219e-19

/* ARGSUSED */
int
BSIM3v32temp (GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v32model *model = (BSIM3v32model*) inModel;
BSIM3v32instance *here;
struct bsim3v32SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double Nvtm, SourceSatCurrent, DrainSatCurrent;
int Size_Not_Found, error;

    /*  loop through all the BSIM3v32 device models */
    for (; model != NULL; model = BSIM3v32nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3v32bulkJctPotential < 0.1)
         {   model->BSIM3v32bulkJctPotential = 0.1;
             fprintf(stderr, "Given pb is less than 0.1. Pb is set to 0.1.\n");
         }
         if (model->BSIM3v32sidewallJctPotential < 0.1)
         {   model->BSIM3v32sidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbsw is less than 0.1. Pbsw is set to 0.1.\n");
         }
         if (model->BSIM3v32GatesidewallJctPotential < 0.1)
         {   model->BSIM3v32GatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswg is less than 0.1. Pbswg is set to 0.1.\n");
         }

         struct bsim3v32SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim3v32SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM3v32tnom;
         TRatio = Temp / Tnom;

         model->BSIM3v32vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3v32factor1 = sqrt(EPSSI / EPSOX * model->BSIM3v32tox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3v32vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3v32vtm + model->BSIM3v32jctTempExponent
                * log(Temp / Tnom);
             T1 = exp(T0 / model->BSIM3v32jctEmissionCoeff);
             model->BSIM3v32jctTempSatCurDensity = model->BSIM3v32jctSatCurDensity
                                              * T1;
             model->BSIM3v32jctSidewallTempSatCurDensity
                         = model->BSIM3v32jctSidewallSatCurDensity * T1;
         }
         else
         {   model->BSIM3v32jctTempSatCurDensity = model->BSIM3v32jctSatCurDensity;
             model->BSIM3v32jctSidewallTempSatCurDensity
                        = model->BSIM3v32jctSidewallSatCurDensity;
         }

         if (model->BSIM3v32jctTempSatCurDensity < 0.0)
             model->BSIM3v32jctTempSatCurDensity = 0.0;
         if (model->BSIM3v32jctSidewallTempSatCurDensity < 0.0)
             model->BSIM3v32jctSidewallTempSatCurDensity = 0.0;

         /* Temperature dependence of D/B and S/B diode capacitance begins */
         delTemp = ckt->CKTtemp - model->BSIM3v32tnom;
         T0 = model->BSIM3v32tcj * delTemp;
         if (T0 >= -1.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitAreaTempJctCap =
                                model->BSIM3v32unitAreaJctCap * (1.0 + T0);
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitAreaJctCap *= 1.0 + T0;
                }
         }
         else if (model->BSIM3v32unitAreaJctCap > 0.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitAreaTempJctCap = 0.0;
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitAreaJctCap = 0.0;
                }
             fprintf(stderr, "Temperature effect has caused cj to be negative. Cj is clamped to zero.\n");
         }
         T0 = model->BSIM3v32tcjsw * delTemp;
         if (T0 >= -1.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitLengthSidewallTempJctCap =
                                model->BSIM3v32unitLengthSidewallJctCap * (1.0 + T0);
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitLengthSidewallJctCap *= 1.0 + T0;
                }
         }
         else if (model->BSIM3v32unitLengthSidewallJctCap > 0.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitLengthSidewallTempJctCap = 0.0;
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitLengthSidewallJctCap = 0.0;
                }
             fprintf(stderr, "Temperature effect has caused cjsw to be negative. Cjsw is clamped to zero.\n");
         }
         T0 = model->BSIM3v32tcjswg * delTemp;
         if (T0 >= -1.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitLengthGateSidewallTempJctCap =
                                model->BSIM3v32unitLengthGateSidewallJctCap * (1.0 + T0);
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitLengthGateSidewallJctCap *= 1.0 + T0;
                }
         }
         else if (model->BSIM3v32unitLengthGateSidewallJctCap > 0.0)
         {
                /* Added revision dependent code */
                switch (model->BSIM3v32intVersion) {
                case BSIM3v32V324:
                case BSIM3v32V323:
                        model->BSIM3v32unitLengthGateSidewallTempJctCap = 0.0;
                        break;
                case BSIM3v32V322:
                case BSIM3v32V32:
                default:
                        model->BSIM3v32unitLengthGateSidewallJctCap = 0.0;
                }
             fprintf(stderr, "Temperature effect has caused cjswg to be negative. Cjswg is clamped to zero.\n");
         }

         model->BSIM3v32PhiB = model->BSIM3v32bulkJctPotential
                          - model->BSIM3v32tpb * delTemp;
         if (model->BSIM3v32PhiB < 0.01)
         {   model->BSIM3v32PhiB = 0.01;
             fprintf(stderr, "Temperature effect has caused pb to be less than 0.01. Pb is clamped to 0.01.\n");
         }
         model->BSIM3v32PhiBSW = model->BSIM3v32sidewallJctPotential
                            - model->BSIM3v32tpbsw * delTemp;
         if (model->BSIM3v32PhiBSW <= 0.01)
         {   model->BSIM3v32PhiBSW = 0.01;
             fprintf(stderr, "Temperature effect has caused pbsw to be less than 0.01. Pbsw is clamped to 0.01.\n");
         }
         model->BSIM3v32PhiBSWG = model->BSIM3v32GatesidewallJctPotential
                             - model->BSIM3v32tpbswg * delTemp;
         if (model->BSIM3v32PhiBSWG <= 0.01)
         {   model->BSIM3v32PhiBSWG = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswg to be less than 0.01. Pbswg is clamped to 0.01.\n");
         }
         /* End of junction capacitance */

         /* loop through all the instances of the model */
               /* MCJ: Length and Width not initialized */
         for (here = BSIM3v32instances(model); here != NULL;
              here = BSIM3v32nextInstance(here))
         {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM3v32l == pSizeDependParamKnot->Length)
                      && (here->BSIM3v32w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                      if (model->BSIM3v32intVersion > BSIM3v32V322)
                      {
                        pParam = here->pParam; /*bug-fix  */
                      }
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim3v32SizeDependParam, 1);
                  if (pLastKnot == NULL)
                    model->pSizeDependParamKnot = pParam;
                  else
                    pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->BSIM3v32l;
                  Wdrn = here->BSIM3v32w;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;

                  T0 = pow(Ldrn, model->BSIM3v32Lln);
                  T1 = pow(Wdrn, model->BSIM3v32Lwn);
                  tmp1 = model->BSIM3v32Ll / T0 + model->BSIM3v32Lw / T1
                       + model->BSIM3v32Lwl / (T0 * T1);
                  pParam->BSIM3v32dl = model->BSIM3v32Lint + tmp1;
                  tmp2 = model->BSIM3v32Llc / T0 + model->BSIM3v32Lwc / T1
                       + model->BSIM3v32Lwlc / (T0 * T1);
                  pParam->BSIM3v32dlc = model->BSIM3v32dlc + tmp2;

                  T2 = pow(Ldrn, model->BSIM3v32Wln);
                  T3 = pow(Wdrn, model->BSIM3v32Wwn);
                  tmp1 = model->BSIM3v32Wl / T2 + model->BSIM3v32Ww / T3
                       + model->BSIM3v32Wwl / (T2 * T3);
                  pParam->BSIM3v32dw = model->BSIM3v32Wint + tmp1;
                  tmp2 = model->BSIM3v32Wlc / T2 + model->BSIM3v32Wwc / T3
                       + model->BSIM3v32Wwlc / (T2 * T3);
                  pParam->BSIM3v32dwc = model->BSIM3v32dwc + tmp2;

                  pParam->BSIM3v32leff = here->BSIM3v32l + model->BSIM3v32xl - 2.0 * pParam->BSIM3v32dl;
                  if (pParam->BSIM3v32leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v32: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM3v32modName, here->BSIM3v32name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v32weff = here->BSIM3v32w + model->BSIM3v32xw - 2.0 * pParam->BSIM3v32dw;
                  if (pParam->BSIM3v32weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v32: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM3v32modName, here->BSIM3v32name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v32leffCV = here->BSIM3v32l + model->BSIM3v32xl - 2.0 * pParam->BSIM3v32dlc;
                  if (pParam->BSIM3v32leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v32: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM3v32modName, here->BSIM3v32name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v32weffCV = here->BSIM3v32w + model->BSIM3v32xw - 2.0 * pParam->BSIM3v32dwc;
                  if (pParam->BSIM3v32weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v32: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM3v32modName, here->BSIM3v32name);
                      return(E_BADPARM);
                  }


                  if (model->BSIM3v32binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM3v32leff;
                      Inv_W = 1.0e-6 / pParam->BSIM3v32weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM3v32leff
                             * pParam->BSIM3v32weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM3v32leff;
                      Inv_W = 1.0 / pParam->BSIM3v32weff;
                      Inv_LW = 1.0 / (pParam->BSIM3v32leff
                             * pParam->BSIM3v32weff);
                  }
                  pParam->BSIM3v32cdsc = model->BSIM3v32cdsc
                                    + model->BSIM3v32lcdsc * Inv_L
                                    + model->BSIM3v32wcdsc * Inv_W
                                    + model->BSIM3v32pcdsc * Inv_LW;
                  pParam->BSIM3v32cdscb = model->BSIM3v32cdscb
                                     + model->BSIM3v32lcdscb * Inv_L
                                     + model->BSIM3v32wcdscb * Inv_W
                                     + model->BSIM3v32pcdscb * Inv_LW;

                  pParam->BSIM3v32cdscd = model->BSIM3v32cdscd
                                     + model->BSIM3v32lcdscd * Inv_L
                                     + model->BSIM3v32wcdscd * Inv_W
                                     + model->BSIM3v32pcdscd * Inv_LW;

                  pParam->BSIM3v32cit = model->BSIM3v32cit
                                   + model->BSIM3v32lcit * Inv_L
                                   + model->BSIM3v32wcit * Inv_W
                                   + model->BSIM3v32pcit * Inv_LW;
                  pParam->BSIM3v32nfactor = model->BSIM3v32nfactor
                                       + model->BSIM3v32lnfactor * Inv_L
                                       + model->BSIM3v32wnfactor * Inv_W
                                       + model->BSIM3v32pnfactor * Inv_LW;
                  pParam->BSIM3v32xj = model->BSIM3v32xj
                                  + model->BSIM3v32lxj * Inv_L
                                  + model->BSIM3v32wxj * Inv_W
                                  + model->BSIM3v32pxj * Inv_LW;
                  pParam->BSIM3v32vsat = model->BSIM3v32vsat
                                    + model->BSIM3v32lvsat * Inv_L
                                    + model->BSIM3v32wvsat * Inv_W
                                    + model->BSIM3v32pvsat * Inv_LW;
                  pParam->BSIM3v32at = model->BSIM3v32at
                                  + model->BSIM3v32lat * Inv_L
                                  + model->BSIM3v32wat * Inv_W
                                  + model->BSIM3v32pat * Inv_LW;
                  pParam->BSIM3v32a0 = model->BSIM3v32a0
                                  + model->BSIM3v32la0 * Inv_L
                                  + model->BSIM3v32wa0 * Inv_W
                                  + model->BSIM3v32pa0 * Inv_LW;

                  pParam->BSIM3v32ags = model->BSIM3v32ags
                                  + model->BSIM3v32lags * Inv_L
                                  + model->BSIM3v32wags * Inv_W
                                  + model->BSIM3v32pags * Inv_LW;

                  pParam->BSIM3v32a1 = model->BSIM3v32a1
                                  + model->BSIM3v32la1 * Inv_L
                                  + model->BSIM3v32wa1 * Inv_W
                                  + model->BSIM3v32pa1 * Inv_LW;
                  pParam->BSIM3v32a2 = model->BSIM3v32a2
                                  + model->BSIM3v32la2 * Inv_L
                                  + model->BSIM3v32wa2 * Inv_W
                                  + model->BSIM3v32pa2 * Inv_LW;
                  pParam->BSIM3v32keta = model->BSIM3v32keta
                                    + model->BSIM3v32lketa * Inv_L
                                    + model->BSIM3v32wketa * Inv_W
                                    + model->BSIM3v32pketa * Inv_LW;
                  pParam->BSIM3v32nsub = model->BSIM3v32nsub
                                    + model->BSIM3v32lnsub * Inv_L
                                    + model->BSIM3v32wnsub * Inv_W
                                    + model->BSIM3v32pnsub * Inv_LW;
                  pParam->BSIM3v32npeak = model->BSIM3v32npeak
                                     + model->BSIM3v32lnpeak * Inv_L
                                     + model->BSIM3v32wnpeak * Inv_W
                                     + model->BSIM3v32pnpeak * Inv_LW;
                  pParam->BSIM3v32ngate = model->BSIM3v32ngate
                                     + model->BSIM3v32lngate * Inv_L
                                     + model->BSIM3v32wngate * Inv_W
                                     + model->BSIM3v32pngate * Inv_LW;
                  pParam->BSIM3v32gamma1 = model->BSIM3v32gamma1
                                      + model->BSIM3v32lgamma1 * Inv_L
                                      + model->BSIM3v32wgamma1 * Inv_W
                                      + model->BSIM3v32pgamma1 * Inv_LW;
                  pParam->BSIM3v32gamma2 = model->BSIM3v32gamma2
                                      + model->BSIM3v32lgamma2 * Inv_L
                                      + model->BSIM3v32wgamma2 * Inv_W
                                      + model->BSIM3v32pgamma2 * Inv_LW;
                  pParam->BSIM3v32vbx = model->BSIM3v32vbx
                                   + model->BSIM3v32lvbx * Inv_L
                                   + model->BSIM3v32wvbx * Inv_W
                                   + model->BSIM3v32pvbx * Inv_LW;
                  pParam->BSIM3v32vbm = model->BSIM3v32vbm
                                   + model->BSIM3v32lvbm * Inv_L
                                   + model->BSIM3v32wvbm * Inv_W
                                   + model->BSIM3v32pvbm * Inv_LW;
                  pParam->BSIM3v32xt = model->BSIM3v32xt
                                   + model->BSIM3v32lxt * Inv_L
                                   + model->BSIM3v32wxt * Inv_W
                                   + model->BSIM3v32pxt * Inv_LW;
                  pParam->BSIM3v32vfb = model->BSIM3v32vfb
                             + model->BSIM3v32lvfb * Inv_L
                             + model->BSIM3v32wvfb * Inv_W
                             + model->BSIM3v32pvfb * Inv_LW;
                  pParam->BSIM3v32k1 = model->BSIM3v32k1
                                  + model->BSIM3v32lk1 * Inv_L
                                  + model->BSIM3v32wk1 * Inv_W
                                  + model->BSIM3v32pk1 * Inv_LW;
                  pParam->BSIM3v32kt1 = model->BSIM3v32kt1
                                   + model->BSIM3v32lkt1 * Inv_L
                                   + model->BSIM3v32wkt1 * Inv_W
                                   + model->BSIM3v32pkt1 * Inv_LW;
                  pParam->BSIM3v32kt1l = model->BSIM3v32kt1l
                                    + model->BSIM3v32lkt1l * Inv_L
                                    + model->BSIM3v32wkt1l * Inv_W
                                    + model->BSIM3v32pkt1l * Inv_LW;
                  pParam->BSIM3v32k2 = model->BSIM3v32k2
                                  + model->BSIM3v32lk2 * Inv_L
                                  + model->BSIM3v32wk2 * Inv_W
                                  + model->BSIM3v32pk2 * Inv_LW;
                  pParam->BSIM3v32kt2 = model->BSIM3v32kt2
                                   + model->BSIM3v32lkt2 * Inv_L
                                   + model->BSIM3v32wkt2 * Inv_W
                                   + model->BSIM3v32pkt2 * Inv_LW;
                  pParam->BSIM3v32k3 = model->BSIM3v32k3
                                  + model->BSIM3v32lk3 * Inv_L
                                  + model->BSIM3v32wk3 * Inv_W
                                  + model->BSIM3v32pk3 * Inv_LW;
                  pParam->BSIM3v32k3b = model->BSIM3v32k3b
                                   + model->BSIM3v32lk3b * Inv_L
                                   + model->BSIM3v32wk3b * Inv_W
                                   + model->BSIM3v32pk3b * Inv_LW;
                  pParam->BSIM3v32w0 = model->BSIM3v32w0
                                  + model->BSIM3v32lw0 * Inv_L
                                  + model->BSIM3v32ww0 * Inv_W
                                  + model->BSIM3v32pw0 * Inv_LW;
                  pParam->BSIM3v32nlx = model->BSIM3v32nlx
                                   + model->BSIM3v32lnlx * Inv_L
                                   + model->BSIM3v32wnlx * Inv_W
                                   + model->BSIM3v32pnlx * Inv_LW;
                  pParam->BSIM3v32dvt0 = model->BSIM3v32dvt0
                                    + model->BSIM3v32ldvt0 * Inv_L
                                    + model->BSIM3v32wdvt0 * Inv_W
                                    + model->BSIM3v32pdvt0 * Inv_LW;
                  pParam->BSIM3v32dvt1 = model->BSIM3v32dvt1
                                    + model->BSIM3v32ldvt1 * Inv_L
                                    + model->BSIM3v32wdvt1 * Inv_W
                                    + model->BSIM3v32pdvt1 * Inv_LW;
                  pParam->BSIM3v32dvt2 = model->BSIM3v32dvt2
                                    + model->BSIM3v32ldvt2 * Inv_L
                                    + model->BSIM3v32wdvt2 * Inv_W
                                    + model->BSIM3v32pdvt2 * Inv_LW;
                  pParam->BSIM3v32dvt0w = model->BSIM3v32dvt0w
                                    + model->BSIM3v32ldvt0w * Inv_L
                                    + model->BSIM3v32wdvt0w * Inv_W
                                    + model->BSIM3v32pdvt0w * Inv_LW;
                  pParam->BSIM3v32dvt1w = model->BSIM3v32dvt1w
                                    + model->BSIM3v32ldvt1w * Inv_L
                                    + model->BSIM3v32wdvt1w * Inv_W
                                    + model->BSIM3v32pdvt1w * Inv_LW;
                  pParam->BSIM3v32dvt2w = model->BSIM3v32dvt2w
                                    + model->BSIM3v32ldvt2w * Inv_L
                                    + model->BSIM3v32wdvt2w * Inv_W
                                    + model->BSIM3v32pdvt2w * Inv_LW;
                  pParam->BSIM3v32drout = model->BSIM3v32drout
                                     + model->BSIM3v32ldrout * Inv_L
                                     + model->BSIM3v32wdrout * Inv_W
                                     + model->BSIM3v32pdrout * Inv_LW;
                  pParam->BSIM3v32dsub = model->BSIM3v32dsub
                                    + model->BSIM3v32ldsub * Inv_L
                                    + model->BSIM3v32wdsub * Inv_W
                                    + model->BSIM3v32pdsub * Inv_LW;
                  pParam->BSIM3v32vth0 = model->BSIM3v32vth0
                                    + model->BSIM3v32lvth0 * Inv_L
                                    + model->BSIM3v32wvth0 * Inv_W
                                    + model->BSIM3v32pvth0 * Inv_LW;
                  pParam->BSIM3v32ua = model->BSIM3v32ua
                                  + model->BSIM3v32lua * Inv_L
                                  + model->BSIM3v32wua * Inv_W
                                  + model->BSIM3v32pua * Inv_LW;
                  pParam->BSIM3v32ua1 = model->BSIM3v32ua1
                                   + model->BSIM3v32lua1 * Inv_L
                                   + model->BSIM3v32wua1 * Inv_W
                                   + model->BSIM3v32pua1 * Inv_LW;
                  pParam->BSIM3v32ub = model->BSIM3v32ub
                                  + model->BSIM3v32lub * Inv_L
                                  + model->BSIM3v32wub * Inv_W
                                  + model->BSIM3v32pub * Inv_LW;
                  pParam->BSIM3v32ub1 = model->BSIM3v32ub1
                                   + model->BSIM3v32lub1 * Inv_L
                                   + model->BSIM3v32wub1 * Inv_W
                                   + model->BSIM3v32pub1 * Inv_LW;
                  pParam->BSIM3v32uc = model->BSIM3v32uc
                                  + model->BSIM3v32luc * Inv_L
                                  + model->BSIM3v32wuc * Inv_W
                                  + model->BSIM3v32puc * Inv_LW;
                  pParam->BSIM3v32uc1 = model->BSIM3v32uc1
                                   + model->BSIM3v32luc1 * Inv_L
                                   + model->BSIM3v32wuc1 * Inv_W
                                   + model->BSIM3v32puc1 * Inv_LW;
                  pParam->BSIM3v32u0 = model->BSIM3v32u0
                                  + model->BSIM3v32lu0 * Inv_L
                                  + model->BSIM3v32wu0 * Inv_W
                                  + model->BSIM3v32pu0 * Inv_LW;
                  pParam->BSIM3v32ute = model->BSIM3v32ute
                                   + model->BSIM3v32lute * Inv_L
                                   + model->BSIM3v32wute * Inv_W
                                   + model->BSIM3v32pute * Inv_LW;
                  pParam->BSIM3v32voff = model->BSIM3v32voff
                                    + model->BSIM3v32lvoff * Inv_L
                                    + model->BSIM3v32wvoff * Inv_W
                                    + model->BSIM3v32pvoff * Inv_LW;
                  pParam->BSIM3v32delta = model->BSIM3v32delta
                                     + model->BSIM3v32ldelta * Inv_L
                                     + model->BSIM3v32wdelta * Inv_W
                                     + model->BSIM3v32pdelta * Inv_LW;
                  pParam->BSIM3v32rdsw = model->BSIM3v32rdsw
                                    + model->BSIM3v32lrdsw * Inv_L
                                    + model->BSIM3v32wrdsw * Inv_W
                                    + model->BSIM3v32prdsw * Inv_LW;
                  pParam->BSIM3v32prwg = model->BSIM3v32prwg
                                    + model->BSIM3v32lprwg * Inv_L
                                    + model->BSIM3v32wprwg * Inv_W
                                    + model->BSIM3v32pprwg * Inv_LW;
                  pParam->BSIM3v32prwb = model->BSIM3v32prwb
                                    + model->BSIM3v32lprwb * Inv_L
                                    + model->BSIM3v32wprwb * Inv_W
                                    + model->BSIM3v32pprwb * Inv_LW;
                  pParam->BSIM3v32prt = model->BSIM3v32prt
                                    + model->BSIM3v32lprt * Inv_L
                                    + model->BSIM3v32wprt * Inv_W
                                    + model->BSIM3v32pprt * Inv_LW;
                  pParam->BSIM3v32eta0 = model->BSIM3v32eta0
                                    + model->BSIM3v32leta0 * Inv_L
                                    + model->BSIM3v32weta0 * Inv_W
                                    + model->BSIM3v32peta0 * Inv_LW;
                  pParam->BSIM3v32etab = model->BSIM3v32etab
                                    + model->BSIM3v32letab * Inv_L
                                    + model->BSIM3v32wetab * Inv_W
                                    + model->BSIM3v32petab * Inv_LW;
                  pParam->BSIM3v32pclm = model->BSIM3v32pclm
                                    + model->BSIM3v32lpclm * Inv_L
                                    + model->BSIM3v32wpclm * Inv_W
                                    + model->BSIM3v32ppclm * Inv_LW;
                  pParam->BSIM3v32pdibl1 = model->BSIM3v32pdibl1
                                      + model->BSIM3v32lpdibl1 * Inv_L
                                      + model->BSIM3v32wpdibl1 * Inv_W
                                      + model->BSIM3v32ppdibl1 * Inv_LW;
                  pParam->BSIM3v32pdibl2 = model->BSIM3v32pdibl2
                                      + model->BSIM3v32lpdibl2 * Inv_L
                                      + model->BSIM3v32wpdibl2 * Inv_W
                                      + model->BSIM3v32ppdibl2 * Inv_LW;
                  pParam->BSIM3v32pdiblb = model->BSIM3v32pdiblb
                                      + model->BSIM3v32lpdiblb * Inv_L
                                      + model->BSIM3v32wpdiblb * Inv_W
                                      + model->BSIM3v32ppdiblb * Inv_LW;
                  pParam->BSIM3v32pscbe1 = model->BSIM3v32pscbe1
                                      + model->BSIM3v32lpscbe1 * Inv_L
                                      + model->BSIM3v32wpscbe1 * Inv_W
                                      + model->BSIM3v32ppscbe1 * Inv_LW;
                  pParam->BSIM3v32pscbe2 = model->BSIM3v32pscbe2
                                      + model->BSIM3v32lpscbe2 * Inv_L
                                      + model->BSIM3v32wpscbe2 * Inv_W
                                      + model->BSIM3v32ppscbe2 * Inv_LW;
                  pParam->BSIM3v32pvag = model->BSIM3v32pvag
                                    + model->BSIM3v32lpvag * Inv_L
                                    + model->BSIM3v32wpvag * Inv_W
                                    + model->BSIM3v32ppvag * Inv_LW;
                  pParam->BSIM3v32wr = model->BSIM3v32wr
                                  + model->BSIM3v32lwr * Inv_L
                                  + model->BSIM3v32wwr * Inv_W
                                  + model->BSIM3v32pwr * Inv_LW;
                  pParam->BSIM3v32dwg = model->BSIM3v32dwg
                                   + model->BSIM3v32ldwg * Inv_L
                                   + model->BSIM3v32wdwg * Inv_W
                                   + model->BSIM3v32pdwg * Inv_LW;
                  pParam->BSIM3v32dwb = model->BSIM3v32dwb
                                   + model->BSIM3v32ldwb * Inv_L
                                   + model->BSIM3v32wdwb * Inv_W
                                   + model->BSIM3v32pdwb * Inv_LW;
                  pParam->BSIM3v32b0 = model->BSIM3v32b0
                                  + model->BSIM3v32lb0 * Inv_L
                                  + model->BSIM3v32wb0 * Inv_W
                                  + model->BSIM3v32pb0 * Inv_LW;
                  pParam->BSIM3v32b1 = model->BSIM3v32b1
                                  + model->BSIM3v32lb1 * Inv_L
                                  + model->BSIM3v32wb1 * Inv_W
                                  + model->BSIM3v32pb1 * Inv_LW;
                  pParam->BSIM3v32alpha0 = model->BSIM3v32alpha0
                                      + model->BSIM3v32lalpha0 * Inv_L
                                      + model->BSIM3v32walpha0 * Inv_W
                                      + model->BSIM3v32palpha0 * Inv_LW;
                  pParam->BSIM3v32alpha1 = model->BSIM3v32alpha1
                                + model->BSIM3v32lalpha1 * Inv_L
                                + model->BSIM3v32walpha1 * Inv_W
                                + model->BSIM3v32palpha1 * Inv_LW;
                  pParam->BSIM3v32beta0 = model->BSIM3v32beta0
                                     + model->BSIM3v32lbeta0 * Inv_L
                                     + model->BSIM3v32wbeta0 * Inv_W
                                     + model->BSIM3v32pbeta0 * Inv_LW;
                  /* CV model */
                  pParam->BSIM3v32elm = model->BSIM3v32elm
                                  + model->BSIM3v32lelm * Inv_L
                                  + model->BSIM3v32welm * Inv_W
                                  + model->BSIM3v32pelm * Inv_LW;
                  pParam->BSIM3v32cgsl = model->BSIM3v32cgsl
                                    + model->BSIM3v32lcgsl * Inv_L
                                    + model->BSIM3v32wcgsl * Inv_W
                                    + model->BSIM3v32pcgsl * Inv_LW;
                  pParam->BSIM3v32cgdl = model->BSIM3v32cgdl
                                    + model->BSIM3v32lcgdl * Inv_L
                                    + model->BSIM3v32wcgdl * Inv_W
                                    + model->BSIM3v32pcgdl * Inv_LW;
                  pParam->BSIM3v32ckappa = model->BSIM3v32ckappa
                                      + model->BSIM3v32lckappa * Inv_L
                                      + model->BSIM3v32wckappa * Inv_W
                                      + model->BSIM3v32pckappa * Inv_LW;
                  pParam->BSIM3v32cf = model->BSIM3v32cf
                                  + model->BSIM3v32lcf * Inv_L
                                  + model->BSIM3v32wcf * Inv_W
                                  + model->BSIM3v32pcf * Inv_LW;
                  pParam->BSIM3v32clc = model->BSIM3v32clc
                                   + model->BSIM3v32lclc * Inv_L
                                   + model->BSIM3v32wclc * Inv_W
                                   + model->BSIM3v32pclc * Inv_LW;
                  pParam->BSIM3v32cle = model->BSIM3v32cle
                                   + model->BSIM3v32lcle * Inv_L
                                   + model->BSIM3v32wcle * Inv_W
                                   + model->BSIM3v32pcle * Inv_LW;
                  pParam->BSIM3v32vfbcv = model->BSIM3v32vfbcv
                                     + model->BSIM3v32lvfbcv * Inv_L
                                     + model->BSIM3v32wvfbcv * Inv_W
                                     + model->BSIM3v32pvfbcv * Inv_LW;
                  pParam->BSIM3v32acde = model->BSIM3v32acde
                         + model->BSIM3v32lacde * Inv_L
                         + model->BSIM3v32wacde * Inv_W
                         + model->BSIM3v32pacde * Inv_LW;
                  pParam->BSIM3v32moin = model->BSIM3v32moin
                         + model->BSIM3v32lmoin * Inv_L
                         + model->BSIM3v32wmoin * Inv_W
                         + model->BSIM3v32pmoin * Inv_LW;
                  pParam->BSIM3v32noff = model->BSIM3v32noff
                         + model->BSIM3v32lnoff * Inv_L
                         + model->BSIM3v32wnoff * Inv_W
                         + model->BSIM3v32pnoff * Inv_LW;
                  pParam->BSIM3v32voffcv = model->BSIM3v32voffcv
                         + model->BSIM3v32lvoffcv * Inv_L
                         + model->BSIM3v32wvoffcv * Inv_W
                         + model->BSIM3v32pvoffcv * Inv_LW;

                  pParam->BSIM3v32abulkCVfactor = 1.0 + pow((pParam->BSIM3v32clc
                               / pParam->BSIM3v32leffCV),
                               pParam->BSIM3v32cle);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM3v32ua = pParam->BSIM3v32ua + pParam->BSIM3v32ua1 * T0;
                  pParam->BSIM3v32ub = pParam->BSIM3v32ub + pParam->BSIM3v32ub1 * T0;
                  pParam->BSIM3v32uc = pParam->BSIM3v32uc + pParam->BSIM3v32uc1 * T0;
                  if (pParam->BSIM3v32u0 > 1.0)
                      pParam->BSIM3v32u0 = pParam->BSIM3v32u0 / 1.0e4;

                  pParam->BSIM3v32u0temp = pParam->BSIM3v32u0
                                      * pow(TRatio, pParam->BSIM3v32ute);
                  pParam->BSIM3v32vsattemp = pParam->BSIM3v32vsat - pParam->BSIM3v32at
                                        * T0;
                  pParam->BSIM3v32rds0 = (pParam->BSIM3v32rdsw + pParam->BSIM3v32prt * T0)
                                    / pow(pParam->BSIM3v32weff * 1E6, pParam->BSIM3v32wr);

                  if (BSIM3v32checkModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during BSIM3v32V3.2 parameter checking for %s in model %s", model->BSIM3v32modName, here->BSIM3v32name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v32cgdo = (model->BSIM3v32cgdo + pParam->BSIM3v32cf)
                                    * pParam->BSIM3v32weffCV;
                  pParam->BSIM3v32cgso = (model->BSIM3v32cgso + pParam->BSIM3v32cf)
                                    * pParam->BSIM3v32weffCV;
                  pParam->BSIM3v32cgbo = model->BSIM3v32cgbo * pParam->BSIM3v32leffCV;

                  T0 = pParam->BSIM3v32leffCV * pParam->BSIM3v32leffCV;
                  pParam->BSIM3v32tconst = pParam->BSIM3v32u0temp * pParam->BSIM3v32elm / (model->BSIM3v32cox
                                      * pParam->BSIM3v32weffCV * pParam->BSIM3v32leffCV * T0);

                  if (!model->BSIM3v32npeakGiven && model->BSIM3v32gamma1Given)
                  {   T0 = pParam->BSIM3v32gamma1 * model->BSIM3v32cox;
                      pParam->BSIM3v32npeak = 3.021E22 * T0 * T0;
                  }

                  pParam->BSIM3v32phi = 2.0 * Vtm0
                                   * log(pParam->BSIM3v32npeak / ni);

                  pParam->BSIM3v32sqrtPhi = sqrt(pParam->BSIM3v32phi);
                  pParam->BSIM3v32phis3 = pParam->BSIM3v32sqrtPhi * pParam->BSIM3v32phi;

                  pParam->BSIM3v32Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM3v32npeak * 1.0e6))
                                     * pParam->BSIM3v32sqrtPhi;
                  pParam->BSIM3v32sqrtXdep0 = sqrt(pParam->BSIM3v32Xdep0);
                  pParam->BSIM3v32litl = sqrt(3.0 * pParam->BSIM3v32xj
                                    * model->BSIM3v32tox);
                  pParam->BSIM3v32vbi = Vtm0 * log(1.0e20
                                   * pParam->BSIM3v32npeak / (ni * ni));
                  pParam->BSIM3v32cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM3v32npeak * 1.0e6 / 2.0
                                     / pParam->BSIM3v32phi);

                  pParam->BSIM3v32ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM3v32npeak * 1.0e6)) / 3.0;
                  pParam->BSIM3v32acde *= pow((pParam->BSIM3v32npeak / 2.0e16), -0.25);


                  if (model->BSIM3v32k1Given || model->BSIM3v32k2Given)
                  {   if (!model->BSIM3v32k1Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3v32k1 = 0.53;
                      }
                      if (!model->BSIM3v32k2Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3v32k2 = -0.0186;
                      }
                      if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) { /* don't print in sensitivity */
                          if (model->BSIM3v32nsubGiven)
                              fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3v32xtGiven)
                              fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3v32vbxGiven)
                              fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3v32gamma1Given)
                              fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3v32gamma2Given)
                              fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                      }
                  }
                  else
                  {   if (!model->BSIM3v32vbxGiven)
                          pParam->BSIM3v32vbx = pParam->BSIM3v32phi - 7.7348e-4
                                           * pParam->BSIM3v32npeak
                                           * pParam->BSIM3v32xt * pParam->BSIM3v32xt;
                      if (pParam->BSIM3v32vbx > 0.0)
                          pParam->BSIM3v32vbx = -pParam->BSIM3v32vbx;
                      if (pParam->BSIM3v32vbm > 0.0)
                          pParam->BSIM3v32vbm = -pParam->BSIM3v32vbm;

                      if (!model->BSIM3v32gamma1Given)
                          pParam->BSIM3v32gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM3v32npeak)
                                              / model->BSIM3v32cox;
                      if (!model->BSIM3v32gamma2Given)
                          pParam->BSIM3v32gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM3v32nsub)
                                              / model->BSIM3v32cox;

                      T0 = pParam->BSIM3v32gamma1 - pParam->BSIM3v32gamma2;
                      T1 = sqrt(pParam->BSIM3v32phi - pParam->BSIM3v32vbx)
                         - pParam->BSIM3v32sqrtPhi;
                      T2 = sqrt(pParam->BSIM3v32phi * (pParam->BSIM3v32phi
                         - pParam->BSIM3v32vbm)) - pParam->BSIM3v32phi;
                      pParam->BSIM3v32k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3v32vbm);
                      pParam->BSIM3v32k1 = pParam->BSIM3v32gamma2 - 2.0
                                      * pParam->BSIM3v32k2 * sqrt(pParam->BSIM3v32phi
                                      - pParam->BSIM3v32vbm);
                  }

                  if (pParam->BSIM3v32k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM3v32k1 / pParam->BSIM3v32k2;
                      pParam->BSIM3v32vbsc = 0.9 * (pParam->BSIM3v32phi - T0 * T0);
                      if (pParam->BSIM3v32vbsc > -3.0)
                          pParam->BSIM3v32vbsc = -3.0;
                      else if (pParam->BSIM3v32vbsc < -30.0)
                          pParam->BSIM3v32vbsc = -30.0;
                  }
                  else
                  {   pParam->BSIM3v32vbsc = -30.0;
                  }
                  if (pParam->BSIM3v32vbsc > pParam->BSIM3v32vbm)
                      pParam->BSIM3v32vbsc = pParam->BSIM3v32vbm;

                  if (!model->BSIM3v32vfbGiven)
                  {   if (model->BSIM3v32vth0Given)
                      {   pParam->BSIM3v32vfb = model->BSIM3v32type * pParam->BSIM3v32vth0
                                           - pParam->BSIM3v32phi - pParam->BSIM3v32k1
                                           * pParam->BSIM3v32sqrtPhi;
                      }
                      else
                      {   pParam->BSIM3v32vfb = -1.0;
                      }
                  }
                  if (!model->BSIM3v32vth0Given)
                  {   pParam->BSIM3v32vth0 = model->BSIM3v32type * (pParam->BSIM3v32vfb
                                        + pParam->BSIM3v32phi + pParam->BSIM3v32k1
                                        * pParam->BSIM3v32sqrtPhi);
                  }

                  pParam->BSIM3v32k1ox = pParam->BSIM3v32k1 * model->BSIM3v32tox
                                    / model->BSIM3v32toxm;
                  pParam->BSIM3v32k2ox = pParam->BSIM3v32k2 * model->BSIM3v32tox
                                    / model->BSIM3v32toxm;

                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3v32tox
                     * pParam->BSIM3v32Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3v32dsub * pParam->BSIM3v32leff / T1);
                  pParam->BSIM3v32theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3v32drout * pParam->BSIM3v32leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3v32thetaRout = pParam->BSIM3v32pdibl1 * T2
                                         + pParam->BSIM3v32pdibl2;

                  tmp = sqrt(pParam->BSIM3v32Xdep0);
                  tmp1 = pParam->BSIM3v32vbi - pParam->BSIM3v32phi;
                  tmp2 = model->BSIM3v32factor1 * tmp;

                  T0 = -0.5 * pParam->BSIM3v32dvt1w * pParam->BSIM3v32weff
                     * pParam->BSIM3v32leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  T0 = pParam->BSIM3v32dvt0w * T2;
                  T2 = T0 * tmp1;

                  T0 = -0.5 * pParam->BSIM3v32dvt1 * pParam->BSIM3v32leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  T3 = pParam->BSIM3v32dvt0 * T3 * tmp1;

                  T4 = model->BSIM3v32tox * pParam->BSIM3v32phi
                     / (pParam->BSIM3v32weff + pParam->BSIM3v32w0);

                  T0 = sqrt(1.0 + pParam->BSIM3v32nlx / pParam->BSIM3v32leff);
                  T5 = pParam->BSIM3v32k1ox * (T0 - 1.0) * pParam->BSIM3v32sqrtPhi
                     + (pParam->BSIM3v32kt1 + pParam->BSIM3v32kt1l / pParam->BSIM3v32leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM3v32type * pParam->BSIM3v32vth0
                       - T2 - T3 + pParam->BSIM3v32k3 * T4 + T5;
                  pParam->BSIM3v32vfbzb = tmp3 - pParam->BSIM3v32phi - pParam->BSIM3v32k1
                                     * pParam->BSIM3v32sqrtPhi;
                  /* End of vfbzb */
              }

              /* adding delvto  */
              here->BSIM3v32vth0 = pParam->BSIM3v32vth0 + here->BSIM3v32delvto;
              here->BSIM3v32vfb = pParam->BSIM3v32vfb + model->BSIM3v32type * here->BSIM3v32delvto;
              here->BSIM3v32vfbzb = pParam->BSIM3v32vfbzb + model->BSIM3v32type * here->BSIM3v32delvto;

              /* low field mobility multiplier */
              here->BSIM3v32u0temp = pParam->BSIM3v32u0temp * here->BSIM3v32mulu0;

              here->BSIM3v32tconst = here->BSIM3v32u0temp * pParam->BSIM3v32elm / (model->BSIM3v32cox
                                    * pParam->BSIM3v32weffCV * pParam->BSIM3v32leffCV * T0);

              /* process source/drain series resistance */
              /* ACM model */

              double DrainResistance, SourceResistance;

              if (model->BSIM3v32acmMod == 0)
              {
                  DrainResistance = model->BSIM3v32sheetResistance
                                                  * here->BSIM3v32drainSquares;
                  SourceResistance = model->BSIM3v32sheetResistance
                                                   * here->BSIM3v32sourceSquares;
              }
              else /* ACM > 0 */
              {
                  error = ACM_SourceDrainResistances(
                  model->BSIM3v32acmMod,
                  model->BSIM3v32ld,
                  model->BSIM3v32ldif,
                  model->BSIM3v32hdif,
                  model->BSIM3v32wmlt,
                  here->BSIM3v32w,
                  model->BSIM3v32xw,
                  model->BSIM3v32sheetResistance,
                  here->BSIM3v32drainSquaresGiven,
                  model->BSIM3v32rd,
                  model->BSIM3v32rdc,
                  here->BSIM3v32drainSquares,
                  here->BSIM3v32sourceSquaresGiven,
                  model->BSIM3v32rs,
                  model->BSIM3v32rsc,
                  here->BSIM3v32sourceSquares,
                  &DrainResistance,
                  &SourceResistance
                  );
                  if (error)
                      return(error);
              }
              if (DrainResistance > 0.0)
                  here->BSIM3v32drainConductance = 1.0 / DrainResistance;
              else
                  here->BSIM3v32drainConductance = 0.0;

              if (SourceResistance > 0.0)
                  here->BSIM3v32sourceConductance = 1.0 / SourceResistance;
              else
                  here->BSIM3v32sourceConductance = 0.0;

              here->BSIM3v32cgso = pParam->BSIM3v32cgso;
              here->BSIM3v32cgdo = pParam->BSIM3v32cgdo;

              Nvtm = model->BSIM3v32vtm * model->BSIM3v32jctEmissionCoeff;
              if (model->BSIM3v32acmMod == 0)
              {
                if ((here->BSIM3v32sourceArea <= 0.0) &&
                    (here->BSIM3v32sourcePerimeter <= 0.0))
                {   SourceSatCurrent = 1.0e-14;
                }
                else
                {   SourceSatCurrent = here->BSIM3v32sourceArea
                                     * model->BSIM3v32jctTempSatCurDensity
                                     + here->BSIM3v32sourcePerimeter
                                     * model->BSIM3v32jctSidewallTempSatCurDensity;
                }

                if ((here->BSIM3v32drainArea <= 0.0) &&
                    (here->BSIM3v32drainPerimeter <= 0.0))
                {   DrainSatCurrent = 1.0e-14;
                }
                else
                {   DrainSatCurrent = here->BSIM3v32drainArea
                                    * model->BSIM3v32jctTempSatCurDensity
                                    + here->BSIM3v32drainPerimeter
                                    * model->BSIM3v32jctSidewallTempSatCurDensity;
                }
              }
              else /* ACM > 0 */
              {
                error = ACM_saturationCurrents(
                model->BSIM3v32acmMod,
                model->BSIM3v32calcacm,
                here->BSIM3v32geo,
                model->BSIM3v32hdif,
                model->BSIM3v32wmlt,
                here->BSIM3v32w,
                model->BSIM3v32xw,
                model->BSIM3v32jctTempSatCurDensity,
                model->BSIM3v32jctSidewallTempSatCurDensity,
                here->BSIM3v32drainAreaGiven,
                here->BSIM3v32drainArea,
                here->BSIM3v32drainPerimeterGiven,
                here->BSIM3v32drainPerimeter,
                here->BSIM3v32sourceAreaGiven,
                here->BSIM3v32sourceArea,
                here->BSIM3v32sourcePerimeterGiven,
                here->BSIM3v32sourcePerimeter,
                &DrainSatCurrent,
                &SourceSatCurrent
                );
                if (error)
                    return(error);
              }

              if ((SourceSatCurrent > 0.0) && (model->BSIM3v32ijth > 0.0))
              {   here->BSIM3v32vjsm = Nvtm * log(model->BSIM3v32ijth
                                  / SourceSatCurrent + 1.0);
                  /* Added revision dependent code */
                  switch (model->BSIM3v32intVersion) {
                  case BSIM3v32V324:
                  case BSIM3v32V323:
                  case BSIM3v32V322:
                          here->BSIM3v32IsEvjsm =
                                  SourceSatCurrent * exp(here->BSIM3v32vjsm / Nvtm);
                          break;
                  case BSIM3v32V32:
                  default:
                          /* Do nothing */
                      break;
                  }
              }

              if ((DrainSatCurrent > 0.0) && (model->BSIM3v32ijth > 0.0))
              {   here->BSIM3v32vjdm = Nvtm * log(model->BSIM3v32ijth
                                  / DrainSatCurrent + 1.0);
                  /* Added revision dependent code */
                  switch (model->BSIM3v32intVersion) {
                  case BSIM3v32V324:
                  case BSIM3v32V323:
                  case BSIM3v32V322:
                          here->BSIM3v32IsEvjdm =
                                  DrainSatCurrent * exp(here->BSIM3v32vjdm / Nvtm);
                          break;
                  case BSIM3v32V32:
                  default:
                          /* Do nothing */
                      break;
                  }
              }
         }
    }
    return(OK);
}

