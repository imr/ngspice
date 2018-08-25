/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3temp.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
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
BSIM3temp(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM3model *model = (BSIM3model*) inModel;
BSIM3instance *here;
struct bsim3SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double Nvtm, SourceSatCurrent, DrainSatCurrent;
int Size_Not_Found, error;

/*  loop through all the BSIM3 device models */
    for (; model != NULL; model = BSIM3nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3bulkJctPotential < 0.1)
         {   model->BSIM3bulkJctPotential = 0.1;
             fprintf(stderr, "Given pb is less than 0.1. Pb is set to 0.1.\n");
         }
         if (model->BSIM3sidewallJctPotential < 0.1)
         {   model->BSIM3sidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbsw is less than 0.1. Pbsw is set to 0.1.\n");
         }
         if (model->BSIM3GatesidewallJctPotential < 0.1)
         {   model->BSIM3GatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswg is less than 0.1. Pbswg is set to 0.1.\n");
         }

         struct bsim3SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim3SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM3tnom;
         TRatio = Temp / Tnom;

         model->BSIM3vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3factor1 = sqrt(EPSSI / EPSOX * model->BSIM3tox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3vtm + model->BSIM3jctTempExponent
                * log(Temp / Tnom);
             T1 = exp(T0 / model->BSIM3jctEmissionCoeff);
             model->BSIM3jctTempSatCurDensity = model->BSIM3jctSatCurDensity
                                              * T1;
             model->BSIM3jctSidewallTempSatCurDensity
                         = model->BSIM3jctSidewallSatCurDensity * T1;
         }
         else
         {   model->BSIM3jctTempSatCurDensity = model->BSIM3jctSatCurDensity;
             model->BSIM3jctSidewallTempSatCurDensity
                        = model->BSIM3jctSidewallSatCurDensity;
         }

         if (model->BSIM3jctTempSatCurDensity < 0.0)
             model->BSIM3jctTempSatCurDensity = 0.0;
         if (model->BSIM3jctSidewallTempSatCurDensity < 0.0)
             model->BSIM3jctSidewallTempSatCurDensity = 0.0;

         /* Temperature dependence of D/B and S/B diode capacitance begins */
         delTemp = ckt->CKTtemp - model->BSIM3tnom;
         T0 = model->BSIM3tcj * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM3unitAreaTempJctCap = model->BSIM3unitAreaJctCap * (1.0 + T0);
         }
         else if (model->BSIM3unitAreaJctCap > 0.0)
         {   model->BSIM3unitAreaTempJctCap = 0.0;
             fprintf(stderr, "Temperature effect has caused cj to be negative. Cj is clamped to zero.\n");
         }
         T0 = model->BSIM3tcjsw * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM3unitLengthSidewallTempJctCap = model->BSIM3unitLengthSidewallJctCap * (1.0 + T0);
         }
         else if (model->BSIM3unitLengthSidewallJctCap > 0.0)
         {   model->BSIM3unitLengthSidewallTempJctCap = 0.0;
             fprintf(stderr, "Temperature effect has caused cjsw to be negative. Cjsw is clamped to zero.\n");
         }
         T0 = model->BSIM3tcjswg * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM3unitLengthGateSidewallTempJctCap = model->BSIM3unitLengthGateSidewallJctCap * (1.0 + T0);
         }
         else if (model->BSIM3unitLengthGateSidewallJctCap > 0.0)
         {   model->BSIM3unitLengthGateSidewallTempJctCap = 0.0;
             fprintf(stderr, "Temperature effect has caused cjswg to be negative. Cjswg is clamped to zero.\n");
         }

         model->BSIM3PhiB = model->BSIM3bulkJctPotential
                          - model->BSIM3tpb * delTemp;
         if (model->BSIM3PhiB < 0.01)
         {   model->BSIM3PhiB = 0.01;
             fprintf(stderr, "Temperature effect has caused pb to be less than 0.01. Pb is clamped to 0.01.\n");
         }
         model->BSIM3PhiBSW = model->BSIM3sidewallJctPotential
                            - model->BSIM3tpbsw * delTemp;
         if (model->BSIM3PhiBSW <= 0.01)
         {   model->BSIM3PhiBSW = 0.01;
             fprintf(stderr, "Temperature effect has caused pbsw to be less than 0.01. Pbsw is clamped to 0.01.\n");
         }
         model->BSIM3PhiBSWG = model->BSIM3GatesidewallJctPotential
                             - model->BSIM3tpbswg * delTemp;
         if (model->BSIM3PhiBSWG <= 0.01)
         {   model->BSIM3PhiBSWG = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswg to be less than 0.01. Pbswg is clamped to 0.01.\n");
         }
         /* End of junction capacitance */

         /* loop through all the instances of the model */
         /* MCJ: Length and Width not initialized */
         for (here = BSIM3instances(model); here != NULL;
              here = BSIM3nextInstance(here))
         {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM3l == pSizeDependParamKnot->Length)
                      && (here->BSIM3w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                      pParam = here->pParam; /*bug-fix  */
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim3SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->BSIM3l;
                  Wdrn = here->BSIM3w;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;

                  T0 = pow(Ldrn, model->BSIM3Lln);
                  T1 = pow(Wdrn, model->BSIM3Lwn);
                  tmp1 = model->BSIM3Ll / T0 + model->BSIM3Lw / T1
                       + model->BSIM3Lwl / (T0 * T1);
                  pParam->BSIM3dl = model->BSIM3Lint + tmp1;
                  tmp2 = model->BSIM3Llc / T0 + model->BSIM3Lwc / T1
                       + model->BSIM3Lwlc / (T0 * T1);
                  pParam->BSIM3dlc = model->BSIM3dlc + tmp2;

                  T2 = pow(Ldrn, model->BSIM3Wln);
                  T3 = pow(Wdrn, model->BSIM3Wwn);
                  tmp1 = model->BSIM3Wl / T2 + model->BSIM3Ww / T3
                       + model->BSIM3Wwl / (T2 * T3);
                  pParam->BSIM3dw = model->BSIM3Wint + tmp1;
                  tmp2 = model->BSIM3Wlc / T2 + model->BSIM3Wwc / T3
                       + model->BSIM3Wwlc / (T2 * T3);
                  pParam->BSIM3dwc = model->BSIM3dwc + tmp2;

                  pParam->BSIM3leff = here->BSIM3l + model->BSIM3xl - 2.0 * pParam->BSIM3dl;
                  if (pParam->BSIM3leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM3modName, here->BSIM3name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3weff = here->BSIM3w + model->BSIM3xw - 2.0 * pParam->BSIM3dw;
                  if (pParam->BSIM3weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM3modName, here->BSIM3name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3leffCV = here->BSIM3l + model->BSIM3xl - 2.0 * pParam->BSIM3dlc;
                  if (pParam->BSIM3leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM3modName, here->BSIM3name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3weffCV = here->BSIM3w + model->BSIM3xw - 2.0 * pParam->BSIM3dwc;
                  if (pParam->BSIM3weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM3modName, here->BSIM3name);
                      return(E_BADPARM);
                  }


                  if (model->BSIM3binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM3leff;
                      Inv_W = 1.0e-6 / pParam->BSIM3weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM3leff
                             * pParam->BSIM3weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM3leff;
                      Inv_W = 1.0 / pParam->BSIM3weff;
                      Inv_LW = 1.0 / (pParam->BSIM3leff
                             * pParam->BSIM3weff);
                  }
                  pParam->BSIM3cdsc = model->BSIM3cdsc
                                    + model->BSIM3lcdsc * Inv_L
                                    + model->BSIM3wcdsc * Inv_W
                                    + model->BSIM3pcdsc * Inv_LW;
                  pParam->BSIM3cdscb = model->BSIM3cdscb
                                     + model->BSIM3lcdscb * Inv_L
                                     + model->BSIM3wcdscb * Inv_W
                                     + model->BSIM3pcdscb * Inv_LW;

                  pParam->BSIM3cdscd = model->BSIM3cdscd
                                     + model->BSIM3lcdscd * Inv_L
                                     + model->BSIM3wcdscd * Inv_W
                                     + model->BSIM3pcdscd * Inv_LW;

                  pParam->BSIM3cit = model->BSIM3cit
                                   + model->BSIM3lcit * Inv_L
                                   + model->BSIM3wcit * Inv_W
                                   + model->BSIM3pcit * Inv_LW;
                  pParam->BSIM3nfactor = model->BSIM3nfactor
                                       + model->BSIM3lnfactor * Inv_L
                                       + model->BSIM3wnfactor * Inv_W
                                       + model->BSIM3pnfactor * Inv_LW;
                  pParam->BSIM3xj = model->BSIM3xj
                                  + model->BSIM3lxj * Inv_L
                                  + model->BSIM3wxj * Inv_W
                                  + model->BSIM3pxj * Inv_LW;
                  pParam->BSIM3vsat = model->BSIM3vsat
                                    + model->BSIM3lvsat * Inv_L
                                    + model->BSIM3wvsat * Inv_W
                                    + model->BSIM3pvsat * Inv_LW;
                  pParam->BSIM3at = model->BSIM3at
                                  + model->BSIM3lat * Inv_L
                                  + model->BSIM3wat * Inv_W
                                  + model->BSIM3pat * Inv_LW;
                  pParam->BSIM3a0 = model->BSIM3a0
                                  + model->BSIM3la0 * Inv_L
                                  + model->BSIM3wa0 * Inv_W
                                  + model->BSIM3pa0 * Inv_LW;

                  pParam->BSIM3ags = model->BSIM3ags
                                  + model->BSIM3lags * Inv_L
                                  + model->BSIM3wags * Inv_W
                                  + model->BSIM3pags * Inv_LW;

                  pParam->BSIM3a1 = model->BSIM3a1
                                  + model->BSIM3la1 * Inv_L
                                  + model->BSIM3wa1 * Inv_W
                                  + model->BSIM3pa1 * Inv_LW;
                  pParam->BSIM3a2 = model->BSIM3a2
                                  + model->BSIM3la2 * Inv_L
                                  + model->BSIM3wa2 * Inv_W
                                  + model->BSIM3pa2 * Inv_LW;
                  pParam->BSIM3keta = model->BSIM3keta
                                    + model->BSIM3lketa * Inv_L
                                    + model->BSIM3wketa * Inv_W
                                    + model->BSIM3pketa * Inv_LW;
                  pParam->BSIM3nsub = model->BSIM3nsub
                                    + model->BSIM3lnsub * Inv_L
                                    + model->BSIM3wnsub * Inv_W
                                    + model->BSIM3pnsub * Inv_LW;
                  pParam->BSIM3npeak = model->BSIM3npeak
                                     + model->BSIM3lnpeak * Inv_L
                                     + model->BSIM3wnpeak * Inv_W
                                     + model->BSIM3pnpeak * Inv_LW;
                  pParam->BSIM3ngate = model->BSIM3ngate
                                     + model->BSIM3lngate * Inv_L
                                     + model->BSIM3wngate * Inv_W
                                     + model->BSIM3pngate * Inv_LW;
                  pParam->BSIM3gamma1 = model->BSIM3gamma1
                                      + model->BSIM3lgamma1 * Inv_L
                                      + model->BSIM3wgamma1 * Inv_W
                                      + model->BSIM3pgamma1 * Inv_LW;
                  pParam->BSIM3gamma2 = model->BSIM3gamma2
                                      + model->BSIM3lgamma2 * Inv_L
                                      + model->BSIM3wgamma2 * Inv_W
                                      + model->BSIM3pgamma2 * Inv_LW;
                  pParam->BSIM3vbx = model->BSIM3vbx
                                   + model->BSIM3lvbx * Inv_L
                                   + model->BSIM3wvbx * Inv_W
                                   + model->BSIM3pvbx * Inv_LW;
                  pParam->BSIM3vbm = model->BSIM3vbm
                                   + model->BSIM3lvbm * Inv_L
                                   + model->BSIM3wvbm * Inv_W
                                   + model->BSIM3pvbm * Inv_LW;
                  pParam->BSIM3xt = model->BSIM3xt
                                   + model->BSIM3lxt * Inv_L
                                   + model->BSIM3wxt * Inv_W
                                   + model->BSIM3pxt * Inv_LW;
                  pParam->BSIM3vfb = model->BSIM3vfb
                                   + model->BSIM3lvfb * Inv_L
                                   + model->BSIM3wvfb * Inv_W
                                   + model->BSIM3pvfb * Inv_LW;
                  pParam->BSIM3k1 = model->BSIM3k1
                                  + model->BSIM3lk1 * Inv_L
                                  + model->BSIM3wk1 * Inv_W
                                  + model->BSIM3pk1 * Inv_LW;
                  pParam->BSIM3kt1 = model->BSIM3kt1
                                   + model->BSIM3lkt1 * Inv_L
                                   + model->BSIM3wkt1 * Inv_W
                                   + model->BSIM3pkt1 * Inv_LW;
                  pParam->BSIM3kt1l = model->BSIM3kt1l
                                    + model->BSIM3lkt1l * Inv_L
                                    + model->BSIM3wkt1l * Inv_W
                                    + model->BSIM3pkt1l * Inv_LW;
                  pParam->BSIM3k2 = model->BSIM3k2
                                  + model->BSIM3lk2 * Inv_L
                                  + model->BSIM3wk2 * Inv_W
                                  + model->BSIM3pk2 * Inv_LW;
                  pParam->BSIM3kt2 = model->BSIM3kt2
                                   + model->BSIM3lkt2 * Inv_L
                                   + model->BSIM3wkt2 * Inv_W
                                   + model->BSIM3pkt2 * Inv_LW;
                  pParam->BSIM3k3 = model->BSIM3k3
                                  + model->BSIM3lk3 * Inv_L
                                  + model->BSIM3wk3 * Inv_W
                                  + model->BSIM3pk3 * Inv_LW;
                  pParam->BSIM3k3b = model->BSIM3k3b
                                   + model->BSIM3lk3b * Inv_L
                                   + model->BSIM3wk3b * Inv_W
                                   + model->BSIM3pk3b * Inv_LW;
                  pParam->BSIM3w0 = model->BSIM3w0
                                  + model->BSIM3lw0 * Inv_L
                                  + model->BSIM3ww0 * Inv_W
                                  + model->BSIM3pw0 * Inv_LW;
                  pParam->BSIM3nlx = model->BSIM3nlx
                                   + model->BSIM3lnlx * Inv_L
                                   + model->BSIM3wnlx * Inv_W
                                   + model->BSIM3pnlx * Inv_LW;
                  pParam->BSIM3dvt0 = model->BSIM3dvt0
                                    + model->BSIM3ldvt0 * Inv_L
                                    + model->BSIM3wdvt0 * Inv_W
                                    + model->BSIM3pdvt0 * Inv_LW;
                  pParam->BSIM3dvt1 = model->BSIM3dvt1
                                    + model->BSIM3ldvt1 * Inv_L
                                    + model->BSIM3wdvt1 * Inv_W
                                    + model->BSIM3pdvt1 * Inv_LW;
                  pParam->BSIM3dvt2 = model->BSIM3dvt2
                                    + model->BSIM3ldvt2 * Inv_L
                                    + model->BSIM3wdvt2 * Inv_W
                                    + model->BSIM3pdvt2 * Inv_LW;
                  pParam->BSIM3dvt0w = model->BSIM3dvt0w
                                    + model->BSIM3ldvt0w * Inv_L
                                    + model->BSIM3wdvt0w * Inv_W
                                    + model->BSIM3pdvt0w * Inv_LW;
                  pParam->BSIM3dvt1w = model->BSIM3dvt1w
                                    + model->BSIM3ldvt1w * Inv_L
                                    + model->BSIM3wdvt1w * Inv_W
                                    + model->BSIM3pdvt1w * Inv_LW;
                  pParam->BSIM3dvt2w = model->BSIM3dvt2w
                                    + model->BSIM3ldvt2w * Inv_L
                                    + model->BSIM3wdvt2w * Inv_W
                                    + model->BSIM3pdvt2w * Inv_LW;
                  pParam->BSIM3drout = model->BSIM3drout
                                     + model->BSIM3ldrout * Inv_L
                                     + model->BSIM3wdrout * Inv_W
                                     + model->BSIM3pdrout * Inv_LW;
                  pParam->BSIM3dsub = model->BSIM3dsub
                                    + model->BSIM3ldsub * Inv_L
                                    + model->BSIM3wdsub * Inv_W
                                    + model->BSIM3pdsub * Inv_LW;
                  pParam->BSIM3vth0 = model->BSIM3vth0
                                    + model->BSIM3lvth0 * Inv_L
                                    + model->BSIM3wvth0 * Inv_W
                                    + model->BSIM3pvth0 * Inv_LW;
                  pParam->BSIM3ua = model->BSIM3ua
                                  + model->BSIM3lua * Inv_L
                                  + model->BSIM3wua * Inv_W
                                  + model->BSIM3pua * Inv_LW;
                  pParam->BSIM3ua1 = model->BSIM3ua1
                                   + model->BSIM3lua1 * Inv_L
                                   + model->BSIM3wua1 * Inv_W
                                   + model->BSIM3pua1 * Inv_LW;
                  pParam->BSIM3ub = model->BSIM3ub
                                  + model->BSIM3lub * Inv_L
                                  + model->BSIM3wub * Inv_W
                                  + model->BSIM3pub * Inv_LW;
                  pParam->BSIM3ub1 = model->BSIM3ub1
                                   + model->BSIM3lub1 * Inv_L
                                   + model->BSIM3wub1 * Inv_W
                                   + model->BSIM3pub1 * Inv_LW;
                  pParam->BSIM3uc = model->BSIM3uc
                                  + model->BSIM3luc * Inv_L
                                  + model->BSIM3wuc * Inv_W
                                  + model->BSIM3puc * Inv_LW;
                  pParam->BSIM3uc1 = model->BSIM3uc1
                                   + model->BSIM3luc1 * Inv_L
                                   + model->BSIM3wuc1 * Inv_W
                                   + model->BSIM3puc1 * Inv_LW;
                  pParam->BSIM3u0 = model->BSIM3u0
                                  + model->BSIM3lu0 * Inv_L
                                  + model->BSIM3wu0 * Inv_W
                                  + model->BSIM3pu0 * Inv_LW;
                  pParam->BSIM3ute = model->BSIM3ute
                                   + model->BSIM3lute * Inv_L
                                   + model->BSIM3wute * Inv_W
                                   + model->BSIM3pute * Inv_LW;
                  pParam->BSIM3voff = model->BSIM3voff
                                    + model->BSIM3lvoff * Inv_L
                                    + model->BSIM3wvoff * Inv_W
                                    + model->BSIM3pvoff * Inv_LW;
                  pParam->BSIM3delta = model->BSIM3delta
                                     + model->BSIM3ldelta * Inv_L
                                     + model->BSIM3wdelta * Inv_W
                                     + model->BSIM3pdelta * Inv_LW;
                  pParam->BSIM3rdsw = model->BSIM3rdsw
                                    + model->BSIM3lrdsw * Inv_L
                                    + model->BSIM3wrdsw * Inv_W
                                    + model->BSIM3prdsw * Inv_LW;
                  pParam->BSIM3prwg = model->BSIM3prwg
                                    + model->BSIM3lprwg * Inv_L
                                    + model->BSIM3wprwg * Inv_W
                                    + model->BSIM3pprwg * Inv_LW;
                  pParam->BSIM3prwb = model->BSIM3prwb
                                    + model->BSIM3lprwb * Inv_L
                                    + model->BSIM3wprwb * Inv_W
                                    + model->BSIM3pprwb * Inv_LW;
                  pParam->BSIM3prt = model->BSIM3prt
                                    + model->BSIM3lprt * Inv_L
                                    + model->BSIM3wprt * Inv_W
                                    + model->BSIM3pprt * Inv_LW;
                  pParam->BSIM3eta0 = model->BSIM3eta0
                                    + model->BSIM3leta0 * Inv_L
                                    + model->BSIM3weta0 * Inv_W
                                    + model->BSIM3peta0 * Inv_LW;
                  pParam->BSIM3etab = model->BSIM3etab
                                    + model->BSIM3letab * Inv_L
                                    + model->BSIM3wetab * Inv_W
                                    + model->BSIM3petab * Inv_LW;
                  pParam->BSIM3pclm = model->BSIM3pclm
                                    + model->BSIM3lpclm * Inv_L
                                    + model->BSIM3wpclm * Inv_W
                                    + model->BSIM3ppclm * Inv_LW;
                  pParam->BSIM3pdibl1 = model->BSIM3pdibl1
                                      + model->BSIM3lpdibl1 * Inv_L
                                      + model->BSIM3wpdibl1 * Inv_W
                                      + model->BSIM3ppdibl1 * Inv_LW;
                  pParam->BSIM3pdibl2 = model->BSIM3pdibl2
                                      + model->BSIM3lpdibl2 * Inv_L
                                      + model->BSIM3wpdibl2 * Inv_W
                                      + model->BSIM3ppdibl2 * Inv_LW;
                  pParam->BSIM3pdiblb = model->BSIM3pdiblb
                                      + model->BSIM3lpdiblb * Inv_L
                                      + model->BSIM3wpdiblb * Inv_W
                                      + model->BSIM3ppdiblb * Inv_LW;
                  pParam->BSIM3pscbe1 = model->BSIM3pscbe1
                                      + model->BSIM3lpscbe1 * Inv_L
                                      + model->BSIM3wpscbe1 * Inv_W
                                      + model->BSIM3ppscbe1 * Inv_LW;
                  pParam->BSIM3pscbe2 = model->BSIM3pscbe2
                                      + model->BSIM3lpscbe2 * Inv_L
                                      + model->BSIM3wpscbe2 * Inv_W
                                      + model->BSIM3ppscbe2 * Inv_LW;
                  pParam->BSIM3pvag = model->BSIM3pvag
                                    + model->BSIM3lpvag * Inv_L
                                    + model->BSIM3wpvag * Inv_W
                                    + model->BSIM3ppvag * Inv_LW;
                  pParam->BSIM3wr = model->BSIM3wr
                                  + model->BSIM3lwr * Inv_L
                                  + model->BSIM3wwr * Inv_W
                                  + model->BSIM3pwr * Inv_LW;
                  pParam->BSIM3dwg = model->BSIM3dwg
                                   + model->BSIM3ldwg * Inv_L
                                   + model->BSIM3wdwg * Inv_W
                                   + model->BSIM3pdwg * Inv_LW;
                  pParam->BSIM3dwb = model->BSIM3dwb
                                   + model->BSIM3ldwb * Inv_L
                                   + model->BSIM3wdwb * Inv_W
                                   + model->BSIM3pdwb * Inv_LW;
                  pParam->BSIM3b0 = model->BSIM3b0
                                  + model->BSIM3lb0 * Inv_L
                                  + model->BSIM3wb0 * Inv_W
                                  + model->BSIM3pb0 * Inv_LW;
                  pParam->BSIM3b1 = model->BSIM3b1
                                  + model->BSIM3lb1 * Inv_L
                                  + model->BSIM3wb1 * Inv_W
                                  + model->BSIM3pb1 * Inv_LW;
                  pParam->BSIM3alpha0 = model->BSIM3alpha0
                                      + model->BSIM3lalpha0 * Inv_L
                                      + model->BSIM3walpha0 * Inv_W
                                      + model->BSIM3palpha0 * Inv_LW;
                  pParam->BSIM3alpha1 = model->BSIM3alpha1
                                      + model->BSIM3lalpha1 * Inv_L
                                      + model->BSIM3walpha1 * Inv_W
                                      + model->BSIM3palpha1 * Inv_LW;
                  pParam->BSIM3beta0 = model->BSIM3beta0
                                     + model->BSIM3lbeta0 * Inv_L
                                     + model->BSIM3wbeta0 * Inv_W
                                     + model->BSIM3pbeta0 * Inv_LW;
                  /* CV model */
                  pParam->BSIM3elm = model->BSIM3elm
                                  + model->BSIM3lelm * Inv_L
                                  + model->BSIM3welm * Inv_W
                                  + model->BSIM3pelm * Inv_LW;
                  pParam->BSIM3cgsl = model->BSIM3cgsl
                                    + model->BSIM3lcgsl * Inv_L
                                    + model->BSIM3wcgsl * Inv_W
                                    + model->BSIM3pcgsl * Inv_LW;
                  pParam->BSIM3cgdl = model->BSIM3cgdl
                                    + model->BSIM3lcgdl * Inv_L
                                    + model->BSIM3wcgdl * Inv_W
                                    + model->BSIM3pcgdl * Inv_LW;
                  pParam->BSIM3ckappa = model->BSIM3ckappa
                                      + model->BSIM3lckappa * Inv_L
                                      + model->BSIM3wckappa * Inv_W
                                      + model->BSIM3pckappa * Inv_LW;
                  pParam->BSIM3cf = model->BSIM3cf
                                  + model->BSIM3lcf * Inv_L
                                  + model->BSIM3wcf * Inv_W
                                  + model->BSIM3pcf * Inv_LW;
                  pParam->BSIM3clc = model->BSIM3clc
                                   + model->BSIM3lclc * Inv_L
                                   + model->BSIM3wclc * Inv_W
                                   + model->BSIM3pclc * Inv_LW;
                  pParam->BSIM3cle = model->BSIM3cle
                                   + model->BSIM3lcle * Inv_L
                                   + model->BSIM3wcle * Inv_W
                                   + model->BSIM3pcle * Inv_LW;
                  pParam->BSIM3vfbcv = model->BSIM3vfbcv
                                     + model->BSIM3lvfbcv * Inv_L
                                     + model->BSIM3wvfbcv * Inv_W
                                     + model->BSIM3pvfbcv * Inv_LW;
                  pParam->BSIM3acde = model->BSIM3acde
                                    + model->BSIM3lacde * Inv_L
                                    + model->BSIM3wacde * Inv_W
                                    + model->BSIM3pacde * Inv_LW;
                  pParam->BSIM3moin = model->BSIM3moin
                                    + model->BSIM3lmoin * Inv_L
                                    + model->BSIM3wmoin * Inv_W
                                    + model->BSIM3pmoin * Inv_LW;
                  pParam->BSIM3noff = model->BSIM3noff
                                    + model->BSIM3lnoff * Inv_L
                                    + model->BSIM3wnoff * Inv_W
                                    + model->BSIM3pnoff * Inv_LW;
                  pParam->BSIM3voffcv = model->BSIM3voffcv
                                      + model->BSIM3lvoffcv * Inv_L
                                      + model->BSIM3wvoffcv * Inv_W
                                      + model->BSIM3pvoffcv * Inv_LW;

                  pParam->BSIM3abulkCVfactor = 1.0 + pow((pParam->BSIM3clc
                                             / pParam->BSIM3leffCV),
                                             pParam->BSIM3cle);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM3ua = pParam->BSIM3ua + pParam->BSIM3ua1 * T0;
                  pParam->BSIM3ub = pParam->BSIM3ub + pParam->BSIM3ub1 * T0;
                  pParam->BSIM3uc = pParam->BSIM3uc + pParam->BSIM3uc1 * T0;
                  if (pParam->BSIM3u0 > 1.0)
                      pParam->BSIM3u0 = pParam->BSIM3u0 / 1.0e4;

                  pParam->BSIM3u0temp = pParam->BSIM3u0
                                      * pow(TRatio, pParam->BSIM3ute);
                  pParam->BSIM3vsattemp = pParam->BSIM3vsat - pParam->BSIM3at
                                        * T0;
                  pParam->BSIM3rds0 = (pParam->BSIM3rdsw + pParam->BSIM3prt * T0)
                                    / pow(pParam->BSIM3weff * 1E6, pParam->BSIM3wr);

                  if (BSIM3checkModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during BSIM3V3.3 parameter checking for %s in model %s", model->BSIM3modName, here->BSIM3name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3cgdo = (model->BSIM3cgdo + pParam->BSIM3cf)
                                    * pParam->BSIM3weffCV;
                  pParam->BSIM3cgso = (model->BSIM3cgso + pParam->BSIM3cf)
                                    * pParam->BSIM3weffCV;
                  pParam->BSIM3cgbo = model->BSIM3cgbo * pParam->BSIM3leffCV;

                  T0 = pParam->BSIM3leffCV * pParam->BSIM3leffCV;
                  pParam->BSIM3tconst = pParam->BSIM3u0temp * pParam->BSIM3elm / (model->BSIM3cox
                                      * pParam->BSIM3weffCV * pParam->BSIM3leffCV * T0);

                  if (!model->BSIM3npeakGiven && model->BSIM3gamma1Given)
                  {   T0 = pParam->BSIM3gamma1 * model->BSIM3cox;
                      pParam->BSIM3npeak = 3.021E22 * T0 * T0;
                  }

                  pParam->BSIM3phi = 2.0 * Vtm0
                                   * log(pParam->BSIM3npeak / ni);

                  pParam->BSIM3sqrtPhi = sqrt(pParam->BSIM3phi);
                  pParam->BSIM3phis3 = pParam->BSIM3sqrtPhi * pParam->BSIM3phi;

                  pParam->BSIM3Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM3npeak * 1.0e6))
                                     * pParam->BSIM3sqrtPhi;
                  pParam->BSIM3sqrtXdep0 = sqrt(pParam->BSIM3Xdep0);
                  pParam->BSIM3litl = sqrt(3.0 * pParam->BSIM3xj
                                    * model->BSIM3tox);
                  pParam->BSIM3vbi = Vtm0 * log(1.0e20
                                   * pParam->BSIM3npeak / (ni * ni));
                  pParam->BSIM3cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM3npeak * 1.0e6 / 2.0
                                     / pParam->BSIM3phi);

                  pParam->BSIM3ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM3npeak * 1.0e6)) / 3.0;
                  pParam->BSIM3acde *= pow((pParam->BSIM3npeak / 2.0e16), -0.25);


                  if (model->BSIM3k1Given || model->BSIM3k2Given)
                  {   if (!model->BSIM3k1Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3k1 = 0.53;
                      }
                      if (!model->BSIM3k2Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3k2 = -0.0186;
                      }
                      if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) { /* don't print in sensitivity */
                          if (model->BSIM3nsubGiven)
                              fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3xtGiven)
                              fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3vbxGiven)
                              fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3gamma1Given)
                              fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                          if (model->BSIM3gamma2Given)
                              fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                      }
                  }
                  else
                  {   if (!model->BSIM3vbxGiven)
                          pParam->BSIM3vbx = pParam->BSIM3phi - 7.7348e-4
                                           * pParam->BSIM3npeak
                                           * pParam->BSIM3xt * pParam->BSIM3xt;
                      if (pParam->BSIM3vbx > 0.0)
                          pParam->BSIM3vbx = -pParam->BSIM3vbx;
                      if (pParam->BSIM3vbm > 0.0)
                          pParam->BSIM3vbm = -pParam->BSIM3vbm;

                      if (!model->BSIM3gamma1Given)
                          pParam->BSIM3gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM3npeak)
                                              / model->BSIM3cox;
                      if (!model->BSIM3gamma2Given)
                          pParam->BSIM3gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM3nsub)
                                              / model->BSIM3cox;

                      T0 = pParam->BSIM3gamma1 - pParam->BSIM3gamma2;
                      T1 = sqrt(pParam->BSIM3phi - pParam->BSIM3vbx)
                         - pParam->BSIM3sqrtPhi;
                      T2 = sqrt(pParam->BSIM3phi * (pParam->BSIM3phi
                         - pParam->BSIM3vbm)) - pParam->BSIM3phi;
                      pParam->BSIM3k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3vbm);
                      pParam->BSIM3k1 = pParam->BSIM3gamma2 - 2.0
                                      * pParam->BSIM3k2 * sqrt(pParam->BSIM3phi
                                      - pParam->BSIM3vbm);
                  }

                  if (pParam->BSIM3k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM3k1 / pParam->BSIM3k2;
                      pParam->BSIM3vbsc = 0.9 * (pParam->BSIM3phi - T0 * T0);
                      if (pParam->BSIM3vbsc > -3.0)
                          pParam->BSIM3vbsc = -3.0;
                      else if (pParam->BSIM3vbsc < -30.0)
                          pParam->BSIM3vbsc = -30.0;
                  }
                  else
                  {   pParam->BSIM3vbsc = -30.0;
                  }
                  if (pParam->BSIM3vbsc > pParam->BSIM3vbm)
                      pParam->BSIM3vbsc = pParam->BSIM3vbm;

                  if (!model->BSIM3vfbGiven)
                  {   if (model->BSIM3vth0Given)
                      {   pParam->BSIM3vfb = model->BSIM3type * pParam->BSIM3vth0
                                           - pParam->BSIM3phi - pParam->BSIM3k1
                                           * pParam->BSIM3sqrtPhi;
                      }
                      else
                      {   pParam->BSIM3vfb = -1.0;
                      }
                  }
                  if (!model->BSIM3vth0Given)
                  {   pParam->BSIM3vth0 = model->BSIM3type * (pParam->BSIM3vfb
                                        + pParam->BSIM3phi + pParam->BSIM3k1
                                        * pParam->BSIM3sqrtPhi);
                  }

                  pParam->BSIM3k1ox = pParam->BSIM3k1 * model->BSIM3tox
                                    / model->BSIM3toxm;
                  pParam->BSIM3k2ox = pParam->BSIM3k2 * model->BSIM3tox
                                    / model->BSIM3toxm;

                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3tox
                     * pParam->BSIM3Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3dsub * pParam->BSIM3leff / T1);
                  pParam->BSIM3theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3drout * pParam->BSIM3leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3thetaRout = pParam->BSIM3pdibl1 * T2
                                         + pParam->BSIM3pdibl2;

                  tmp = sqrt(pParam->BSIM3Xdep0);
                  tmp1 = pParam->BSIM3vbi - pParam->BSIM3phi;
                  tmp2 = model->BSIM3factor1 * tmp;

                  T0 = -0.5 * pParam->BSIM3dvt1w * pParam->BSIM3weff
                     * pParam->BSIM3leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  T0 = pParam->BSIM3dvt0w * T2;
                  T2 = T0 * tmp1;

                  T0 = -0.5 * pParam->BSIM3dvt1 * pParam->BSIM3leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  T3 = pParam->BSIM3dvt0 * T3 * tmp1;

                  T4 = model->BSIM3tox * pParam->BSIM3phi
                     / (pParam->BSIM3weff + pParam->BSIM3w0);

                  T0 = sqrt(1.0 + pParam->BSIM3nlx / pParam->BSIM3leff);
                  T5 = pParam->BSIM3k1ox * (T0 - 1.0) * pParam->BSIM3sqrtPhi
                     + (pParam->BSIM3kt1 + pParam->BSIM3kt1l / pParam->BSIM3leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM3type * pParam->BSIM3vth0
                       - T2 - T3 + pParam->BSIM3k3 * T4 + T5;
                  pParam->BSIM3vfbzb = tmp3 - pParam->BSIM3phi - pParam->BSIM3k1
                                     * pParam->BSIM3sqrtPhi;
                  /* End of vfbzb */
              }

              /* adding delvto  */
              here->BSIM3vth0 = pParam->BSIM3vth0 + here->BSIM3delvto;
              here->BSIM3vfb = pParam->BSIM3vfb + model->BSIM3type * here->BSIM3delvto;
              here->BSIM3vfbzb = pParam->BSIM3vfbzb + model->BSIM3type * here->BSIM3delvto;

              /* low field mobility multiplier */
              here->BSIM3u0temp = pParam->BSIM3u0temp * here->BSIM3mulu0;
              here->BSIM3tconst = here->BSIM3u0temp * pParam->BSIM3elm / (model->BSIM3cox
                                    * pParam->BSIM3weffCV * pParam->BSIM3leffCV * T0);

              /* process source/drain series resistance */
              /* ACM model */

              double drainResistance, sourceResistance;

              if (model->BSIM3acmMod == 0)
              {
                  drainResistance = model->BSIM3sheetResistance
                                                  * here->BSIM3drainSquares;
                  sourceResistance = model->BSIM3sheetResistance
                                               * here->BSIM3sourceSquares;
              }
              else /* ACM > 0 */
              {
                  error = ACM_SourceDrainResistances(
                  model->BSIM3acmMod,
                  model->BSIM3ld,
                  model->BSIM3ldif,
                  model->BSIM3hdif,
                  model->BSIM3wmlt,
                  here->BSIM3w,
                  model->BSIM3xw,
                  model->BSIM3sheetResistance,
                  here->BSIM3drainSquaresGiven,
                  model->BSIM3rd,
                  model->BSIM3rdc,
                  here->BSIM3drainSquares,
                  here->BSIM3sourceSquaresGiven,
                  model->BSIM3rs,
                  model->BSIM3rsc,
                  here->BSIM3sourceSquares,
                  &drainResistance,
                  &sourceResistance
                  );
                  if (error)
                      return(error);
              }
              if (drainResistance > 0.0)
                  here->BSIM3drainConductance = 1.0 / drainResistance;
              else
                  here->BSIM3drainConductance = 0.0;

              if (sourceResistance > 0.0)
                  here->BSIM3sourceConductance = 1.0 / sourceResistance;
              else
                  here->BSIM3sourceConductance = 0.0;

              here->BSIM3cgso = pParam->BSIM3cgso;
              here->BSIM3cgdo = pParam->BSIM3cgdo;

              Nvtm = model->BSIM3vtm * model->BSIM3jctEmissionCoeff;
              if ((here->BSIM3sourceArea <= 0.0) &&
                  (here->BSIM3sourcePerimeter <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM3sourceArea
                                   * model->BSIM3jctTempSatCurDensity
                                   + here->BSIM3sourcePerimeter
                                   * model->BSIM3jctSidewallTempSatCurDensity;
              }
              if ((SourceSatCurrent > 0.0) && (model->BSIM3ijth > 0.0))
              {   here->BSIM3vjsm = Nvtm * log(model->BSIM3ijth
                                  / SourceSatCurrent + 1.0);
                  here->BSIM3IsEvjsm = SourceSatCurrent * exp(here->BSIM3vjsm
                                     / Nvtm);
              }

              if ((here->BSIM3drainArea <= 0.0) &&
                  (here->BSIM3drainPerimeter <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM3drainArea
                                  * model->BSIM3jctTempSatCurDensity
                                  + here->BSIM3drainPerimeter
                                  * model->BSIM3jctSidewallTempSatCurDensity;
              }
              if ((DrainSatCurrent > 0.0) && (model->BSIM3ijth > 0.0))
              {   here->BSIM3vjdm = Nvtm * log(model->BSIM3ijth
                                  / DrainSatCurrent + 1.0);
                  here->BSIM3IsEvjdm = DrainSatCurrent * exp(here->BSIM3vjdm
                                     / Nvtm);
              }
         }
    }
    return(OK);
}

