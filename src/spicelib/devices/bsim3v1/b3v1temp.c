/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1ld.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Modified by Paolo Nenzi 2002
 **********/

/*
 * Release Notes:
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
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
BSIM3v1temp(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1model *model = (BSIM3v1model*) inModel;
BSIM3v1instance *here;
struct bsim3v1SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
int Size_Not_Found;

    /*  loop through all the BSIM3v1 device models */
    for (; model != NULL; model = BSIM3v1nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3v1bulkJctPotential < 0.1)
             model->BSIM3v1bulkJctPotential = 0.1;
         if (model->BSIM3v1sidewallJctPotential < 0.1)
             model->BSIM3v1sidewallJctPotential = 0.1;
         if (model->BSIM3v1GatesidewallJctPotential < 0.1)
             model->BSIM3v1GatesidewallJctPotential = 0.1;

         struct bsim3v1SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim3v1SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM3v1tnom;
         TRatio = Temp / Tnom;

         model->BSIM3v1vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3v1factor1 = sqrt(EPSSI / EPSOX * model->BSIM3v1tox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3v1vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3v1vtm + model->BSIM3v1jctTempExponent
                * log(Temp / Tnom);
             T1 = exp(T0 / model->BSIM3v1jctEmissionCoeff);
             model->BSIM3v1jctTempSatCurDensity = model->BSIM3v1jctSatCurDensity
                                              * T1;
             model->BSIM3v1jctSidewallTempSatCurDensity
                         = model->BSIM3v1jctSidewallSatCurDensity * T1;
         }
         else
         {   model->BSIM3v1jctTempSatCurDensity = model->BSIM3v1jctSatCurDensity;
             model->BSIM3v1jctSidewallTempSatCurDensity
                        = model->BSIM3v1jctSidewallSatCurDensity;
         }

         if (model->BSIM3v1jctTempSatCurDensity < 0.0)
             model->BSIM3v1jctTempSatCurDensity = 0.0;
         if (model->BSIM3v1jctSidewallTempSatCurDensity < 0.0)
             model->BSIM3v1jctSidewallTempSatCurDensity = 0.0;

         /* loop through all the instances of the model */
         /* MCJ: Length and Width not initialized */
         for (here = BSIM3v1instances(model); here != NULL;
              here = BSIM3v1nextInstance(here))
         {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM3v1l == pSizeDependParamKnot->Length)
                      && (here->BSIM3v1w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim3v1SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->BSIM3v1l;
                  Wdrn = here->BSIM3v1w;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;

                  T0 = pow(Ldrn, model->BSIM3v1Lln);
                  T1 = pow(Wdrn, model->BSIM3v1Lwn);
                  tmp1 = model->BSIM3v1Ll / T0 + model->BSIM3v1Lw / T1
                       + model->BSIM3v1Lwl / (T0 * T1);
                  pParam->BSIM3v1dl = model->BSIM3v1Lint + tmp1;
                  pParam->BSIM3v1dlc = model->BSIM3v1dlc + tmp1;

                  T2 = pow(Ldrn, model->BSIM3v1Wln);
                  T3 = pow(Wdrn, model->BSIM3v1Wwn);
                  tmp2 = model->BSIM3v1Wl / T2 + model->BSIM3v1Ww / T3
                       + model->BSIM3v1Wwl / (T2 * T3);
                  pParam->BSIM3v1dw = model->BSIM3v1Wint + tmp2;
                  pParam->BSIM3v1dwc = model->BSIM3v1dwc + tmp2;

                  pParam->BSIM3v1leff = here->BSIM3v1l - 2.0 * pParam->BSIM3v1dl;
                  if (pParam->BSIM3v1leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v1: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM3v1modName, here->BSIM3v1name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1weff = here->BSIM3v1w - 2.0 * pParam->BSIM3v1dw;
                  if (pParam->BSIM3v1weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v1: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM3v1modName, here->BSIM3v1name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1leffCV = here->BSIM3v1l - 2.0 * pParam->BSIM3v1dlc;
                  if (pParam->BSIM3v1leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v1: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM3v1modName, here->BSIM3v1name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1weffCV = here->BSIM3v1w - 2.0 * pParam->BSIM3v1dwc;
                  if (pParam->BSIM3v1weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v1: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM3v1modName, here->BSIM3v1name);
                      return(E_BADPARM);
                  }


                  if (model->BSIM3v1binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM3v1leff;
                      Inv_W = 1.0e-6 / pParam->BSIM3v1weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM3v1leff
                             * pParam->BSIM3v1weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM3v1leff;
                      Inv_W = 1.0 / pParam->BSIM3v1weff;
                      Inv_LW = 1.0 / (pParam->BSIM3v1leff
                             * pParam->BSIM3v1weff);
                  }
                  pParam->BSIM3v1cdsc = model->BSIM3v1cdsc
                                    + model->BSIM3v1lcdsc * Inv_L
                                    + model->BSIM3v1wcdsc * Inv_W
                                    + model->BSIM3v1pcdsc * Inv_LW;
                  pParam->BSIM3v1cdscb = model->BSIM3v1cdscb
                                     + model->BSIM3v1lcdscb * Inv_L
                                     + model->BSIM3v1wcdscb * Inv_W
                                     + model->BSIM3v1pcdscb * Inv_LW;

                      pParam->BSIM3v1cdscd = model->BSIM3v1cdscd
                                     + model->BSIM3v1lcdscd * Inv_L
                                     + model->BSIM3v1wcdscd * Inv_W
                                     + model->BSIM3v1pcdscd * Inv_LW;

                  pParam->BSIM3v1cit = model->BSIM3v1cit
                                   + model->BSIM3v1lcit * Inv_L
                                   + model->BSIM3v1wcit * Inv_W
                                   + model->BSIM3v1pcit * Inv_LW;
                  pParam->BSIM3v1nfactor = model->BSIM3v1nfactor
                                       + model->BSIM3v1lnfactor * Inv_L
                                       + model->BSIM3v1wnfactor * Inv_W
                                       + model->BSIM3v1pnfactor * Inv_LW;
                  pParam->BSIM3v1xj = model->BSIM3v1xj
                                  + model->BSIM3v1lxj * Inv_L
                                  + model->BSIM3v1wxj * Inv_W
                                  + model->BSIM3v1pxj * Inv_LW;
                  pParam->BSIM3v1vsat = model->BSIM3v1vsat
                                    + model->BSIM3v1lvsat * Inv_L
                                    + model->BSIM3v1wvsat * Inv_W
                                    + model->BSIM3v1pvsat * Inv_LW;
                  pParam->BSIM3v1at = model->BSIM3v1at
                                  + model->BSIM3v1lat * Inv_L
                                  + model->BSIM3v1wat * Inv_W
                                  + model->BSIM3v1pat * Inv_LW;
                  pParam->BSIM3v1a0 = model->BSIM3v1a0
                                  + model->BSIM3v1la0 * Inv_L
                                  + model->BSIM3v1wa0 * Inv_W
                                  + model->BSIM3v1pa0 * Inv_LW;

                  pParam->BSIM3v1ags = model->BSIM3v1ags
                                  + model->BSIM3v1lags * Inv_L
                                  + model->BSIM3v1wags * Inv_W
                                  + model->BSIM3v1pags * Inv_LW;

                  pParam->BSIM3v1a1 = model->BSIM3v1a1
                                  + model->BSIM3v1la1 * Inv_L
                                  + model->BSIM3v1wa1 * Inv_W
                                  + model->BSIM3v1pa1 * Inv_LW;
                  pParam->BSIM3v1a2 = model->BSIM3v1a2
                                  + model->BSIM3v1la2 * Inv_L
                                  + model->BSIM3v1wa2 * Inv_W
                                  + model->BSIM3v1pa2 * Inv_LW;
                  pParam->BSIM3v1keta = model->BSIM3v1keta
                                    + model->BSIM3v1lketa * Inv_L
                                    + model->BSIM3v1wketa * Inv_W
                                    + model->BSIM3v1pketa * Inv_LW;
                  pParam->BSIM3v1nsub = model->BSIM3v1nsub
                                    + model->BSIM3v1lnsub * Inv_L
                                    + model->BSIM3v1wnsub * Inv_W
                                    + model->BSIM3v1pnsub * Inv_LW;
                  pParam->BSIM3v1npeak = model->BSIM3v1npeak
                                     + model->BSIM3v1lnpeak * Inv_L
                                     + model->BSIM3v1wnpeak * Inv_W
                                     + model->BSIM3v1pnpeak * Inv_LW;
                  pParam->BSIM3v1ngate = model->BSIM3v1ngate
                                     + model->BSIM3v1lngate * Inv_L
                                     + model->BSIM3v1wngate * Inv_W
                                     + model->BSIM3v1pngate * Inv_LW;
                  pParam->BSIM3v1gamma1 = model->BSIM3v1gamma1
                                      + model->BSIM3v1lgamma1 * Inv_L
                                      + model->BSIM3v1wgamma1 * Inv_W
                                      + model->BSIM3v1pgamma1 * Inv_LW;
                  pParam->BSIM3v1gamma2 = model->BSIM3v1gamma2
                                      + model->BSIM3v1lgamma2 * Inv_L
                                      + model->BSIM3v1wgamma2 * Inv_W
                                      + model->BSIM3v1pgamma2 * Inv_LW;
                  pParam->BSIM3v1vbx = model->BSIM3v1vbx
                                   + model->BSIM3v1lvbx * Inv_L
                                   + model->BSIM3v1wvbx * Inv_W
                                   + model->BSIM3v1pvbx * Inv_LW;
                  pParam->BSIM3v1vbm = model->BSIM3v1vbm
                                   + model->BSIM3v1lvbm * Inv_L
                                   + model->BSIM3v1wvbm * Inv_W
                                   + model->BSIM3v1pvbm * Inv_LW;
                  pParam->BSIM3v1xt = model->BSIM3v1xt
                                   + model->BSIM3v1lxt * Inv_L
                                   + model->BSIM3v1wxt * Inv_W
                                   + model->BSIM3v1pxt * Inv_LW;
                  pParam->BSIM3v1k1 = model->BSIM3v1k1
                                  + model->BSIM3v1lk1 * Inv_L
                                  + model->BSIM3v1wk1 * Inv_W
                                  + model->BSIM3v1pk1 * Inv_LW;
                  pParam->BSIM3v1kt1 = model->BSIM3v1kt1
                                   + model->BSIM3v1lkt1 * Inv_L
                                   + model->BSIM3v1wkt1 * Inv_W
                                   + model->BSIM3v1pkt1 * Inv_LW;
                  pParam->BSIM3v1kt1l = model->BSIM3v1kt1l
                                    + model->BSIM3v1lkt1l * Inv_L
                                    + model->BSIM3v1wkt1l * Inv_W
                                    + model->BSIM3v1pkt1l * Inv_LW;
                  pParam->BSIM3v1k2 = model->BSIM3v1k2
                                  + model->BSIM3v1lk2 * Inv_L
                                  + model->BSIM3v1wk2 * Inv_W
                                  + model->BSIM3v1pk2 * Inv_LW;
                  pParam->BSIM3v1kt2 = model->BSIM3v1kt2
                                   + model->BSIM3v1lkt2 * Inv_L
                                   + model->BSIM3v1wkt2 * Inv_W
                                   + model->BSIM3v1pkt2 * Inv_LW;
                  pParam->BSIM3v1k3 = model->BSIM3v1k3
                                  + model->BSIM3v1lk3 * Inv_L
                                  + model->BSIM3v1wk3 * Inv_W
                                  + model->BSIM3v1pk3 * Inv_LW;
                  pParam->BSIM3v1k3b = model->BSIM3v1k3b
                                   + model->BSIM3v1lk3b * Inv_L
                                   + model->BSIM3v1wk3b * Inv_W
                                   + model->BSIM3v1pk3b * Inv_LW;
                  pParam->BSIM3v1w0 = model->BSIM3v1w0
                                  + model->BSIM3v1lw0 * Inv_L
                                  + model->BSIM3v1ww0 * Inv_W
                                  + model->BSIM3v1pw0 * Inv_LW;
                  pParam->BSIM3v1nlx = model->BSIM3v1nlx
                                   + model->BSIM3v1lnlx * Inv_L
                                   + model->BSIM3v1wnlx * Inv_W
                                   + model->BSIM3v1pnlx * Inv_LW;
                  pParam->BSIM3v1dvt0 = model->BSIM3v1dvt0
                                    + model->BSIM3v1ldvt0 * Inv_L
                                    + model->BSIM3v1wdvt0 * Inv_W
                                    + model->BSIM3v1pdvt0 * Inv_LW;
                  pParam->BSIM3v1dvt1 = model->BSIM3v1dvt1
                                    + model->BSIM3v1ldvt1 * Inv_L
                                    + model->BSIM3v1wdvt1 * Inv_W
                                    + model->BSIM3v1pdvt1 * Inv_LW;
                  pParam->BSIM3v1dvt2 = model->BSIM3v1dvt2
                                    + model->BSIM3v1ldvt2 * Inv_L
                                    + model->BSIM3v1wdvt2 * Inv_W
                                    + model->BSIM3v1pdvt2 * Inv_LW;
                  pParam->BSIM3v1dvt0w = model->BSIM3v1dvt0w
                                    + model->BSIM3v1ldvt0w * Inv_L
                                    + model->BSIM3v1wdvt0w * Inv_W
                                    + model->BSIM3v1pdvt0w * Inv_LW;
                  pParam->BSIM3v1dvt1w = model->BSIM3v1dvt1w
                                    + model->BSIM3v1ldvt1w * Inv_L
                                    + model->BSIM3v1wdvt1w * Inv_W
                                    + model->BSIM3v1pdvt1w * Inv_LW;
                  pParam->BSIM3v1dvt2w = model->BSIM3v1dvt2w
                                    + model->BSIM3v1ldvt2w * Inv_L
                                    + model->BSIM3v1wdvt2w * Inv_W
                                    + model->BSIM3v1pdvt2w * Inv_LW;
                  pParam->BSIM3v1drout = model->BSIM3v1drout
                                     + model->BSIM3v1ldrout * Inv_L
                                     + model->BSIM3v1wdrout * Inv_W
                                     + model->BSIM3v1pdrout * Inv_LW;
                  pParam->BSIM3v1dsub = model->BSIM3v1dsub
                                    + model->BSIM3v1ldsub * Inv_L
                                    + model->BSIM3v1wdsub * Inv_W
                                    + model->BSIM3v1pdsub * Inv_LW;
                  pParam->BSIM3v1vth0 = model->BSIM3v1vth0
                                    + model->BSIM3v1lvth0 * Inv_L
                                    + model->BSIM3v1wvth0 * Inv_W
                                    + model->BSIM3v1pvth0 * Inv_LW;
                  pParam->BSIM3v1ua = model->BSIM3v1ua
                                  + model->BSIM3v1lua * Inv_L
                                  + model->BSIM3v1wua * Inv_W
                                  + model->BSIM3v1pua * Inv_LW;
                  pParam->BSIM3v1ua1 = model->BSIM3v1ua1
                                   + model->BSIM3v1lua1 * Inv_L
                                   + model->BSIM3v1wua1 * Inv_W
                                   + model->BSIM3v1pua1 * Inv_LW;
                  pParam->BSIM3v1ub = model->BSIM3v1ub
                                  + model->BSIM3v1lub * Inv_L
                                  + model->BSIM3v1wub * Inv_W
                                  + model->BSIM3v1pub * Inv_LW;
                  pParam->BSIM3v1ub1 = model->BSIM3v1ub1
                                   + model->BSIM3v1lub1 * Inv_L
                                   + model->BSIM3v1wub1 * Inv_W
                                   + model->BSIM3v1pub1 * Inv_LW;
                  pParam->BSIM3v1uc = model->BSIM3v1uc
                                  + model->BSIM3v1luc * Inv_L
                                  + model->BSIM3v1wuc * Inv_W
                                  + model->BSIM3v1puc * Inv_LW;
                  pParam->BSIM3v1uc1 = model->BSIM3v1uc1
                                   + model->BSIM3v1luc1 * Inv_L
                                   + model->BSIM3v1wuc1 * Inv_W
                                   + model->BSIM3v1puc1 * Inv_LW;
                  pParam->BSIM3v1u0 = model->BSIM3v1u0
                                  + model->BSIM3v1lu0 * Inv_L
                                  + model->BSIM3v1wu0 * Inv_W
                                  + model->BSIM3v1pu0 * Inv_LW;
                  pParam->BSIM3v1ute = model->BSIM3v1ute
                                   + model->BSIM3v1lute * Inv_L
                                   + model->BSIM3v1wute * Inv_W
                                   + model->BSIM3v1pute * Inv_LW;
                  pParam->BSIM3v1voff = model->BSIM3v1voff
                                    + model->BSIM3v1lvoff * Inv_L
                                    + model->BSIM3v1wvoff * Inv_W
                                    + model->BSIM3v1pvoff * Inv_LW;
                  pParam->BSIM3v1delta = model->BSIM3v1delta
                                     + model->BSIM3v1ldelta * Inv_L
                                     + model->BSIM3v1wdelta * Inv_W
                                     + model->BSIM3v1pdelta * Inv_LW;
                  pParam->BSIM3v1rdsw = model->BSIM3v1rdsw
                                    + model->BSIM3v1lrdsw * Inv_L
                                    + model->BSIM3v1wrdsw * Inv_W
                                    + model->BSIM3v1prdsw * Inv_LW;
                  pParam->BSIM3v1prwg = model->BSIM3v1prwg
                                    + model->BSIM3v1lprwg * Inv_L
                                    + model->BSIM3v1wprwg * Inv_W
                                    + model->BSIM3v1pprwg * Inv_LW;
                  pParam->BSIM3v1prwb = model->BSIM3v1prwb
                                    + model->BSIM3v1lprwb * Inv_L
                                    + model->BSIM3v1wprwb * Inv_W
                                    + model->BSIM3v1pprwb * Inv_LW;
                  pParam->BSIM3v1prt = model->BSIM3v1prt
                                    + model->BSIM3v1lprt * Inv_L
                                    + model->BSIM3v1wprt * Inv_W
                                    + model->BSIM3v1pprt * Inv_LW;
                  pParam->BSIM3v1eta0 = model->BSIM3v1eta0
                                    + model->BSIM3v1leta0 * Inv_L
                                    + model->BSIM3v1weta0 * Inv_W
                                    + model->BSIM3v1peta0 * Inv_LW;
                  pParam->BSIM3v1etab = model->BSIM3v1etab
                                    + model->BSIM3v1letab * Inv_L
                                    + model->BSIM3v1wetab * Inv_W
                                    + model->BSIM3v1petab * Inv_LW;
                  pParam->BSIM3v1pclm = model->BSIM3v1pclm
                                    + model->BSIM3v1lpclm * Inv_L
                                    + model->BSIM3v1wpclm * Inv_W
                                    + model->BSIM3v1ppclm * Inv_LW;
                  pParam->BSIM3v1pdibl1 = model->BSIM3v1pdibl1
                                      + model->BSIM3v1lpdibl1 * Inv_L
                                      + model->BSIM3v1wpdibl1 * Inv_W
                                      + model->BSIM3v1ppdibl1 * Inv_LW;
                  pParam->BSIM3v1pdibl2 = model->BSIM3v1pdibl2
                                      + model->BSIM3v1lpdibl2 * Inv_L
                                      + model->BSIM3v1wpdibl2 * Inv_W
                                      + model->BSIM3v1ppdibl2 * Inv_LW;
                  pParam->BSIM3v1pdiblb = model->BSIM3v1pdiblb
                                      + model->BSIM3v1lpdiblb * Inv_L
                                      + model->BSIM3v1wpdiblb * Inv_W
                                      + model->BSIM3v1ppdiblb * Inv_LW;
                  pParam->BSIM3v1pscbe1 = model->BSIM3v1pscbe1
                                      + model->BSIM3v1lpscbe1 * Inv_L
                                      + model->BSIM3v1wpscbe1 * Inv_W
                                      + model->BSIM3v1ppscbe1 * Inv_LW;
                  pParam->BSIM3v1pscbe2 = model->BSIM3v1pscbe2
                                      + model->BSIM3v1lpscbe2 * Inv_L
                                      + model->BSIM3v1wpscbe2 * Inv_W
                                      + model->BSIM3v1ppscbe2 * Inv_LW;
                  pParam->BSIM3v1pvag = model->BSIM3v1pvag
                                    + model->BSIM3v1lpvag * Inv_L
                                    + model->BSIM3v1wpvag * Inv_W
                                    + model->BSIM3v1ppvag * Inv_LW;
                  pParam->BSIM3v1wr = model->BSIM3v1wr
                                  + model->BSIM3v1lwr * Inv_L
                                  + model->BSIM3v1wwr * Inv_W
                                  + model->BSIM3v1pwr * Inv_LW;
                  pParam->BSIM3v1dwg = model->BSIM3v1dwg
                                   + model->BSIM3v1ldwg * Inv_L
                                   + model->BSIM3v1wdwg * Inv_W
                                   + model->BSIM3v1pdwg * Inv_LW;
                  pParam->BSIM3v1dwb = model->BSIM3v1dwb
                                   + model->BSIM3v1ldwb * Inv_L
                                   + model->BSIM3v1wdwb * Inv_W
                                   + model->BSIM3v1pdwb * Inv_LW;
                  pParam->BSIM3v1b0 = model->BSIM3v1b0
                                  + model->BSIM3v1lb0 * Inv_L
                                  + model->BSIM3v1wb0 * Inv_W
                                  + model->BSIM3v1pb0 * Inv_LW;
                  pParam->BSIM3v1b1 = model->BSIM3v1b1
                                  + model->BSIM3v1lb1 * Inv_L
                                  + model->BSIM3v1wb1 * Inv_W
                                  + model->BSIM3v1pb1 * Inv_LW;
                  pParam->BSIM3v1alpha0 = model->BSIM3v1alpha0
                                      + model->BSIM3v1lalpha0 * Inv_L
                                      + model->BSIM3v1walpha0 * Inv_W
                                      + model->BSIM3v1palpha0 * Inv_LW;
                  pParam->BSIM3v1beta0 = model->BSIM3v1beta0
                                     + model->BSIM3v1lbeta0 * Inv_L
                                     + model->BSIM3v1wbeta0 * Inv_W
                                     + model->BSIM3v1pbeta0 * Inv_LW;
                  /* CV model */
                  pParam->BSIM3v1elm = model->BSIM3v1elm
                                  + model->BSIM3v1lelm * Inv_L
                                  + model->BSIM3v1welm * Inv_W
                                  + model->BSIM3v1pelm * Inv_LW;
                  pParam->BSIM3v1cgsl = model->BSIM3v1cgsl
                                    + model->BSIM3v1lcgsl * Inv_L
                                    + model->BSIM3v1wcgsl * Inv_W
                                    + model->BSIM3v1pcgsl * Inv_LW;
                  pParam->BSIM3v1cgdl = model->BSIM3v1cgdl
                                    + model->BSIM3v1lcgdl * Inv_L
                                    + model->BSIM3v1wcgdl * Inv_W
                                    + model->BSIM3v1pcgdl * Inv_LW;
                  pParam->BSIM3v1ckappa = model->BSIM3v1ckappa
                                      + model->BSIM3v1lckappa * Inv_L
                                      + model->BSIM3v1wckappa * Inv_W
                                      + model->BSIM3v1pckappa * Inv_LW;
                  pParam->BSIM3v1cf = model->BSIM3v1cf
                                  + model->BSIM3v1lcf * Inv_L
                                  + model->BSIM3v1wcf * Inv_W
                                  + model->BSIM3v1pcf * Inv_LW;
                  pParam->BSIM3v1clc = model->BSIM3v1clc
                                   + model->BSIM3v1lclc * Inv_L
                                   + model->BSIM3v1wclc * Inv_W
                                   + model->BSIM3v1pclc * Inv_LW;
                  pParam->BSIM3v1cle = model->BSIM3v1cle
                                   + model->BSIM3v1lcle * Inv_L
                                   + model->BSIM3v1wcle * Inv_W
                                   + model->BSIM3v1pcle * Inv_LW;
                  pParam->BSIM3v1vfbcv = model->BSIM3v1vfbcv
                                  + model->BSIM3v1lvfbcv * Inv_L
                                  + model->BSIM3v1wvfbcv * Inv_W
                                  + model->BSIM3v1pvfbcv * Inv_LW;
                  pParam->BSIM3v1abulkCVfactor = 1.0 + pow((pParam->BSIM3v1clc
                                             / pParam->BSIM3v1leff),
                                             pParam->BSIM3v1cle);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM3v1ua = pParam->BSIM3v1ua + pParam->BSIM3v1ua1 * T0;
                  pParam->BSIM3v1ub = pParam->BSIM3v1ub + pParam->BSIM3v1ub1 * T0;
                  pParam->BSIM3v1uc = pParam->BSIM3v1uc + pParam->BSIM3v1uc1 * T0;
                  if (pParam->BSIM3v1u0 > 1.0)
                      pParam->BSIM3v1u0 = pParam->BSIM3v1u0 / 1.0e4;

                  pParam->BSIM3v1u0temp = pParam->BSIM3v1u0
                                      * pow(TRatio, pParam->BSIM3v1ute);
                  pParam->BSIM3v1vsattemp = pParam->BSIM3v1vsat - pParam->BSIM3v1at
                                        * T0;
                  pParam->BSIM3v1rds0 = (pParam->BSIM3v1rdsw + pParam->BSIM3v1prt * T0)
                                    / pow(pParam->BSIM3v1weff * 1E6, pParam->BSIM3v1wr);

                  if (BSIM3v1checkModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during BSIM3V3.1 parameter checking for %s in model %s", model->BSIM3v1modName, here->BSIM3v1name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1cgdo = (model->BSIM3v1cgdo + pParam->BSIM3v1cf)
                                    * pParam->BSIM3v1weffCV;
                  pParam->BSIM3v1cgso = (model->BSIM3v1cgso + pParam->BSIM3v1cf)
                                    * pParam->BSIM3v1weffCV;
                  pParam->BSIM3v1cgbo = model->BSIM3v1cgbo * pParam->BSIM3v1leffCV;

                  if (!model->BSIM3v1npeakGiven && model->BSIM3v1gamma1Given)
                  {   T0 = pParam->BSIM3v1gamma1 * model->BSIM3v1cox;
                      pParam->BSIM3v1npeak = 3.021E22 * T0 * T0;
                  }

                  pParam->BSIM3v1phi = 2.0 * Vtm0
                                   * log(pParam->BSIM3v1npeak / ni);

                  pParam->BSIM3v1sqrtPhi = sqrt(pParam->BSIM3v1phi);
                  pParam->BSIM3v1phis3 = pParam->BSIM3v1sqrtPhi * pParam->BSIM3v1phi;

                  pParam->BSIM3v1Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM3v1npeak * 1.0e6))
                                     * pParam->BSIM3v1sqrtPhi;
                  pParam->BSIM3v1sqrtXdep0 = sqrt(pParam->BSIM3v1Xdep0);
                  pParam->BSIM3v1litl = sqrt(3.0 * pParam->BSIM3v1xj
                                    * model->BSIM3v1tox);
                  pParam->BSIM3v1vbi = Vtm0 * log(1.0e20
                                   * pParam->BSIM3v1npeak / (ni * ni));
                  pParam->BSIM3v1cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM3v1npeak * 1.0e6 / 2.0
                                     / pParam->BSIM3v1phi);

                  if (model->BSIM3v1k1Given || model->BSIM3v1k2Given)
                  {   if (!model->BSIM3v1k1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3v1k1 = 0.53;
                      }
                      if (!model->BSIM3v1k2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3v1k2 = -0.0186;
                      }
                      if (model->BSIM3v1nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1vbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->BSIM3v1vbxGiven)
                          pParam->BSIM3v1vbx = pParam->BSIM3v1phi - 7.7348e-4
                                           * pParam->BSIM3v1npeak
                                           * pParam->BSIM3v1xt * pParam->BSIM3v1xt;
                      if (pParam->BSIM3v1vbx > 0.0)
                          pParam->BSIM3v1vbx = -pParam->BSIM3v1vbx;
                      if (pParam->BSIM3v1vbm > 0.0)
                          pParam->BSIM3v1vbm = -pParam->BSIM3v1vbm;

                      if (!model->BSIM3v1gamma1Given)
                          pParam->BSIM3v1gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM3v1npeak)
                                              / model->BSIM3v1cox;
                      if (!model->BSIM3v1gamma2Given)
                          pParam->BSIM3v1gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM3v1nsub)
                                              / model->BSIM3v1cox;

                      T0 = pParam->BSIM3v1gamma1 - pParam->BSIM3v1gamma2;
                      T1 = sqrt(pParam->BSIM3v1phi - pParam->BSIM3v1vbx)
                         - pParam->BSIM3v1sqrtPhi;
                      T2 = sqrt(pParam->BSIM3v1phi * (pParam->BSIM3v1phi
                         - pParam->BSIM3v1vbm)) - pParam->BSIM3v1phi;
                      pParam->BSIM3v1k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3v1vbm);
                      pParam->BSIM3v1k1 = pParam->BSIM3v1gamma2 - 2.0
                                      * pParam->BSIM3v1k2 * sqrt(pParam->BSIM3v1phi
                                      - pParam->BSIM3v1vbm);
                  }

                  if (pParam->BSIM3v1k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM3v1k1 / pParam->BSIM3v1k2;
                      pParam->BSIM3v1vbsc = 0.9 * (pParam->BSIM3v1phi - T0 * T0);
                      if (pParam->BSIM3v1vbsc > -3.0)
                          pParam->BSIM3v1vbsc = -3.0;
                      else if (pParam->BSIM3v1vbsc < -30.0)
                          pParam->BSIM3v1vbsc = -30.0;
                  }
                  else
                  {   pParam->BSIM3v1vbsc = -30.0;
                  }
                  if (pParam->BSIM3v1vbsc > pParam->BSIM3v1vbm)
                      pParam->BSIM3v1vbsc = pParam->BSIM3v1vbm;

                  if (model->BSIM3v1vth0Given)
                  {   pParam->BSIM3v1vfb = model->BSIM3v1type * pParam->BSIM3v1vth0
                                       - pParam->BSIM3v1phi - pParam->BSIM3v1k1
                                       * pParam->BSIM3v1sqrtPhi;
                  }
                  else
                  {   pParam->BSIM3v1vfb = -1.0;
                      pParam->BSIM3v1vth0 = model->BSIM3v1type * (pParam->BSIM3v1vfb
                                        + pParam->BSIM3v1phi + pParam->BSIM3v1k1
                                        * pParam->BSIM3v1sqrtPhi);
                  }
                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3v1tox
                     * pParam->BSIM3v1Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3v1dsub * pParam->BSIM3v1leff / T1);
                  pParam->BSIM3v1theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3v1drout * pParam->BSIM3v1leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3v1thetaRout = pParam->BSIM3v1pdibl1 * T2
                                         + pParam->BSIM3v1pdibl2;
              }

              /* process source/drain series resistance */
              here->BSIM3v1drainConductance = model->BSIM3v1sheetResistance
                                              * here->BSIM3v1drainSquares;
              if (here->BSIM3v1drainConductance > 0.0)
                  here->BSIM3v1drainConductance = 1.0
                                              / here->BSIM3v1drainConductance;
              else
                  here->BSIM3v1drainConductance = 0.0;

              here->BSIM3v1sourceConductance = model->BSIM3v1sheetResistance
                                           * here->BSIM3v1sourceSquares;
              if (here->BSIM3v1sourceConductance > 0.0)
                  here->BSIM3v1sourceConductance = 1.0
                                               / here->BSIM3v1sourceConductance;
              else
                  here->BSIM3v1sourceConductance = 0.0;
              here->BSIM3v1cgso = pParam->BSIM3v1cgso;
              here->BSIM3v1cgdo = pParam->BSIM3v1cgdo;
         }
    }
    return(OK);
}

