/***********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3temp.c
**********/
/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
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
BSIM3v0temp(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v0model *model = (BSIM3v0model*) inModel;
BSIM3v0instance *here;
struct bsim3v0SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp1, tmp2, Eg, ni, T0, T1, T2, T3, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
int Size_Not_Found;

    /*  loop through all the BSIM3v0 device models */
    for (; model != NULL; model = BSIM3v0nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3v0bulkJctPotential < 0.1)
             model->BSIM3v0bulkJctPotential = 0.1;
         if (model->BSIM3v0sidewallJctPotential < 0.1)
             model->BSIM3v0sidewallJctPotential = 0.1;

         struct bsim3v0SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim3v0SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM3v0tnom;
         TRatio = Temp / Tnom;

         /* loop through all the instances of the model */
         for (here = BSIM3v0instances(model); here != NULL;
              here=BSIM3v0nextInstance(here))
          {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM3v0l == pSizeDependParamKnot->Length)
                      && (here->BSIM3v0w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim3v0SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                     Ldrn = here->BSIM3v0l;
                     Wdrn = here->BSIM3v0w;

                  T0 = pow(Ldrn, model->BSIM3v0Lln);
                  T1 = pow(Wdrn, model->BSIM3v0Lwn);
                  tmp1 = model->BSIM3v0Ll / T0 + model->BSIM3v0Lw / T1
                       + model->BSIM3v0Lwl / (T0 * T1);
                  pParam->BSIM3v0dl = model->BSIM3v0Lint + tmp1;
                  pParam->BSIM3v0dlc = model->BSIM3v0dlc + tmp1;

                  T2 = pow(Ldrn, model->BSIM3v0Wln);
                  T3 = pow(Wdrn, model->BSIM3v0Wwn);
                  tmp2 = model->BSIM3v0Wl / T2 + model->BSIM3v0Ww / T3
                       + model->BSIM3v0Wwl / (T2 * T3);
                  pParam->BSIM3v0dw = model->BSIM3v0Wint + tmp2;
                  pParam->BSIM3v0dwc = model->BSIM3v0dwc + tmp2;

                  pParam->BSIM3v0leff = here->BSIM3v0l - 2.0 * pParam->BSIM3v0dl;
                  if (pParam->BSIM3v0leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v0: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM3v0modName, here->BSIM3v0name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v0weff = here->BSIM3v0w - 2.0 * pParam->BSIM3v0dw;
                  if (pParam->BSIM3v0weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v0: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM3v0modName, here->BSIM3v0name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v0leffCV = here->BSIM3v0l - 2.0 * pParam->BSIM3v0dlc;
                  if (pParam->BSIM3v0leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v0: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM3v0modName, here->BSIM3v0name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v0weffCV = here->BSIM3v0w - 2.0 * pParam->BSIM3v0dwc;
                  if (pParam->BSIM3v0weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM3v0: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM3v0modName, here->BSIM3v0name);
                      return(E_BADPARM);
                  }

                  model->BSIM3v0vcrit = CONSTvt0 * log(CONSTvt0
                                    / (CONSTroot2 * 1.0e-14));
                  model->BSIM3v0factor1 = sqrt(EPSSI / EPSOX * model->BSIM3v0tox);


                  if (model->BSIM3v0binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM3v0leff;
                      Inv_W = 1.0e-6 / pParam->BSIM3v0weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM3v0leff
                             * pParam->BSIM3v0weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM3v0leff;
                      Inv_W = 1.0 / pParam->BSIM3v0weff;
                      Inv_LW = 1.0 / (pParam->BSIM3v0leff
                             * pParam->BSIM3v0weff);
                  }
                  pParam->BSIM3v0cdsc = model->BSIM3v0cdsc
                                    + model->BSIM3v0lcdsc * Inv_L
                                    + model->BSIM3v0wcdsc * Inv_W
                                    + model->BSIM3v0pcdsc * Inv_LW;
                  pParam->BSIM3v0cdscb = model->BSIM3v0cdscb
                                     + model->BSIM3v0lcdscb * Inv_L
                                     + model->BSIM3v0wcdscb * Inv_W
                                     + model->BSIM3v0pcdscb * Inv_LW;

                      pParam->BSIM3v0cdscd = model->BSIM3v0cdscd
                                     + model->BSIM3v0lcdscd * Inv_L
                                     + model->BSIM3v0wcdscd * Inv_W
                                     + model->BSIM3v0pcdscd * Inv_LW;

                  pParam->BSIM3v0cit = model->BSIM3v0cit
                                   + model->BSIM3v0lcit * Inv_L
                                   + model->BSIM3v0wcit * Inv_W
                                   + model->BSIM3v0pcit * Inv_LW;
                  pParam->BSIM3v0nfactor = model->BSIM3v0nfactor
                                       + model->BSIM3v0lnfactor * Inv_L
                                       + model->BSIM3v0wnfactor * Inv_W
                                       + model->BSIM3v0pnfactor * Inv_LW;
                  pParam->BSIM3v0xj = model->BSIM3v0xj
                                  + model->BSIM3v0lxj * Inv_L
                                  + model->BSIM3v0wxj * Inv_W
                                  + model->BSIM3v0pxj * Inv_LW;
                  pParam->BSIM3v0vsat = model->BSIM3v0vsat
                                    + model->BSIM3v0lvsat * Inv_L
                                    + model->BSIM3v0wvsat * Inv_W
                                    + model->BSIM3v0pvsat * Inv_LW;
                  pParam->BSIM3v0at = model->BSIM3v0at
                                  + model->BSIM3v0lat * Inv_L
                                  + model->BSIM3v0wat * Inv_W
                                  + model->BSIM3v0pat * Inv_LW;
                  pParam->BSIM3v0a0 = model->BSIM3v0a0
                                  + model->BSIM3v0la0 * Inv_L
                                  + model->BSIM3v0wa0 * Inv_W
                                  + model->BSIM3v0pa0 * Inv_LW;

                  pParam->BSIM3v0ags = model->BSIM3v0ags
                                  + model->BSIM3v0lags * Inv_L
                                  + model->BSIM3v0wags * Inv_W
                                  + model->BSIM3v0pags * Inv_LW;

                  pParam->BSIM3v0a1 = model->BSIM3v0a1
                                  + model->BSIM3v0la1 * Inv_L
                                  + model->BSIM3v0wa1 * Inv_W
                                  + model->BSIM3v0pa1 * Inv_LW;
                  pParam->BSIM3v0a2 = model->BSIM3v0a2
                                  + model->BSIM3v0la2 * Inv_L
                                  + model->BSIM3v0wa2 * Inv_W
                                  + model->BSIM3v0pa2 * Inv_LW;
                  pParam->BSIM3v0keta = model->BSIM3v0keta
                                    + model->BSIM3v0lketa * Inv_L
                                    + model->BSIM3v0wketa * Inv_W
                                    + model->BSIM3v0pketa * Inv_LW;
                  pParam->BSIM3v0nsub = model->BSIM3v0nsub
                                    + model->BSIM3v0lnsub * Inv_L
                                    + model->BSIM3v0wnsub * Inv_W
                                    + model->BSIM3v0pnsub * Inv_LW;
                  pParam->BSIM3v0npeak = model->BSIM3v0npeak
                                     + model->BSIM3v0lnpeak * Inv_L
                                     + model->BSIM3v0wnpeak * Inv_W
                                     + model->BSIM3v0pnpeak * Inv_LW;
                  pParam->BSIM3v0ngate = model->BSIM3v0ngate
                                     + model->BSIM3v0lngate * Inv_L
                                     + model->BSIM3v0wngate * Inv_W
                                     + model->BSIM3v0pngate * Inv_LW;
                  pParam->BSIM3v0gamma1 = model->BSIM3v0gamma1
                                      + model->BSIM3v0lgamma1 * Inv_L
                                      + model->BSIM3v0wgamma1 * Inv_W
                                      + model->BSIM3v0pgamma1 * Inv_LW;
                  pParam->BSIM3v0gamma2 = model->BSIM3v0gamma2
                                      + model->BSIM3v0lgamma2 * Inv_L
                                      + model->BSIM3v0wgamma2 * Inv_W
                                      + model->BSIM3v0pgamma2 * Inv_LW;
                  pParam->BSIM3v0vbx = model->BSIM3v0vbx
                                   + model->BSIM3v0lvbx * Inv_L
                                   + model->BSIM3v0wvbx * Inv_W
                                   + model->BSIM3v0pvbx * Inv_LW;
                  pParam->BSIM3v0vbm = model->BSIM3v0vbm
                                   + model->BSIM3v0lvbm * Inv_L
                                   + model->BSIM3v0wvbm * Inv_W
                                   + model->BSIM3v0pvbm * Inv_LW;
                  pParam->BSIM3v0xt = model->BSIM3v0xt
                                   + model->BSIM3v0lxt * Inv_L
                                   + model->BSIM3v0wxt * Inv_W
                                   + model->BSIM3v0pxt * Inv_LW;
                  pParam->BSIM3v0k1 = model->BSIM3v0k1
                                  + model->BSIM3v0lk1 * Inv_L
                                  + model->BSIM3v0wk1 * Inv_W
                                  + model->BSIM3v0pk1 * Inv_LW;
                  pParam->BSIM3v0kt1 = model->BSIM3v0kt1
                                   + model->BSIM3v0lkt1 * Inv_L
                                   + model->BSIM3v0wkt1 * Inv_W
                                   + model->BSIM3v0pkt1 * Inv_LW;
                  pParam->BSIM3v0kt1l = model->BSIM3v0kt1l
                                    + model->BSIM3v0lkt1l * Inv_L
                                    + model->BSIM3v0wkt1l * Inv_W
                                    + model->BSIM3v0pkt1l * Inv_LW;
                  pParam->BSIM3v0k2 = model->BSIM3v0k2
                                  + model->BSIM3v0lk2 * Inv_L
                                  + model->BSIM3v0wk2 * Inv_W
                                  + model->BSIM3v0pk2 * Inv_LW;
                  pParam->BSIM3v0kt2 = model->BSIM3v0kt2
                                   + model->BSIM3v0lkt2 * Inv_L
                                   + model->BSIM3v0wkt2 * Inv_W
                                   + model->BSIM3v0pkt2 * Inv_LW;
                  pParam->BSIM3v0k3 = model->BSIM3v0k3
                                  + model->BSIM3v0lk3 * Inv_L
                                  + model->BSIM3v0wk3 * Inv_W
                                  + model->BSIM3v0pk3 * Inv_LW;
                  pParam->BSIM3v0k3b = model->BSIM3v0k3b
                                   + model->BSIM3v0lk3b * Inv_L
                                   + model->BSIM3v0wk3b * Inv_W
                                   + model->BSIM3v0pk3b * Inv_LW;
                  pParam->BSIM3v0w0 = model->BSIM3v0w0
                                  + model->BSIM3v0lw0 * Inv_L
                                  + model->BSIM3v0ww0 * Inv_W
                                  + model->BSIM3v0pw0 * Inv_LW;
                  pParam->BSIM3v0nlx = model->BSIM3v0nlx
                                   + model->BSIM3v0lnlx * Inv_L
                                   + model->BSIM3v0wnlx * Inv_W
                                   + model->BSIM3v0pnlx * Inv_LW;
                  pParam->BSIM3v0dvt0 = model->BSIM3v0dvt0
                                    + model->BSIM3v0ldvt0 * Inv_L
                                    + model->BSIM3v0wdvt0 * Inv_W
                                    + model->BSIM3v0pdvt0 * Inv_LW;
                  pParam->BSIM3v0dvt1 = model->BSIM3v0dvt1
                                    + model->BSIM3v0ldvt1 * Inv_L
                                    + model->BSIM3v0wdvt1 * Inv_W
                                    + model->BSIM3v0pdvt1 * Inv_LW;
                  pParam->BSIM3v0dvt2 = model->BSIM3v0dvt2
                                    + model->BSIM3v0ldvt2 * Inv_L
                                    + model->BSIM3v0wdvt2 * Inv_W
                                    + model->BSIM3v0pdvt2 * Inv_LW;
                  pParam->BSIM3v0dvt0w = model->BSIM3v0dvt0w
                                    + model->BSIM3v0ldvt0w * Inv_L
                                    + model->BSIM3v0wdvt0w * Inv_W
                                    + model->BSIM3v0pdvt0w * Inv_LW;
                  pParam->BSIM3v0dvt1w = model->BSIM3v0dvt1w
                                    + model->BSIM3v0ldvt1w * Inv_L
                                    + model->BSIM3v0wdvt1w * Inv_W
                                    + model->BSIM3v0pdvt1w * Inv_LW;
                  pParam->BSIM3v0dvt2w = model->BSIM3v0dvt2w
                                    + model->BSIM3v0ldvt2w * Inv_L
                                    + model->BSIM3v0wdvt2w * Inv_W
                                    + model->BSIM3v0pdvt2w * Inv_LW;
                  pParam->BSIM3v0drout = model->BSIM3v0drout
                                     + model->BSIM3v0ldrout * Inv_L
                                     + model->BSIM3v0wdrout * Inv_W
                                     + model->BSIM3v0pdrout * Inv_LW;
                  pParam->BSIM3v0dsub = model->BSIM3v0dsub
                                    + model->BSIM3v0ldsub * Inv_L
                                    + model->BSIM3v0wdsub * Inv_W
                                    + model->BSIM3v0pdsub * Inv_LW;
                  pParam->BSIM3v0vth0 = model->BSIM3v0vth0
                                    + model->BSIM3v0lvth0 * Inv_L
                                    + model->BSIM3v0wvth0 * Inv_W
                                    + model->BSIM3v0pvth0 * Inv_LW;
                  pParam->BSIM3v0ua = model->BSIM3v0ua
                                  + model->BSIM3v0lua * Inv_L
                                  + model->BSIM3v0wua * Inv_W
                                  + model->BSIM3v0pua * Inv_LW;
                  pParam->BSIM3v0ua1 = model->BSIM3v0ua1
                                   + model->BSIM3v0lua1 * Inv_L
                                   + model->BSIM3v0wua1 * Inv_W
                                   + model->BSIM3v0pua1 * Inv_LW;
                  pParam->BSIM3v0ub = model->BSIM3v0ub
                                  + model->BSIM3v0lub * Inv_L
                                  + model->BSIM3v0wub * Inv_W
                                  + model->BSIM3v0pub * Inv_LW;
                  pParam->BSIM3v0ub1 = model->BSIM3v0ub1
                                   + model->BSIM3v0lub1 * Inv_L
                                   + model->BSIM3v0wub1 * Inv_W
                                   + model->BSIM3v0pub1 * Inv_LW;
                  pParam->BSIM3v0uc = model->BSIM3v0uc
                                  + model->BSIM3v0luc * Inv_L
                                  + model->BSIM3v0wuc * Inv_W
                                  + model->BSIM3v0puc * Inv_LW;
                  pParam->BSIM3v0uc1 = model->BSIM3v0uc1
                                   + model->BSIM3v0luc1 * Inv_L
                                   + model->BSIM3v0wuc1 * Inv_W
                                   + model->BSIM3v0puc1 * Inv_LW;
                  pParam->BSIM3v0u0 = model->BSIM3v0u0
                                  + model->BSIM3v0lu0 * Inv_L
                                  + model->BSIM3v0wu0 * Inv_W
                                  + model->BSIM3v0pu0 * Inv_LW;
                  pParam->BSIM3v0ute = model->BSIM3v0ute
                                   + model->BSIM3v0lute * Inv_L
                                   + model->BSIM3v0wute * Inv_W
                                   + model->BSIM3v0pute * Inv_LW;
                  pParam->BSIM3v0voff = model->BSIM3v0voff
                                    + model->BSIM3v0lvoff * Inv_L
                                    + model->BSIM3v0wvoff * Inv_W
                                    + model->BSIM3v0pvoff * Inv_LW;
                  pParam->BSIM3v0delta = model->BSIM3v0delta
                                     + model->BSIM3v0ldelta * Inv_L
                                     + model->BSIM3v0wdelta * Inv_W
                                     + model->BSIM3v0pdelta * Inv_LW;
                  pParam->BSIM3v0rdsw = model->BSIM3v0rdsw
                                    + model->BSIM3v0lrdsw * Inv_L
                                    + model->BSIM3v0wrdsw * Inv_W
                                    + model->BSIM3v0prdsw * Inv_LW;
                  pParam->BSIM3v0prwg = model->BSIM3v0prwg
                                    + model->BSIM3v0lprwg * Inv_L
                                    + model->BSIM3v0wprwg * Inv_W
                                    + model->BSIM3v0pprwg * Inv_LW;
                  pParam->BSIM3v0prwb = model->BSIM3v0prwb
                                    + model->BSIM3v0lprwb * Inv_L
                                    + model->BSIM3v0wprwb * Inv_W
                                    + model->BSIM3v0pprwb * Inv_LW;
                  pParam->BSIM3v0prt = model->BSIM3v0prt
                                    + model->BSIM3v0lprt * Inv_L
                                    + model->BSIM3v0wprt * Inv_W
                                    + model->BSIM3v0pprt * Inv_LW;
                  pParam->BSIM3v0eta0 = model->BSIM3v0eta0
                                    + model->BSIM3v0leta0 * Inv_L
                                    + model->BSIM3v0weta0 * Inv_W
                                    + model->BSIM3v0peta0 * Inv_LW;
                  pParam->BSIM3v0etab = model->BSIM3v0etab
                                    + model->BSIM3v0letab * Inv_L
                                    + model->BSIM3v0wetab * Inv_W
                                    + model->BSIM3v0petab * Inv_LW;
                  pParam->BSIM3v0pclm = model->BSIM3v0pclm
                                    + model->BSIM3v0lpclm * Inv_L
                                    + model->BSIM3v0wpclm * Inv_W
                                    + model->BSIM3v0ppclm * Inv_LW;
                  pParam->BSIM3v0pdibl1 = model->BSIM3v0pdibl1
                                      + model->BSIM3v0lpdibl1 * Inv_L
                                      + model->BSIM3v0wpdibl1 * Inv_W
                                      + model->BSIM3v0ppdibl1 * Inv_LW;
                  pParam->BSIM3v0pdibl2 = model->BSIM3v0pdibl2
                                      + model->BSIM3v0lpdibl2 * Inv_L
                                      + model->BSIM3v0wpdibl2 * Inv_W
                                      + model->BSIM3v0ppdibl2 * Inv_LW;
                  pParam->BSIM3v0pdiblb = model->BSIM3v0pdiblb
                                      + model->BSIM3v0lpdiblb * Inv_L
                                      + model->BSIM3v0wpdiblb * Inv_W
                                      + model->BSIM3v0ppdiblb * Inv_LW;
                  pParam->BSIM3v0pscbe1 = model->BSIM3v0pscbe1
                                      + model->BSIM3v0lpscbe1 * Inv_L
                                      + model->BSIM3v0wpscbe1 * Inv_W
                                      + model->BSIM3v0ppscbe1 * Inv_LW;
                  pParam->BSIM3v0pscbe2 = model->BSIM3v0pscbe2
                                      + model->BSIM3v0lpscbe2 * Inv_L
                                      + model->BSIM3v0wpscbe2 * Inv_W
                                      + model->BSIM3v0ppscbe2 * Inv_LW;
                  pParam->BSIM3v0pvag = model->BSIM3v0pvag
                                    + model->BSIM3v0lpvag * Inv_L
                                    + model->BSIM3v0wpvag * Inv_W
                                    + model->BSIM3v0ppvag * Inv_LW;
                  pParam->BSIM3v0wr = model->BSIM3v0wr
                                  + model->BSIM3v0lwr * Inv_L
                                  + model->BSIM3v0wwr * Inv_W
                                  + model->BSIM3v0pwr * Inv_LW;
                  pParam->BSIM3v0dwg = model->BSIM3v0dwg
                                   + model->BSIM3v0ldwg * Inv_L
                                   + model->BSIM3v0wdwg * Inv_W
                                   + model->BSIM3v0pdwg * Inv_LW;
                  pParam->BSIM3v0dwb = model->BSIM3v0dwb
                                   + model->BSIM3v0ldwb * Inv_L
                                   + model->BSIM3v0wdwb * Inv_W
                                   + model->BSIM3v0pdwb * Inv_LW;
                  pParam->BSIM3v0b0 = model->BSIM3v0b0
                                  + model->BSIM3v0lb0 * Inv_L
                                  + model->BSIM3v0wb0 * Inv_W
                                  + model->BSIM3v0pb0 * Inv_LW;
                  pParam->BSIM3v0b1 = model->BSIM3v0b1
                                  + model->BSIM3v0lb1 * Inv_L
                                  + model->BSIM3v0wb1 * Inv_W
                                  + model->BSIM3v0pb1 * Inv_LW;
                  pParam->BSIM3v0alpha0 = model->BSIM3v0alpha0
                                      + model->BSIM3v0lalpha0 * Inv_L
                                      + model->BSIM3v0walpha0 * Inv_W
                                      + model->BSIM3v0palpha0 * Inv_LW;
                  pParam->BSIM3v0beta0 = model->BSIM3v0beta0
                                     + model->BSIM3v0lbeta0 * Inv_L
                                     + model->BSIM3v0wbeta0 * Inv_W
                                     + model->BSIM3v0pbeta0 * Inv_LW;
                  /* CV model */
                  pParam->BSIM3v0elm = model->BSIM3v0elm
                                  + model->BSIM3v0lelm * Inv_L
                                  + model->BSIM3v0welm * Inv_W
                                  + model->BSIM3v0pelm * Inv_LW;
                  pParam->BSIM3v0cgsl = model->BSIM3v0cgsl
                                    + model->BSIM3v0lcgsl * Inv_L
                                    + model->BSIM3v0wcgsl * Inv_W
                                    + model->BSIM3v0pcgsl * Inv_LW;
                  pParam->BSIM3v0cgdl = model->BSIM3v0cgdl
                                    + model->BSIM3v0lcgdl * Inv_L
                                    + model->BSIM3v0wcgdl * Inv_W
                                    + model->BSIM3v0pcgdl * Inv_LW;
                  pParam->BSIM3v0ckappa = model->BSIM3v0ckappa
                                      + model->BSIM3v0lckappa * Inv_L
                                      + model->BSIM3v0wckappa * Inv_W
                                      + model->BSIM3v0pckappa * Inv_LW;
                  pParam->BSIM3v0cf = model->BSIM3v0cf
                                  + model->BSIM3v0lcf * Inv_L
                                  + model->BSIM3v0wcf * Inv_W
                                  + model->BSIM3v0pcf * Inv_LW;
                  pParam->BSIM3v0clc = model->BSIM3v0clc
                                   + model->BSIM3v0lclc * Inv_L
                                   + model->BSIM3v0wclc * Inv_W
                                   + model->BSIM3v0pclc * Inv_LW;
                  pParam->BSIM3v0cle = model->BSIM3v0cle
                                   + model->BSIM3v0lcle * Inv_L
                                   + model->BSIM3v0wcle * Inv_W
                                   + model->BSIM3v0pcle * Inv_LW;
                  pParam->BSIM3v0abulkCVfactor = 1.0 + pow((pParam->BSIM3v0clc
                                             / pParam->BSIM3v0leff),
                                             pParam->BSIM3v0cle);

                  pParam->BSIM3v0cgdo = (model->BSIM3v0cgdo + pParam->BSIM3v0cf)
                                    * pParam->BSIM3v0weffCV;
                  pParam->BSIM3v0cgso = (model->BSIM3v0cgso + pParam->BSIM3v0cf)
                                    * pParam->BSIM3v0weffCV;
                  pParam->BSIM3v0cgbo = model->BSIM3v0cgbo * pParam->BSIM3v0leffCV;

                  T0 = (TRatio - 1.0);
                  pParam->BSIM3v0ua = pParam->BSIM3v0ua + pParam->BSIM3v0ua1 * T0;
                  pParam->BSIM3v0ub = pParam->BSIM3v0ub + pParam->BSIM3v0ub1 * T0;
                  pParam->BSIM3v0uc = pParam->BSIM3v0uc + pParam->BSIM3v0uc1 * T0;

                  pParam->BSIM3v0u0temp = pParam->BSIM3v0u0
                                      * pow(TRatio, pParam->BSIM3v0ute);
                  pParam->BSIM3v0vsattemp = pParam->BSIM3v0vsat - pParam->BSIM3v0at
                                        * T0;
                  pParam->BSIM3v0rds0 = (pParam->BSIM3v0rdsw + pParam->BSIM3v0prt * T0)
                                    / pow(pParam->BSIM3v0weff * 1E6, pParam->BSIM3v0wr);

                  if (!model->BSIM3v0npeakGiven && model->BSIM3v0gamma1Given)
                  {   T0 = pParam->BSIM3v0gamma1 * model->BSIM3v0cox;
                      pParam->BSIM3v0npeak = 3.021E22 * T0 * T0;
                  }

                  Vtm0 = KboQ * Tnom;
                  Eg = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
                  ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
                     * exp(21.5565981 - Eg / (2.0 * Vtm0));

                  pParam->BSIM3v0phi = 2.0 * Vtm0
                                   * log(pParam->BSIM3v0npeak / ni);

                  pParam->BSIM3v0sqrtPhi = sqrt(pParam->BSIM3v0phi);
                  pParam->BSIM3v0phis3 = pParam->BSIM3v0sqrtPhi * pParam->BSIM3v0phi;

                  pParam->BSIM3v0Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM3v0npeak * 1.0e6))
                                     * pParam->BSIM3v0sqrtPhi;
                  pParam->BSIM3v0sqrtXdep0 = sqrt(pParam->BSIM3v0Xdep0);
                  pParam->BSIM3v0litl = sqrt(3.0 * pParam->BSIM3v0xj
                                    * model->BSIM3v0tox);
                  pParam->BSIM3v0vbi = Vtm0 * log(1.0e20
                                   * pParam->BSIM3v0npeak / (ni * ni));
                  pParam->BSIM3v0cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM3v0npeak * 1.0e6 / 2.0
                                     / pParam->BSIM3v0phi);

                  if (model->BSIM3v0k1Given || model->BSIM3v0k2Given)
                  {   if (!model->BSIM3v0k1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3v0k1 = 0.53;
                      }
                      if (!model->BSIM3v0k2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3v0k2 = -0.0186;
                      }
                      if (model->BSIM3v0nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v0xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v0vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v0vbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v0gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v0gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->BSIM3v0vbxGiven)
                          pParam->BSIM3v0vbx = pParam->BSIM3v0phi - 7.7348e-4
                                           * pParam->BSIM3v0npeak
                                           * pParam->BSIM3v0xt * pParam->BSIM3v0xt;
                      if (pParam->BSIM3v0vbx > 0.0)
                          pParam->BSIM3v0vbx = -pParam->BSIM3v0vbx;
                      if (pParam->BSIM3v0vbm > 0.0)
                          pParam->BSIM3v0vbm = -pParam->BSIM3v0vbm;

                      if (!model->BSIM3v0gamma1Given)
                          pParam->BSIM3v0gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM3v0npeak)
                                              / model->BSIM3v0cox;
                      if (!model->BSIM3v0gamma2Given)
                          pParam->BSIM3v0gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM3v0nsub)
                                              / model->BSIM3v0cox;

                      T0 = pParam->BSIM3v0gamma1 - pParam->BSIM3v0gamma2;
                      T1 = sqrt(pParam->BSIM3v0phi - pParam->BSIM3v0vbx)
                         - pParam->BSIM3v0sqrtPhi;
                      T2 = sqrt(pParam->BSIM3v0phi * (pParam->BSIM3v0phi
                         - pParam->BSIM3v0vbm)) - pParam->BSIM3v0phi;
                      pParam->BSIM3v0k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3v0vbm);
                      pParam->BSIM3v0k1 = pParam->BSIM3v0gamma2 - 2.0
                                      * pParam->BSIM3v0k2 * sqrt(pParam->BSIM3v0phi
                                      - pParam->BSIM3v0vbm);
                  }

                  if (pParam->BSIM3v0k2 > 0.0)
                  {   T0 = 0.5 * pParam->BSIM3v0k1 / pParam->BSIM3v0k2;
                      pParam->BSIM3v0vbsc = 0.9 * (pParam->BSIM3v0phi - T0 * T0);
                      if (pParam->BSIM3v0vbsc > -3.0)
                          pParam->BSIM3v0vbsc = -3.0;
                      else if (pParam->BSIM3v0vbsc < -30.0)
                          pParam->BSIM3v0vbsc = -30.0;
                  }
                  else
                  {   pParam->BSIM3v0vbsc = -10.0;
                  }

                  model->BSIM3v0vtm = KboQ * Temp;

                  if (model->BSIM3v0vth0Given)
                      pParam->BSIM3v0vfb = model->BSIM3v0type * pParam->BSIM3v0vth0
                                       - pParam->BSIM3v0phi - pParam->BSIM3v0k1
                                       * pParam->BSIM3v0sqrtPhi;
                  else
                      pParam->BSIM3v0vth0 = model->BSIM3v0type * (-1.0
                                        + pParam->BSIM3v0phi + pParam->BSIM3v0k1
                                        * pParam->BSIM3v0sqrtPhi);

                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3v0tox
                     * pParam->BSIM3v0Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3v0dsub * pParam->BSIM3v0leff / T1);
                  pParam->BSIM3v0theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3v0drout * pParam->BSIM3v0leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3v0thetaRout = pParam->BSIM3v0pdibl1 * T2
                                         + pParam->BSIM3v0pdibl2;

                  /* process source/drain series resistance */
                  here->BSIM3v0drainConductance = model->BSIM3v0sheetResistance
                                              * here->BSIM3v0drainSquares;
                  if (here->BSIM3v0drainConductance > 0.0)
                      here->BSIM3v0drainConductance = 1.0
                                                  / here->BSIM3v0drainConductance;
                  else
                      here->BSIM3v0drainConductance = 0.0;

                  here->BSIM3v0sourceConductance = model->BSIM3v0sheetResistance
                                               * here->BSIM3v0sourceSquares;
                  if (here->BSIM3v0sourceConductance > 0.0)
                      here->BSIM3v0sourceConductance = 1.0
                                                   / here->BSIM3v0sourceConductance;
                  else
                      here->BSIM3v0sourceConductance = 0.0;
              }
              here->BSIM3v0cgso = pParam->BSIM3v0cgso;
              here->BSIM3v0cgdo = pParam->BSIM3v0cgdo;
         }
    }
    return(OK);
}




