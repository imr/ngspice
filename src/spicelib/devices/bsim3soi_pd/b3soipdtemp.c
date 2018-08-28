/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdtemp.c          98/5/01
Modified by Pin Su	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su, Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su and Hui Wan 02/3/5
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su
 * BSIMPD2.2.3 release
 */

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5  /* Kb / q  where q = 1.60219e-19 */
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Eg300 1.115   /*  energy gap at 300K  */

#define MAX_EXPL 2.688117142e+43
#define MIN_EXPL 3.720075976e-44
#define EXPL_THRESHOLD 100.0
#define DEXP(A,B) {                                                        \
        if (A > EXPL_THRESHOLD) {                                              \
            B = MAX_EXPL*(1.0+(A)-EXPL_THRESHOLD);                              \
        } else if (A < -EXPL_THRESHOLD)  {                                                \
            B = MIN_EXPL;                                                      \
        } else   {                                                            \
            B = exp(A);                                                       \
        }                                                                     \
    }

/* ARGSUSED */
int
B3SOIPDtemp(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIPDmodel *model = (B3SOIPDmodel*) inModel;
B3SOIPDinstance *here;
struct b3soipdSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double Temp, TempRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double SDphi, SDgamma;
int Size_Not_Found;

/* v2.0 release */
double tmp3, T7;


    /*  loop through all the B3SOIPD device models */
    for (; model != NULL; model = B3SOIPDnextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->B3SOIPDGatesidewallJctPotential < 0.1)
             model->B3SOIPDGatesidewallJctPotential = 0.1;

        struct b3soipdSizeDependParam *p = model->pSizeDependParamKnot;
        while (p) {
            struct b3soipdSizeDependParam *next_p = p->pNext;
            FREE(p);
            p = next_p;
        }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->B3SOIPDtnom;
         TempRatio = Temp / Tnom;

         model->B3SOIPDvcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->B3SOIPDfactor1 = sqrt(EPSSI / EPSOX * model->B3SOIPDtox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         model->B3SOIPDeg0 = Eg0;
         model->B3SOIPDvtm = KboQ * Temp;

         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         /* ni is in cm^-3 */
         ni = 1.45e10 * (Temp / 300.15) * sqrt(Temp / 300.15)
            * exp(21.5565981 - Eg / (2.0 * model->B3SOIPDvtm));


         /* loop through all the instances of the model */
         /* MCJ: Length and Width not initialized */
         for (here = B3SOIPDinstances(model); here != NULL;
              here = B3SOIPDnextInstance(here))
         {
              here->B3SOIPDrbodyext = here->B3SOIPDbodySquares *
                                    model->B3SOIPDrbsh;
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->B3SOIPDl == pSizeDependParamKnot->Length)
                      && (here->B3SOIPDw == pSizeDependParamKnot->Width)
                      && (here->B3SOIPDrth0 == pSizeDependParamKnot->Rth0)
                      && (here->B3SOIPDcth0 == pSizeDependParamKnot->Cth0))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                      pParam = here->pParam; /* v2.2.3 bug fix */
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct b3soipdSizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->B3SOIPDl;
                  Wdrn = here->B3SOIPDw;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
                  pParam->Rth0 = here->B3SOIPDrth0;
                  pParam->Cth0 = here->B3SOIPDcth0;

                  T0 = pow(Ldrn, model->B3SOIPDLln);
                  T1 = pow(Wdrn, model->B3SOIPDLwn);
                  tmp1 = model->B3SOIPDLl / T0 + model->B3SOIPDLw / T1
                       + model->B3SOIPDLwl / (T0 * T1);
                  pParam->B3SOIPDdl = model->B3SOIPDLint + tmp1;

/* v2.2.3 */
                  tmp1 = model->B3SOIPDLlc / T0 + model->B3SOIPDLwc / T1
                       + model->B3SOIPDLwlc / (T0 * T1);
                  pParam->B3SOIPDdlc = model->B3SOIPDdlc + tmp1;


                  T2 = pow(Ldrn, model->B3SOIPDWln);
                  T3 = pow(Wdrn, model->B3SOIPDWwn);
                  tmp2 = model->B3SOIPDWl / T2 + model->B3SOIPDWw / T3
                       + model->B3SOIPDWwl / (T2 * T3);
                  pParam->B3SOIPDdw = model->B3SOIPDWint + tmp2;

/* v2.2.3 */
                  tmp2 = model->B3SOIPDWlc / T2 + model->B3SOIPDWwc / T3
                       + model->B3SOIPDWwlc / (T2 * T3);
                  pParam->B3SOIPDdwc = model->B3SOIPDdwc + tmp2;


                  pParam->B3SOIPDleff = here->B3SOIPDl - 2.0 * pParam->B3SOIPDdl;
                  if (pParam->B3SOIPDleff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIPD: mosfet %s, model %s: Effective channel length <= 0",
                       model->B3SOIPDmodName, here->B3SOIPDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIPDweff = here->B3SOIPDw - here->B3SOIPDnbc * model->B3SOIPDdwbc
                     - (2.0 - here->B3SOIPDnbc) * pParam->B3SOIPDdw;
                  if (pParam->B3SOIPDweff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIPD: mosfet %s, model %s: Effective channel width <= 0",
                       model->B3SOIPDmodName, here->B3SOIPDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIPDwdiod = pParam->B3SOIPDweff / here->B3SOIPDnseg + here->B3SOIPDpdbcp;
                  pParam->B3SOIPDwdios = pParam->B3SOIPDweff / here->B3SOIPDnseg + here->B3SOIPDpsbcp;

                  pParam->B3SOIPDleffCV = here->B3SOIPDl - 2.0 * pParam->B3SOIPDdlc;
                  if (pParam->B3SOIPDleffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIPD: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->B3SOIPDmodName, here->B3SOIPDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIPDweffCV = here->B3SOIPDw - here->B3SOIPDnbc * model->B3SOIPDdwbc
                     - (2.0 - here->B3SOIPDnbc) * pParam->B3SOIPDdwc;
                  if (pParam->B3SOIPDweffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIPD: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->B3SOIPDmodName, here->B3SOIPDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIPDwdiodCV = pParam->B3SOIPDweffCV / here->B3SOIPDnseg + here->B3SOIPDpdbcp;
                  pParam->B3SOIPDwdiosCV = pParam->B3SOIPDweffCV / here->B3SOIPDnseg + here->B3SOIPDpsbcp;

                  pParam->B3SOIPDleffCVb = here->B3SOIPDl - 2.0 * pParam->B3SOIPDdlc - model->B3SOIPDdlcb;
                  if (pParam->B3SOIPDleffCVb <= 0.0)
                  {
                     SPfrontEnd->IFerrorf (ERR_FATAL,
                     "B3SOIPD: mosfet %s, model %s: Effective channel length for C-V (body) <= 0",
                     model->B3SOIPDmodName, here->B3SOIPDname);
                     return(E_BADPARM);
                  }

                  pParam->B3SOIPDleffCVbg = pParam->B3SOIPDleffCVb + 2 * model->B3SOIPDdlbg;
                  if (pParam->B3SOIPDleffCVbg <= 0.0)
                  {
                     SPfrontEnd->IFerrorf (ERR_FATAL,
                     "B3SOIPD: mosfet %s, model %s: Effective channel length for C-V (backgate) <= 0",
                     model->B3SOIPDmodName, here->B3SOIPDname);
                     return(E_BADPARM);
                  }

                  /* Not binned - START */
                  pParam->B3SOIPDat = model->B3SOIPDat;
                  pParam->B3SOIPDgamma1 = model->B3SOIPDgamma1;
                  pParam->B3SOIPDgamma2 = model->B3SOIPDgamma2;
                  pParam->B3SOIPDvbx = model->B3SOIPDvbx;
                  pParam->B3SOIPDvbm = model->B3SOIPDvbm;
                  pParam->B3SOIPDxt = model->B3SOIPDxt;
                  pParam->B3SOIPDkt1 = model->B3SOIPDkt1;
                  pParam->B3SOIPDkt1l = model->B3SOIPDkt1l;
                  pParam->B3SOIPDkt2 = model->B3SOIPDkt2;
                  pParam->B3SOIPDua1 = model->B3SOIPDua1;
                  pParam->B3SOIPDub1 = model->B3SOIPDub1;
                  pParam->B3SOIPDuc1 = model->B3SOIPDuc1;
                  pParam->B3SOIPDute = model->B3SOIPDute;
                  pParam->B3SOIPDprt = model->B3SOIPDprt;
                  /* Not binned - END */

                  /* CV model */
                  pParam->B3SOIPDcgsl = model->B3SOIPDcgsl;
                  pParam->B3SOIPDcgdl = model->B3SOIPDcgdl;
                  pParam->B3SOIPDckappa = model->B3SOIPDckappa;
                  pParam->B3SOIPDcf = model->B3SOIPDcf;
                  pParam->B3SOIPDclc = model->B3SOIPDclc;
                  pParam->B3SOIPDcle = model->B3SOIPDcle;

                  pParam->B3SOIPDabulkCVfactor = 1.0 + pow((pParam->B3SOIPDclc / pParam->B3SOIPDleff),
                                             pParam->B3SOIPDcle);

                  /* Added for binning - START */
                  if (model->B3SOIPDbinUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->B3SOIPDleff;
                      Inv_W = 1.0e-6 / pParam->B3SOIPDweff;
                      Inv_LW = 1.0e-12 / (pParam->B3SOIPDleff
                             * pParam->B3SOIPDweff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->B3SOIPDleff;
                      Inv_W = 1.0 / pParam->B3SOIPDweff;
                      Inv_LW = 1.0 / (pParam->B3SOIPDleff
                             * pParam->B3SOIPDweff);
                  }
                  pParam->B3SOIPDnpeak = model->B3SOIPDnpeak
                                     + model->B3SOIPDlnpeak * Inv_L
                                     + model->B3SOIPDwnpeak * Inv_W
                                     + model->B3SOIPDpnpeak * Inv_LW;
                  pParam->B3SOIPDnsub = model->B3SOIPDnsub
                                    + model->B3SOIPDlnsub * Inv_L
                                    + model->B3SOIPDwnsub * Inv_W
                                    + model->B3SOIPDpnsub * Inv_LW;
                  pParam->B3SOIPDngate = model->B3SOIPDngate
                                     + model->B3SOIPDlngate * Inv_L
                                     + model->B3SOIPDwngate * Inv_W
                                     + model->B3SOIPDpngate * Inv_LW;
                  pParam->B3SOIPDvth0 = model->B3SOIPDvth0
                                    + model->B3SOIPDlvth0 * Inv_L
                                    + model->B3SOIPDwvth0 * Inv_W
                                    + model->B3SOIPDpvth0 * Inv_LW;
                  pParam->B3SOIPDk1 = model->B3SOIPDk1
                                  + model->B3SOIPDlk1 * Inv_L
                                  + model->B3SOIPDwk1 * Inv_W
                                  + model->B3SOIPDpk1 * Inv_LW;
                  pParam->B3SOIPDk2 = model->B3SOIPDk2
                                  + model->B3SOIPDlk2 * Inv_L
                                  + model->B3SOIPDwk2 * Inv_W
                                  + model->B3SOIPDpk2 * Inv_LW;
                  pParam->B3SOIPDk1w1 = model->B3SOIPDk1w1
                                  + model->B3SOIPDlk1w1 * Inv_L
                                  + model->B3SOIPDwk1w1 * Inv_W
                                  + model->B3SOIPDpk1w1 * Inv_LW;
                  pParam->B3SOIPDk1w2 = model->B3SOIPDk1w2
                                  + model->B3SOIPDlk1w2 * Inv_L
                                  + model->B3SOIPDwk1w2 * Inv_W
                                  + model->B3SOIPDpk1w2 * Inv_LW;
                  pParam->B3SOIPDk3 = model->B3SOIPDk3
                                  + model->B3SOIPDlk3 * Inv_L
                                  + model->B3SOIPDwk3 * Inv_W
                                  + model->B3SOIPDpk3 * Inv_LW;
                  pParam->B3SOIPDk3b = model->B3SOIPDk3b
                                   + model->B3SOIPDlk3b * Inv_L
                                   + model->B3SOIPDwk3b * Inv_W
                                   + model->B3SOIPDpk3b * Inv_LW;
                  pParam->B3SOIPDkb1 = model->B3SOIPDkb1
                                   + model->B3SOIPDlkb1 * Inv_L
                                   + model->B3SOIPDwkb1 * Inv_W
                                   + model->B3SOIPDpkb1 * Inv_LW;
                  pParam->B3SOIPDw0 = model->B3SOIPDw0
                                  + model->B3SOIPDlw0 * Inv_L
                                  + model->B3SOIPDww0 * Inv_W
                                  + model->B3SOIPDpw0 * Inv_LW;
                  pParam->B3SOIPDnlx = model->B3SOIPDnlx
                                   + model->B3SOIPDlnlx * Inv_L
                                   + model->B3SOIPDwnlx * Inv_W
                                   + model->B3SOIPDpnlx * Inv_LW;
                  pParam->B3SOIPDdvt0 = model->B3SOIPDdvt0
                                    + model->B3SOIPDldvt0 * Inv_L
                                    + model->B3SOIPDwdvt0 * Inv_W
                                    + model->B3SOIPDpdvt0 * Inv_LW;
                  pParam->B3SOIPDdvt1 = model->B3SOIPDdvt1
                                    + model->B3SOIPDldvt1 * Inv_L
                                    + model->B3SOIPDwdvt1 * Inv_W
                                    + model->B3SOIPDpdvt1 * Inv_LW;
                  pParam->B3SOIPDdvt2 = model->B3SOIPDdvt2
                                    + model->B3SOIPDldvt2 * Inv_L
                                    + model->B3SOIPDwdvt2 * Inv_W
                                    + model->B3SOIPDpdvt2 * Inv_LW;
                  pParam->B3SOIPDdvt0w = model->B3SOIPDdvt0w
                                    + model->B3SOIPDldvt0w * Inv_L
                                    + model->B3SOIPDwdvt0w * Inv_W
                                    + model->B3SOIPDpdvt0w * Inv_LW;
                  pParam->B3SOIPDdvt1w = model->B3SOIPDdvt1w
                                    + model->B3SOIPDldvt1w * Inv_L
                                    + model->B3SOIPDwdvt1w * Inv_W
                                    + model->B3SOIPDpdvt1w * Inv_LW;
                  pParam->B3SOIPDdvt2w = model->B3SOIPDdvt2w
                                    + model->B3SOIPDldvt2w * Inv_L
                                    + model->B3SOIPDwdvt2w * Inv_W
                                    + model->B3SOIPDpdvt2w * Inv_LW;
                  pParam->B3SOIPDu0 = model->B3SOIPDu0
                                  + model->B3SOIPDlu0 * Inv_L
                                  + model->B3SOIPDwu0 * Inv_W
                                  + model->B3SOIPDpu0 * Inv_LW;
                  pParam->B3SOIPDua = model->B3SOIPDua
                                  + model->B3SOIPDlua * Inv_L
                                  + model->B3SOIPDwua * Inv_W
                                  + model->B3SOIPDpua * Inv_LW;
                  pParam->B3SOIPDub = model->B3SOIPDub
                                  + model->B3SOIPDlub * Inv_L
                                  + model->B3SOIPDwub * Inv_W
                                  + model->B3SOIPDpub * Inv_LW;
                  pParam->B3SOIPDuc = model->B3SOIPDuc
                                  + model->B3SOIPDluc * Inv_L
                                  + model->B3SOIPDwuc * Inv_W
                                  + model->B3SOIPDpuc * Inv_LW;
                  pParam->B3SOIPDvsat = model->B3SOIPDvsat
                                    + model->B3SOIPDlvsat * Inv_L
                                    + model->B3SOIPDwvsat * Inv_W
                                    + model->B3SOIPDpvsat * Inv_LW;
                  pParam->B3SOIPDa0 = model->B3SOIPDa0
                                  + model->B3SOIPDla0 * Inv_L
                                  + model->B3SOIPDwa0 * Inv_W
                                  + model->B3SOIPDpa0 * Inv_LW;
                  pParam->B3SOIPDags = model->B3SOIPDags
                                  + model->B3SOIPDlags * Inv_L
                                  + model->B3SOIPDwags * Inv_W
                                  + model->B3SOIPDpags * Inv_LW;
                  pParam->B3SOIPDb0 = model->B3SOIPDb0
                                  + model->B3SOIPDlb0 * Inv_L
                                  + model->B3SOIPDwb0 * Inv_W
                                  + model->B3SOIPDpb0 * Inv_LW;
                  pParam->B3SOIPDb1 = model->B3SOIPDb1
                                  + model->B3SOIPDlb1 * Inv_L
                                  + model->B3SOIPDwb1 * Inv_W
                                  + model->B3SOIPDpb1 * Inv_LW;
                  pParam->B3SOIPDketa = model->B3SOIPDketa
                                    + model->B3SOIPDlketa * Inv_L
                                    + model->B3SOIPDwketa * Inv_W
                                    + model->B3SOIPDpketa * Inv_LW;
                  pParam->B3SOIPDketas = model->B3SOIPDketas
                                    + model->B3SOIPDlketas * Inv_L
                                    + model->B3SOIPDwketas * Inv_W
                                    + model->B3SOIPDpketas * Inv_LW;
                  pParam->B3SOIPDa1 = model->B3SOIPDa1
                                  + model->B3SOIPDla1 * Inv_L
                                  + model->B3SOIPDwa1 * Inv_W
                                  + model->B3SOIPDpa1 * Inv_LW;
                  pParam->B3SOIPDa2 = model->B3SOIPDa2
                                  + model->B3SOIPDla2 * Inv_L
                                  + model->B3SOIPDwa2 * Inv_W
                                  + model->B3SOIPDpa2 * Inv_LW;
                  pParam->B3SOIPDrdsw = model->B3SOIPDrdsw
                                    + model->B3SOIPDlrdsw * Inv_L
                                    + model->B3SOIPDwrdsw * Inv_W
                                    + model->B3SOIPDprdsw * Inv_LW;
                  pParam->B3SOIPDprwb = model->B3SOIPDprwb
                                    + model->B3SOIPDlprwb * Inv_L
                                    + model->B3SOIPDwprwb * Inv_W
                                    + model->B3SOIPDpprwb * Inv_LW;
                  pParam->B3SOIPDprwg = model->B3SOIPDprwg
                                    + model->B3SOIPDlprwg * Inv_L
                                    + model->B3SOIPDwprwg * Inv_W
                                    + model->B3SOIPDpprwg * Inv_LW;
                  pParam->B3SOIPDwr = model->B3SOIPDwr
                                  + model->B3SOIPDlwr * Inv_L
                                  + model->B3SOIPDwwr * Inv_W
                                  + model->B3SOIPDpwr * Inv_LW;
                  pParam->B3SOIPDnfactor = model->B3SOIPDnfactor
                                       + model->B3SOIPDlnfactor * Inv_L
                                       + model->B3SOIPDwnfactor * Inv_W
                                       + model->B3SOIPDpnfactor * Inv_LW;
                  pParam->B3SOIPDdwg = model->B3SOIPDdwg
                                   + model->B3SOIPDldwg * Inv_L
                                   + model->B3SOIPDwdwg * Inv_W
                                   + model->B3SOIPDpdwg * Inv_LW;
                  pParam->B3SOIPDdwb = model->B3SOIPDdwb
                                   + model->B3SOIPDldwb * Inv_L
                                   + model->B3SOIPDwdwb * Inv_W
                                   + model->B3SOIPDpdwb * Inv_LW;
                  pParam->B3SOIPDvoff = model->B3SOIPDvoff
                                    + model->B3SOIPDlvoff * Inv_L
                                    + model->B3SOIPDwvoff * Inv_W
                                    + model->B3SOIPDpvoff * Inv_LW;
                  pParam->B3SOIPDeta0 = model->B3SOIPDeta0
                                    + model->B3SOIPDleta0 * Inv_L
                                    + model->B3SOIPDweta0 * Inv_W
                                    + model->B3SOIPDpeta0 * Inv_LW;
                  pParam->B3SOIPDetab = model->B3SOIPDetab
                                    + model->B3SOIPDletab * Inv_L
                                    + model->B3SOIPDwetab * Inv_W
                                    + model->B3SOIPDpetab * Inv_LW;
                  pParam->B3SOIPDdsub = model->B3SOIPDdsub
                                    + model->B3SOIPDldsub * Inv_L
                                    + model->B3SOIPDwdsub * Inv_W
                                    + model->B3SOIPDpdsub * Inv_LW;
                  pParam->B3SOIPDcit = model->B3SOIPDcit
                                   + model->B3SOIPDlcit * Inv_L
                                   + model->B3SOIPDwcit * Inv_W
                                   + model->B3SOIPDpcit * Inv_LW;
                  pParam->B3SOIPDcdsc = model->B3SOIPDcdsc
                                    + model->B3SOIPDlcdsc * Inv_L
                                    + model->B3SOIPDwcdsc * Inv_W
                                    + model->B3SOIPDpcdsc * Inv_LW;
                  pParam->B3SOIPDcdscb = model->B3SOIPDcdscb
                                     + model->B3SOIPDlcdscb * Inv_L
                                     + model->B3SOIPDwcdscb * Inv_W
                                     + model->B3SOIPDpcdscb * Inv_LW;
                      pParam->B3SOIPDcdscd = model->B3SOIPDcdscd
                                     + model->B3SOIPDlcdscd * Inv_L
                                     + model->B3SOIPDwcdscd * Inv_W
                                     + model->B3SOIPDpcdscd * Inv_LW;
                  pParam->B3SOIPDpclm = model->B3SOIPDpclm
                                    + model->B3SOIPDlpclm * Inv_L
                                    + model->B3SOIPDwpclm * Inv_W
                                    + model->B3SOIPDppclm * Inv_LW;
                  pParam->B3SOIPDpdibl1 = model->B3SOIPDpdibl1
                                      + model->B3SOIPDlpdibl1 * Inv_L
                                      + model->B3SOIPDwpdibl1 * Inv_W
                                      + model->B3SOIPDppdibl1 * Inv_LW;
                  pParam->B3SOIPDpdibl2 = model->B3SOIPDpdibl2
                                      + model->B3SOIPDlpdibl2 * Inv_L
                                      + model->B3SOIPDwpdibl2 * Inv_W
                                      + model->B3SOIPDppdibl2 * Inv_LW;
                  pParam->B3SOIPDpdiblb = model->B3SOIPDpdiblb
                                      + model->B3SOIPDlpdiblb * Inv_L
                                      + model->B3SOIPDwpdiblb * Inv_W
                                      + model->B3SOIPDppdiblb * Inv_LW;
                  pParam->B3SOIPDdrout = model->B3SOIPDdrout
                                     + model->B3SOIPDldrout * Inv_L
                                     + model->B3SOIPDwdrout * Inv_W
                                     + model->B3SOIPDpdrout * Inv_LW;
                  pParam->B3SOIPDpvag = model->B3SOIPDpvag
                                    + model->B3SOIPDlpvag * Inv_L
                                    + model->B3SOIPDwpvag * Inv_W
                                    + model->B3SOIPDppvag * Inv_LW;
                  pParam->B3SOIPDdelta = model->B3SOIPDdelta
                                     + model->B3SOIPDldelta * Inv_L
                                     + model->B3SOIPDwdelta * Inv_W
                                     + model->B3SOIPDpdelta * Inv_LW;
                  pParam->B3SOIPDalpha0 = model->B3SOIPDalpha0
                                      + model->B3SOIPDlalpha0 * Inv_L
                                      + model->B3SOIPDwalpha0 * Inv_W
                                      + model->B3SOIPDpalpha0 * Inv_LW;
                  pParam->B3SOIPDfbjtii = model->B3SOIPDfbjtii
                                      + model->B3SOIPDlfbjtii * Inv_L
                                      + model->B3SOIPDwfbjtii * Inv_W
                                      + model->B3SOIPDpfbjtii * Inv_LW;
                  pParam->B3SOIPDbeta0 = model->B3SOIPDbeta0
                                     + model->B3SOIPDlbeta0 * Inv_L
                                     + model->B3SOIPDwbeta0 * Inv_W
                                     + model->B3SOIPDpbeta0 * Inv_LW;
                  pParam->B3SOIPDbeta1 = model->B3SOIPDbeta1
                                     + model->B3SOIPDlbeta1 * Inv_L
                                     + model->B3SOIPDwbeta1 * Inv_W
                                     + model->B3SOIPDpbeta1 * Inv_LW;
                  pParam->B3SOIPDbeta2 = model->B3SOIPDbeta2
                                     + model->B3SOIPDlbeta2 * Inv_L
                                     + model->B3SOIPDwbeta2 * Inv_W
                                     + model->B3SOIPDpbeta2 * Inv_LW;
                  pParam->B3SOIPDvdsatii0 = model->B3SOIPDvdsatii0
                                      + model->B3SOIPDlvdsatii0 * Inv_L
                                      + model->B3SOIPDwvdsatii0 * Inv_W
                                      + model->B3SOIPDpvdsatii0 * Inv_LW;
                  pParam->B3SOIPDlii = model->B3SOIPDlii
                                      + model->B3SOIPDllii * Inv_L
                                      + model->B3SOIPDwlii * Inv_W
                                      + model->B3SOIPDplii * Inv_LW;
                  pParam->B3SOIPDesatii = model->B3SOIPDesatii
                                      + model->B3SOIPDlesatii * Inv_L
                                      + model->B3SOIPDwesatii * Inv_W
                                      + model->B3SOIPDpesatii * Inv_LW;
                  pParam->B3SOIPDsii0 = model->B3SOIPDsii0
                                      + model->B3SOIPDlsii0 * Inv_L
                                      + model->B3SOIPDwsii0 * Inv_W
                                      + model->B3SOIPDpsii0 * Inv_LW;
                  pParam->B3SOIPDsii1 = model->B3SOIPDsii1
                                      + model->B3SOIPDlsii1 * Inv_L
                                      + model->B3SOIPDwsii1 * Inv_W
                                      + model->B3SOIPDpsii1 * Inv_LW;
                  pParam->B3SOIPDsii2 = model->B3SOIPDsii2
                                      + model->B3SOIPDlsii2 * Inv_L
                                      + model->B3SOIPDwsii2 * Inv_W
                                      + model->B3SOIPDpsii2 * Inv_LW;
                  pParam->B3SOIPDsiid = model->B3SOIPDsiid
                                      + model->B3SOIPDlsiid * Inv_L
                                      + model->B3SOIPDwsiid * Inv_W
                                      + model->B3SOIPDpsiid * Inv_LW;
                  pParam->B3SOIPDagidl = model->B3SOIPDagidl
                                      + model->B3SOIPDlagidl * Inv_L
                                      + model->B3SOIPDwagidl * Inv_W
                                      + model->B3SOIPDpagidl * Inv_LW;
                  pParam->B3SOIPDbgidl = model->B3SOIPDbgidl
                                      + model->B3SOIPDlbgidl * Inv_L
                                      + model->B3SOIPDwbgidl * Inv_W
                                      + model->B3SOIPDpbgidl * Inv_LW;
                  pParam->B3SOIPDngidl = model->B3SOIPDngidl
                                      + model->B3SOIPDlngidl * Inv_L
                                      + model->B3SOIPDwngidl * Inv_W
                                      + model->B3SOIPDpngidl * Inv_LW;
                  pParam->B3SOIPDntun = model->B3SOIPDntun
                                      + model->B3SOIPDlntun * Inv_L
                                      + model->B3SOIPDwntun * Inv_W
                                      + model->B3SOIPDpntun * Inv_LW;
                  pParam->B3SOIPDndiode = model->B3SOIPDndiode
                                      + model->B3SOIPDlndiode * Inv_L
                                      + model->B3SOIPDwndiode * Inv_W
                                      + model->B3SOIPDpndiode * Inv_LW;
                  pParam->B3SOIPDnrecf0 = model->B3SOIPDnrecf0
                                  + model->B3SOIPDlnrecf0 * Inv_L
                                  + model->B3SOIPDwnrecf0 * Inv_W
                                  + model->B3SOIPDpnrecf0 * Inv_LW;
                  pParam->B3SOIPDnrecr0 = model->B3SOIPDnrecr0
                                  + model->B3SOIPDlnrecr0 * Inv_L
                                  + model->B3SOIPDwnrecr0 * Inv_W
                                  + model->B3SOIPDpnrecr0 * Inv_LW;
                  pParam->B3SOIPDisbjt = model->B3SOIPDisbjt
                                  + model->B3SOIPDlisbjt * Inv_L
                                  + model->B3SOIPDwisbjt * Inv_W
                                  + model->B3SOIPDpisbjt * Inv_LW;
                  pParam->B3SOIPDisdif = model->B3SOIPDisdif
                                  + model->B3SOIPDlisdif * Inv_L
                                  + model->B3SOIPDwisdif * Inv_W
                                  + model->B3SOIPDpisdif * Inv_LW;
                  pParam->B3SOIPDisrec = model->B3SOIPDisrec
                                  + model->B3SOIPDlisrec * Inv_L
                                  + model->B3SOIPDwisrec * Inv_W
                                  + model->B3SOIPDpisrec * Inv_LW;
                  pParam->B3SOIPDistun = model->B3SOIPDistun
                                  + model->B3SOIPDlistun * Inv_L
                                  + model->B3SOIPDwistun * Inv_W
                                  + model->B3SOIPDpistun * Inv_LW;
                  pParam->B3SOIPDvrec0 = model->B3SOIPDvrec0
                                  + model->B3SOIPDlvrec0 * Inv_L
                                  + model->B3SOIPDwvrec0 * Inv_W
                                  + model->B3SOIPDpvrec0 * Inv_LW;
                  pParam->B3SOIPDvtun0 = model->B3SOIPDvtun0
                                  + model->B3SOIPDlvtun0 * Inv_L
                                  + model->B3SOIPDwvtun0 * Inv_W
                                  + model->B3SOIPDpvtun0 * Inv_LW;
                  pParam->B3SOIPDnbjt = model->B3SOIPDnbjt
                                  + model->B3SOIPDlnbjt * Inv_L
                                  + model->B3SOIPDwnbjt * Inv_W
                                  + model->B3SOIPDpnbjt * Inv_LW;
                  pParam->B3SOIPDlbjt0 = model->B3SOIPDlbjt0
                                  + model->B3SOIPDllbjt0 * Inv_L
                                  + model->B3SOIPDwlbjt0 * Inv_W
                                  + model->B3SOIPDplbjt0 * Inv_LW;
                  pParam->B3SOIPDvabjt = model->B3SOIPDvabjt
                                  + model->B3SOIPDlvabjt * Inv_L
                                  + model->B3SOIPDwvabjt * Inv_W
                                  + model->B3SOIPDpvabjt * Inv_LW;
                  pParam->B3SOIPDaely = model->B3SOIPDaely
                                  + model->B3SOIPDlaely * Inv_L
                                  + model->B3SOIPDwaely * Inv_W
                                  + model->B3SOIPDpaely * Inv_LW;
                  pParam->B3SOIPDahli = model->B3SOIPDahli
                                  + model->B3SOIPDlahli * Inv_L
                                  + model->B3SOIPDwahli * Inv_W
                                  + model->B3SOIPDpahli * Inv_LW;
                  /* CV model */
                  pParam->B3SOIPDvsdfb = model->B3SOIPDvsdfb
                                  + model->B3SOIPDlvsdfb * Inv_L
                                  + model->B3SOIPDwvsdfb * Inv_W
                                  + model->B3SOIPDpvsdfb * Inv_LW;
                  pParam->B3SOIPDvsdth = model->B3SOIPDvsdth
                                  + model->B3SOIPDlvsdth * Inv_L
                                  + model->B3SOIPDwvsdth * Inv_W
                                  + model->B3SOIPDpvsdth * Inv_LW;
                  pParam->B3SOIPDdelvt = model->B3SOIPDdelvt
                                  + model->B3SOIPDldelvt * Inv_L
                                  + model->B3SOIPDwdelvt * Inv_W
                                  + model->B3SOIPDpdelvt * Inv_LW;
                  pParam->B3SOIPDacde = model->B3SOIPDacde
                                  + model->B3SOIPDlacde * Inv_L
                                  + model->B3SOIPDwacde * Inv_W
                                  + model->B3SOIPDpacde * Inv_LW;
                  pParam->B3SOIPDmoin = model->B3SOIPDmoin
                                  + model->B3SOIPDlmoin * Inv_L
                                  + model->B3SOIPDwmoin * Inv_W
                                  + model->B3SOIPDpmoin * Inv_LW;
                  /* Added for binning - END */

                  T0 = (TempRatio - 1.0);

                  pParam->B3SOIPDuatemp = pParam->B3SOIPDua;  /*  save ua, ub, and uc for b3soipdld.c */
                  pParam->B3SOIPDubtemp = pParam->B3SOIPDub;
                  pParam->B3SOIPDuctemp = pParam->B3SOIPDuc;
                  pParam->B3SOIPDrds0denom = pow(pParam->B3SOIPDweff * 1E6, pParam->B3SOIPDwr);


/* v2.2 release */
                  pParam->B3SOIPDrth = here->B3SOIPDrth0 / (pParam->B3SOIPDweff + model->B3SOIPDwth0)
                                   * here->B3SOIPDnseg;
                  pParam->B3SOIPDcth = here->B3SOIPDcth0 * (pParam->B3SOIPDweff + model->B3SOIPDwth0)
                                   / here->B3SOIPDnseg;

/* v2.2.2 adding layout-dependent Frbody multiplier */
                  pParam->B3SOIPDrbody = here->B3SOIPDfrbody *model->B3SOIPDrbody * model->B3SOIPDrhalo
                                     / (2 * model->B3SOIPDrbody + model->B3SOIPDrhalo * pParam->B3SOIPDleff)
                                     * pParam->B3SOIPDweff / here->B3SOIPDnseg;

                  pParam->B3SOIPDoxideRatio = pow(model->B3SOIPDtoxref/model->B3SOIPDtoxqm,
                                  model->B3SOIPDntox) /model->B3SOIPDtoxqm/model->B3SOIPDtoxqm;
/* v2.2 release */


                  pParam->B3SOIPDua = pParam->B3SOIPDua + pParam->B3SOIPDua1 * T0;
                  pParam->B3SOIPDub = pParam->B3SOIPDub + pParam->B3SOIPDub1 * T0;
                  pParam->B3SOIPDuc = pParam->B3SOIPDuc + pParam->B3SOIPDuc1 * T0;
                  if (pParam->B3SOIPDu0 > 1.0)
                      pParam->B3SOIPDu0 = pParam->B3SOIPDu0 / 1.0e4;

                  pParam->B3SOIPDu0temp = pParam->B3SOIPDu0
                                      * pow(TempRatio, pParam->B3SOIPDute);
                  pParam->B3SOIPDvsattemp = pParam->B3SOIPDvsat - pParam->B3SOIPDat
                                        * T0;
                  pParam->B3SOIPDrds0 = (pParam->B3SOIPDrdsw + pParam->B3SOIPDprt * T0)
                                    / pow(pParam->B3SOIPDweff * 1E6, pParam->B3SOIPDwr);

                  if (B3SOIPDcheckModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during B3SOIPDV3 parameter checking for %s in model %s", model->B3SOIPDmodName, here->B3SOIPDname);
                      return(E_BADPARM);
                  }


                  pParam->B3SOIPDcgdo = (model->B3SOIPDcgdo + pParam->B3SOIPDcf)
                                    * pParam->B3SOIPDwdiodCV;
                  pParam->B3SOIPDcgso = (model->B3SOIPDcgso + pParam->B3SOIPDcf)
                                    * pParam->B3SOIPDwdiosCV;

                  pParam->B3SOIPDcgeo = model->B3SOIPDcgeo
                                    * pParam->B3SOIPDleffCV;


                  if (!model->B3SOIPDnpeakGiven && model->B3SOIPDgamma1Given)
                  {   T0 = pParam->B3SOIPDgamma1 * model->B3SOIPDcox;
                      pParam->B3SOIPDnpeak = 3.021E22 * T0 * T0;
                  }


                  T4 = Eg300 / model->B3SOIPDvtm * (TempRatio - 1.0);
                  T7 = model->B3SOIPDxbjt * T4 / pParam->B3SOIPDndiode;
                  DEXP(T7, T0);
                  T7 = model->B3SOIPDxdif * T4 / pParam->B3SOIPDndiode;
                  DEXP(T7, T1);
                  T7 = model->B3SOIPDxrec * T4 / pParam->B3SOIPDnrecf0;
                  DEXP(T7, T2);

                  /* v2.2.2 bug fix */
                  pParam->B3SOIPDahli0 = pParam->B3SOIPDahli * T0;

                  pParam->B3SOIPDjbjt = pParam->B3SOIPDisbjt * T0;
                  pParam->B3SOIPDjdif = pParam->B3SOIPDisdif * T1;
                  pParam->B3SOIPDjrec = pParam->B3SOIPDisrec * T2;

                  T7 = model->B3SOIPDxtun * (TempRatio - 1);
                  DEXP(T7, T0);
                  pParam->B3SOIPDjtun = pParam->B3SOIPDistun * T0;


                  if (pParam->B3SOIPDnsub > 0)
                     pParam->B3SOIPDvfbb = -model->B3SOIPDtype * model->B3SOIPDvtm *
                                log(pParam->B3SOIPDnpeak/ pParam->B3SOIPDnsub);
                  else
                     pParam->B3SOIPDvfbb = -model->B3SOIPDtype * model->B3SOIPDvtm *
                                log(-pParam->B3SOIPDnpeak* pParam->B3SOIPDnsub/ni/ni);

                  if (!model->B3SOIPDvsdfbGiven)
                  {
                     if (pParam->B3SOIPDnsub > 0)
                        pParam->B3SOIPDvsdfb = -model->B3SOIPDtype * (model->B3SOIPDvtm*log(1e20 *
                                            pParam->B3SOIPDnsub / ni /ni) - 0.3);
                     else if (pParam->B3SOIPDnsub < 0)
                        pParam->B3SOIPDvsdfb = -model->B3SOIPDtype * (model->B3SOIPDvtm*log(-1e20 /
                                            pParam->B3SOIPDnsub) + 0.3);
                  }

                  /* Phi  & Gamma */
                  SDphi = 2.0*model->B3SOIPDvtm*log(fabs(pParam->B3SOIPDnsub) / ni);
                  SDgamma = 5.753e-12 * sqrt(fabs(pParam->B3SOIPDnsub)) / model->B3SOIPDcbox;

                  if (!model->B3SOIPDvsdthGiven)
                  {
                     if ( ((pParam->B3SOIPDnsub > 0) && (model->B3SOIPDtype > 0)) ||
                          ((pParam->B3SOIPDnsub < 0) && (model->B3SOIPDtype < 0)) )
                        pParam->B3SOIPDvsdth = pParam->B3SOIPDvsdfb + SDphi +
                                            SDgamma * sqrt(SDphi);
                     else
                        pParam->B3SOIPDvsdth = pParam->B3SOIPDvsdfb - SDphi -
                                            SDgamma * sqrt(SDphi);
                  }

                  if (!model->B3SOIPDcsdminGiven) {
                     /* Cdmin */
                     tmp = sqrt(2.0 * EPSSI * SDphi / (Charge_q *
                                fabs(pParam->B3SOIPDnsub) * 1.0e6));
                     tmp1 = EPSSI / tmp;
                     model->B3SOIPDcsdmin = tmp1 * model->B3SOIPDcbox /
                                          (tmp1 + model->B3SOIPDcbox);
                  }


                  pParam->B3SOIPDphi = 2.0 * model->B3SOIPDvtm
                                   * log(pParam->B3SOIPDnpeak / ni);

                  pParam->B3SOIPDsqrtPhi = sqrt(pParam->B3SOIPDphi);
                  pParam->B3SOIPDphis3 = pParam->B3SOIPDsqrtPhi * pParam->B3SOIPDphi;

                  pParam->B3SOIPDXdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->B3SOIPDnpeak * 1.0e6))
                                     * pParam->B3SOIPDsqrtPhi;
                  pParam->B3SOIPDsqrtXdep0 = sqrt(pParam->B3SOIPDXdep0);
                  pParam->B3SOIPDlitl = sqrt(3.0 * model->B3SOIPDxj
                                    * model->B3SOIPDtox);
                  pParam->B3SOIPDvbi = model->B3SOIPDvtm * log(1.0e20
                                   * pParam->B3SOIPDnpeak / (ni * ni));
                  pParam->B3SOIPDcdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->B3SOIPDnpeak * 1.0e6 / 2.0
                                     / pParam->B3SOIPDphi);

                  if (model->B3SOIPDk1Given || model->B3SOIPDk2Given)
                  {   if (!model->B3SOIPDk1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->B3SOIPDk1 = 0.53;
                      }
                      if (!model->B3SOIPDk2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->B3SOIPDk2 = -0.0186;
                      }
                      if (model->B3SOIPDxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIPDvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIPDvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIPDgamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIPDgamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->B3SOIPDvbxGiven)
                          pParam->B3SOIPDvbx = pParam->B3SOIPDphi - 7.7348e-4
                                           * pParam->B3SOIPDnpeak
                                           * pParam->B3SOIPDxt * pParam->B3SOIPDxt;
                      if (pParam->B3SOIPDvbx > 0.0)
                          pParam->B3SOIPDvbx = -pParam->B3SOIPDvbx;
                      if (pParam->B3SOIPDvbm > 0.0)
                          pParam->B3SOIPDvbm = -pParam->B3SOIPDvbm;

                      if (!model->B3SOIPDgamma1Given)
                          pParam->B3SOIPDgamma1 = 5.753e-12
                                              * sqrt(pParam->B3SOIPDnpeak)
                                              / model->B3SOIPDcox;
                      if (!model->B3SOIPDgamma2Given)
                          pParam->B3SOIPDgamma2 = 5.753e-12
                                              * sqrt(pParam->B3SOIPDnsub)
                                              / model->B3SOIPDcox;

                      T0 = pParam->B3SOIPDgamma1 - pParam->B3SOIPDgamma2;
                      T1 = sqrt(pParam->B3SOIPDphi - pParam->B3SOIPDvbx)
                         - pParam->B3SOIPDsqrtPhi;
                      T2 = sqrt(pParam->B3SOIPDphi * (pParam->B3SOIPDphi
                         - pParam->B3SOIPDvbm)) - pParam->B3SOIPDphi;
                      pParam->B3SOIPDk2 = T0 * T1 / (2.0 * T2 + pParam->B3SOIPDvbm);
                      pParam->B3SOIPDk1 = pParam->B3SOIPDgamma2 - 2.0
                                      * pParam->B3SOIPDk2 * sqrt(pParam->B3SOIPDphi
                                      - pParam->B3SOIPDvbm);
                  }

                  if (pParam->B3SOIPDk2 < 0.0)
                  {   T0 = 0.5 * pParam->B3SOIPDk1 / pParam->B3SOIPDk2;
                      pParam->B3SOIPDvbsc = 0.9 * (pParam->B3SOIPDphi - T0 * T0);
                      if (pParam->B3SOIPDvbsc > -3.0)
                          pParam->B3SOIPDvbsc = -3.0;
                      else if (pParam->B3SOIPDvbsc < -30.0)
                          pParam->B3SOIPDvbsc = -30.0;
                  }
                  else
                  {   pParam->B3SOIPDvbsc = -30.0;
                  }
                  if (pParam->B3SOIPDvbsc > pParam->B3SOIPDvbm)
                      pParam->B3SOIPDvbsc = pParam->B3SOIPDvbm;

                  if ((T0 = pParam->B3SOIPDweff + pParam->B3SOIPDk1w2) < 1e-8)
                     T0 = 1e-8;
                  pParam->B3SOIPDk1eff = pParam->B3SOIPDk1 * (1 + pParam->B3SOIPDk1w1/T0);

                  if (model->B3SOIPDvth0Given)
                  {   pParam->B3SOIPDvfb = model->B3SOIPDtype * pParam->B3SOIPDvth0
                                       - pParam->B3SOIPDphi - pParam->B3SOIPDk1eff
                                       * pParam->B3SOIPDsqrtPhi;
                  }
                  else
                  {   pParam->B3SOIPDvfb = -1.0;
                      pParam->B3SOIPDvth0 = model->B3SOIPDtype * (pParam->B3SOIPDvfb
                                        + pParam->B3SOIPDphi + pParam->B3SOIPDk1eff
                                        * pParam->B3SOIPDsqrtPhi);
                  }
                  T1 = sqrt(EPSSI / EPSOX * model->B3SOIPDtox
                     * pParam->B3SOIPDXdep0);
                  T0 = exp(-0.5 * pParam->B3SOIPDdsub * pParam->B3SOIPDleff / T1);
                  pParam->B3SOIPDtheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->B3SOIPDdrout * pParam->B3SOIPDleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->B3SOIPDthetaRout = pParam->B3SOIPDpdibl1 * T2
                                         + pParam->B3SOIPDpdibl2;
              }

              here->B3SOIPDcsbox = model->B3SOIPDcbox*here->B3SOIPDsourceArea;
              here->B3SOIPDcsmin = model->B3SOIPDcsdmin*here->B3SOIPDsourceArea;
              here->B3SOIPDcdbox = model->B3SOIPDcbox*here->B3SOIPDdrainArea;
              here->B3SOIPDcdmin = model->B3SOIPDcsdmin*here->B3SOIPDdrainArea;

              if ( ((pParam->B3SOIPDnsub > 0) && (model->B3SOIPDtype > 0)) ||
                   ((pParam->B3SOIPDnsub < 0) && (model->B3SOIPDtype < 0)) )
              {
                 T0 = pParam->B3SOIPDvsdth - pParam->B3SOIPDvsdfb;
                 pParam->B3SOIPDsdt1 = pParam->B3SOIPDvsdfb + model->B3SOIPDasd * T0;
                 T1 = here->B3SOIPDcsbox - here->B3SOIPDcsmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIPDst2 = T2 / model->B3SOIPDasd;
                 pParam->B3SOIPDst3 = T2 /( 1 - model->B3SOIPDasd);
                 here->B3SOIPDst4 =  T0 * T1 * (1 + model->B3SOIPDasd) / 3
                                  - here->B3SOIPDcsmin * pParam->B3SOIPDvsdfb;

                 T1 = here->B3SOIPDcdbox - here->B3SOIPDcdmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIPDdt2 = T2 / model->B3SOIPDasd;
                 pParam->B3SOIPDdt3 = T2 /( 1 - model->B3SOIPDasd);
                 here->B3SOIPDdt4 =  T0 * T1 * (1 + model->B3SOIPDasd) / 3
                                  - here->B3SOIPDcdmin * pParam->B3SOIPDvsdfb;
              } else
              {
                 T0 = pParam->B3SOIPDvsdfb - pParam->B3SOIPDvsdth;
                 pParam->B3SOIPDsdt1 = pParam->B3SOIPDvsdth + model->B3SOIPDasd * T0;
                 T1 = here->B3SOIPDcsmin - here->B3SOIPDcsbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIPDst2 = T2 / model->B3SOIPDasd;
                 pParam->B3SOIPDst3 = T2 /( 1 - model->B3SOIPDasd);
                 here->B3SOIPDst4 =  T0 * T1 * (1 + model->B3SOIPDasd) / 3
                                  - here->B3SOIPDcsbox * pParam->B3SOIPDvsdth;

                 T1 = here->B3SOIPDcdmin - here->B3SOIPDcdbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIPDdt2 = T2 / model->B3SOIPDasd;
                 pParam->B3SOIPDdt3 = T2 /( 1 - model->B3SOIPDasd);
                 here->B3SOIPDdt4 =  T0 * T1 * (1 + model->B3SOIPDasd) / 3
                                      - here->B3SOIPDcdbox * pParam->B3SOIPDvsdth;
              }

              /* v2.2.2 bug fix */
              T0 = model->B3SOIPDcsdesw * log(1 + model->B3SOIPDtsi /
                 model->B3SOIPDtbox);
              T1 = here->B3SOIPDsourcePerimeter - here->B3SOIPDw;
              if (T1 > 0.0)
                 here->B3SOIPDcsesw = T0 * T1;
              else
                 here->B3SOIPDcsesw = 0.0;
              T1 = here->B3SOIPDdrainPerimeter - here->B3SOIPDw;
              if (T1 > 0.0)
                 here->B3SOIPDcdesw = T0 * T1;
              else
                 here->B3SOIPDcdesw = 0.0;


              here->B3SOIPDphi = pParam->B3SOIPDphi;
              /* process source/drain series resistance */
              here->B3SOIPDdrainConductance = model->B3SOIPDsheetResistance
                                              * here->B3SOIPDdrainSquares;
              if (here->B3SOIPDdrainConductance > 0.0)
                  here->B3SOIPDdrainConductance = 1.0
                                              / here->B3SOIPDdrainConductance;
              else
                  here->B3SOIPDdrainConductance = 0.0;

              here->B3SOIPDsourceConductance = model->B3SOIPDsheetResistance
                                           * here->B3SOIPDsourceSquares;
              if (here->B3SOIPDsourceConductance > 0.0)
                  here->B3SOIPDsourceConductance = 1.0
                                               / here->B3SOIPDsourceConductance;
              else
                  here->B3SOIPDsourceConductance = 0.0;
              here->B3SOIPDcgso = pParam->B3SOIPDcgso;
              here->B3SOIPDcgdo = pParam->B3SOIPDcgdo;


/* v2.0 release */
              if (model->B3SOIPDln < 1e-15) model->B3SOIPDln = 1e-15;
              T0 = -0.5 * pParam->B3SOIPDleff * pParam->B3SOIPDleff / model->B3SOIPDln / model->B3SOIPDln;
              DEXP(T0,T1);
              pParam->B3SOIPDarfabjt = T1;

              T0 = pParam->B3SOIPDlbjt0 * (1.0 / pParam->B3SOIPDleff + 1.0 / model->B3SOIPDln);
              pParam->B3SOIPDlratio = pow(T0,pParam->B3SOIPDnbjt);
              pParam->B3SOIPDlratiodif = 1.0 + model->B3SOIPDldif0 * pow(T0,model->B3SOIPDndif);

              if ((pParam->B3SOIPDvearly = pParam->B3SOIPDvabjt + pParam->B3SOIPDaely * pParam->B3SOIPDleff) < 1)
                 pParam->B3SOIPDvearly = 1;

              /* vfbzb calculation for capMod 3 */
              tmp = sqrt(pParam->B3SOIPDXdep0);
              tmp1 = pParam->B3SOIPDvbi - pParam->B3SOIPDphi;
              tmp2 = model->B3SOIPDfactor1 * tmp;

              T0 = -0.5 * pParam->B3SOIPDdvt1w * pParam->B3SOIPDweff
                 * pParam->B3SOIPDleff / tmp2;
              if (T0 > -EXPL_THRESHOLD)
              {   T1 = exp(T0);
                  T2 = T1 * (1.0 + 2.0 * T1);
              }
              else
              {   T1 = MIN_EXPL;
                  T2 = T1 * (1.0 + 2.0 * T1);
              }
              T0 = pParam->B3SOIPDdvt0w * T2;
              T2 = T0 * tmp1;

              T0 = -0.5 * pParam->B3SOIPDdvt1 * pParam->B3SOIPDleff / tmp2;
              if (T0 > -EXPL_THRESHOLD)
              {   T1 = exp(T0);
                  T3 = T1 * (1.0 + 2.0 * T1);
              }
              else
              {   T1 = MIN_EXPL;
                  T3 = T1 * (1.0 + 2.0 * T1);
              }
              T3 = pParam->B3SOIPDdvt0 * T3 * tmp1;

/* v2.2.3 */
              T4 = (model->B3SOIPDtox - model->B3SOIPDdtoxcv) * pParam->B3SOIPDphi
                 / (pParam->B3SOIPDweff + pParam->B3SOIPDw0);

              T0 = sqrt(1.0 + pParam->B3SOIPDnlx / pParam->B3SOIPDleff);
              T5 = pParam->B3SOIPDk1eff * (T0 - 1.0) * pParam->B3SOIPDsqrtPhi
                 + (pParam->B3SOIPDkt1 + pParam->B3SOIPDkt1l / pParam->B3SOIPDleff)
                 * (TempRatio - 1.0);

              tmp3 = model->B3SOIPDtype * pParam->B3SOIPDvth0
                   - T2 - T3 + pParam->B3SOIPDk3 * T4 + T5;
              pParam->B3SOIPDvfbzb = tmp3 - pParam->B3SOIPDphi - pParam->B3SOIPDk1eff
                                 * pParam->B3SOIPDsqrtPhi;
              /* End of vfbzb */

              pParam->B3SOIPDldeb = sqrt(EPSSI * Vtm0 / (Charge_q * pParam->B3SOIPDnpeak * 1.0e6)) / 3.0;
              pParam->B3SOIPDacde = pParam->B3SOIPDacde * pow((pParam->B3SOIPDnpeak / 2.0e16), -0.25);
         }
    }
    return(OK);
}

