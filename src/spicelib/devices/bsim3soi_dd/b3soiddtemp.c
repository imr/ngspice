/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Pin Su, Wei Jin 99/9/27
File: b3soiddtemp.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMDD2.1 release
 */

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
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
B3SOIDDtemp(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIDDmodel *model = (B3SOIDDmodel*) inModel;
B3SOIDDinstance *here;
struct b3soiddSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, T6, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double SDphi, SDgamma;
int Size_Not_Found;

    /*  loop through all the B3SOIDD device models */
    for (; model != NULL; model = B3SOIDDnextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->B3SOIDDGatesidewallJctPotential < 0.1)
             model->B3SOIDDGatesidewallJctPotential = 0.1;

         struct b3soiddSizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct b3soiddSizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->B3SOIDDtnom;
         TRatio = Temp / Tnom;

         model->B3SOIDDvcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->B3SOIDDfactor1 = sqrt(EPSSI / EPSOX * model->B3SOIDDtox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         model->B3SOIDDeg0 = Eg0;
         model->B3SOIDDvtm = KboQ * Temp;

         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         /* ni is in cm^-3 */
         ni = 1.45e10 * (Temp / 300.15) * sqrt(Temp / 300.15)
            * exp(21.5565981 - Eg / (2.0 * model->B3SOIDDvtm));


         /* loop through all the instances of the model */
         /* MCJ: Length and Width not initialized */
         for (here = B3SOIDDinstances(model); here != NULL;
              here = B3SOIDDnextInstance(here))
         {
              here->B3SOIDDrbodyext = here->B3SOIDDbodySquares *
                                    model->B3SOIDDrbsh;
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->B3SOIDDl == pSizeDependParamKnot->Length)
                      && (here->B3SOIDDw == pSizeDependParamKnot->Width)
                      && (here->B3SOIDDrth0 == pSizeDependParamKnot->Rth0)
                      && (here->B3SOIDDcth0 == pSizeDependParamKnot->Cth0))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct b3soiddSizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->B3SOIDDl;
                  Wdrn = here->B3SOIDDw;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
                  pParam->Rth0 = here->B3SOIDDrth0;
                  pParam->Cth0 = here->B3SOIDDcth0;

                  T0 = pow(Ldrn, model->B3SOIDDLln);
                  T1 = pow(Wdrn, model->B3SOIDDLwn);
                  tmp1 = model->B3SOIDDLl / T0 + model->B3SOIDDLw / T1
                       + model->B3SOIDDLwl / (T0 * T1);
                  pParam->B3SOIDDdl = model->B3SOIDDLint + tmp1;
                  pParam->B3SOIDDdlc = model->B3SOIDDdlc + tmp1;

                  T2 = pow(Ldrn, model->B3SOIDDWln);
                  T3 = pow(Wdrn, model->B3SOIDDWwn);
                  tmp2 = model->B3SOIDDWl / T2 + model->B3SOIDDWw / T3
                       + model->B3SOIDDWwl / (T2 * T3);
                  pParam->B3SOIDDdw = model->B3SOIDDWint + tmp2;
                  pParam->B3SOIDDdwc = model->B3SOIDDdwc + tmp2;

                  pParam->B3SOIDDleff = here->B3SOIDDl - 2.0 * pParam->B3SOIDDdl;
                  if (pParam->B3SOIDDleff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIDD: mosfet %s, model %s: Effective channel length <= 0",
                       model->B3SOIDDmodName, here->B3SOIDDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIDDweff = here->B3SOIDDw - 2.0 * pParam->B3SOIDDdw;
                  if (pParam->B3SOIDDweff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIDD: mosfet %s, model %s: Effective channel width <= 0",
                       model->B3SOIDDmodName, here->B3SOIDDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIDDleffCV = here->B3SOIDDl - 2.0 * pParam->B3SOIDDdlc;
                  if (pParam->B3SOIDDleffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIDD: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->B3SOIDDmodName, here->B3SOIDDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIDDweffCV = here->B3SOIDDw - 2.0 * pParam->B3SOIDDdwc;
                  if (pParam->B3SOIDDweffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIDD: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->B3SOIDDmodName, here->B3SOIDDname);
                      return(E_BADPARM);
                  }

                  /* Not binned - START */
                  pParam->B3SOIDDat = model->B3SOIDDat;
                  pParam->B3SOIDDgamma1 = model->B3SOIDDgamma1;
                  pParam->B3SOIDDgamma2 = model->B3SOIDDgamma2;
                  pParam->B3SOIDDvbx = model->B3SOIDDvbx;
                  pParam->B3SOIDDvbm = model->B3SOIDDvbm;
                  pParam->B3SOIDDxt = model->B3SOIDDxt;
                  pParam->B3SOIDDkt1 = model->B3SOIDDkt1;
                  pParam->B3SOIDDkt1l = model->B3SOIDDkt1l;
                  pParam->B3SOIDDkt2 = model->B3SOIDDkt2;
                  pParam->B3SOIDDua1 = model->B3SOIDDua1;
                  pParam->B3SOIDDub1 = model->B3SOIDDub1;
                  pParam->B3SOIDDuc1 = model->B3SOIDDuc1;
                  pParam->B3SOIDDute = model->B3SOIDDute;
                  pParam->B3SOIDDprt = model->B3SOIDDprt;
                  /* Not binned - END */

                  /* CV model */
                  pParam->B3SOIDDcgsl = model->B3SOIDDcgsl;
                  pParam->B3SOIDDcgdl = model->B3SOIDDcgdl;
                  pParam->B3SOIDDckappa = model->B3SOIDDckappa;
                  pParam->B3SOIDDcf = model->B3SOIDDcf;
                  pParam->B3SOIDDclc = model->B3SOIDDclc;
                  pParam->B3SOIDDcle = model->B3SOIDDcle;

                  pParam->B3SOIDDabulkCVfactor = pow(1.0+(pParam->B3SOIDDclc
                                             / pParam->B3SOIDDleff),
                                             pParam->B3SOIDDcle);

/* Added for binning - START */
                  if (model->B3SOIDDbinUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->B3SOIDDleff;
                      Inv_W = 1.0e-6 / pParam->B3SOIDDweff;
                      Inv_LW = 1.0e-12 / (pParam->B3SOIDDleff
                             * pParam->B3SOIDDweff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->B3SOIDDleff;
                      Inv_W = 1.0 / pParam->B3SOIDDweff;
                      Inv_LW = 1.0 / (pParam->B3SOIDDleff
                             * pParam->B3SOIDDweff);
                  }
                  pParam->B3SOIDDnpeak = model->B3SOIDDnpeak
                                     + model->B3SOIDDlnpeak * Inv_L
                                     + model->B3SOIDDwnpeak * Inv_W
                                     + model->B3SOIDDpnpeak * Inv_LW;
                  pParam->B3SOIDDnsub = model->B3SOIDDnsub
                                    + model->B3SOIDDlnsub * Inv_L
                                    + model->B3SOIDDwnsub * Inv_W
                                    + model->B3SOIDDpnsub * Inv_LW;
                  pParam->B3SOIDDngate = model->B3SOIDDngate
                                     + model->B3SOIDDlngate * Inv_L
                                     + model->B3SOIDDwngate * Inv_W
                                     + model->B3SOIDDpngate * Inv_LW;
                  pParam->B3SOIDDvth0 = model->B3SOIDDvth0
                                    + model->B3SOIDDlvth0 * Inv_L
                                    + model->B3SOIDDwvth0 * Inv_W
                                    + model->B3SOIDDpvth0 * Inv_LW;
                  pParam->B3SOIDDk1 = model->B3SOIDDk1
                                  + model->B3SOIDDlk1 * Inv_L
                                  + model->B3SOIDDwk1 * Inv_W
                                  + model->B3SOIDDpk1 * Inv_LW;
                  pParam->B3SOIDDk2 = model->B3SOIDDk2
                                  + model->B3SOIDDlk2 * Inv_L
                                  + model->B3SOIDDwk2 * Inv_W
                                  + model->B3SOIDDpk2 * Inv_LW;
                  pParam->B3SOIDDk3 = model->B3SOIDDk3
                                  + model->B3SOIDDlk3 * Inv_L
                                  + model->B3SOIDDwk3 * Inv_W
                                  + model->B3SOIDDpk3 * Inv_LW;
                  pParam->B3SOIDDk3b = model->B3SOIDDk3b
                                   + model->B3SOIDDlk3b * Inv_L
                                   + model->B3SOIDDwk3b * Inv_W
                                   + model->B3SOIDDpk3b * Inv_LW;
                  pParam->B3SOIDDvbsa = model->B3SOIDDvbsa
                                   + model->B3SOIDDlvbsa * Inv_L
                                   + model->B3SOIDDwvbsa * Inv_W
                                   + model->B3SOIDDpvbsa * Inv_LW;
                  pParam->B3SOIDDdelp = model->B3SOIDDdelp
                                   + model->B3SOIDDldelp * Inv_L
                                   + model->B3SOIDDwdelp * Inv_W
                                   + model->B3SOIDDpdelp * Inv_LW;
                  pParam->B3SOIDDkb1 = model->B3SOIDDkb1
                                   + model->B3SOIDDlkb1 * Inv_L
                                   + model->B3SOIDDwkb1 * Inv_W
                                   + model->B3SOIDDpkb1 * Inv_LW;
                  pParam->B3SOIDDkb3 = model->B3SOIDDkb3
                                   + model->B3SOIDDlkb3 * Inv_L
                                   + model->B3SOIDDwkb3 * Inv_W
                                   + model->B3SOIDDpkb3 * Inv_LW;
                  pParam->B3SOIDDdvbd0 = model->B3SOIDDdvbd0
                                   + model->B3SOIDDldvbd0 * Inv_L
                                   + model->B3SOIDDwdvbd0 * Inv_W
                                   + model->B3SOIDDpdvbd0 * Inv_LW;
                  pParam->B3SOIDDdvbd1 = model->B3SOIDDdvbd1
                                   + model->B3SOIDDldvbd1 * Inv_L
                                   + model->B3SOIDDwdvbd1 * Inv_W
                                   + model->B3SOIDDpdvbd1 * Inv_LW;
                  pParam->B3SOIDDw0 = model->B3SOIDDw0
                                  + model->B3SOIDDlw0 * Inv_L
                                  + model->B3SOIDDww0 * Inv_W
                                  + model->B3SOIDDpw0 * Inv_LW;
                  pParam->B3SOIDDnlx = model->B3SOIDDnlx
                                   + model->B3SOIDDlnlx * Inv_L
                                   + model->B3SOIDDwnlx * Inv_W
                                   + model->B3SOIDDpnlx * Inv_LW;
                  pParam->B3SOIDDdvt0 = model->B3SOIDDdvt0
                                    + model->B3SOIDDldvt0 * Inv_L
                                    + model->B3SOIDDwdvt0 * Inv_W
                                    + model->B3SOIDDpdvt0 * Inv_LW;
                  pParam->B3SOIDDdvt1 = model->B3SOIDDdvt1
                                    + model->B3SOIDDldvt1 * Inv_L
                                    + model->B3SOIDDwdvt1 * Inv_W
                                    + model->B3SOIDDpdvt1 * Inv_LW;
                  pParam->B3SOIDDdvt2 = model->B3SOIDDdvt2
                                    + model->B3SOIDDldvt2 * Inv_L
                                    + model->B3SOIDDwdvt2 * Inv_W
                                    + model->B3SOIDDpdvt2 * Inv_LW;
                  pParam->B3SOIDDdvt0w = model->B3SOIDDdvt0w
                                    + model->B3SOIDDldvt0w * Inv_L
                                    + model->B3SOIDDwdvt0w * Inv_W
                                    + model->B3SOIDDpdvt0w * Inv_LW;
                  pParam->B3SOIDDdvt1w = model->B3SOIDDdvt1w
                                    + model->B3SOIDDldvt1w * Inv_L
                                    + model->B3SOIDDwdvt1w * Inv_W
                                    + model->B3SOIDDpdvt1w * Inv_LW;
                  pParam->B3SOIDDdvt2w = model->B3SOIDDdvt2w
                                    + model->B3SOIDDldvt2w * Inv_L
                                    + model->B3SOIDDwdvt2w * Inv_W
                                    + model->B3SOIDDpdvt2w * Inv_LW;
                  pParam->B3SOIDDu0 = model->B3SOIDDu0
                                  + model->B3SOIDDlu0 * Inv_L
                                  + model->B3SOIDDwu0 * Inv_W
                                  + model->B3SOIDDpu0 * Inv_LW;
                  pParam->B3SOIDDua = model->B3SOIDDua
                                  + model->B3SOIDDlua * Inv_L
                                  + model->B3SOIDDwua * Inv_W
                                  + model->B3SOIDDpua * Inv_LW;
                  pParam->B3SOIDDub = model->B3SOIDDub
                                  + model->B3SOIDDlub * Inv_L
                                  + model->B3SOIDDwub * Inv_W
                                  + model->B3SOIDDpub * Inv_LW;
                  pParam->B3SOIDDuc = model->B3SOIDDuc
                                  + model->B3SOIDDluc * Inv_L
                                  + model->B3SOIDDwuc * Inv_W
                                  + model->B3SOIDDpuc * Inv_LW;
                  pParam->B3SOIDDvsat = model->B3SOIDDvsat
                                    + model->B3SOIDDlvsat * Inv_L
                                    + model->B3SOIDDwvsat * Inv_W
                                    + model->B3SOIDDpvsat * Inv_LW;
                  pParam->B3SOIDDa0 = model->B3SOIDDa0
                                  + model->B3SOIDDla0 * Inv_L
                                  + model->B3SOIDDwa0 * Inv_W
                                  + model->B3SOIDDpa0 * Inv_LW;
                  pParam->B3SOIDDags = model->B3SOIDDags
                                  + model->B3SOIDDlags * Inv_L
                                  + model->B3SOIDDwags * Inv_W
                                  + model->B3SOIDDpags * Inv_LW;
                  pParam->B3SOIDDb0 = model->B3SOIDDb0
                                  + model->B3SOIDDlb0 * Inv_L
                                  + model->B3SOIDDwb0 * Inv_W
                                  + model->B3SOIDDpb0 * Inv_LW;
                  pParam->B3SOIDDb1 = model->B3SOIDDb1
                                  + model->B3SOIDDlb1 * Inv_L
                                  + model->B3SOIDDwb1 * Inv_W
                                  + model->B3SOIDDpb1 * Inv_LW;
                  pParam->B3SOIDDketa = model->B3SOIDDketa
                                    + model->B3SOIDDlketa * Inv_L
                                    + model->B3SOIDDwketa * Inv_W
                                    + model->B3SOIDDpketa * Inv_LW;
                  pParam->B3SOIDDabp = model->B3SOIDDabp
                                  + model->B3SOIDDlabp * Inv_L
                                  + model->B3SOIDDwabp * Inv_W
                                  + model->B3SOIDDpabp * Inv_LW;
                  pParam->B3SOIDDmxc = model->B3SOIDDmxc
                                  + model->B3SOIDDlmxc * Inv_L
                                  + model->B3SOIDDwmxc * Inv_W
                                  + model->B3SOIDDpmxc * Inv_LW;
                  pParam->B3SOIDDadice0 = model->B3SOIDDadice0
                                  + model->B3SOIDDladice0 * Inv_L
                                  + model->B3SOIDDwadice0 * Inv_W
                                  + model->B3SOIDDpadice0 * Inv_LW;
                  pParam->B3SOIDDa1 = model->B3SOIDDa1
                                  + model->B3SOIDDla1 * Inv_L
                                  + model->B3SOIDDwa1 * Inv_W
                                  + model->B3SOIDDpa1 * Inv_LW;
                  pParam->B3SOIDDa2 = model->B3SOIDDa2
                                  + model->B3SOIDDla2 * Inv_L
                                  + model->B3SOIDDwa2 * Inv_W
                                  + model->B3SOIDDpa2 * Inv_LW;
                  pParam->B3SOIDDrdsw = model->B3SOIDDrdsw
                                    + model->B3SOIDDlrdsw * Inv_L
                                    + model->B3SOIDDwrdsw * Inv_W
                                    + model->B3SOIDDprdsw * Inv_LW;
                  pParam->B3SOIDDprwb = model->B3SOIDDprwb
                                    + model->B3SOIDDlprwb * Inv_L
                                    + model->B3SOIDDwprwb * Inv_W
                                    + model->B3SOIDDpprwb * Inv_LW;
                  pParam->B3SOIDDprwg = model->B3SOIDDprwg
                                    + model->B3SOIDDlprwg * Inv_L
                                    + model->B3SOIDDwprwg * Inv_W
                                    + model->B3SOIDDpprwg * Inv_LW;
                  pParam->B3SOIDDwr = model->B3SOIDDwr
                                  + model->B3SOIDDlwr * Inv_L
                                  + model->B3SOIDDwwr * Inv_W
                                  + model->B3SOIDDpwr * Inv_LW;
                  pParam->B3SOIDDnfactor = model->B3SOIDDnfactor
                                       + model->B3SOIDDlnfactor * Inv_L
                                       + model->B3SOIDDwnfactor * Inv_W
                                       + model->B3SOIDDpnfactor * Inv_LW;
                  pParam->B3SOIDDdwg = model->B3SOIDDdwg
                                   + model->B3SOIDDldwg * Inv_L
                                   + model->B3SOIDDwdwg * Inv_W
                                   + model->B3SOIDDpdwg * Inv_LW;
                  pParam->B3SOIDDdwb = model->B3SOIDDdwb
                                   + model->B3SOIDDldwb * Inv_L
                                   + model->B3SOIDDwdwb * Inv_W
                                   + model->B3SOIDDpdwb * Inv_LW;
                  pParam->B3SOIDDvoff = model->B3SOIDDvoff
                                    + model->B3SOIDDlvoff * Inv_L
                                    + model->B3SOIDDwvoff * Inv_W
                                    + model->B3SOIDDpvoff * Inv_LW;
                  pParam->B3SOIDDeta0 = model->B3SOIDDeta0
                                    + model->B3SOIDDleta0 * Inv_L
                                    + model->B3SOIDDweta0 * Inv_W
                                    + model->B3SOIDDpeta0 * Inv_LW;
                  pParam->B3SOIDDetab = model->B3SOIDDetab
                                    + model->B3SOIDDletab * Inv_L
                                    + model->B3SOIDDwetab * Inv_W
                                    + model->B3SOIDDpetab * Inv_LW;
                  pParam->B3SOIDDdsub = model->B3SOIDDdsub
                                    + model->B3SOIDDldsub * Inv_L
                                    + model->B3SOIDDwdsub * Inv_W
                                    + model->B3SOIDDpdsub * Inv_LW;
                  pParam->B3SOIDDcit = model->B3SOIDDcit
                                   + model->B3SOIDDlcit * Inv_L
                                   + model->B3SOIDDwcit * Inv_W
                                   + model->B3SOIDDpcit * Inv_LW;
                  pParam->B3SOIDDcdsc = model->B3SOIDDcdsc
                                    + model->B3SOIDDlcdsc * Inv_L
                                    + model->B3SOIDDwcdsc * Inv_W
                                    + model->B3SOIDDpcdsc * Inv_LW;
                  pParam->B3SOIDDcdscb = model->B3SOIDDcdscb
                                     + model->B3SOIDDlcdscb * Inv_L
                                     + model->B3SOIDDwcdscb * Inv_W
                                     + model->B3SOIDDpcdscb * Inv_LW;
                      pParam->B3SOIDDcdscd = model->B3SOIDDcdscd
                                     + model->B3SOIDDlcdscd * Inv_L
                                     + model->B3SOIDDwcdscd * Inv_W
                                     + model->B3SOIDDpcdscd * Inv_LW;
                  pParam->B3SOIDDpclm = model->B3SOIDDpclm
                                    + model->B3SOIDDlpclm * Inv_L
                                    + model->B3SOIDDwpclm * Inv_W
                                    + model->B3SOIDDppclm * Inv_LW;
                  pParam->B3SOIDDpdibl1 = model->B3SOIDDpdibl1
                                      + model->B3SOIDDlpdibl1 * Inv_L
                                      + model->B3SOIDDwpdibl1 * Inv_W
                                      + model->B3SOIDDppdibl1 * Inv_LW;
                  pParam->B3SOIDDpdibl2 = model->B3SOIDDpdibl2
                                      + model->B3SOIDDlpdibl2 * Inv_L
                                      + model->B3SOIDDwpdibl2 * Inv_W
                                      + model->B3SOIDDppdibl2 * Inv_LW;
                  pParam->B3SOIDDpdiblb = model->B3SOIDDpdiblb
                                      + model->B3SOIDDlpdiblb * Inv_L
                                      + model->B3SOIDDwpdiblb * Inv_W
                                      + model->B3SOIDDppdiblb * Inv_LW;
                  pParam->B3SOIDDdrout = model->B3SOIDDdrout
                                     + model->B3SOIDDldrout * Inv_L
                                     + model->B3SOIDDwdrout * Inv_W
                                     + model->B3SOIDDpdrout * Inv_LW;
                  pParam->B3SOIDDpvag = model->B3SOIDDpvag
                                    + model->B3SOIDDlpvag * Inv_L
                                    + model->B3SOIDDwpvag * Inv_W
                                    + model->B3SOIDDppvag * Inv_LW;
                  pParam->B3SOIDDdelta = model->B3SOIDDdelta
                                     + model->B3SOIDDldelta * Inv_L
                                     + model->B3SOIDDwdelta * Inv_W
                                     + model->B3SOIDDpdelta * Inv_LW;
                  pParam->B3SOIDDaii = model->B3SOIDDaii
                                     + model->B3SOIDDlaii * Inv_L
                                     + model->B3SOIDDwaii * Inv_W
                                     + model->B3SOIDDpaii * Inv_LW;
                  pParam->B3SOIDDbii = model->B3SOIDDbii
                                     + model->B3SOIDDlbii * Inv_L
                                     + model->B3SOIDDwbii * Inv_W
                                     + model->B3SOIDDpbii * Inv_LW;
                  pParam->B3SOIDDcii = model->B3SOIDDcii
                                     + model->B3SOIDDlcii * Inv_L
                                     + model->B3SOIDDwcii * Inv_W
                                     + model->B3SOIDDpcii * Inv_LW;
                  pParam->B3SOIDDdii = model->B3SOIDDdii
                                     + model->B3SOIDDldii * Inv_L
                                     + model->B3SOIDDwdii * Inv_W
                                     + model->B3SOIDDpdii * Inv_LW;
                  pParam->B3SOIDDalpha0 = model->B3SOIDDalpha0
                                      + model->B3SOIDDlalpha0 * Inv_L
                                      + model->B3SOIDDwalpha0 * Inv_W
                                      + model->B3SOIDDpalpha0 * Inv_LW;
                  pParam->B3SOIDDalpha1 = model->B3SOIDDalpha1
                                      + model->B3SOIDDlalpha1 * Inv_L
                                      + model->B3SOIDDwalpha1 * Inv_W
                                      + model->B3SOIDDpalpha1 * Inv_LW;
                  pParam->B3SOIDDbeta0 = model->B3SOIDDbeta0
                                     + model->B3SOIDDlbeta0 * Inv_L
                                     + model->B3SOIDDwbeta0 * Inv_W
                                     + model->B3SOIDDpbeta0 * Inv_LW;
                  pParam->B3SOIDDagidl = model->B3SOIDDagidl
                                      + model->B3SOIDDlagidl * Inv_L
                                      + model->B3SOIDDwagidl * Inv_W
                                      + model->B3SOIDDpagidl * Inv_LW;
                  pParam->B3SOIDDbgidl = model->B3SOIDDbgidl
                                      + model->B3SOIDDlbgidl * Inv_L
                                      + model->B3SOIDDwbgidl * Inv_W
                                      + model->B3SOIDDpbgidl * Inv_LW;
                  pParam->B3SOIDDngidl = model->B3SOIDDngidl
                                      + model->B3SOIDDlngidl * Inv_L
                                      + model->B3SOIDDwngidl * Inv_W
                                      + model->B3SOIDDpngidl * Inv_LW;
                  pParam->B3SOIDDntun = model->B3SOIDDntun
                                      + model->B3SOIDDlntun * Inv_L
                                      + model->B3SOIDDwntun * Inv_W
                                      + model->B3SOIDDpntun * Inv_LW;
                  pParam->B3SOIDDndiode = model->B3SOIDDndiode
                                      + model->B3SOIDDlndiode * Inv_L
                                      + model->B3SOIDDwndiode * Inv_W
                                      + model->B3SOIDDpndiode * Inv_LW;
                  pParam->B3SOIDDisbjt = model->B3SOIDDisbjt
                                  + model->B3SOIDDlisbjt * Inv_L
                                  + model->B3SOIDDwisbjt * Inv_W
                                  + model->B3SOIDDpisbjt * Inv_LW;
                  pParam->B3SOIDDisdif = model->B3SOIDDisdif
                                  + model->B3SOIDDlisdif * Inv_L
                                  + model->B3SOIDDwisdif * Inv_W
                                  + model->B3SOIDDpisdif * Inv_LW;
                  pParam->B3SOIDDisrec = model->B3SOIDDisrec
                                  + model->B3SOIDDlisrec * Inv_L
                                  + model->B3SOIDDwisrec * Inv_W
                                  + model->B3SOIDDpisrec * Inv_LW;
                  pParam->B3SOIDDistun = model->B3SOIDDistun
                                  + model->B3SOIDDlistun * Inv_L
                                  + model->B3SOIDDwistun * Inv_W
                                  + model->B3SOIDDpistun * Inv_LW;
                  pParam->B3SOIDDedl = model->B3SOIDDedl
                                  + model->B3SOIDDledl * Inv_L
                                  + model->B3SOIDDwedl * Inv_W
                                  + model->B3SOIDDpedl * Inv_LW;
                  pParam->B3SOIDDkbjt1 = model->B3SOIDDkbjt1
                                  + model->B3SOIDDlkbjt1 * Inv_L
                                  + model->B3SOIDDwkbjt1 * Inv_W
                                  + model->B3SOIDDpkbjt1 * Inv_LW;
                  /* CV model */
                  pParam->B3SOIDDvsdfb = model->B3SOIDDvsdfb
                                  + model->B3SOIDDlvsdfb * Inv_L
                                  + model->B3SOIDDwvsdfb * Inv_W
                                  + model->B3SOIDDpvsdfb * Inv_LW;
                  pParam->B3SOIDDvsdth = model->B3SOIDDvsdth
                                  + model->B3SOIDDlvsdth * Inv_L
                                  + model->B3SOIDDwvsdth * Inv_W
                                  + model->B3SOIDDpvsdth * Inv_LW;
/* Added for binning - END */

                  T0 = (TRatio - 1.0);

                  pParam->B3SOIDDuatemp = pParam->B3SOIDDua;  /*  save ua, ub, and uc for b3soiddld.c */
                  pParam->B3SOIDDubtemp = pParam->B3SOIDDub;
                  pParam->B3SOIDDuctemp = pParam->B3SOIDDuc;
                  pParam->B3SOIDDrds0denom = pow(pParam->B3SOIDDweff * 1E6, pParam->B3SOIDDwr);
                  pParam->B3SOIDDrth = here->B3SOIDDrth0 * sqrt(model->B3SOIDDtbox
                      / model->B3SOIDDtsi) / pParam->B3SOIDDweff;
                  pParam->B3SOIDDcth = here->B3SOIDDcth0 * model->B3SOIDDtsi;
                  pParam->B3SOIDDrbody = model->B3SOIDDrbody *
                                     pParam->B3SOIDDweff / pParam->B3SOIDDleff;
                  pParam->B3SOIDDua = pParam->B3SOIDDua + pParam->B3SOIDDua1 * T0;
                  pParam->B3SOIDDub = pParam->B3SOIDDub + pParam->B3SOIDDub1 * T0;
                  pParam->B3SOIDDuc = pParam->B3SOIDDuc + pParam->B3SOIDDuc1 * T0;
                  if (pParam->B3SOIDDu0 > 1.0)
                      pParam->B3SOIDDu0 = pParam->B3SOIDDu0 / 1.0e4;

                  pParam->B3SOIDDu0temp = pParam->B3SOIDDu0
                                      * pow(TRatio, pParam->B3SOIDDute);
                  pParam->B3SOIDDvsattemp = pParam->B3SOIDDvsat - pParam->B3SOIDDat
                                        * T0;
                  pParam->B3SOIDDrds0 = (pParam->B3SOIDDrdsw + pParam->B3SOIDDprt * T0)
                                    / pow(pParam->B3SOIDDweff * 1E6, pParam->B3SOIDDwr);

                  if (B3SOIDDcheckModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during B3SOIDDV3 parameter checking for %s in model %s", model->B3SOIDDmodName, here->B3SOIDDname);
                      return(E_BADPARM);
                  }


                  pParam->B3SOIDDcgdo = (model->B3SOIDDcgdo + pParam->B3SOIDDcf)
                                    * pParam->B3SOIDDweffCV;
                  pParam->B3SOIDDcgso = (model->B3SOIDDcgso + pParam->B3SOIDDcf)
                                    * pParam->B3SOIDDweffCV;
                  pParam->B3SOIDDcgeo = model->B3SOIDDcgeo
                                    * pParam->B3SOIDDleffCV;


                  if (!model->B3SOIDDnpeakGiven && model->B3SOIDDgamma1Given)
                  {   T0 = pParam->B3SOIDDgamma1 * model->B3SOIDDcox;
                      pParam->B3SOIDDnpeak = 3.021E22 * T0 * T0;
                  }

                  T0 = pow(TRatio, model->B3SOIDDxbjt / pParam->B3SOIDDndiode);
                  T1 = pow(TRatio, model->B3SOIDDxdif / pParam->B3SOIDDndiode);
                  T2 = pow(TRatio, model->B3SOIDDxrec / pParam->B3SOIDDndiode / 2);
                  T4 = -Eg0 / pParam->B3SOIDDndiode / model->B3SOIDDvtm * (1 - TRatio);
                  T5 = exp(T4);
                  T6 = sqrt(T5);
                  pParam->B3SOIDDjbjt = pParam->B3SOIDDisbjt * T0 * T5;
                  pParam->B3SOIDDjdif = pParam->B3SOIDDisdif * T1 * T5;
                  pParam->B3SOIDDjrec = pParam->B3SOIDDisrec * T2 * T6;
                  T0 = pow(TRatio, model->B3SOIDDxtun / pParam->B3SOIDDntun);
                  pParam->B3SOIDDjtun = pParam->B3SOIDDistun * T0 ;

                  if (pParam->B3SOIDDnsub > 0)
                     pParam->B3SOIDDvfbb = -model->B3SOIDDtype * model->B3SOIDDvtm *
                                log(pParam->B3SOIDDnpeak/ pParam->B3SOIDDnsub);
                  else
                     pParam->B3SOIDDvfbb = -model->B3SOIDDtype * model->B3SOIDDvtm *
                                log(-pParam->B3SOIDDnpeak* pParam->B3SOIDDnsub/ni/ni);

                  if (!model->B3SOIDDvsdfbGiven)
                  {
                     if (pParam->B3SOIDDnsub > 0)
                        pParam->B3SOIDDvsdfb = -model->B3SOIDDtype * (model->B3SOIDDvtm*log(1e20 *
                                            pParam->B3SOIDDnsub / ni /ni) - 0.3);
                     else if (pParam->B3SOIDDnsub < 0)
                        pParam->B3SOIDDvsdfb = -model->B3SOIDDtype * (model->B3SOIDDvtm*log(-1e20 /
                                            pParam->B3SOIDDnsub) + 0.3);
                  }

                  /* Phi  & Gamma */
                  SDphi = 2.0*model->B3SOIDDvtm*log(fabs(pParam->B3SOIDDnsub) / ni);
                  SDgamma = 5.753e-12 * sqrt(fabs(pParam->B3SOIDDnsub)) / model->B3SOIDDcbox;

                  if (!model->B3SOIDDvsdthGiven)
                  {
                     if ( ((pParam->B3SOIDDnsub > 0) && (model->B3SOIDDtype > 0)) ||
                          ((pParam->B3SOIDDnsub < 0) && (model->B3SOIDDtype < 0)) )
                        pParam->B3SOIDDvsdth = pParam->B3SOIDDvsdfb + SDphi +
                                            SDgamma * sqrt(SDphi);
                     else
                        pParam->B3SOIDDvsdth = pParam->B3SOIDDvsdfb - SDphi -
                                            SDgamma * sqrt(SDphi);
                  }
                  if (!model->B3SOIDDcsdminGiven)
                  {
                     /* Cdmin */
                     tmp = sqrt(2.0 * EPSSI * SDphi / (Charge_q *
                                fabs(pParam->B3SOIDDnsub) * 1.0e6));
                     tmp1 = EPSSI / tmp;
                     model->B3SOIDDcsdmin = tmp1 * model->B3SOIDDcbox /
                                          (tmp1 + model->B3SOIDDcbox);
                  }


                  T0 = model->B3SOIDDcsdesw * log(1 + model->B3SOIDDtsi /
                       model->B3SOIDDtbox);
                  T1 = here->B3SOIDDsourcePerimeter - pParam->B3SOIDDweff;
                  if (T1 > 0.0)
                     pParam->B3SOIDDcsesw = T0 * T1;
                  else
                     pParam->B3SOIDDcsesw = 0.0;
                  T1 = here->B3SOIDDdrainPerimeter - pParam->B3SOIDDweff;
                  if (T1 > 0.0)
                     pParam->B3SOIDDcdesw = T0 * T1;
                  else
                     pParam->B3SOIDDcdesw = 0.0;

                  pParam->B3SOIDDphi = 2.0 * model->B3SOIDDvtm
                                   * log(pParam->B3SOIDDnpeak / ni);

                  pParam->B3SOIDDsqrtPhi = sqrt(pParam->B3SOIDDphi);
                  pParam->B3SOIDDphis3 = pParam->B3SOIDDsqrtPhi * pParam->B3SOIDDphi;

                  pParam->B3SOIDDXdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->B3SOIDDnpeak * 1.0e6))
                                     * pParam->B3SOIDDsqrtPhi;
                  pParam->B3SOIDDsqrtXdep0 = sqrt(pParam->B3SOIDDXdep0);
                  pParam->B3SOIDDlitl = sqrt(3.0 * model->B3SOIDDxj
                                    * model->B3SOIDDtox);
                  pParam->B3SOIDDvbi = model->B3SOIDDvtm * log(1.0e20
                                   * pParam->B3SOIDDnpeak / (ni * ni));
                  pParam->B3SOIDDcdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->B3SOIDDnpeak * 1.0e6 / 2.0
                                     / pParam->B3SOIDDphi);

                  if (model->B3SOIDDk1Given || model->B3SOIDDk2Given)
                  {   if (!model->B3SOIDDk1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->B3SOIDDk1 = 0.53;
                      }
                      if (!model->B3SOIDDk2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->B3SOIDDk2 = -0.0186;
                      }
                      if (model->B3SOIDDxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIDDvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIDDvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIDDgamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIDDgamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->B3SOIDDvbxGiven)
                          pParam->B3SOIDDvbx = pParam->B3SOIDDphi - 7.7348e-4
                                           * pParam->B3SOIDDnpeak
                                           * pParam->B3SOIDDxt * pParam->B3SOIDDxt;
                      if (pParam->B3SOIDDvbx > 0.0)
                          pParam->B3SOIDDvbx = -pParam->B3SOIDDvbx;
                      if (pParam->B3SOIDDvbm > 0.0)
                          pParam->B3SOIDDvbm = -pParam->B3SOIDDvbm;

                      if (!model->B3SOIDDgamma1Given)
                          pParam->B3SOIDDgamma1 = 5.753e-12
                                              * sqrt(pParam->B3SOIDDnpeak)
                                              / model->B3SOIDDcox;
                      if (!model->B3SOIDDgamma2Given)
                          pParam->B3SOIDDgamma2 = 5.753e-12
                                              * sqrt(pParam->B3SOIDDnsub)
                                              / model->B3SOIDDcox;

                      T0 = pParam->B3SOIDDgamma1 - pParam->B3SOIDDgamma2;
                      T1 = sqrt(pParam->B3SOIDDphi - pParam->B3SOIDDvbx)
                         - pParam->B3SOIDDsqrtPhi;
                      T2 = sqrt(pParam->B3SOIDDphi * (pParam->B3SOIDDphi
                         - pParam->B3SOIDDvbm)) - pParam->B3SOIDDphi;
                      pParam->B3SOIDDk2 = T0 * T1 / (2.0 * T2 + pParam->B3SOIDDvbm);
                      pParam->B3SOIDDk1 = pParam->B3SOIDDgamma2 - 2.0
                                      * pParam->B3SOIDDk2 * sqrt(pParam->B3SOIDDphi
                                      - pParam->B3SOIDDvbm);
                  }

                  if (pParam->B3SOIDDk2 < 0.0)
                  {   T0 = 0.5 * pParam->B3SOIDDk1 / pParam->B3SOIDDk2;
                      pParam->B3SOIDDvbsc = 0.9 * (pParam->B3SOIDDphi - T0 * T0);
                      if (pParam->B3SOIDDvbsc > -3.0)
                          pParam->B3SOIDDvbsc = -3.0;
                      else if (pParam->B3SOIDDvbsc < -30.0)
                          pParam->B3SOIDDvbsc = -30.0;
                  }
                  else
                  {   pParam->B3SOIDDvbsc = -30.0;
                  }
                  if (pParam->B3SOIDDvbsc > pParam->B3SOIDDvbm)
                      pParam->B3SOIDDvbsc = pParam->B3SOIDDvbm;

                  if (model->B3SOIDDvth0Given)
                  {   pParam->B3SOIDDvfb = model->B3SOIDDtype * pParam->B3SOIDDvth0
                                       - pParam->B3SOIDDphi - pParam->B3SOIDDk1
                                       * pParam->B3SOIDDsqrtPhi;
                  }
                  else
                  {   pParam->B3SOIDDvfb = -1.0;
                      pParam->B3SOIDDvth0 = model->B3SOIDDtype * (pParam->B3SOIDDvfb
                                        + pParam->B3SOIDDphi + pParam->B3SOIDDk1
                                        * pParam->B3SOIDDsqrtPhi);
                  }
                  T1 = sqrt(EPSSI / EPSOX * model->B3SOIDDtox
                     * pParam->B3SOIDDXdep0);
                  T0 = exp(-0.5 * pParam->B3SOIDDdsub * pParam->B3SOIDDleff / T1);
                  pParam->B3SOIDDtheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->B3SOIDDdrout * pParam->B3SOIDDleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->B3SOIDDthetaRout = pParam->B3SOIDDpdibl1 * T2
                                         + pParam->B3SOIDDpdibl2;

                  here->B3SOIDDminIsub = 5.0e-2 * pParam->B3SOIDDweff * model->B3SOIDDtsi
                                     * MAX(pParam->B3SOIDDisdif, pParam->B3SOIDDisrec);
              }

              here->B3SOIDDcsbox = model->B3SOIDDcbox*here->B3SOIDDsourceArea;
              here->B3SOIDDcsmin = model->B3SOIDDcsdmin*here->B3SOIDDsourceArea;
              here->B3SOIDDcdbox = model->B3SOIDDcbox*here->B3SOIDDdrainArea;
              here->B3SOIDDcdmin = model->B3SOIDDcsdmin*here->B3SOIDDdrainArea;

              if ( ((pParam->B3SOIDDnsub > 0) && (model->B3SOIDDtype > 0)) ||
                   ((pParam->B3SOIDDnsub < 0) && (model->B3SOIDDtype < 0)) )
              {
                 T0 = pParam->B3SOIDDvsdth - pParam->B3SOIDDvsdfb;
                 pParam->B3SOIDDsdt1 = pParam->B3SOIDDvsdfb + model->B3SOIDDasd * T0;
                 T1 = here->B3SOIDDcsbox - here->B3SOIDDcsmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIDDst2 = T2 / model->B3SOIDDasd;
                 pParam->B3SOIDDst3 = T2 /( 1 - model->B3SOIDDasd);
                 here->B3SOIDDst4 =  T0 * T1 * (1 + model->B3SOIDDasd) / 3
                                  - here->B3SOIDDcsmin * pParam->B3SOIDDvsdfb;

                 T1 = here->B3SOIDDcdbox - here->B3SOIDDcdmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIDDdt2 = T2 / model->B3SOIDDasd;
                 pParam->B3SOIDDdt3 = T2 /( 1 - model->B3SOIDDasd);
                 here->B3SOIDDdt4 =  T0 * T1 * (1 + model->B3SOIDDasd) / 3
                                  - here->B3SOIDDcdmin * pParam->B3SOIDDvsdfb;
              } else
              {
                 T0 = pParam->B3SOIDDvsdfb - pParam->B3SOIDDvsdth;
                 pParam->B3SOIDDsdt1 = pParam->B3SOIDDvsdth + model->B3SOIDDasd * T0;
                 T1 = here->B3SOIDDcsmin - here->B3SOIDDcsbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIDDst2 = T2 / model->B3SOIDDasd;
                 pParam->B3SOIDDst3 = T2 /( 1 - model->B3SOIDDasd);
                 here->B3SOIDDst4 =  T0 * T1 * (1 + model->B3SOIDDasd) / 3
                                  - here->B3SOIDDcsbox * pParam->B3SOIDDvsdth;

                 T1 = here->B3SOIDDcdmin - here->B3SOIDDcdbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIDDdt2 = T2 / model->B3SOIDDasd;
                 pParam->B3SOIDDdt3 = T2 /( 1 - model->B3SOIDDasd);
                 here->B3SOIDDdt4 =  T0 * T1 * (1 + model->B3SOIDDasd) / 3
                                  - here->B3SOIDDcdbox * pParam->B3SOIDDvsdth;
              }

              here->B3SOIDDphi = pParam->B3SOIDDphi;
              /* process source/drain series resistance */
              here->B3SOIDDdrainConductance = model->B3SOIDDsheetResistance
                                              * here->B3SOIDDdrainSquares;
              if (here->B3SOIDDdrainConductance > 0.0)
                  here->B3SOIDDdrainConductance = 1.0
                                              / here->B3SOIDDdrainConductance;
              else
                  here->B3SOIDDdrainConductance = 0.0;

              here->B3SOIDDsourceConductance = model->B3SOIDDsheetResistance
                                           * here->B3SOIDDsourceSquares;
              if (here->B3SOIDDsourceConductance > 0.0)
                  here->B3SOIDDsourceConductance = 1.0
                                               / here->B3SOIDDsourceConductance;
              else
                  here->B3SOIDDsourceConductance = 0.0;
              here->B3SOIDDcgso = pParam->B3SOIDDcgso;
              here->B3SOIDDcgdo = pParam->B3SOIDDcgdo;

         }
    }
    return(OK);
}

