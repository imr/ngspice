/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdtemp.c          98/5/01
Modified by Pin Su, Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su
 * BSIMFD2.1 release
 */

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
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
B3SOIFDtemp(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIFDmodel *model = (B3SOIFDmodel*) inModel;
B3SOIFDinstance *here;
struct b3soifdSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, T6, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double SDphi, SDgamma;
int Size_Not_Found;

    /*  loop through all the B3SOIFD device models */
    for (; model != NULL; model = B3SOIFDnextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->B3SOIFDGatesidewallJctPotential < 0.1)
             model->B3SOIFDGatesidewallJctPotential = 0.1;

         struct b3soifdSizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct b3soifdSizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->B3SOIFDtnom;
         TRatio = Temp / Tnom;

         model->B3SOIFDvcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->B3SOIFDfactor1 = sqrt(EPSSI / EPSOX * model->B3SOIFDtox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         model->B3SOIFDeg0 = Eg0;
         model->B3SOIFDvtm = KboQ * Temp;

         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         /* ni is in cm^-3 */
         ni = 1.45e10 * (Temp / 300.15) * sqrt(Temp / 300.15)
            * exp(21.5565981 - Eg / (2.0 * model->B3SOIFDvtm));


         /* loop through all the instances of the model */
         /* MCJ: Length and Width not initialized */
         for (here = B3SOIFDinstances(model); here != NULL;
              here = B3SOIFDnextInstance(here))
         {
              here->B3SOIFDrbodyext = here->B3SOIFDbodySquares *
                                    model->B3SOIFDrbsh;
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->B3SOIFDl == pSizeDependParamKnot->Length)
                      && (here->B3SOIFDw == pSizeDependParamKnot->Width)
                      && (here->B3SOIFDrth0 == pSizeDependParamKnot->Rth0)
                      && (here->B3SOIFDcth0 == pSizeDependParamKnot->Cth0))
                  {   Size_Not_Found = 0;
                      here->pParam = pSizeDependParamKnot;
                  }
                  else
                  {   pLastKnot = pSizeDependParamKnot;
                      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
                  }
              }

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct b3soifdSizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  Ldrn = here->B3SOIFDl;
                  Wdrn = here->B3SOIFDw;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
                  pParam->Rth0 = here->B3SOIFDrth0;
                  pParam->Cth0 = here->B3SOIFDcth0;

                  T0 = pow(Ldrn, model->B3SOIFDLln);
                  T1 = pow(Wdrn, model->B3SOIFDLwn);
                  tmp1 = model->B3SOIFDLl / T0 + model->B3SOIFDLw / T1
                       + model->B3SOIFDLwl / (T0 * T1);
                  pParam->B3SOIFDdl = model->B3SOIFDLint + tmp1;
                  pParam->B3SOIFDdlc = model->B3SOIFDdlc + tmp1;

                  T2 = pow(Ldrn, model->B3SOIFDWln);
                  T3 = pow(Wdrn, model->B3SOIFDWwn);
                  tmp2 = model->B3SOIFDWl / T2 + model->B3SOIFDWw / T3
                       + model->B3SOIFDWwl / (T2 * T3);
                  pParam->B3SOIFDdw = model->B3SOIFDWint + tmp2;
                  pParam->B3SOIFDdwc = model->B3SOIFDdwc + tmp2;

                  pParam->B3SOIFDleff = here->B3SOIFDl - 2.0 * pParam->B3SOIFDdl;
                  if (pParam->B3SOIFDleff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIFD: mosfet %s, model %s: Effective channel length <= 0",
                       model->B3SOIFDmodName, here->B3SOIFDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIFDweff = here->B3SOIFDw - 2.0 * pParam->B3SOIFDdw;
                  if (pParam->B3SOIFDweff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIFD: mosfet %s, model %s: Effective channel width <= 0",
                       model->B3SOIFDmodName, here->B3SOIFDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIFDleffCV = here->B3SOIFDl - 2.0 * pParam->B3SOIFDdlc;
                  if (pParam->B3SOIFDleffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIFD: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->B3SOIFDmodName, here->B3SOIFDname);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIFDweffCV = here->B3SOIFDw - 2.0 * pParam->B3SOIFDdwc;
                  if (pParam->B3SOIFDweffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "B3SOIFD: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->B3SOIFDmodName, here->B3SOIFDname);
                      return(E_BADPARM);
                  }

                  /* Not binned - START */
                  pParam->B3SOIFDat = model->B3SOIFDat;
                  pParam->B3SOIFDgamma1 = model->B3SOIFDgamma1;
                  pParam->B3SOIFDgamma2 = model->B3SOIFDgamma2;
                  pParam->B3SOIFDvbx = model->B3SOIFDvbx;
                  pParam->B3SOIFDvbm = model->B3SOIFDvbm;
                  pParam->B3SOIFDxt = model->B3SOIFDxt;
                  pParam->B3SOIFDkt1 = model->B3SOIFDkt1;
                  pParam->B3SOIFDkt1l = model->B3SOIFDkt1l;
                  pParam->B3SOIFDkt2 = model->B3SOIFDkt2;
                  pParam->B3SOIFDua1 = model->B3SOIFDua1;
                  pParam->B3SOIFDub1 = model->B3SOIFDub1;
                  pParam->B3SOIFDuc1 = model->B3SOIFDuc1;
                  pParam->B3SOIFDute = model->B3SOIFDute;
                  pParam->B3SOIFDprt = model->B3SOIFDprt;
                  /* Not binned - END */

                  /* CV model */
                  pParam->B3SOIFDcgsl = model->B3SOIFDcgsl;
                  pParam->B3SOIFDcgdl = model->B3SOIFDcgdl;
                  pParam->B3SOIFDckappa = model->B3SOIFDckappa;
                  pParam->B3SOIFDcf = model->B3SOIFDcf;
                  pParam->B3SOIFDclc = model->B3SOIFDclc;
                  pParam->B3SOIFDcle = model->B3SOIFDcle;

                  pParam->B3SOIFDabulkCVfactor = pow(1.0+(pParam->B3SOIFDclc
                                             / pParam->B3SOIFDleff),
                                             pParam->B3SOIFDcle);

/* Added for binning - START */
                  if (model->B3SOIFDbinUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->B3SOIFDleff;
                      Inv_W = 1.0e-6 / pParam->B3SOIFDweff;
                      Inv_LW = 1.0e-12 / (pParam->B3SOIFDleff
                             * pParam->B3SOIFDweff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->B3SOIFDleff;
                      Inv_W = 1.0 / pParam->B3SOIFDweff;
                      Inv_LW = 1.0 / (pParam->B3SOIFDleff
                             * pParam->B3SOIFDweff);
                  }
                  pParam->B3SOIFDnpeak = model->B3SOIFDnpeak
                                     + model->B3SOIFDlnpeak * Inv_L
                                     + model->B3SOIFDwnpeak * Inv_W
                                     + model->B3SOIFDpnpeak * Inv_LW;
                  pParam->B3SOIFDnsub = model->B3SOIFDnsub
                                    + model->B3SOIFDlnsub * Inv_L
                                    + model->B3SOIFDwnsub * Inv_W
                                    + model->B3SOIFDpnsub * Inv_LW;
                  pParam->B3SOIFDngate = model->B3SOIFDngate
                                     + model->B3SOIFDlngate * Inv_L
                                     + model->B3SOIFDwngate * Inv_W
                                     + model->B3SOIFDpngate * Inv_LW;
                  pParam->B3SOIFDvth0 = model->B3SOIFDvth0
                                    + model->B3SOIFDlvth0 * Inv_L
                                    + model->B3SOIFDwvth0 * Inv_W
                                    + model->B3SOIFDpvth0 * Inv_LW;
                  pParam->B3SOIFDk1 = model->B3SOIFDk1
                                  + model->B3SOIFDlk1 * Inv_L
                                  + model->B3SOIFDwk1 * Inv_W
                                  + model->B3SOIFDpk1 * Inv_LW;
                  pParam->B3SOIFDk2 = model->B3SOIFDk2
                                  + model->B3SOIFDlk2 * Inv_L
                                  + model->B3SOIFDwk2 * Inv_W
                                  + model->B3SOIFDpk2 * Inv_LW;
                  pParam->B3SOIFDk3 = model->B3SOIFDk3
                                  + model->B3SOIFDlk3 * Inv_L
                                  + model->B3SOIFDwk3 * Inv_W
                                  + model->B3SOIFDpk3 * Inv_LW;
                  pParam->B3SOIFDk3b = model->B3SOIFDk3b
                                   + model->B3SOIFDlk3b * Inv_L
                                   + model->B3SOIFDwk3b * Inv_W
                                   + model->B3SOIFDpk3b * Inv_LW;
                  pParam->B3SOIFDvbsa = model->B3SOIFDvbsa
                                   + model->B3SOIFDlvbsa * Inv_L
                                   + model->B3SOIFDwvbsa * Inv_W
                                   + model->B3SOIFDpvbsa * Inv_LW;
                  pParam->B3SOIFDdelp = model->B3SOIFDdelp
                                   + model->B3SOIFDldelp * Inv_L
                                   + model->B3SOIFDwdelp * Inv_W
                                   + model->B3SOIFDpdelp * Inv_LW;
                  pParam->B3SOIFDkb1 = model->B3SOIFDkb1
                                   + model->B3SOIFDlkb1 * Inv_L
                                   + model->B3SOIFDwkb1 * Inv_W
                                   + model->B3SOIFDpkb1 * Inv_LW;
                  pParam->B3SOIFDkb3 = model->B3SOIFDkb3
                                   + model->B3SOIFDlkb3 * Inv_L
                                   + model->B3SOIFDwkb3 * Inv_W
                                   + model->B3SOIFDpkb3 * Inv_LW;
                  pParam->B3SOIFDdvbd0 = model->B3SOIFDdvbd0
                                   + model->B3SOIFDldvbd0 * Inv_L
                                   + model->B3SOIFDwdvbd0 * Inv_W
                                   + model->B3SOIFDpdvbd0 * Inv_LW;
                  pParam->B3SOIFDdvbd1 = model->B3SOIFDdvbd1
                                   + model->B3SOIFDldvbd1 * Inv_L
                                   + model->B3SOIFDwdvbd1 * Inv_W
                                   + model->B3SOIFDpdvbd1 * Inv_LW;
                  pParam->B3SOIFDw0 = model->B3SOIFDw0
                                  + model->B3SOIFDlw0 * Inv_L
                                  + model->B3SOIFDww0 * Inv_W
                                  + model->B3SOIFDpw0 * Inv_LW;
                  pParam->B3SOIFDnlx = model->B3SOIFDnlx
                                   + model->B3SOIFDlnlx * Inv_L
                                   + model->B3SOIFDwnlx * Inv_W
                                   + model->B3SOIFDpnlx * Inv_LW;
                  pParam->B3SOIFDdvt0 = model->B3SOIFDdvt0
                                    + model->B3SOIFDldvt0 * Inv_L
                                    + model->B3SOIFDwdvt0 * Inv_W
                                    + model->B3SOIFDpdvt0 * Inv_LW;
                  pParam->B3SOIFDdvt1 = model->B3SOIFDdvt1
                                    + model->B3SOIFDldvt1 * Inv_L
                                    + model->B3SOIFDwdvt1 * Inv_W
                                    + model->B3SOIFDpdvt1 * Inv_LW;
                  pParam->B3SOIFDdvt2 = model->B3SOIFDdvt2
                                    + model->B3SOIFDldvt2 * Inv_L
                                    + model->B3SOIFDwdvt2 * Inv_W
                                    + model->B3SOIFDpdvt2 * Inv_LW;
                  pParam->B3SOIFDdvt0w = model->B3SOIFDdvt0w
                                    + model->B3SOIFDldvt0w * Inv_L
                                    + model->B3SOIFDwdvt0w * Inv_W
                                    + model->B3SOIFDpdvt0w * Inv_LW;
                  pParam->B3SOIFDdvt1w = model->B3SOIFDdvt1w
                                    + model->B3SOIFDldvt1w * Inv_L
                                    + model->B3SOIFDwdvt1w * Inv_W
                                    + model->B3SOIFDpdvt1w * Inv_LW;
                  pParam->B3SOIFDdvt2w = model->B3SOIFDdvt2w
                                    + model->B3SOIFDldvt2w * Inv_L
                                    + model->B3SOIFDwdvt2w * Inv_W
                                    + model->B3SOIFDpdvt2w * Inv_LW;
                  pParam->B3SOIFDu0 = model->B3SOIFDu0
                                  + model->B3SOIFDlu0 * Inv_L
                                  + model->B3SOIFDwu0 * Inv_W
                                  + model->B3SOIFDpu0 * Inv_LW;
                  pParam->B3SOIFDua = model->B3SOIFDua
                                  + model->B3SOIFDlua * Inv_L
                                  + model->B3SOIFDwua * Inv_W
                                  + model->B3SOIFDpua * Inv_LW;
                  pParam->B3SOIFDub = model->B3SOIFDub
                                  + model->B3SOIFDlub * Inv_L
                                  + model->B3SOIFDwub * Inv_W
                                  + model->B3SOIFDpub * Inv_LW;
                  pParam->B3SOIFDuc = model->B3SOIFDuc
                                  + model->B3SOIFDluc * Inv_L
                                  + model->B3SOIFDwuc * Inv_W
                                  + model->B3SOIFDpuc * Inv_LW;
                  pParam->B3SOIFDvsat = model->B3SOIFDvsat
                                    + model->B3SOIFDlvsat * Inv_L
                                    + model->B3SOIFDwvsat * Inv_W
                                    + model->B3SOIFDpvsat * Inv_LW;
                  pParam->B3SOIFDa0 = model->B3SOIFDa0
                                  + model->B3SOIFDla0 * Inv_L
                                  + model->B3SOIFDwa0 * Inv_W
                                  + model->B3SOIFDpa0 * Inv_LW;
                  pParam->B3SOIFDags = model->B3SOIFDags
                                  + model->B3SOIFDlags * Inv_L
                                  + model->B3SOIFDwags * Inv_W
                                  + model->B3SOIFDpags * Inv_LW;
                  pParam->B3SOIFDb0 = model->B3SOIFDb0
                                  + model->B3SOIFDlb0 * Inv_L
                                  + model->B3SOIFDwb0 * Inv_W
                                  + model->B3SOIFDpb0 * Inv_LW;
                  pParam->B3SOIFDb1 = model->B3SOIFDb1
                                  + model->B3SOIFDlb1 * Inv_L
                                  + model->B3SOIFDwb1 * Inv_W
                                  + model->B3SOIFDpb1 * Inv_LW;
                  pParam->B3SOIFDketa = model->B3SOIFDketa
                                    + model->B3SOIFDlketa * Inv_L
                                    + model->B3SOIFDwketa * Inv_W
                                    + model->B3SOIFDpketa * Inv_LW;
                  pParam->B3SOIFDabp = model->B3SOIFDabp
                                  + model->B3SOIFDlabp * Inv_L
                                  + model->B3SOIFDwabp * Inv_W
                                  + model->B3SOIFDpabp * Inv_LW;
                  pParam->B3SOIFDmxc = model->B3SOIFDmxc
                                  + model->B3SOIFDlmxc * Inv_L
                                  + model->B3SOIFDwmxc * Inv_W
                                  + model->B3SOIFDpmxc * Inv_LW;
                  pParam->B3SOIFDadice0 = model->B3SOIFDadice0
                                  + model->B3SOIFDladice0 * Inv_L
                                  + model->B3SOIFDwadice0 * Inv_W
                                  + model->B3SOIFDpadice0 * Inv_LW;
                  pParam->B3SOIFDa1 = model->B3SOIFDa1
                                  + model->B3SOIFDla1 * Inv_L
                                  + model->B3SOIFDwa1 * Inv_W
                                  + model->B3SOIFDpa1 * Inv_LW;
                  pParam->B3SOIFDa2 = model->B3SOIFDa2
                                  + model->B3SOIFDla2 * Inv_L
                                  + model->B3SOIFDwa2 * Inv_W
                                  + model->B3SOIFDpa2 * Inv_LW;
                  pParam->B3SOIFDrdsw = model->B3SOIFDrdsw
                                    + model->B3SOIFDlrdsw * Inv_L
                                    + model->B3SOIFDwrdsw * Inv_W
                                    + model->B3SOIFDprdsw * Inv_LW;
                  pParam->B3SOIFDprwb = model->B3SOIFDprwb
                                    + model->B3SOIFDlprwb * Inv_L
                                    + model->B3SOIFDwprwb * Inv_W
                                    + model->B3SOIFDpprwb * Inv_LW;
                  pParam->B3SOIFDprwg = model->B3SOIFDprwg
                                    + model->B3SOIFDlprwg * Inv_L
                                    + model->B3SOIFDwprwg * Inv_W
                                    + model->B3SOIFDpprwg * Inv_LW;
                  pParam->B3SOIFDwr = model->B3SOIFDwr
                                  + model->B3SOIFDlwr * Inv_L
                                  + model->B3SOIFDwwr * Inv_W
                                  + model->B3SOIFDpwr * Inv_LW;
                  pParam->B3SOIFDnfactor = model->B3SOIFDnfactor
                                       + model->B3SOIFDlnfactor * Inv_L
                                       + model->B3SOIFDwnfactor * Inv_W
                                       + model->B3SOIFDpnfactor * Inv_LW;
                  pParam->B3SOIFDdwg = model->B3SOIFDdwg
                                   + model->B3SOIFDldwg * Inv_L
                                   + model->B3SOIFDwdwg * Inv_W
                                   + model->B3SOIFDpdwg * Inv_LW;
                  pParam->B3SOIFDdwb = model->B3SOIFDdwb
                                   + model->B3SOIFDldwb * Inv_L
                                   + model->B3SOIFDwdwb * Inv_W
                                   + model->B3SOIFDpdwb * Inv_LW;
                  pParam->B3SOIFDvoff = model->B3SOIFDvoff
                                    + model->B3SOIFDlvoff * Inv_L
                                    + model->B3SOIFDwvoff * Inv_W
                                    + model->B3SOIFDpvoff * Inv_LW;
                  pParam->B3SOIFDeta0 = model->B3SOIFDeta0
                                    + model->B3SOIFDleta0 * Inv_L
                                    + model->B3SOIFDweta0 * Inv_W
                                    + model->B3SOIFDpeta0 * Inv_LW;
                  pParam->B3SOIFDetab = model->B3SOIFDetab
                                    + model->B3SOIFDletab * Inv_L
                                    + model->B3SOIFDwetab * Inv_W
                                    + model->B3SOIFDpetab * Inv_LW;
                  pParam->B3SOIFDdsub = model->B3SOIFDdsub
                                    + model->B3SOIFDldsub * Inv_L
                                    + model->B3SOIFDwdsub * Inv_W
                                    + model->B3SOIFDpdsub * Inv_LW;
                  pParam->B3SOIFDcit = model->B3SOIFDcit
                                   + model->B3SOIFDlcit * Inv_L
                                   + model->B3SOIFDwcit * Inv_W
                                   + model->B3SOIFDpcit * Inv_LW;
                  pParam->B3SOIFDcdsc = model->B3SOIFDcdsc
                                    + model->B3SOIFDlcdsc * Inv_L
                                    + model->B3SOIFDwcdsc * Inv_W
                                    + model->B3SOIFDpcdsc * Inv_LW;
                  pParam->B3SOIFDcdscb = model->B3SOIFDcdscb
                                     + model->B3SOIFDlcdscb * Inv_L
                                     + model->B3SOIFDwcdscb * Inv_W
                                     + model->B3SOIFDpcdscb * Inv_LW;
                      pParam->B3SOIFDcdscd = model->B3SOIFDcdscd
                                     + model->B3SOIFDlcdscd * Inv_L
                                     + model->B3SOIFDwcdscd * Inv_W
                                     + model->B3SOIFDpcdscd * Inv_LW;
                  pParam->B3SOIFDpclm = model->B3SOIFDpclm
                                    + model->B3SOIFDlpclm * Inv_L
                                    + model->B3SOIFDwpclm * Inv_W
                                    + model->B3SOIFDppclm * Inv_LW;
                  pParam->B3SOIFDpdibl1 = model->B3SOIFDpdibl1
                                      + model->B3SOIFDlpdibl1 * Inv_L
                                      + model->B3SOIFDwpdibl1 * Inv_W
                                      + model->B3SOIFDppdibl1 * Inv_LW;
                  pParam->B3SOIFDpdibl2 = model->B3SOIFDpdibl2
                                      + model->B3SOIFDlpdibl2 * Inv_L
                                      + model->B3SOIFDwpdibl2 * Inv_W
                                      + model->B3SOIFDppdibl2 * Inv_LW;
                  pParam->B3SOIFDpdiblb = model->B3SOIFDpdiblb
                                      + model->B3SOIFDlpdiblb * Inv_L
                                      + model->B3SOIFDwpdiblb * Inv_W
                                      + model->B3SOIFDppdiblb * Inv_LW;
                  pParam->B3SOIFDdrout = model->B3SOIFDdrout
                                     + model->B3SOIFDldrout * Inv_L
                                     + model->B3SOIFDwdrout * Inv_W
                                     + model->B3SOIFDpdrout * Inv_LW;
                  pParam->B3SOIFDpvag = model->B3SOIFDpvag
                                    + model->B3SOIFDlpvag * Inv_L
                                    + model->B3SOIFDwpvag * Inv_W
                                    + model->B3SOIFDppvag * Inv_LW;
                  pParam->B3SOIFDdelta = model->B3SOIFDdelta
                                     + model->B3SOIFDldelta * Inv_L
                                     + model->B3SOIFDwdelta * Inv_W
                                     + model->B3SOIFDpdelta * Inv_LW;
                  pParam->B3SOIFDaii = model->B3SOIFDaii
                                     + model->B3SOIFDlaii * Inv_L
                                     + model->B3SOIFDwaii * Inv_W
                                     + model->B3SOIFDpaii * Inv_LW;
                  pParam->B3SOIFDbii = model->B3SOIFDbii
                                     + model->B3SOIFDlbii * Inv_L
                                     + model->B3SOIFDwbii * Inv_W
                                     + model->B3SOIFDpbii * Inv_LW;
                  pParam->B3SOIFDcii = model->B3SOIFDcii
                                     + model->B3SOIFDlcii * Inv_L
                                     + model->B3SOIFDwcii * Inv_W
                                     + model->B3SOIFDpcii * Inv_LW;
                  pParam->B3SOIFDdii = model->B3SOIFDdii
                                     + model->B3SOIFDldii * Inv_L
                                     + model->B3SOIFDwdii * Inv_W
                                     + model->B3SOIFDpdii * Inv_LW;
                  pParam->B3SOIFDalpha0 = model->B3SOIFDalpha0
                                      + model->B3SOIFDlalpha0 * Inv_L
                                      + model->B3SOIFDwalpha0 * Inv_W
                                      + model->B3SOIFDpalpha0 * Inv_LW;
                  pParam->B3SOIFDalpha1 = model->B3SOIFDalpha1
                                      + model->B3SOIFDlalpha1 * Inv_L
                                      + model->B3SOIFDwalpha1 * Inv_W
                                      + model->B3SOIFDpalpha1 * Inv_LW;
                  pParam->B3SOIFDbeta0 = model->B3SOIFDbeta0
                                     + model->B3SOIFDlbeta0 * Inv_L
                                     + model->B3SOIFDwbeta0 * Inv_W
                                     + model->B3SOIFDpbeta0 * Inv_LW;
                  pParam->B3SOIFDagidl = model->B3SOIFDagidl
                                      + model->B3SOIFDlagidl * Inv_L
                                      + model->B3SOIFDwagidl * Inv_W
                                      + model->B3SOIFDpagidl * Inv_LW;
                  pParam->B3SOIFDbgidl = model->B3SOIFDbgidl
                                      + model->B3SOIFDlbgidl * Inv_L
                                      + model->B3SOIFDwbgidl * Inv_W
                                      + model->B3SOIFDpbgidl * Inv_LW;
                  pParam->B3SOIFDngidl = model->B3SOIFDngidl
                                      + model->B3SOIFDlngidl * Inv_L
                                      + model->B3SOIFDwngidl * Inv_W
                                      + model->B3SOIFDpngidl * Inv_LW;
                  pParam->B3SOIFDntun = model->B3SOIFDntun
                                      + model->B3SOIFDlntun * Inv_L
                                      + model->B3SOIFDwntun * Inv_W
                                      + model->B3SOIFDpntun * Inv_LW;
                  pParam->B3SOIFDndiode = model->B3SOIFDndiode
                                      + model->B3SOIFDlndiode * Inv_L
                                      + model->B3SOIFDwndiode * Inv_W
                                      + model->B3SOIFDpndiode * Inv_LW;
                  pParam->B3SOIFDisbjt = model->B3SOIFDisbjt
                                  + model->B3SOIFDlisbjt * Inv_L
                                  + model->B3SOIFDwisbjt * Inv_W
                                  + model->B3SOIFDpisbjt * Inv_LW;
                  pParam->B3SOIFDisdif = model->B3SOIFDisdif
                                  + model->B3SOIFDlisdif * Inv_L
                                  + model->B3SOIFDwisdif * Inv_W
                                  + model->B3SOIFDpisdif * Inv_LW;
                  pParam->B3SOIFDisrec = model->B3SOIFDisrec
                                  + model->B3SOIFDlisrec * Inv_L
                                  + model->B3SOIFDwisrec * Inv_W
                                  + model->B3SOIFDpisrec * Inv_LW;
                  pParam->B3SOIFDistun = model->B3SOIFDistun
                                  + model->B3SOIFDlistun * Inv_L
                                  + model->B3SOIFDwistun * Inv_W
                                  + model->B3SOIFDpistun * Inv_LW;
                  pParam->B3SOIFDedl = model->B3SOIFDedl
                                  + model->B3SOIFDledl * Inv_L
                                  + model->B3SOIFDwedl * Inv_W
                                  + model->B3SOIFDpedl * Inv_LW;
                  pParam->B3SOIFDkbjt1 = model->B3SOIFDkbjt1
                                  + model->B3SOIFDlkbjt1 * Inv_L
                                  + model->B3SOIFDwkbjt1 * Inv_W
                                  + model->B3SOIFDpkbjt1 * Inv_LW;
                  /* CV model */
                  pParam->B3SOIFDvsdfb = model->B3SOIFDvsdfb
                                  + model->B3SOIFDlvsdfb * Inv_L
                                  + model->B3SOIFDwvsdfb * Inv_W
                                  + model->B3SOIFDpvsdfb * Inv_LW;
                  pParam->B3SOIFDvsdth = model->B3SOIFDvsdth
                                  + model->B3SOIFDlvsdth * Inv_L
                                  + model->B3SOIFDwvsdth * Inv_W
                                  + model->B3SOIFDpvsdth * Inv_LW;
/* Added for binning - END */

                  T0 = (TRatio - 1.0);

                  pParam->B3SOIFDuatemp = pParam->B3SOIFDua;  /*  save ua, ub, and uc for b3soifdld.c */
                  pParam->B3SOIFDubtemp = pParam->B3SOIFDub;
                  pParam->B3SOIFDuctemp = pParam->B3SOIFDuc;
                  pParam->B3SOIFDrds0denom = pow(pParam->B3SOIFDweff * 1E6, pParam->B3SOIFDwr);
                  pParam->B3SOIFDrth = here->B3SOIFDrth0 * sqrt(model->B3SOIFDtbox
                      / model->B3SOIFDtsi) / pParam->B3SOIFDweff;
                  pParam->B3SOIFDcth = here->B3SOIFDcth0 * model->B3SOIFDtsi;
                  pParam->B3SOIFDrbody = model->B3SOIFDrbody *
                                     pParam->B3SOIFDweff / pParam->B3SOIFDleff;
                  pParam->B3SOIFDua = pParam->B3SOIFDua + pParam->B3SOIFDua1 * T0;
                  pParam->B3SOIFDub = pParam->B3SOIFDub + pParam->B3SOIFDub1 * T0;
                  pParam->B3SOIFDuc = pParam->B3SOIFDuc + pParam->B3SOIFDuc1 * T0;
                  if (pParam->B3SOIFDu0 > 1.0)
                      pParam->B3SOIFDu0 = pParam->B3SOIFDu0 / 1.0e4;

                  pParam->B3SOIFDu0temp = pParam->B3SOIFDu0
                                      * pow(TRatio, pParam->B3SOIFDute);
                  pParam->B3SOIFDvsattemp = pParam->B3SOIFDvsat - pParam->B3SOIFDat
                                        * T0;
                  pParam->B3SOIFDrds0 = (pParam->B3SOIFDrdsw + pParam->B3SOIFDprt * T0)
                                    / pow(pParam->B3SOIFDweff * 1E6, pParam->B3SOIFDwr);

                  if (B3SOIFDcheckModel(model, here, ckt))
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL, "Fatal error(s) detected during B3SOIFDV3 parameter checking for %s in model %s", model->B3SOIFDmodName, here->B3SOIFDname);
                      return(E_BADPARM);
                  }


                  pParam->B3SOIFDcgdo = (model->B3SOIFDcgdo + pParam->B3SOIFDcf)
                                    * pParam->B3SOIFDweffCV;
                  pParam->B3SOIFDcgso = (model->B3SOIFDcgso + pParam->B3SOIFDcf)
                                    * pParam->B3SOIFDweffCV;


                  pParam->B3SOIFDcgeo = model->B3SOIFDcgeo * pParam->B3SOIFDleffCV;


                  if (!model->B3SOIFDnpeakGiven && model->B3SOIFDgamma1Given)
                  {   T0 = pParam->B3SOIFDgamma1 * model->B3SOIFDcox;
                      pParam->B3SOIFDnpeak = 3.021E22 * T0 * T0;
                  }

                  T0 = pow(TRatio, model->B3SOIFDxbjt / pParam->B3SOIFDndiode);
                  T1 = pow(TRatio, model->B3SOIFDxdif / pParam->B3SOIFDndiode);
                  T2 = pow(TRatio, model->B3SOIFDxrec / pParam->B3SOIFDndiode / 2);
                  T4 = -Eg0 / pParam->B3SOIFDndiode / model->B3SOIFDvtm * (1 - TRatio);
                  T5 = exp(T4);
                  T6 = sqrt(T5);
                  pParam->B3SOIFDjbjt = pParam->B3SOIFDisbjt * T0 * T5;
                  pParam->B3SOIFDjdif = pParam->B3SOIFDisdif * T1 * T5;
                  pParam->B3SOIFDjrec = pParam->B3SOIFDisrec * T2 * T6;
                  T0 = pow(TRatio, model->B3SOIFDxtun / pParam->B3SOIFDntun);
                  pParam->B3SOIFDjtun = pParam->B3SOIFDistun * T0 ;

                  if (pParam->B3SOIFDnsub > 0)
                     pParam->B3SOIFDvfbb = -model->B3SOIFDtype * model->B3SOIFDvtm *
                                log(pParam->B3SOIFDnpeak/ pParam->B3SOIFDnsub);
                  else
                     pParam->B3SOIFDvfbb = -model->B3SOIFDtype * model->B3SOIFDvtm *
                                log(-pParam->B3SOIFDnpeak* pParam->B3SOIFDnsub/ni/ni);

                  if (!model->B3SOIFDvsdfbGiven)
                  {
                     if (pParam->B3SOIFDnsub > 0)
                        pParam->B3SOIFDvsdfb = -model->B3SOIFDtype * (model->B3SOIFDvtm*log(1e20 *
                                            pParam->B3SOIFDnsub / ni /ni) - 0.3);
                     else if (pParam->B3SOIFDnsub < 0)
                        pParam->B3SOIFDvsdfb = -model->B3SOIFDtype * (model->B3SOIFDvtm*log(-1e20 /
                                            pParam->B3SOIFDnsub) + 0.3);
                  }

                  /* Phi  & Gamma */
                  SDphi = 2.0*model->B3SOIFDvtm*log(fabs(pParam->B3SOIFDnsub) / ni);
                  SDgamma = 5.753e-12 * sqrt(fabs(pParam->B3SOIFDnsub)) / model->B3SOIFDcbox;

                  if (!model->B3SOIFDvsdthGiven)
                  {
                     if ( ((pParam->B3SOIFDnsub > 0) && (model->B3SOIFDtype > 0)) ||
                          ((pParam->B3SOIFDnsub < 0) && (model->B3SOIFDtype < 0)) )
                        pParam->B3SOIFDvsdth = pParam->B3SOIFDvsdfb + SDphi +
                                            SDgamma * sqrt(SDphi);
                     else
                        pParam->B3SOIFDvsdth = pParam->B3SOIFDvsdfb - SDphi -
                                            SDgamma * sqrt(SDphi);
                  }
                  if (!model->B3SOIFDcsdminGiven)
                  {
                     /* Cdmin */
                     tmp = sqrt(2.0 * EPSSI * SDphi / (Charge_q *
                                fabs(pParam->B3SOIFDnsub) * 1.0e6));
                     tmp1 = EPSSI / tmp;
                     model->B3SOIFDcsdmin = tmp1 * model->B3SOIFDcbox /
                                          (tmp1 + model->B3SOIFDcbox);
                  }

                  T0 = model->B3SOIFDcsdesw * log(1 + model->B3SOIFDtsi /
                       model->B3SOIFDtbox);
                  T1 = here->B3SOIFDsourcePerimeter - pParam->B3SOIFDweff;
                  if (T1 > 0.0)
                     pParam->B3SOIFDcsesw = T0 * T1;
                  else
                     pParam->B3SOIFDcsesw = 0.0;
                  T1 = here->B3SOIFDdrainPerimeter - pParam->B3SOIFDweff;
                  if (T1 > 0.0)
                     pParam->B3SOIFDcdesw = T0 * T1;
                  else
                     pParam->B3SOIFDcdesw = 0.0;

                  pParam->B3SOIFDphi = 2.0 * model->B3SOIFDvtm
                                   * log(pParam->B3SOIFDnpeak / ni);

                  pParam->B3SOIFDsqrtPhi = sqrt(pParam->B3SOIFDphi);
                  pParam->B3SOIFDphis3 = pParam->B3SOIFDsqrtPhi * pParam->B3SOIFDphi;

                  pParam->B3SOIFDXdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->B3SOIFDnpeak * 1.0e6))
                                     * pParam->B3SOIFDsqrtPhi;
                  pParam->B3SOIFDsqrtXdep0 = sqrt(pParam->B3SOIFDXdep0);
                  pParam->B3SOIFDlitl = sqrt(3.0 * model->B3SOIFDxj
                                    * model->B3SOIFDtox);
                  pParam->B3SOIFDvbi = model->B3SOIFDvtm * log(1.0e20
                                   * pParam->B3SOIFDnpeak / (ni * ni));
                  pParam->B3SOIFDcdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->B3SOIFDnpeak * 1.0e6 / 2.0
                                     / pParam->B3SOIFDphi);

                  if (model->B3SOIFDk1Given || model->B3SOIFDk2Given)
                  {   if (!model->B3SOIFDk1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->B3SOIFDk1 = 0.53;
                      }
                      if (!model->B3SOIFDk2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->B3SOIFDk2 = -0.0186;
                      }
                      if (model->B3SOIFDxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIFDvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIFDvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIFDgamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIFDgamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->B3SOIFDvbxGiven)
                          pParam->B3SOIFDvbx = pParam->B3SOIFDphi - 7.7348e-4
                                           * pParam->B3SOIFDnpeak
                                           * pParam->B3SOIFDxt * pParam->B3SOIFDxt;
                      if (pParam->B3SOIFDvbx > 0.0)
                          pParam->B3SOIFDvbx = -pParam->B3SOIFDvbx;
                      if (pParam->B3SOIFDvbm > 0.0)
                          pParam->B3SOIFDvbm = -pParam->B3SOIFDvbm;

                      if (!model->B3SOIFDgamma1Given)
                          pParam->B3SOIFDgamma1 = 5.753e-12
                                              * sqrt(pParam->B3SOIFDnpeak)
                                              / model->B3SOIFDcox;
                      if (!model->B3SOIFDgamma2Given)
                          pParam->B3SOIFDgamma2 = 5.753e-12
                                              * sqrt(pParam->B3SOIFDnsub)
                                              / model->B3SOIFDcox;

                      T0 = pParam->B3SOIFDgamma1 - pParam->B3SOIFDgamma2;
                      T1 = sqrt(pParam->B3SOIFDphi - pParam->B3SOIFDvbx)
                         - pParam->B3SOIFDsqrtPhi;
                      T2 = sqrt(pParam->B3SOIFDphi * (pParam->B3SOIFDphi
                         - pParam->B3SOIFDvbm)) - pParam->B3SOIFDphi;
                      pParam->B3SOIFDk2 = T0 * T1 / (2.0 * T2 + pParam->B3SOIFDvbm);
                      pParam->B3SOIFDk1 = pParam->B3SOIFDgamma2 - 2.0
                                      * pParam->B3SOIFDk2 * sqrt(pParam->B3SOIFDphi
                                      - pParam->B3SOIFDvbm);
                  }

                  if (pParam->B3SOIFDk2 < 0.0)
                  {   T0 = 0.5 * pParam->B3SOIFDk1 / pParam->B3SOIFDk2;
                      pParam->B3SOIFDvbsc = 0.9 * (pParam->B3SOIFDphi - T0 * T0);
                      if (pParam->B3SOIFDvbsc > -3.0)
                          pParam->B3SOIFDvbsc = -3.0;
                      else if (pParam->B3SOIFDvbsc < -30.0)
                          pParam->B3SOIFDvbsc = -30.0;
                  }
                  else
                  {   pParam->B3SOIFDvbsc = -30.0;
                  }
                  if (pParam->B3SOIFDvbsc > pParam->B3SOIFDvbm)
                      pParam->B3SOIFDvbsc = pParam->B3SOIFDvbm;

                  if (model->B3SOIFDvth0Given)
                  {   pParam->B3SOIFDvfb = model->B3SOIFDtype * pParam->B3SOIFDvth0
                                       - pParam->B3SOIFDphi - pParam->B3SOIFDk1
                                       * pParam->B3SOIFDsqrtPhi;
                  }
                  else
                  {   pParam->B3SOIFDvfb = -1.0;
                      pParam->B3SOIFDvth0 = model->B3SOIFDtype * (pParam->B3SOIFDvfb
                                        + pParam->B3SOIFDphi + pParam->B3SOIFDk1
                                        * pParam->B3SOIFDsqrtPhi);
                  }
                  T1 = sqrt(EPSSI / EPSOX * model->B3SOIFDtox
                     * pParam->B3SOIFDXdep0);
                  T0 = exp(-0.5 * pParam->B3SOIFDdsub * pParam->B3SOIFDleff / T1);
                  pParam->B3SOIFDtheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->B3SOIFDdrout * pParam->B3SOIFDleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->B3SOIFDthetaRout = pParam->B3SOIFDpdibl1 * T2
                                         + pParam->B3SOIFDpdibl2;

                  here->B3SOIFDminIsub = 5.0e-2 * pParam->B3SOIFDweff * model->B3SOIFDtsi
                                     * MAX(pParam->B3SOIFDisdif, pParam->B3SOIFDisrec);
              }

              here->B3SOIFDcsbox = model->B3SOIFDcbox*here->B3SOIFDsourceArea;
              here->B3SOIFDcsmin = model->B3SOIFDcsdmin*here->B3SOIFDsourceArea;
              here->B3SOIFDcdbox = model->B3SOIFDcbox*here->B3SOIFDdrainArea;
              here->B3SOIFDcdmin = model->B3SOIFDcsdmin*here->B3SOIFDdrainArea;

              if ( ((pParam->B3SOIFDnsub > 0) && (model->B3SOIFDtype > 0)) ||
                   ((pParam->B3SOIFDnsub < 0) && (model->B3SOIFDtype < 0)) )
              {
                 T0 = pParam->B3SOIFDvsdth - pParam->B3SOIFDvsdfb;
                 pParam->B3SOIFDsdt1 = pParam->B3SOIFDvsdfb + model->B3SOIFDasd * T0;
                 T1 = here->B3SOIFDcsbox - here->B3SOIFDcsmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIFDst2 = T2 / model->B3SOIFDasd;
                 pParam->B3SOIFDst3 = T2 /( 1 - model->B3SOIFDasd);
                 here->B3SOIFDst4 =  T0 * T1 * (1 + model->B3SOIFDasd) / 3
                                  - here->B3SOIFDcsmin * pParam->B3SOIFDvsdfb;

                 T1 = here->B3SOIFDcdbox - here->B3SOIFDcdmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIFDdt2 = T2 / model->B3SOIFDasd;
                 pParam->B3SOIFDdt3 = T2 /( 1 - model->B3SOIFDasd);
                 here->B3SOIFDdt4 =  T0 * T1 * (1 + model->B3SOIFDasd) / 3
                                  - here->B3SOIFDcdmin * pParam->B3SOIFDvsdfb;
              } else
              {
                 T0 = pParam->B3SOIFDvsdfb - pParam->B3SOIFDvsdth;
                 pParam->B3SOIFDsdt1 = pParam->B3SOIFDvsdth + model->B3SOIFDasd * T0;
                 T1 = here->B3SOIFDcsmin - here->B3SOIFDcsbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIFDst2 = T2 / model->B3SOIFDasd;
                 pParam->B3SOIFDst3 = T2 /( 1 - model->B3SOIFDasd);
                 here->B3SOIFDst4 =  T0 * T1 * (1 + model->B3SOIFDasd) / 3
                                  - here->B3SOIFDcsbox * pParam->B3SOIFDvsdth;

                 T1 = here->B3SOIFDcdmin - here->B3SOIFDcdbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIFDdt2 = T2 / model->B3SOIFDasd;
                 pParam->B3SOIFDdt3 = T2 /( 1 - model->B3SOIFDasd);
                 here->B3SOIFDdt4 =  T0 * T1 * (1 + model->B3SOIFDasd) / 3
                                  - here->B3SOIFDcdbox * pParam->B3SOIFDvsdth;
              }

              here->B3SOIFDphi = pParam->B3SOIFDphi;
              /* process source/drain series resistance */
              here->B3SOIFDdrainConductance = model->B3SOIFDsheetResistance
                                              * here->B3SOIFDdrainSquares;
              if (here->B3SOIFDdrainConductance > 0.0)
                  here->B3SOIFDdrainConductance = 1.0
                                              / here->B3SOIFDdrainConductance;
              else
                  here->B3SOIFDdrainConductance = 0.0;

              here->B3SOIFDsourceConductance = model->B3SOIFDsheetResistance
                                           * here->B3SOIFDsourceSquares;
              if (here->B3SOIFDsourceConductance > 0.0)
                  here->B3SOIFDsourceConductance = 1.0
                                               / here->B3SOIFDsourceConductance;
              else
                  here->B3SOIFDsourceConductance = 0.0;
              here->B3SOIFDcgso = pParam->B3SOIFDcgso;
              here->B3SOIFDcgdo = pParam->B3SOIFDcgdo;

         }
    }
    return(OK);
}

