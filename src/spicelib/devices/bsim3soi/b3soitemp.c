/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soitemp.c          98/5/01
Modified by Pin Su	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su, Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su and Hui Wan 02/3/5
Modified by Pin Su 02/5/20
Modified by Paolo Nenzi 2002
**********/

/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

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
B3SOItemp(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOImodel *model = (B3SOImodel*) inModel;
B3SOIinstance *here;
struct b3soiSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp, tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double Temp, TempRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double SDphi, SDgamma;
int Size_Not_Found;

/* v2.0 release */
double tmp3, T7;


    /*  loop through all the B3SOI device models */
    for (; model != NULL; model = model->B3SOInextModel)
    {    Temp = ckt->CKTtemp;
         if (model->B3SOIGatesidewallJctPotential < 0.1)
             model->B3SOIGatesidewallJctPotential = 0.1;
         model->pSizeDependParamKnot = NULL;
	 pLastKnot = NULL;

	 Tnom = model->B3SOItnom;
	 TempRatio = Temp / Tnom;

	 model->B3SOIvcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->B3SOIfactor1 = sqrt(EPSSI / EPSOX * model->B3SOItox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         model->B3SOIeg0 = Eg0; 
         model->B3SOIvtm = KboQ * Temp;

         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         /* ni is in cm^-3 */
         ni = 1.45e10 * (Temp / 300.15) * sqrt(Temp / 300.15) 
            * exp(21.5565981 - Eg / (2.0 * model->B3SOIvtm));


         /* loop through all the instances of the model */
	 /* MCJ: Length and Width not initialized */
         for (here = model->B3SOIinstances; here != NULL;
              here = here->B3SOInextInstance) 
	 {    
              
	      if (here->B3SOIowner != ARCHme)
                      continue;

	      here->B3SOIrbodyext = here->B3SOIbodySquares *
                                    model->B3SOIrbsh;
	      pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->B3SOIl == pSizeDependParamKnot->Length)
		      && (here->B3SOIw == pSizeDependParamKnot->Width)
                      && (here->B3SOIrth0 == pSizeDependParamKnot->Rth0)
                      && (here->B3SOIcth0 == pSizeDependParamKnot->Cth0))
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
	      {   pParam = (struct b3soiSizeDependParam *)tmalloc(
	                    sizeof(struct b3soiSizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

		  Ldrn = here->B3SOIl;
		  Wdrn = here->B3SOIw;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
                  pParam->Rth0 = here->B3SOIrth0;
                  pParam->Cth0 = here->B3SOIcth0;
		  
                  T0 = pow(Ldrn, model->B3SOILln);
                  T1 = pow(Wdrn, model->B3SOILwn);
                  tmp1 = model->B3SOILl / T0 + model->B3SOILw / T1
                       + model->B3SOILwl / (T0 * T1);
                  pParam->B3SOIdl = model->B3SOILint + tmp1;

/* v2.2.3 */
                  tmp1 = model->B3SOILlc / T0 + model->B3SOILwc / T1
                       + model->B3SOILwlc / (T0 * T1);
                  pParam->B3SOIdlc = model->B3SOIdlc + tmp1;

/* v3.0 */
                  pParam->B3SOIdlcig = model->B3SOIdlcig + tmp1;


                  T2 = pow(Ldrn, model->B3SOIWln);
                  T3 = pow(Wdrn, model->B3SOIWwn);
                  tmp2 = model->B3SOIWl / T2 + model->B3SOIWw / T3
                       + model->B3SOIWwl / (T2 * T3);
                  pParam->B3SOIdw = model->B3SOIWint + tmp2;

/* v2.2.3 */
                  tmp2 = model->B3SOIWlc / T2 + model->B3SOIWwc / T3
                       + model->B3SOIWwlc / (T2 * T3);
                  pParam->B3SOIdwc = model->B3SOIdwc + tmp2;


                  pParam->B3SOIleff = here->B3SOIl - 2.0 * pParam->B3SOIdl;
                  if (pParam->B3SOIleff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->B3SOImodName;
                      namarray[1] = here->B3SOIname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "B3SOI: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIweff = here->B3SOIw - here->B3SOInbc * model->B3SOIdwbc
                     - (2.0 - here->B3SOInbc) * pParam->B3SOIdw;
                  if (pParam->B3SOIweff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->B3SOImodName;
                      namarray[1] = here->B3SOIname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "B3SOI: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIwdiod = pParam->B3SOIweff / here->B3SOInseg + here->B3SOIpdbcp;
                  pParam->B3SOIwdios = pParam->B3SOIweff / here->B3SOInseg + here->B3SOIpsbcp;

                  pParam->B3SOIleffCV = here->B3SOIl - 2.0 * pParam->B3SOIdlc;
                  if (pParam->B3SOIleffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->B3SOImodName;
                      namarray[1] = here->B3SOIname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "B3SOI: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIweffCV = here->B3SOIw - here->B3SOInbc * model->B3SOIdwbc
                     - (2.0 - here->B3SOInbc) * pParam->B3SOIdwc;
                  if (pParam->B3SOIweffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->B3SOImodName;
                      namarray[1] = here->B3SOIname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "B3SOI: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->B3SOIwdiodCV = pParam->B3SOIweffCV / here->B3SOInseg + here->B3SOIpdbcp;
                  pParam->B3SOIwdiosCV = pParam->B3SOIweffCV / here->B3SOInseg + here->B3SOIpsbcp;

                  pParam->B3SOIleffCVb = here->B3SOIl - 2.0 * pParam->B3SOIdlc - model->B3SOIdlcb;
                  if (pParam->B3SOIleffCVb <= 0.0)
                  {  
                     IFuid namarray[2];
                     namarray[0] = model->B3SOImodName;
                     namarray[1] = here->B3SOIname;
                     (*(SPfrontEnd->IFerror))(ERR_FATAL,
                     "B3SOI: mosfet %s, model %s: Effective channel length for C-V (body) <= 0",
                     namarray);
                     return(E_BADPARM);
                  }

                  pParam->B3SOIleffCVbg = pParam->B3SOIleffCVb + 2 * model->B3SOIdlbg;
                  if (pParam->B3SOIleffCVbg <= 0.0)
                  { 
                     IFuid namarray[2];
                     namarray[0] = model->B3SOImodName;
                     namarray[1] = here->B3SOIname;
                     (*(SPfrontEnd->IFerror))(ERR_FATAL,
                     "B3SOI: mosfet %s, model %s: Effective channel length for C-V (backgate) <= 0",
                     namarray);
                     return(E_BADPARM);
                  }

                  /* Not binned - START */
		  pParam->B3SOIat = model->B3SOIat;
		  pParam->B3SOIgamma1 = model->B3SOIgamma1;
		  pParam->B3SOIgamma2 = model->B3SOIgamma2;
		  pParam->B3SOIvbx = model->B3SOIvbx;
		  pParam->B3SOIvbm = model->B3SOIvbm;
		  pParam->B3SOIxt = model->B3SOIxt;
		  pParam->B3SOIkt1 = model->B3SOIkt1;
		  pParam->B3SOIkt1l = model->B3SOIkt1l;
		  pParam->B3SOIkt2 = model->B3SOIkt2;
		  pParam->B3SOIua1 = model->B3SOIua1;
		  pParam->B3SOIub1 = model->B3SOIub1;
		  pParam->B3SOIuc1 = model->B3SOIuc1;
		  pParam->B3SOIute = model->B3SOIute;
		  pParam->B3SOIprt = model->B3SOIprt;
                  /* Not binned - END */

		  /* CV model */
		  pParam->B3SOIcgsl = model->B3SOIcgsl;
		  pParam->B3SOIcgdl = model->B3SOIcgdl;
		  pParam->B3SOIckappa = model->B3SOIckappa;
		  pParam->B3SOIcf = model->B3SOIcf;
		  pParam->B3SOIclc = model->B3SOIclc;
		  pParam->B3SOIcle = model->B3SOIcle;

                  pParam->B3SOIabulkCVfactor = 1.0 + pow((pParam->B3SOIclc / pParam->B3SOIleff),
					     pParam->B3SOIcle);

                  /* Added for binning - START */
		  if (model->B3SOIbinUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->B3SOIleff;
		      Inv_W = 1.0e-6 / pParam->B3SOIweff;
		      Inv_LW = 1.0e-12 / (pParam->B3SOIleff
			     * pParam->B3SOIweff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->B3SOIleff;
		      Inv_W = 1.0 / pParam->B3SOIweff;
		      Inv_LW = 1.0 / (pParam->B3SOIleff
			     * pParam->B3SOIweff);
		  }
		  pParam->B3SOInpeak = model->B3SOInpeak
				     + model->B3SOIlnpeak * Inv_L
				     + model->B3SOIwnpeak * Inv_W
				     + model->B3SOIpnpeak * Inv_LW;
		  pParam->B3SOInsub = model->B3SOInsub
				    + model->B3SOIlnsub * Inv_L
				    + model->B3SOIwnsub * Inv_W
				    + model->B3SOIpnsub * Inv_LW;
		  pParam->B3SOIngate = model->B3SOIngate
				     + model->B3SOIlngate * Inv_L
				     + model->B3SOIwngate * Inv_W
				     + model->B3SOIpngate * Inv_LW;
		  pParam->B3SOIvth0 = model->B3SOIvth0
				    + model->B3SOIlvth0 * Inv_L
				    + model->B3SOIwvth0 * Inv_W
				    + model->B3SOIpvth0 * Inv_LW;
		  pParam->B3SOIk1 = model->B3SOIk1
				  + model->B3SOIlk1 * Inv_L
				  + model->B3SOIwk1 * Inv_W
				  + model->B3SOIpk1 * Inv_LW;
		  pParam->B3SOIk2 = model->B3SOIk2
				  + model->B3SOIlk2 * Inv_L
				  + model->B3SOIwk2 * Inv_W
				  + model->B3SOIpk2 * Inv_LW;
		  pParam->B3SOIk1w1 = model->B3SOIk1w1
				  + model->B3SOIlk1w1 * Inv_L
				  + model->B3SOIwk1w1 * Inv_W
				  + model->B3SOIpk1w1 * Inv_LW;
		  pParam->B3SOIk1w2 = model->B3SOIk1w2
				  + model->B3SOIlk1w2 * Inv_L
				  + model->B3SOIwk1w2 * Inv_W
				  + model->B3SOIpk1w2 * Inv_LW;
		  pParam->B3SOIk3 = model->B3SOIk3
				  + model->B3SOIlk3 * Inv_L
				  + model->B3SOIwk3 * Inv_W
				  + model->B3SOIpk3 * Inv_LW;
		  pParam->B3SOIk3b = model->B3SOIk3b
				   + model->B3SOIlk3b * Inv_L
				   + model->B3SOIwk3b * Inv_W
				   + model->B3SOIpk3b * Inv_LW;
		  pParam->B3SOIkb1 = model->B3SOIkb1
				   + model->B3SOIlkb1 * Inv_L
				   + model->B3SOIwkb1 * Inv_W
				   + model->B3SOIpkb1 * Inv_LW;
		  pParam->B3SOIw0 = model->B3SOIw0
				  + model->B3SOIlw0 * Inv_L
				  + model->B3SOIww0 * Inv_W
				  + model->B3SOIpw0 * Inv_LW;
		  pParam->B3SOInlx = model->B3SOInlx
				   + model->B3SOIlnlx * Inv_L
				   + model->B3SOIwnlx * Inv_W
				   + model->B3SOIpnlx * Inv_LW;
		  pParam->B3SOIdvt0 = model->B3SOIdvt0
				    + model->B3SOIldvt0 * Inv_L
				    + model->B3SOIwdvt0 * Inv_W
				    + model->B3SOIpdvt0 * Inv_LW;
		  pParam->B3SOIdvt1 = model->B3SOIdvt1
				    + model->B3SOIldvt1 * Inv_L
				    + model->B3SOIwdvt1 * Inv_W
				    + model->B3SOIpdvt1 * Inv_LW;
		  pParam->B3SOIdvt2 = model->B3SOIdvt2
				    + model->B3SOIldvt2 * Inv_L
				    + model->B3SOIwdvt2 * Inv_W
				    + model->B3SOIpdvt2 * Inv_LW;
		  pParam->B3SOIdvt0w = model->B3SOIdvt0w
				    + model->B3SOIldvt0w * Inv_L
				    + model->B3SOIwdvt0w * Inv_W
				    + model->B3SOIpdvt0w * Inv_LW;
		  pParam->B3SOIdvt1w = model->B3SOIdvt1w
				    + model->B3SOIldvt1w * Inv_L
				    + model->B3SOIwdvt1w * Inv_W
				    + model->B3SOIpdvt1w * Inv_LW;
		  pParam->B3SOIdvt2w = model->B3SOIdvt2w
				    + model->B3SOIldvt2w * Inv_L
				    + model->B3SOIwdvt2w * Inv_W
				    + model->B3SOIpdvt2w * Inv_LW;
		  pParam->B3SOIu0 = model->B3SOIu0
				  + model->B3SOIlu0 * Inv_L
				  + model->B3SOIwu0 * Inv_W
				  + model->B3SOIpu0 * Inv_LW;
		  pParam->B3SOIua = model->B3SOIua
				  + model->B3SOIlua * Inv_L
				  + model->B3SOIwua * Inv_W
				  + model->B3SOIpua * Inv_LW;
		  pParam->B3SOIub = model->B3SOIub
				  + model->B3SOIlub * Inv_L
				  + model->B3SOIwub * Inv_W
				  + model->B3SOIpub * Inv_LW;
		  pParam->B3SOIuc = model->B3SOIuc
				  + model->B3SOIluc * Inv_L
				  + model->B3SOIwuc * Inv_W
				  + model->B3SOIpuc * Inv_LW;
		  pParam->B3SOIvsat = model->B3SOIvsat
				    + model->B3SOIlvsat * Inv_L
				    + model->B3SOIwvsat * Inv_W
				    + model->B3SOIpvsat * Inv_LW;
		  pParam->B3SOIa0 = model->B3SOIa0
				  + model->B3SOIla0 * Inv_L
				  + model->B3SOIwa0 * Inv_W
				  + model->B3SOIpa0 * Inv_LW; 
		  pParam->B3SOIags = model->B3SOIags
				  + model->B3SOIlags * Inv_L
				  + model->B3SOIwags * Inv_W
				  + model->B3SOIpags * Inv_LW;
		  pParam->B3SOIb0 = model->B3SOIb0
				  + model->B3SOIlb0 * Inv_L
				  + model->B3SOIwb0 * Inv_W
				  + model->B3SOIpb0 * Inv_LW;
		  pParam->B3SOIb1 = model->B3SOIb1
				  + model->B3SOIlb1 * Inv_L
				  + model->B3SOIwb1 * Inv_W
				  + model->B3SOIpb1 * Inv_LW;
		  pParam->B3SOIketa = model->B3SOIketa
				    + model->B3SOIlketa * Inv_L
				    + model->B3SOIwketa * Inv_W
				    + model->B3SOIpketa * Inv_LW;
		  pParam->B3SOIketas = model->B3SOIketas
				    + model->B3SOIlketas * Inv_L
				    + model->B3SOIwketas * Inv_W
				    + model->B3SOIpketas * Inv_LW;
		  pParam->B3SOIa1 = model->B3SOIa1
				  + model->B3SOIla1 * Inv_L
				  + model->B3SOIwa1 * Inv_W
				  + model->B3SOIpa1 * Inv_LW;
		  pParam->B3SOIa2 = model->B3SOIa2
				  + model->B3SOIla2 * Inv_L
				  + model->B3SOIwa2 * Inv_W
				  + model->B3SOIpa2 * Inv_LW;
		  pParam->B3SOIrdsw = model->B3SOIrdsw
				    + model->B3SOIlrdsw * Inv_L
				    + model->B3SOIwrdsw * Inv_W
				    + model->B3SOIprdsw * Inv_LW;
		  pParam->B3SOIprwb = model->B3SOIprwb
				    + model->B3SOIlprwb * Inv_L
				    + model->B3SOIwprwb * Inv_W
				    + model->B3SOIpprwb * Inv_LW;
		  pParam->B3SOIprwg = model->B3SOIprwg
				    + model->B3SOIlprwg * Inv_L
				    + model->B3SOIwprwg * Inv_W
				    + model->B3SOIpprwg * Inv_LW;
		  pParam->B3SOIwr = model->B3SOIwr
				  + model->B3SOIlwr * Inv_L
				  + model->B3SOIwwr * Inv_W
				  + model->B3SOIpwr * Inv_LW;
		  pParam->B3SOInfactor = model->B3SOInfactor
				       + model->B3SOIlnfactor * Inv_L
				       + model->B3SOIwnfactor * Inv_W
				       + model->B3SOIpnfactor * Inv_LW;
		  pParam->B3SOIdwg = model->B3SOIdwg
				   + model->B3SOIldwg * Inv_L
				   + model->B3SOIwdwg * Inv_W
				   + model->B3SOIpdwg * Inv_LW;
		  pParam->B3SOIdwb = model->B3SOIdwb
				   + model->B3SOIldwb * Inv_L
				   + model->B3SOIwdwb * Inv_W
				   + model->B3SOIpdwb * Inv_LW;
		  pParam->B3SOIvoff = model->B3SOIvoff
				    + model->B3SOIlvoff * Inv_L
				    + model->B3SOIwvoff * Inv_W
				    + model->B3SOIpvoff * Inv_LW;
		  pParam->B3SOIeta0 = model->B3SOIeta0
				    + model->B3SOIleta0 * Inv_L
				    + model->B3SOIweta0 * Inv_W
				    + model->B3SOIpeta0 * Inv_LW;
		  pParam->B3SOIetab = model->B3SOIetab
				    + model->B3SOIletab * Inv_L
				    + model->B3SOIwetab * Inv_W
				    + model->B3SOIpetab * Inv_LW;
		  pParam->B3SOIdsub = model->B3SOIdsub
				    + model->B3SOIldsub * Inv_L
				    + model->B3SOIwdsub * Inv_W
				    + model->B3SOIpdsub * Inv_LW;
		  pParam->B3SOIcit = model->B3SOIcit
				   + model->B3SOIlcit * Inv_L
				   + model->B3SOIwcit * Inv_W
				   + model->B3SOIpcit * Inv_LW;
		  pParam->B3SOIcdsc = model->B3SOIcdsc
				    + model->B3SOIlcdsc * Inv_L
				    + model->B3SOIwcdsc * Inv_W
				    + model->B3SOIpcdsc * Inv_LW;
		  pParam->B3SOIcdscb = model->B3SOIcdscb
				     + model->B3SOIlcdscb * Inv_L
				     + model->B3SOIwcdscb * Inv_W
				     + model->B3SOIpcdscb * Inv_LW; 
    		  pParam->B3SOIcdscd = model->B3SOIcdscd
				     + model->B3SOIlcdscd * Inv_L
				     + model->B3SOIwcdscd * Inv_W
				     + model->B3SOIpcdscd * Inv_LW; 
		  pParam->B3SOIpclm = model->B3SOIpclm
				    + model->B3SOIlpclm * Inv_L
				    + model->B3SOIwpclm * Inv_W
				    + model->B3SOIppclm * Inv_LW;
		  pParam->B3SOIpdibl1 = model->B3SOIpdibl1
				      + model->B3SOIlpdibl1 * Inv_L
				      + model->B3SOIwpdibl1 * Inv_W
				      + model->B3SOIppdibl1 * Inv_LW;
		  pParam->B3SOIpdibl2 = model->B3SOIpdibl2
				      + model->B3SOIlpdibl2 * Inv_L
				      + model->B3SOIwpdibl2 * Inv_W
				      + model->B3SOIppdibl2 * Inv_LW;
		  pParam->B3SOIpdiblb = model->B3SOIpdiblb
				      + model->B3SOIlpdiblb * Inv_L
				      + model->B3SOIwpdiblb * Inv_W
				      + model->B3SOIppdiblb * Inv_LW;
		  pParam->B3SOIdrout = model->B3SOIdrout
				     + model->B3SOIldrout * Inv_L
				     + model->B3SOIwdrout * Inv_W
				     + model->B3SOIpdrout * Inv_LW;
		  pParam->B3SOIpvag = model->B3SOIpvag
				    + model->B3SOIlpvag * Inv_L
				    + model->B3SOIwpvag * Inv_W
				    + model->B3SOIppvag * Inv_LW;
		  pParam->B3SOIdelta = model->B3SOIdelta
				     + model->B3SOIldelta * Inv_L
				     + model->B3SOIwdelta * Inv_W
				     + model->B3SOIpdelta * Inv_LW;
		  pParam->B3SOIalpha0 = model->B3SOIalpha0
				      + model->B3SOIlalpha0 * Inv_L
				      + model->B3SOIwalpha0 * Inv_W
				      + model->B3SOIpalpha0 * Inv_LW;
		  pParam->B3SOIfbjtii = model->B3SOIfbjtii
				      + model->B3SOIlfbjtii * Inv_L
				      + model->B3SOIwfbjtii * Inv_W
				      + model->B3SOIpfbjtii * Inv_LW;
		  pParam->B3SOIbeta0 = model->B3SOIbeta0
				     + model->B3SOIlbeta0 * Inv_L
				     + model->B3SOIwbeta0 * Inv_W
				     + model->B3SOIpbeta0 * Inv_LW;
		  pParam->B3SOIbeta1 = model->B3SOIbeta1
				     + model->B3SOIlbeta1 * Inv_L
				     + model->B3SOIwbeta1 * Inv_W
				     + model->B3SOIpbeta1 * Inv_LW;
		  pParam->B3SOIbeta2 = model->B3SOIbeta2
				     + model->B3SOIlbeta2 * Inv_L
				     + model->B3SOIwbeta2 * Inv_W
				     + model->B3SOIpbeta2 * Inv_LW;
		  pParam->B3SOIvdsatii0 = model->B3SOIvdsatii0
				      + model->B3SOIlvdsatii0 * Inv_L
				      + model->B3SOIwvdsatii0 * Inv_W
				      + model->B3SOIpvdsatii0 * Inv_LW;
		  pParam->B3SOIlii = model->B3SOIlii
				      + model->B3SOIllii * Inv_L
				      + model->B3SOIwlii * Inv_W
				      + model->B3SOIplii * Inv_LW;
		  pParam->B3SOIesatii = model->B3SOIesatii
				      + model->B3SOIlesatii * Inv_L
				      + model->B3SOIwesatii * Inv_W
				      + model->B3SOIpesatii * Inv_LW;
		  pParam->B3SOIsii0 = model->B3SOIsii0
				      + model->B3SOIlsii0 * Inv_L
				      + model->B3SOIwsii0 * Inv_W
				      + model->B3SOIpsii0 * Inv_LW;
		  pParam->B3SOIsii1 = model->B3SOIsii1
				      + model->B3SOIlsii1 * Inv_L
				      + model->B3SOIwsii1 * Inv_W
				      + model->B3SOIpsii1 * Inv_LW;
		  pParam->B3SOIsii2 = model->B3SOIsii2
				      + model->B3SOIlsii2 * Inv_L
				      + model->B3SOIwsii2 * Inv_W
				      + model->B3SOIpsii2 * Inv_LW;
		  pParam->B3SOIsiid = model->B3SOIsiid
				      + model->B3SOIlsiid * Inv_L
				      + model->B3SOIwsiid * Inv_W
				      + model->B3SOIpsiid * Inv_LW;
		  pParam->B3SOIagidl = model->B3SOIagidl
				      + model->B3SOIlagidl * Inv_L
				      + model->B3SOIwagidl * Inv_W
				      + model->B3SOIpagidl * Inv_LW;
		  pParam->B3SOIbgidl = model->B3SOIbgidl
				      + model->B3SOIlbgidl * Inv_L
				      + model->B3SOIwbgidl * Inv_W
				      + model->B3SOIpbgidl * Inv_LW;
		  pParam->B3SOIngidl = model->B3SOIngidl
				      + model->B3SOIlngidl * Inv_L
				      + model->B3SOIwngidl * Inv_W
				      + model->B3SOIpngidl * Inv_LW;
		  pParam->B3SOIntun = model->B3SOIntun
				      + model->B3SOIlntun * Inv_L
				      + model->B3SOIwntun * Inv_W
				      + model->B3SOIpntun * Inv_LW;
		  pParam->B3SOIndiode = model->B3SOIndiode
				      + model->B3SOIlndiode * Inv_L
				      + model->B3SOIwndiode * Inv_W
				      + model->B3SOIpndiode * Inv_LW;
		  pParam->B3SOInrecf0 = model->B3SOInrecf0
				  + model->B3SOIlnrecf0 * Inv_L
				  + model->B3SOIwnrecf0 * Inv_W
				  + model->B3SOIpnrecf0 * Inv_LW;
		  pParam->B3SOInrecr0 = model->B3SOInrecr0
				  + model->B3SOIlnrecr0 * Inv_L
				  + model->B3SOIwnrecr0 * Inv_W
				  + model->B3SOIpnrecr0 * Inv_LW;
		  pParam->B3SOIisbjt = model->B3SOIisbjt
				  + model->B3SOIlisbjt * Inv_L
				  + model->B3SOIwisbjt * Inv_W
				  + model->B3SOIpisbjt * Inv_LW;
		  pParam->B3SOIisdif = model->B3SOIisdif
				  + model->B3SOIlisdif * Inv_L
				  + model->B3SOIwisdif * Inv_W
				  + model->B3SOIpisdif * Inv_LW;
		  pParam->B3SOIisrec = model->B3SOIisrec
				  + model->B3SOIlisrec * Inv_L
				  + model->B3SOIwisrec * Inv_W
				  + model->B3SOIpisrec * Inv_LW;
		  pParam->B3SOIistun = model->B3SOIistun
				  + model->B3SOIlistun * Inv_L
				  + model->B3SOIwistun * Inv_W
				  + model->B3SOIpistun * Inv_LW;
		  pParam->B3SOIvrec0 = model->B3SOIvrec0
				  + model->B3SOIlvrec0 * Inv_L
				  + model->B3SOIwvrec0 * Inv_W
				  + model->B3SOIpvrec0 * Inv_LW;
		  pParam->B3SOIvtun0 = model->B3SOIvtun0
				  + model->B3SOIlvtun0 * Inv_L
				  + model->B3SOIwvtun0 * Inv_W
				  + model->B3SOIpvtun0 * Inv_LW;
		  pParam->B3SOInbjt = model->B3SOInbjt
				  + model->B3SOIlnbjt * Inv_L
				  + model->B3SOIwnbjt * Inv_W
				  + model->B3SOIpnbjt * Inv_LW;
		  pParam->B3SOIlbjt0 = model->B3SOIlbjt0
				  + model->B3SOIllbjt0 * Inv_L
				  + model->B3SOIwlbjt0 * Inv_W
				  + model->B3SOIplbjt0 * Inv_LW;
		  pParam->B3SOIvabjt = model->B3SOIvabjt
				  + model->B3SOIlvabjt * Inv_L
				  + model->B3SOIwvabjt * Inv_W
				  + model->B3SOIpvabjt * Inv_LW;
		  pParam->B3SOIaely = model->B3SOIaely
				  + model->B3SOIlaely * Inv_L
				  + model->B3SOIwaely * Inv_W
				  + model->B3SOIpaely * Inv_LW;
		  pParam->B3SOIahli = model->B3SOIahli
				  + model->B3SOIlahli * Inv_L
				  + model->B3SOIwahli * Inv_W
				  + model->B3SOIpahli * Inv_LW;

/* v3.0 */
                  pParam->B3SOInigc = model->B3SOInigc
                                     + model->B3SOIlnigc * Inv_L
                                     + model->B3SOIwnigc * Inv_W
                                     + model->B3SOIpnigc * Inv_LW;
                  pParam->B3SOIaigc = model->B3SOIaigc
                                     + model->B3SOIlaigc * Inv_L
                                     + model->B3SOIwaigc * Inv_W
                                     + model->B3SOIpaigc * Inv_LW;
                  pParam->B3SOIbigc = model->B3SOIbigc
                                     + model->B3SOIlbigc * Inv_L
                                     + model->B3SOIwbigc * Inv_W
                                     + model->B3SOIpbigc * Inv_LW;
                  pParam->B3SOIcigc = model->B3SOIcigc
                                     + model->B3SOIlcigc * Inv_L
                                     + model->B3SOIwcigc * Inv_W
                                     + model->B3SOIpcigc * Inv_LW;
                  pParam->B3SOIaigsd = model->B3SOIaigsd
                                     + model->B3SOIlaigsd * Inv_L
                                     + model->B3SOIwaigsd * Inv_W
                                     + model->B3SOIpaigsd * Inv_LW;
                  pParam->B3SOIbigsd = model->B3SOIbigsd
                                     + model->B3SOIlbigsd * Inv_L
                                     + model->B3SOIwbigsd * Inv_W
                                     + model->B3SOIpbigsd * Inv_LW;
                  pParam->B3SOIcigsd = model->B3SOIcigsd
                                     + model->B3SOIlcigsd * Inv_L
                                     + model->B3SOIwcigsd * Inv_W
                                     + model->B3SOIpcigsd * Inv_LW;
                  pParam->B3SOIpigcd = model->B3SOIpigcd
                                     + model->B3SOIlpigcd * Inv_L
                                     + model->B3SOIwpigcd * Inv_W
                                     + model->B3SOIppigcd * Inv_LW;
                  pParam->B3SOIpoxedge = model->B3SOIpoxedge
                                       + model->B3SOIlpoxedge * Inv_L
                                       + model->B3SOIwpoxedge * Inv_W
                                       + model->B3SOIppoxedge * Inv_LW;
/* v3.0 */


		  /* CV model */
		  pParam->B3SOIvsdfb = model->B3SOIvsdfb
				  + model->B3SOIlvsdfb * Inv_L
				  + model->B3SOIwvsdfb * Inv_W
				  + model->B3SOIpvsdfb * Inv_LW;
		  pParam->B3SOIvsdth = model->B3SOIvsdth
				  + model->B3SOIlvsdth * Inv_L
				  + model->B3SOIwvsdth * Inv_W
				  + model->B3SOIpvsdth * Inv_LW;
		  pParam->B3SOIdelvt = model->B3SOIdelvt
				  + model->B3SOIldelvt * Inv_L
				  + model->B3SOIwdelvt * Inv_W
				  + model->B3SOIpdelvt * Inv_LW;
		  pParam->B3SOIacde = model->B3SOIacde
				  + model->B3SOIlacde * Inv_L
				  + model->B3SOIwacde * Inv_W
				  + model->B3SOIpacde * Inv_LW;
		  pParam->B3SOImoin = model->B3SOImoin
				  + model->B3SOIlmoin * Inv_L
				  + model->B3SOIwmoin * Inv_W
				  + model->B3SOIpmoin * Inv_LW;
                  /* Added for binning - END */

	          T0 = (TempRatio - 1.0);

                  pParam->B3SOIuatemp = pParam->B3SOIua;  /*  save ua, ub, and uc for b3soild.c */
                  pParam->B3SOIubtemp = pParam->B3SOIub;
                  pParam->B3SOIuctemp = pParam->B3SOIuc;
                  pParam->B3SOIrds0denom = pow(pParam->B3SOIweff * 1E6, pParam->B3SOIwr);


/* v2.2 release */
                  pParam->B3SOIrth = here->B3SOIrth0 / (pParam->B3SOIweff + model->B3SOIwth0) 
                                   * here->B3SOInseg;
                  pParam->B3SOIcth = here->B3SOIcth0 * (pParam->B3SOIweff + model->B3SOIwth0)
                                   / here->B3SOInseg;

/* v2.2.2 adding layout-dependent Frbody multiplier */
                  pParam->B3SOIrbody = here->B3SOIfrbody *model->B3SOIrbody * model->B3SOIrhalo
                                     / (2 * model->B3SOIrbody + model->B3SOIrhalo * pParam->B3SOIleff)
                                     * pParam->B3SOIweff / here->B3SOInseg;

                  pParam->B3SOIoxideRatio = pow(model->B3SOItoxref/model->B3SOItoxqm,
                                  model->B3SOIntox) /model->B3SOItoxqm/model->B3SOItoxqm;
/* v2.2 release */


	          pParam->B3SOIua = pParam->B3SOIua + pParam->B3SOIua1 * T0;
	          pParam->B3SOIub = pParam->B3SOIub + pParam->B3SOIub1 * T0;
	          pParam->B3SOIuc = pParam->B3SOIuc + pParam->B3SOIuc1 * T0;
                  if (pParam->B3SOIu0 > 1.0) 
                      pParam->B3SOIu0 = pParam->B3SOIu0 / 1.0e4;

                  pParam->B3SOIu0temp = pParam->B3SOIu0
				      * pow(TempRatio, pParam->B3SOIute); 
                  pParam->B3SOIvsattemp = pParam->B3SOIvsat - pParam->B3SOIat 
			                * T0;
	          pParam->B3SOIrds0 = (pParam->B3SOIrdsw + pParam->B3SOIprt * T0)
                                    / pow(pParam->B3SOIweff * 1E6, pParam->B3SOIwr);

		  if (B3SOIcheckModel(model, here, ckt))
		  {   IFuid namarray[2];
                      namarray[0] = model->B3SOImodName;
                      namarray[1] = here->B3SOIname;
                      (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during B3SOIV3 parameter checking for %s in model %s", namarray);
                      return(E_BADPARM);   
		  }


                  pParam->B3SOIcgdo = (model->B3SOIcgdo + pParam->B3SOIcf)
				    * pParam->B3SOIwdiodCV;
                  pParam->B3SOIcgso = (model->B3SOIcgso + pParam->B3SOIcf)
				    * pParam->B3SOIwdiosCV;

                  pParam->B3SOIcgeo = model->B3SOIcgeo
                                    * pParam->B3SOIleffCV;


                  if (!model->B3SOInpeakGiven && model->B3SOIgamma1Given)
                  {   T0 = pParam->B3SOIgamma1 * model->B3SOIcox;
                      pParam->B3SOInpeak = 3.021E22 * T0 * T0;
                  }


                  T4 = Eg300 / model->B3SOIvtm * (TempRatio - 1.0);
                  T7 = model->B3SOIxbjt * T4 / pParam->B3SOIndiode;
                  DEXP(T7, T0);
                  T7 = model->B3SOIxdif * T4 / pParam->B3SOIndiode;
                  DEXP(T7, T1);
                  T7 = model->B3SOIxrec * T4 / pParam->B3SOInrecf0;
                  DEXP(T7, T2);

                  /* v2.2.2 bug fix */
                  pParam->B3SOIahli0 = pParam->B3SOIahli * T0;
 
                  pParam->B3SOIjbjt = pParam->B3SOIisbjt * T0;
                  pParam->B3SOIjdif = pParam->B3SOIisdif * T1;
                  pParam->B3SOIjrec = pParam->B3SOIisrec * T2;

                  T7 = model->B3SOIxtun * (TempRatio - 1);
                  DEXP(T7, T0);
                  pParam->B3SOIjtun = pParam->B3SOIistun * T0;
 

                  if (pParam->B3SOInsub > 0)
                     pParam->B3SOIvfbb = -model->B3SOItype * model->B3SOIvtm *
                                log(pParam->B3SOInpeak/ pParam->B3SOInsub);
                  else
                     pParam->B3SOIvfbb = -model->B3SOItype * model->B3SOIvtm *
                                log(-pParam->B3SOInpeak* pParam->B3SOInsub/ni/ni);

                  if (!model->B3SOIvsdfbGiven)
                  {
                     if (pParam->B3SOInsub > 0)
                        pParam->B3SOIvsdfb = -model->B3SOItype * (model->B3SOIvtm*log(1e20 *
                                            pParam->B3SOInsub / ni /ni) - 0.3);
                     else if (pParam->B3SOInsub < 0)
                        pParam->B3SOIvsdfb = -model->B3SOItype * (model->B3SOIvtm*log(-1e20 /
                                            pParam->B3SOInsub) + 0.3);
                  }
                 
                  /* Phi  & Gamma */
                  SDphi = 2.0*model->B3SOIvtm*log(fabs(pParam->B3SOInsub) / ni);
                  SDgamma = 5.753e-12 * sqrt(fabs(pParam->B3SOInsub)) / model->B3SOIcbox;

                  if (!model->B3SOIvsdthGiven)
                  {
                     if ( ((pParam->B3SOInsub > 0) && (model->B3SOItype > 0)) ||
                          ((pParam->B3SOInsub < 0) && (model->B3SOItype < 0)) )
                        pParam->B3SOIvsdth = pParam->B3SOIvsdfb + SDphi +
                                            SDgamma * sqrt(SDphi);
                     else
                        pParam->B3SOIvsdth = pParam->B3SOIvsdfb - SDphi -
                                            SDgamma * sqrt(SDphi);
                  }

                  if (!model->B3SOIcsdminGiven) {
                     /* Cdmin */
                     tmp = sqrt(2.0 * EPSSI * SDphi / (Charge_q * 
                                fabs(pParam->B3SOInsub) * 1.0e6));
                     tmp1 = EPSSI / tmp;
                     model->B3SOIcsdmin = tmp1 * model->B3SOIcbox /
                                          (tmp1 + model->B3SOIcbox);
                  } 


		  pParam->B3SOIphi = 2.0 * model->B3SOIvtm 
			           * log(pParam->B3SOInpeak / ni);

	          pParam->B3SOIsqrtPhi = sqrt(pParam->B3SOIphi);
	          pParam->B3SOIphis3 = pParam->B3SOIsqrtPhi * pParam->B3SOIphi;

                  pParam->B3SOIXdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->B3SOInpeak * 1.0e6))
                                     * pParam->B3SOIsqrtPhi; 
                  pParam->B3SOIsqrtXdep0 = sqrt(pParam->B3SOIXdep0);
                  pParam->B3SOIlitl = sqrt(3.0 * model->B3SOIxj
				    * model->B3SOItox);
                  pParam->B3SOIvbi = model->B3SOIvtm * log(1.0e20
			           * pParam->B3SOInpeak / (ni * ni));
                  pParam->B3SOIcdep0 = sqrt(Charge_q * EPSSI
				     * pParam->B3SOInpeak * 1.0e6 / 2.0
				     / pParam->B3SOIphi);

/* v3.0 */
                  if (pParam->B3SOIngate > 0.0)
                  {   pParam->B3SOIvfbsd = Vtm0 * log(pParam->B3SOIngate
                                         / 1.0e20);
                  }
                  else
                      pParam->B3SOIvfbsd = 0.0;

                  pParam->B3SOIToxRatio = exp(model->B3SOIntox
                                        * log(model->B3SOItoxref /model->B3SOItoxqm))
                                        /model->B3SOItoxqm /model->B3SOItoxqm;
                  pParam->B3SOIToxRatioEdge = exp(model->B3SOIntox
                                            * log(model->B3SOItoxref
                                            / (model->B3SOItoxqm * pParam->B3SOIpoxedge)))
                                            / model->B3SOItoxqm / model->B3SOItoxqm
                                            / pParam->B3SOIpoxedge / pParam->B3SOIpoxedge;
                  pParam->B3SOIAechvb = (model->B3SOItype == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->B3SOIBechvb = (model->B3SOItype == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->B3SOIAechvbEdge = pParam->B3SOIAechvb * pParam->B3SOIweff
                                          * pParam->B3SOIdlcig * pParam->B3SOIToxRatioEdge;
                  pParam->B3SOIBechvbEdge = -pParam->B3SOIBechvb
                                          * model->B3SOItoxqm * pParam->B3SOIpoxedge;
                  pParam->B3SOIAechvb *= pParam->B3SOIweff * pParam->B3SOIleff
                                       * pParam->B3SOIToxRatio;
                  pParam->B3SOIBechvb *= -model->B3SOItoxqm;
/* v3.0 */

        
                  if (model->B3SOIk1Given || model->B3SOIk2Given)
	          {   if (!model->B3SOIk1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->B3SOIk1 = 0.53;
                      }
                      if (!model->B3SOIk2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->B3SOIk2 = -0.0186;
                      }
                      if (model->B3SOIxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIgamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->B3SOIgamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->B3SOIvbxGiven)
                          pParam->B3SOIvbx = pParam->B3SOIphi - 7.7348e-4 
                                           * pParam->B3SOInpeak
					   * pParam->B3SOIxt * pParam->B3SOIxt;
	              if (pParam->B3SOIvbx > 0.0)
		          pParam->B3SOIvbx = -pParam->B3SOIvbx;
	              if (pParam->B3SOIvbm > 0.0)
                          pParam->B3SOIvbm = -pParam->B3SOIvbm;
           
                      if (!model->B3SOIgamma1Given)
                          pParam->B3SOIgamma1 = 5.753e-12
					      * sqrt(pParam->B3SOInpeak)
                                              / model->B3SOIcox;
                      if (!model->B3SOIgamma2Given)
                          pParam->B3SOIgamma2 = 5.753e-12
					      * sqrt(pParam->B3SOInsub)
                                              / model->B3SOIcox;

                      T0 = pParam->B3SOIgamma1 - pParam->B3SOIgamma2;
                      T1 = sqrt(pParam->B3SOIphi - pParam->B3SOIvbx)
			 - pParam->B3SOIsqrtPhi;
                      T2 = sqrt(pParam->B3SOIphi * (pParam->B3SOIphi
			 - pParam->B3SOIvbm)) - pParam->B3SOIphi;
                      pParam->B3SOIk2 = T0 * T1 / (2.0 * T2 + pParam->B3SOIvbm);
                      pParam->B3SOIk1 = pParam->B3SOIgamma2 - 2.0
				      * pParam->B3SOIk2 * sqrt(pParam->B3SOIphi
				      - pParam->B3SOIvbm);
                  }
 
		  if (pParam->B3SOIk2 < 0.0)
		  {   T0 = 0.5 * pParam->B3SOIk1 / pParam->B3SOIk2;
                      pParam->B3SOIvbsc = 0.9 * (pParam->B3SOIphi - T0 * T0);
		      if (pParam->B3SOIvbsc > -3.0)
		          pParam->B3SOIvbsc = -3.0;
		      else if (pParam->B3SOIvbsc < -30.0)
		          pParam->B3SOIvbsc = -30.0;
		  }
		  else
		  {   pParam->B3SOIvbsc = -30.0;
		  }
		  if (pParam->B3SOIvbsc > pParam->B3SOIvbm)
		      pParam->B3SOIvbsc = pParam->B3SOIvbm;

                  if ((T0 = pParam->B3SOIweff + pParam->B3SOIk1w2) < 1e-8)
                     T0 = 1e-8;
                  pParam->B3SOIk1eff = pParam->B3SOIk1 * (1 + pParam->B3SOIk1w1/T0);

	          if (model->B3SOIvth0Given)
		  {   pParam->B3SOIvfb = model->B3SOItype * pParam->B3SOIvth0 
                                       - pParam->B3SOIphi - pParam->B3SOIk1eff 
                                       * pParam->B3SOIsqrtPhi;
		  }
		  else
		  {   pParam->B3SOIvfb = -1.0;
		      pParam->B3SOIvth0 = model->B3SOItype * (pParam->B3SOIvfb
                                        + pParam->B3SOIphi + pParam->B3SOIk1eff 
                                        * pParam->B3SOIsqrtPhi);
		  }
                  T1 = sqrt(EPSSI / EPSOX * model->B3SOItox
		     * pParam->B3SOIXdep0);
                  T0 = exp(-0.5 * pParam->B3SOIdsub * pParam->B3SOIleff / T1);
                  pParam->B3SOItheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->B3SOIdrout * pParam->B3SOIleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->B3SOIthetaRout = pParam->B3SOIpdibl1 * T2
				         + pParam->B3SOIpdibl2;
	      }

              here->B3SOIcsbox = model->B3SOIcbox*here->B3SOIsourceArea;
              here->B3SOIcsmin = model->B3SOIcsdmin*here->B3SOIsourceArea;
              here->B3SOIcdbox = model->B3SOIcbox*here->B3SOIdrainArea;
              here->B3SOIcdmin = model->B3SOIcsdmin*here->B3SOIdrainArea;

	      if ( ((pParam->B3SOInsub > 0) && (model->B3SOItype > 0)) ||
	           ((pParam->B3SOInsub < 0) && (model->B3SOItype < 0)) )
	      {
                 T0 = pParam->B3SOIvsdth - pParam->B3SOIvsdfb;
                 pParam->B3SOIsdt1 = pParam->B3SOIvsdfb + model->B3SOIasd * T0;
                 T1 = here->B3SOIcsbox - here->B3SOIcsmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIst2 = T2 / model->B3SOIasd;
                 pParam->B3SOIst3 = T2 /( 1 - model->B3SOIasd);
                 here->B3SOIst4 =  T0 * T1 * (1 + model->B3SOIasd) / 3
                                  - here->B3SOIcsmin * pParam->B3SOIvsdfb;

                 T1 = here->B3SOIcdbox - here->B3SOIcdmin;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIdt2 = T2 / model->B3SOIasd;
                 pParam->B3SOIdt3 = T2 /( 1 - model->B3SOIasd);
                 here->B3SOIdt4 =  T0 * T1 * (1 + model->B3SOIasd) / 3
                                  - here->B3SOIcdmin * pParam->B3SOIvsdfb;
	      } else
	      {
                 T0 = pParam->B3SOIvsdfb - pParam->B3SOIvsdth;
                 pParam->B3SOIsdt1 = pParam->B3SOIvsdth + model->B3SOIasd * T0;
                 T1 = here->B3SOIcsmin - here->B3SOIcsbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIst2 = T2 / model->B3SOIasd;
                 pParam->B3SOIst3 = T2 /( 1 - model->B3SOIasd);
                 here->B3SOIst4 =  T0 * T1 * (1 + model->B3SOIasd) / 3
                                  - here->B3SOIcsbox * pParam->B3SOIvsdth;
 
                 T1 = here->B3SOIcdmin - here->B3SOIcdbox;
                 T2 = T1 / T0 / T0;
                 pParam->B3SOIdt2 = T2 / model->B3SOIasd;
                 pParam->B3SOIdt3 = T2 /( 1 - model->B3SOIasd);
                 here->B3SOIdt4 =  T0 * T1 * (1 + model->B3SOIasd) / 3
                                      - here->B3SOIcdbox * pParam->B3SOIvsdth;
	      } 

              /* v2.2.2 bug fix */
              T0 = model->B3SOIcsdesw * log(1 + model->B3SOItsi /
                 model->B3SOItbox);
              T1 = here->B3SOIsourcePerimeter - here->B3SOIw;
              if (T1 > 0.0)
                 here->B3SOIcsesw = T0 * T1; 
              else
                 here->B3SOIcsesw = 0.0;
              T1 = here->B3SOIdrainPerimeter - here->B3SOIw;
              if (T1 > 0.0)
                 here->B3SOIcdesw = T0 * T1;
              else
                 here->B3SOIcdesw = 0.0;


	      here->B3SOIphi = pParam->B3SOIphi;
              /* process source/drain series resistance */
              here->B3SOIdrainConductance = model->B3SOIsheetResistance 
		                              * here->B3SOIdrainSquares;
              if (here->B3SOIdrainConductance > 0.0)
                  here->B3SOIdrainConductance = 1.0
					      / here->B3SOIdrainConductance;
	      else
                  here->B3SOIdrainConductance = 0.0;
                  
              here->B3SOIsourceConductance = model->B3SOIsheetResistance 
		                           * here->B3SOIsourceSquares;
              if (here->B3SOIsourceConductance > 0.0) 
                  here->B3SOIsourceConductance = 1.0
					       / here->B3SOIsourceConductance;
	      else
                  here->B3SOIsourceConductance = 0.0;
	      here->B3SOIcgso = pParam->B3SOIcgso;
	      here->B3SOIcgdo = pParam->B3SOIcgdo;


/* v2.0 release */
              if (model->B3SOIln < 1e-15) model->B3SOIln = 1e-15;
              T0 = -0.5 * pParam->B3SOIleff * pParam->B3SOIleff / model->B3SOIln / model->B3SOIln;
              DEXP(T0,T1);
              pParam->B3SOIarfabjt = T1;

              T0 = pParam->B3SOIlbjt0 * (1.0 / pParam->B3SOIleff + 1.0 / model->B3SOIln);
              pParam->B3SOIlratio = pow(T0,pParam->B3SOInbjt);
              pParam->B3SOIlratiodif = 1.0 + model->B3SOIldif0 * pow(T0,model->B3SOIndif);

              if ((pParam->B3SOIvearly = pParam->B3SOIvabjt + pParam->B3SOIaely * pParam->B3SOIleff) < 1) 
                 pParam->B3SOIvearly = 1; 

              /* vfbzb calculation for capMod 3 */
              tmp = sqrt(pParam->B3SOIXdep0);
              tmp1 = pParam->B3SOIvbi - pParam->B3SOIphi;
              tmp2 = model->B3SOIfactor1 * tmp;

              T0 = -0.5 * pParam->B3SOIdvt1w * pParam->B3SOIweff
                 * pParam->B3SOIleff / tmp2;
              if (T0 > -EXPL_THRESHOLD)
              {   T1 = exp(T0);
                  T2 = T1 * (1.0 + 2.0 * T1);
              }
              else
              {   T1 = MIN_EXPL;
                  T2 = T1 * (1.0 + 2.0 * T1);
              }
              T0 = pParam->B3SOIdvt0w * T2;
              T2 = T0 * tmp1;

              T0 = -0.5 * pParam->B3SOIdvt1 * pParam->B3SOIleff / tmp2;
              if (T0 > -EXPL_THRESHOLD)
              {   T1 = exp(T0);
                  T3 = T1 * (1.0 + 2.0 * T1);
              }
              else
              {   T1 = MIN_EXPL;
                  T3 = T1 * (1.0 + 2.0 * T1);
              }
              T3 = pParam->B3SOIdvt0 * T3 * tmp1;

/* v2.2.3 */
              T4 = (model->B3SOItox - model->B3SOIdtoxcv) * pParam->B3SOIphi
                 / (pParam->B3SOIweff + pParam->B3SOIw0);

              T0 = sqrt(1.0 + pParam->B3SOInlx / pParam->B3SOIleff);
              T5 = pParam->B3SOIk1eff * (T0 - 1.0) * pParam->B3SOIsqrtPhi
                 + (pParam->B3SOIkt1 + pParam->B3SOIkt1l / pParam->B3SOIleff)
                 * (TempRatio - 1.0);

              tmp3 = model->B3SOItype * pParam->B3SOIvth0
                   - T2 - T3 + pParam->B3SOIk3 * T4 + T5;
              pParam->B3SOIvfbzb = tmp3 - pParam->B3SOIphi - pParam->B3SOIk1eff
                                 * pParam->B3SOIsqrtPhi;
              /* End of vfbzb */

              pParam->B3SOIldeb = sqrt(EPSSI * Vtm0 / (Charge_q * pParam->B3SOInpeak * 1.0e6)) / 3.0;
              pParam->B3SOIacde = pParam->B3SOIacde * pow((pParam->B3SOInpeak / 2.0e16), -0.25);
         }
    }
    return(OK);
}

