/***********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1stemp.c
**********/
/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

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
BSIM3v1Stemp(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*) inModel;
BSIM3v1Sinstance *here;
struct bsim3v1sSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
int Size_Not_Found;

    /*  loop through all the BSIM3v1S device models */
    for (; model != NULL; model = model->BSIM3v1SnextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3v1SbulkJctPotential < 0.1)  
	     model->BSIM3v1SbulkJctPotential = 0.1;
         if (model->BSIM3v1SsidewallJctPotential < 0.1)
             model->BSIM3v1SsidewallJctPotential = 0.1;
         if (model->BSIM3v1SGatesidewallJctPotential < 0.1)
             model->BSIM3v1SGatesidewallJctPotential = 0.1;
         model->pSizeDependParamKnot = NULL;
	 pLastKnot = NULL;

	 Tnom = model->BSIM3v1Stnom;
	 TRatio = Temp / Tnom;

	 model->BSIM3v1Svcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3v1Sfactor1 = sqrt(EPSSI / EPSOX * model->BSIM3v1Stox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3v1Svtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3v1Svtm + model->BSIM3v1SjctTempExponent
		* log(Temp / Tnom);
	     T1 = exp(T0 / model->BSIM3v1SjctEmissionCoeff);
	     model->BSIM3v1SjctTempSatCurDensity = model->BSIM3v1SjctSatCurDensity
					      * T1;
	     model->BSIM3v1SjctSidewallTempSatCurDensity
			 = model->BSIM3v1SjctSidewallSatCurDensity * T1;
	 }
	 else
	 {   model->BSIM3v1SjctTempSatCurDensity = model->BSIM3v1SjctSatCurDensity;
	     model->BSIM3v1SjctSidewallTempSatCurDensity
			= model->BSIM3v1SjctSidewallSatCurDensity;
	 }

	 if (model->BSIM3v1SjctTempSatCurDensity < 0.0)
	     model->BSIM3v1SjctTempSatCurDensity = 0.0;
	 if (model->BSIM3v1SjctSidewallTempSatCurDensity < 0.0)
	     model->BSIM3v1SjctSidewallTempSatCurDensity = 0.0;

         /* loop through all the instances of the model */
	 /* MCJ: Length and Width not initialized */
         for (here = model->BSIM3v1Sinstances; here != NULL;
              here = here->BSIM3v1SnextInstance) 
	 {    
              
	      if (here->BSIM3v1Sowner != ARCHme) 
	              continue;
		      
              pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM3v1Sl == pSizeDependParamKnot->Length)
		      && (here->BSIM3v1Sw == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      if (Size_Not_Found)
	      {   pParam = (struct bsim3v1sSizeDependParam *)tmalloc(
	                    sizeof(struct bsim3v1sSizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

		  Ldrn = here->BSIM3v1Sl;
		  Wdrn = here->BSIM3v1Sw;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
		  
                  T0 = pow(Ldrn, model->BSIM3v1SLln);
                  T1 = pow(Wdrn, model->BSIM3v1SLwn);
                  tmp1 = model->BSIM3v1SLl / T0 + model->BSIM3v1SLw / T1
                       + model->BSIM3v1SLwl / (T0 * T1);
                  pParam->BSIM3v1Sdl = model->BSIM3v1SLint + tmp1;
                  pParam->BSIM3v1Sdlc = model->BSIM3v1Sdlc + tmp1;

                  T2 = pow(Ldrn, model->BSIM3v1SWln);
                  T3 = pow(Wdrn, model->BSIM3v1SWwn);
                  tmp2 = model->BSIM3v1SWl / T2 + model->BSIM3v1SWw / T3
                       + model->BSIM3v1SWwl / (T2 * T3);
                  pParam->BSIM3v1Sdw = model->BSIM3v1SWint + tmp2;
                  pParam->BSIM3v1Sdwc = model->BSIM3v1Sdwc + tmp2;

                  pParam->BSIM3v1Sleff = here->BSIM3v1Sl - 2.0 * pParam->BSIM3v1Sdl;
                  if (pParam->BSIM3v1Sleff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1SmodName;
                      namarray[1] = here->BSIM3v1Sname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1S: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1Sweff = here->BSIM3v1Sw - 2.0 * pParam->BSIM3v1Sdw;
                  if (pParam->BSIM3v1Sweff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1SmodName;
                      namarray[1] = here->BSIM3v1Sname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1S: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1SleffCV = here->BSIM3v1Sl - 2.0 * pParam->BSIM3v1Sdlc;
                  if (pParam->BSIM3v1SleffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1SmodName;
                      namarray[1] = here->BSIM3v1Sname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1S: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1SweffCV = here->BSIM3v1Sw - 2.0 * pParam->BSIM3v1Sdwc;
                  if (pParam->BSIM3v1SweffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1SmodName;
                      namarray[1] = here->BSIM3v1Sname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1S: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM3v1SbinUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM3v1Sleff;
		      Inv_W = 1.0e-6 / pParam->BSIM3v1Sweff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM3v1Sleff
			     * pParam->BSIM3v1Sweff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM3v1Sleff;
		      Inv_W = 1.0 / pParam->BSIM3v1Sweff;
		      Inv_LW = 1.0 / (pParam->BSIM3v1Sleff
			     * pParam->BSIM3v1Sweff);
		  }
		  pParam->BSIM3v1Scdsc = model->BSIM3v1Scdsc
				    + model->BSIM3v1Slcdsc * Inv_L
				    + model->BSIM3v1Swcdsc * Inv_W
				    + model->BSIM3v1Spcdsc * Inv_LW;
		  pParam->BSIM3v1Scdscb = model->BSIM3v1Scdscb
				     + model->BSIM3v1Slcdscb * Inv_L
				     + model->BSIM3v1Swcdscb * Inv_W
				     + model->BSIM3v1Spcdscb * Inv_LW; 
				     
    		  pParam->BSIM3v1Scdscd = model->BSIM3v1Scdscd
				     + model->BSIM3v1Slcdscd * Inv_L
				     + model->BSIM3v1Swcdscd * Inv_W
				     + model->BSIM3v1Spcdscd * Inv_LW; 
				     
		  pParam->BSIM3v1Scit = model->BSIM3v1Scit
				   + model->BSIM3v1Slcit * Inv_L
				   + model->BSIM3v1Swcit * Inv_W
				   + model->BSIM3v1Spcit * Inv_LW;
		  pParam->BSIM3v1Snfactor = model->BSIM3v1Snfactor
				       + model->BSIM3v1Slnfactor * Inv_L
				       + model->BSIM3v1Swnfactor * Inv_W
				       + model->BSIM3v1Spnfactor * Inv_LW;
		  pParam->BSIM3v1Sxj = model->BSIM3v1Sxj
				  + model->BSIM3v1Slxj * Inv_L
				  + model->BSIM3v1Swxj * Inv_W
				  + model->BSIM3v1Spxj * Inv_LW;
		  pParam->BSIM3v1Svsat = model->BSIM3v1Svsat
				    + model->BSIM3v1Slvsat * Inv_L
				    + model->BSIM3v1Swvsat * Inv_W
				    + model->BSIM3v1Spvsat * Inv_LW;
		  pParam->BSIM3v1Sat = model->BSIM3v1Sat
				  + model->BSIM3v1Slat * Inv_L
				  + model->BSIM3v1Swat * Inv_W
				  + model->BSIM3v1Spat * Inv_LW;
		  pParam->BSIM3v1Sa0 = model->BSIM3v1Sa0
				  + model->BSIM3v1Sla0 * Inv_L
				  + model->BSIM3v1Swa0 * Inv_W
				  + model->BSIM3v1Spa0 * Inv_LW; 
				  
		  pParam->BSIM3v1Sags = model->BSIM3v1Sags
				  + model->BSIM3v1Slags * Inv_L
				  + model->BSIM3v1Swags * Inv_W
				  + model->BSIM3v1Spags * Inv_LW;
				  
		  pParam->BSIM3v1Sa1 = model->BSIM3v1Sa1
				  + model->BSIM3v1Sla1 * Inv_L
				  + model->BSIM3v1Swa1 * Inv_W
				  + model->BSIM3v1Spa1 * Inv_LW;
		  pParam->BSIM3v1Sa2 = model->BSIM3v1Sa2
				  + model->BSIM3v1Sla2 * Inv_L
				  + model->BSIM3v1Swa2 * Inv_W
				  + model->BSIM3v1Spa2 * Inv_LW;
		  pParam->BSIM3v1Sketa = model->BSIM3v1Sketa
				    + model->BSIM3v1Slketa * Inv_L
				    + model->BSIM3v1Swketa * Inv_W
				    + model->BSIM3v1Spketa * Inv_LW;
		  pParam->BSIM3v1Snsub = model->BSIM3v1Snsub
				    + model->BSIM3v1Slnsub * Inv_L
				    + model->BSIM3v1Swnsub * Inv_W
				    + model->BSIM3v1Spnsub * Inv_LW;
		  pParam->BSIM3v1Snpeak = model->BSIM3v1Snpeak
				     + model->BSIM3v1Slnpeak * Inv_L
				     + model->BSIM3v1Swnpeak * Inv_W
				     + model->BSIM3v1Spnpeak * Inv_LW;
		  pParam->BSIM3v1Sngate = model->BSIM3v1Sngate
				     + model->BSIM3v1Slngate * Inv_L
				     + model->BSIM3v1Swngate * Inv_W
				     + model->BSIM3v1Spngate * Inv_LW;
		  pParam->BSIM3v1Sgamma1 = model->BSIM3v1Sgamma1
				      + model->BSIM3v1Slgamma1 * Inv_L
				      + model->BSIM3v1Swgamma1 * Inv_W
				      + model->BSIM3v1Spgamma1 * Inv_LW;
		  pParam->BSIM3v1Sgamma2 = model->BSIM3v1Sgamma2
				      + model->BSIM3v1Slgamma2 * Inv_L
				      + model->BSIM3v1Swgamma2 * Inv_W
				      + model->BSIM3v1Spgamma2 * Inv_LW;
		  pParam->BSIM3v1Svbx = model->BSIM3v1Svbx
				   + model->BSIM3v1Slvbx * Inv_L
				   + model->BSIM3v1Swvbx * Inv_W
				   + model->BSIM3v1Spvbx * Inv_LW;
		  pParam->BSIM3v1Svbm = model->BSIM3v1Svbm
				   + model->BSIM3v1Slvbm * Inv_L
				   + model->BSIM3v1Swvbm * Inv_W
				   + model->BSIM3v1Spvbm * Inv_LW;
		  pParam->BSIM3v1Sxt = model->BSIM3v1Sxt
				   + model->BSIM3v1Slxt * Inv_L
				   + model->BSIM3v1Swxt * Inv_W
				   + model->BSIM3v1Spxt * Inv_LW;
		  pParam->BSIM3v1Sk1 = model->BSIM3v1Sk1
				  + model->BSIM3v1Slk1 * Inv_L
				  + model->BSIM3v1Swk1 * Inv_W
				  + model->BSIM3v1Spk1 * Inv_LW;
		  pParam->BSIM3v1Skt1 = model->BSIM3v1Skt1
				   + model->BSIM3v1Slkt1 * Inv_L
				   + model->BSIM3v1Swkt1 * Inv_W
				   + model->BSIM3v1Spkt1 * Inv_LW;
		  pParam->BSIM3v1Skt1l = model->BSIM3v1Skt1l
				    + model->BSIM3v1Slkt1l * Inv_L
				    + model->BSIM3v1Swkt1l * Inv_W
				    + model->BSIM3v1Spkt1l * Inv_LW;
		  pParam->BSIM3v1Sk2 = model->BSIM3v1Sk2
				  + model->BSIM3v1Slk2 * Inv_L
				  + model->BSIM3v1Swk2 * Inv_W
				  + model->BSIM3v1Spk2 * Inv_LW;
		  pParam->BSIM3v1Skt2 = model->BSIM3v1Skt2
				   + model->BSIM3v1Slkt2 * Inv_L
				   + model->BSIM3v1Swkt2 * Inv_W
				   + model->BSIM3v1Spkt2 * Inv_LW;
		  pParam->BSIM3v1Sk3 = model->BSIM3v1Sk3
				  + model->BSIM3v1Slk3 * Inv_L
				  + model->BSIM3v1Swk3 * Inv_W
				  + model->BSIM3v1Spk3 * Inv_LW;
		  pParam->BSIM3v1Sk3b = model->BSIM3v1Sk3b
				   + model->BSIM3v1Slk3b * Inv_L
				   + model->BSIM3v1Swk3b * Inv_W
				   + model->BSIM3v1Spk3b * Inv_LW;
		  pParam->BSIM3v1Sw0 = model->BSIM3v1Sw0
				  + model->BSIM3v1Slw0 * Inv_L
				  + model->BSIM3v1Sww0 * Inv_W
				  + model->BSIM3v1Spw0 * Inv_LW;
		  pParam->BSIM3v1Snlx = model->BSIM3v1Snlx
				   + model->BSIM3v1Slnlx * Inv_L
				   + model->BSIM3v1Swnlx * Inv_W
				   + model->BSIM3v1Spnlx * Inv_LW;
		  pParam->BSIM3v1Sdvt0 = model->BSIM3v1Sdvt0
				    + model->BSIM3v1Sldvt0 * Inv_L
				    + model->BSIM3v1Swdvt0 * Inv_W
				    + model->BSIM3v1Spdvt0 * Inv_LW;
		  pParam->BSIM3v1Sdvt1 = model->BSIM3v1Sdvt1
				    + model->BSIM3v1Sldvt1 * Inv_L
				    + model->BSIM3v1Swdvt1 * Inv_W
				    + model->BSIM3v1Spdvt1 * Inv_LW;
		  pParam->BSIM3v1Sdvt2 = model->BSIM3v1Sdvt2
				    + model->BSIM3v1Sldvt2 * Inv_L
				    + model->BSIM3v1Swdvt2 * Inv_W
				    + model->BSIM3v1Spdvt2 * Inv_LW;
		  pParam->BSIM3v1Sdvt0w = model->BSIM3v1Sdvt0w
				    + model->BSIM3v1Sldvt0w * Inv_L
				    + model->BSIM3v1Swdvt0w * Inv_W
				    + model->BSIM3v1Spdvt0w * Inv_LW;
		  pParam->BSIM3v1Sdvt1w = model->BSIM3v1Sdvt1w
				    + model->BSIM3v1Sldvt1w * Inv_L
				    + model->BSIM3v1Swdvt1w * Inv_W
				    + model->BSIM3v1Spdvt1w * Inv_LW;
		  pParam->BSIM3v1Sdvt2w = model->BSIM3v1Sdvt2w
				    + model->BSIM3v1Sldvt2w * Inv_L
				    + model->BSIM3v1Swdvt2w * Inv_W
				    + model->BSIM3v1Spdvt2w * Inv_LW;
		  pParam->BSIM3v1Sdrout = model->BSIM3v1Sdrout
				     + model->BSIM3v1Sldrout * Inv_L
				     + model->BSIM3v1Swdrout * Inv_W
				     + model->BSIM3v1Spdrout * Inv_LW;
		  pParam->BSIM3v1Sdsub = model->BSIM3v1Sdsub
				    + model->BSIM3v1Sldsub * Inv_L
				    + model->BSIM3v1Swdsub * Inv_W
				    + model->BSIM3v1Spdsub * Inv_LW;
		  pParam->BSIM3v1Svth0 = model->BSIM3v1Svth0
				    + model->BSIM3v1Slvth0 * Inv_L
				    + model->BSIM3v1Swvth0 * Inv_W
				    + model->BSIM3v1Spvth0 * Inv_LW;
		  pParam->BSIM3v1Sua = model->BSIM3v1Sua
				  + model->BSIM3v1Slua * Inv_L
				  + model->BSIM3v1Swua * Inv_W
				  + model->BSIM3v1Spua * Inv_LW;
		  pParam->BSIM3v1Sua1 = model->BSIM3v1Sua1
				   + model->BSIM3v1Slua1 * Inv_L
				   + model->BSIM3v1Swua1 * Inv_W
				   + model->BSIM3v1Spua1 * Inv_LW;
		  pParam->BSIM3v1Sub = model->BSIM3v1Sub
				  + model->BSIM3v1Slub * Inv_L
				  + model->BSIM3v1Swub * Inv_W
				  + model->BSIM3v1Spub * Inv_LW;
		  pParam->BSIM3v1Sub1 = model->BSIM3v1Sub1
				   + model->BSIM3v1Slub1 * Inv_L
				   + model->BSIM3v1Swub1 * Inv_W
				   + model->BSIM3v1Spub1 * Inv_LW;
		  pParam->BSIM3v1Suc = model->BSIM3v1Suc
				  + model->BSIM3v1Sluc * Inv_L
				  + model->BSIM3v1Swuc * Inv_W
				  + model->BSIM3v1Spuc * Inv_LW;
		  pParam->BSIM3v1Suc1 = model->BSIM3v1Suc1
				   + model->BSIM3v1Sluc1 * Inv_L
				   + model->BSIM3v1Swuc1 * Inv_W
				   + model->BSIM3v1Spuc1 * Inv_LW;
		  pParam->BSIM3v1Su0 = model->BSIM3v1Su0
				  + model->BSIM3v1Slu0 * Inv_L
				  + model->BSIM3v1Swu0 * Inv_W
				  + model->BSIM3v1Spu0 * Inv_LW;
		  pParam->BSIM3v1Sute = model->BSIM3v1Sute
				   + model->BSIM3v1Slute * Inv_L
				   + model->BSIM3v1Swute * Inv_W
				   + model->BSIM3v1Spute * Inv_LW;
		  pParam->BSIM3v1Svoff = model->BSIM3v1Svoff
				    + model->BSIM3v1Slvoff * Inv_L
				    + model->BSIM3v1Swvoff * Inv_W
				    + model->BSIM3v1Spvoff * Inv_LW;
		  pParam->BSIM3v1Sdelta = model->BSIM3v1Sdelta
				     + model->BSIM3v1Sldelta * Inv_L
				     + model->BSIM3v1Swdelta * Inv_W
				     + model->BSIM3v1Spdelta * Inv_LW;
		  pParam->BSIM3v1Srdsw = model->BSIM3v1Srdsw
				    + model->BSIM3v1Slrdsw * Inv_L
				    + model->BSIM3v1Swrdsw * Inv_W
				    + model->BSIM3v1Sprdsw * Inv_LW;
		  pParam->BSIM3v1Sprwg = model->BSIM3v1Sprwg
				    + model->BSIM3v1Slprwg * Inv_L
				    + model->BSIM3v1Swprwg * Inv_W
				    + model->BSIM3v1Spprwg * Inv_LW;
		  pParam->BSIM3v1Sprwb = model->BSIM3v1Sprwb
				    + model->BSIM3v1Slprwb * Inv_L
				    + model->BSIM3v1Swprwb * Inv_W
				    + model->BSIM3v1Spprwb * Inv_LW;
		  pParam->BSIM3v1Sprt = model->BSIM3v1Sprt
				    + model->BSIM3v1Slprt * Inv_L
				    + model->BSIM3v1Swprt * Inv_W
				    + model->BSIM3v1Spprt * Inv_LW;
		  pParam->BSIM3v1Seta0 = model->BSIM3v1Seta0
				    + model->BSIM3v1Sleta0 * Inv_L
				    + model->BSIM3v1Sweta0 * Inv_W
				    + model->BSIM3v1Speta0 * Inv_LW;
		  pParam->BSIM3v1Setab = model->BSIM3v1Setab
				    + model->BSIM3v1Sletab * Inv_L
				    + model->BSIM3v1Swetab * Inv_W
				    + model->BSIM3v1Spetab * Inv_LW;
		  pParam->BSIM3v1Spclm = model->BSIM3v1Spclm
				    + model->BSIM3v1Slpclm * Inv_L
				    + model->BSIM3v1Swpclm * Inv_W
				    + model->BSIM3v1Sppclm * Inv_LW;
		  pParam->BSIM3v1Spdibl1 = model->BSIM3v1Spdibl1
				      + model->BSIM3v1Slpdibl1 * Inv_L
				      + model->BSIM3v1Swpdibl1 * Inv_W
				      + model->BSIM3v1Sppdibl1 * Inv_LW;
		  pParam->BSIM3v1Spdibl2 = model->BSIM3v1Spdibl2
				      + model->BSIM3v1Slpdibl2 * Inv_L
				      + model->BSIM3v1Swpdibl2 * Inv_W
				      + model->BSIM3v1Sppdibl2 * Inv_LW;
		  pParam->BSIM3v1Spdiblb = model->BSIM3v1Spdiblb
				      + model->BSIM3v1Slpdiblb * Inv_L
				      + model->BSIM3v1Swpdiblb * Inv_W
				      + model->BSIM3v1Sppdiblb * Inv_LW;
		  pParam->BSIM3v1Spscbe1 = model->BSIM3v1Spscbe1
				      + model->BSIM3v1Slpscbe1 * Inv_L
				      + model->BSIM3v1Swpscbe1 * Inv_W
				      + model->BSIM3v1Sppscbe1 * Inv_LW;
		  pParam->BSIM3v1Spscbe2 = model->BSIM3v1Spscbe2
				      + model->BSIM3v1Slpscbe2 * Inv_L
				      + model->BSIM3v1Swpscbe2 * Inv_W
				      + model->BSIM3v1Sppscbe2 * Inv_LW;
		  pParam->BSIM3v1Spvag = model->BSIM3v1Spvag
				    + model->BSIM3v1Slpvag * Inv_L
				    + model->BSIM3v1Swpvag * Inv_W
				    + model->BSIM3v1Sppvag * Inv_LW;
		  pParam->BSIM3v1Swr = model->BSIM3v1Swr
				  + model->BSIM3v1Slwr * Inv_L
				  + model->BSIM3v1Swwr * Inv_W
				  + model->BSIM3v1Spwr * Inv_LW;
		  pParam->BSIM3v1Sdwg = model->BSIM3v1Sdwg
				   + model->BSIM3v1Sldwg * Inv_L
				   + model->BSIM3v1Swdwg * Inv_W
				   + model->BSIM3v1Spdwg * Inv_LW;
		  pParam->BSIM3v1Sdwb = model->BSIM3v1Sdwb
				   + model->BSIM3v1Sldwb * Inv_L
				   + model->BSIM3v1Swdwb * Inv_W
				   + model->BSIM3v1Spdwb * Inv_LW;
		  pParam->BSIM3v1Sb0 = model->BSIM3v1Sb0
				  + model->BSIM3v1Slb0 * Inv_L
				  + model->BSIM3v1Swb0 * Inv_W
				  + model->BSIM3v1Spb0 * Inv_LW;
		  pParam->BSIM3v1Sb1 = model->BSIM3v1Sb1
				  + model->BSIM3v1Slb1 * Inv_L
				  + model->BSIM3v1Swb1 * Inv_W
				  + model->BSIM3v1Spb1 * Inv_LW;
		  pParam->BSIM3v1Salpha0 = model->BSIM3v1Salpha0
				      + model->BSIM3v1Slalpha0 * Inv_L
				      + model->BSIM3v1Swalpha0 * Inv_W
				      + model->BSIM3v1Spalpha0 * Inv_LW;
		  pParam->BSIM3v1Sbeta0 = model->BSIM3v1Sbeta0
				     + model->BSIM3v1Slbeta0 * Inv_L
				     + model->BSIM3v1Swbeta0 * Inv_W
				     + model->BSIM3v1Spbeta0 * Inv_LW;
		  /* CV model */
		  pParam->BSIM3v1Selm = model->BSIM3v1Selm
				  + model->BSIM3v1Slelm * Inv_L
				  + model->BSIM3v1Swelm * Inv_W
				  + model->BSIM3v1Spelm * Inv_LW;
		  pParam->BSIM3v1Scgsl = model->BSIM3v1Scgsl
				    + model->BSIM3v1Slcgsl * Inv_L
				    + model->BSIM3v1Swcgsl * Inv_W
				    + model->BSIM3v1Spcgsl * Inv_LW;
		  pParam->BSIM3v1Scgdl = model->BSIM3v1Scgdl
				    + model->BSIM3v1Slcgdl * Inv_L
				    + model->BSIM3v1Swcgdl * Inv_W
				    + model->BSIM3v1Spcgdl * Inv_LW;
		  pParam->BSIM3v1Sckappa = model->BSIM3v1Sckappa
				      + model->BSIM3v1Slckappa * Inv_L
				      + model->BSIM3v1Swckappa * Inv_W
				      + model->BSIM3v1Spckappa * Inv_LW;
		  pParam->BSIM3v1Scf = model->BSIM3v1Scf
				  + model->BSIM3v1Slcf * Inv_L
				  + model->BSIM3v1Swcf * Inv_W
				  + model->BSIM3v1Spcf * Inv_LW;
		  pParam->BSIM3v1Sclc = model->BSIM3v1Sclc
				   + model->BSIM3v1Slclc * Inv_L
				   + model->BSIM3v1Swclc * Inv_W
				   + model->BSIM3v1Spclc * Inv_LW;
		  pParam->BSIM3v1Scle = model->BSIM3v1Scle
				   + model->BSIM3v1Slcle * Inv_L
				   + model->BSIM3v1Swcle * Inv_W
				   + model->BSIM3v1Spcle * Inv_LW;
		  pParam->BSIM3v1Svfbcv = model->BSIM3v1Svfbcv
				  + model->BSIM3v1Slvfbcv * Inv_L
				  + model->BSIM3v1Swvfbcv * Inv_W
				  + model->BSIM3v1Spvfbcv * Inv_LW;
                  pParam->BSIM3v1SabulkCVfactor = 1.0 + pow((pParam->BSIM3v1Sclc
					     / pParam->BSIM3v1Sleff),
					     pParam->BSIM3v1Scle);

	          T0 = (TRatio - 1.0);
	          pParam->BSIM3v1Sua = pParam->BSIM3v1Sua + pParam->BSIM3v1Sua1 * T0;
	          pParam->BSIM3v1Sub = pParam->BSIM3v1Sub + pParam->BSIM3v1Sub1 * T0;
	          pParam->BSIM3v1Suc = pParam->BSIM3v1Suc + pParam->BSIM3v1Suc1 * T0;
                  if (pParam->BSIM3v1Su0 > 1.0) 
                      pParam->BSIM3v1Su0 = pParam->BSIM3v1Su0 / 1.0e4;

                  pParam->BSIM3v1Su0temp = pParam->BSIM3v1Su0
				      * pow(TRatio, pParam->BSIM3v1Sute); 
                  pParam->BSIM3v1Svsattemp = pParam->BSIM3v1Svsat - pParam->BSIM3v1Sat 
			                * T0;
	          pParam->BSIM3v1Srds0 = (pParam->BSIM3v1Srdsw + pParam->BSIM3v1Sprt * T0)
                                    / pow(pParam->BSIM3v1Sweff * 1E6, pParam->BSIM3v1Swr);

		  if (BSIM3v1ScheckModel(model, here, ckt))
		  {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1SmodName;
                      namarray[1] = here->BSIM3v1Sname;
                      (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM3v1SV3 parameter checking for %s in model %s", namarray);
                      return(E_BADPARM);   
		  }

                  pParam->BSIM3v1Scgdo = (model->BSIM3v1Scgdo + pParam->BSIM3v1Scf)
				    * pParam->BSIM3v1SweffCV;
                  pParam->BSIM3v1Scgso = (model->BSIM3v1Scgso + pParam->BSIM3v1Scf)
				    * pParam->BSIM3v1SweffCV;
                  pParam->BSIM3v1Scgbo = model->BSIM3v1Scgbo * pParam->BSIM3v1SleffCV;

                  if (!model->BSIM3v1SnpeakGiven && model->BSIM3v1Sgamma1Given)
                  {   T0 = pParam->BSIM3v1Sgamma1 * model->BSIM3v1Scox;
                      pParam->BSIM3v1Snpeak = 3.021E22 * T0 * T0;
                  }

		  pParam->BSIM3v1Sphi = 2.0 * Vtm0 
			           * log(pParam->BSIM3v1Snpeak / ni);

	          pParam->BSIM3v1SsqrtPhi = sqrt(pParam->BSIM3v1Sphi);
	          pParam->BSIM3v1Sphis3 = pParam->BSIM3v1SsqrtPhi * pParam->BSIM3v1Sphi;

                  pParam->BSIM3v1SXdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM3v1Snpeak * 1.0e6))
                                     * pParam->BSIM3v1SsqrtPhi; 
                  pParam->BSIM3v1SsqrtXdep0 = sqrt(pParam->BSIM3v1SXdep0);
                  pParam->BSIM3v1Slitl = sqrt(3.0 * pParam->BSIM3v1Sxj
				    * model->BSIM3v1Stox);
                  pParam->BSIM3v1Svbi = Vtm0 * log(1.0e20
			           * pParam->BSIM3v1Snpeak / (ni * ni));
                  pParam->BSIM3v1Scdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM3v1Snpeak * 1.0e6 / 2.0
				     / pParam->BSIM3v1Sphi);
        
                  if (model->BSIM3v1Sk1Given || model->BSIM3v1Sk2Given)
	          {   if (!model->BSIM3v1Sk1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3v1Sk1 = 0.53;
                      }
                      if (!model->BSIM3v1Sk2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3v1Sk2 = -0.0186;
                      }
                      if (model->BSIM3v1SnsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1SxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1SvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1SvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1Sgamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1Sgamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM3v1SvbxGiven)
                          pParam->BSIM3v1Svbx = pParam->BSIM3v1Sphi - 7.7348e-4 
                                           * pParam->BSIM3v1Snpeak
					   * pParam->BSIM3v1Sxt * pParam->BSIM3v1Sxt;
	              if (pParam->BSIM3v1Svbx > 0.0)
		          pParam->BSIM3v1Svbx = -pParam->BSIM3v1Svbx;
	              if (pParam->BSIM3v1Svbm > 0.0)
                          pParam->BSIM3v1Svbm = -pParam->BSIM3v1Svbm;
           
                      if (!model->BSIM3v1Sgamma1Given)
                          pParam->BSIM3v1Sgamma1 = 5.753e-12
					      * sqrt(pParam->BSIM3v1Snpeak)
                                              / model->BSIM3v1Scox;
                      if (!model->BSIM3v1Sgamma2Given)
                          pParam->BSIM3v1Sgamma2 = 5.753e-12
					      * sqrt(pParam->BSIM3v1Snsub)
                                              / model->BSIM3v1Scox;

                      T0 = pParam->BSIM3v1Sgamma1 - pParam->BSIM3v1Sgamma2;
                      T1 = sqrt(pParam->BSIM3v1Sphi - pParam->BSIM3v1Svbx)
			 - pParam->BSIM3v1SsqrtPhi;
                      T2 = sqrt(pParam->BSIM3v1Sphi * (pParam->BSIM3v1Sphi
			 - pParam->BSIM3v1Svbm)) - pParam->BSIM3v1Sphi;
                      pParam->BSIM3v1Sk2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3v1Svbm);
                      pParam->BSIM3v1Sk1 = pParam->BSIM3v1Sgamma2 - 2.0
				      * pParam->BSIM3v1Sk2 * sqrt(pParam->BSIM3v1Sphi
				      - pParam->BSIM3v1Svbm);
                  }
 
		  if (pParam->BSIM3v1Sk2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM3v1Sk1 / pParam->BSIM3v1Sk2;
                      pParam->BSIM3v1Svbsc = 0.9 * (pParam->BSIM3v1Sphi - T0 * T0);
		      if (pParam->BSIM3v1Svbsc > -3.0)
		          pParam->BSIM3v1Svbsc = -3.0;
		      else if (pParam->BSIM3v1Svbsc < -30.0)
		          pParam->BSIM3v1Svbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM3v1Svbsc = -30.0;
		  }
		  if (pParam->BSIM3v1Svbsc > pParam->BSIM3v1Svbm)
		      pParam->BSIM3v1Svbsc = pParam->BSIM3v1Svbm;

	          if (model->BSIM3v1Svth0Given)
		  {   pParam->BSIM3v1Svfb = model->BSIM3v1Stype * pParam->BSIM3v1Svth0 
                                       - pParam->BSIM3v1Sphi - pParam->BSIM3v1Sk1 
                                       * pParam->BSIM3v1SsqrtPhi;
		  }
		  else
		  {   pParam->BSIM3v1Svfb = -1.0;
		      pParam->BSIM3v1Svth0 = model->BSIM3v1Stype * (pParam->BSIM3v1Svfb
                                        + pParam->BSIM3v1Sphi + pParam->BSIM3v1Sk1 
                                        * pParam->BSIM3v1SsqrtPhi);
		  }
                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3v1Stox
		     * pParam->BSIM3v1SXdep0);
                  T0 = exp(-0.5 * pParam->BSIM3v1Sdsub * pParam->BSIM3v1Sleff / T1);
                  pParam->BSIM3v1Stheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3v1Sdrout * pParam->BSIM3v1Sleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3v1SthetaRout = pParam->BSIM3v1Spdibl1 * T2
				         + pParam->BSIM3v1Spdibl2;
	      }

              /* process source/drain series resistance */
              here->BSIM3v1SdrainConductance = model->BSIM3v1SsheetResistance 
		                              * here->BSIM3v1SdrainSquares;
              if (here->BSIM3v1SdrainConductance > 0.0)
                  here->BSIM3v1SdrainConductance = 1.0
					      / here->BSIM3v1SdrainConductance;
	      else
                  here->BSIM3v1SdrainConductance = 0.0;
                  
              here->BSIM3v1SsourceConductance = model->BSIM3v1SsheetResistance 
		                           * here->BSIM3v1SsourceSquares;
              if (here->BSIM3v1SsourceConductance > 0.0) 
                  here->BSIM3v1SsourceConductance = 1.0
					       / here->BSIM3v1SsourceConductance;
	      else
                  here->BSIM3v1SsourceConductance = 0.0;
	      here->BSIM3v1Scgso = pParam->BSIM3v1Scgso;
	      here->BSIM3v1Scgdo = pParam->BSIM3v1Scgdo;
         }
    }
    return(OK);
}
