/* $Id$  */
/* 
$Log$
Revision 1.1  2000-04-27 20:03:59  pnenzi
Initial revision

 * Revision 3.1  96/12/08  19:59:49  yuhua
 * BSIM3v3.1 release

 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/***********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1temp.c
**********/
/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1def.h"
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
BSIM3V1temp(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
register BSIM3V1model *model = (BSIM3V1model*) inModel;
register BSIM3V1instance *here;
struct bsim3v1SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam;
double tmp, tmp1, tmp2, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Dw, Dl, Vtm0, Tnom;
int Size_Not_Found;

    /*  loop through all the BSIM3V1 device models */
    for (; model != NULL; model = model->BSIM3V1nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3V1bulkJctPotential < 0.1)  
	     model->BSIM3V1bulkJctPotential = 0.1;
         if (model->BSIM3V1sidewallJctPotential < 0.1)
             model->BSIM3V1sidewallJctPotential = 0.1;
         if (model->BSIM3V1GatesidewallJctPotential < 0.1)
             model->BSIM3V1GatesidewallJctPotential = 0.1;
         model->pSizeDependParamKnot = NULL;
	 pLastKnot = NULL;

	 Tnom = model->BSIM3V1tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM3V1vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3V1factor1 = sqrt(EPSSI / EPSOX * model->BSIM3V1tox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3V1vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3V1vtm + model->BSIM3V1jctTempExponent
		* log(Temp / Tnom);
	     T1 = exp(T0 / model->BSIM3V1jctEmissionCoeff);
	     model->BSIM3V1jctTempSatCurDensity = model->BSIM3V1jctSatCurDensity
					      * T1;
	     model->BSIM3V1jctSidewallTempSatCurDensity
			 = model->BSIM3V1jctSidewallSatCurDensity * T1;
	 }
	 else
	 {   model->BSIM3V1jctTempSatCurDensity = model->BSIM3V1jctSatCurDensity;
	     model->BSIM3V1jctSidewallTempSatCurDensity
			= model->BSIM3V1jctSidewallSatCurDensity;
	 }

	 if (model->BSIM3V1jctTempSatCurDensity < 0.0)
	     model->BSIM3V1jctTempSatCurDensity = 0.0;
	 if (model->BSIM3V1jctSidewallTempSatCurDensity < 0.0)
	     model->BSIM3V1jctSidewallTempSatCurDensity = 0.0;

         /* loop through all the instances of the model */
	 /* MCJ: Length and Width not initialized */
         for (here = model->BSIM3V1instances; here != NULL;
              here = here->BSIM3V1nextInstance) 
	 {    
              if (here->BSIM3V1owner != ARCHme) continue;
              pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM3V1l == pSizeDependParamKnot->Length)
		      && (here->BSIM3V1w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      if (Size_Not_Found)
	      {   pParam = (struct bsim3v1SizeDependParam *)malloc(
	                    sizeof(struct bsim3v1SizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

		  Ldrn = here->BSIM3V1l;
		  Wdrn = here->BSIM3V1w;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
		  
                  T0 = pow(Ldrn, model->BSIM3V1Lln);
                  T1 = pow(Wdrn, model->BSIM3V1Lwn);
                  tmp1 = model->BSIM3V1Ll / T0 + model->BSIM3V1Lw / T1
                       + model->BSIM3V1Lwl / (T0 * T1);
                  pParam->BSIM3V1dl = model->BSIM3V1Lint + tmp1;
                  pParam->BSIM3V1dlc = model->BSIM3V1dlc + tmp1;

                  T2 = pow(Ldrn, model->BSIM3V1Wln);
                  T3 = pow(Wdrn, model->BSIM3V1Wwn);
                  tmp2 = model->BSIM3V1Wl / T2 + model->BSIM3V1Ww / T3
                       + model->BSIM3V1Wwl / (T2 * T3);
                  pParam->BSIM3V1dw = model->BSIM3V1Wint + tmp2;
                  pParam->BSIM3V1dwc = model->BSIM3V1dwc + tmp2;

                  pParam->BSIM3V1leff = here->BSIM3V1l - 2.0 * pParam->BSIM3V1dl;
                  if (pParam->BSIM3V1leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V1modName;
                      namarray[1] = here->BSIM3V1name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V1: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V1weff = here->BSIM3V1w - 2.0 * pParam->BSIM3V1dw;
                  if (pParam->BSIM3V1weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V1modName;
                      namarray[1] = here->BSIM3V1name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V1: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V1leffCV = here->BSIM3V1l - 2.0 * pParam->BSIM3V1dlc;
                  if (pParam->BSIM3V1leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V1modName;
                      namarray[1] = here->BSIM3V1name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V1: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V1weffCV = here->BSIM3V1w - 2.0 * pParam->BSIM3V1dwc;
                  if (pParam->BSIM3V1weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V1modName;
                      namarray[1] = here->BSIM3V1name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V1: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM3V1binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM3V1leff;
		      Inv_W = 1.0e-6 / pParam->BSIM3V1weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM3V1leff
			     * pParam->BSIM3V1weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM3V1leff;
		      Inv_W = 1.0 / pParam->BSIM3V1weff;
		      Inv_LW = 1.0 / (pParam->BSIM3V1leff
			     * pParam->BSIM3V1weff);
		  }
		  pParam->BSIM3V1cdsc = model->BSIM3V1cdsc
				    + model->BSIM3V1lcdsc * Inv_L
				    + model->BSIM3V1wcdsc * Inv_W
				    + model->BSIM3V1pcdsc * Inv_LW;
		  pParam->BSIM3V1cdscb = model->BSIM3V1cdscb
				     + model->BSIM3V1lcdscb * Inv_L
				     + model->BSIM3V1wcdscb * Inv_W
				     + model->BSIM3V1pcdscb * Inv_LW; 
				     
    		  pParam->BSIM3V1cdscd = model->BSIM3V1cdscd
				     + model->BSIM3V1lcdscd * Inv_L
				     + model->BSIM3V1wcdscd * Inv_W
				     + model->BSIM3V1pcdscd * Inv_LW; 
				     
		  pParam->BSIM3V1cit = model->BSIM3V1cit
				   + model->BSIM3V1lcit * Inv_L
				   + model->BSIM3V1wcit * Inv_W
				   + model->BSIM3V1pcit * Inv_LW;
		  pParam->BSIM3V1nfactor = model->BSIM3V1nfactor
				       + model->BSIM3V1lnfactor * Inv_L
				       + model->BSIM3V1wnfactor * Inv_W
				       + model->BSIM3V1pnfactor * Inv_LW;
		  pParam->BSIM3V1xj = model->BSIM3V1xj
				  + model->BSIM3V1lxj * Inv_L
				  + model->BSIM3V1wxj * Inv_W
				  + model->BSIM3V1pxj * Inv_LW;
		  pParam->BSIM3V1vsat = model->BSIM3V1vsat
				    + model->BSIM3V1lvsat * Inv_L
				    + model->BSIM3V1wvsat * Inv_W
				    + model->BSIM3V1pvsat * Inv_LW;
		  pParam->BSIM3V1at = model->BSIM3V1at
				  + model->BSIM3V1lat * Inv_L
				  + model->BSIM3V1wat * Inv_W
				  + model->BSIM3V1pat * Inv_LW;
		  pParam->BSIM3V1a0 = model->BSIM3V1a0
				  + model->BSIM3V1la0 * Inv_L
				  + model->BSIM3V1wa0 * Inv_W
				  + model->BSIM3V1pa0 * Inv_LW; 
				  
		  pParam->BSIM3V1ags = model->BSIM3V1ags
				  + model->BSIM3V1lags * Inv_L
				  + model->BSIM3V1wags * Inv_W
				  + model->BSIM3V1pags * Inv_LW;
				  
		  pParam->BSIM3V1a1 = model->BSIM3V1a1
				  + model->BSIM3V1la1 * Inv_L
				  + model->BSIM3V1wa1 * Inv_W
				  + model->BSIM3V1pa1 * Inv_LW;
		  pParam->BSIM3V1a2 = model->BSIM3V1a2
				  + model->BSIM3V1la2 * Inv_L
				  + model->BSIM3V1wa2 * Inv_W
				  + model->BSIM3V1pa2 * Inv_LW;
		  pParam->BSIM3V1keta = model->BSIM3V1keta
				    + model->BSIM3V1lketa * Inv_L
				    + model->BSIM3V1wketa * Inv_W
				    + model->BSIM3V1pketa * Inv_LW;
		  pParam->BSIM3V1nsub = model->BSIM3V1nsub
				    + model->BSIM3V1lnsub * Inv_L
				    + model->BSIM3V1wnsub * Inv_W
				    + model->BSIM3V1pnsub * Inv_LW;
		  pParam->BSIM3V1npeak = model->BSIM3V1npeak
				     + model->BSIM3V1lnpeak * Inv_L
				     + model->BSIM3V1wnpeak * Inv_W
				     + model->BSIM3V1pnpeak * Inv_LW;
		  pParam->BSIM3V1ngate = model->BSIM3V1ngate
				     + model->BSIM3V1lngate * Inv_L
				     + model->BSIM3V1wngate * Inv_W
				     + model->BSIM3V1pngate * Inv_LW;
		  pParam->BSIM3V1gamma1 = model->BSIM3V1gamma1
				      + model->BSIM3V1lgamma1 * Inv_L
				      + model->BSIM3V1wgamma1 * Inv_W
				      + model->BSIM3V1pgamma1 * Inv_LW;
		  pParam->BSIM3V1gamma2 = model->BSIM3V1gamma2
				      + model->BSIM3V1lgamma2 * Inv_L
				      + model->BSIM3V1wgamma2 * Inv_W
				      + model->BSIM3V1pgamma2 * Inv_LW;
		  pParam->BSIM3V1vbx = model->BSIM3V1vbx
				   + model->BSIM3V1lvbx * Inv_L
				   + model->BSIM3V1wvbx * Inv_W
				   + model->BSIM3V1pvbx * Inv_LW;
		  pParam->BSIM3V1vbm = model->BSIM3V1vbm
				   + model->BSIM3V1lvbm * Inv_L
				   + model->BSIM3V1wvbm * Inv_W
				   + model->BSIM3V1pvbm * Inv_LW;
		  pParam->BSIM3V1xt = model->BSIM3V1xt
				   + model->BSIM3V1lxt * Inv_L
				   + model->BSIM3V1wxt * Inv_W
				   + model->BSIM3V1pxt * Inv_LW;
		  pParam->BSIM3V1k1 = model->BSIM3V1k1
				  + model->BSIM3V1lk1 * Inv_L
				  + model->BSIM3V1wk1 * Inv_W
				  + model->BSIM3V1pk1 * Inv_LW;
		  pParam->BSIM3V1kt1 = model->BSIM3V1kt1
				   + model->BSIM3V1lkt1 * Inv_L
				   + model->BSIM3V1wkt1 * Inv_W
				   + model->BSIM3V1pkt1 * Inv_LW;
		  pParam->BSIM3V1kt1l = model->BSIM3V1kt1l
				    + model->BSIM3V1lkt1l * Inv_L
				    + model->BSIM3V1wkt1l * Inv_W
				    + model->BSIM3V1pkt1l * Inv_LW;
		  pParam->BSIM3V1k2 = model->BSIM3V1k2
				  + model->BSIM3V1lk2 * Inv_L
				  + model->BSIM3V1wk2 * Inv_W
				  + model->BSIM3V1pk2 * Inv_LW;
		  pParam->BSIM3V1kt2 = model->BSIM3V1kt2
				   + model->BSIM3V1lkt2 * Inv_L
				   + model->BSIM3V1wkt2 * Inv_W
				   + model->BSIM3V1pkt2 * Inv_LW;
		  pParam->BSIM3V1k3 = model->BSIM3V1k3
				  + model->BSIM3V1lk3 * Inv_L
				  + model->BSIM3V1wk3 * Inv_W
				  + model->BSIM3V1pk3 * Inv_LW;
		  pParam->BSIM3V1k3b = model->BSIM3V1k3b
				   + model->BSIM3V1lk3b * Inv_L
				   + model->BSIM3V1wk3b * Inv_W
				   + model->BSIM3V1pk3b * Inv_LW;
		  pParam->BSIM3V1w0 = model->BSIM3V1w0
				  + model->BSIM3V1lw0 * Inv_L
				  + model->BSIM3V1ww0 * Inv_W
				  + model->BSIM3V1pw0 * Inv_LW;
		  pParam->BSIM3V1nlx = model->BSIM3V1nlx
				   + model->BSIM3V1lnlx * Inv_L
				   + model->BSIM3V1wnlx * Inv_W
				   + model->BSIM3V1pnlx * Inv_LW;
		  pParam->BSIM3V1dvt0 = model->BSIM3V1dvt0
				    + model->BSIM3V1ldvt0 * Inv_L
				    + model->BSIM3V1wdvt0 * Inv_W
				    + model->BSIM3V1pdvt0 * Inv_LW;
		  pParam->BSIM3V1dvt1 = model->BSIM3V1dvt1
				    + model->BSIM3V1ldvt1 * Inv_L
				    + model->BSIM3V1wdvt1 * Inv_W
				    + model->BSIM3V1pdvt1 * Inv_LW;
		  pParam->BSIM3V1dvt2 = model->BSIM3V1dvt2
				    + model->BSIM3V1ldvt2 * Inv_L
				    + model->BSIM3V1wdvt2 * Inv_W
				    + model->BSIM3V1pdvt2 * Inv_LW;
		  pParam->BSIM3V1dvt0w = model->BSIM3V1dvt0w
				    + model->BSIM3V1ldvt0w * Inv_L
				    + model->BSIM3V1wdvt0w * Inv_W
				    + model->BSIM3V1pdvt0w * Inv_LW;
		  pParam->BSIM3V1dvt1w = model->BSIM3V1dvt1w
				    + model->BSIM3V1ldvt1w * Inv_L
				    + model->BSIM3V1wdvt1w * Inv_W
				    + model->BSIM3V1pdvt1w * Inv_LW;
		  pParam->BSIM3V1dvt2w = model->BSIM3V1dvt2w
				    + model->BSIM3V1ldvt2w * Inv_L
				    + model->BSIM3V1wdvt2w * Inv_W
				    + model->BSIM3V1pdvt2w * Inv_LW;
		  pParam->BSIM3V1drout = model->BSIM3V1drout
				     + model->BSIM3V1ldrout * Inv_L
				     + model->BSIM3V1wdrout * Inv_W
				     + model->BSIM3V1pdrout * Inv_LW;
		  pParam->BSIM3V1dsub = model->BSIM3V1dsub
				    + model->BSIM3V1ldsub * Inv_L
				    + model->BSIM3V1wdsub * Inv_W
				    + model->BSIM3V1pdsub * Inv_LW;
		  pParam->BSIM3V1vth0 = model->BSIM3V1vth0
				    + model->BSIM3V1lvth0 * Inv_L
				    + model->BSIM3V1wvth0 * Inv_W
				    + model->BSIM3V1pvth0 * Inv_LW;
		  pParam->BSIM3V1ua = model->BSIM3V1ua
				  + model->BSIM3V1lua * Inv_L
				  + model->BSIM3V1wua * Inv_W
				  + model->BSIM3V1pua * Inv_LW;
		  pParam->BSIM3V1ua1 = model->BSIM3V1ua1
				   + model->BSIM3V1lua1 * Inv_L
				   + model->BSIM3V1wua1 * Inv_W
				   + model->BSIM3V1pua1 * Inv_LW;
		  pParam->BSIM3V1ub = model->BSIM3V1ub
				  + model->BSIM3V1lub * Inv_L
				  + model->BSIM3V1wub * Inv_W
				  + model->BSIM3V1pub * Inv_LW;
		  pParam->BSIM3V1ub1 = model->BSIM3V1ub1
				   + model->BSIM3V1lub1 * Inv_L
				   + model->BSIM3V1wub1 * Inv_W
				   + model->BSIM3V1pub1 * Inv_LW;
		  pParam->BSIM3V1uc = model->BSIM3V1uc
				  + model->BSIM3V1luc * Inv_L
				  + model->BSIM3V1wuc * Inv_W
				  + model->BSIM3V1puc * Inv_LW;
		  pParam->BSIM3V1uc1 = model->BSIM3V1uc1
				   + model->BSIM3V1luc1 * Inv_L
				   + model->BSIM3V1wuc1 * Inv_W
				   + model->BSIM3V1puc1 * Inv_LW;
		  pParam->BSIM3V1u0 = model->BSIM3V1u0
				  + model->BSIM3V1lu0 * Inv_L
				  + model->BSIM3V1wu0 * Inv_W
				  + model->BSIM3V1pu0 * Inv_LW;
		  pParam->BSIM3V1ute = model->BSIM3V1ute
				   + model->BSIM3V1lute * Inv_L
				   + model->BSIM3V1wute * Inv_W
				   + model->BSIM3V1pute * Inv_LW;
		  pParam->BSIM3V1voff = model->BSIM3V1voff
				    + model->BSIM3V1lvoff * Inv_L
				    + model->BSIM3V1wvoff * Inv_W
				    + model->BSIM3V1pvoff * Inv_LW;
		  pParam->BSIM3V1delta = model->BSIM3V1delta
				     + model->BSIM3V1ldelta * Inv_L
				     + model->BSIM3V1wdelta * Inv_W
				     + model->BSIM3V1pdelta * Inv_LW;
		  pParam->BSIM3V1rdsw = model->BSIM3V1rdsw
				    + model->BSIM3V1lrdsw * Inv_L
				    + model->BSIM3V1wrdsw * Inv_W
				    + model->BSIM3V1prdsw * Inv_LW;
		  pParam->BSIM3V1prwg = model->BSIM3V1prwg
				    + model->BSIM3V1lprwg * Inv_L
				    + model->BSIM3V1wprwg * Inv_W
				    + model->BSIM3V1pprwg * Inv_LW;
		  pParam->BSIM3V1prwb = model->BSIM3V1prwb
				    + model->BSIM3V1lprwb * Inv_L
				    + model->BSIM3V1wprwb * Inv_W
				    + model->BSIM3V1pprwb * Inv_LW;
		  pParam->BSIM3V1prt = model->BSIM3V1prt
				    + model->BSIM3V1lprt * Inv_L
				    + model->BSIM3V1wprt * Inv_W
				    + model->BSIM3V1pprt * Inv_LW;
		  pParam->BSIM3V1eta0 = model->BSIM3V1eta0
				    + model->BSIM3V1leta0 * Inv_L
				    + model->BSIM3V1weta0 * Inv_W
				    + model->BSIM3V1peta0 * Inv_LW;
		  pParam->BSIM3V1etab = model->BSIM3V1etab
				    + model->BSIM3V1letab * Inv_L
				    + model->BSIM3V1wetab * Inv_W
				    + model->BSIM3V1petab * Inv_LW;
		  pParam->BSIM3V1pclm = model->BSIM3V1pclm
				    + model->BSIM3V1lpclm * Inv_L
				    + model->BSIM3V1wpclm * Inv_W
				    + model->BSIM3V1ppclm * Inv_LW;
		  pParam->BSIM3V1pdibl1 = model->BSIM3V1pdibl1
				      + model->BSIM3V1lpdibl1 * Inv_L
				      + model->BSIM3V1wpdibl1 * Inv_W
				      + model->BSIM3V1ppdibl1 * Inv_LW;
		  pParam->BSIM3V1pdibl2 = model->BSIM3V1pdibl2
				      + model->BSIM3V1lpdibl2 * Inv_L
				      + model->BSIM3V1wpdibl2 * Inv_W
				      + model->BSIM3V1ppdibl2 * Inv_LW;
		  pParam->BSIM3V1pdiblb = model->BSIM3V1pdiblb
				      + model->BSIM3V1lpdiblb * Inv_L
				      + model->BSIM3V1wpdiblb * Inv_W
				      + model->BSIM3V1ppdiblb * Inv_LW;
		  pParam->BSIM3V1pscbe1 = model->BSIM3V1pscbe1
				      + model->BSIM3V1lpscbe1 * Inv_L
				      + model->BSIM3V1wpscbe1 * Inv_W
				      + model->BSIM3V1ppscbe1 * Inv_LW;
		  pParam->BSIM3V1pscbe2 = model->BSIM3V1pscbe2
				      + model->BSIM3V1lpscbe2 * Inv_L
				      + model->BSIM3V1wpscbe2 * Inv_W
				      + model->BSIM3V1ppscbe2 * Inv_LW;
		  pParam->BSIM3V1pvag = model->BSIM3V1pvag
				    + model->BSIM3V1lpvag * Inv_L
				    + model->BSIM3V1wpvag * Inv_W
				    + model->BSIM3V1ppvag * Inv_LW;
		  pParam->BSIM3V1wr = model->BSIM3V1wr
				  + model->BSIM3V1lwr * Inv_L
				  + model->BSIM3V1wwr * Inv_W
				  + model->BSIM3V1pwr * Inv_LW;
		  pParam->BSIM3V1dwg = model->BSIM3V1dwg
				   + model->BSIM3V1ldwg * Inv_L
				   + model->BSIM3V1wdwg * Inv_W
				   + model->BSIM3V1pdwg * Inv_LW;
		  pParam->BSIM3V1dwb = model->BSIM3V1dwb
				   + model->BSIM3V1ldwb * Inv_L
				   + model->BSIM3V1wdwb * Inv_W
				   + model->BSIM3V1pdwb * Inv_LW;
		  pParam->BSIM3V1b0 = model->BSIM3V1b0
				  + model->BSIM3V1lb0 * Inv_L
				  + model->BSIM3V1wb0 * Inv_W
				  + model->BSIM3V1pb0 * Inv_LW;
		  pParam->BSIM3V1b1 = model->BSIM3V1b1
				  + model->BSIM3V1lb1 * Inv_L
				  + model->BSIM3V1wb1 * Inv_W
				  + model->BSIM3V1pb1 * Inv_LW;
		  pParam->BSIM3V1alpha0 = model->BSIM3V1alpha0
				      + model->BSIM3V1lalpha0 * Inv_L
				      + model->BSIM3V1walpha0 * Inv_W
				      + model->BSIM3V1palpha0 * Inv_LW;
		  pParam->BSIM3V1beta0 = model->BSIM3V1beta0
				     + model->BSIM3V1lbeta0 * Inv_L
				     + model->BSIM3V1wbeta0 * Inv_W
				     + model->BSIM3V1pbeta0 * Inv_LW;
		  /* CV model */
		  pParam->BSIM3V1elm = model->BSIM3V1elm
				  + model->BSIM3V1lelm * Inv_L
				  + model->BSIM3V1welm * Inv_W
				  + model->BSIM3V1pelm * Inv_LW;
		  pParam->BSIM3V1cgsl = model->BSIM3V1cgsl
				    + model->BSIM3V1lcgsl * Inv_L
				    + model->BSIM3V1wcgsl * Inv_W
				    + model->BSIM3V1pcgsl * Inv_LW;
		  pParam->BSIM3V1cgdl = model->BSIM3V1cgdl
				    + model->BSIM3V1lcgdl * Inv_L
				    + model->BSIM3V1wcgdl * Inv_W
				    + model->BSIM3V1pcgdl * Inv_LW;
		  pParam->BSIM3V1ckappa = model->BSIM3V1ckappa
				      + model->BSIM3V1lckappa * Inv_L
				      + model->BSIM3V1wckappa * Inv_W
				      + model->BSIM3V1pckappa * Inv_LW;
		  pParam->BSIM3V1cf = model->BSIM3V1cf
				  + model->BSIM3V1lcf * Inv_L
				  + model->BSIM3V1wcf * Inv_W
				  + model->BSIM3V1pcf * Inv_LW;
		  pParam->BSIM3V1clc = model->BSIM3V1clc
				   + model->BSIM3V1lclc * Inv_L
				   + model->BSIM3V1wclc * Inv_W
				   + model->BSIM3V1pclc * Inv_LW;
		  pParam->BSIM3V1cle = model->BSIM3V1cle
				   + model->BSIM3V1lcle * Inv_L
				   + model->BSIM3V1wcle * Inv_W
				   + model->BSIM3V1pcle * Inv_LW;
		  pParam->BSIM3V1vfbcv = model->BSIM3V1vfbcv
				  + model->BSIM3V1lvfbcv * Inv_L
				  + model->BSIM3V1wvfbcv * Inv_W
				  + model->BSIM3V1pvfbcv * Inv_LW;
                  pParam->BSIM3V1abulkCVfactor = 1.0 + pow((pParam->BSIM3V1clc
					     / pParam->BSIM3V1leff),
					     pParam->BSIM3V1cle);

	          T0 = (TRatio - 1.0);
	          pParam->BSIM3V1ua = pParam->BSIM3V1ua + pParam->BSIM3V1ua1 * T0;
	          pParam->BSIM3V1ub = pParam->BSIM3V1ub + pParam->BSIM3V1ub1 * T0;
	          pParam->BSIM3V1uc = pParam->BSIM3V1uc + pParam->BSIM3V1uc1 * T0;
                  if (pParam->BSIM3V1u0 > 1.0) 
                      pParam->BSIM3V1u0 = pParam->BSIM3V1u0 / 1.0e4;

                  pParam->BSIM3V1u0temp = pParam->BSIM3V1u0
				      * pow(TRatio, pParam->BSIM3V1ute); 
                  pParam->BSIM3V1vsattemp = pParam->BSIM3V1vsat - pParam->BSIM3V1at 
			                * T0;
	          pParam->BSIM3V1rds0 = (pParam->BSIM3V1rdsw + pParam->BSIM3V1prt * T0)
                                    / pow(pParam->BSIM3V1weff * 1E6, pParam->BSIM3V1wr);

		  if (BSIM3V1checkModel(model, here, ckt))
		  {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V1modName;
                      namarray[1] = here->BSIM3V1name;
                      (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM3V1V3 parameter checking for %s in model %s", namarray);
                      return(E_BADPARM);   
		  }

                  pParam->BSIM3V1cgdo = (model->BSIM3V1cgdo + pParam->BSIM3V1cf)
				    * pParam->BSIM3V1weffCV;
                  pParam->BSIM3V1cgso = (model->BSIM3V1cgso + pParam->BSIM3V1cf)
				    * pParam->BSIM3V1weffCV;
                  pParam->BSIM3V1cgbo = model->BSIM3V1cgbo * pParam->BSIM3V1leffCV;

                  if (!model->BSIM3V1npeakGiven && model->BSIM3V1gamma1Given)
                  {   T0 = pParam->BSIM3V1gamma1 * model->BSIM3V1cox;
                      pParam->BSIM3V1npeak = 3.021E22 * T0 * T0;
                  }

		  pParam->BSIM3V1phi = 2.0 * Vtm0 
			           * log(pParam->BSIM3V1npeak / ni);

	          pParam->BSIM3V1sqrtPhi = sqrt(pParam->BSIM3V1phi);
	          pParam->BSIM3V1phis3 = pParam->BSIM3V1sqrtPhi * pParam->BSIM3V1phi;

                  pParam->BSIM3V1Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM3V1npeak * 1.0e6))
                                     * pParam->BSIM3V1sqrtPhi; 
                  pParam->BSIM3V1sqrtXdep0 = sqrt(pParam->BSIM3V1Xdep0);
                  pParam->BSIM3V1litl = sqrt(3.0 * pParam->BSIM3V1xj
				    * model->BSIM3V1tox);
                  pParam->BSIM3V1vbi = Vtm0 * log(1.0e20
			           * pParam->BSIM3V1npeak / (ni * ni));
                  pParam->BSIM3V1cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM3V1npeak * 1.0e6 / 2.0
				     / pParam->BSIM3V1phi);
        
                  if (model->BSIM3V1k1Given || model->BSIM3V1k2Given)
	          {   if (!model->BSIM3V1k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3V1k1 = 0.53;
                      }
                      if (!model->BSIM3V1k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3V1k2 = -0.0186;
                      }
                      if (model->BSIM3V1nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V1xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V1vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V1vbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V1gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V1gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM3V1vbxGiven)
                          pParam->BSIM3V1vbx = pParam->BSIM3V1phi - 7.7348e-4 
                                           * pParam->BSIM3V1npeak
					   * pParam->BSIM3V1xt * pParam->BSIM3V1xt;
	              if (pParam->BSIM3V1vbx > 0.0)
		          pParam->BSIM3V1vbx = -pParam->BSIM3V1vbx;
	              if (pParam->BSIM3V1vbm > 0.0)
                          pParam->BSIM3V1vbm = -pParam->BSIM3V1vbm;
           
                      if (!model->BSIM3V1gamma1Given)
                          pParam->BSIM3V1gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM3V1npeak)
                                              / model->BSIM3V1cox;
                      if (!model->BSIM3V1gamma2Given)
                          pParam->BSIM3V1gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM3V1nsub)
                                              / model->BSIM3V1cox;

                      T0 = pParam->BSIM3V1gamma1 - pParam->BSIM3V1gamma2;
                      T1 = sqrt(pParam->BSIM3V1phi - pParam->BSIM3V1vbx)
			 - pParam->BSIM3V1sqrtPhi;
                      T2 = sqrt(pParam->BSIM3V1phi * (pParam->BSIM3V1phi
			 - pParam->BSIM3V1vbm)) - pParam->BSIM3V1phi;
                      pParam->BSIM3V1k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3V1vbm);
                      pParam->BSIM3V1k1 = pParam->BSIM3V1gamma2 - 2.0
				      * pParam->BSIM3V1k2 * sqrt(pParam->BSIM3V1phi
				      - pParam->BSIM3V1vbm);
                  }
 
		  if (pParam->BSIM3V1k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM3V1k1 / pParam->BSIM3V1k2;
                      pParam->BSIM3V1vbsc = 0.9 * (pParam->BSIM3V1phi - T0 * T0);
		      if (pParam->BSIM3V1vbsc > -3.0)
		          pParam->BSIM3V1vbsc = -3.0;
		      else if (pParam->BSIM3V1vbsc < -30.0)
		          pParam->BSIM3V1vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM3V1vbsc = -30.0;
		  }
		  if (pParam->BSIM3V1vbsc > pParam->BSIM3V1vbm)
		      pParam->BSIM3V1vbsc = pParam->BSIM3V1vbm;

	          if (model->BSIM3V1vth0Given)
		  {   pParam->BSIM3V1vfb = model->BSIM3V1type * pParam->BSIM3V1vth0 
                                       - pParam->BSIM3V1phi - pParam->BSIM3V1k1 
                                       * pParam->BSIM3V1sqrtPhi;
		  }
		  else
		  {   pParam->BSIM3V1vfb = -1.0;
		      pParam->BSIM3V1vth0 = model->BSIM3V1type * (pParam->BSIM3V1vfb
                                        + pParam->BSIM3V1phi + pParam->BSIM3V1k1 
                                        * pParam->BSIM3V1sqrtPhi);
		  }
                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3V1tox
		     * pParam->BSIM3V1Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3V1dsub * pParam->BSIM3V1leff / T1);
                  pParam->BSIM3V1theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3V1drout * pParam->BSIM3V1leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3V1thetaRout = pParam->BSIM3V1pdibl1 * T2
				         + pParam->BSIM3V1pdibl2;
	      }

              /* process source/drain series resistance */
              here->BSIM3V1drainConductance = model->BSIM3V1sheetResistance 
		                              * here->BSIM3V1drainSquares;
              if (here->BSIM3V1drainConductance > 0.0)
                  here->BSIM3V1drainConductance = 1.0
					      / here->BSIM3V1drainConductance;
	      else
                  here->BSIM3V1drainConductance = 0.0;
                  
              here->BSIM3V1sourceConductance = model->BSIM3V1sheetResistance 
		                           * here->BSIM3V1sourceSquares;
              if (here->BSIM3V1sourceConductance > 0.0) 
                  here->BSIM3V1sourceConductance = 1.0
					       / here->BSIM3V1sourceConductance;
	      else
                  here->BSIM3V1sourceConductance = 0.0;
	      here->BSIM3V1cgso = pParam->BSIM3V1cgso;
	      here->BSIM3V1cgdo = pParam->BSIM3V1cgdo;
         }
    }
    return(OK);
}
