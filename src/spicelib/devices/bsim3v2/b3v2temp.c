/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/***********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3v2temp.c
**********/
/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v2def.h"
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
BSIM3V2temp(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
register BSIM3V2model *model = (BSIM3V2model*) inModel;
register BSIM3V2instance *here;
struct BSIM3V2SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni, T0, T1, T2, T3, T4, T5, Ldrn, Wdrn;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Dw, Dl, Vtm0, Tnom;
double Nvtm, SourceSatCurrent, DrainSatCurrent;
int Size_Not_Found;

    /*  loop through all the BSIM3V2 device models */
    for (; model != NULL; model = model->BSIM3V2nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3V2bulkJctPotential < 0.1)  
	 {   model->BSIM3V2bulkJctPotential = 0.1;
	     fprintf(stderr, "Given pb is less than 0.1. Pb is set to 0.1.\n");
	 }
         if (model->BSIM3V2sidewallJctPotential < 0.1)
	 {   model->BSIM3V2sidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbsw is less than 0.1. Pbsw is set to 0.1.\n");
	 }
         if (model->BSIM3V2GatesidewallJctPotential < 0.1)
	 {   model->BSIM3V2GatesidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbswg is less than 0.1. Pbswg is set to 0.1.\n");
	 }
         model->pSizeDependParamKnot = NULL;
	 pLastKnot = NULL;

	 Tnom = model->BSIM3V2tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM3V2vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM3V2factor1 = sqrt(EPSSI / EPSOX * model->BSIM3V2tox);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM3V2vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM3V2vtm + model->BSIM3V2jctTempExponent
		* log(Temp / Tnom);
	     T1 = exp(T0 / model->BSIM3V2jctEmissionCoeff);
	     model->BSIM3V2jctTempSatCurDensity = model->BSIM3V2jctSatCurDensity
					      * T1;
	     model->BSIM3V2jctSidewallTempSatCurDensity
			 = model->BSIM3V2jctSidewallSatCurDensity * T1;
	 }
	 else
	 {   model->BSIM3V2jctTempSatCurDensity = model->BSIM3V2jctSatCurDensity;
	     model->BSIM3V2jctSidewallTempSatCurDensity
			= model->BSIM3V2jctSidewallSatCurDensity;
	 }

	 if (model->BSIM3V2jctTempSatCurDensity < 0.0)
	     model->BSIM3V2jctTempSatCurDensity = 0.0;
	 if (model->BSIM3V2jctSidewallTempSatCurDensity < 0.0)
	     model->BSIM3V2jctSidewallTempSatCurDensity = 0.0;

	 /* Temperature dependence of D/B and S/B diode capacitance begins */
	 delTemp = ckt->CKTtemp - model->BSIM3V2tnom;
	 T0 = model->BSIM3V2tcj * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM3V2unitAreaJctCap *= 1.0 + T0;
	 }
	 else if (model->BSIM3V2unitAreaJctCap > 0.0)
	 {   model->BSIM3V2unitAreaJctCap = 0.0;
	     fprintf(stderr, "Temperature effect has caused cj to be negative. Cj is clamped to zero.\n");
	 }
         T0 = model->BSIM3V2tcjsw * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM3V2unitLengthSidewallJctCap *= 1.0 + T0;
	 }
	 else if (model->BSIM3V2unitLengthSidewallJctCap > 0.0)
	 {   model->BSIM3V2unitLengthSidewallJctCap = 0.0;
	     fprintf(stderr, "Temperature effect has caused cjsw to be negative. Cjsw is clamped to zero.\n");
	 }
         T0 = model->BSIM3V2tcjswg * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM3V2unitLengthGateSidewallJctCap *= 1.0 + T0;
	 }
	 else if (model->BSIM3V2unitLengthGateSidewallJctCap > 0.0)
	 {   model->BSIM3V2unitLengthGateSidewallJctCap = 0.0;
	     fprintf(stderr, "Temperature effect has caused cjswg to be negative. Cjswg is clamped to zero.\n");
	 }

         model->BSIM3V2PhiB = model->BSIM3V2bulkJctPotential
			  - model->BSIM3V2tpb * delTemp;
         if (model->BSIM3V2PhiB < 0.01)
	 {   model->BSIM3V2PhiB = 0.01;
	     fprintf(stderr, "Temperature effect has caused pb to be less than 0.01. Pb is clamped to 0.01.\n");
	 }
         model->BSIM3V2PhiBSW = model->BSIM3V2sidewallJctPotential
                            - model->BSIM3V2tpbsw * delTemp;
         if (model->BSIM3V2PhiBSW <= 0.01)
	 {   model->BSIM3V2PhiBSW = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbsw to be less than 0.01. Pbsw is clamped to 0.01.\n");
	 }
	 model->BSIM3V2PhiBSWG = model->BSIM3V2GatesidewallJctPotential
                             - model->BSIM3V2tpbswg * delTemp;
         if (model->BSIM3V2PhiBSWG <= 0.01)
	 {   model->BSIM3V2PhiBSWG = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbswg to be less than 0.01. Pbswg is clamped to 0.01.\n");
	 }
         /* End of junction capacitance - Weidong & Min-Chie 5/1998 */

         /* loop through all the instances of the model */
	 /* MCJ: Length and Width not initialized */
         for (here = model->BSIM3V2instances; here != NULL;
              here = here->BSIM3V2nextInstance) 
	 {    
              if (here->BSIM3V2owner != ARCHme) continue;
              pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM3V2l == pSizeDependParamKnot->Length)
		      && (here->BSIM3V2w == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      if (Size_Not_Found)
	      {   pParam = (struct BSIM3V2SizeDependParam *)malloc(
	                    sizeof(struct BSIM3V2SizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

		  Ldrn = here->BSIM3V2l;
		  Wdrn = here->BSIM3V2w;
                  pParam->Length = Ldrn;
                  pParam->Width = Wdrn;
		  
                  T0 = pow(Ldrn, model->BSIM3V2Lln);
                  T1 = pow(Wdrn, model->BSIM3V2Lwn);
                  tmp1 = model->BSIM3V2Ll / T0 + model->BSIM3V2Lw / T1
                       + model->BSIM3V2Lwl / (T0 * T1);
                  pParam->BSIM3V2dl = model->BSIM3V2Lint + tmp1;
                  tmp2 = model->BSIM3V2Llc / T0 + model->BSIM3V2Lwc / T1
                       + model->BSIM3V2Lwlc / (T0 * T1);
                  pParam->BSIM3V2dlc = model->BSIM3V2dlc + tmp2;

                  T2 = pow(Ldrn, model->BSIM3V2Wln);
                  T3 = pow(Wdrn, model->BSIM3V2Wwn);
                  tmp1 = model->BSIM3V2Wl / T2 + model->BSIM3V2Ww / T3
                       + model->BSIM3V2Wwl / (T2 * T3);
                  pParam->BSIM3V2dw = model->BSIM3V2Wint + tmp1;
                  tmp2 = model->BSIM3V2Wlc / T2 + model->BSIM3V2Wwc / T3
                       + model->BSIM3V2Wwlc / (T2 * T3);
                  pParam->BSIM3V2dwc = model->BSIM3V2dwc + tmp2;

                  pParam->BSIM3V2leff = here->BSIM3V2l - 2.0 * pParam->BSIM3V2dl;
                  if (pParam->BSIM3V2leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V2modName;
                      namarray[1] = here->BSIM3V2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V2: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V2weff = here->BSIM3V2w - 2.0 * pParam->BSIM3V2dw;
                  if (pParam->BSIM3V2weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V2modName;
                      namarray[1] = here->BSIM3V2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V2: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V2leffCV = here->BSIM3V2l - 2.0 * pParam->BSIM3V2dlc;
                  if (pParam->BSIM3V2leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V2modName;
                      namarray[1] = here->BSIM3V2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V2: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3V2weffCV = here->BSIM3V2w - 2.0 * pParam->BSIM3V2dwc;
                  if (pParam->BSIM3V2weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V2modName;
                      namarray[1] = here->BSIM3V2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3V2: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM3V2binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM3V2leff;
		      Inv_W = 1.0e-6 / pParam->BSIM3V2weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM3V2leff
			     * pParam->BSIM3V2weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM3V2leff;
		      Inv_W = 1.0 / pParam->BSIM3V2weff;
		      Inv_LW = 1.0 / (pParam->BSIM3V2leff
			     * pParam->BSIM3V2weff);
		  }
		  pParam->BSIM3V2cdsc = model->BSIM3V2cdsc
				    + model->BSIM3V2lcdsc * Inv_L
				    + model->BSIM3V2wcdsc * Inv_W
				    + model->BSIM3V2pcdsc * Inv_LW;
		  pParam->BSIM3V2cdscb = model->BSIM3V2cdscb
				     + model->BSIM3V2lcdscb * Inv_L
				     + model->BSIM3V2wcdscb * Inv_W
				     + model->BSIM3V2pcdscb * Inv_LW; 
				     
    		  pParam->BSIM3V2cdscd = model->BSIM3V2cdscd
				     + model->BSIM3V2lcdscd * Inv_L
				     + model->BSIM3V2wcdscd * Inv_W
				     + model->BSIM3V2pcdscd * Inv_LW; 
				     
		  pParam->BSIM3V2cit = model->BSIM3V2cit
				   + model->BSIM3V2lcit * Inv_L
				   + model->BSIM3V2wcit * Inv_W
				   + model->BSIM3V2pcit * Inv_LW;
		  pParam->BSIM3V2nfactor = model->BSIM3V2nfactor
				       + model->BSIM3V2lnfactor * Inv_L
				       + model->BSIM3V2wnfactor * Inv_W
				       + model->BSIM3V2pnfactor * Inv_LW;
		  pParam->BSIM3V2xj = model->BSIM3V2xj
				  + model->BSIM3V2lxj * Inv_L
				  + model->BSIM3V2wxj * Inv_W
				  + model->BSIM3V2pxj * Inv_LW;
		  pParam->BSIM3V2vsat = model->BSIM3V2vsat
				    + model->BSIM3V2lvsat * Inv_L
				    + model->BSIM3V2wvsat * Inv_W
				    + model->BSIM3V2pvsat * Inv_LW;
		  pParam->BSIM3V2at = model->BSIM3V2at
				  + model->BSIM3V2lat * Inv_L
				  + model->BSIM3V2wat * Inv_W
				  + model->BSIM3V2pat * Inv_LW;
		  pParam->BSIM3V2a0 = model->BSIM3V2a0
				  + model->BSIM3V2la0 * Inv_L
				  + model->BSIM3V2wa0 * Inv_W
				  + model->BSIM3V2pa0 * Inv_LW; 
				  
		  pParam->BSIM3V2ags = model->BSIM3V2ags
				  + model->BSIM3V2lags * Inv_L
				  + model->BSIM3V2wags * Inv_W
				  + model->BSIM3V2pags * Inv_LW;
				  
		  pParam->BSIM3V2a1 = model->BSIM3V2a1
				  + model->BSIM3V2la1 * Inv_L
				  + model->BSIM3V2wa1 * Inv_W
				  + model->BSIM3V2pa1 * Inv_LW;
		  pParam->BSIM3V2a2 = model->BSIM3V2a2
				  + model->BSIM3V2la2 * Inv_L
				  + model->BSIM3V2wa2 * Inv_W
				  + model->BSIM3V2pa2 * Inv_LW;
		  pParam->BSIM3V2keta = model->BSIM3V2keta
				    + model->BSIM3V2lketa * Inv_L
				    + model->BSIM3V2wketa * Inv_W
				    + model->BSIM3V2pketa * Inv_LW;
		  pParam->BSIM3V2nsub = model->BSIM3V2nsub
				    + model->BSIM3V2lnsub * Inv_L
				    + model->BSIM3V2wnsub * Inv_W
				    + model->BSIM3V2pnsub * Inv_LW;
		  pParam->BSIM3V2npeak = model->BSIM3V2npeak
				     + model->BSIM3V2lnpeak * Inv_L
				     + model->BSIM3V2wnpeak * Inv_W
				     + model->BSIM3V2pnpeak * Inv_LW;
		  pParam->BSIM3V2ngate = model->BSIM3V2ngate
				     + model->BSIM3V2lngate * Inv_L
				     + model->BSIM3V2wngate * Inv_W
				     + model->BSIM3V2pngate * Inv_LW;
		  pParam->BSIM3V2gamma1 = model->BSIM3V2gamma1
				      + model->BSIM3V2lgamma1 * Inv_L
				      + model->BSIM3V2wgamma1 * Inv_W
				      + model->BSIM3V2pgamma1 * Inv_LW;
		  pParam->BSIM3V2gamma2 = model->BSIM3V2gamma2
				      + model->BSIM3V2lgamma2 * Inv_L
				      + model->BSIM3V2wgamma2 * Inv_W
				      + model->BSIM3V2pgamma2 * Inv_LW;
		  pParam->BSIM3V2vbx = model->BSIM3V2vbx
				   + model->BSIM3V2lvbx * Inv_L
				   + model->BSIM3V2wvbx * Inv_W
				   + model->BSIM3V2pvbx * Inv_LW;
		  pParam->BSIM3V2vbm = model->BSIM3V2vbm
				   + model->BSIM3V2lvbm * Inv_L
				   + model->BSIM3V2wvbm * Inv_W
				   + model->BSIM3V2pvbm * Inv_LW;
		  pParam->BSIM3V2xt = model->BSIM3V2xt
				   + model->BSIM3V2lxt * Inv_L
				   + model->BSIM3V2wxt * Inv_W
				   + model->BSIM3V2pxt * Inv_LW;
                  pParam->BSIM3V2vfb = model->BSIM3V2vfb
                                   + model->BSIM3V2lvfb * Inv_L
                                   + model->BSIM3V2wvfb * Inv_W
                                   + model->BSIM3V2pvfb * Inv_LW;
		  pParam->BSIM3V2k1 = model->BSIM3V2k1
				  + model->BSIM3V2lk1 * Inv_L
				  + model->BSIM3V2wk1 * Inv_W
				  + model->BSIM3V2pk1 * Inv_LW;
		  pParam->BSIM3V2kt1 = model->BSIM3V2kt1
				   + model->BSIM3V2lkt1 * Inv_L
				   + model->BSIM3V2wkt1 * Inv_W
				   + model->BSIM3V2pkt1 * Inv_LW;
		  pParam->BSIM3V2kt1l = model->BSIM3V2kt1l
				    + model->BSIM3V2lkt1l * Inv_L
				    + model->BSIM3V2wkt1l * Inv_W
				    + model->BSIM3V2pkt1l * Inv_LW;
		  pParam->BSIM3V2k2 = model->BSIM3V2k2
				  + model->BSIM3V2lk2 * Inv_L
				  + model->BSIM3V2wk2 * Inv_W
				  + model->BSIM3V2pk2 * Inv_LW;
		  pParam->BSIM3V2kt2 = model->BSIM3V2kt2
				   + model->BSIM3V2lkt2 * Inv_L
				   + model->BSIM3V2wkt2 * Inv_W
				   + model->BSIM3V2pkt2 * Inv_LW;
		  pParam->BSIM3V2k3 = model->BSIM3V2k3
				  + model->BSIM3V2lk3 * Inv_L
				  + model->BSIM3V2wk3 * Inv_W
				  + model->BSIM3V2pk3 * Inv_LW;
		  pParam->BSIM3V2k3b = model->BSIM3V2k3b
				   + model->BSIM3V2lk3b * Inv_L
				   + model->BSIM3V2wk3b * Inv_W
				   + model->BSIM3V2pk3b * Inv_LW;
		  pParam->BSIM3V2w0 = model->BSIM3V2w0
				  + model->BSIM3V2lw0 * Inv_L
				  + model->BSIM3V2ww0 * Inv_W
				  + model->BSIM3V2pw0 * Inv_LW;
		  pParam->BSIM3V2nlx = model->BSIM3V2nlx
				   + model->BSIM3V2lnlx * Inv_L
				   + model->BSIM3V2wnlx * Inv_W
				   + model->BSIM3V2pnlx * Inv_LW;
		  pParam->BSIM3V2dvt0 = model->BSIM3V2dvt0
				    + model->BSIM3V2ldvt0 * Inv_L
				    + model->BSIM3V2wdvt0 * Inv_W
				    + model->BSIM3V2pdvt0 * Inv_LW;
		  pParam->BSIM3V2dvt1 = model->BSIM3V2dvt1
				    + model->BSIM3V2ldvt1 * Inv_L
				    + model->BSIM3V2wdvt1 * Inv_W
				    + model->BSIM3V2pdvt1 * Inv_LW;
		  pParam->BSIM3V2dvt2 = model->BSIM3V2dvt2
				    + model->BSIM3V2ldvt2 * Inv_L
				    + model->BSIM3V2wdvt2 * Inv_W
				    + model->BSIM3V2pdvt2 * Inv_LW;
		  pParam->BSIM3V2dvt0w = model->BSIM3V2dvt0w
				    + model->BSIM3V2ldvt0w * Inv_L
				    + model->BSIM3V2wdvt0w * Inv_W
				    + model->BSIM3V2pdvt0w * Inv_LW;
		  pParam->BSIM3V2dvt1w = model->BSIM3V2dvt1w
				    + model->BSIM3V2ldvt1w * Inv_L
				    + model->BSIM3V2wdvt1w * Inv_W
				    + model->BSIM3V2pdvt1w * Inv_LW;
		  pParam->BSIM3V2dvt2w = model->BSIM3V2dvt2w
				    + model->BSIM3V2ldvt2w * Inv_L
				    + model->BSIM3V2wdvt2w * Inv_W
				    + model->BSIM3V2pdvt2w * Inv_LW;
		  pParam->BSIM3V2drout = model->BSIM3V2drout
				     + model->BSIM3V2ldrout * Inv_L
				     + model->BSIM3V2wdrout * Inv_W
				     + model->BSIM3V2pdrout * Inv_LW;
		  pParam->BSIM3V2dsub = model->BSIM3V2dsub
				    + model->BSIM3V2ldsub * Inv_L
				    + model->BSIM3V2wdsub * Inv_W
				    + model->BSIM3V2pdsub * Inv_LW;
		  pParam->BSIM3V2vth0 = model->BSIM3V2vth0
				    + model->BSIM3V2lvth0 * Inv_L
				    + model->BSIM3V2wvth0 * Inv_W
				    + model->BSIM3V2pvth0 * Inv_LW;
		  pParam->BSIM3V2ua = model->BSIM3V2ua
				  + model->BSIM3V2lua * Inv_L
				  + model->BSIM3V2wua * Inv_W
				  + model->BSIM3V2pua * Inv_LW;
		  pParam->BSIM3V2ua1 = model->BSIM3V2ua1
				   + model->BSIM3V2lua1 * Inv_L
				   + model->BSIM3V2wua1 * Inv_W
				   + model->BSIM3V2pua1 * Inv_LW;
		  pParam->BSIM3V2ub = model->BSIM3V2ub
				  + model->BSIM3V2lub * Inv_L
				  + model->BSIM3V2wub * Inv_W
				  + model->BSIM3V2pub * Inv_LW;
		  pParam->BSIM3V2ub1 = model->BSIM3V2ub1
				   + model->BSIM3V2lub1 * Inv_L
				   + model->BSIM3V2wub1 * Inv_W
				   + model->BSIM3V2pub1 * Inv_LW;
		  pParam->BSIM3V2uc = model->BSIM3V2uc
				  + model->BSIM3V2luc * Inv_L
				  + model->BSIM3V2wuc * Inv_W
				  + model->BSIM3V2puc * Inv_LW;
		  pParam->BSIM3V2uc1 = model->BSIM3V2uc1
				   + model->BSIM3V2luc1 * Inv_L
				   + model->BSIM3V2wuc1 * Inv_W
				   + model->BSIM3V2puc1 * Inv_LW;
		  pParam->BSIM3V2u0 = model->BSIM3V2u0
				  + model->BSIM3V2lu0 * Inv_L
				  + model->BSIM3V2wu0 * Inv_W
				  + model->BSIM3V2pu0 * Inv_LW;
		  pParam->BSIM3V2ute = model->BSIM3V2ute
				   + model->BSIM3V2lute * Inv_L
				   + model->BSIM3V2wute * Inv_W
				   + model->BSIM3V2pute * Inv_LW;
		  pParam->BSIM3V2voff = model->BSIM3V2voff
				    + model->BSIM3V2lvoff * Inv_L
				    + model->BSIM3V2wvoff * Inv_W
				    + model->BSIM3V2pvoff * Inv_LW;
		  pParam->BSIM3V2delta = model->BSIM3V2delta
				     + model->BSIM3V2ldelta * Inv_L
				     + model->BSIM3V2wdelta * Inv_W
				     + model->BSIM3V2pdelta * Inv_LW;
		  pParam->BSIM3V2rdsw = model->BSIM3V2rdsw
				    + model->BSIM3V2lrdsw * Inv_L
				    + model->BSIM3V2wrdsw * Inv_W
				    + model->BSIM3V2prdsw * Inv_LW;
		  pParam->BSIM3V2prwg = model->BSIM3V2prwg
				    + model->BSIM3V2lprwg * Inv_L
				    + model->BSIM3V2wprwg * Inv_W
				    + model->BSIM3V2pprwg * Inv_LW;
		  pParam->BSIM3V2prwb = model->BSIM3V2prwb
				    + model->BSIM3V2lprwb * Inv_L
				    + model->BSIM3V2wprwb * Inv_W
				    + model->BSIM3V2pprwb * Inv_LW;
		  pParam->BSIM3V2prt = model->BSIM3V2prt
				    + model->BSIM3V2lprt * Inv_L
				    + model->BSIM3V2wprt * Inv_W
				    + model->BSIM3V2pprt * Inv_LW;
		  pParam->BSIM3V2eta0 = model->BSIM3V2eta0
				    + model->BSIM3V2leta0 * Inv_L
				    + model->BSIM3V2weta0 * Inv_W
				    + model->BSIM3V2peta0 * Inv_LW;
		  pParam->BSIM3V2etab = model->BSIM3V2etab
				    + model->BSIM3V2letab * Inv_L
				    + model->BSIM3V2wetab * Inv_W
				    + model->BSIM3V2petab * Inv_LW;
		  pParam->BSIM3V2pclm = model->BSIM3V2pclm
				    + model->BSIM3V2lpclm * Inv_L
				    + model->BSIM3V2wpclm * Inv_W
				    + model->BSIM3V2ppclm * Inv_LW;
		  pParam->BSIM3V2pdibl1 = model->BSIM3V2pdibl1
				      + model->BSIM3V2lpdibl1 * Inv_L
				      + model->BSIM3V2wpdibl1 * Inv_W
				      + model->BSIM3V2ppdibl1 * Inv_LW;
		  pParam->BSIM3V2pdibl2 = model->BSIM3V2pdibl2
				      + model->BSIM3V2lpdibl2 * Inv_L
				      + model->BSIM3V2wpdibl2 * Inv_W
				      + model->BSIM3V2ppdibl2 * Inv_LW;
		  pParam->BSIM3V2pdiblb = model->BSIM3V2pdiblb
				      + model->BSIM3V2lpdiblb * Inv_L
				      + model->BSIM3V2wpdiblb * Inv_W
				      + model->BSIM3V2ppdiblb * Inv_LW;
		  pParam->BSIM3V2pscbe1 = model->BSIM3V2pscbe1
				      + model->BSIM3V2lpscbe1 * Inv_L
				      + model->BSIM3V2wpscbe1 * Inv_W
				      + model->BSIM3V2ppscbe1 * Inv_LW;
		  pParam->BSIM3V2pscbe2 = model->BSIM3V2pscbe2
				      + model->BSIM3V2lpscbe2 * Inv_L
				      + model->BSIM3V2wpscbe2 * Inv_W
				      + model->BSIM3V2ppscbe2 * Inv_LW;
		  pParam->BSIM3V2pvag = model->BSIM3V2pvag
				    + model->BSIM3V2lpvag * Inv_L
				    + model->BSIM3V2wpvag * Inv_W
				    + model->BSIM3V2ppvag * Inv_LW;
		  pParam->BSIM3V2wr = model->BSIM3V2wr
				  + model->BSIM3V2lwr * Inv_L
				  + model->BSIM3V2wwr * Inv_W
				  + model->BSIM3V2pwr * Inv_LW;
		  pParam->BSIM3V2dwg = model->BSIM3V2dwg
				   + model->BSIM3V2ldwg * Inv_L
				   + model->BSIM3V2wdwg * Inv_W
				   + model->BSIM3V2pdwg * Inv_LW;
		  pParam->BSIM3V2dwb = model->BSIM3V2dwb
				   + model->BSIM3V2ldwb * Inv_L
				   + model->BSIM3V2wdwb * Inv_W
				   + model->BSIM3V2pdwb * Inv_LW;
		  pParam->BSIM3V2b0 = model->BSIM3V2b0
				  + model->BSIM3V2lb0 * Inv_L
				  + model->BSIM3V2wb0 * Inv_W
				  + model->BSIM3V2pb0 * Inv_LW;
		  pParam->BSIM3V2b1 = model->BSIM3V2b1
				  + model->BSIM3V2lb1 * Inv_L
				  + model->BSIM3V2wb1 * Inv_W
				  + model->BSIM3V2pb1 * Inv_LW;
		  pParam->BSIM3V2alpha0 = model->BSIM3V2alpha0
				      + model->BSIM3V2lalpha0 * Inv_L
				      + model->BSIM3V2walpha0 * Inv_W
				      + model->BSIM3V2palpha0 * Inv_LW;
                  pParam->BSIM3V2alpha1 = model->BSIM3V2alpha1
                                      + model->BSIM3V2lalpha1 * Inv_L
                                      + model->BSIM3V2walpha1 * Inv_W
                                      + model->BSIM3V2palpha1 * Inv_LW;
		  pParam->BSIM3V2beta0 = model->BSIM3V2beta0
				     + model->BSIM3V2lbeta0 * Inv_L
				     + model->BSIM3V2wbeta0 * Inv_W
				     + model->BSIM3V2pbeta0 * Inv_LW;
		  /* CV model */
		  pParam->BSIM3V2elm = model->BSIM3V2elm
				  + model->BSIM3V2lelm * Inv_L
				  + model->BSIM3V2welm * Inv_W
				  + model->BSIM3V2pelm * Inv_LW;
		  pParam->BSIM3V2cgsl = model->BSIM3V2cgsl
				    + model->BSIM3V2lcgsl * Inv_L
				    + model->BSIM3V2wcgsl * Inv_W
				    + model->BSIM3V2pcgsl * Inv_LW;
		  pParam->BSIM3V2cgdl = model->BSIM3V2cgdl
				    + model->BSIM3V2lcgdl * Inv_L
				    + model->BSIM3V2wcgdl * Inv_W
				    + model->BSIM3V2pcgdl * Inv_LW;
		  pParam->BSIM3V2ckappa = model->BSIM3V2ckappa
				      + model->BSIM3V2lckappa * Inv_L
				      + model->BSIM3V2wckappa * Inv_W
				      + model->BSIM3V2pckappa * Inv_LW;
		  pParam->BSIM3V2cf = model->BSIM3V2cf
				  + model->BSIM3V2lcf * Inv_L
				  + model->BSIM3V2wcf * Inv_W
				  + model->BSIM3V2pcf * Inv_LW;
		  pParam->BSIM3V2clc = model->BSIM3V2clc
				   + model->BSIM3V2lclc * Inv_L
				   + model->BSIM3V2wclc * Inv_W
				   + model->BSIM3V2pclc * Inv_LW;
		  pParam->BSIM3V2cle = model->BSIM3V2cle
				   + model->BSIM3V2lcle * Inv_L
				   + model->BSIM3V2wcle * Inv_W
				   + model->BSIM3V2pcle * Inv_LW;
		  pParam->BSIM3V2vfbcv = model->BSIM3V2vfbcv
				     + model->BSIM3V2lvfbcv * Inv_L
				     + model->BSIM3V2wvfbcv * Inv_W
				     + model->BSIM3V2pvfbcv * Inv_LW;
                  pParam->BSIM3V2acde = model->BSIM3V2acde
                                    + model->BSIM3V2lacde * Inv_L
                                    + model->BSIM3V2wacde * Inv_W
                                    + model->BSIM3V2pacde * Inv_LW;
                  pParam->BSIM3V2moin = model->BSIM3V2moin
                                    + model->BSIM3V2lmoin * Inv_L
                                    + model->BSIM3V2wmoin * Inv_W
                                    + model->BSIM3V2pmoin * Inv_LW;
                  pParam->BSIM3V2noff = model->BSIM3V2noff
                                    + model->BSIM3V2lnoff * Inv_L
                                    + model->BSIM3V2wnoff * Inv_W
                                    + model->BSIM3V2pnoff * Inv_LW;
                  pParam->BSIM3V2voffcv = model->BSIM3V2voffcv
                                      + model->BSIM3V2lvoffcv * Inv_L
                                      + model->BSIM3V2wvoffcv * Inv_W
                                      + model->BSIM3V2pvoffcv * Inv_LW;

                  pParam->BSIM3V2abulkCVfactor = 1.0 + pow((pParam->BSIM3V2clc
					     / pParam->BSIM3V2leffCV),
					     pParam->BSIM3V2cle);

	          T0 = (TRatio - 1.0);
	          pParam->BSIM3V2ua = pParam->BSIM3V2ua + pParam->BSIM3V2ua1 * T0;
	          pParam->BSIM3V2ub = pParam->BSIM3V2ub + pParam->BSIM3V2ub1 * T0;
	          pParam->BSIM3V2uc = pParam->BSIM3V2uc + pParam->BSIM3V2uc1 * T0;
                  if (pParam->BSIM3V2u0 > 1.0) 
                      pParam->BSIM3V2u0 = pParam->BSIM3V2u0 / 1.0e4;

                  pParam->BSIM3V2u0temp = pParam->BSIM3V2u0
				      * pow(TRatio, pParam->BSIM3V2ute); 
                  pParam->BSIM3V2vsattemp = pParam->BSIM3V2vsat - pParam->BSIM3V2at 
			                * T0;
	          pParam->BSIM3V2rds0 = (pParam->BSIM3V2rdsw + pParam->BSIM3V2prt * T0)
                                    / pow(pParam->BSIM3V2weff * 1E6, pParam->BSIM3V2wr);

		  if (BSIM3V2checkModel(model, here, ckt))
		  {   IFuid namarray[2];
                      namarray[0] = model->BSIM3V2modName;
                      namarray[1] = here->BSIM3V2name;
                      (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM3V2V3.2 parameter checking for %s in model %s", namarray);
                      return(E_BADPARM);   
		  }

                  pParam->BSIM3V2cgdo = (model->BSIM3V2cgdo + pParam->BSIM3V2cf)
				    * pParam->BSIM3V2weffCV;
                  pParam->BSIM3V2cgso = (model->BSIM3V2cgso + pParam->BSIM3V2cf)
				    * pParam->BSIM3V2weffCV;
                  pParam->BSIM3V2cgbo = model->BSIM3V2cgbo * pParam->BSIM3V2leffCV;

                  T0 = pParam->BSIM3V2leffCV * pParam->BSIM3V2leffCV;
                  pParam->BSIM3V2tconst = pParam->BSIM3V2u0temp * pParam->BSIM3V2elm / (model->BSIM3V2cox
                                      * pParam->BSIM3V2weffCV * pParam->BSIM3V2leffCV * T0);

                  if (!model->BSIM3V2npeakGiven && model->BSIM3V2gamma1Given)
                  {   T0 = pParam->BSIM3V2gamma1 * model->BSIM3V2cox;
                      pParam->BSIM3V2npeak = 3.021E22 * T0 * T0;
                  }

		  pParam->BSIM3V2phi = 2.0 * Vtm0 
			           * log(pParam->BSIM3V2npeak / ni);

	          pParam->BSIM3V2sqrtPhi = sqrt(pParam->BSIM3V2phi);
	          pParam->BSIM3V2phis3 = pParam->BSIM3V2sqrtPhi * pParam->BSIM3V2phi;

                  pParam->BSIM3V2Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM3V2npeak * 1.0e6))
                                     * pParam->BSIM3V2sqrtPhi; 
                  pParam->BSIM3V2sqrtXdep0 = sqrt(pParam->BSIM3V2Xdep0);
                  pParam->BSIM3V2litl = sqrt(3.0 * pParam->BSIM3V2xj
				    * model->BSIM3V2tox);
                  pParam->BSIM3V2vbi = Vtm0 * log(1.0e20
			           * pParam->BSIM3V2npeak / (ni * ni));
                  pParam->BSIM3V2cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM3V2npeak * 1.0e6 / 2.0
				     / pParam->BSIM3V2phi);

                  pParam->BSIM3V2ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM3V2npeak * 1.0e6)) / 3.0;
                  pParam->BSIM3V2acde *= pow((pParam->BSIM3V2npeak / 2.0e16), -0.25);


                  if (model->BSIM3V2k1Given || model->BSIM3V2k2Given)
	          {   if (!model->BSIM3V2k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3V2k1 = 0.53;
                      }
                      if (!model->BSIM3V2k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3V2k2 = -0.0186;
                      }
                      if (model->BSIM3V2nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V2xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V2vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V2gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3V2gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM3V2vbxGiven)
                          pParam->BSIM3V2vbx = pParam->BSIM3V2phi - 7.7348e-4 
                                           * pParam->BSIM3V2npeak
					   * pParam->BSIM3V2xt * pParam->BSIM3V2xt;
	              if (pParam->BSIM3V2vbx > 0.0)
		          pParam->BSIM3V2vbx = -pParam->BSIM3V2vbx;
	              if (pParam->BSIM3V2vbm > 0.0)
                          pParam->BSIM3V2vbm = -pParam->BSIM3V2vbm;
           
                      if (!model->BSIM3V2gamma1Given)
                          pParam->BSIM3V2gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM3V2npeak)
                                              / model->BSIM3V2cox;
                      if (!model->BSIM3V2gamma2Given)
                          pParam->BSIM3V2gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM3V2nsub)
                                              / model->BSIM3V2cox;

                      T0 = pParam->BSIM3V2gamma1 - pParam->BSIM3V2gamma2;
                      T1 = sqrt(pParam->BSIM3V2phi - pParam->BSIM3V2vbx)
			 - pParam->BSIM3V2sqrtPhi;
                      T2 = sqrt(pParam->BSIM3V2phi * (pParam->BSIM3V2phi
			 - pParam->BSIM3V2vbm)) - pParam->BSIM3V2phi;
                      pParam->BSIM3V2k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3V2vbm);
                      pParam->BSIM3V2k1 = pParam->BSIM3V2gamma2 - 2.0
				      * pParam->BSIM3V2k2 * sqrt(pParam->BSIM3V2phi
				      - pParam->BSIM3V2vbm);
                  }
 
		  if (pParam->BSIM3V2k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM3V2k1 / pParam->BSIM3V2k2;
                      pParam->BSIM3V2vbsc = 0.9 * (pParam->BSIM3V2phi - T0 * T0);
		      if (pParam->BSIM3V2vbsc > -3.0)
		          pParam->BSIM3V2vbsc = -3.0;
		      else if (pParam->BSIM3V2vbsc < -30.0)
		          pParam->BSIM3V2vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM3V2vbsc = -30.0;
		  }
		  if (pParam->BSIM3V2vbsc > pParam->BSIM3V2vbm)
		      pParam->BSIM3V2vbsc = pParam->BSIM3V2vbm;

                  if (!model->BSIM3V2vfbGiven)
                  {   if (model->BSIM3V2vth0Given)
                      {   pParam->BSIM3V2vfb = model->BSIM3V2type * pParam->BSIM3V2vth0
                                           - pParam->BSIM3V2phi - pParam->BSIM3V2k1
                                           * pParam->BSIM3V2sqrtPhi;
                      }
                      else
                      {   pParam->BSIM3V2vfb = -1.0;
                      }
                  }
                  if (!model->BSIM3V2vth0Given)
                  {   pParam->BSIM3V2vth0 = model->BSIM3V2type * (pParam->BSIM3V2vfb
                                        + pParam->BSIM3V2phi + pParam->BSIM3V2k1
                                        * pParam->BSIM3V2sqrtPhi);
                  }

                  pParam->BSIM3V2k1ox = pParam->BSIM3V2k1 * model->BSIM3V2tox
                                    / model->BSIM3V2toxm;
                  pParam->BSIM3V2k2ox = pParam->BSIM3V2k2 * model->BSIM3V2tox
                                    / model->BSIM3V2toxm;

                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3V2tox
		     * pParam->BSIM3V2Xdep0);
                  T0 = exp(-0.5 * pParam->BSIM3V2dsub * pParam->BSIM3V2leff / T1);
                  pParam->BSIM3V2theta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3V2drout * pParam->BSIM3V2leff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3V2thetaRout = pParam->BSIM3V2pdibl1 * T2
				         + pParam->BSIM3V2pdibl2;

                  /* vfbzb for capMod 1, 2 & 3 - Weidong 4/1997 */
                  tmp = sqrt(pParam->BSIM3V2Xdep0);
                  tmp1 = pParam->BSIM3V2vbi - pParam->BSIM3V2phi;
                  tmp2 = model->BSIM3V2factor1 * tmp;

                  T0 = -0.5 * pParam->BSIM3V2dvt1w * pParam->BSIM3V2weff
                     * pParam->BSIM3V2leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T2 = T1 * (1.0 + 2.0 * T1);
                  }
                  T0 = pParam->BSIM3V2dvt0w * T2;
                  T2 = T0 * tmp1;

                  T0 = -0.5 * pParam->BSIM3V2dvt1 * pParam->BSIM3V2leff / tmp2;
                  if (T0 > -EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  else
                  {   T1 = MIN_EXP;
                      T3 = T1 * (1.0 + 2.0 * T1);
                  }
                  T3 = pParam->BSIM3V2dvt0 * T3 * tmp1;

                  T4 = model->BSIM3V2tox * pParam->BSIM3V2phi
                     / (pParam->BSIM3V2weff + pParam->BSIM3V2w0);

                  T0 = sqrt(1.0 + pParam->BSIM3V2nlx / pParam->BSIM3V2leff);
                  T5 = pParam->BSIM3V2k1ox * (T0 - 1.0) * pParam->BSIM3V2sqrtPhi
                     + (pParam->BSIM3V2kt1 + pParam->BSIM3V2kt1l / pParam->BSIM3V2leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM3V2type * pParam->BSIM3V2vth0
                       - T2 - T3 + pParam->BSIM3V2k3 * T4 + T5;
                  pParam->BSIM3V2vfbzb = tmp3 - pParam->BSIM3V2phi - pParam->BSIM3V2k1
                                     * pParam->BSIM3V2sqrtPhi;
                  /* End of vfbzb */
              }

              /* process source/drain series resistance */
              here->BSIM3V2drainConductance = model->BSIM3V2sheetResistance 
		                              * here->BSIM3V2drainSquares;
              if (here->BSIM3V2drainConductance > 0.0)
                  here->BSIM3V2drainConductance = 1.0
					      / here->BSIM3V2drainConductance;
	      else
                  here->BSIM3V2drainConductance = 0.0;
                  
              here->BSIM3V2sourceConductance = model->BSIM3V2sheetResistance 
		                           * here->BSIM3V2sourceSquares;
              if (here->BSIM3V2sourceConductance > 0.0) 
                  here->BSIM3V2sourceConductance = 1.0
					       / here->BSIM3V2sourceConductance;
	      else
                  here->BSIM3V2sourceConductance = 0.0;
	      here->BSIM3V2cgso = pParam->BSIM3V2cgso;
	      here->BSIM3V2cgdo = pParam->BSIM3V2cgdo;

              Nvtm = model->BSIM3V2vtm * model->BSIM3V2jctEmissionCoeff;
              if ((here->BSIM3V2sourceArea <= 0.0) &&
                  (here->BSIM3V2sourcePerimeter <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM3V2sourceArea
                                   * model->BSIM3V2jctTempSatCurDensity
                                   + here->BSIM3V2sourcePerimeter
                                   * model->BSIM3V2jctSidewallTempSatCurDensity;
              }
              if ((SourceSatCurrent > 0.0) && (model->BSIM3V2ijth > 0.0))
              {   here->BSIM3V2vjsm = Nvtm * log(model->BSIM3V2ijth
                                  / SourceSatCurrent + 1.0);
              }

              if ((here->BSIM3V2drainArea <= 0.0) &&
                  (here->BSIM3V2drainPerimeter <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM3V2drainArea
                                  * model->BSIM3V2jctTempSatCurDensity
                                  + here->BSIM3V2drainPerimeter
                                  * model->BSIM3V2jctSidewallTempSatCurDensity;
              }
              if ((DrainSatCurrent > 0.0) && (model->BSIM3V2ijth > 0.0))
              {   here->BSIM3V2vjdm = Nvtm * log(model->BSIM3V2ijth
                                  / DrainSatCurrent + 1.0);
              }
         }
    }
    return(OK);
}

