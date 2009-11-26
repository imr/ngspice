/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4temp.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "const.h"
#include "sperror.h"

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5 
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define Charge_q 1.60219e-19

int
BSIM4v2RdseffGeo(double nf, int geo, int rgeo, int minSD, double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG, int Type,double *Rtot);

int
BSIM4v2PAeffGeo(double nf, int geo, int minSD, double Weffcj, double DMCG, double DMCI, double DMDG, double *Ps, double *Pd, double *As, double *Ad);

int
BSIM4v2DioIjthVjmEval(
double Nvtm, double Ijth, double Isb, double XExpBV,
double *Vjm)
{
double Tb, Tc, EVjmovNv;

       Tc = XExpBV;
       Tb = 1.0 + Ijth / Isb - Tc;
       EVjmovNv = 0.5 * (Tb + sqrt(Tb * Tb + 4.0 * Tc));
       *Vjm = Nvtm * log(EVjmovNv);

return 0;
}


int
BSIM4v2temp(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v2model *model = (BSIM4v2model*) inModel;
BSIM4v2instance *here;
struct bsim4SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T8, T9, Lnew=0.0, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
int Size_Not_Found;

    /*  loop through all the BSIM4v2 device models */
    for (; model != NULL; model = model->BSIM4v2nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v2SbulkJctPotential < 0.1)  
	 {   model->BSIM4v2SbulkJctPotential = 0.1;
	     fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
	 }
         if (model->BSIM4v2SsidewallJctPotential < 0.1)
	 {   model->BSIM4v2SsidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
	 }
         if (model->BSIM4v2SGatesidewallJctPotential < 0.1)
	 {   model->BSIM4v2SGatesidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
	 }

         if (model->BSIM4v2DbulkJctPotential < 0.1) 
         {   model->BSIM4v2DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v2DsidewallJctPotential < 0.1)
         {   model->BSIM4v2DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v2DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v2DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4v2toxeGiven) && (model->BSIM4v2toxpGiven) && (model->BSIM4v2dtoxGiven)
             && (model->BSIM4v2toxe != (model->BSIM4v2toxp + model->BSIM4v2dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
	 else if ((model->BSIM4v2toxeGiven) && (!model->BSIM4v2toxpGiven))
	     model->BSIM4v2toxp = model->BSIM4v2toxe - model->BSIM4v2dtox;
	 else if ((!model->BSIM4v2toxeGiven) && (model->BSIM4v2toxpGiven))
             model->BSIM4v2toxe = model->BSIM4v2toxp + model->BSIM4v2dtox;

         model->BSIM4v2coxe = model->BSIM4v2epsrox * EPS0 / model->BSIM4v2toxe;
         model->BSIM4v2coxp = model->BSIM4v2epsrox * EPS0 / model->BSIM4v2toxp;

         if (!model->BSIM4v2cgdoGiven)
         {   if (model->BSIM4v2dlcGiven && (model->BSIM4v2dlc > 0.0))
                 model->BSIM4v2cgdo = model->BSIM4v2dlc * model->BSIM4v2coxe
                                  - model->BSIM4v2cgdl ;
             else
                 model->BSIM4v2cgdo = 0.6 * model->BSIM4v2xj * model->BSIM4v2coxe;
         }
         if (!model->BSIM4v2cgsoGiven)
         {   if (model->BSIM4v2dlcGiven && (model->BSIM4v2dlc > 0.0))
                 model->BSIM4v2cgso = model->BSIM4v2dlc * model->BSIM4v2coxe
                                  - model->BSIM4v2cgsl ;
             else
                 model->BSIM4v2cgso = 0.6 * model->BSIM4v2xj * model->BSIM4v2coxe;
         }
         if (!model->BSIM4v2cgboGiven)
             model->BSIM4v2cgbo = 2.0 * model->BSIM4v2dwc * model->BSIM4v2coxe;

         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

	 Tnom = model->BSIM4v2tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM4v2vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v2factor1 = sqrt(EPSSI / (model->BSIM4v2epsrox * EPS0)
                             * model->BSIM4v2toxe);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4v2vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v2vtm;
	     T1 = log(Temp / Tnom);
	     T2 = T0 + model->BSIM4v2SjctTempExponent * T1;
	     T3 = exp(T2 / model->BSIM4v2SjctEmissionCoeff);
	     model->BSIM4v2SjctTempSatCurDensity = model->BSIM4v2SjctSatCurDensity
					       * T3;
	     model->BSIM4v2SjctSidewallTempSatCurDensity
			 = model->BSIM4v2SjctSidewallSatCurDensity * T3;
             model->BSIM4v2SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v2SjctGateSidewallSatCurDensity * T3;

	     T2 = T0 + model->BSIM4v2DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v2DjctEmissionCoeff);
             model->BSIM4v2DjctTempSatCurDensity = model->BSIM4v2DjctSatCurDensity
                                               * T3;
             model->BSIM4v2DjctSidewallTempSatCurDensity
                         = model->BSIM4v2DjctSidewallSatCurDensity * T3;
             model->BSIM4v2DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v2DjctGateSidewallSatCurDensity * T3;
	 }
	 else
	 {   model->BSIM4v2SjctTempSatCurDensity = model->BSIM4v2SjctSatCurDensity;
	     model->BSIM4v2SjctSidewallTempSatCurDensity
			= model->BSIM4v2SjctSidewallSatCurDensity;
             model->BSIM4v2SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v2SjctGateSidewallSatCurDensity;
             model->BSIM4v2DjctTempSatCurDensity = model->BSIM4v2DjctSatCurDensity;
             model->BSIM4v2DjctSidewallTempSatCurDensity
                        = model->BSIM4v2DjctSidewallSatCurDensity;
             model->BSIM4v2DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v2DjctGateSidewallSatCurDensity;
	 }

	 if (model->BSIM4v2SjctTempSatCurDensity < 0.0)
	     model->BSIM4v2SjctTempSatCurDensity = 0.0;
	 if (model->BSIM4v2SjctSidewallTempSatCurDensity < 0.0)
	     model->BSIM4v2SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v2SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v2SjctGateSidewallTempSatCurDensity = 0.0;

         if (model->BSIM4v2DjctTempSatCurDensity < 0.0)
             model->BSIM4v2DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v2DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v2DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v2DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v2DjctGateSidewallTempSatCurDensity = 0.0;

	 /* Temperature dependence of D/B and S/B diode capacitance begins */
	 delTemp = ckt->CKTtemp - model->BSIM4v2tnom;
	 T0 = model->BSIM4v2tcj * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v2SunitAreaTempJctCap = model->BSIM4v2SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4v2DunitAreaTempJctCap = model->BSIM4v2DunitAreaJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v2SunitAreaJctCap > 0.0)
	     {   model->BSIM4v2SunitAreaTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
	     if (model->BSIM4v2DunitAreaJctCap > 0.0)
             {   model->BSIM4v2DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
	 }
         T0 = model->BSIM4v2tcjsw * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v2SunitLengthSidewallTempJctCap = model->BSIM4v2SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4v2DunitLengthSidewallTempJctCap = model->BSIM4v2DunitLengthSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v2SunitLengthSidewallJctCap > 0.0)
	     {   model->BSIM4v2SunitLengthSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
	     }
	     if (model->BSIM4v2DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v2DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }	
	 }
         T0 = model->BSIM4v2tcjswg * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v2SunitLengthGateSidewallTempJctCap = model->BSIM4v2SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4v2DunitLengthGateSidewallTempJctCap = model->BSIM4v2DunitLengthGateSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v2SunitLengthGateSidewallJctCap > 0.0)
	     {   model->BSIM4v2SunitLengthGateSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
	     }
	     if (model->BSIM4v2DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v2DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
	 }

         model->BSIM4v2PhiBS = model->BSIM4v2SbulkJctPotential
			   - model->BSIM4v2tpb * delTemp;
         if (model->BSIM4v2PhiBS < 0.01)
	 {   model->BSIM4v2PhiBS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
	 }
         model->BSIM4v2PhiBD = model->BSIM4v2DbulkJctPotential
                           - model->BSIM4v2tpb * delTemp;
         if (model->BSIM4v2PhiBD < 0.01)
         {   model->BSIM4v2PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v2PhiBSWS = model->BSIM4v2SsidewallJctPotential
                             - model->BSIM4v2tpbsw * delTemp;
         if (model->BSIM4v2PhiBSWS <= 0.01)
	 {   model->BSIM4v2PhiBSWS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
	 }
         model->BSIM4v2PhiBSWD = model->BSIM4v2DsidewallJctPotential
                             - model->BSIM4v2tpbsw * delTemp;
         if (model->BSIM4v2PhiBSWD <= 0.01)
         {   model->BSIM4v2PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

	 model->BSIM4v2PhiBSWGS = model->BSIM4v2SGatesidewallJctPotential
                              - model->BSIM4v2tpbswg * delTemp;
         if (model->BSIM4v2PhiBSWGS <= 0.01)
	 {   model->BSIM4v2PhiBSWGS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
	 }
         model->BSIM4v2PhiBSWGD = model->BSIM4v2DGatesidewallJctPotential
                              - model->BSIM4v2tpbswg * delTemp;
         if (model->BSIM4v2PhiBSWGD <= 0.01)
         {   model->BSIM4v2PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v2ijthdfwd <= 0.0)
         {   model->BSIM4v2ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v2ijthdfwd);
         }
         if (model->BSIM4v2ijthsfwd <= 0.0)
         {   model->BSIM4v2ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v2ijthsfwd);
         }
	 if (model->BSIM4v2ijthdrev <= 0.0)
         {   model->BSIM4v2ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v2ijthdrev);
         }
         if (model->BSIM4v2ijthsrev <= 0.0)
         {   model->BSIM4v2ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v2ijthsrev);
         }

         if ((model->BSIM4v2xjbvd <= 0.0) && (model->BSIM4v2dioMod == 2))
         {   model->BSIM4v2xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v2xjbvd);
         }
         else if ((model->BSIM4v2xjbvd < 0.0) && (model->BSIM4v2dioMod == 0))
         {   model->BSIM4v2xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v2xjbvd);
         }

         if (model->BSIM4v2bvd <= 0.0)
         {   model->BSIM4v2bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v2bvd);
         }

         if ((model->BSIM4v2xjbvs <= 0.0) && (model->BSIM4v2dioMod == 2))
         {   model->BSIM4v2xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v2xjbvs);
         }
         else if ((model->BSIM4v2xjbvs < 0.0) && (model->BSIM4v2dioMod == 0))
         {   model->BSIM4v2xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v2xjbvs);
         }

         if (model->BSIM4v2bvs <= 0.0)
         {   model->BSIM4v2bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v2bvs);
         }


         /* loop through all the instances of the model */
         for (here = model->BSIM4v2instances; here != NULL;
              here = here->BSIM4v2nextInstance) 
	 {    if (here->BSIM4v2owner != ARCHme) continue;
	      pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM4v2l == pSizeDependParamKnot->Length)
		      && (here->BSIM4v2w == pSizeDependParamKnot->Width)
		      && (here->BSIM4v2nf == pSizeDependParamKnot->NFinger))
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
	      {   pParam = (struct bsim4SizeDependParam *)tmalloc(
	                    sizeof(struct bsim4SizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v2l;
                  pParam->Width = here->BSIM4v2w;
		  pParam->NFinger = here->BSIM4v2nf;
                  Lnew = here->BSIM4v2l  + model->BSIM4v2xl ;
                  Wnew = here->BSIM4v2w / here->BSIM4v2nf + model->BSIM4v2xw;
		  
                  T0 = pow(Lnew, model->BSIM4v2Lln);
                  T1 = pow(Wnew, model->BSIM4v2Lwn);
                  tmp1 = model->BSIM4v2Ll / T0 + model->BSIM4v2Lw / T1
                       + model->BSIM4v2Lwl / (T0 * T1);
                  pParam->BSIM4v2dl = model->BSIM4v2Lint + tmp1;
                  tmp2 = model->BSIM4v2Llc / T0 + model->BSIM4v2Lwc / T1
                       + model->BSIM4v2Lwlc / (T0 * T1);
                  pParam->BSIM4v2dlc = model->BSIM4v2dlc + tmp2;
                  pParam->BSIM4v2dlcig = model->BSIM4v2dlcig + tmp2;

                  T2 = pow(Lnew, model->BSIM4v2Wln);
                  T3 = pow(Wnew, model->BSIM4v2Wwn);
                  tmp1 = model->BSIM4v2Wl / T2 + model->BSIM4v2Ww / T3
                       + model->BSIM4v2Wwl / (T2 * T3);
                  pParam->BSIM4v2dw = model->BSIM4v2Wint + tmp1;
                  tmp2 = model->BSIM4v2Wlc / T2 + model->BSIM4v2Wwc / T3
                       + model->BSIM4v2Wwlc / (T2 * T3); 
                  pParam->BSIM4v2dwc = model->BSIM4v2dwc + tmp2;
                  pParam->BSIM4v2dwj = model->BSIM4v2dwj + tmp2;

                  pParam->BSIM4v2leff = Lnew - 2.0 * pParam->BSIM4v2dl;
                  if (pParam->BSIM4v2leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v2modName;
                      namarray[1] = here->BSIM4v2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v2: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v2weff = Wnew - 2.0 * pParam->BSIM4v2dw;
                  if (pParam->BSIM4v2weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v2modName;
                      namarray[1] = here->BSIM4v2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v2: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v2leffCV = Lnew - 2.0 * pParam->BSIM4v2dlc;
                  if (pParam->BSIM4v2leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v2modName;
                      namarray[1] = here->BSIM4v2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v2: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v2weffCV = Wnew - 2.0 * pParam->BSIM4v2dwc;
                  if (pParam->BSIM4v2weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v2modName;
                      namarray[1] = here->BSIM4v2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v2: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v2weffCJ = Wnew - 2.0 * pParam->BSIM4v2dwj;
                  if (pParam->BSIM4v2weffCJ <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v2modName;
                      namarray[1] = here->BSIM4v2name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v2: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM4v2binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM4v2leff;
		      Inv_W = 1.0e-6 / pParam->BSIM4v2weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM4v2leff
			     * pParam->BSIM4v2weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM4v2leff;
		      Inv_W = 1.0 / pParam->BSIM4v2weff;
		      Inv_LW = 1.0 / (pParam->BSIM4v2leff
			     * pParam->BSIM4v2weff);
		  }
		  pParam->BSIM4v2cdsc = model->BSIM4v2cdsc
				    + model->BSIM4v2lcdsc * Inv_L
				    + model->BSIM4v2wcdsc * Inv_W
				    + model->BSIM4v2pcdsc * Inv_LW;
		  pParam->BSIM4v2cdscb = model->BSIM4v2cdscb
				     + model->BSIM4v2lcdscb * Inv_L
				     + model->BSIM4v2wcdscb * Inv_W
				     + model->BSIM4v2pcdscb * Inv_LW; 
				     
    		  pParam->BSIM4v2cdscd = model->BSIM4v2cdscd
				     + model->BSIM4v2lcdscd * Inv_L
				     + model->BSIM4v2wcdscd * Inv_W
				     + model->BSIM4v2pcdscd * Inv_LW; 
				     
		  pParam->BSIM4v2cit = model->BSIM4v2cit
				   + model->BSIM4v2lcit * Inv_L
				   + model->BSIM4v2wcit * Inv_W
				   + model->BSIM4v2pcit * Inv_LW;
		  pParam->BSIM4v2nfactor = model->BSIM4v2nfactor
				       + model->BSIM4v2lnfactor * Inv_L
				       + model->BSIM4v2wnfactor * Inv_W
				       + model->BSIM4v2pnfactor * Inv_LW;
		  pParam->BSIM4v2xj = model->BSIM4v2xj
				  + model->BSIM4v2lxj * Inv_L
				  + model->BSIM4v2wxj * Inv_W
				  + model->BSIM4v2pxj * Inv_LW;
		  pParam->BSIM4v2vsat = model->BSIM4v2vsat
				    + model->BSIM4v2lvsat * Inv_L
				    + model->BSIM4v2wvsat * Inv_W
				    + model->BSIM4v2pvsat * Inv_LW;
		  pParam->BSIM4v2at = model->BSIM4v2at
				  + model->BSIM4v2lat * Inv_L
				  + model->BSIM4v2wat * Inv_W
				  + model->BSIM4v2pat * Inv_LW;
		  pParam->BSIM4v2a0 = model->BSIM4v2a0
				  + model->BSIM4v2la0 * Inv_L
				  + model->BSIM4v2wa0 * Inv_W
				  + model->BSIM4v2pa0 * Inv_LW; 
				  
		  pParam->BSIM4v2ags = model->BSIM4v2ags
				  + model->BSIM4v2lags * Inv_L
				  + model->BSIM4v2wags * Inv_W
				  + model->BSIM4v2pags * Inv_LW;
				  
		  pParam->BSIM4v2a1 = model->BSIM4v2a1
				  + model->BSIM4v2la1 * Inv_L
				  + model->BSIM4v2wa1 * Inv_W
				  + model->BSIM4v2pa1 * Inv_LW;
		  pParam->BSIM4v2a2 = model->BSIM4v2a2
				  + model->BSIM4v2la2 * Inv_L
				  + model->BSIM4v2wa2 * Inv_W
				  + model->BSIM4v2pa2 * Inv_LW;
		  pParam->BSIM4v2keta = model->BSIM4v2keta
				    + model->BSIM4v2lketa * Inv_L
				    + model->BSIM4v2wketa * Inv_W
				    + model->BSIM4v2pketa * Inv_LW;
		  pParam->BSIM4v2nsub = model->BSIM4v2nsub
				    + model->BSIM4v2lnsub * Inv_L
				    + model->BSIM4v2wnsub * Inv_W
				    + model->BSIM4v2pnsub * Inv_LW;
		  pParam->BSIM4v2ndep = model->BSIM4v2ndep
				    + model->BSIM4v2lndep * Inv_L
				    + model->BSIM4v2wndep * Inv_W
				    + model->BSIM4v2pndep * Inv_LW;
                  pParam->BSIM4v2nsd = model->BSIM4v2nsd
                                   + model->BSIM4v2lnsd * Inv_L
                                   + model->BSIM4v2wnsd * Inv_W
                                   + model->BSIM4v2pnsd * Inv_LW;
                  pParam->BSIM4v2phin = model->BSIM4v2phin
                                    + model->BSIM4v2lphin * Inv_L
                                    + model->BSIM4v2wphin * Inv_W
                                    + model->BSIM4v2pphin * Inv_LW;
		  pParam->BSIM4v2ngate = model->BSIM4v2ngate
				     + model->BSIM4v2lngate * Inv_L
				     + model->BSIM4v2wngate * Inv_W
				     + model->BSIM4v2pngate * Inv_LW;
		  pParam->BSIM4v2gamma1 = model->BSIM4v2gamma1
				      + model->BSIM4v2lgamma1 * Inv_L
				      + model->BSIM4v2wgamma1 * Inv_W
				      + model->BSIM4v2pgamma1 * Inv_LW;
		  pParam->BSIM4v2gamma2 = model->BSIM4v2gamma2
				      + model->BSIM4v2lgamma2 * Inv_L
				      + model->BSIM4v2wgamma2 * Inv_W
				      + model->BSIM4v2pgamma2 * Inv_LW;
		  pParam->BSIM4v2vbx = model->BSIM4v2vbx
				   + model->BSIM4v2lvbx * Inv_L
				   + model->BSIM4v2wvbx * Inv_W
				   + model->BSIM4v2pvbx * Inv_LW;
		  pParam->BSIM4v2vbm = model->BSIM4v2vbm
				   + model->BSIM4v2lvbm * Inv_L
				   + model->BSIM4v2wvbm * Inv_W
				   + model->BSIM4v2pvbm * Inv_LW;
		  pParam->BSIM4v2xt = model->BSIM4v2xt
				   + model->BSIM4v2lxt * Inv_L
				   + model->BSIM4v2wxt * Inv_W
				   + model->BSIM4v2pxt * Inv_LW;
                  pParam->BSIM4v2vfb = model->BSIM4v2vfb
                                   + model->BSIM4v2lvfb * Inv_L
                                   + model->BSIM4v2wvfb * Inv_W
                                   + model->BSIM4v2pvfb * Inv_LW;
		  pParam->BSIM4v2k1 = model->BSIM4v2k1
				  + model->BSIM4v2lk1 * Inv_L
				  + model->BSIM4v2wk1 * Inv_W
				  + model->BSIM4v2pk1 * Inv_LW;
		  pParam->BSIM4v2kt1 = model->BSIM4v2kt1
				   + model->BSIM4v2lkt1 * Inv_L
				   + model->BSIM4v2wkt1 * Inv_W
				   + model->BSIM4v2pkt1 * Inv_LW;
		  pParam->BSIM4v2kt1l = model->BSIM4v2kt1l
				    + model->BSIM4v2lkt1l * Inv_L
				    + model->BSIM4v2wkt1l * Inv_W
				    + model->BSIM4v2pkt1l * Inv_LW;
		  pParam->BSIM4v2k2 = model->BSIM4v2k2
				  + model->BSIM4v2lk2 * Inv_L
				  + model->BSIM4v2wk2 * Inv_W
				  + model->BSIM4v2pk2 * Inv_LW;
		  pParam->BSIM4v2kt2 = model->BSIM4v2kt2
				   + model->BSIM4v2lkt2 * Inv_L
				   + model->BSIM4v2wkt2 * Inv_W
				   + model->BSIM4v2pkt2 * Inv_LW;
		  pParam->BSIM4v2k3 = model->BSIM4v2k3
				  + model->BSIM4v2lk3 * Inv_L
				  + model->BSIM4v2wk3 * Inv_W
				  + model->BSIM4v2pk3 * Inv_LW;
		  pParam->BSIM4v2k3b = model->BSIM4v2k3b
				   + model->BSIM4v2lk3b * Inv_L
				   + model->BSIM4v2wk3b * Inv_W
				   + model->BSIM4v2pk3b * Inv_LW;
		  pParam->BSIM4v2w0 = model->BSIM4v2w0
				  + model->BSIM4v2lw0 * Inv_L
				  + model->BSIM4v2ww0 * Inv_W
				  + model->BSIM4v2pw0 * Inv_LW;
		  pParam->BSIM4v2lpe0 = model->BSIM4v2lpe0
				    + model->BSIM4v2llpe0 * Inv_L
 				    + model->BSIM4v2wlpe0 * Inv_W
				    + model->BSIM4v2plpe0 * Inv_LW;
                  pParam->BSIM4v2lpeb = model->BSIM4v2lpeb
                                    + model->BSIM4v2llpeb * Inv_L
                                    + model->BSIM4v2wlpeb * Inv_W
                                    + model->BSIM4v2plpeb * Inv_LW;
                  pParam->BSIM4v2dvtp0 = model->BSIM4v2dvtp0
                                     + model->BSIM4v2ldvtp0 * Inv_L
                                     + model->BSIM4v2wdvtp0 * Inv_W
                                     + model->BSIM4v2pdvtp0 * Inv_LW;
                  pParam->BSIM4v2dvtp1 = model->BSIM4v2dvtp1
                                     + model->BSIM4v2ldvtp1 * Inv_L
                                     + model->BSIM4v2wdvtp1 * Inv_W
                                     + model->BSIM4v2pdvtp1 * Inv_LW;
		  pParam->BSIM4v2dvt0 = model->BSIM4v2dvt0
				    + model->BSIM4v2ldvt0 * Inv_L
				    + model->BSIM4v2wdvt0 * Inv_W
				    + model->BSIM4v2pdvt0 * Inv_LW;
		  pParam->BSIM4v2dvt1 = model->BSIM4v2dvt1
				    + model->BSIM4v2ldvt1 * Inv_L
				    + model->BSIM4v2wdvt1 * Inv_W
				    + model->BSIM4v2pdvt1 * Inv_LW;
		  pParam->BSIM4v2dvt2 = model->BSIM4v2dvt2
				    + model->BSIM4v2ldvt2 * Inv_L
				    + model->BSIM4v2wdvt2 * Inv_W
				    + model->BSIM4v2pdvt2 * Inv_LW;
		  pParam->BSIM4v2dvt0w = model->BSIM4v2dvt0w
				    + model->BSIM4v2ldvt0w * Inv_L
				    + model->BSIM4v2wdvt0w * Inv_W
				    + model->BSIM4v2pdvt0w * Inv_LW;
		  pParam->BSIM4v2dvt1w = model->BSIM4v2dvt1w
				    + model->BSIM4v2ldvt1w * Inv_L
				    + model->BSIM4v2wdvt1w * Inv_W
				    + model->BSIM4v2pdvt1w * Inv_LW;
		  pParam->BSIM4v2dvt2w = model->BSIM4v2dvt2w
				    + model->BSIM4v2ldvt2w * Inv_L
				    + model->BSIM4v2wdvt2w * Inv_W
				    + model->BSIM4v2pdvt2w * Inv_LW;
		  pParam->BSIM4v2drout = model->BSIM4v2drout
				     + model->BSIM4v2ldrout * Inv_L
				     + model->BSIM4v2wdrout * Inv_W
				     + model->BSIM4v2pdrout * Inv_LW;
		  pParam->BSIM4v2dsub = model->BSIM4v2dsub
				    + model->BSIM4v2ldsub * Inv_L
				    + model->BSIM4v2wdsub * Inv_W
				    + model->BSIM4v2pdsub * Inv_LW;
		  pParam->BSIM4v2vth0 = model->BSIM4v2vth0
				    + model->BSIM4v2lvth0 * Inv_L
				    + model->BSIM4v2wvth0 * Inv_W
				    + model->BSIM4v2pvth0 * Inv_LW;
		  pParam->BSIM4v2ua = model->BSIM4v2ua
				  + model->BSIM4v2lua * Inv_L
				  + model->BSIM4v2wua * Inv_W
				  + model->BSIM4v2pua * Inv_LW;
		  pParam->BSIM4v2ua1 = model->BSIM4v2ua1
				   + model->BSIM4v2lua1 * Inv_L
				   + model->BSIM4v2wua1 * Inv_W
				   + model->BSIM4v2pua1 * Inv_LW;
		  pParam->BSIM4v2ub = model->BSIM4v2ub
				  + model->BSIM4v2lub * Inv_L
				  + model->BSIM4v2wub * Inv_W
				  + model->BSIM4v2pub * Inv_LW;
		  pParam->BSIM4v2ub1 = model->BSIM4v2ub1
				   + model->BSIM4v2lub1 * Inv_L
				   + model->BSIM4v2wub1 * Inv_W
				   + model->BSIM4v2pub1 * Inv_LW;
		  pParam->BSIM4v2uc = model->BSIM4v2uc
				  + model->BSIM4v2luc * Inv_L
				  + model->BSIM4v2wuc * Inv_W
				  + model->BSIM4v2puc * Inv_LW;
		  pParam->BSIM4v2uc1 = model->BSIM4v2uc1
				   + model->BSIM4v2luc1 * Inv_L
				   + model->BSIM4v2wuc1 * Inv_W
				   + model->BSIM4v2puc1 * Inv_LW;
                  pParam->BSIM4v2eu = model->BSIM4v2eu
                                  + model->BSIM4v2leu * Inv_L
                                  + model->BSIM4v2weu * Inv_W
                                  + model->BSIM4v2peu * Inv_LW;
		  pParam->BSIM4v2u0 = model->BSIM4v2u0
				  + model->BSIM4v2lu0 * Inv_L
				  + model->BSIM4v2wu0 * Inv_W
				  + model->BSIM4v2pu0 * Inv_LW;
		  pParam->BSIM4v2ute = model->BSIM4v2ute
				   + model->BSIM4v2lute * Inv_L
				   + model->BSIM4v2wute * Inv_W
				   + model->BSIM4v2pute * Inv_LW;
		  pParam->BSIM4v2voff = model->BSIM4v2voff
				    + model->BSIM4v2lvoff * Inv_L
				    + model->BSIM4v2wvoff * Inv_W
				    + model->BSIM4v2pvoff * Inv_LW;
                  pParam->BSIM4v2minv = model->BSIM4v2minv
                                    + model->BSIM4v2lminv * Inv_L
                                    + model->BSIM4v2wminv * Inv_W
                                    + model->BSIM4v2pminv * Inv_LW;
                  pParam->BSIM4v2fprout = model->BSIM4v2fprout
                                     + model->BSIM4v2lfprout * Inv_L
                                     + model->BSIM4v2wfprout * Inv_W
                                     + model->BSIM4v2pfprout * Inv_LW;
                  pParam->BSIM4v2pdits = model->BSIM4v2pdits
                                     + model->BSIM4v2lpdits * Inv_L
                                     + model->BSIM4v2wpdits * Inv_W
                                     + model->BSIM4v2ppdits * Inv_LW;
                  pParam->BSIM4v2pditsd = model->BSIM4v2pditsd
                                      + model->BSIM4v2lpditsd * Inv_L
                                      + model->BSIM4v2wpditsd * Inv_W
                                      + model->BSIM4v2ppditsd * Inv_LW;
		  pParam->BSIM4v2delta = model->BSIM4v2delta
				     + model->BSIM4v2ldelta * Inv_L
				     + model->BSIM4v2wdelta * Inv_W
				     + model->BSIM4v2pdelta * Inv_LW;
		  pParam->BSIM4v2rdsw = model->BSIM4v2rdsw
				    + model->BSIM4v2lrdsw * Inv_L
				    + model->BSIM4v2wrdsw * Inv_W
				    + model->BSIM4v2prdsw * Inv_LW;
                  pParam->BSIM4v2rdw = model->BSIM4v2rdw
                                    + model->BSIM4v2lrdw * Inv_L
                                    + model->BSIM4v2wrdw * Inv_W
                                    + model->BSIM4v2prdw * Inv_LW;
                  pParam->BSIM4v2rsw = model->BSIM4v2rsw
                                    + model->BSIM4v2lrsw * Inv_L
                                    + model->BSIM4v2wrsw * Inv_W
                                    + model->BSIM4v2prsw * Inv_LW;
		  pParam->BSIM4v2prwg = model->BSIM4v2prwg
				    + model->BSIM4v2lprwg * Inv_L
				    + model->BSIM4v2wprwg * Inv_W
				    + model->BSIM4v2pprwg * Inv_LW;
		  pParam->BSIM4v2prwb = model->BSIM4v2prwb
				    + model->BSIM4v2lprwb * Inv_L
				    + model->BSIM4v2wprwb * Inv_W
				    + model->BSIM4v2pprwb * Inv_LW;
		  pParam->BSIM4v2prt = model->BSIM4v2prt
				    + model->BSIM4v2lprt * Inv_L
				    + model->BSIM4v2wprt * Inv_W
				    + model->BSIM4v2pprt * Inv_LW;
		  pParam->BSIM4v2eta0 = model->BSIM4v2eta0
				    + model->BSIM4v2leta0 * Inv_L
				    + model->BSIM4v2weta0 * Inv_W
				    + model->BSIM4v2peta0 * Inv_LW;
		  pParam->BSIM4v2etab = model->BSIM4v2etab
				    + model->BSIM4v2letab * Inv_L
				    + model->BSIM4v2wetab * Inv_W
				    + model->BSIM4v2petab * Inv_LW;
		  pParam->BSIM4v2pclm = model->BSIM4v2pclm
				    + model->BSIM4v2lpclm * Inv_L
				    + model->BSIM4v2wpclm * Inv_W
				    + model->BSIM4v2ppclm * Inv_LW;
		  pParam->BSIM4v2pdibl1 = model->BSIM4v2pdibl1
				      + model->BSIM4v2lpdibl1 * Inv_L
				      + model->BSIM4v2wpdibl1 * Inv_W
				      + model->BSIM4v2ppdibl1 * Inv_LW;
		  pParam->BSIM4v2pdibl2 = model->BSIM4v2pdibl2
				      + model->BSIM4v2lpdibl2 * Inv_L
				      + model->BSIM4v2wpdibl2 * Inv_W
				      + model->BSIM4v2ppdibl2 * Inv_LW;
		  pParam->BSIM4v2pdiblb = model->BSIM4v2pdiblb
				      + model->BSIM4v2lpdiblb * Inv_L
				      + model->BSIM4v2wpdiblb * Inv_W
				      + model->BSIM4v2ppdiblb * Inv_LW;
		  pParam->BSIM4v2pscbe1 = model->BSIM4v2pscbe1
				      + model->BSIM4v2lpscbe1 * Inv_L
				      + model->BSIM4v2wpscbe1 * Inv_W
				      + model->BSIM4v2ppscbe1 * Inv_LW;
		  pParam->BSIM4v2pscbe2 = model->BSIM4v2pscbe2
				      + model->BSIM4v2lpscbe2 * Inv_L
				      + model->BSIM4v2wpscbe2 * Inv_W
				      + model->BSIM4v2ppscbe2 * Inv_LW;
		  pParam->BSIM4v2pvag = model->BSIM4v2pvag
				    + model->BSIM4v2lpvag * Inv_L
				    + model->BSIM4v2wpvag * Inv_W
				    + model->BSIM4v2ppvag * Inv_LW;
		  pParam->BSIM4v2wr = model->BSIM4v2wr
				  + model->BSIM4v2lwr * Inv_L
				  + model->BSIM4v2wwr * Inv_W
				  + model->BSIM4v2pwr * Inv_LW;
		  pParam->BSIM4v2dwg = model->BSIM4v2dwg
				   + model->BSIM4v2ldwg * Inv_L
				   + model->BSIM4v2wdwg * Inv_W
				   + model->BSIM4v2pdwg * Inv_LW;
		  pParam->BSIM4v2dwb = model->BSIM4v2dwb
				   + model->BSIM4v2ldwb * Inv_L
				   + model->BSIM4v2wdwb * Inv_W
				   + model->BSIM4v2pdwb * Inv_LW;
		  pParam->BSIM4v2b0 = model->BSIM4v2b0
				  + model->BSIM4v2lb0 * Inv_L
				  + model->BSIM4v2wb0 * Inv_W
				  + model->BSIM4v2pb0 * Inv_LW;
		  pParam->BSIM4v2b1 = model->BSIM4v2b1
				  + model->BSIM4v2lb1 * Inv_L
				  + model->BSIM4v2wb1 * Inv_W
				  + model->BSIM4v2pb1 * Inv_LW;
		  pParam->BSIM4v2alpha0 = model->BSIM4v2alpha0
				      + model->BSIM4v2lalpha0 * Inv_L
				      + model->BSIM4v2walpha0 * Inv_W
				      + model->BSIM4v2palpha0 * Inv_LW;
                  pParam->BSIM4v2alpha1 = model->BSIM4v2alpha1
                                      + model->BSIM4v2lalpha1 * Inv_L
                                      + model->BSIM4v2walpha1 * Inv_W
                                      + model->BSIM4v2palpha1 * Inv_LW;
		  pParam->BSIM4v2beta0 = model->BSIM4v2beta0
				     + model->BSIM4v2lbeta0 * Inv_L
				     + model->BSIM4v2wbeta0 * Inv_W
				     + model->BSIM4v2pbeta0 * Inv_LW;
                  pParam->BSIM4v2agidl = model->BSIM4v2agidl
                                     + model->BSIM4v2lagidl * Inv_L
                                     + model->BSIM4v2wagidl * Inv_W
                                     + model->BSIM4v2pagidl * Inv_LW;
                  pParam->BSIM4v2bgidl = model->BSIM4v2bgidl
                                     + model->BSIM4v2lbgidl * Inv_L
                                     + model->BSIM4v2wbgidl * Inv_W
                                     + model->BSIM4v2pbgidl * Inv_LW;
                  pParam->BSIM4v2cgidl = model->BSIM4v2cgidl
                                     + model->BSIM4v2lcgidl * Inv_L
                                     + model->BSIM4v2wcgidl * Inv_W
                                     + model->BSIM4v2pcgidl * Inv_LW;
                  pParam->BSIM4v2egidl = model->BSIM4v2egidl
                                     + model->BSIM4v2legidl * Inv_L
                                     + model->BSIM4v2wegidl * Inv_W
                                     + model->BSIM4v2pegidl * Inv_LW;
                  pParam->BSIM4v2aigc = model->BSIM4v2aigc
                                     + model->BSIM4v2laigc * Inv_L
                                     + model->BSIM4v2waigc * Inv_W
                                     + model->BSIM4v2paigc * Inv_LW;
                  pParam->BSIM4v2bigc = model->BSIM4v2bigc
                                     + model->BSIM4v2lbigc * Inv_L
                                     + model->BSIM4v2wbigc * Inv_W
                                     + model->BSIM4v2pbigc * Inv_LW;
                  pParam->BSIM4v2cigc = model->BSIM4v2cigc
                                     + model->BSIM4v2lcigc * Inv_L
                                     + model->BSIM4v2wcigc * Inv_W
                                     + model->BSIM4v2pcigc * Inv_LW;
                  pParam->BSIM4v2aigsd = model->BSIM4v2aigsd
                                     + model->BSIM4v2laigsd * Inv_L
                                     + model->BSIM4v2waigsd * Inv_W
                                     + model->BSIM4v2paigsd * Inv_LW;
                  pParam->BSIM4v2bigsd = model->BSIM4v2bigsd
                                     + model->BSIM4v2lbigsd * Inv_L
                                     + model->BSIM4v2wbigsd * Inv_W
                                     + model->BSIM4v2pbigsd * Inv_LW;
                  pParam->BSIM4v2cigsd = model->BSIM4v2cigsd
                                     + model->BSIM4v2lcigsd * Inv_L
                                     + model->BSIM4v2wcigsd * Inv_W
                                     + model->BSIM4v2pcigsd * Inv_LW;
                  pParam->BSIM4v2aigbacc = model->BSIM4v2aigbacc
                                       + model->BSIM4v2laigbacc * Inv_L
                                       + model->BSIM4v2waigbacc * Inv_W
                                       + model->BSIM4v2paigbacc * Inv_LW;
                  pParam->BSIM4v2bigbacc = model->BSIM4v2bigbacc
                                       + model->BSIM4v2lbigbacc * Inv_L
                                       + model->BSIM4v2wbigbacc * Inv_W
                                       + model->BSIM4v2pbigbacc * Inv_LW;
                  pParam->BSIM4v2cigbacc = model->BSIM4v2cigbacc
                                       + model->BSIM4v2lcigbacc * Inv_L
                                       + model->BSIM4v2wcigbacc * Inv_W
                                       + model->BSIM4v2pcigbacc * Inv_LW;
                  pParam->BSIM4v2aigbinv = model->BSIM4v2aigbinv
                                       + model->BSIM4v2laigbinv * Inv_L
                                       + model->BSIM4v2waigbinv * Inv_W
                                       + model->BSIM4v2paigbinv * Inv_LW;
                  pParam->BSIM4v2bigbinv = model->BSIM4v2bigbinv
                                       + model->BSIM4v2lbigbinv * Inv_L
                                       + model->BSIM4v2wbigbinv * Inv_W
                                       + model->BSIM4v2pbigbinv * Inv_LW;
                  pParam->BSIM4v2cigbinv = model->BSIM4v2cigbinv
                                       + model->BSIM4v2lcigbinv * Inv_L
                                       + model->BSIM4v2wcigbinv * Inv_W
                                       + model->BSIM4v2pcigbinv * Inv_LW;
                  pParam->BSIM4v2nigc = model->BSIM4v2nigc
                                       + model->BSIM4v2lnigc * Inv_L
                                       + model->BSIM4v2wnigc * Inv_W
                                       + model->BSIM4v2pnigc * Inv_LW;
                  pParam->BSIM4v2nigbacc = model->BSIM4v2nigbacc
                                       + model->BSIM4v2lnigbacc * Inv_L
                                       + model->BSIM4v2wnigbacc * Inv_W
                                       + model->BSIM4v2pnigbacc * Inv_LW;
                  pParam->BSIM4v2nigbinv = model->BSIM4v2nigbinv
                                       + model->BSIM4v2lnigbinv * Inv_L
                                       + model->BSIM4v2wnigbinv * Inv_W
                                       + model->BSIM4v2pnigbinv * Inv_LW;
                  pParam->BSIM4v2ntox = model->BSIM4v2ntox
                                    + model->BSIM4v2lntox * Inv_L
                                    + model->BSIM4v2wntox * Inv_W
                                    + model->BSIM4v2pntox * Inv_LW;
                  pParam->BSIM4v2eigbinv = model->BSIM4v2eigbinv
                                       + model->BSIM4v2leigbinv * Inv_L
                                       + model->BSIM4v2weigbinv * Inv_W
                                       + model->BSIM4v2peigbinv * Inv_LW;
                  pParam->BSIM4v2pigcd = model->BSIM4v2pigcd
                                     + model->BSIM4v2lpigcd * Inv_L
                                     + model->BSIM4v2wpigcd * Inv_W
                                     + model->BSIM4v2ppigcd * Inv_LW;
                  pParam->BSIM4v2poxedge = model->BSIM4v2poxedge
                                       + model->BSIM4v2lpoxedge * Inv_L
                                       + model->BSIM4v2wpoxedge * Inv_W
                                       + model->BSIM4v2ppoxedge * Inv_LW;
                  pParam->BSIM4v2xrcrg1 = model->BSIM4v2xrcrg1
                                      + model->BSIM4v2lxrcrg1 * Inv_L
                                      + model->BSIM4v2wxrcrg1 * Inv_W
                                      + model->BSIM4v2pxrcrg1 * Inv_LW;
                  pParam->BSIM4v2xrcrg2 = model->BSIM4v2xrcrg2
                                      + model->BSIM4v2lxrcrg2 * Inv_L
                                      + model->BSIM4v2wxrcrg2 * Inv_W
                                      + model->BSIM4v2pxrcrg2 * Inv_LW;

		  pParam->BSIM4v2cgsl = model->BSIM4v2cgsl
				    + model->BSIM4v2lcgsl * Inv_L
				    + model->BSIM4v2wcgsl * Inv_W
				    + model->BSIM4v2pcgsl * Inv_LW;
		  pParam->BSIM4v2cgdl = model->BSIM4v2cgdl
				    + model->BSIM4v2lcgdl * Inv_L
				    + model->BSIM4v2wcgdl * Inv_W
				    + model->BSIM4v2pcgdl * Inv_LW;
		  pParam->BSIM4v2ckappas = model->BSIM4v2ckappas
				       + model->BSIM4v2lckappas * Inv_L
				       + model->BSIM4v2wckappas * Inv_W
 				       + model->BSIM4v2pckappas * Inv_LW;
                  pParam->BSIM4v2ckappad = model->BSIM4v2ckappad
                                       + model->BSIM4v2lckappad * Inv_L
                                       + model->BSIM4v2wckappad * Inv_W
                                       + model->BSIM4v2pckappad * Inv_LW;
		  pParam->BSIM4v2cf = model->BSIM4v2cf
				  + model->BSIM4v2lcf * Inv_L
				  + model->BSIM4v2wcf * Inv_W
				  + model->BSIM4v2pcf * Inv_LW;
		  pParam->BSIM4v2clc = model->BSIM4v2clc
				   + model->BSIM4v2lclc * Inv_L
				   + model->BSIM4v2wclc * Inv_W
				   + model->BSIM4v2pclc * Inv_LW;
		  pParam->BSIM4v2cle = model->BSIM4v2cle
				   + model->BSIM4v2lcle * Inv_L
				   + model->BSIM4v2wcle * Inv_W
				   + model->BSIM4v2pcle * Inv_LW;
		  pParam->BSIM4v2vfbcv = model->BSIM4v2vfbcv
				     + model->BSIM4v2lvfbcv * Inv_L
				     + model->BSIM4v2wvfbcv * Inv_W
				     + model->BSIM4v2pvfbcv * Inv_LW;
                  pParam->BSIM4v2acde = model->BSIM4v2acde
                                    + model->BSIM4v2lacde * Inv_L
                                    + model->BSIM4v2wacde * Inv_W
                                    + model->BSIM4v2pacde * Inv_LW;
                  pParam->BSIM4v2moin = model->BSIM4v2moin
                                    + model->BSIM4v2lmoin * Inv_L
                                    + model->BSIM4v2wmoin * Inv_W
                                    + model->BSIM4v2pmoin * Inv_LW;
                  pParam->BSIM4v2noff = model->BSIM4v2noff
                                    + model->BSIM4v2lnoff * Inv_L
                                    + model->BSIM4v2wnoff * Inv_W
                                    + model->BSIM4v2pnoff * Inv_LW;
                  pParam->BSIM4v2voffcv = model->BSIM4v2voffcv
                                      + model->BSIM4v2lvoffcv * Inv_L
                                      + model->BSIM4v2wvoffcv * Inv_W
                                      + model->BSIM4v2pvoffcv * Inv_LW;

                  pParam->BSIM4v2abulkCVfactor = 1.0 + pow((pParam->BSIM4v2clc
					     / pParam->BSIM4v2leffCV),
					     pParam->BSIM4v2cle);

	          T0 = (TRatio - 1.0);
	          pParam->BSIM4v2ua = pParam->BSIM4v2ua + pParam->BSIM4v2ua1 * T0;
	          pParam->BSIM4v2ub = pParam->BSIM4v2ub + pParam->BSIM4v2ub1 * T0;
	          pParam->BSIM4v2uc = pParam->BSIM4v2uc + pParam->BSIM4v2uc1 * T0;
                  if (pParam->BSIM4v2u0 > 1.0) 
                      pParam->BSIM4v2u0 = pParam->BSIM4v2u0 / 1.0e4;

                  pParam->BSIM4v2u0temp = pParam->BSIM4v2u0
				      * pow(TRatio, pParam->BSIM4v2ute); 
                  pParam->BSIM4v2vsattemp = pParam->BSIM4v2vsat - pParam->BSIM4v2at 
			                * T0;
                  if (pParam->BSIM4v2eu < 0.0)
                  {   pParam->BSIM4v2eu = 0.0;
		      printf("Warning: eu has been negative; reset to 0.0.\n");
		  }


		  PowWeffWr = pow(pParam->BSIM4v2weffCJ * 1.0e6, pParam->BSIM4v2wr) * here->BSIM4v2nf;
		  /* External Rd(V) */
		  T1 = pParam->BSIM4v2rdw + pParam->BSIM4v2prt * T0;
		  if (T1 < 0.0)
		  {   T1 = 0.0;
		      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
		  }
                  T2 = model->BSIM4v2rdwmin + pParam->BSIM4v2prt * T0;
		  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
		  pParam->BSIM4v2rd0 = T1 / PowWeffWr;
                  pParam->BSIM4v2rdwmin = T2 / PowWeffWr;


		  /* External Rs(V) */
		  T1 = pParam->BSIM4v2rsw + pParam->BSIM4v2prt * T0;
                  if (T1 < 0.0)
                  {   T1 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  T2 = model->BSIM4v2rswmin + pParam->BSIM4v2prt * T0;
                  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v2rs0 = T1 / PowWeffWr;
                  pParam->BSIM4v2rswmin = T2 / PowWeffWr;

		  /* Internal Rds(V) in IV */
	          pParam->BSIM4v2rds0 = (pParam->BSIM4v2rdsw + pParam->BSIM4v2prt * T0)
				    * here->BSIM4v2nf / PowWeffWr;
		  pParam->BSIM4v2rdswmin = (model->BSIM4v2rdswmin + pParam->BSIM4v2prt * T0)
				       * here->BSIM4v2nf / PowWeffWr;

                  pParam->BSIM4v2cgdo = (model->BSIM4v2cgdo + pParam->BSIM4v2cf)
				    * pParam->BSIM4v2weffCV;
                  pParam->BSIM4v2cgso = (model->BSIM4v2cgso + pParam->BSIM4v2cf)
				    * pParam->BSIM4v2weffCV;
                  pParam->BSIM4v2cgbo = model->BSIM4v2cgbo * pParam->BSIM4v2leffCV * here->BSIM4v2nf;

                  if (!model->BSIM4v2ndepGiven && model->BSIM4v2gamma1Given)
                  {   T0 = pParam->BSIM4v2gamma1 * model->BSIM4v2coxe;
                      pParam->BSIM4v2ndep = 3.01248e22 * T0 * T0;
                  }

		  pParam->BSIM4v2phi = Vtm0 * log(pParam->BSIM4v2ndep / ni)
				   + pParam->BSIM4v2phin + 0.4;

	          pParam->BSIM4v2sqrtPhi = sqrt(pParam->BSIM4v2phi);
	          pParam->BSIM4v2phis3 = pParam->BSIM4v2sqrtPhi * pParam->BSIM4v2phi;

                  pParam->BSIM4v2Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM4v2ndep * 1.0e6))
                                     * pParam->BSIM4v2sqrtPhi; 
                  pParam->BSIM4v2sqrtXdep0 = sqrt(pParam->BSIM4v2Xdep0);
                  pParam->BSIM4v2litl = sqrt(3.0 * pParam->BSIM4v2xj
				    * model->BSIM4v2toxe);
                  pParam->BSIM4v2vbi = Vtm0 * log(pParam->BSIM4v2nsd
			           * pParam->BSIM4v2ndep / (ni * ni));

		  if (pParam->BSIM4v2ngate > 0.0)
                  {   pParam->BSIM4v2vfbsd = Vtm0 * log(pParam->BSIM4v2ngate
                                         / pParam->BSIM4v2nsd);
		  }
		  else
		      pParam->BSIM4v2vfbsd = 0.0;

                  pParam->BSIM4v2cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM4v2ndep * 1.0e6 / 2.0
				     / pParam->BSIM4v2phi);

                  pParam->BSIM4v2ToxRatio = exp(pParam->BSIM4v2ntox
					* log(model->BSIM4v2toxref / model->BSIM4v2toxe))
					/ model->BSIM4v2toxe / model->BSIM4v2toxe;
                  pParam->BSIM4v2ToxRatioEdge = exp(pParam->BSIM4v2ntox
                                            * log(model->BSIM4v2toxref
                                            / (model->BSIM4v2toxe * pParam->BSIM4v2poxedge)))
                                            / model->BSIM4v2toxe / model->BSIM4v2toxe
                                            / pParam->BSIM4v2poxedge / pParam->BSIM4v2poxedge;
                  pParam->BSIM4v2Aechvb = (model->BSIM4v2type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v2Bechvb = (model->BSIM4v2type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v2AechvbEdge = pParam->BSIM4v2Aechvb * pParam->BSIM4v2weff
					  * pParam->BSIM4v2dlcig * pParam->BSIM4v2ToxRatioEdge;
                  pParam->BSIM4v2BechvbEdge = -pParam->BSIM4v2Bechvb
					  * model->BSIM4v2toxe * pParam->BSIM4v2poxedge;
                  pParam->BSIM4v2Aechvb *= pParam->BSIM4v2weff * pParam->BSIM4v2leff
				       * pParam->BSIM4v2ToxRatio;
                  pParam->BSIM4v2Bechvb *= -model->BSIM4v2toxe;


                  pParam->BSIM4v2mstar = 0.5 + atan(pParam->BSIM4v2minv) / PI;
                  pParam->BSIM4v2voffcbn =  pParam->BSIM4v2voff + model->BSIM4v2voffl / pParam->BSIM4v2leff;

                  pParam->BSIM4v2ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4v2ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v2acde *= pow((pParam->BSIM4v2ndep / 2.0e16), -0.25);


                  if (model->BSIM4v2k1Given || model->BSIM4v2k2Given)
	          {   if (!model->BSIM4v2k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v2k1 = 0.53;
                      }
                      if (!model->BSIM4v2k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v2k2 = -0.0186;
                      }
                      if (model->BSIM4v2nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v2xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v2vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v2gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v2gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM4v2vbxGiven)
                          pParam->BSIM4v2vbx = pParam->BSIM4v2phi - 7.7348e-4 
                                           * pParam->BSIM4v2ndep
					   * pParam->BSIM4v2xt * pParam->BSIM4v2xt;
	              if (pParam->BSIM4v2vbx > 0.0)
		          pParam->BSIM4v2vbx = -pParam->BSIM4v2vbx;
	              if (pParam->BSIM4v2vbm > 0.0)
                          pParam->BSIM4v2vbm = -pParam->BSIM4v2vbm;
           
                      if (!model->BSIM4v2gamma1Given)
                          pParam->BSIM4v2gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM4v2ndep)
                                              / model->BSIM4v2coxe;
                      if (!model->BSIM4v2gamma2Given)
                          pParam->BSIM4v2gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM4v2nsub)
                                              / model->BSIM4v2coxe;

                      T0 = pParam->BSIM4v2gamma1 - pParam->BSIM4v2gamma2;
                      T1 = sqrt(pParam->BSIM4v2phi - pParam->BSIM4v2vbx)
			 - pParam->BSIM4v2sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v2phi * (pParam->BSIM4v2phi
			 - pParam->BSIM4v2vbm)) - pParam->BSIM4v2phi;
                      pParam->BSIM4v2k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v2vbm);
                      pParam->BSIM4v2k1 = pParam->BSIM4v2gamma2 - 2.0
				      * pParam->BSIM4v2k2 * sqrt(pParam->BSIM4v2phi
				      - pParam->BSIM4v2vbm);
                  }
 
		  if (pParam->BSIM4v2k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM4v2k1 / pParam->BSIM4v2k2;
                      pParam->BSIM4v2vbsc = 0.9 * (pParam->BSIM4v2phi - T0 * T0);
		      if (pParam->BSIM4v2vbsc > -3.0)
		          pParam->BSIM4v2vbsc = -3.0;
		      else if (pParam->BSIM4v2vbsc < -30.0)
		          pParam->BSIM4v2vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM4v2vbsc = -30.0;
		  }
		  if (pParam->BSIM4v2vbsc > pParam->BSIM4v2vbm)
		      pParam->BSIM4v2vbsc = pParam->BSIM4v2vbm;

                  if (!model->BSIM4v2vfbGiven)
                  {   if (model->BSIM4v2vth0Given)
                      {   pParam->BSIM4v2vfb = model->BSIM4v2type * pParam->BSIM4v2vth0
                                           - pParam->BSIM4v2phi - pParam->BSIM4v2k1
                                           * pParam->BSIM4v2sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4v2vfb = -1.0;
                      }
                  }
                  if (!model->BSIM4v2vth0Given)
                  {   pParam->BSIM4v2vth0 = model->BSIM4v2type * (pParam->BSIM4v2vfb
                                        + pParam->BSIM4v2phi + pParam->BSIM4v2k1
                                        * pParam->BSIM4v2sqrtPhi);
                  }

                  pParam->BSIM4v2k1ox = pParam->BSIM4v2k1 * model->BSIM4v2toxe
                                    / model->BSIM4v2toxm;
                  pParam->BSIM4v2k2ox = pParam->BSIM4v2k2 * model->BSIM4v2toxe
                                    / model->BSIM4v2toxm;

		  T3 = model->BSIM4v2type * pParam->BSIM4v2vth0
		     - pParam->BSIM4v2vfb - pParam->BSIM4v2phi;
		  T4 = T3 + T3;
		  T5 = 2.5 * T3;
                  pParam->BSIM4v2vtfbphi1 = (model->BSIM4v2type == NMOS) ? T4 : T5; 
		  if (pParam->BSIM4v2vtfbphi1 < 0.0)
		      pParam->BSIM4v2vtfbphi1 = 0.0;

                  pParam->BSIM4v2vtfbphi2 = 4.0 * T3;
                  if (pParam->BSIM4v2vtfbphi2 < 0.0)
                      pParam->BSIM4v2vtfbphi2 = 0.0;

                  tmp = sqrt(EPSSI / (model->BSIM4v2epsrox * EPS0)
                      * model->BSIM4v2toxe * pParam->BSIM4v2Xdep0);
          	  T0 = pParam->BSIM4v2dsub * pParam->BSIM4v2leff / tmp;
                  if (T0 < EXP_THRESHOLD)
          	  {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
              	      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v2theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v2theta0vb0 = 1.0 / (MAX_EXP - 2.0);

 	          T0 = pParam->BSIM4v2drout * pParam->BSIM4v2leff / tmp;
        	  if (T0 < EXP_THRESHOLD)
       	          {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v2thetaRout = pParam->BSIM4v2pdibl1 * T5
                                         + pParam->BSIM4v2pdibl2;

                  tmp = sqrt(pParam->BSIM4v2Xdep0);
                  tmp1 = pParam->BSIM4v2vbi - pParam->BSIM4v2phi;
                  tmp2 = model->BSIM4v2factor1 * tmp;

                  T0 = pParam->BSIM4v2dvt1w * pParam->BSIM4v2weff
                     * pParam->BSIM4v2leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v2dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v2dvt1 * pParam->BSIM4v2leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  }
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v2dvt0 * T9 * tmp1;

                  T4 = model->BSIM4v2toxe * pParam->BSIM4v2phi
                     / (pParam->BSIM4v2weff + pParam->BSIM4v2w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v2lpe0 / pParam->BSIM4v2leff);
                  T5 = pParam->BSIM4v2k1ox * (T0 - 1.0) * pParam->BSIM4v2sqrtPhi
                     + (pParam->BSIM4v2kt1 + pParam->BSIM4v2kt1l / pParam->BSIM4v2leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM4v2type * pParam->BSIM4v2vth0
                       - T8 - T9 + pParam->BSIM4v2k3 * T4 + T5;
                  pParam->BSIM4v2vfbzb = tmp3 - pParam->BSIM4v2phi - pParam->BSIM4v2k1
                                     * pParam->BSIM4v2sqrtPhi; /* End of vfbzb */
              } /* End of SizeNotFound */

              here->BSIM4v2cgso = pParam->BSIM4v2cgso;
              here->BSIM4v2cgdo = pParam->BSIM4v2cgdo;

              if (here->BSIM4v2rbodyMod)
              {   if (here->BSIM4v2rbdb < 1.0e-3)
                      here->BSIM4v2grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v2grbdb = model->BSIM4v2gbmin + 1.0 / here->BSIM4v2rbdb;
                  if (here->BSIM4v2rbpb < 1.0e-3)
                      here->BSIM4v2grbpb = 1.0e3;
                  else
                      here->BSIM4v2grbpb = model->BSIM4v2gbmin + 1.0 / here->BSIM4v2rbpb;
                  if (here->BSIM4v2rbps < 1.0e-3)
                      here->BSIM4v2grbps = 1.0e3;
                  else
                      here->BSIM4v2grbps = model->BSIM4v2gbmin + 1.0 / here->BSIM4v2rbps;
                  if (here->BSIM4v2rbsb < 1.0e-3)
                      here->BSIM4v2grbsb = 1.0e3;
                  else
                      here->BSIM4v2grbsb = model->BSIM4v2gbmin + 1.0 / here->BSIM4v2rbsb;
                  if (here->BSIM4v2rbpd < 1.0e-3)
                      here->BSIM4v2grbpd = 1.0e3;
                  else
                      here->BSIM4v2grbpd = model->BSIM4v2gbmin + 1.0 / here->BSIM4v2rbpd;
              }


              /* 
               * Process geomertry dependent parasitics
	       */

              here->BSIM4v2grgeltd = model->BSIM4v2rshg * (model->BSIM4v2xgw
                      + pParam->BSIM4v2weffCJ / 3.0 / model->BSIM4v2ngcon) /
                      (model->BSIM4v2ngcon * here->BSIM4v2nf *
                      (Lnew - model->BSIM4v2xgl));
              if (here->BSIM4v2grgeltd > 0.0)
                  here->BSIM4v2grgeltd = 1.0 / here->BSIM4v2grgeltd;
              else
              {   here->BSIM4v2grgeltd = 1.0e3; /* mho */
		  if (here->BSIM4v2rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

	      DMCGeff = model->BSIM4v2dmcg - model->BSIM4v2dmcgt;
              DMCIeff = model->BSIM4v2dmci;
              DMDGeff = model->BSIM4v2dmdg - model->BSIM4v2dmcgt;

	      if (here->BSIM4v2sourcePerimeterGiven)
	      {   if (model->BSIM4v2perMod == 0)
	              here->BSIM4v2Pseff = here->BSIM4v2sourcePerimeter;
		  else
		      here->BSIM4v2Pseff = here->BSIM4v2sourcePerimeter 
				       - pParam->BSIM4v2weffCJ * here->BSIM4v2nf;
	      }
	      else
	          BSIM4v2PAeffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod, here->BSIM4v2min, 
                                    pParam->BSIM4v2weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &(here->BSIM4v2Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4v2drainPerimeterGiven)
              {   if (model->BSIM4v2perMod == 0)
                      here->BSIM4v2Pdeff = here->BSIM4v2drainPerimeter;
                  else
                      here->BSIM4v2Pdeff = here->BSIM4v2drainPerimeter 
				       - pParam->BSIM4v2weffCJ * here->BSIM4v2nf;
              }
              else
                  BSIM4v2PAeffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod, here->BSIM4v2min,
                                    pParam->BSIM4v2weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &(here->BSIM4v2Pdeff), &dumAs, &dumAd);

              if (here->BSIM4v2sourceAreaGiven)
                  here->BSIM4v2Aseff = here->BSIM4v2sourceArea;
              else
                  BSIM4v2PAeffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod, here->BSIM4v2min,
                                    pParam->BSIM4v2weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &(here->BSIM4v2Aseff), &dumAd);

              if (here->BSIM4v2drainAreaGiven)
                  here->BSIM4v2Adeff = here->BSIM4v2drainArea;
              else
                  BSIM4v2PAeffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod, here->BSIM4v2min,
                                    pParam->BSIM4v2weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &dumAs, &(here->BSIM4v2Adeff));

	      /* Processing S/D resistance and conductance below */
              if(here->BSIM4v2sNodePrime != here->BSIM4v2sNode)
              {
                 here->BSIM4v2sourceConductance = 0.0;
                 if(here->BSIM4v2sourceSquaresGiven)
                 {
                    here->BSIM4v2sourceConductance = model->BSIM4v2sheetResistance
                                               * here->BSIM4v2sourceSquares;
                 } else if (here->BSIM4v2rgeoMod > 0)
                 {
                    BSIM4v2RdseffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod,
                      here->BSIM4v2rgeoMod, here->BSIM4v2min,
                      pParam->BSIM4v2weffCJ, model->BSIM4v2sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v2sourceConductance));
                 } else
                 {
                    here->BSIM4v2sourceConductance = 0.0;
                 }

                 if (here->BSIM4v2sourceConductance > 0.0)
                     here->BSIM4v2sourceConductance = 1.0
                                            / here->BSIM4v2sourceConductance;
                 else
                 {
                     here->BSIM4v2sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4v2sourceConductance = 0.0;
              }

              if(here->BSIM4v2dNodePrime != here->BSIM4v2dNode)
              {
                 here->BSIM4v2drainConductance = 0.0;
                 if(here->BSIM4v2drainSquaresGiven)
                 {
                    here->BSIM4v2drainConductance = model->BSIM4v2sheetResistance
                                              * here->BSIM4v2drainSquares;
                 } else if (here->BSIM4v2rgeoMod > 0)
                 {
                    BSIM4v2RdseffGeo(here->BSIM4v2nf, here->BSIM4v2geoMod,
                      here->BSIM4v2rgeoMod, here->BSIM4v2min,
                      pParam->BSIM4v2weffCJ, model->BSIM4v2sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v2drainConductance));
                 } else
                 {
                    here->BSIM4v2drainConductance = 0.0;
                 }

                 if (here->BSIM4v2drainConductance > 0.0)
                     here->BSIM4v2drainConductance = 1.0
                                           / here->BSIM4v2drainConductance;
                 else
                 {
                     here->BSIM4v2drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4v2drainConductance = 0.0;
              }
           
               /* End of Rsd processing */


              Nvtms = model->BSIM4v2vtm * model->BSIM4v2SjctEmissionCoeff;
              if ((here->BSIM4v2Aseff <= 0.0) && (here->BSIM4v2Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4v2Aseff * model->BSIM4v2SjctTempSatCurDensity
				   + here->BSIM4v2Pseff * model->BSIM4v2SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v2weffCJ * here->BSIM4v2nf
                                   * model->BSIM4v2SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v2dioMod)
                  {   case 0:
			  if ((model->BSIM4v2bvs / Nvtms) > EXP_THRESHOLD)
			      here->BSIM4v2XExpBVS = model->BSIM4v2xjbvs * MIN_EXP;
			  else
	                      here->BSIM4v2XExpBVS = model->BSIM4v2xjbvs * exp(-model->BSIM4v2bvs / Nvtms);	
		          break;
                      case 1:
                          BSIM4v2DioIjthVjmEval(Nvtms, model->BSIM4v2ijthsfwd, SourceSatCurrent, 
			                      0.0, &(here->BSIM4v2vjsmFwd));
                          here->BSIM4v2IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v2vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v2bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v2XExpBVS = model->BSIM4v2xjbvs * MIN_EXP;
			      tmp = MIN_EXP;
			  }
                          else
			  {   here->BSIM4v2XExpBVS = exp(-model->BSIM4v2bvs / Nvtms);
			      tmp = here->BSIM4v2XExpBVS;
		              here->BSIM4v2XExpBVS *= model->BSIM4v2xjbvs;	
			  }

                          BSIM4v2DioIjthVjmEval(Nvtms, model->BSIM4v2ijthsfwd, SourceSatCurrent, 
                               		      here->BSIM4v2XExpBVS, &(here->BSIM4v2vjsmFwd));
		          T0 = exp(here->BSIM4v2vjsmFwd / Nvtms);
                          here->BSIM4v2IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v2XExpBVS / T0
			  		      + here->BSIM4v2XExpBVS - 1.0);
		          here->BSIM4v2SslpFwd = SourceSatCurrent
					       * (T0 + here->BSIM4v2XExpBVS / T0) / Nvtms;

			  T2 = model->BSIM4v2ijthsrev / SourceSatCurrent;
			  if (T2 < 1.0)
			  {   T2 = 10.0;
			      fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
			  } 
                          here->BSIM4v2vjsmRev = -model->BSIM4v2bvs
					     - Nvtms * log((T2 - 1.0) / model->BSIM4v2xjbvs);
			  T1 = model->BSIM4v2xjbvs * exp(-(model->BSIM4v2bvs
			     + here->BSIM4v2vjsmRev) / Nvtms);
			  here->BSIM4v2IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v2SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v2dioMod);
                  }
              }

              Nvtmd = model->BSIM4v2vtm * model->BSIM4v2DjctEmissionCoeff;
	      if ((here->BSIM4v2Adeff <= 0.0) && (here->BSIM4v2Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4v2Adeff * model->BSIM4v2DjctTempSatCurDensity
				  + here->BSIM4v2Pdeff * model->BSIM4v2DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v2weffCJ * here->BSIM4v2nf
                                  * model->BSIM4v2DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v2dioMod)
                  {   case 0:
                          if ((model->BSIM4v2bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v2XExpBVD = model->BSIM4v2xjbvd * MIN_EXP;
                          else
                          here->BSIM4v2XExpBVD = model->BSIM4v2xjbvd * exp(-model->BSIM4v2bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v2DioIjthVjmEval(Nvtmd, model->BSIM4v2ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v2vjdmFwd));
                          here->BSIM4v2IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v2vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v2bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v2XExpBVD = model->BSIM4v2xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v2XExpBVD = exp(-model->BSIM4v2bvd / Nvtmd);
                              tmp = here->BSIM4v2XExpBVD;
                              here->BSIM4v2XExpBVD *= model->BSIM4v2xjbvd;
                          }

                          BSIM4v2DioIjthVjmEval(Nvtmd, model->BSIM4v2ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v2XExpBVD, &(here->BSIM4v2vjdmFwd));
                          T0 = exp(here->BSIM4v2vjdmFwd / Nvtmd);
                          here->BSIM4v2IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v2XExpBVD / T0
                                              + here->BSIM4v2XExpBVD - 1.0);
                          here->BSIM4v2DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v2XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v2ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0) 
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4v2vjdmRev = -model->BSIM4v2bvd
                                             - Nvtms * log((T2 - 1.0) / model->BSIM4v2xjbvd);
                          T1 = model->BSIM4v2xjbvd * exp(-(model->BSIM4v2bvd
                             + here->BSIM4v2vjdmRev) / Nvtmd);
                          here->BSIM4v2IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v2DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v2dioMod);
                  }
              }

              if (BSIM4v2checkModel(model, here, ckt))
              {   IFuid namarray[2];
                  namarray[0] = model->BSIM4v2modName;
                  namarray[1] = here->BSIM4v2name;
                  (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM4v2.2.1 parameter checking for %s in model %s", namarray);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
