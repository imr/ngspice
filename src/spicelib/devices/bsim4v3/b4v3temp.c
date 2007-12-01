/**** BSIM4.3.0  Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3temp.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "const.h"
#include "sperror.h"

int
BSIM4v3PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
int
BSIM4v3RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5 
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define Charge_q 1.60219e-19
#define DELTA  1.0E-9

int
BSIM4v3DioIjthVjmEval(Nvtm, Ijth, Isb, XExpBV, Vjm)
double Nvtm, Ijth, Isb, XExpBV;
double *Vjm;
{
double Tb, Tc, EVjmovNv;

       Tc = XExpBV;
       Tb = 1.0 + Ijth / Isb - Tc;
       EVjmovNv = 0.5 * (Tb + sqrt(Tb * Tb + 4.0 * Tc));
       *Vjm = Nvtm * log(EVjmovNv);

return 0;
}


int
BSIM4v3temp(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4v3model *model = (BSIM4v3model*) inModel;
BSIM4v3instance *here;
struct bsim4v3SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T8, T9, Lnew, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
double T10;
double Inv_saref, Inv_sbref, Inv_sa, Inv_sb, rho, Ldrn, dvth0_lod;
double W_tmp, Inv_ODeff, OD_offset, dk2_lod, deta0_lod;

int Size_Not_Found, i;

    /*  loop through all the BSIM4v3 device models */
    for (; model != NULL; model = model->BSIM4v3nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v3SbulkJctPotential < 0.1)  
	 {   model->BSIM4v3SbulkJctPotential = 0.1;
	     fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
	 }
         if (model->BSIM4v3SsidewallJctPotential < 0.1)
	 {   model->BSIM4v3SsidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
	 }
         if (model->BSIM4v3SGatesidewallJctPotential < 0.1)
	 {   model->BSIM4v3SGatesidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
	 }

         if (model->BSIM4v3DbulkJctPotential < 0.1) 
         {   model->BSIM4v3DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v3DsidewallJctPotential < 0.1)
         {   model->BSIM4v3DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v3DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v3DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4v3toxeGiven) && (model->BSIM4v3toxpGiven) && (model->BSIM4v3dtoxGiven)
             && (model->BSIM4v3toxe != (model->BSIM4v3toxp + model->BSIM4v3dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
	 else if ((model->BSIM4v3toxeGiven) && (!model->BSIM4v3toxpGiven))
	     model->BSIM4v3toxp = model->BSIM4v3toxe - model->BSIM4v3dtox;
	 else if ((!model->BSIM4v3toxeGiven) && (model->BSIM4v3toxpGiven))
             model->BSIM4v3toxe = model->BSIM4v3toxp + model->BSIM4v3dtox;

         model->BSIM4v3coxe = model->BSIM4v3epsrox * EPS0 / model->BSIM4v3toxe;
         model->BSIM4v3coxp = model->BSIM4v3epsrox * EPS0 / model->BSIM4v3toxp;

         if (!model->BSIM4v3cgdoGiven)
         {   if (model->BSIM4v3dlcGiven && (model->BSIM4v3dlc > 0.0))
                 model->BSIM4v3cgdo = model->BSIM4v3dlc * model->BSIM4v3coxe
                                  - model->BSIM4v3cgdl ;
             else
                 model->BSIM4v3cgdo = 0.6 * model->BSIM4v3xj * model->BSIM4v3coxe;
         }
         if (!model->BSIM4v3cgsoGiven)
         {   if (model->BSIM4v3dlcGiven && (model->BSIM4v3dlc > 0.0))
                 model->BSIM4v3cgso = model->BSIM4v3dlc * model->BSIM4v3coxe
                                  - model->BSIM4v3cgsl ;
             else
                 model->BSIM4v3cgso = 0.6 * model->BSIM4v3xj * model->BSIM4v3coxe;
         }
         if (!model->BSIM4v3cgboGiven)
             model->BSIM4v3cgbo = 2.0 * model->BSIM4v3dwc * model->BSIM4v3coxe;
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

	 Tnom = model->BSIM4v3tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM4v3vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v3factor1 = sqrt(EPSSI / (model->BSIM4v3epsrox * EPS0)
                             * model->BSIM4v3toxe);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4v3vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v3vtm;
	     T1 = log(Temp / Tnom);
	     T2 = T0 + model->BSIM4v3SjctTempExponent * T1;
	     T3 = exp(T2 / model->BSIM4v3SjctEmissionCoeff);
	     model->BSIM4v3SjctTempSatCurDensity = model->BSIM4v3SjctSatCurDensity
					       * T3;
	     model->BSIM4v3SjctSidewallTempSatCurDensity
			 = model->BSIM4v3SjctSidewallSatCurDensity * T3;
             model->BSIM4v3SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v3SjctGateSidewallSatCurDensity * T3;

	     T2 = T0 + model->BSIM4v3DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v3DjctEmissionCoeff);
             model->BSIM4v3DjctTempSatCurDensity = model->BSIM4v3DjctSatCurDensity
                                               * T3;
             model->BSIM4v3DjctSidewallTempSatCurDensity
                         = model->BSIM4v3DjctSidewallSatCurDensity * T3;
             model->BSIM4v3DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v3DjctGateSidewallSatCurDensity * T3;
	 }
	 else
	 {   model->BSIM4v3SjctTempSatCurDensity = model->BSIM4v3SjctSatCurDensity;
	     model->BSIM4v3SjctSidewallTempSatCurDensity
			= model->BSIM4v3SjctSidewallSatCurDensity;
             model->BSIM4v3SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v3SjctGateSidewallSatCurDensity;
             model->BSIM4v3DjctTempSatCurDensity = model->BSIM4v3DjctSatCurDensity;
             model->BSIM4v3DjctSidewallTempSatCurDensity
                        = model->BSIM4v3DjctSidewallSatCurDensity;
             model->BSIM4v3DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v3DjctGateSidewallSatCurDensity;
	 }

	 if (model->BSIM4v3SjctTempSatCurDensity < 0.0)
	     model->BSIM4v3SjctTempSatCurDensity = 0.0;
	 if (model->BSIM4v3SjctSidewallTempSatCurDensity < 0.0)
	     model->BSIM4v3SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v3SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v3SjctGateSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v3DjctTempSatCurDensity < 0.0)
             model->BSIM4v3DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v3DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v3DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v3DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v3DjctGateSidewallTempSatCurDensity = 0.0;

	 /* Temperature dependence of D/B and S/B diode capacitance begins */
	 delTemp = ckt->CKTtemp - model->BSIM4v3tnom;
	 T0 = model->BSIM4v3tcj * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v3SunitAreaTempJctCap = model->BSIM4v3SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4v3DunitAreaTempJctCap = model->BSIM4v3DunitAreaJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v3SunitAreaJctCap > 0.0)
	     {   model->BSIM4v3SunitAreaTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
	     if (model->BSIM4v3DunitAreaJctCap > 0.0)
             {   model->BSIM4v3DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
	 }
         T0 = model->BSIM4v3tcjsw * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v3SunitLengthSidewallTempJctCap = model->BSIM4v3SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4v3DunitLengthSidewallTempJctCap = model->BSIM4v3DunitLengthSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v3SunitLengthSidewallJctCap > 0.0)
	     {   model->BSIM4v3SunitLengthSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
	     }
	     if (model->BSIM4v3DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v3DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }	
	 }
         T0 = model->BSIM4v3tcjswg * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v3SunitLengthGateSidewallTempJctCap = model->BSIM4v3SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4v3DunitLengthGateSidewallTempJctCap = model->BSIM4v3DunitLengthGateSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4v3SunitLengthGateSidewallJctCap > 0.0)
	     {   model->BSIM4v3SunitLengthGateSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
	     }
	     if (model->BSIM4v3DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v3DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
	 }

         model->BSIM4v3PhiBS = model->BSIM4v3SbulkJctPotential
			   - model->BSIM4v3tpb * delTemp;
         if (model->BSIM4v3PhiBS < 0.01)
	 {   model->BSIM4v3PhiBS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
	 }
         model->BSIM4v3PhiBD = model->BSIM4v3DbulkJctPotential
                           - model->BSIM4v3tpb * delTemp;
         if (model->BSIM4v3PhiBD < 0.01)
         {   model->BSIM4v3PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v3PhiBSWS = model->BSIM4v3SsidewallJctPotential
                             - model->BSIM4v3tpbsw * delTemp;
         if (model->BSIM4v3PhiBSWS <= 0.01)
	 {   model->BSIM4v3PhiBSWS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
	 }
         model->BSIM4v3PhiBSWD = model->BSIM4v3DsidewallJctPotential
                             - model->BSIM4v3tpbsw * delTemp;
         if (model->BSIM4v3PhiBSWD <= 0.01)
         {   model->BSIM4v3PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

	 model->BSIM4v3PhiBSWGS = model->BSIM4v3SGatesidewallJctPotential
                              - model->BSIM4v3tpbswg * delTemp;
         if (model->BSIM4v3PhiBSWGS <= 0.01)
	 {   model->BSIM4v3PhiBSWGS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
	 }
         model->BSIM4v3PhiBSWGD = model->BSIM4v3DGatesidewallJctPotential
                              - model->BSIM4v3tpbswg * delTemp;
         if (model->BSIM4v3PhiBSWGD <= 0.01)
         {   model->BSIM4v3PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v3ijthdfwd <= 0.0)
         {   model->BSIM4v3ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v3ijthdfwd);
         }
         if (model->BSIM4v3ijthsfwd <= 0.0)
         {   model->BSIM4v3ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v3ijthsfwd);
         }
	 if (model->BSIM4v3ijthdrev <= 0.0)
         {   model->BSIM4v3ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v3ijthdrev);
         }
         if (model->BSIM4v3ijthsrev <= 0.0)
         {   model->BSIM4v3ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v3ijthsrev);
         }

         if ((model->BSIM4v3xjbvd <= 0.0) && (model->BSIM4v3dioMod == 2))
         {   model->BSIM4v3xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v3xjbvd);
         }
         else if ((model->BSIM4v3xjbvd < 0.0) && (model->BSIM4v3dioMod == 0))
         {   model->BSIM4v3xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v3xjbvd);
         }

         if (model->BSIM4v3bvd <= 0.0)
         {   model->BSIM4v3bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v3bvd);
         }

         if ((model->BSIM4v3xjbvs <= 0.0) && (model->BSIM4v3dioMod == 2))
         {   model->BSIM4v3xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v3xjbvs);
         }
         else if ((model->BSIM4v3xjbvs < 0.0) && (model->BSIM4v3dioMod == 0))
         {   model->BSIM4v3xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v3xjbvs);
         }

         if (model->BSIM4v3bvs <= 0.0)
         {   model->BSIM4v3bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v3bvs);
         }


         /* loop through all the instances of the model */
         for (here = model->BSIM4v3instances; here != NULL;
              here = here->BSIM4v3nextInstance) 
	   {  if (here->BSIM4v3owner != ARCHme) continue;   
	      pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM4v3l == pSizeDependParamKnot->Length)
		      && (here->BSIM4v3w == pSizeDependParamKnot->Width)
		      && (here->BSIM4v3nf == pSizeDependParamKnot->NFinger))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		      pParam = here->pParam; /*bug-fix  */
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      /* stress effect */
	      Ldrn = here->BSIM4v3l;

	      if (Size_Not_Found)
	      {   pParam = (struct bsim4v3SizeDependParam *)malloc(
	                    sizeof(struct bsim4v3SizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v3l;
                  pParam->Width = here->BSIM4v3w;
		  pParam->NFinger = here->BSIM4v3nf;
                  Lnew = here->BSIM4v3l  + model->BSIM4v3xl ;
                  Wnew = here->BSIM4v3w / here->BSIM4v3nf + model->BSIM4v3xw;

                  T0 = pow(Lnew, model->BSIM4v3Lln);
                  T1 = pow(Wnew, model->BSIM4v3Lwn);
                  tmp1 = model->BSIM4v3Ll / T0 + model->BSIM4v3Lw / T1
                       + model->BSIM4v3Lwl / (T0 * T1);
                  pParam->BSIM4v3dl = model->BSIM4v3Lint + tmp1;
                  tmp2 = model->BSIM4v3Llc / T0 + model->BSIM4v3Lwc / T1
                       + model->BSIM4v3Lwlc / (T0 * T1);
                  pParam->BSIM4v3dlc = model->BSIM4v3dlc + tmp2;
                  pParam->BSIM4v3dlcig = model->BSIM4v3dlcig + tmp2;

                  T2 = pow(Lnew, model->BSIM4v3Wln);
                  T3 = pow(Wnew, model->BSIM4v3Wwn);
                  tmp1 = model->BSIM4v3Wl / T2 + model->BSIM4v3Ww / T3
                       + model->BSIM4v3Wwl / (T2 * T3);
                  pParam->BSIM4v3dw = model->BSIM4v3Wint + tmp1;
                  tmp2 = model->BSIM4v3Wlc / T2 + model->BSIM4v3Wwc / T3
                       + model->BSIM4v3Wwlc / (T2 * T3); 
                  pParam->BSIM4v3dwc = model->BSIM4v3dwc + tmp2;
                  pParam->BSIM4v3dwj = model->BSIM4v3dwj + tmp2;

                  pParam->BSIM4v3leff = Lnew - 2.0 * pParam->BSIM4v3dl;
                  if (pParam->BSIM4v3leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v3modName;
                      namarray[1] = here->BSIM4v3name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v3: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v3weff = Wnew - 2.0 * pParam->BSIM4v3dw;
                  if (pParam->BSIM4v3weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v3modName;
                      namarray[1] = here->BSIM4v3name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v3: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v3leffCV = Lnew - 2.0 * pParam->BSIM4v3dlc;
                  if (pParam->BSIM4v3leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v3modName;
                      namarray[1] = here->BSIM4v3name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v3: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v3weffCV = Wnew - 2.0 * pParam->BSIM4v3dwc;
                  if (pParam->BSIM4v3weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v3modName;
                      namarray[1] = here->BSIM4v3name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v3: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v3weffCJ = Wnew - 2.0 * pParam->BSIM4v3dwj;
                  if (pParam->BSIM4v3weffCJ <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v3modName;
                      namarray[1] = here->BSIM4v3name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v3: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM4v3binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM4v3leff;
		      Inv_W = 1.0e-6 / pParam->BSIM4v3weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM4v3leff
			     * pParam->BSIM4v3weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM4v3leff;
		      Inv_W = 1.0 / pParam->BSIM4v3weff;
		      Inv_LW = 1.0 / (pParam->BSIM4v3leff
			     * pParam->BSIM4v3weff);
		  }
		  pParam->BSIM4v3cdsc = model->BSIM4v3cdsc
				    + model->BSIM4v3lcdsc * Inv_L
				    + model->BSIM4v3wcdsc * Inv_W
				    + model->BSIM4v3pcdsc * Inv_LW;
		  pParam->BSIM4v3cdscb = model->BSIM4v3cdscb
				     + model->BSIM4v3lcdscb * Inv_L
				     + model->BSIM4v3wcdscb * Inv_W
				     + model->BSIM4v3pcdscb * Inv_LW; 
				     
    		  pParam->BSIM4v3cdscd = model->BSIM4v3cdscd
				     + model->BSIM4v3lcdscd * Inv_L
				     + model->BSIM4v3wcdscd * Inv_W
				     + model->BSIM4v3pcdscd * Inv_LW; 
				     
		  pParam->BSIM4v3cit = model->BSIM4v3cit
				   + model->BSIM4v3lcit * Inv_L
				   + model->BSIM4v3wcit * Inv_W
				   + model->BSIM4v3pcit * Inv_LW;
		  pParam->BSIM4v3nfactor = model->BSIM4v3nfactor
				       + model->BSIM4v3lnfactor * Inv_L
				       + model->BSIM4v3wnfactor * Inv_W
				       + model->BSIM4v3pnfactor * Inv_LW;
		  pParam->BSIM4v3xj = model->BSIM4v3xj
				  + model->BSIM4v3lxj * Inv_L
				  + model->BSIM4v3wxj * Inv_W
				  + model->BSIM4v3pxj * Inv_LW;
		  pParam->BSIM4v3vsat = model->BSIM4v3vsat
				    + model->BSIM4v3lvsat * Inv_L
				    + model->BSIM4v3wvsat * Inv_W
				    + model->BSIM4v3pvsat * Inv_LW;
		  pParam->BSIM4v3at = model->BSIM4v3at
				  + model->BSIM4v3lat * Inv_L
				  + model->BSIM4v3wat * Inv_W
				  + model->BSIM4v3pat * Inv_LW;
		  pParam->BSIM4v3a0 = model->BSIM4v3a0
				  + model->BSIM4v3la0 * Inv_L
				  + model->BSIM4v3wa0 * Inv_W
				  + model->BSIM4v3pa0 * Inv_LW; 
				  
		  pParam->BSIM4v3ags = model->BSIM4v3ags
				  + model->BSIM4v3lags * Inv_L
				  + model->BSIM4v3wags * Inv_W
				  + model->BSIM4v3pags * Inv_LW;
				  
		  pParam->BSIM4v3a1 = model->BSIM4v3a1
				  + model->BSIM4v3la1 * Inv_L
				  + model->BSIM4v3wa1 * Inv_W
				  + model->BSIM4v3pa1 * Inv_LW;
		  pParam->BSIM4v3a2 = model->BSIM4v3a2
				  + model->BSIM4v3la2 * Inv_L
				  + model->BSIM4v3wa2 * Inv_W
				  + model->BSIM4v3pa2 * Inv_LW;
		  pParam->BSIM4v3keta = model->BSIM4v3keta
				    + model->BSIM4v3lketa * Inv_L
				    + model->BSIM4v3wketa * Inv_W
				    + model->BSIM4v3pketa * Inv_LW;
		  pParam->BSIM4v3nsub = model->BSIM4v3nsub
				    + model->BSIM4v3lnsub * Inv_L
				    + model->BSIM4v3wnsub * Inv_W
				    + model->BSIM4v3pnsub * Inv_LW;
		  pParam->BSIM4v3ndep = model->BSIM4v3ndep
				    + model->BSIM4v3lndep * Inv_L
				    + model->BSIM4v3wndep * Inv_W
				    + model->BSIM4v3pndep * Inv_LW;
                  pParam->BSIM4v3nsd = model->BSIM4v3nsd
                                   + model->BSIM4v3lnsd * Inv_L
                                   + model->BSIM4v3wnsd * Inv_W
                                   + model->BSIM4v3pnsd * Inv_LW;
                  pParam->BSIM4v3phin = model->BSIM4v3phin
                                    + model->BSIM4v3lphin * Inv_L
                                    + model->BSIM4v3wphin * Inv_W
                                    + model->BSIM4v3pphin * Inv_LW;
		  pParam->BSIM4v3ngate = model->BSIM4v3ngate
				     + model->BSIM4v3lngate * Inv_L
				     + model->BSIM4v3wngate * Inv_W
				     + model->BSIM4v3pngate * Inv_LW;
		  pParam->BSIM4v3gamma1 = model->BSIM4v3gamma1
				      + model->BSIM4v3lgamma1 * Inv_L
				      + model->BSIM4v3wgamma1 * Inv_W
				      + model->BSIM4v3pgamma1 * Inv_LW;
		  pParam->BSIM4v3gamma2 = model->BSIM4v3gamma2
				      + model->BSIM4v3lgamma2 * Inv_L
				      + model->BSIM4v3wgamma2 * Inv_W
				      + model->BSIM4v3pgamma2 * Inv_LW;
		  pParam->BSIM4v3vbx = model->BSIM4v3vbx
				   + model->BSIM4v3lvbx * Inv_L
				   + model->BSIM4v3wvbx * Inv_W
				   + model->BSIM4v3pvbx * Inv_LW;
		  pParam->BSIM4v3vbm = model->BSIM4v3vbm
				   + model->BSIM4v3lvbm * Inv_L
				   + model->BSIM4v3wvbm * Inv_W
				   + model->BSIM4v3pvbm * Inv_LW;
		  pParam->BSIM4v3xt = model->BSIM4v3xt
				   + model->BSIM4v3lxt * Inv_L
				   + model->BSIM4v3wxt * Inv_W
				   + model->BSIM4v3pxt * Inv_LW;
                  pParam->BSIM4v3vfb = model->BSIM4v3vfb
                                   + model->BSIM4v3lvfb * Inv_L
                                   + model->BSIM4v3wvfb * Inv_W
                                   + model->BSIM4v3pvfb * Inv_LW;
		  pParam->BSIM4v3k1 = model->BSIM4v3k1
				  + model->BSIM4v3lk1 * Inv_L
				  + model->BSIM4v3wk1 * Inv_W
				  + model->BSIM4v3pk1 * Inv_LW;
		  pParam->BSIM4v3kt1 = model->BSIM4v3kt1
				   + model->BSIM4v3lkt1 * Inv_L
				   + model->BSIM4v3wkt1 * Inv_W
				   + model->BSIM4v3pkt1 * Inv_LW;
		  pParam->BSIM4v3kt1l = model->BSIM4v3kt1l
				    + model->BSIM4v3lkt1l * Inv_L
				    + model->BSIM4v3wkt1l * Inv_W
				    + model->BSIM4v3pkt1l * Inv_LW;
		  pParam->BSIM4v3k2 = model->BSIM4v3k2
				  + model->BSIM4v3lk2 * Inv_L
				  + model->BSIM4v3wk2 * Inv_W
				  + model->BSIM4v3pk2 * Inv_LW;
		  pParam->BSIM4v3kt2 = model->BSIM4v3kt2
				   + model->BSIM4v3lkt2 * Inv_L
				   + model->BSIM4v3wkt2 * Inv_W
				   + model->BSIM4v3pkt2 * Inv_LW;
		  pParam->BSIM4v3k3 = model->BSIM4v3k3
				  + model->BSIM4v3lk3 * Inv_L
				  + model->BSIM4v3wk3 * Inv_W
				  + model->BSIM4v3pk3 * Inv_LW;
		  pParam->BSIM4v3k3b = model->BSIM4v3k3b
				   + model->BSIM4v3lk3b * Inv_L
				   + model->BSIM4v3wk3b * Inv_W
				   + model->BSIM4v3pk3b * Inv_LW;
		  pParam->BSIM4v3w0 = model->BSIM4v3w0
				  + model->BSIM4v3lw0 * Inv_L
				  + model->BSIM4v3ww0 * Inv_W
				  + model->BSIM4v3pw0 * Inv_LW;
		  pParam->BSIM4v3lpe0 = model->BSIM4v3lpe0
				    + model->BSIM4v3llpe0 * Inv_L
 				    + model->BSIM4v3wlpe0 * Inv_W
				    + model->BSIM4v3plpe0 * Inv_LW;
                  pParam->BSIM4v3lpeb = model->BSIM4v3lpeb
                                    + model->BSIM4v3llpeb * Inv_L
                                    + model->BSIM4v3wlpeb * Inv_W
                                    + model->BSIM4v3plpeb * Inv_LW;
                  pParam->BSIM4v3dvtp0 = model->BSIM4v3dvtp0
                                     + model->BSIM4v3ldvtp0 * Inv_L
                                     + model->BSIM4v3wdvtp0 * Inv_W
                                     + model->BSIM4v3pdvtp0 * Inv_LW;
                  pParam->BSIM4v3dvtp1 = model->BSIM4v3dvtp1
                                     + model->BSIM4v3ldvtp1 * Inv_L
                                     + model->BSIM4v3wdvtp1 * Inv_W
                                     + model->BSIM4v3pdvtp1 * Inv_LW;
		  pParam->BSIM4v3dvt0 = model->BSIM4v3dvt0
				    + model->BSIM4v3ldvt0 * Inv_L
				    + model->BSIM4v3wdvt0 * Inv_W
				    + model->BSIM4v3pdvt0 * Inv_LW;
		  pParam->BSIM4v3dvt1 = model->BSIM4v3dvt1
				    + model->BSIM4v3ldvt1 * Inv_L
				    + model->BSIM4v3wdvt1 * Inv_W
				    + model->BSIM4v3pdvt1 * Inv_LW;
		  pParam->BSIM4v3dvt2 = model->BSIM4v3dvt2
				    + model->BSIM4v3ldvt2 * Inv_L
				    + model->BSIM4v3wdvt2 * Inv_W
				    + model->BSIM4v3pdvt2 * Inv_LW;
		  pParam->BSIM4v3dvt0w = model->BSIM4v3dvt0w
				    + model->BSIM4v3ldvt0w * Inv_L
				    + model->BSIM4v3wdvt0w * Inv_W
				    + model->BSIM4v3pdvt0w * Inv_LW;
		  pParam->BSIM4v3dvt1w = model->BSIM4v3dvt1w
				    + model->BSIM4v3ldvt1w * Inv_L
				    + model->BSIM4v3wdvt1w * Inv_W
				    + model->BSIM4v3pdvt1w * Inv_LW;
		  pParam->BSIM4v3dvt2w = model->BSIM4v3dvt2w
				    + model->BSIM4v3ldvt2w * Inv_L
				    + model->BSIM4v3wdvt2w * Inv_W
				    + model->BSIM4v3pdvt2w * Inv_LW;
		  pParam->BSIM4v3drout = model->BSIM4v3drout
				     + model->BSIM4v3ldrout * Inv_L
				     + model->BSIM4v3wdrout * Inv_W
				     + model->BSIM4v3pdrout * Inv_LW;
		  pParam->BSIM4v3dsub = model->BSIM4v3dsub
				    + model->BSIM4v3ldsub * Inv_L
				    + model->BSIM4v3wdsub * Inv_W
				    + model->BSIM4v3pdsub * Inv_LW;
		  pParam->BSIM4v3vth0 = model->BSIM4v3vth0
				    + model->BSIM4v3lvth0 * Inv_L
				    + model->BSIM4v3wvth0 * Inv_W
				    + model->BSIM4v3pvth0 * Inv_LW;
		  pParam->BSIM4v3ua = model->BSIM4v3ua
				  + model->BSIM4v3lua * Inv_L
				  + model->BSIM4v3wua * Inv_W
				  + model->BSIM4v3pua * Inv_LW;
		  pParam->BSIM4v3ua1 = model->BSIM4v3ua1
				   + model->BSIM4v3lua1 * Inv_L
				   + model->BSIM4v3wua1 * Inv_W
				   + model->BSIM4v3pua1 * Inv_LW;
		  pParam->BSIM4v3ub = model->BSIM4v3ub
				  + model->BSIM4v3lub * Inv_L
				  + model->BSIM4v3wub * Inv_W
				  + model->BSIM4v3pub * Inv_LW;
		  pParam->BSIM4v3ub1 = model->BSIM4v3ub1
				   + model->BSIM4v3lub1 * Inv_L
				   + model->BSIM4v3wub1 * Inv_W
				   + model->BSIM4v3pub1 * Inv_LW;
		  pParam->BSIM4v3uc = model->BSIM4v3uc
				  + model->BSIM4v3luc * Inv_L
				  + model->BSIM4v3wuc * Inv_W
				  + model->BSIM4v3puc * Inv_LW;
		  pParam->BSIM4v3uc1 = model->BSIM4v3uc1
				   + model->BSIM4v3luc1 * Inv_L
				   + model->BSIM4v3wuc1 * Inv_W
				   + model->BSIM4v3puc1 * Inv_LW;
                  pParam->BSIM4v3eu = model->BSIM4v3eu
                                  + model->BSIM4v3leu * Inv_L
                                  + model->BSIM4v3weu * Inv_W
                                  + model->BSIM4v3peu * Inv_LW;
		  pParam->BSIM4v3u0 = model->BSIM4v3u0
				  + model->BSIM4v3lu0 * Inv_L
				  + model->BSIM4v3wu0 * Inv_W
				  + model->BSIM4v3pu0 * Inv_LW;
		  pParam->BSIM4v3ute = model->BSIM4v3ute
				   + model->BSIM4v3lute * Inv_L
				   + model->BSIM4v3wute * Inv_W
				   + model->BSIM4v3pute * Inv_LW;
		  pParam->BSIM4v3voff = model->BSIM4v3voff
				    + model->BSIM4v3lvoff * Inv_L
				    + model->BSIM4v3wvoff * Inv_W
				    + model->BSIM4v3pvoff * Inv_LW;
                  pParam->BSIM4v3minv = model->BSIM4v3minv
                                    + model->BSIM4v3lminv * Inv_L
                                    + model->BSIM4v3wminv * Inv_W
                                    + model->BSIM4v3pminv * Inv_LW;
                  pParam->BSIM4v3fprout = model->BSIM4v3fprout
                                     + model->BSIM4v3lfprout * Inv_L
                                     + model->BSIM4v3wfprout * Inv_W
                                     + model->BSIM4v3pfprout * Inv_LW;
                  pParam->BSIM4v3pdits = model->BSIM4v3pdits
                                     + model->BSIM4v3lpdits * Inv_L
                                     + model->BSIM4v3wpdits * Inv_W
                                     + model->BSIM4v3ppdits * Inv_LW;
                  pParam->BSIM4v3pditsd = model->BSIM4v3pditsd
                                      + model->BSIM4v3lpditsd * Inv_L
                                      + model->BSIM4v3wpditsd * Inv_W
                                      + model->BSIM4v3ppditsd * Inv_LW;
		  pParam->BSIM4v3delta = model->BSIM4v3delta
				     + model->BSIM4v3ldelta * Inv_L
				     + model->BSIM4v3wdelta * Inv_W
				     + model->BSIM4v3pdelta * Inv_LW;
		  pParam->BSIM4v3rdsw = model->BSIM4v3rdsw
				    + model->BSIM4v3lrdsw * Inv_L
				    + model->BSIM4v3wrdsw * Inv_W
				    + model->BSIM4v3prdsw * Inv_LW;
                  pParam->BSIM4v3rdw = model->BSIM4v3rdw
                                    + model->BSIM4v3lrdw * Inv_L
                                    + model->BSIM4v3wrdw * Inv_W
                                    + model->BSIM4v3prdw * Inv_LW;
                  pParam->BSIM4v3rsw = model->BSIM4v3rsw
                                    + model->BSIM4v3lrsw * Inv_L
                                    + model->BSIM4v3wrsw * Inv_W
                                    + model->BSIM4v3prsw * Inv_LW;
		  pParam->BSIM4v3prwg = model->BSIM4v3prwg
				    + model->BSIM4v3lprwg * Inv_L
				    + model->BSIM4v3wprwg * Inv_W
				    + model->BSIM4v3pprwg * Inv_LW;
		  pParam->BSIM4v3prwb = model->BSIM4v3prwb
				    + model->BSIM4v3lprwb * Inv_L
				    + model->BSIM4v3wprwb * Inv_W
				    + model->BSIM4v3pprwb * Inv_LW;
		  pParam->BSIM4v3prt = model->BSIM4v3prt
				    + model->BSIM4v3lprt * Inv_L
				    + model->BSIM4v3wprt * Inv_W
				    + model->BSIM4v3pprt * Inv_LW;
		  pParam->BSIM4v3eta0 = model->BSIM4v3eta0
				    + model->BSIM4v3leta0 * Inv_L
				    + model->BSIM4v3weta0 * Inv_W
				    + model->BSIM4v3peta0 * Inv_LW;
		  pParam->BSIM4v3etab = model->BSIM4v3etab
				    + model->BSIM4v3letab * Inv_L
				    + model->BSIM4v3wetab * Inv_W
				    + model->BSIM4v3petab * Inv_LW;
		  pParam->BSIM4v3pclm = model->BSIM4v3pclm
				    + model->BSIM4v3lpclm * Inv_L
				    + model->BSIM4v3wpclm * Inv_W
				    + model->BSIM4v3ppclm * Inv_LW;
		  pParam->BSIM4v3pdibl1 = model->BSIM4v3pdibl1
				      + model->BSIM4v3lpdibl1 * Inv_L
				      + model->BSIM4v3wpdibl1 * Inv_W
				      + model->BSIM4v3ppdibl1 * Inv_LW;
		  pParam->BSIM4v3pdibl2 = model->BSIM4v3pdibl2
				      + model->BSIM4v3lpdibl2 * Inv_L
				      + model->BSIM4v3wpdibl2 * Inv_W
				      + model->BSIM4v3ppdibl2 * Inv_LW;
		  pParam->BSIM4v3pdiblb = model->BSIM4v3pdiblb
				      + model->BSIM4v3lpdiblb * Inv_L
				      + model->BSIM4v3wpdiblb * Inv_W
				      + model->BSIM4v3ppdiblb * Inv_LW;
		  pParam->BSIM4v3pscbe1 = model->BSIM4v3pscbe1
				      + model->BSIM4v3lpscbe1 * Inv_L
				      + model->BSIM4v3wpscbe1 * Inv_W
				      + model->BSIM4v3ppscbe1 * Inv_LW;
		  pParam->BSIM4v3pscbe2 = model->BSIM4v3pscbe2
				      + model->BSIM4v3lpscbe2 * Inv_L
				      + model->BSIM4v3wpscbe2 * Inv_W
				      + model->BSIM4v3ppscbe2 * Inv_LW;
		  pParam->BSIM4v3pvag = model->BSIM4v3pvag
				    + model->BSIM4v3lpvag * Inv_L
				    + model->BSIM4v3wpvag * Inv_W
				    + model->BSIM4v3ppvag * Inv_LW;
		  pParam->BSIM4v3wr = model->BSIM4v3wr
				  + model->BSIM4v3lwr * Inv_L
				  + model->BSIM4v3wwr * Inv_W
				  + model->BSIM4v3pwr * Inv_LW;
		  pParam->BSIM4v3dwg = model->BSIM4v3dwg
				   + model->BSIM4v3ldwg * Inv_L
				   + model->BSIM4v3wdwg * Inv_W
				   + model->BSIM4v3pdwg * Inv_LW;
		  pParam->BSIM4v3dwb = model->BSIM4v3dwb
				   + model->BSIM4v3ldwb * Inv_L
				   + model->BSIM4v3wdwb * Inv_W
				   + model->BSIM4v3pdwb * Inv_LW;
		  pParam->BSIM4v3b0 = model->BSIM4v3b0
				  + model->BSIM4v3lb0 * Inv_L
				  + model->BSIM4v3wb0 * Inv_W
				  + model->BSIM4v3pb0 * Inv_LW;
		  pParam->BSIM4v3b1 = model->BSIM4v3b1
				  + model->BSIM4v3lb1 * Inv_L
				  + model->BSIM4v3wb1 * Inv_W
				  + model->BSIM4v3pb1 * Inv_LW;
		  pParam->BSIM4v3alpha0 = model->BSIM4v3alpha0
				      + model->BSIM4v3lalpha0 * Inv_L
				      + model->BSIM4v3walpha0 * Inv_W
				      + model->BSIM4v3palpha0 * Inv_LW;
                  pParam->BSIM4v3alpha1 = model->BSIM4v3alpha1
                                      + model->BSIM4v3lalpha1 * Inv_L
                                      + model->BSIM4v3walpha1 * Inv_W
                                      + model->BSIM4v3palpha1 * Inv_LW;
		  pParam->BSIM4v3beta0 = model->BSIM4v3beta0
				     + model->BSIM4v3lbeta0 * Inv_L
				     + model->BSIM4v3wbeta0 * Inv_W
				     + model->BSIM4v3pbeta0 * Inv_LW;
                  pParam->BSIM4v3agidl = model->BSIM4v3agidl
                                     + model->BSIM4v3lagidl * Inv_L
                                     + model->BSIM4v3wagidl * Inv_W
                                     + model->BSIM4v3pagidl * Inv_LW;
                  pParam->BSIM4v3bgidl = model->BSIM4v3bgidl
                                     + model->BSIM4v3lbgidl * Inv_L
                                     + model->BSIM4v3wbgidl * Inv_W
                                     + model->BSIM4v3pbgidl * Inv_LW;
                  pParam->BSIM4v3cgidl = model->BSIM4v3cgidl
                                     + model->BSIM4v3lcgidl * Inv_L
                                     + model->BSIM4v3wcgidl * Inv_W
                                     + model->BSIM4v3pcgidl * Inv_LW;
                  pParam->BSIM4v3egidl = model->BSIM4v3egidl
                                     + model->BSIM4v3legidl * Inv_L
                                     + model->BSIM4v3wegidl * Inv_W
                                     + model->BSIM4v3pegidl * Inv_LW;
                  pParam->BSIM4v3aigc = model->BSIM4v3aigc
                                     + model->BSIM4v3laigc * Inv_L
                                     + model->BSIM4v3waigc * Inv_W
                                     + model->BSIM4v3paigc * Inv_LW;
                  pParam->BSIM4v3bigc = model->BSIM4v3bigc
                                     + model->BSIM4v3lbigc * Inv_L
                                     + model->BSIM4v3wbigc * Inv_W
                                     + model->BSIM4v3pbigc * Inv_LW;
                  pParam->BSIM4v3cigc = model->BSIM4v3cigc
                                     + model->BSIM4v3lcigc * Inv_L
                                     + model->BSIM4v3wcigc * Inv_W
                                     + model->BSIM4v3pcigc * Inv_LW;
                  pParam->BSIM4v3aigsd = model->BSIM4v3aigsd
                                     + model->BSIM4v3laigsd * Inv_L
                                     + model->BSIM4v3waigsd * Inv_W
                                     + model->BSIM4v3paigsd * Inv_LW;
                  pParam->BSIM4v3bigsd = model->BSIM4v3bigsd
                                     + model->BSIM4v3lbigsd * Inv_L
                                     + model->BSIM4v3wbigsd * Inv_W
                                     + model->BSIM4v3pbigsd * Inv_LW;
                  pParam->BSIM4v3cigsd = model->BSIM4v3cigsd
                                     + model->BSIM4v3lcigsd * Inv_L
                                     + model->BSIM4v3wcigsd * Inv_W
                                     + model->BSIM4v3pcigsd * Inv_LW;
                  pParam->BSIM4v3aigbacc = model->BSIM4v3aigbacc
                                       + model->BSIM4v3laigbacc * Inv_L
                                       + model->BSIM4v3waigbacc * Inv_W
                                       + model->BSIM4v3paigbacc * Inv_LW;
                  pParam->BSIM4v3bigbacc = model->BSIM4v3bigbacc
                                       + model->BSIM4v3lbigbacc * Inv_L
                                       + model->BSIM4v3wbigbacc * Inv_W
                                       + model->BSIM4v3pbigbacc * Inv_LW;
                  pParam->BSIM4v3cigbacc = model->BSIM4v3cigbacc
                                       + model->BSIM4v3lcigbacc * Inv_L
                                       + model->BSIM4v3wcigbacc * Inv_W
                                       + model->BSIM4v3pcigbacc * Inv_LW;
                  pParam->BSIM4v3aigbinv = model->BSIM4v3aigbinv
                                       + model->BSIM4v3laigbinv * Inv_L
                                       + model->BSIM4v3waigbinv * Inv_W
                                       + model->BSIM4v3paigbinv * Inv_LW;
                  pParam->BSIM4v3bigbinv = model->BSIM4v3bigbinv
                                       + model->BSIM4v3lbigbinv * Inv_L
                                       + model->BSIM4v3wbigbinv * Inv_W
                                       + model->BSIM4v3pbigbinv * Inv_LW;
                  pParam->BSIM4v3cigbinv = model->BSIM4v3cigbinv
                                       + model->BSIM4v3lcigbinv * Inv_L
                                       + model->BSIM4v3wcigbinv * Inv_W
                                       + model->BSIM4v3pcigbinv * Inv_LW;
                  pParam->BSIM4v3nigc = model->BSIM4v3nigc
                                       + model->BSIM4v3lnigc * Inv_L
                                       + model->BSIM4v3wnigc * Inv_W
                                       + model->BSIM4v3pnigc * Inv_LW;
                  pParam->BSIM4v3nigbacc = model->BSIM4v3nigbacc
                                       + model->BSIM4v3lnigbacc * Inv_L
                                       + model->BSIM4v3wnigbacc * Inv_W
                                       + model->BSIM4v3pnigbacc * Inv_LW;
                  pParam->BSIM4v3nigbinv = model->BSIM4v3nigbinv
                                       + model->BSIM4v3lnigbinv * Inv_L
                                       + model->BSIM4v3wnigbinv * Inv_W
                                       + model->BSIM4v3pnigbinv * Inv_LW;
                  pParam->BSIM4v3ntox = model->BSIM4v3ntox
                                    + model->BSIM4v3lntox * Inv_L
                                    + model->BSIM4v3wntox * Inv_W
                                    + model->BSIM4v3pntox * Inv_LW;
                  pParam->BSIM4v3eigbinv = model->BSIM4v3eigbinv
                                       + model->BSIM4v3leigbinv * Inv_L
                                       + model->BSIM4v3weigbinv * Inv_W
                                       + model->BSIM4v3peigbinv * Inv_LW;
                  pParam->BSIM4v3pigcd = model->BSIM4v3pigcd
                                     + model->BSIM4v3lpigcd * Inv_L
                                     + model->BSIM4v3wpigcd * Inv_W
                                     + model->BSIM4v3ppigcd * Inv_LW;
                  pParam->BSIM4v3poxedge = model->BSIM4v3poxedge
                                       + model->BSIM4v3lpoxedge * Inv_L
                                       + model->BSIM4v3wpoxedge * Inv_W
                                       + model->BSIM4v3ppoxedge * Inv_LW;
                  pParam->BSIM4v3xrcrg1 = model->BSIM4v3xrcrg1
                                      + model->BSIM4v3lxrcrg1 * Inv_L
                                      + model->BSIM4v3wxrcrg1 * Inv_W
                                      + model->BSIM4v3pxrcrg1 * Inv_LW;
                  pParam->BSIM4v3xrcrg2 = model->BSIM4v3xrcrg2
                                      + model->BSIM4v3lxrcrg2 * Inv_L
                                      + model->BSIM4v3wxrcrg2 * Inv_W
                                      + model->BSIM4v3pxrcrg2 * Inv_LW;
                  pParam->BSIM4v3lambda = model->BSIM4v3lambda
                                      + model->BSIM4v3llambda * Inv_L
                                      + model->BSIM4v3wlambda * Inv_W
                                      + model->BSIM4v3plambda * Inv_LW;
                  pParam->BSIM4v3vtl = model->BSIM4v3vtl
                                      + model->BSIM4v3lvtl * Inv_L
                                      + model->BSIM4v3wvtl * Inv_W
                                      + model->BSIM4v3pvtl * Inv_LW;
                  pParam->BSIM4v3xn = model->BSIM4v3xn
                                      + model->BSIM4v3lxn * Inv_L
                                      + model->BSIM4v3wxn * Inv_W
                                      + model->BSIM4v3pxn * Inv_LW;

		  pParam->BSIM4v3cgsl = model->BSIM4v3cgsl
				    + model->BSIM4v3lcgsl * Inv_L
				    + model->BSIM4v3wcgsl * Inv_W
				    + model->BSIM4v3pcgsl * Inv_LW;
		  pParam->BSIM4v3cgdl = model->BSIM4v3cgdl
				    + model->BSIM4v3lcgdl * Inv_L
				    + model->BSIM4v3wcgdl * Inv_W
				    + model->BSIM4v3pcgdl * Inv_LW;
		  pParam->BSIM4v3ckappas = model->BSIM4v3ckappas
				       + model->BSIM4v3lckappas * Inv_L
				       + model->BSIM4v3wckappas * Inv_W
 				       + model->BSIM4v3pckappas * Inv_LW;
                  pParam->BSIM4v3ckappad = model->BSIM4v3ckappad
                                       + model->BSIM4v3lckappad * Inv_L
                                       + model->BSIM4v3wckappad * Inv_W
                                       + model->BSIM4v3pckappad * Inv_LW;
		  pParam->BSIM4v3cf = model->BSIM4v3cf
				  + model->BSIM4v3lcf * Inv_L
				  + model->BSIM4v3wcf * Inv_W
				  + model->BSIM4v3pcf * Inv_LW;
		  pParam->BSIM4v3clc = model->BSIM4v3clc
				   + model->BSIM4v3lclc * Inv_L
				   + model->BSIM4v3wclc * Inv_W
				   + model->BSIM4v3pclc * Inv_LW;
		  pParam->BSIM4v3cle = model->BSIM4v3cle
				   + model->BSIM4v3lcle * Inv_L
				   + model->BSIM4v3wcle * Inv_W
				   + model->BSIM4v3pcle * Inv_LW;
		  pParam->BSIM4v3vfbcv = model->BSIM4v3vfbcv
				     + model->BSIM4v3lvfbcv * Inv_L
				     + model->BSIM4v3wvfbcv * Inv_W
				     + model->BSIM4v3pvfbcv * Inv_LW;
                  pParam->BSIM4v3acde = model->BSIM4v3acde
                                    + model->BSIM4v3lacde * Inv_L
                                    + model->BSIM4v3wacde * Inv_W
                                    + model->BSIM4v3pacde * Inv_LW;
                  pParam->BSIM4v3moin = model->BSIM4v3moin
                                    + model->BSIM4v3lmoin * Inv_L
                                    + model->BSIM4v3wmoin * Inv_W
                                    + model->BSIM4v3pmoin * Inv_LW;
                  pParam->BSIM4v3noff = model->BSIM4v3noff
                                    + model->BSIM4v3lnoff * Inv_L
                                    + model->BSIM4v3wnoff * Inv_W
                                    + model->BSIM4v3pnoff * Inv_LW;
                  pParam->BSIM4v3voffcv = model->BSIM4v3voffcv
                                      + model->BSIM4v3lvoffcv * Inv_L
                                      + model->BSIM4v3wvoffcv * Inv_W
                                      + model->BSIM4v3pvoffcv * Inv_LW;

                  pParam->BSIM4v3abulkCVfactor = 1.0 + pow((pParam->BSIM4v3clc
					     / pParam->BSIM4v3leffCV),
					     pParam->BSIM4v3cle);

	          T0 = (TRatio - 1.0);

		  PowWeffWr = pow(pParam->BSIM4v3weffCJ * 1.0e6, pParam->BSIM4v3wr) * here->BSIM4v3nf;

	          T1 = T2 = T3 = T4 = 0.0;
	          if(model->BSIM4v3tempMod == 0) {
	          	pParam->BSIM4v3ua = pParam->BSIM4v3ua + pParam->BSIM4v3ua1 * T0;
	          	pParam->BSIM4v3ub = pParam->BSIM4v3ub + pParam->BSIM4v3ub1 * T0;
	          	pParam->BSIM4v3uc = pParam->BSIM4v3uc + pParam->BSIM4v3uc1 * T0;
                  	pParam->BSIM4v3vsattemp = pParam->BSIM4v3vsat - pParam->BSIM4v3at * T0;
		  	T10 = pParam->BSIM4v3prt * T0;
		     if(model->BSIM4v3rdsMod) {
		  	/* External Rd(V) */
		  	T1 = pParam->BSIM4v3rdw + T10;
                  	T2 = model->BSIM4v3rdwmin + T10;
		  	/* External Rs(V) */
		  	T3 = pParam->BSIM4v3rsw + T10;
                  	T4 = model->BSIM4v3rswmin + T10;
                     }
		  	/* Internal Rds(V) in IV */
	          	pParam->BSIM4v3rds0 = (pParam->BSIM4v3rdsw + T10)
				    	* here->BSIM4v3nf / PowWeffWr;
		  	pParam->BSIM4v3rdswmin = (model->BSIM4v3rdswmin + T10)
				       	* here->BSIM4v3nf / PowWeffWr;
                  } else { /* tempMod = 1 */
	          	pParam->BSIM4v3ua = pParam->BSIM4v3ua * (1.0 + pParam->BSIM4v3ua1 * delTemp) ;
	          	pParam->BSIM4v3ub = pParam->BSIM4v3ub * (1.0 + pParam->BSIM4v3ub1 * delTemp);
	          	pParam->BSIM4v3uc = pParam->BSIM4v3uc * (1.0 + pParam->BSIM4v3uc1 * delTemp);
                  	pParam->BSIM4v3vsattemp = pParam->BSIM4v3vsat * (1.0 - pParam->BSIM4v3at * delTemp);
		  	T10 = 1.0 + pParam->BSIM4v3prt * delTemp;
		     if(model->BSIM4v3rdsMod) {
		  	/* External Rd(V) */
		  	T1 = pParam->BSIM4v3rdw * T10;
                  	T2 = model->BSIM4v3rdwmin * T10;
		  	/* External Rs(V) */
		  	T3 = pParam->BSIM4v3rsw * T10;
                  	T4 = model->BSIM4v3rswmin * T10;
                     }
		  	/* Internal Rds(V) in IV */
	          	pParam->BSIM4v3rds0 = pParam->BSIM4v3rdsw * T10 * here->BSIM4v3nf / PowWeffWr;
		  	pParam->BSIM4v3rdswmin = model->BSIM4v3rdswmin * T10 * here->BSIM4v3nf / PowWeffWr;
                  }
		  if (T1 < 0.0)
		  {   T1 = 0.0;
		      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
		  }
		  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
		  pParam->BSIM4v3rd0 = T1 / PowWeffWr;
                  pParam->BSIM4v3rdwmin = T2 / PowWeffWr;
                  if (T3 < 0.0)
                  {   T3 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  if (T4 < 0.0)
                  {   T4 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v3rs0 = T3 / PowWeffWr;
                  pParam->BSIM4v3rswmin = T4 / PowWeffWr;

                  if (pParam->BSIM4v3u0 > 1.0) 
                      pParam->BSIM4v3u0 = pParam->BSIM4v3u0 / 1.0e4;

                  pParam->BSIM4v3u0temp = pParam->BSIM4v3u0
				      * pow(TRatio, pParam->BSIM4v3ute); 
                  if (pParam->BSIM4v3eu < 0.0)
                  {   pParam->BSIM4v3eu = 0.0;
		      printf("Warning: eu has been negative; reset to 0.0.\n");
		  }

                /* Source End Velocity Limit  */
      	          if((model->BSIM4v3vtlGiven) && (model->BSIM4v3vtl > 0.0) )
            	  {  
                     if(model->BSIM4v3lc < 0.0) pParam->BSIM4v3lc = 0.0;
                     else   pParam->BSIM4v3lc = model->BSIM4v3lc ;
                     T0 = pParam->BSIM4v3leff / (pParam->BSIM4v3xn * pParam->BSIM4v3leff + pParam->BSIM4v3lc);
                     pParam->BSIM4v3tfactor = (1.0 - T0) / (1.0 + T0 );
             	  }

                  pParam->BSIM4v3cgdo = (model->BSIM4v3cgdo + pParam->BSIM4v3cf)
				    * pParam->BSIM4v3weffCV;
                  pParam->BSIM4v3cgso = (model->BSIM4v3cgso + pParam->BSIM4v3cf)
				    * pParam->BSIM4v3weffCV;
                  pParam->BSIM4v3cgbo = model->BSIM4v3cgbo * pParam->BSIM4v3leffCV * here->BSIM4v3nf;

                  if (!model->BSIM4v3ndepGiven && model->BSIM4v3gamma1Given)
                  {   T0 = pParam->BSIM4v3gamma1 * model->BSIM4v3coxe;
                      pParam->BSIM4v3ndep = 3.01248e22 * T0 * T0;
                  }

		  pParam->BSIM4v3phi = Vtm0 * log(pParam->BSIM4v3ndep / ni)
				   + pParam->BSIM4v3phin + 0.4;

	          pParam->BSIM4v3sqrtPhi = sqrt(pParam->BSIM4v3phi);
	          pParam->BSIM4v3phis3 = pParam->BSIM4v3sqrtPhi * pParam->BSIM4v3phi;

                  pParam->BSIM4v3Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM4v3ndep * 1.0e6))
                                     * pParam->BSIM4v3sqrtPhi; 
                  pParam->BSIM4v3sqrtXdep0 = sqrt(pParam->BSIM4v3Xdep0);
                  pParam->BSIM4v3litl = sqrt(3.0 * pParam->BSIM4v3xj
				    * model->BSIM4v3toxe);
                  pParam->BSIM4v3vbi = Vtm0 * log(pParam->BSIM4v3nsd
			           * pParam->BSIM4v3ndep / (ni * ni));

		  if (pParam->BSIM4v3ngate > 0.0)
                  {   pParam->BSIM4v3vfbsd = Vtm0 * log(pParam->BSIM4v3ngate
                                         / pParam->BSIM4v3nsd);
		  }
		  else
		      pParam->BSIM4v3vfbsd = 0.0;

                  pParam->BSIM4v3cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM4v3ndep * 1.0e6 / 2.0
				     / pParam->BSIM4v3phi);

                  pParam->BSIM4v3ToxRatio = exp(pParam->BSIM4v3ntox
					* log(model->BSIM4v3toxref / model->BSIM4v3toxe))
					/ model->BSIM4v3toxe / model->BSIM4v3toxe;
                  pParam->BSIM4v3ToxRatioEdge = exp(pParam->BSIM4v3ntox
                                            * log(model->BSIM4v3toxref
                                            / (model->BSIM4v3toxe * pParam->BSIM4v3poxedge)))
                                            / model->BSIM4v3toxe / model->BSIM4v3toxe
                                            / pParam->BSIM4v3poxedge / pParam->BSIM4v3poxedge;
                  pParam->BSIM4v3Aechvb = (model->BSIM4v3type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v3Bechvb = (model->BSIM4v3type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v3AechvbEdge = pParam->BSIM4v3Aechvb * pParam->BSIM4v3weff
					  * pParam->BSIM4v3dlcig * pParam->BSIM4v3ToxRatioEdge;
                  pParam->BSIM4v3BechvbEdge = -pParam->BSIM4v3Bechvb
					  * model->BSIM4v3toxe * pParam->BSIM4v3poxedge;
                  pParam->BSIM4v3Aechvb *= pParam->BSIM4v3weff * pParam->BSIM4v3leff
				       * pParam->BSIM4v3ToxRatio;
                  pParam->BSIM4v3Bechvb *= -model->BSIM4v3toxe;


                  pParam->BSIM4v3mstar = 0.5 + atan(pParam->BSIM4v3minv) / PI;
                  pParam->BSIM4v3voffcbn =  pParam->BSIM4v3voff + model->BSIM4v3voffl / pParam->BSIM4v3leff;

                  pParam->BSIM4v3ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4v3ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v3acde *= pow((pParam->BSIM4v3ndep / 2.0e16), -0.25);


                  if (model->BSIM4v3k1Given || model->BSIM4v3k2Given)
	          {   if (!model->BSIM4v3k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v3k1 = 0.53;
                      }
                      if (!model->BSIM4v3k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v3k2 = -0.0186;
                      }
                      if (model->BSIM4v3nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v3xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v3vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v3gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v3gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM4v3vbxGiven)
                          pParam->BSIM4v3vbx = pParam->BSIM4v3phi - 7.7348e-4 
                                           * pParam->BSIM4v3ndep
					   * pParam->BSIM4v3xt * pParam->BSIM4v3xt;
	              if (pParam->BSIM4v3vbx > 0.0)
		          pParam->BSIM4v3vbx = -pParam->BSIM4v3vbx;
	              if (pParam->BSIM4v3vbm > 0.0)
                          pParam->BSIM4v3vbm = -pParam->BSIM4v3vbm;
           
                      if (!model->BSIM4v3gamma1Given)
                          pParam->BSIM4v3gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM4v3ndep)
                                              / model->BSIM4v3coxe;
                      if (!model->BSIM4v3gamma2Given)
                          pParam->BSIM4v3gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM4v3nsub)
                                              / model->BSIM4v3coxe;

                      T0 = pParam->BSIM4v3gamma1 - pParam->BSIM4v3gamma2;
                      T1 = sqrt(pParam->BSIM4v3phi - pParam->BSIM4v3vbx)
			 - pParam->BSIM4v3sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v3phi * (pParam->BSIM4v3phi
			 - pParam->BSIM4v3vbm)) - pParam->BSIM4v3phi;
                      pParam->BSIM4v3k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v3vbm);
                      pParam->BSIM4v3k1 = pParam->BSIM4v3gamma2 - 2.0
				      * pParam->BSIM4v3k2 * sqrt(pParam->BSIM4v3phi
				      - pParam->BSIM4v3vbm);
                  }
 
		  if (pParam->BSIM4v3k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM4v3k1 / pParam->BSIM4v3k2;
                      pParam->BSIM4v3vbsc = 0.9 * (pParam->BSIM4v3phi - T0 * T0);
		      if (pParam->BSIM4v3vbsc > -3.0)
		          pParam->BSIM4v3vbsc = -3.0;
		      else if (pParam->BSIM4v3vbsc < -30.0)
		          pParam->BSIM4v3vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM4v3vbsc = -30.0;
		  }
		  if (pParam->BSIM4v3vbsc > pParam->BSIM4v3vbm)
		      pParam->BSIM4v3vbsc = pParam->BSIM4v3vbm;

                  if (!model->BSIM4v3vfbGiven)
                  {   if (model->BSIM4v3vth0Given)
                      {   pParam->BSIM4v3vfb = model->BSIM4v3type * pParam->BSIM4v3vth0
                                           - pParam->BSIM4v3phi - pParam->BSIM4v3k1
                                           * pParam->BSIM4v3sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4v3vfb = -1.0;
                      }
                  }
                   if (!model->BSIM4v3vth0Given)
                  {   pParam->BSIM4v3vth0 = model->BSIM4v3type * (pParam->BSIM4v3vfb
                                        + pParam->BSIM4v3phi + pParam->BSIM4v3k1
                                        * pParam->BSIM4v3sqrtPhi);
                  }

                  pParam->BSIM4v3k1ox = pParam->BSIM4v3k1 * model->BSIM4v3toxe
                                    / model->BSIM4v3toxm;
                  pParam->BSIM4v3k2ox = pParam->BSIM4v3k2 * model->BSIM4v3toxe
                                    / model->BSIM4v3toxm;

		  T3 = model->BSIM4v3type * pParam->BSIM4v3vth0
		     - pParam->BSIM4v3vfb - pParam->BSIM4v3phi;
		  T4 = T3 + T3;
		  T5 = 2.5 * T3;
                  pParam->BSIM4v3vtfbphi1 = (model->BSIM4v3type == NMOS) ? T4 : T5; 
		  if (pParam->BSIM4v3vtfbphi1 < 0.0)
		      pParam->BSIM4v3vtfbphi1 = 0.0;

                  pParam->BSIM4v3vtfbphi2 = 4.0 * T3;
                  if (pParam->BSIM4v3vtfbphi2 < 0.0)
                      pParam->BSIM4v3vtfbphi2 = 0.0;

                  tmp = sqrt(EPSSI / (model->BSIM4v3epsrox * EPS0)
                      * model->BSIM4v3toxe * pParam->BSIM4v3Xdep0);
          	  T0 = pParam->BSIM4v3dsub * pParam->BSIM4v3leff / tmp;
                  if (T0 < EXP_THRESHOLD)
          	  {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
              	      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v3theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v3theta0vb0 = 1.0 / (MAX_EXP - 2.0);

 	          T0 = pParam->BSIM4v3drout * pParam->BSIM4v3leff / tmp;
        	  if (T0 < EXP_THRESHOLD)
       	          {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v3thetaRout = pParam->BSIM4v3pdibl1 * T5
                                         + pParam->BSIM4v3pdibl2;

                  tmp = sqrt(pParam->BSIM4v3Xdep0);
                  tmp1 = pParam->BSIM4v3vbi - pParam->BSIM4v3phi;
                  tmp2 = model->BSIM4v3factor1 * tmp;

                  T0 = pParam->BSIM4v3dvt1w * pParam->BSIM4v3weff
                     * pParam->BSIM4v3leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v3dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v3dvt1 * pParam->BSIM4v3leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  } 
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v3dvt0 * T9 * tmp1;

                  T4 = model->BSIM4v3toxe * pParam->BSIM4v3phi
                     / (pParam->BSIM4v3weff + pParam->BSIM4v3w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v3lpe0 / pParam->BSIM4v3leff);
                  T5 = pParam->BSIM4v3k1ox * (T0 - 1.0) * pParam->BSIM4v3sqrtPhi
                     + (pParam->BSIM4v3kt1 + pParam->BSIM4v3kt1l / pParam->BSIM4v3leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM4v3type * pParam->BSIM4v3vth0
                       - T8 - T9 + pParam->BSIM4v3k3 * T4 + T5;
                  pParam->BSIM4v3vfbzb = tmp3 - pParam->BSIM4v3phi - pParam->BSIM4v3k1
                                     * pParam->BSIM4v3sqrtPhi; /* End of vfbzb */

		  /* stress effect */
                  T0 = pow(Lnew, model->BSIM4v3llodku0);
		  W_tmp = Wnew + model->BSIM4v3wlod;
                  T1 = pow(W_tmp, model->BSIM4v3wlodku0);
                  tmp1 = model->BSIM4v3lku0 / T0 + model->BSIM4v3wku0 / T1
                         + model->BSIM4v3pku0 / (T0 * T1);
                  pParam->BSIM4v3ku0 = 1.0 + tmp1;

                  T0 = pow(Lnew, model->BSIM4v3llodvth);
                  T1 = pow(W_tmp, model->BSIM4v3wlodvth);
                  tmp1 = model->BSIM4v3lkvth0 / T0 + model->BSIM4v3wkvth0 / T1
                       + model->BSIM4v3pkvth0 / (T0 * T1);
                  pParam->BSIM4v3kvth0 = 1.0 + tmp1;
		  pParam->BSIM4v3kvth0 = sqrt(pParam->BSIM4v3kvth0*pParam->BSIM4v3kvth0 + DELTA);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM4v3ku0temp = pParam->BSIM4v3ku0 * (1.0 + model->BSIM4v3tku0 *T0) + DELTA;

                  Inv_saref = 1.0/(model->BSIM4v3saref + 0.5*Ldrn);
                  Inv_sbref = 1.0/(model->BSIM4v3sbref + 0.5*Ldrn);
		  pParam->BSIM4v3inv_od_ref = Inv_saref + Inv_sbref;
		  pParam->BSIM4v3rho_ref = model->BSIM4v3ku0 / pParam->BSIM4v3ku0temp * pParam->BSIM4v3inv_od_ref;

              } /* End of SizeNotFound */

              /*  stress effect */
              if( (here->BSIM4v3sa > 0.0) && (here->BSIM4v3sb > 0.0) && 
               	 ((here->BSIM4v3nf == 1.0) || ((here->BSIM4v3nf > 1.0) && (here->BSIM4v3sd > 0.0))) )
	      {	  Inv_sa = 0;
              	  Inv_sb = 0;
	      	  for(i = 0; i < here->BSIM4v3nf; i++){
                   	T0 = 1.0 / here->BSIM4v3nf / (here->BSIM4v3sa + 0.5*Ldrn + i * (here->BSIM4v3sd +Ldrn));
                    	T1 = 1.0 / here->BSIM4v3nf / (here->BSIM4v3sb + 0.5*Ldrn + i * (here->BSIM4v3sd +Ldrn));
                   	Inv_sa += T0;
                    	Inv_sb += T1;
                  }
                  Inv_ODeff = Inv_sa + Inv_sb; 
                  rho = model->BSIM4v3ku0 / pParam->BSIM4v3ku0temp * Inv_ODeff;
                  T0 = (1.0 + rho)/(1.0 + pParam->BSIM4v3rho_ref);
                  here->BSIM4v3u0temp = pParam->BSIM4v3u0temp * T0;

                  T1 = (1.0 + model->BSIM4v3kvsat * rho)/(1.0 + model->BSIM4v3kvsat * pParam->BSIM4v3rho_ref);
                  here->BSIM4v3vsattemp = pParam->BSIM4v3vsattemp * T1;

		  OD_offset = Inv_ODeff - pParam->BSIM4v3inv_od_ref;
		  dvth0_lod = model->BSIM4v3kvth0 / pParam->BSIM4v3kvth0 * OD_offset;
                  dk2_lod = model->BSIM4v3stk2 / pow(pParam->BSIM4v3kvth0, model->BSIM4v3lodk2) *
                                   OD_offset;
                  deta0_lod = model->BSIM4v3steta0 / pow(pParam->BSIM4v3kvth0, model->BSIM4v3lodeta0) *
                                     OD_offset;
		  here->BSIM4v3vth0 = pParam->BSIM4v3vth0 + dvth0_lod;

	          if (!model->BSIM4v3vfbGiven && !model->BSIM4v3vth0Given)
                       here->BSIM4v3vfb = -1.0;
                  else  
                       here->BSIM4v3vfb = pParam->BSIM4v3vfb + model->BSIM4v3type * dvth0_lod;
                  here->BSIM4v3vfbzb = pParam->BSIM4v3vfbzb + model->BSIM4v3type * dvth0_lod;

                  T3 = model->BSIM4v3type * here->BSIM4v3vth0
                     - here->BSIM4v3vfb - pParam->BSIM4v3phi;
                  T4 = T3 + T3;
                  T5 = 2.5 * T3;
                  here->BSIM4v3vtfbphi1 = (model->BSIM4v3type == NMOS) ? T4 : T5;
                  if (here->BSIM4v3vtfbphi1 < 0.0)
                      here->BSIM4v3vtfbphi1 = 0.0;

                  here->BSIM4v3vtfbphi2 = 4.0 * T3;
                  if (here->BSIM4v3vtfbphi2 < 0.0)
                      here->BSIM4v3vtfbphi2 = 0.0;
		  
		  here->BSIM4v3k2 = pParam->BSIM4v3k2 + dk2_lod;
                  if (here->BSIM4v3k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM4v3k1 / here->BSIM4v3k2;
                      here->BSIM4v3vbsc = 0.9 * (pParam->BSIM4v3phi - T0 * T0);
                      if (here->BSIM4v3vbsc > -3.0)
                          here->BSIM4v3vbsc = -3.0;
                      else if (here->BSIM4v3vbsc < -30.0)
                          here->BSIM4v3vbsc = -30.0;
                  }
                  else
                      here->BSIM4v3vbsc = -30.0;
                  if (here->BSIM4v3vbsc > pParam->BSIM4v3vbm)
                      here->BSIM4v3vbsc = pParam->BSIM4v3vbm;
		  here->BSIM4v3k2ox = here->BSIM4v3k2 * model->BSIM4v3toxe
                                    / model->BSIM4v3toxm;

                  here->BSIM4v3eta0 = pParam->BSIM4v3eta0 + deta0_lod;
	       } else {
		      here->BSIM4v3u0temp = pParam->BSIM4v3u0temp;
                      here->BSIM4v3vth0 = pParam->BSIM4v3vth0;
                      here->BSIM4v3vsattemp = pParam->BSIM4v3vsattemp;
                      here->BSIM4v3vfb = pParam->BSIM4v3vfb;
                      here->BSIM4v3vfbzb = pParam->BSIM4v3vfbzb;
		      here->BSIM4v3vtfbphi1 = pParam->BSIM4v3vtfbphi1;
		      here->BSIM4v3vtfbphi2 = pParam->BSIM4v3vtfbphi2;
                      here->BSIM4v3k2 = pParam->BSIM4v3k2;
                      here->BSIM4v3vbsc = pParam->BSIM4v3vbsc;
                      here->BSIM4v3k2ox = pParam->BSIM4v3k2ox;
                      here->BSIM4v3eta0 = pParam->BSIM4v3eta0;
              }
                   
              here->BSIM4v3cgso = pParam->BSIM4v3cgso;
              here->BSIM4v3cgdo = pParam->BSIM4v3cgdo;
              
              if (here->BSIM4v3rbodyMod)
              {   if (here->BSIM4v3rbdb < 1.0e-3)
                      here->BSIM4v3grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v3grbdb = model->BSIM4v3gbmin + 1.0 / here->BSIM4v3rbdb;
                  if (here->BSIM4v3rbpb < 1.0e-3)
                      here->BSIM4v3grbpb = 1.0e3;
                  else
                      here->BSIM4v3grbpb = model->BSIM4v3gbmin + 1.0 / here->BSIM4v3rbpb;
                  if (here->BSIM4v3rbps < 1.0e-3)
                      here->BSIM4v3grbps = 1.0e3;
                  else
                      here->BSIM4v3grbps = model->BSIM4v3gbmin + 1.0 / here->BSIM4v3rbps;
                  if (here->BSIM4v3rbsb < 1.0e-3)
                      here->BSIM4v3grbsb = 1.0e3;
                  else
                      here->BSIM4v3grbsb = model->BSIM4v3gbmin + 1.0 / here->BSIM4v3rbsb;
                  if (here->BSIM4v3rbpd < 1.0e-3)
                      here->BSIM4v3grbpd = 1.0e3;
                  else
                      here->BSIM4v3grbpd = model->BSIM4v3gbmin + 1.0 / here->BSIM4v3rbpd;
              }


              /* 
               * Process geomertry dependent parasitics
	       */

              here->BSIM4v3grgeltd = model->BSIM4v3rshg * (model->BSIM4v3xgw
                      + pParam->BSIM4v3weffCJ / 3.0 / model->BSIM4v3ngcon) /
                      (model->BSIM4v3ngcon * here->BSIM4v3nf *
                      (Lnew - model->BSIM4v3xgl));
              if (here->BSIM4v3grgeltd > 0.0)
                  here->BSIM4v3grgeltd = 1.0 / here->BSIM4v3grgeltd;
              else
              {   here->BSIM4v3grgeltd = 1.0e3; /* mho */
		  if (here->BSIM4v3rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

	      DMCGeff = model->BSIM4v3dmcg - model->BSIM4v3dmcgt;
              DMCIeff = model->BSIM4v3dmci;
              DMDGeff = model->BSIM4v3dmdg - model->BSIM4v3dmcgt;

	      if (here->BSIM4v3sourcePerimeterGiven)
	      {   if (model->BSIM4v3perMod == 0)
	              here->BSIM4v3Pseff = here->BSIM4v3sourcePerimeter;
		  else
		      here->BSIM4v3Pseff = here->BSIM4v3sourcePerimeter 
				       - pParam->BSIM4v3weffCJ * here->BSIM4v3nf;
	      }
	      else
	          BSIM4v3PAeffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod, here->BSIM4v3min, 
                                    pParam->BSIM4v3weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &(here->BSIM4v3Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4v3drainPerimeterGiven)
              {   if (model->BSIM4v3perMod == 0)
                      here->BSIM4v3Pdeff = here->BSIM4v3drainPerimeter;
                  else
                      here->BSIM4v3Pdeff = here->BSIM4v3drainPerimeter 
				       - pParam->BSIM4v3weffCJ * here->BSIM4v3nf;
              }
              else
                  BSIM4v3PAeffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod, here->BSIM4v3min,
                                    pParam->BSIM4v3weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &(here->BSIM4v3Pdeff), &dumAs, &dumAd);

              if (here->BSIM4v3sourceAreaGiven)
                  here->BSIM4v3Aseff = here->BSIM4v3sourceArea;
              else
                  BSIM4v3PAeffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod, here->BSIM4v3min,
                                    pParam->BSIM4v3weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &(here->BSIM4v3Aseff), &dumAd);

              if (here->BSIM4v3drainAreaGiven)
                  here->BSIM4v3Adeff = here->BSIM4v3drainArea;
              else
                  BSIM4v3PAeffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod, here->BSIM4v3min,
                                    pParam->BSIM4v3weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &dumAs, &(here->BSIM4v3Adeff));

	      /* Processing S/D resistance and conductance below */
              if(here->BSIM4v3sNodePrime != here->BSIM4v3sNode)
              {
                 here->BSIM4v3sourceConductance = 0.0;
                 if(here->BSIM4v3sourceSquaresGiven)
                 {
                    here->BSIM4v3sourceConductance = model->BSIM4v3sheetResistance
                                               * here->BSIM4v3sourceSquares;
                 } else if (here->BSIM4v3rgeoMod > 0)
                 {
                    BSIM4v3RdseffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod,
                      here->BSIM4v3rgeoMod, here->BSIM4v3min,
                      pParam->BSIM4v3weffCJ, model->BSIM4v3sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v3sourceConductance));
                 } else
                 {
                    here->BSIM4v3sourceConductance = 0.0;
                 }

                 if (here->BSIM4v3sourceConductance > 0.0)
                     here->BSIM4v3sourceConductance = 1.0
                                            / here->BSIM4v3sourceConductance;
                 else
                 {
                     here->BSIM4v3sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4v3sourceConductance = 0.0;
              }

              if(here->BSIM4v3dNodePrime != here->BSIM4v3dNode)
              {
                 here->BSIM4v3drainConductance = 0.0;
                 if(here->BSIM4v3drainSquaresGiven)
                 {
                    here->BSIM4v3drainConductance = model->BSIM4v3sheetResistance
                                              * here->BSIM4v3drainSquares;
                 } else if (here->BSIM4v3rgeoMod > 0)
                 {
                    BSIM4v3RdseffGeo(here->BSIM4v3nf, here->BSIM4v3geoMod,
                      here->BSIM4v3rgeoMod, here->BSIM4v3min,
                      pParam->BSIM4v3weffCJ, model->BSIM4v3sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v3drainConductance));
                 } else
                 {
                    here->BSIM4v3drainConductance = 0.0;
                 }

                 if (here->BSIM4v3drainConductance > 0.0)
                     here->BSIM4v3drainConductance = 1.0
                                           / here->BSIM4v3drainConductance;
                 else
                 {
                     here->BSIM4v3drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4v3drainConductance = 0.0;
              }
           
               /* End of Rsd processing */


              Nvtms = model->BSIM4v3vtm * model->BSIM4v3SjctEmissionCoeff;
              if ((here->BSIM4v3Aseff <= 0.0) && (here->BSIM4v3Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4v3Aseff * model->BSIM4v3SjctTempSatCurDensity
				   + here->BSIM4v3Pseff * model->BSIM4v3SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v3weffCJ * here->BSIM4v3nf
                                   * model->BSIM4v3SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v3dioMod)
                  {   case 0:
			  if ((model->BSIM4v3bvs / Nvtms) > EXP_THRESHOLD)
			      here->BSIM4v3XExpBVS = model->BSIM4v3xjbvs * MIN_EXP;
			  else
	                      here->BSIM4v3XExpBVS = model->BSIM4v3xjbvs * exp(-model->BSIM4v3bvs / Nvtms);	
		          break;
                      case 1:
                          BSIM4v3DioIjthVjmEval(Nvtms, model->BSIM4v3ijthsfwd, SourceSatCurrent, 
			                      0.0, &(here->BSIM4v3vjsmFwd));
                          here->BSIM4v3IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v3vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v3bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v3XExpBVS = model->BSIM4v3xjbvs * MIN_EXP;
			      tmp = MIN_EXP;
			  }
                          else
			  {   here->BSIM4v3XExpBVS = exp(-model->BSIM4v3bvs / Nvtms);
			      tmp = here->BSIM4v3XExpBVS;
		              here->BSIM4v3XExpBVS *= model->BSIM4v3xjbvs;	
			  }

                          BSIM4v3DioIjthVjmEval(Nvtms, model->BSIM4v3ijthsfwd, SourceSatCurrent, 
                               		      here->BSIM4v3XExpBVS, &(here->BSIM4v3vjsmFwd));
		          T0 = exp(here->BSIM4v3vjsmFwd / Nvtms);
                          here->BSIM4v3IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v3XExpBVS / T0
			  		      + here->BSIM4v3XExpBVS - 1.0);
		          here->BSIM4v3SslpFwd = SourceSatCurrent
					       * (T0 + here->BSIM4v3XExpBVS / T0) / Nvtms;

			  T2 = model->BSIM4v3ijthsrev / SourceSatCurrent;
			  if (T2 < 1.0)
			  {   T2 = 10.0;
			      fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
			  } 
                          here->BSIM4v3vjsmRev = -model->BSIM4v3bvs
					     - Nvtms * log((T2 - 1.0) / model->BSIM4v3xjbvs);
			  T1 = model->BSIM4v3xjbvs * exp(-(model->BSIM4v3bvs
			     + here->BSIM4v3vjsmRev) / Nvtms);
			  here->BSIM4v3IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v3SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v3dioMod);
                  }
              }

              Nvtmd = model->BSIM4v3vtm * model->BSIM4v3DjctEmissionCoeff;
	      if ((here->BSIM4v3Adeff <= 0.0) && (here->BSIM4v3Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4v3Adeff * model->BSIM4v3DjctTempSatCurDensity
				  + here->BSIM4v3Pdeff * model->BSIM4v3DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v3weffCJ * here->BSIM4v3nf
                                  * model->BSIM4v3DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v3dioMod)
                  {   case 0:
                          if ((model->BSIM4v3bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v3XExpBVD = model->BSIM4v3xjbvd * MIN_EXP;
                          else
                          here->BSIM4v3XExpBVD = model->BSIM4v3xjbvd * exp(-model->BSIM4v3bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v3DioIjthVjmEval(Nvtmd, model->BSIM4v3ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v3vjdmFwd));
                          here->BSIM4v3IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v3vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v3bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v3XExpBVD = model->BSIM4v3xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v3XExpBVD = exp(-model->BSIM4v3bvd / Nvtmd);
                              tmp = here->BSIM4v3XExpBVD;
                              here->BSIM4v3XExpBVD *= model->BSIM4v3xjbvd;
                          }

                          BSIM4v3DioIjthVjmEval(Nvtmd, model->BSIM4v3ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v3XExpBVD, &(here->BSIM4v3vjdmFwd));
                          T0 = exp(here->BSIM4v3vjdmFwd / Nvtmd);
                          here->BSIM4v3IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v3XExpBVD / T0
                                              + here->BSIM4v3XExpBVD - 1.0);
                          here->BSIM4v3DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v3XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v3ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0) 
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4v3vjdmRev = -model->BSIM4v3bvd
                                             - Nvtmd * log((T2 - 1.0) / model->BSIM4v3xjbvd); /* bugfix */
                          T1 = model->BSIM4v3xjbvd * exp(-(model->BSIM4v3bvd
                             + here->BSIM4v3vjdmRev) / Nvtmd);
                          here->BSIM4v3IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v3DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v3dioMod);
                  }
              }

              if (BSIM4v3checkModel(model, here, ckt))
              {   IFuid namarray[2];
                  namarray[0] = model->BSIM4v3modName;
                  namarray[1] = here->BSIM4v3name;
                  (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM4v3.3.0 parameter checking for %s in model %s", namarray);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
