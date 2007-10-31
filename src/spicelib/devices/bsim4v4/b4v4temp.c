/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4temp.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 **********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

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
#define DEXP(A,B) {                                                        \
        if (A > EXP_THRESHOLD) {                                           \
            B = MAX_EXP*(1.0+(A)-EXP_THRESHOLD);                           \
        } else if (A < -EXP_THRESHOLD)  {                                  \
            B = MIN_EXP;                                                   \
        } else   {                                                         \
            B = exp(A);                                                    \
        }                                                                  \
    }


int
BSIM4V4PAeffGeo(double, int, int, double, double, double, double, double *, double *, double *, double *);
int
BSIM4V4RdseffGeo(double, int, int, int, double, double, double, double, double, int, double *);

int
BSIM4V4DioIjthVjmEval(Nvtm, Ijth, Isb, XExpBV, Vjm)
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
BSIM4V4temp(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4V4model *model = (BSIM4V4model*) inModel;
BSIM4V4instance *here;
struct bsim4SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Lnew=0.0, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
double T10;
double Inv_saref, Inv_sbref, Inv_sa, Inv_sb, rho, Ldrn, dvth0_lod;
double W_tmp, Inv_ODeff, OD_offset, dk2_lod, deta0_lod;

int Size_Not_Found, i;

    /*  loop through all the BSIM4V4 device models */
    for (; model != NULL; model = model->BSIM4V4nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4V4SbulkJctPotential < 0.1)  
	 {   model->BSIM4V4SbulkJctPotential = 0.1;
	     fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
	 }
         if (model->BSIM4V4SsidewallJctPotential < 0.1)
	 {   model->BSIM4V4SsidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
	 }
         if (model->BSIM4V4SGatesidewallJctPotential < 0.1)
	 {   model->BSIM4V4SGatesidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
	 }

         if (model->BSIM4V4DbulkJctPotential < 0.1) 
         {   model->BSIM4V4DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4V4DsidewallJctPotential < 0.1)
         {   model->BSIM4V4DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4V4DGatesidewallJctPotential < 0.1)
         {   model->BSIM4V4DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4V4toxeGiven) && (model->BSIM4V4toxpGiven) && (model->BSIM4V4dtoxGiven)
             && (model->BSIM4V4toxe != (model->BSIM4V4toxp + model->BSIM4V4dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
	 else if ((model->BSIM4V4toxeGiven) && (!model->BSIM4V4toxpGiven))
	     model->BSIM4V4toxp = model->BSIM4V4toxe - model->BSIM4V4dtox;
	 else if ((!model->BSIM4V4toxeGiven) && (model->BSIM4V4toxpGiven))
             model->BSIM4V4toxe = model->BSIM4V4toxp + model->BSIM4V4dtox;

         model->BSIM4V4coxe = model->BSIM4V4epsrox * EPS0 / model->BSIM4V4toxe;
         model->BSIM4V4coxp = model->BSIM4V4epsrox * EPS0 / model->BSIM4V4toxp;

         if (!model->BSIM4V4cgdoGiven)
         {   if (model->BSIM4V4dlcGiven && (model->BSIM4V4dlc > 0.0))
                 model->BSIM4V4cgdo = model->BSIM4V4dlc * model->BSIM4V4coxe
                                  - model->BSIM4V4cgdl ;
             else
                 model->BSIM4V4cgdo = 0.6 * model->BSIM4V4xj * model->BSIM4V4coxe;
         }
         if (!model->BSIM4V4cgsoGiven)
         {   if (model->BSIM4V4dlcGiven && (model->BSIM4V4dlc > 0.0))
                 model->BSIM4V4cgso = model->BSIM4V4dlc * model->BSIM4V4coxe
                                  - model->BSIM4V4cgsl ;
             else
                 model->BSIM4V4cgso = 0.6 * model->BSIM4V4xj * model->BSIM4V4coxe;
         }
         if (!model->BSIM4V4cgboGiven)
             model->BSIM4V4cgbo = 2.0 * model->BSIM4V4dwc * model->BSIM4V4coxe;
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

	 Tnom = model->BSIM4V4tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM4V4vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4V4factor1 = sqrt(EPSSI / (model->BSIM4V4epsrox * EPS0)
                             * model->BSIM4V4toxe);

         Vtm0 = model->BSIM4V4vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4V4vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4V4vtm;
	     T1 = log(Temp / Tnom);
	     T2 = T0 + model->BSIM4V4SjctTempExponent * T1;
	     T3 = exp(T2 / model->BSIM4V4SjctEmissionCoeff);
	     model->BSIM4V4SjctTempSatCurDensity = model->BSIM4V4SjctSatCurDensity
					       * T3;
	     model->BSIM4V4SjctSidewallTempSatCurDensity
			 = model->BSIM4V4SjctSidewallSatCurDensity * T3;
             model->BSIM4V4SjctGateSidewallTempSatCurDensity
                         = model->BSIM4V4SjctGateSidewallSatCurDensity * T3;

	     T2 = T0 + model->BSIM4V4DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4V4DjctEmissionCoeff);
             model->BSIM4V4DjctTempSatCurDensity = model->BSIM4V4DjctSatCurDensity
                                               * T3;
             model->BSIM4V4DjctSidewallTempSatCurDensity
                         = model->BSIM4V4DjctSidewallSatCurDensity * T3;
             model->BSIM4V4DjctGateSidewallTempSatCurDensity
                         = model->BSIM4V4DjctGateSidewallSatCurDensity * T3;
	 }
	 else
	 {   model->BSIM4V4SjctTempSatCurDensity = model->BSIM4V4SjctSatCurDensity;
	     model->BSIM4V4SjctSidewallTempSatCurDensity
			= model->BSIM4V4SjctSidewallSatCurDensity;
             model->BSIM4V4SjctGateSidewallTempSatCurDensity
                        = model->BSIM4V4SjctGateSidewallSatCurDensity;
             model->BSIM4V4DjctTempSatCurDensity = model->BSIM4V4DjctSatCurDensity;
             model->BSIM4V4DjctSidewallTempSatCurDensity
                        = model->BSIM4V4DjctSidewallSatCurDensity;
             model->BSIM4V4DjctGateSidewallTempSatCurDensity
                        = model->BSIM4V4DjctGateSidewallSatCurDensity;
	 }

	 if (model->BSIM4V4SjctTempSatCurDensity < 0.0)
	     model->BSIM4V4SjctTempSatCurDensity = 0.0;
	 if (model->BSIM4V4SjctSidewallTempSatCurDensity < 0.0)
	     model->BSIM4V4SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4V4SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4V4SjctGateSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4V4DjctTempSatCurDensity < 0.0)
             model->BSIM4V4DjctTempSatCurDensity = 0.0;
         if (model->BSIM4V4DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4V4DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4V4DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4V4DjctGateSidewallTempSatCurDensity = 0.0;

	 /* Temperature dependence of D/B and S/B diode capacitance begins */
	 delTemp = ckt->CKTtemp - model->BSIM4V4tnom;
	 T0 = model->BSIM4V4tcj * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4V4SunitAreaTempJctCap = model->BSIM4V4SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4V4DunitAreaTempJctCap = model->BSIM4V4DunitAreaJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4V4SunitAreaJctCap > 0.0)
	     {   model->BSIM4V4SunitAreaTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
	     if (model->BSIM4V4DunitAreaJctCap > 0.0)
             {   model->BSIM4V4DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
	 }
         T0 = model->BSIM4V4tcjsw * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4V4SunitLengthSidewallTempJctCap = model->BSIM4V4SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4V4DunitLengthSidewallTempJctCap = model->BSIM4V4DunitLengthSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4V4SunitLengthSidewallJctCap > 0.0)
	     {   model->BSIM4V4SunitLengthSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
	     }
	     if (model->BSIM4V4DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4V4DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }	
	 }
         T0 = model->BSIM4V4tcjswg * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4V4SunitLengthGateSidewallTempJctCap = model->BSIM4V4SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4V4DunitLengthGateSidewallTempJctCap = model->BSIM4V4DunitLengthGateSidewallJctCap *(1.0 + T0);
	 }
	 else
	 {   if (model->BSIM4V4SunitLengthGateSidewallJctCap > 0.0)
	     {   model->BSIM4V4SunitLengthGateSidewallTempJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
	     }
	     if (model->BSIM4V4DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4V4DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
	 }

         model->BSIM4V4PhiBS = model->BSIM4V4SbulkJctPotential
			   - model->BSIM4V4tpb * delTemp;
         if (model->BSIM4V4PhiBS < 0.01)
	 {   model->BSIM4V4PhiBS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
	 }
         model->BSIM4V4PhiBD = model->BSIM4V4DbulkJctPotential
                           - model->BSIM4V4tpb * delTemp;
         if (model->BSIM4V4PhiBD < 0.01)
         {   model->BSIM4V4PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4V4PhiBSWS = model->BSIM4V4SsidewallJctPotential
                             - model->BSIM4V4tpbsw * delTemp;
         if (model->BSIM4V4PhiBSWS <= 0.01)
	 {   model->BSIM4V4PhiBSWS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
	 }
         model->BSIM4V4PhiBSWD = model->BSIM4V4DsidewallJctPotential
                             - model->BSIM4V4tpbsw * delTemp;
         if (model->BSIM4V4PhiBSWD <= 0.01)
         {   model->BSIM4V4PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

	 model->BSIM4V4PhiBSWGS = model->BSIM4V4SGatesidewallJctPotential
                              - model->BSIM4V4tpbswg * delTemp;
         if (model->BSIM4V4PhiBSWGS <= 0.01)
	 {   model->BSIM4V4PhiBSWGS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
	 }
         model->BSIM4V4PhiBSWGD = model->BSIM4V4DGatesidewallJctPotential
                              - model->BSIM4V4tpbswg * delTemp;
         if (model->BSIM4V4PhiBSWGD <= 0.01)
         {   model->BSIM4V4PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4V4ijthdfwd <= 0.0)
         {   model->BSIM4V4ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4V4ijthdfwd);
         }
         if (model->BSIM4V4ijthsfwd <= 0.0)
         {   model->BSIM4V4ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4V4ijthsfwd);
         }
	 if (model->BSIM4V4ijthdrev <= 0.0)
         {   model->BSIM4V4ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4V4ijthdrev);
         }
         if (model->BSIM4V4ijthsrev <= 0.0)
         {   model->BSIM4V4ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4V4ijthsrev);
         }

         if ((model->BSIM4V4xjbvd <= 0.0) && (model->BSIM4V4dioMod == 2))
         {   model->BSIM4V4xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4V4xjbvd);
         }
         else if ((model->BSIM4V4xjbvd < 0.0) && (model->BSIM4V4dioMod == 0))
         {   model->BSIM4V4xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4V4xjbvd);
         }

         if (model->BSIM4V4bvd <= 0.0)
         {   model->BSIM4V4bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4V4bvd);
         }

         if ((model->BSIM4V4xjbvs <= 0.0) && (model->BSIM4V4dioMod == 2))
         {   model->BSIM4V4xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4V4xjbvs);
         }
         else if ((model->BSIM4V4xjbvs < 0.0) && (model->BSIM4V4dioMod == 0))
         {   model->BSIM4V4xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4V4xjbvs);
         }

         if (model->BSIM4V4bvs <= 0.0)
         {   model->BSIM4V4bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4V4bvs);
         }


         /* loop through all the instances of the model */
         for (here = model->BSIM4V4instances; here != NULL;
              here = here->BSIM4V4nextInstance) 
      { if (here->BSIM4V4owner != ARCHme) continue;
	      pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM4V4l == pSizeDependParamKnot->Length)
		      && (here->BSIM4V4w == pSizeDependParamKnot->Width)
		      && (here->BSIM4V4nf == pSizeDependParamKnot->NFinger))
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
	      Ldrn = here->BSIM4V4l;

	      if (Size_Not_Found)
	      {   pParam = (struct bsim4SizeDependParam *)malloc(
	                    sizeof(struct bsim4SizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4V4l;
                  pParam->Width = here->BSIM4V4w;
		  pParam->NFinger = here->BSIM4V4nf;
                  Lnew = here->BSIM4V4l  + model->BSIM4V4xl ;
                  Wnew = here->BSIM4V4w / here->BSIM4V4nf + model->BSIM4V4xw;

                  T0 = pow(Lnew, model->BSIM4V4Lln);
                  T1 = pow(Wnew, model->BSIM4V4Lwn);
                  tmp1 = model->BSIM4V4Ll / T0 + model->BSIM4V4Lw / T1
                       + model->BSIM4V4Lwl / (T0 * T1);
                  pParam->BSIM4V4dl = model->BSIM4V4Lint + tmp1;
                  tmp2 = model->BSIM4V4Llc / T0 + model->BSIM4V4Lwc / T1
                       + model->BSIM4V4Lwlc / (T0 * T1);
                  pParam->BSIM4V4dlc = model->BSIM4V4dlc + tmp2;
                  pParam->BSIM4V4dlcig = model->BSIM4V4dlcig;

                  T2 = pow(Lnew, model->BSIM4V4Wln);
                  T3 = pow(Wnew, model->BSIM4V4Wwn);
                  tmp1 = model->BSIM4V4Wl / T2 + model->BSIM4V4Ww / T3
                       + model->BSIM4V4Wwl / (T2 * T3);
                  pParam->BSIM4V4dw = model->BSIM4V4Wint + tmp1;
                  tmp2 = model->BSIM4V4Wlc / T2 + model->BSIM4V4Wwc / T3
                       + model->BSIM4V4Wwlc / (T2 * T3); 
                  pParam->BSIM4V4dwc = model->BSIM4V4dwc + tmp2;
                  pParam->BSIM4V4dwj = model->BSIM4V4dwj + tmp2;

                  pParam->BSIM4V4leff = Lnew - 2.0 * pParam->BSIM4V4dl;
                  if (pParam->BSIM4V4leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4V4modName;
                      namarray[1] = here->BSIM4V4name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4V4: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4V4weff = Wnew - 2.0 * pParam->BSIM4V4dw;
                  if (pParam->BSIM4V4weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4V4modName;
                      namarray[1] = here->BSIM4V4name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4V4: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4V4leffCV = Lnew - 2.0 * pParam->BSIM4V4dlc;
                  if (pParam->BSIM4V4leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4V4modName;
                      namarray[1] = here->BSIM4V4name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4V4: mosfet %s, model %s: Effective channel length for C-V <= 0",
		      namarray);
		      fprintf(stderr,"BSIM4V4: mosfet %s, model %s: Effective channel length for C-V <= 0 (Lnew - 2.0*BSIM4V4dlc == %g - 2.0*%g)\n",
			      model->BSIM4V4modName, here->BSIM4V4name, Lnew, pParam->BSIM4V4dlc );
                      return(E_BADPARM);
                  }

                  pParam->BSIM4V4weffCV = Wnew - 2.0 * pParam->BSIM4V4dwc;
                  if (pParam->BSIM4V4weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4V4modName;
                      namarray[1] = here->BSIM4V4name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4V4: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4V4weffCJ = Wnew - 2.0 * pParam->BSIM4V4dwj;
                  if (pParam->BSIM4V4weffCJ <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4V4modName;
                      namarray[1] = here->BSIM4V4name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4V4: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM4V4binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM4V4leff;
		      Inv_W = 1.0e-6 / pParam->BSIM4V4weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM4V4leff
			     * pParam->BSIM4V4weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM4V4leff;
		      Inv_W = 1.0 / pParam->BSIM4V4weff;
		      Inv_LW = 1.0 / (pParam->BSIM4V4leff
			     * pParam->BSIM4V4weff);
		  }
		  pParam->BSIM4V4cdsc = model->BSIM4V4cdsc
				    + model->BSIM4V4lcdsc * Inv_L
				    + model->BSIM4V4wcdsc * Inv_W
				    + model->BSIM4V4pcdsc * Inv_LW;
		  pParam->BSIM4V4cdscb = model->BSIM4V4cdscb
				     + model->BSIM4V4lcdscb * Inv_L
				     + model->BSIM4V4wcdscb * Inv_W
				     + model->BSIM4V4pcdscb * Inv_LW; 
				     
    		  pParam->BSIM4V4cdscd = model->BSIM4V4cdscd
				     + model->BSIM4V4lcdscd * Inv_L
				     + model->BSIM4V4wcdscd * Inv_W
				     + model->BSIM4V4pcdscd * Inv_LW; 
				     
		  pParam->BSIM4V4cit = model->BSIM4V4cit
				   + model->BSIM4V4lcit * Inv_L
				   + model->BSIM4V4wcit * Inv_W
				   + model->BSIM4V4pcit * Inv_LW;
		  pParam->BSIM4V4nfactor = model->BSIM4V4nfactor
				       + model->BSIM4V4lnfactor * Inv_L
				       + model->BSIM4V4wnfactor * Inv_W
				       + model->BSIM4V4pnfactor * Inv_LW;
		  pParam->BSIM4V4xj = model->BSIM4V4xj
				  + model->BSIM4V4lxj * Inv_L
				  + model->BSIM4V4wxj * Inv_W
				  + model->BSIM4V4pxj * Inv_LW;
		  pParam->BSIM4V4vsat = model->BSIM4V4vsat
				    + model->BSIM4V4lvsat * Inv_L
				    + model->BSIM4V4wvsat * Inv_W
				    + model->BSIM4V4pvsat * Inv_LW;
		  pParam->BSIM4V4at = model->BSIM4V4at
				  + model->BSIM4V4lat * Inv_L
				  + model->BSIM4V4wat * Inv_W
				  + model->BSIM4V4pat * Inv_LW;
		  pParam->BSIM4V4a0 = model->BSIM4V4a0
				  + model->BSIM4V4la0 * Inv_L
				  + model->BSIM4V4wa0 * Inv_W
				  + model->BSIM4V4pa0 * Inv_LW; 
				  
		  pParam->BSIM4V4ags = model->BSIM4V4ags
				  + model->BSIM4V4lags * Inv_L
				  + model->BSIM4V4wags * Inv_W
				  + model->BSIM4V4pags * Inv_LW;
				  
		  pParam->BSIM4V4a1 = model->BSIM4V4a1
				  + model->BSIM4V4la1 * Inv_L
				  + model->BSIM4V4wa1 * Inv_W
				  + model->BSIM4V4pa1 * Inv_LW;
		  pParam->BSIM4V4a2 = model->BSIM4V4a2
				  + model->BSIM4V4la2 * Inv_L
				  + model->BSIM4V4wa2 * Inv_W
				  + model->BSIM4V4pa2 * Inv_LW;
		  pParam->BSIM4V4keta = model->BSIM4V4keta
				    + model->BSIM4V4lketa * Inv_L
				    + model->BSIM4V4wketa * Inv_W
				    + model->BSIM4V4pketa * Inv_LW;
		  pParam->BSIM4V4nsub = model->BSIM4V4nsub
				    + model->BSIM4V4lnsub * Inv_L
				    + model->BSIM4V4wnsub * Inv_W
				    + model->BSIM4V4pnsub * Inv_LW;
		  pParam->BSIM4V4ndep = model->BSIM4V4ndep
				    + model->BSIM4V4lndep * Inv_L
				    + model->BSIM4V4wndep * Inv_W
				    + model->BSIM4V4pndep * Inv_LW;
                  pParam->BSIM4V4nsd = model->BSIM4V4nsd
                                   + model->BSIM4V4lnsd * Inv_L
                                   + model->BSIM4V4wnsd * Inv_W
                                   + model->BSIM4V4pnsd * Inv_LW;
                  pParam->BSIM4V4phin = model->BSIM4V4phin
                                    + model->BSIM4V4lphin * Inv_L
                                    + model->BSIM4V4wphin * Inv_W
                                    + model->BSIM4V4pphin * Inv_LW;
		  pParam->BSIM4V4ngate = model->BSIM4V4ngate
				     + model->BSIM4V4lngate * Inv_L
				     + model->BSIM4V4wngate * Inv_W
				     + model->BSIM4V4pngate * Inv_LW;
		  pParam->BSIM4V4gamma1 = model->BSIM4V4gamma1
				      + model->BSIM4V4lgamma1 * Inv_L
				      + model->BSIM4V4wgamma1 * Inv_W
				      + model->BSIM4V4pgamma1 * Inv_LW;
		  pParam->BSIM4V4gamma2 = model->BSIM4V4gamma2
				      + model->BSIM4V4lgamma2 * Inv_L
				      + model->BSIM4V4wgamma2 * Inv_W
				      + model->BSIM4V4pgamma2 * Inv_LW;
		  pParam->BSIM4V4vbx = model->BSIM4V4vbx
				   + model->BSIM4V4lvbx * Inv_L
				   + model->BSIM4V4wvbx * Inv_W
				   + model->BSIM4V4pvbx * Inv_LW;
		  pParam->BSIM4V4vbm = model->BSIM4V4vbm
				   + model->BSIM4V4lvbm * Inv_L
				   + model->BSIM4V4wvbm * Inv_W
				   + model->BSIM4V4pvbm * Inv_LW;
		  pParam->BSIM4V4xt = model->BSIM4V4xt
				   + model->BSIM4V4lxt * Inv_L
				   + model->BSIM4V4wxt * Inv_W
				   + model->BSIM4V4pxt * Inv_LW;
                  pParam->BSIM4V4vfb = model->BSIM4V4vfb
                                   + model->BSIM4V4lvfb * Inv_L
                                   + model->BSIM4V4wvfb * Inv_W
                                   + model->BSIM4V4pvfb * Inv_LW;
		  pParam->BSIM4V4k1 = model->BSIM4V4k1
				  + model->BSIM4V4lk1 * Inv_L
				  + model->BSIM4V4wk1 * Inv_W
				  + model->BSIM4V4pk1 * Inv_LW;
		  pParam->BSIM4V4kt1 = model->BSIM4V4kt1
				   + model->BSIM4V4lkt1 * Inv_L
				   + model->BSIM4V4wkt1 * Inv_W
				   + model->BSIM4V4pkt1 * Inv_LW;
		  pParam->BSIM4V4kt1l = model->BSIM4V4kt1l
				    + model->BSIM4V4lkt1l * Inv_L
				    + model->BSIM4V4wkt1l * Inv_W
				    + model->BSIM4V4pkt1l * Inv_LW;
		  pParam->BSIM4V4k2 = model->BSIM4V4k2
				  + model->BSIM4V4lk2 * Inv_L
				  + model->BSIM4V4wk2 * Inv_W
				  + model->BSIM4V4pk2 * Inv_LW;
		  pParam->BSIM4V4kt2 = model->BSIM4V4kt2
				   + model->BSIM4V4lkt2 * Inv_L
				   + model->BSIM4V4wkt2 * Inv_W
				   + model->BSIM4V4pkt2 * Inv_LW;
		  pParam->BSIM4V4k3 = model->BSIM4V4k3
				  + model->BSIM4V4lk3 * Inv_L
				  + model->BSIM4V4wk3 * Inv_W
				  + model->BSIM4V4pk3 * Inv_LW;
		  pParam->BSIM4V4k3b = model->BSIM4V4k3b
				   + model->BSIM4V4lk3b * Inv_L
				   + model->BSIM4V4wk3b * Inv_W
				   + model->BSIM4V4pk3b * Inv_LW;
		  pParam->BSIM4V4w0 = model->BSIM4V4w0
				  + model->BSIM4V4lw0 * Inv_L
				  + model->BSIM4V4ww0 * Inv_W
				  + model->BSIM4V4pw0 * Inv_LW;
		  pParam->BSIM4V4lpe0 = model->BSIM4V4lpe0
				    + model->BSIM4V4llpe0 * Inv_L
 				    + model->BSIM4V4wlpe0 * Inv_W
				    + model->BSIM4V4plpe0 * Inv_LW;
                  pParam->BSIM4V4lpeb = model->BSIM4V4lpeb
                                    + model->BSIM4V4llpeb * Inv_L
                                    + model->BSIM4V4wlpeb * Inv_W
                                    + model->BSIM4V4plpeb * Inv_LW;
                  pParam->BSIM4V4dvtp0 = model->BSIM4V4dvtp0
                                     + model->BSIM4V4ldvtp0 * Inv_L
                                     + model->BSIM4V4wdvtp0 * Inv_W
                                     + model->BSIM4V4pdvtp0 * Inv_LW;
                  pParam->BSIM4V4dvtp1 = model->BSIM4V4dvtp1
                                     + model->BSIM4V4ldvtp1 * Inv_L
                                     + model->BSIM4V4wdvtp1 * Inv_W
                                     + model->BSIM4V4pdvtp1 * Inv_LW;
		  pParam->BSIM4V4dvt0 = model->BSIM4V4dvt0
				    + model->BSIM4V4ldvt0 * Inv_L
				    + model->BSIM4V4wdvt0 * Inv_W
				    + model->BSIM4V4pdvt0 * Inv_LW;
		  pParam->BSIM4V4dvt1 = model->BSIM4V4dvt1
				    + model->BSIM4V4ldvt1 * Inv_L
				    + model->BSIM4V4wdvt1 * Inv_W
				    + model->BSIM4V4pdvt1 * Inv_LW;
		  pParam->BSIM4V4dvt2 = model->BSIM4V4dvt2
				    + model->BSIM4V4ldvt2 * Inv_L
				    + model->BSIM4V4wdvt2 * Inv_W
				    + model->BSIM4V4pdvt2 * Inv_LW;
		  pParam->BSIM4V4dvt0w = model->BSIM4V4dvt0w
				    + model->BSIM4V4ldvt0w * Inv_L
				    + model->BSIM4V4wdvt0w * Inv_W
				    + model->BSIM4V4pdvt0w * Inv_LW;
		  pParam->BSIM4V4dvt1w = model->BSIM4V4dvt1w
				    + model->BSIM4V4ldvt1w * Inv_L
				    + model->BSIM4V4wdvt1w * Inv_W
				    + model->BSIM4V4pdvt1w * Inv_LW;
		  pParam->BSIM4V4dvt2w = model->BSIM4V4dvt2w
				    + model->BSIM4V4ldvt2w * Inv_L
				    + model->BSIM4V4wdvt2w * Inv_W
				    + model->BSIM4V4pdvt2w * Inv_LW;
		  pParam->BSIM4V4drout = model->BSIM4V4drout
				     + model->BSIM4V4ldrout * Inv_L
				     + model->BSIM4V4wdrout * Inv_W
				     + model->BSIM4V4pdrout * Inv_LW;
		  pParam->BSIM4V4dsub = model->BSIM4V4dsub
				    + model->BSIM4V4ldsub * Inv_L
				    + model->BSIM4V4wdsub * Inv_W
				    + model->BSIM4V4pdsub * Inv_LW;
		  pParam->BSIM4V4vth0 = model->BSIM4V4vth0
				    + model->BSIM4V4lvth0 * Inv_L
				    + model->BSIM4V4wvth0 * Inv_W
				    + model->BSIM4V4pvth0 * Inv_LW;
		  pParam->BSIM4V4ua = model->BSIM4V4ua
				  + model->BSIM4V4lua * Inv_L
				  + model->BSIM4V4wua * Inv_W
				  + model->BSIM4V4pua * Inv_LW;
		  pParam->BSIM4V4ua1 = model->BSIM4V4ua1
				   + model->BSIM4V4lua1 * Inv_L
				   + model->BSIM4V4wua1 * Inv_W
				   + model->BSIM4V4pua1 * Inv_LW;
		  pParam->BSIM4V4ub = model->BSIM4V4ub
				  + model->BSIM4V4lub * Inv_L
				  + model->BSIM4V4wub * Inv_W
				  + model->BSIM4V4pub * Inv_LW;
		  pParam->BSIM4V4ub1 = model->BSIM4V4ub1
				   + model->BSIM4V4lub1 * Inv_L
				   + model->BSIM4V4wub1 * Inv_W
				   + model->BSIM4V4pub1 * Inv_LW;
		  pParam->BSIM4V4uc = model->BSIM4V4uc
				  + model->BSIM4V4luc * Inv_L
				  + model->BSIM4V4wuc * Inv_W
				  + model->BSIM4V4puc * Inv_LW;
		  pParam->BSIM4V4uc1 = model->BSIM4V4uc1
				   + model->BSIM4V4luc1 * Inv_L
				   + model->BSIM4V4wuc1 * Inv_W
				   + model->BSIM4V4puc1 * Inv_LW;
                  pParam->BSIM4V4eu = model->BSIM4V4eu
                                  + model->BSIM4V4leu * Inv_L
                                  + model->BSIM4V4weu * Inv_W
                                  + model->BSIM4V4peu * Inv_LW;
		  pParam->BSIM4V4u0 = model->BSIM4V4u0
				  + model->BSIM4V4lu0 * Inv_L
				  + model->BSIM4V4wu0 * Inv_W
				  + model->BSIM4V4pu0 * Inv_LW;
		  pParam->BSIM4V4ute = model->BSIM4V4ute
				   + model->BSIM4V4lute * Inv_L
				   + model->BSIM4V4wute * Inv_W
				   + model->BSIM4V4pute * Inv_LW;
		  pParam->BSIM4V4voff = model->BSIM4V4voff
				    + model->BSIM4V4lvoff * Inv_L
				    + model->BSIM4V4wvoff * Inv_W
				    + model->BSIM4V4pvoff * Inv_LW;
                  pParam->BSIM4V4minv = model->BSIM4V4minv
                                    + model->BSIM4V4lminv * Inv_L
                                    + model->BSIM4V4wminv * Inv_W
                                    + model->BSIM4V4pminv * Inv_LW;
                  pParam->BSIM4V4fprout = model->BSIM4V4fprout
                                     + model->BSIM4V4lfprout * Inv_L
                                     + model->BSIM4V4wfprout * Inv_W
                                     + model->BSIM4V4pfprout * Inv_LW;
                  pParam->BSIM4V4pdits = model->BSIM4V4pdits
                                     + model->BSIM4V4lpdits * Inv_L
                                     + model->BSIM4V4wpdits * Inv_W
                                     + model->BSIM4V4ppdits * Inv_LW;
                  pParam->BSIM4V4pditsd = model->BSIM4V4pditsd
                                      + model->BSIM4V4lpditsd * Inv_L
                                      + model->BSIM4V4wpditsd * Inv_W
                                      + model->BSIM4V4ppditsd * Inv_LW;
		  pParam->BSIM4V4delta = model->BSIM4V4delta
				     + model->BSIM4V4ldelta * Inv_L
				     + model->BSIM4V4wdelta * Inv_W
				     + model->BSIM4V4pdelta * Inv_LW;
		  pParam->BSIM4V4rdsw = model->BSIM4V4rdsw
				    + model->BSIM4V4lrdsw * Inv_L
				    + model->BSIM4V4wrdsw * Inv_W
				    + model->BSIM4V4prdsw * Inv_LW;
                  pParam->BSIM4V4rdw = model->BSIM4V4rdw
                                    + model->BSIM4V4lrdw * Inv_L
                                    + model->BSIM4V4wrdw * Inv_W
                                    + model->BSIM4V4prdw * Inv_LW;
                  pParam->BSIM4V4rsw = model->BSIM4V4rsw
                                    + model->BSIM4V4lrsw * Inv_L
                                    + model->BSIM4V4wrsw * Inv_W
                                    + model->BSIM4V4prsw * Inv_LW;
		  pParam->BSIM4V4prwg = model->BSIM4V4prwg
				    + model->BSIM4V4lprwg * Inv_L
				    + model->BSIM4V4wprwg * Inv_W
				    + model->BSIM4V4pprwg * Inv_LW;
		  pParam->BSIM4V4prwb = model->BSIM4V4prwb
				    + model->BSIM4V4lprwb * Inv_L
				    + model->BSIM4V4wprwb * Inv_W
				    + model->BSIM4V4pprwb * Inv_LW;
		  pParam->BSIM4V4prt = model->BSIM4V4prt
				    + model->BSIM4V4lprt * Inv_L
				    + model->BSIM4V4wprt * Inv_W
				    + model->BSIM4V4pprt * Inv_LW;
		  pParam->BSIM4V4eta0 = model->BSIM4V4eta0
				    + model->BSIM4V4leta0 * Inv_L
				    + model->BSIM4V4weta0 * Inv_W
				    + model->BSIM4V4peta0 * Inv_LW;
		  pParam->BSIM4V4etab = model->BSIM4V4etab
				    + model->BSIM4V4letab * Inv_L
				    + model->BSIM4V4wetab * Inv_W
				    + model->BSIM4V4petab * Inv_LW;
		  pParam->BSIM4V4pclm = model->BSIM4V4pclm
				    + model->BSIM4V4lpclm * Inv_L
				    + model->BSIM4V4wpclm * Inv_W
				    + model->BSIM4V4ppclm * Inv_LW;
		  pParam->BSIM4V4pdibl1 = model->BSIM4V4pdibl1
				      + model->BSIM4V4lpdibl1 * Inv_L
				      + model->BSIM4V4wpdibl1 * Inv_W
				      + model->BSIM4V4ppdibl1 * Inv_LW;
		  pParam->BSIM4V4pdibl2 = model->BSIM4V4pdibl2
				      + model->BSIM4V4lpdibl2 * Inv_L
				      + model->BSIM4V4wpdibl2 * Inv_W
				      + model->BSIM4V4ppdibl2 * Inv_LW;
		  pParam->BSIM4V4pdiblb = model->BSIM4V4pdiblb
				      + model->BSIM4V4lpdiblb * Inv_L
				      + model->BSIM4V4wpdiblb * Inv_W
				      + model->BSIM4V4ppdiblb * Inv_LW;
		  pParam->BSIM4V4pscbe1 = model->BSIM4V4pscbe1
				      + model->BSIM4V4lpscbe1 * Inv_L
				      + model->BSIM4V4wpscbe1 * Inv_W
				      + model->BSIM4V4ppscbe1 * Inv_LW;
		  pParam->BSIM4V4pscbe2 = model->BSIM4V4pscbe2
				      + model->BSIM4V4lpscbe2 * Inv_L
				      + model->BSIM4V4wpscbe2 * Inv_W
				      + model->BSIM4V4ppscbe2 * Inv_LW;
		  pParam->BSIM4V4pvag = model->BSIM4V4pvag
				    + model->BSIM4V4lpvag * Inv_L
				    + model->BSIM4V4wpvag * Inv_W
				    + model->BSIM4V4ppvag * Inv_LW;
		  pParam->BSIM4V4wr = model->BSIM4V4wr
				  + model->BSIM4V4lwr * Inv_L
				  + model->BSIM4V4wwr * Inv_W
				  + model->BSIM4V4pwr * Inv_LW;
		  pParam->BSIM4V4dwg = model->BSIM4V4dwg
				   + model->BSIM4V4ldwg * Inv_L
				   + model->BSIM4V4wdwg * Inv_W
				   + model->BSIM4V4pdwg * Inv_LW;
		  pParam->BSIM4V4dwb = model->BSIM4V4dwb
				   + model->BSIM4V4ldwb * Inv_L
				   + model->BSIM4V4wdwb * Inv_W
				   + model->BSIM4V4pdwb * Inv_LW;
		  pParam->BSIM4V4b0 = model->BSIM4V4b0
				  + model->BSIM4V4lb0 * Inv_L
				  + model->BSIM4V4wb0 * Inv_W
				  + model->BSIM4V4pb0 * Inv_LW;
		  pParam->BSIM4V4b1 = model->BSIM4V4b1
				  + model->BSIM4V4lb1 * Inv_L
				  + model->BSIM4V4wb1 * Inv_W
				  + model->BSIM4V4pb1 * Inv_LW;
		  pParam->BSIM4V4alpha0 = model->BSIM4V4alpha0
				      + model->BSIM4V4lalpha0 * Inv_L
				      + model->BSIM4V4walpha0 * Inv_W
				      + model->BSIM4V4palpha0 * Inv_LW;
                  pParam->BSIM4V4alpha1 = model->BSIM4V4alpha1
                                      + model->BSIM4V4lalpha1 * Inv_L
                                      + model->BSIM4V4walpha1 * Inv_W
                                      + model->BSIM4V4palpha1 * Inv_LW;
		  pParam->BSIM4V4beta0 = model->BSIM4V4beta0
				     + model->BSIM4V4lbeta0 * Inv_L
				     + model->BSIM4V4wbeta0 * Inv_W
				     + model->BSIM4V4pbeta0 * Inv_LW;
                  pParam->BSIM4V4agidl = model->BSIM4V4agidl
                                     + model->BSIM4V4lagidl * Inv_L
                                     + model->BSIM4V4wagidl * Inv_W
                                     + model->BSIM4V4pagidl * Inv_LW;
                  pParam->BSIM4V4bgidl = model->BSIM4V4bgidl
                                     + model->BSIM4V4lbgidl * Inv_L
                                     + model->BSIM4V4wbgidl * Inv_W
                                     + model->BSIM4V4pbgidl * Inv_LW;
                  pParam->BSIM4V4cgidl = model->BSIM4V4cgidl
                                     + model->BSIM4V4lcgidl * Inv_L
                                     + model->BSIM4V4wcgidl * Inv_W
                                     + model->BSIM4V4pcgidl * Inv_LW;
                  pParam->BSIM4V4egidl = model->BSIM4V4egidl
                                     + model->BSIM4V4legidl * Inv_L
                                     + model->BSIM4V4wegidl * Inv_W
                                     + model->BSIM4V4pegidl * Inv_LW;
                  pParam->BSIM4V4aigc = model->BSIM4V4aigc
                                     + model->BSIM4V4laigc * Inv_L
                                     + model->BSIM4V4waigc * Inv_W
                                     + model->BSIM4V4paigc * Inv_LW;
                  pParam->BSIM4V4bigc = model->BSIM4V4bigc
                                     + model->BSIM4V4lbigc * Inv_L
                                     + model->BSIM4V4wbigc * Inv_W
                                     + model->BSIM4V4pbigc * Inv_LW;
                  pParam->BSIM4V4cigc = model->BSIM4V4cigc
                                     + model->BSIM4V4lcigc * Inv_L
                                     + model->BSIM4V4wcigc * Inv_W
                                     + model->BSIM4V4pcigc * Inv_LW;
                  pParam->BSIM4V4aigsd = model->BSIM4V4aigsd
                                     + model->BSIM4V4laigsd * Inv_L
                                     + model->BSIM4V4waigsd * Inv_W
                                     + model->BSIM4V4paigsd * Inv_LW;
                  pParam->BSIM4V4bigsd = model->BSIM4V4bigsd
                                     + model->BSIM4V4lbigsd * Inv_L
                                     + model->BSIM4V4wbigsd * Inv_W
                                     + model->BSIM4V4pbigsd * Inv_LW;
                  pParam->BSIM4V4cigsd = model->BSIM4V4cigsd
                                     + model->BSIM4V4lcigsd * Inv_L
                                     + model->BSIM4V4wcigsd * Inv_W
                                     + model->BSIM4V4pcigsd * Inv_LW;
                  pParam->BSIM4V4aigbacc = model->BSIM4V4aigbacc
                                       + model->BSIM4V4laigbacc * Inv_L
                                       + model->BSIM4V4waigbacc * Inv_W
                                       + model->BSIM4V4paigbacc * Inv_LW;
                  pParam->BSIM4V4bigbacc = model->BSIM4V4bigbacc
                                       + model->BSIM4V4lbigbacc * Inv_L
                                       + model->BSIM4V4wbigbacc * Inv_W
                                       + model->BSIM4V4pbigbacc * Inv_LW;
                  pParam->BSIM4V4cigbacc = model->BSIM4V4cigbacc
                                       + model->BSIM4V4lcigbacc * Inv_L
                                       + model->BSIM4V4wcigbacc * Inv_W
                                       + model->BSIM4V4pcigbacc * Inv_LW;
                  pParam->BSIM4V4aigbinv = model->BSIM4V4aigbinv
                                       + model->BSIM4V4laigbinv * Inv_L
                                       + model->BSIM4V4waigbinv * Inv_W
                                       + model->BSIM4V4paigbinv * Inv_LW;
                  pParam->BSIM4V4bigbinv = model->BSIM4V4bigbinv
                                       + model->BSIM4V4lbigbinv * Inv_L
                                       + model->BSIM4V4wbigbinv * Inv_W
                                       + model->BSIM4V4pbigbinv * Inv_LW;
                  pParam->BSIM4V4cigbinv = model->BSIM4V4cigbinv
                                       + model->BSIM4V4lcigbinv * Inv_L
                                       + model->BSIM4V4wcigbinv * Inv_W
                                       + model->BSIM4V4pcigbinv * Inv_LW;
                  pParam->BSIM4V4nigc = model->BSIM4V4nigc
                                       + model->BSIM4V4lnigc * Inv_L
                                       + model->BSIM4V4wnigc * Inv_W
                                       + model->BSIM4V4pnigc * Inv_LW;
                  pParam->BSIM4V4nigbacc = model->BSIM4V4nigbacc
                                       + model->BSIM4V4lnigbacc * Inv_L
                                       + model->BSIM4V4wnigbacc * Inv_W
                                       + model->BSIM4V4pnigbacc * Inv_LW;
                  pParam->BSIM4V4nigbinv = model->BSIM4V4nigbinv
                                       + model->BSIM4V4lnigbinv * Inv_L
                                       + model->BSIM4V4wnigbinv * Inv_W
                                       + model->BSIM4V4pnigbinv * Inv_LW;
                  pParam->BSIM4V4ntox = model->BSIM4V4ntox
                                    + model->BSIM4V4lntox * Inv_L
                                    + model->BSIM4V4wntox * Inv_W
                                    + model->BSIM4V4pntox * Inv_LW;
                  pParam->BSIM4V4eigbinv = model->BSIM4V4eigbinv
                                       + model->BSIM4V4leigbinv * Inv_L
                                       + model->BSIM4V4weigbinv * Inv_W
                                       + model->BSIM4V4peigbinv * Inv_LW;
                  pParam->BSIM4V4pigcd = model->BSIM4V4pigcd
                                     + model->BSIM4V4lpigcd * Inv_L
                                     + model->BSIM4V4wpigcd * Inv_W
                                     + model->BSIM4V4ppigcd * Inv_LW;
                  pParam->BSIM4V4poxedge = model->BSIM4V4poxedge
                                       + model->BSIM4V4lpoxedge * Inv_L
                                       + model->BSIM4V4wpoxedge * Inv_W
                                       + model->BSIM4V4ppoxedge * Inv_LW;
                  pParam->BSIM4V4xrcrg1 = model->BSIM4V4xrcrg1
                                      + model->BSIM4V4lxrcrg1 * Inv_L
                                      + model->BSIM4V4wxrcrg1 * Inv_W
                                      + model->BSIM4V4pxrcrg1 * Inv_LW;
                  pParam->BSIM4V4xrcrg2 = model->BSIM4V4xrcrg2
                                      + model->BSIM4V4lxrcrg2 * Inv_L
                                      + model->BSIM4V4wxrcrg2 * Inv_W
                                      + model->BSIM4V4pxrcrg2 * Inv_LW;
                  pParam->BSIM4V4lambda = model->BSIM4V4lambda
                                      + model->BSIM4V4llambda * Inv_L
                                      + model->BSIM4V4wlambda * Inv_W
                                      + model->BSIM4V4plambda * Inv_LW;
                  pParam->BSIM4V4vtl = model->BSIM4V4vtl
                                      + model->BSIM4V4lvtl * Inv_L
                                      + model->BSIM4V4wvtl * Inv_W
                                      + model->BSIM4V4pvtl * Inv_LW;
                  pParam->BSIM4V4xn = model->BSIM4V4xn
                                      + model->BSIM4V4lxn * Inv_L
                                      + model->BSIM4V4wxn * Inv_W
                                      + model->BSIM4V4pxn * Inv_LW;
                  pParam->BSIM4V4vfbsdoff = model->BSIM4V4vfbsdoff
                                      + model->BSIM4V4lvfbsdoff * Inv_L
                                      + model->BSIM4V4wvfbsdoff * Inv_W
                                      + model->BSIM4V4pvfbsdoff * Inv_LW;

		  pParam->BSIM4V4cgsl = model->BSIM4V4cgsl
				    + model->BSIM4V4lcgsl * Inv_L
				    + model->BSIM4V4wcgsl * Inv_W
				    + model->BSIM4V4pcgsl * Inv_LW;
		  pParam->BSIM4V4cgdl = model->BSIM4V4cgdl
				    + model->BSIM4V4lcgdl * Inv_L
				    + model->BSIM4V4wcgdl * Inv_W
				    + model->BSIM4V4pcgdl * Inv_LW;
		  pParam->BSIM4V4ckappas = model->BSIM4V4ckappas
				       + model->BSIM4V4lckappas * Inv_L
				       + model->BSIM4V4wckappas * Inv_W
 				       + model->BSIM4V4pckappas * Inv_LW;
                  pParam->BSIM4V4ckappad = model->BSIM4V4ckappad
                                       + model->BSIM4V4lckappad * Inv_L
                                       + model->BSIM4V4wckappad * Inv_W
                                       + model->BSIM4V4pckappad * Inv_LW;
		  pParam->BSIM4V4cf = model->BSIM4V4cf
				  + model->BSIM4V4lcf * Inv_L
				  + model->BSIM4V4wcf * Inv_W
				  + model->BSIM4V4pcf * Inv_LW;
		  pParam->BSIM4V4clc = model->BSIM4V4clc
				   + model->BSIM4V4lclc * Inv_L
				   + model->BSIM4V4wclc * Inv_W
				   + model->BSIM4V4pclc * Inv_LW;
		  pParam->BSIM4V4cle = model->BSIM4V4cle
				   + model->BSIM4V4lcle * Inv_L
				   + model->BSIM4V4wcle * Inv_W
				   + model->BSIM4V4pcle * Inv_LW;
		  pParam->BSIM4V4vfbcv = model->BSIM4V4vfbcv
				     + model->BSIM4V4lvfbcv * Inv_L
				     + model->BSIM4V4wvfbcv * Inv_W
				     + model->BSIM4V4pvfbcv * Inv_LW;
                  pParam->BSIM4V4acde = model->BSIM4V4acde
                                    + model->BSIM4V4lacde * Inv_L
                                    + model->BSIM4V4wacde * Inv_W
                                    + model->BSIM4V4pacde * Inv_LW;
                  pParam->BSIM4V4moin = model->BSIM4V4moin
                                    + model->BSIM4V4lmoin * Inv_L
                                    + model->BSIM4V4wmoin * Inv_W
                                    + model->BSIM4V4pmoin * Inv_LW;
                  pParam->BSIM4V4noff = model->BSIM4V4noff
                                    + model->BSIM4V4lnoff * Inv_L
                                    + model->BSIM4V4wnoff * Inv_W
                                    + model->BSIM4V4pnoff * Inv_LW;
                  pParam->BSIM4V4voffcv = model->BSIM4V4voffcv
                                      + model->BSIM4V4lvoffcv * Inv_L
                                      + model->BSIM4V4wvoffcv * Inv_W
                                      + model->BSIM4V4pvoffcv * Inv_LW;

                  pParam->BSIM4V4abulkCVfactor = 1.0 + pow((pParam->BSIM4V4clc
					     / pParam->BSIM4V4leffCV),
					     pParam->BSIM4V4cle);

	          T0 = (TRatio - 1.0);

		  PowWeffWr = pow(pParam->BSIM4V4weffCJ * 1.0e6, pParam->BSIM4V4wr) * here->BSIM4V4nf;

	          T1 = T2 = T3 = T4 = 0.0;
	          if(model->BSIM4V4tempMod == 0) {
	          	pParam->BSIM4V4ua = pParam->BSIM4V4ua + pParam->BSIM4V4ua1 * T0;
	          	pParam->BSIM4V4ub = pParam->BSIM4V4ub + pParam->BSIM4V4ub1 * T0;
	          	pParam->BSIM4V4uc = pParam->BSIM4V4uc + pParam->BSIM4V4uc1 * T0;
                  	pParam->BSIM4V4vsattemp = pParam->BSIM4V4vsat - pParam->BSIM4V4at * T0;
		  	T10 = pParam->BSIM4V4prt * T0;
		     if(model->BSIM4V4rdsMod) {
		  	/* External Rd(V) */
		  	T1 = pParam->BSIM4V4rdw + T10;
                  	T2 = model->BSIM4V4rdwmin + T10;
		  	/* External Rs(V) */
		  	T3 = pParam->BSIM4V4rsw + T10;
                  	T4 = model->BSIM4V4rswmin + T10;
                     }
		  	/* Internal Rds(V) in IV */
	          	pParam->BSIM4V4rds0 = (pParam->BSIM4V4rdsw + T10)
				    	* here->BSIM4V4nf / PowWeffWr;
		  	pParam->BSIM4V4rdswmin = (model->BSIM4V4rdswmin + T10)
				       	* here->BSIM4V4nf / PowWeffWr;
                  } else { /* tempMod = 1 */
	          	pParam->BSIM4V4ua = pParam->BSIM4V4ua * (1.0 + pParam->BSIM4V4ua1 * delTemp) ;
	          	pParam->BSIM4V4ub = pParam->BSIM4V4ub * (1.0 + pParam->BSIM4V4ub1 * delTemp);
	          	pParam->BSIM4V4uc = pParam->BSIM4V4uc * (1.0 + pParam->BSIM4V4uc1 * delTemp);
                  	pParam->BSIM4V4vsattemp = pParam->BSIM4V4vsat * (1.0 - pParam->BSIM4V4at * delTemp);
		  	T10 = 1.0 + pParam->BSIM4V4prt * delTemp;
		     if(model->BSIM4V4rdsMod) {
		  	/* External Rd(V) */
		  	T1 = pParam->BSIM4V4rdw * T10;
                  	T2 = model->BSIM4V4rdwmin * T10;
		  	/* External Rs(V) */
		  	T3 = pParam->BSIM4V4rsw * T10;
                  	T4 = model->BSIM4V4rswmin * T10;
                     }
		  	/* Internal Rds(V) in IV */
	          	pParam->BSIM4V4rds0 = pParam->BSIM4V4rdsw * T10 * here->BSIM4V4nf / PowWeffWr;
		  	pParam->BSIM4V4rdswmin = model->BSIM4V4rdswmin * T10 * here->BSIM4V4nf / PowWeffWr;
                  }
		  if (T1 < 0.0)
		  {   T1 = 0.0;
		      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
		  }
		  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
		  pParam->BSIM4V4rd0 = T1 / PowWeffWr;
                  pParam->BSIM4V4rdwmin = T2 / PowWeffWr;
                  if (T3 < 0.0)
                  {   T3 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  if (T4 < 0.0)
                  {   T4 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4V4rs0 = T3 / PowWeffWr;
                  pParam->BSIM4V4rswmin = T4 / PowWeffWr;

                  if (pParam->BSIM4V4u0 > 1.0) 
                      pParam->BSIM4V4u0 = pParam->BSIM4V4u0 / 1.0e4;

                  pParam->BSIM4V4u0temp = pParam->BSIM4V4u0
				      * pow(TRatio, pParam->BSIM4V4ute); 
                  if (pParam->BSIM4V4eu < 0.0)
                  {   pParam->BSIM4V4eu = 0.0;
		      printf("Warning: eu has been negative; reset to 0.0.\n");
		  }

                /* Source End Velocity Limit  */
      	          if((model->BSIM4V4vtlGiven) && (model->BSIM4V4vtl > 0.0) )
            	  {  
                     if(model->BSIM4V4lc < 0.0) pParam->BSIM4V4lc = 0.0;
                     else   pParam->BSIM4V4lc = model->BSIM4V4lc ;
                     T0 = pParam->BSIM4V4leff / (pParam->BSIM4V4xn * pParam->BSIM4V4leff + pParam->BSIM4V4lc);
                     pParam->BSIM4V4tfactor = (1.0 - T0) / (1.0 + T0 );
             	  }

                  pParam->BSIM4V4cgdo = (model->BSIM4V4cgdo + pParam->BSIM4V4cf)
				    * pParam->BSIM4V4weffCV;
                  pParam->BSIM4V4cgso = (model->BSIM4V4cgso + pParam->BSIM4V4cf)
				    * pParam->BSIM4V4weffCV;
                  pParam->BSIM4V4cgbo = model->BSIM4V4cgbo * pParam->BSIM4V4leffCV * here->BSIM4V4nf;

                  if (!model->BSIM4V4ndepGiven && model->BSIM4V4gamma1Given)
                  {   T0 = pParam->BSIM4V4gamma1 * model->BSIM4V4coxe;
                      pParam->BSIM4V4ndep = 3.01248e22 * T0 * T0;
                  }

		  pParam->BSIM4V4phi = Vtm0 * log(pParam->BSIM4V4ndep / ni)
				   + pParam->BSIM4V4phin + 0.4;

	          pParam->BSIM4V4sqrtPhi = sqrt(pParam->BSIM4V4phi);
	          pParam->BSIM4V4phis3 = pParam->BSIM4V4sqrtPhi * pParam->BSIM4V4phi;

                  pParam->BSIM4V4Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM4V4ndep * 1.0e6))
                                     * pParam->BSIM4V4sqrtPhi; 
                  pParam->BSIM4V4sqrtXdep0 = sqrt(pParam->BSIM4V4Xdep0);
                  pParam->BSIM4V4litl = sqrt(3.0 * pParam->BSIM4V4xj
				    * model->BSIM4V4toxe);
                  pParam->BSIM4V4vbi = Vtm0 * log(pParam->BSIM4V4nsd
			           * pParam->BSIM4V4ndep / (ni * ni));

		  if (pParam->BSIM4V4ngate > 0.0)
                  {   pParam->BSIM4V4vfbsd = Vtm0 * log(pParam->BSIM4V4ngate
                                         / pParam->BSIM4V4nsd);
		  }
		  else
		      pParam->BSIM4V4vfbsd = 0.0;

                  pParam->BSIM4V4cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM4V4ndep * 1.0e6 / 2.0
				     / pParam->BSIM4V4phi);

                  pParam->BSIM4V4ToxRatio = exp(pParam->BSIM4V4ntox
					* log(model->BSIM4V4toxref / model->BSIM4V4toxe))
					/ model->BSIM4V4toxe / model->BSIM4V4toxe;
                  pParam->BSIM4V4ToxRatioEdge = exp(pParam->BSIM4V4ntox
                                            * log(model->BSIM4V4toxref
                                            / (model->BSIM4V4toxe * pParam->BSIM4V4poxedge)))
                                            / model->BSIM4V4toxe / model->BSIM4V4toxe
                                            / pParam->BSIM4V4poxedge / pParam->BSIM4V4poxedge;
                  pParam->BSIM4V4Aechvb = (model->BSIM4V4type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4V4Bechvb = (model->BSIM4V4type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4V4AechvbEdge = pParam->BSIM4V4Aechvb * pParam->BSIM4V4weff
					  * pParam->BSIM4V4dlcig * pParam->BSIM4V4ToxRatioEdge;
                  pParam->BSIM4V4BechvbEdge = -pParam->BSIM4V4Bechvb
					  * model->BSIM4V4toxe * pParam->BSIM4V4poxedge;
                  pParam->BSIM4V4Aechvb *= pParam->BSIM4V4weff * pParam->BSIM4V4leff
				       * pParam->BSIM4V4ToxRatio;
                  pParam->BSIM4V4Bechvb *= -model->BSIM4V4toxe;


                  pParam->BSIM4V4mstar = 0.5 + atan(pParam->BSIM4V4minv) / PI;
                  pParam->BSIM4V4voffcbn =  pParam->BSIM4V4voff + model->BSIM4V4voffl / pParam->BSIM4V4leff;

                  pParam->BSIM4V4ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4V4ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4V4acde *= pow((pParam->BSIM4V4ndep / 2.0e16), -0.25);


                  if (model->BSIM4V4k1Given || model->BSIM4V4k2Given)
	          {   if (!model->BSIM4V4k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4V4k1 = 0.53;
                      }
                      if (!model->BSIM4V4k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4V4k2 = -0.0186;
                      }
                      if (model->BSIM4V4nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4V4xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4V4vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4V4gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4V4gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM4V4vbxGiven)
                          pParam->BSIM4V4vbx = pParam->BSIM4V4phi - 7.7348e-4 
                                           * pParam->BSIM4V4ndep
					   * pParam->BSIM4V4xt * pParam->BSIM4V4xt;
	              if (pParam->BSIM4V4vbx > 0.0)
		          pParam->BSIM4V4vbx = -pParam->BSIM4V4vbx;
	              if (pParam->BSIM4V4vbm > 0.0)
                          pParam->BSIM4V4vbm = -pParam->BSIM4V4vbm;
           
                      if (!model->BSIM4V4gamma1Given)
                          pParam->BSIM4V4gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM4V4ndep)
                                              / model->BSIM4V4coxe;
                      if (!model->BSIM4V4gamma2Given)
                          pParam->BSIM4V4gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM4V4nsub)
                                              / model->BSIM4V4coxe;

                      T0 = pParam->BSIM4V4gamma1 - pParam->BSIM4V4gamma2;
                      T1 = sqrt(pParam->BSIM4V4phi - pParam->BSIM4V4vbx)
			 - pParam->BSIM4V4sqrtPhi;
                      T2 = sqrt(pParam->BSIM4V4phi * (pParam->BSIM4V4phi
			 - pParam->BSIM4V4vbm)) - pParam->BSIM4V4phi;
                      pParam->BSIM4V4k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4V4vbm);
                      pParam->BSIM4V4k1 = pParam->BSIM4V4gamma2 - 2.0
				      * pParam->BSIM4V4k2 * sqrt(pParam->BSIM4V4phi
				      - pParam->BSIM4V4vbm);
                  }
 
		  if (pParam->BSIM4V4k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM4V4k1 / pParam->BSIM4V4k2;
                      pParam->BSIM4V4vbsc = 0.9 * (pParam->BSIM4V4phi - T0 * T0);
		      if (pParam->BSIM4V4vbsc > -3.0)
		          pParam->BSIM4V4vbsc = -3.0;
		      else if (pParam->BSIM4V4vbsc < -30.0)
		          pParam->BSIM4V4vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM4V4vbsc = -30.0;
		  }
		  if (pParam->BSIM4V4vbsc > pParam->BSIM4V4vbm)
		      pParam->BSIM4V4vbsc = pParam->BSIM4V4vbm;

                  if (!model->BSIM4V4vfbGiven)
                  {   if (model->BSIM4V4vth0Given)
                      {   pParam->BSIM4V4vfb = model->BSIM4V4type * pParam->BSIM4V4vth0
                                           - pParam->BSIM4V4phi - pParam->BSIM4V4k1
                                           * pParam->BSIM4V4sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4V4vfb = -1.0;
                      }
                  }
                   if (!model->BSIM4V4vth0Given)
                  {   pParam->BSIM4V4vth0 = model->BSIM4V4type * (pParam->BSIM4V4vfb
                                        + pParam->BSIM4V4phi + pParam->BSIM4V4k1
                                        * pParam->BSIM4V4sqrtPhi);
                  }

                  pParam->BSIM4V4k1ox = pParam->BSIM4V4k1 * model->BSIM4V4toxe
                                    / model->BSIM4V4toxm;
                  pParam->BSIM4V4k2ox = pParam->BSIM4V4k2 * model->BSIM4V4toxe
                                    / model->BSIM4V4toxm;

		  T3 = model->BSIM4V4type * pParam->BSIM4V4vth0
		     - pParam->BSIM4V4vfb - pParam->BSIM4V4phi;
		  T4 = T3 + T3;
		  T5 = 2.5 * T3;
                  pParam->BSIM4V4vtfbphi1 = (model->BSIM4V4type == NMOS) ? T4 : T5; 
		  if (pParam->BSIM4V4vtfbphi1 < 0.0)
		      pParam->BSIM4V4vtfbphi1 = 0.0;

                  pParam->BSIM4V4vtfbphi2 = 4.0 * T3;
                  if (pParam->BSIM4V4vtfbphi2 < 0.0)
                      pParam->BSIM4V4vtfbphi2 = 0.0;

                  tmp = sqrt(EPSSI / (model->BSIM4V4epsrox * EPS0)
                      * model->BSIM4V4toxe * pParam->BSIM4V4Xdep0);
          	  T0 = pParam->BSIM4V4dsub * pParam->BSIM4V4leff / tmp;
                  if (T0 < EXP_THRESHOLD)
          	  {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
              	      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4V4theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4V4theta0vb0 = 1.0 / (MAX_EXP - 2.0);

 	          T0 = pParam->BSIM4V4drout * pParam->BSIM4V4leff / tmp;
        	  if (T0 < EXP_THRESHOLD)
       	          {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4V4thetaRout = pParam->BSIM4V4pdibl1 * T5
                                         + pParam->BSIM4V4pdibl2;

                  tmp = sqrt(pParam->BSIM4V4Xdep0);
                  tmp1 = pParam->BSIM4V4vbi - pParam->BSIM4V4phi;
                  tmp2 = model->BSIM4V4factor1 * tmp;

                  T0 = pParam->BSIM4V4dvt1w * pParam->BSIM4V4weff
                     * pParam->BSIM4V4leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4V4dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4V4dvt1 * pParam->BSIM4V4leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  } 
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4V4dvt0 * T9 * tmp1;

                  T4 = model->BSIM4V4toxe * pParam->BSIM4V4phi
                     / (pParam->BSIM4V4weff + pParam->BSIM4V4w0);

                  T0 = sqrt(1.0 + pParam->BSIM4V4lpe0 / pParam->BSIM4V4leff);
                  T5 = pParam->BSIM4V4k1ox * (T0 - 1.0) * pParam->BSIM4V4sqrtPhi
                     + (pParam->BSIM4V4kt1 + pParam->BSIM4V4kt1l / pParam->BSIM4V4leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM4V4type * pParam->BSIM4V4vth0
                       - T8 - T9 + pParam->BSIM4V4k3 * T4 + T5;
                  pParam->BSIM4V4vfbzb = tmp3 - pParam->BSIM4V4phi - pParam->BSIM4V4k1
                                     * pParam->BSIM4V4sqrtPhi; /* End of vfbzb */

		  /* stress effect */
                  T0 = pow(Lnew, model->BSIM4V4llodku0);
		  W_tmp = Wnew + model->BSIM4V4wlod;
                  T1 = pow(W_tmp, model->BSIM4V4wlodku0);
                  tmp1 = model->BSIM4V4lku0 / T0 + model->BSIM4V4wku0 / T1
                         + model->BSIM4V4pku0 / (T0 * T1);
                  pParam->BSIM4V4ku0 = 1.0 + tmp1;

                  T0 = pow(Lnew, model->BSIM4V4llodvth);
                  T1 = pow(W_tmp, model->BSIM4V4wlodvth);
                  tmp1 = model->BSIM4V4lkvth0 / T0 + model->BSIM4V4wkvth0 / T1
                       + model->BSIM4V4pkvth0 / (T0 * T1);
                  pParam->BSIM4V4kvth0 = 1.0 + tmp1;
		  pParam->BSIM4V4kvth0 = sqrt(pParam->BSIM4V4kvth0*pParam->BSIM4V4kvth0 + DELTA);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM4V4ku0temp = pParam->BSIM4V4ku0 * (1.0 + model->BSIM4V4tku0 *T0) + DELTA;

                  Inv_saref = 1.0/(model->BSIM4V4saref + 0.5*Ldrn);
                  Inv_sbref = 1.0/(model->BSIM4V4sbref + 0.5*Ldrn);
		  pParam->BSIM4V4inv_od_ref = Inv_saref + Inv_sbref;
		  pParam->BSIM4V4rho_ref = model->BSIM4V4ku0 / pParam->BSIM4V4ku0temp * pParam->BSIM4V4inv_od_ref;

              } /* End of SizeNotFound */

              /*  stress effect */
              if( (here->BSIM4V4sa > 0.0) && (here->BSIM4V4sb > 0.0) && 
               	 ((here->BSIM4V4nf == 1.0) || ((here->BSIM4V4nf > 1.0) && (here->BSIM4V4sd > 0.0))) )
	      {	  Inv_sa = 0;
              	  Inv_sb = 0;
	    	  
	    	  if (model->BSIM4V4wlod < 0.0)
	          {   fprintf(stderr, "Warning: WLOD = %g is less than 0. Set to 0.0\n",model->BSIM4V4wlod);
               	      model->BSIM4V4wlod = 0.0;
	          }
	          if (model->BSIM4V4kvsat < -1.0 )
	          {   fprintf(stderr, "Warning: KVSAT = %g is too small; Reset to -1.0.\n",model->BSIM4V4kvsat);
	       	      model->BSIM4V4kvsat = -1.0;
            	  }
            	  if (model->BSIM4V4kvsat > 1.0)
            	  {   fprintf(stderr, "Warning: KVSAT = %g is too big; Reset to 1.0.\n",model->BSIM4V4kvsat);
           	      model->BSIM4V4kvsat = 1.0;
            	  }
              	  
	      	  for(i = 0; i < here->BSIM4V4nf; i++){
                   	T0 = 1.0 / here->BSIM4V4nf / (here->BSIM4V4sa + 0.5*Ldrn + i * (here->BSIM4V4sd +Ldrn));
                    	T1 = 1.0 / here->BSIM4V4nf / (here->BSIM4V4sb + 0.5*Ldrn + i * (here->BSIM4V4sd +Ldrn));
                   	Inv_sa += T0;
                    	Inv_sb += T1;
                  }
                  Inv_ODeff = Inv_sa + Inv_sb; 
                  rho = model->BSIM4V4ku0 / pParam->BSIM4V4ku0temp * Inv_ODeff;
                  T0 = (1.0 + rho)/(1.0 + pParam->BSIM4V4rho_ref);
                  here->BSIM4V4u0temp = pParam->BSIM4V4u0temp * T0;

                  T1 = (1.0 + model->BSIM4V4kvsat * rho)/(1.0 + model->BSIM4V4kvsat * pParam->BSIM4V4rho_ref);
                  here->BSIM4V4vsattemp = pParam->BSIM4V4vsattemp * T1;

		  OD_offset = Inv_ODeff - pParam->BSIM4V4inv_od_ref;
		  dvth0_lod = model->BSIM4V4kvth0 / pParam->BSIM4V4kvth0 * OD_offset;
                  dk2_lod = model->BSIM4V4stk2 / pow(pParam->BSIM4V4kvth0, model->BSIM4V4lodk2) *
                                   OD_offset;
                  deta0_lod = model->BSIM4V4steta0 / pow(pParam->BSIM4V4kvth0, model->BSIM4V4lodeta0) *
                                     OD_offset;
		  here->BSIM4V4vth0 = pParam->BSIM4V4vth0 + dvth0_lod;

	          if (!model->BSIM4V4vfbGiven && !model->BSIM4V4vth0Given)
                       here->BSIM4V4vfb = -1.0;
                  else  
                       here->BSIM4V4vfb = pParam->BSIM4V4vfb + model->BSIM4V4type * dvth0_lod;
                  here->BSIM4V4vfbzb = pParam->BSIM4V4vfbzb + model->BSIM4V4type * dvth0_lod;

                  T3 = model->BSIM4V4type * here->BSIM4V4vth0
                     - here->BSIM4V4vfb - pParam->BSIM4V4phi;
                  T4 = T3 + T3;
                  T5 = 2.5 * T3;
                  here->BSIM4V4vtfbphi1 = (model->BSIM4V4type == NMOS) ? T4 : T5;
                  if (here->BSIM4V4vtfbphi1 < 0.0)
                      here->BSIM4V4vtfbphi1 = 0.0;

                  here->BSIM4V4vtfbphi2 = 4.0 * T3;
                  if (here->BSIM4V4vtfbphi2 < 0.0)
                      here->BSIM4V4vtfbphi2 = 0.0;
		  
		  here->BSIM4V4k2 = pParam->BSIM4V4k2 + dk2_lod;
                  if (here->BSIM4V4k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM4V4k1 / here->BSIM4V4k2;
                      here->BSIM4V4vbsc = 0.9 * (pParam->BSIM4V4phi - T0 * T0);
                      if (here->BSIM4V4vbsc > -3.0)
                          here->BSIM4V4vbsc = -3.0;
                      else if (here->BSIM4V4vbsc < -30.0)
                          here->BSIM4V4vbsc = -30.0;
                  }
                  else
                      here->BSIM4V4vbsc = -30.0;
                  if (here->BSIM4V4vbsc > pParam->BSIM4V4vbm)
                      here->BSIM4V4vbsc = pParam->BSIM4V4vbm;
		  here->BSIM4V4k2ox = here->BSIM4V4k2 * model->BSIM4V4toxe
                                    / model->BSIM4V4toxm;

                  here->BSIM4V4eta0 = pParam->BSIM4V4eta0 + deta0_lod;
	       } else {
		      here->BSIM4V4u0temp = pParam->BSIM4V4u0temp;
                      here->BSIM4V4vth0 = pParam->BSIM4V4vth0;
                      here->BSIM4V4vsattemp = pParam->BSIM4V4vsattemp;
                      here->BSIM4V4vfb = pParam->BSIM4V4vfb;
                      here->BSIM4V4vfbzb = pParam->BSIM4V4vfbzb;
		      here->BSIM4V4vtfbphi1 = pParam->BSIM4V4vtfbphi1;
		      here->BSIM4V4vtfbphi2 = pParam->BSIM4V4vtfbphi2;
                      here->BSIM4V4k2 = pParam->BSIM4V4k2;
                      here->BSIM4V4vbsc = pParam->BSIM4V4vbsc;
                      here->BSIM4V4k2ox = pParam->BSIM4V4k2ox;
                      here->BSIM4V4eta0 = pParam->BSIM4V4eta0;
              }
                   
              here->BSIM4V4cgso = pParam->BSIM4V4cgso;
              here->BSIM4V4cgdo = pParam->BSIM4V4cgdo;
              
              if (here->BSIM4V4rbodyMod)
              {   if (here->BSIM4V4rbdb < 1.0e-3)
                      here->BSIM4V4grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4V4grbdb = model->BSIM4V4gbmin + 1.0 / here->BSIM4V4rbdb;
                  if (here->BSIM4V4rbpb < 1.0e-3)
                      here->BSIM4V4grbpb = 1.0e3;
                  else
                      here->BSIM4V4grbpb = model->BSIM4V4gbmin + 1.0 / here->BSIM4V4rbpb;
                  if (here->BSIM4V4rbps < 1.0e-3)
                      here->BSIM4V4grbps = 1.0e3;
                  else
                      here->BSIM4V4grbps = model->BSIM4V4gbmin + 1.0 / here->BSIM4V4rbps;
                  if (here->BSIM4V4rbsb < 1.0e-3)
                      here->BSIM4V4grbsb = 1.0e3;
                  else
                      here->BSIM4V4grbsb = model->BSIM4V4gbmin + 1.0 / here->BSIM4V4rbsb;
                  if (here->BSIM4V4rbpd < 1.0e-3)
                      here->BSIM4V4grbpd = 1.0e3;
                  else
                      here->BSIM4V4grbpd = model->BSIM4V4gbmin + 1.0 / here->BSIM4V4rbpd;
              }


              /* 
               * Process geomertry dependent parasitics
	       */

              here->BSIM4V4grgeltd = model->BSIM4V4rshg * (model->BSIM4V4xgw
                      + pParam->BSIM4V4weffCJ / 3.0 / model->BSIM4V4ngcon) /
                      (model->BSIM4V4ngcon * here->BSIM4V4nf *
                      (Lnew - model->BSIM4V4xgl));
              if (here->BSIM4V4grgeltd > 0.0)
                  here->BSIM4V4grgeltd = 1.0 / here->BSIM4V4grgeltd;
              else
              {   here->BSIM4V4grgeltd = 1.0e3; /* mho */
		  if (here->BSIM4V4rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

	      DMCGeff = model->BSIM4V4dmcg - model->BSIM4V4dmcgt;
              DMCIeff = model->BSIM4V4dmci;
              DMDGeff = model->BSIM4V4dmdg - model->BSIM4V4dmcgt;

	      if (here->BSIM4V4sourcePerimeterGiven)
	      {   if (model->BSIM4V4perMod == 0)
	              here->BSIM4V4Pseff = here->BSIM4V4sourcePerimeter;
		  else
		      here->BSIM4V4Pseff = here->BSIM4V4sourcePerimeter 
				       - pParam->BSIM4V4weffCJ * here->BSIM4V4nf;
	      }
	      else
	          BSIM4V4PAeffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod, here->BSIM4V4min, 
                                    pParam->BSIM4V4weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &(here->BSIM4V4Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4V4drainPerimeterGiven)
              {   if (model->BSIM4V4perMod == 0)
                      here->BSIM4V4Pdeff = here->BSIM4V4drainPerimeter;
                  else
                      here->BSIM4V4Pdeff = here->BSIM4V4drainPerimeter 
				       - pParam->BSIM4V4weffCJ * here->BSIM4V4nf;
              }
              else
                  BSIM4V4PAeffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod, here->BSIM4V4min,
                                    pParam->BSIM4V4weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &(here->BSIM4V4Pdeff), &dumAs, &dumAd);

              if (here->BSIM4V4sourceAreaGiven)
                  here->BSIM4V4Aseff = here->BSIM4V4sourceArea;
              else
                  BSIM4V4PAeffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod, here->BSIM4V4min,
                                    pParam->BSIM4V4weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &(here->BSIM4V4Aseff), &dumAd);

              if (here->BSIM4V4drainAreaGiven)
                  here->BSIM4V4Adeff = here->BSIM4V4drainArea;
              else
                  BSIM4V4PAeffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod, here->BSIM4V4min,
                                    pParam->BSIM4V4weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &dumAs, &(here->BSIM4V4Adeff));

	      /* Processing S/D resistance and conductance below */
              if(here->BSIM4V4sNodePrime != here->BSIM4V4sNode)
              {
                 here->BSIM4V4sourceConductance = 0.0;
                 if(here->BSIM4V4sourceSquaresGiven)
                 {
                    here->BSIM4V4sourceConductance = model->BSIM4V4sheetResistance
                                               * here->BSIM4V4sourceSquares;
                 } else if (here->BSIM4V4rgeoMod > 0)
                 {
                    BSIM4V4RdseffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod,
                      here->BSIM4V4rgeoMod, here->BSIM4V4min,
                      pParam->BSIM4V4weffCJ, model->BSIM4V4sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4V4sourceConductance));
                 } else
                 {
                    here->BSIM4V4sourceConductance = 0.0;
                 }

                 if (here->BSIM4V4sourceConductance > 0.0)
                     here->BSIM4V4sourceConductance = 1.0
                                            / here->BSIM4V4sourceConductance;
                 else
                 {
                     here->BSIM4V4sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4V4sourceConductance = 0.0;
              }

              if(here->BSIM4V4dNodePrime != here->BSIM4V4dNode)
              {
                 here->BSIM4V4drainConductance = 0.0;
                 if(here->BSIM4V4drainSquaresGiven)
                 {
                    here->BSIM4V4drainConductance = model->BSIM4V4sheetResistance
                                              * here->BSIM4V4drainSquares;
                 } else if (here->BSIM4V4rgeoMod > 0)
                 {
                    BSIM4V4RdseffGeo(here->BSIM4V4nf, here->BSIM4V4geoMod,
                      here->BSIM4V4rgeoMod, here->BSIM4V4min,
                      pParam->BSIM4V4weffCJ, model->BSIM4V4sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4V4drainConductance));
                 } else
                 {
                    here->BSIM4V4drainConductance = 0.0;
                 }

                 if (here->BSIM4V4drainConductance > 0.0)
                     here->BSIM4V4drainConductance = 1.0
                                           / here->BSIM4V4drainConductance;
                 else
                 {
                     here->BSIM4V4drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4V4drainConductance = 0.0;
              }
           
               /* End of Rsd processing */


              Nvtms = model->BSIM4V4vtm * model->BSIM4V4SjctEmissionCoeff;
              if ((here->BSIM4V4Aseff <= 0.0) && (here->BSIM4V4Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4V4Aseff * model->BSIM4V4SjctTempSatCurDensity
				   + here->BSIM4V4Pseff * model->BSIM4V4SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4V4weffCJ * here->BSIM4V4nf
                                   * model->BSIM4V4SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4V4dioMod)
                  {   case 0:
			  if ((model->BSIM4V4bvs / Nvtms) > EXP_THRESHOLD)
			      here->BSIM4V4XExpBVS = model->BSIM4V4xjbvs * MIN_EXP;
			  else
	                      here->BSIM4V4XExpBVS = model->BSIM4V4xjbvs * exp(-model->BSIM4V4bvs / Nvtms);	
		          break;
                      case 1:
                          BSIM4V4DioIjthVjmEval(Nvtms, model->BSIM4V4ijthsfwd, SourceSatCurrent, 
			                      0.0, &(here->BSIM4V4vjsmFwd));
                          here->BSIM4V4IVjsmFwd = SourceSatCurrent * exp(here->BSIM4V4vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4V4bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4V4XExpBVS = model->BSIM4V4xjbvs * MIN_EXP;
			      tmp = MIN_EXP;
			  }
                          else
			  {   here->BSIM4V4XExpBVS = exp(-model->BSIM4V4bvs / Nvtms);
			      tmp = here->BSIM4V4XExpBVS;
		              here->BSIM4V4XExpBVS *= model->BSIM4V4xjbvs;	
			  }

                          BSIM4V4DioIjthVjmEval(Nvtms, model->BSIM4V4ijthsfwd, SourceSatCurrent, 
                               		      here->BSIM4V4XExpBVS, &(here->BSIM4V4vjsmFwd));
		          T0 = exp(here->BSIM4V4vjsmFwd / Nvtms);
                          here->BSIM4V4IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4V4XExpBVS / T0
			  		      + here->BSIM4V4XExpBVS - 1.0);
		          here->BSIM4V4SslpFwd = SourceSatCurrent
					       * (T0 + here->BSIM4V4XExpBVS / T0) / Nvtms;

			  T2 = model->BSIM4V4ijthsrev / SourceSatCurrent;
			  if (T2 < 1.0)
			  {   T2 = 10.0;
			      fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
			  } 
                          here->BSIM4V4vjsmRev = -model->BSIM4V4bvs
					     - Nvtms * log((T2 - 1.0) / model->BSIM4V4xjbvs);
			  T1 = model->BSIM4V4xjbvs * exp(-(model->BSIM4V4bvs
			     + here->BSIM4V4vjsmRev) / Nvtms);
			  here->BSIM4V4IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4V4SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4V4dioMod);
                  }
              }

              Nvtmd = model->BSIM4V4vtm * model->BSIM4V4DjctEmissionCoeff;
	      if ((here->BSIM4V4Adeff <= 0.0) && (here->BSIM4V4Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4V4Adeff * model->BSIM4V4DjctTempSatCurDensity
				  + here->BSIM4V4Pdeff * model->BSIM4V4DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4V4weffCJ * here->BSIM4V4nf
                                  * model->BSIM4V4DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4V4dioMod)
                  {   case 0:
                          if ((model->BSIM4V4bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4V4XExpBVD = model->BSIM4V4xjbvd * MIN_EXP;
                          else
                          here->BSIM4V4XExpBVD = model->BSIM4V4xjbvd * exp(-model->BSIM4V4bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4V4DioIjthVjmEval(Nvtmd, model->BSIM4V4ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4V4vjdmFwd));
                          here->BSIM4V4IVjdmFwd = DrainSatCurrent * exp(here->BSIM4V4vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4V4bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4V4XExpBVD = model->BSIM4V4xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4V4XExpBVD = exp(-model->BSIM4V4bvd / Nvtmd);
                              tmp = here->BSIM4V4XExpBVD;
                              here->BSIM4V4XExpBVD *= model->BSIM4V4xjbvd;
                          }

                          BSIM4V4DioIjthVjmEval(Nvtmd, model->BSIM4V4ijthdfwd, DrainSatCurrent,
                                              here->BSIM4V4XExpBVD, &(here->BSIM4V4vjdmFwd));
                          T0 = exp(here->BSIM4V4vjdmFwd / Nvtmd);
                          here->BSIM4V4IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4V4XExpBVD / T0
                                              + here->BSIM4V4XExpBVD - 1.0);
                          here->BSIM4V4DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4V4XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4V4ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0) 
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4V4vjdmRev = -model->BSIM4V4bvd
                                             - Nvtmd * log((T2 - 1.0) / model->BSIM4V4xjbvd); /* bugfix */
                          T1 = model->BSIM4V4xjbvd * exp(-(model->BSIM4V4bvd
                             + here->BSIM4V4vjdmRev) / Nvtmd);
                          here->BSIM4V4IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4V4DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4V4dioMod);
                  }
              }

		/* GEDL current reverse bias */
	        T0 = (TRatio - 1.0);
                model->BSIM4V4njtstemp = model->BSIM4V4njts * (1.0 + model->BSIM4V4tnjts * T0);
                model->BSIM4V4njtsswtemp = model->BSIM4V4njtssw * (1.0 + model->BSIM4V4tnjtssw * T0);
                model->BSIM4V4njtsswgtemp = model->BSIM4V4njtsswg * (1.0 + model->BSIM4V4tnjtsswg * T0);
                T7 = Eg0 / model->BSIM4V4vtm * T0;
                T9 = model->BSIM4V4xtss * T7;
                DEXP(T9, T1);
                T9 = model->BSIM4V4xtsd * T7;
                DEXP(T9, T2);
                T9 = model->BSIM4V4xtssws * T7;
                DEXP(T9, T3);
                T9 = model->BSIM4V4xtsswd * T7;
                DEXP(T9, T4);
                T9 = model->BSIM4V4xtsswgs * T7;
                DEXP(T9, T5);
                T9 = model->BSIM4V4xtsswgd * T7;
                DEXP(T9, T6);

		T10 = pParam->BSIM4V4weffCJ * here->BSIM4V4nf;
		here->BSIM4V4SjctTempRevSatCur = T1 * here->BSIM4V4Aseff * model->BSIM4V4jtss;
		here->BSIM4V4DjctTempRevSatCur = T2 * here->BSIM4V4Adeff * model->BSIM4V4jtsd;
		here->BSIM4V4SswTempRevSatCur = T3 * here->BSIM4V4Pseff * model->BSIM4V4jtssws;
		here->BSIM4V4DswTempRevSatCur = T4 * here->BSIM4V4Pdeff * model->BSIM4V4jtsswd;
		here->BSIM4V4SswgTempRevSatCur = T5 * T10 * model->BSIM4V4jtsswgs;
		here->BSIM4V4DswgTempRevSatCur = T6 * T10 * model->BSIM4V4jtsswgd;
                

              if (BSIM4V4checkModel(model, here, ckt))
              {   IFuid namarray[2];
                  namarray[0] = model->BSIM4V4modName;
                  namarray[1] = here->BSIM4V4name;
                  (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM4.4.0 parameter checking for %s in model %s", namarray);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
