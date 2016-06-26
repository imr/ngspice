/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4temp.c of BSIM4.0.0.
 * Authors: Weidong Liu, Xiaodong Jin, Kanyu M. Cao, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5 
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define Charge_q 1.60219e-19

static int
BSIM4v0DioIjthVjmEval(Nvtm, Ijth, Isb, XExpBV, Vjm)
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
BSIM4v0temp(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
register BSIM4v0model *model = (BSIM4v0model*) inModel;
register BSIM4v0instance *here;
struct bsim4v0SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T8, T9, Ldrn, Wdrn;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
int Size_Not_Found;

    /*  loop through all the BSIM4v0 device models */
    for (; model != NULL; model = model->BSIM4v0nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v0SbulkJctPotential < 0.1)  
	 {   model->BSIM4v0SbulkJctPotential = 0.1;
	     fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
	 }
         if (model->BSIM4v0SsidewallJctPotential < 0.1)
	 {   model->BSIM4v0SsidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
	 }
         if (model->BSIM4v0SGatesidewallJctPotential < 0.1)
	 {   model->BSIM4v0SGatesidewallJctPotential = 0.1;
	     fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
	 }

         if (model->BSIM4v0DbulkJctPotential < 0.1) 
         {   model->BSIM4v0DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v0DsidewallJctPotential < 0.1)
         {   model->BSIM4v0DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v0DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v0DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4v0toxeGiven) && (model->BSIM4v0toxpGiven) && (model->BSIM4v0dtoxGiven)
             && (model->BSIM4v0toxe != (model->BSIM4v0toxp + model->BSIM4v0dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
	 else if ((model->BSIM4v0toxeGiven) && (!model->BSIM4v0toxpGiven))
	     model->BSIM4v0toxp = model->BSIM4v0toxe - model->BSIM4v0dtox;
	 else if ((!model->BSIM4v0toxeGiven) && (model->BSIM4v0toxpGiven))
             model->BSIM4v0toxe = model->BSIM4v0toxp + model->BSIM4v0dtox;

         model->BSIM4v0coxe = model->BSIM4v0epsrox * EPS0 / model->BSIM4v0toxe;
         model->BSIM4v0coxp = model->BSIM4v0epsrox * EPS0 / model->BSIM4v0toxp;

         if (!model->BSIM4v0cgdoGiven)
         {   if (model->BSIM4v0dlcGiven && (model->BSIM4v0dlc > 0.0))
                 model->BSIM4v0cgdo = model->BSIM4v0dlc * model->BSIM4v0coxe
                                  - model->BSIM4v0cgdl ;
             else
                 model->BSIM4v0cgdo = 0.6 * model->BSIM4v0xj * model->BSIM4v0coxe;
         }
         if (!model->BSIM4v0cgsoGiven)
         {   if (model->BSIM4v0dlcGiven && (model->BSIM4v0dlc > 0.0))
                 model->BSIM4v0cgso = model->BSIM4v0dlc * model->BSIM4v0coxe
                                  - model->BSIM4v0cgsl ;
             else
                 model->BSIM4v0cgso = 0.6 * model->BSIM4v0xj * model->BSIM4v0coxe;
         }
         if (!model->BSIM4v0cgboGiven)
             model->BSIM4v0cgbo = 2.0 * model->BSIM4v0dwc * model->BSIM4v0coxe;

         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

	 Tnom = model->BSIM4v0tnom;
	 TRatio = Temp / Tnom;

	 model->BSIM4v0vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v0factor1 = sqrt(EPSSI / (model->BSIM4v0epsrox * EPS0)
                             * model->BSIM4v0toxe);

         Vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4v0vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
	 if (Temp != Tnom)
	 {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v0vtm;
	     T1 = log(Temp / Tnom);
	     T2 = T0 + model->BSIM4v0SjctTempExponent * T1;
	     T3 = exp(T2 / model->BSIM4v0SjctEmissionCoeff);
	     model->BSIM4v0SjctTempSatCurDensity = model->BSIM4v0SjctSatCurDensity
					       * T3;
	     model->BSIM4v0SjctSidewallTempSatCurDensity
			 = model->BSIM4v0SjctSidewallSatCurDensity * T3;
             model->BSIM4v0SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v0SjctGateSidewallSatCurDensity * T3;

	     T2 = T0 + model->BSIM4v0DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v0DjctEmissionCoeff);
             model->BSIM4v0DjctTempSatCurDensity = model->BSIM4v0DjctSatCurDensity
                                               * T3;
             model->BSIM4v0DjctSidewallTempSatCurDensity
                         = model->BSIM4v0DjctSidewallSatCurDensity * T3;
             model->BSIM4v0DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v0DjctGateSidewallSatCurDensity * T3;
	 }
	 else
	 {   model->BSIM4v0SjctTempSatCurDensity = model->BSIM4v0SjctSatCurDensity;
	     model->BSIM4v0SjctSidewallTempSatCurDensity
			= model->BSIM4v0SjctSidewallSatCurDensity;
             model->BSIM4v0SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v0SjctGateSidewallSatCurDensity;
             model->BSIM4v0DjctTempSatCurDensity = model->BSIM4v0DjctSatCurDensity;
             model->BSIM4v0DjctSidewallTempSatCurDensity
                        = model->BSIM4v0DjctSidewallSatCurDensity;
             model->BSIM4v0DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v0DjctGateSidewallSatCurDensity;
	 }

	 if (model->BSIM4v0SjctTempSatCurDensity < 0.0)
	     model->BSIM4v0SjctTempSatCurDensity = 0.0;
	 if (model->BSIM4v0SjctSidewallTempSatCurDensity < 0.0)
	     model->BSIM4v0SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v0SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v0SjctGateSidewallTempSatCurDensity = 0.0;

         if (model->BSIM4v0DjctTempSatCurDensity < 0.0)
             model->BSIM4v0DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v0DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v0DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v0DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v0DjctGateSidewallTempSatCurDensity = 0.0;

	 /* Temperature dependence of D/B and S/B diode capacitance begins */
	 delTemp = ckt->CKTtemp - model->BSIM4v0tnom;
	 T0 = model->BSIM4v0tcj * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v0SunitAreaJctCap *= 1.0 + T0;
             model->BSIM4v0DunitAreaJctCap *= 1.0 + T0;
	 }
	 else
	 {   if (model->BSIM4v0SunitAreaJctCap > 0.0)
	     {   model->BSIM4v0SunitAreaJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
	     if (model->BSIM4v0DunitAreaJctCap > 0.0)
             {   model->BSIM4v0DunitAreaJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
	 }
         T0 = model->BSIM4v0tcjsw * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v0SunitLengthSidewallJctCap *= 1.0 + T0;
             model->BSIM4v0DunitLengthSidewallJctCap *= 1.0 + T0;
	 }
	 else
	 {   if (model->BSIM4v0SunitLengthSidewallJctCap > 0.0)
	     {   model->BSIM4v0SunitLengthSidewallJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
	     }
	     if (model->BSIM4v0DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v0DunitLengthSidewallJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }	
	 }
         T0 = model->BSIM4v0tcjswg * delTemp;
	 if (T0 >= -1.0)
	 {   model->BSIM4v0SunitLengthGateSidewallJctCap *= 1.0 + T0;
             model->BSIM4v0DunitLengthGateSidewallJctCap *= 1.0 + T0;
	 }
	 else
	 {   if (model->BSIM4v0SunitLengthGateSidewallJctCap > 0.0)
	     {   model->BSIM4v0SunitLengthGateSidewallJctCap = 0.0;
	         fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
	     }
	     if (model->BSIM4v0DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v0DunitLengthGateSidewallJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
	 }

         model->BSIM4v0PhiBS = model->BSIM4v0SbulkJctPotential
			   - model->BSIM4v0tpb * delTemp;
         if (model->BSIM4v0PhiBS < 0.01)
	 {   model->BSIM4v0PhiBS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
	 }
         model->BSIM4v0PhiBD = model->BSIM4v0DbulkJctPotential
                           - model->BSIM4v0tpb * delTemp;
         if (model->BSIM4v0PhiBD < 0.01)
         {   model->BSIM4v0PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v0PhiBSWS = model->BSIM4v0SsidewallJctPotential
                             - model->BSIM4v0tpbsw * delTemp;
         if (model->BSIM4v0PhiBSWS <= 0.01)
	 {   model->BSIM4v0PhiBSWS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
	 }
         model->BSIM4v0PhiBSWD = model->BSIM4v0DsidewallJctPotential
                             - model->BSIM4v0tpbsw * delTemp;
         if (model->BSIM4v0PhiBSWD <= 0.01)
         {   model->BSIM4v0PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

	 model->BSIM4v0PhiBSWGS = model->BSIM4v0SGatesidewallJctPotential
                              - model->BSIM4v0tpbswg * delTemp;
         if (model->BSIM4v0PhiBSWGS <= 0.01)
	 {   model->BSIM4v0PhiBSWGS = 0.01;
	     fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
	 }
         model->BSIM4v0PhiBSWGD = model->BSIM4v0DGatesidewallJctPotential
                              - model->BSIM4v0tpbswg * delTemp;
         if (model->BSIM4v0PhiBSWGD <= 0.01)
         {   model->BSIM4v0PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v0ijthdfwd <= 0.0)
         {   model->BSIM4v0ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v0ijthdfwd);
         }
         if (model->BSIM4v0ijthsfwd <= 0.0)
         {   model->BSIM4v0ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v0ijthsfwd);
         }
	 if (model->BSIM4v0ijthdrev <= 0.0)
         {   model->BSIM4v0ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v0ijthdrev);
         }
         if (model->BSIM4v0ijthsrev <= 0.0)
         {   model->BSIM4v0ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v0ijthsrev);
         }

         if ((model->BSIM4v0xjbvd <= 0.0) && (model->BSIM4v0dioMod == 2))
         {   model->BSIM4v0xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v0xjbvd);
         }
         else if ((model->BSIM4v0xjbvd < 0.0) && (model->BSIM4v0dioMod == 0))
         {   model->BSIM4v0xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v0xjbvd);
         }

         if (model->BSIM4v0bvd <= 0.0)
         {   model->BSIM4v0bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v0bvd);
         }

         if ((model->BSIM4v0xjbvs <= 0.0) && (model->BSIM4v0dioMod == 2))
         {   model->BSIM4v0xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v0xjbvs);
         }
         else if ((model->BSIM4v0xjbvs < 0.0) && (model->BSIM4v0dioMod == 0))
         {   model->BSIM4v0xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v0xjbvs);
         }

         if (model->BSIM4v0bvs <= 0.0)
         {   model->BSIM4v0bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v0bvs);
         }


         /* loop through all the instances of the model */
         for (here = model->BSIM4v0instances; here != NULL;
              here = here->BSIM4v0nextInstance) 
	 {    pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM4v0l == pSizeDependParamKnot->Length)
		      && (here->BSIM4v0w == pSizeDependParamKnot->Width)
		      && (here->BSIM4v0nf == pSizeDependParamKnot->NFinger))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      if (Size_Not_Found)
	      {   pParam = TMALLOC(struct bsim4v0SizeDependParam, 1);
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v0l;
                  pParam->Width = here->BSIM4v0w;
		  pParam->NFinger = here->BSIM4v0nf;
                  Ldrn = here->BSIM4v0l;
                  Wdrn = here->BSIM4v0w / here->BSIM4v0nf;
		  
                  T0 = pow(Ldrn, model->BSIM4v0Lln);
                  T1 = pow(Wdrn, model->BSIM4v0Lwn);
                  tmp1 = model->BSIM4v0Ll / T0 + model->BSIM4v0Lw / T1
                       + model->BSIM4v0Lwl / (T0 * T1);
                  pParam->BSIM4v0dl = model->BSIM4v0Lint + tmp1;
                  tmp2 = model->BSIM4v0Llc / T0 + model->BSIM4v0Lwc / T1
                       + model->BSIM4v0Lwlc / (T0 * T1);
                  pParam->BSIM4v0dlc = model->BSIM4v0dlc + tmp2;
                  pParam->BSIM4v0dlcig = model->BSIM4v0dlcig + tmp2;

                  T2 = pow(Ldrn, model->BSIM4v0Wln);
                  T3 = pow(Wdrn, model->BSIM4v0Wwn);
                  tmp1 = model->BSIM4v0Wl / T2 + model->BSIM4v0Ww / T3
                       + model->BSIM4v0Wwl / (T2 * T3);
                  pParam->BSIM4v0dw = model->BSIM4v0Wint + tmp1;
                  tmp2 = model->BSIM4v0Wlc / T2 + model->BSIM4v0Wwc / T3
                       + model->BSIM4v0Wwlc / (T2 * T3);
                  pParam->BSIM4v0dwc = model->BSIM4v0dwc + tmp2;
                  pParam->BSIM4v0dwj = model->BSIM4v0dwj + tmp2;

                  pParam->BSIM4v0leff = here->BSIM4v0l - 2.0 * pParam->BSIM4v0dl;
                  if (pParam->BSIM4v0leff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v0modName;
                      namarray[1] = here->BSIM4v0name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v0: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v0weff = here->BSIM4v0w / here->BSIM4v0nf 
				    - 2.0 * pParam->BSIM4v0dw;
                  if (pParam->BSIM4v0weff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v0modName;
                      namarray[1] = here->BSIM4v0name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v0: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v0leffCV = here->BSIM4v0l - 2.0 * pParam->BSIM4v0dlc;
                  if (pParam->BSIM4v0leffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v0modName;
                      namarray[1] = here->BSIM4v0name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v0: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v0weffCV = here->BSIM4v0w / here->BSIM4v0nf
				      - 2.0 * pParam->BSIM4v0dwc;
                  if (pParam->BSIM4v0weffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v0modName;
                      namarray[1] = here->BSIM4v0name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v0: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v0weffCJ = here->BSIM4v0w / here->BSIM4v0nf
				      - 2.0 * pParam->BSIM4v0dwj;
                  if (pParam->BSIM4v0weffCJ <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v0modName;
                      namarray[1] = here->BSIM4v0name;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM4v0: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


		  if (model->BSIM4v0binUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM4v0leff;
		      Inv_W = 1.0e-6 / pParam->BSIM4v0weff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM4v0leff
			     * pParam->BSIM4v0weff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM4v0leff;
		      Inv_W = 1.0 / pParam->BSIM4v0weff;
		      Inv_LW = 1.0 / (pParam->BSIM4v0leff
			     * pParam->BSIM4v0weff);
		  }
		  pParam->BSIM4v0cdsc = model->BSIM4v0cdsc
				    + model->BSIM4v0lcdsc * Inv_L
				    + model->BSIM4v0wcdsc * Inv_W
				    + model->BSIM4v0pcdsc * Inv_LW;
		  pParam->BSIM4v0cdscb = model->BSIM4v0cdscb
				     + model->BSIM4v0lcdscb * Inv_L
				     + model->BSIM4v0wcdscb * Inv_W
				     + model->BSIM4v0pcdscb * Inv_LW; 
				     
    		  pParam->BSIM4v0cdscd = model->BSIM4v0cdscd
				     + model->BSIM4v0lcdscd * Inv_L
				     + model->BSIM4v0wcdscd * Inv_W
				     + model->BSIM4v0pcdscd * Inv_LW; 
				     
		  pParam->BSIM4v0cit = model->BSIM4v0cit
				   + model->BSIM4v0lcit * Inv_L
				   + model->BSIM4v0wcit * Inv_W
				   + model->BSIM4v0pcit * Inv_LW;
		  pParam->BSIM4v0nfactor = model->BSIM4v0nfactor
				       + model->BSIM4v0lnfactor * Inv_L
				       + model->BSIM4v0wnfactor * Inv_W
				       + model->BSIM4v0pnfactor * Inv_LW;
		  pParam->BSIM4v0xj = model->BSIM4v0xj
				  + model->BSIM4v0lxj * Inv_L
				  + model->BSIM4v0wxj * Inv_W
				  + model->BSIM4v0pxj * Inv_LW;
		  pParam->BSIM4v0vsat = model->BSIM4v0vsat
				    + model->BSIM4v0lvsat * Inv_L
				    + model->BSIM4v0wvsat * Inv_W
				    + model->BSIM4v0pvsat * Inv_LW;
		  pParam->BSIM4v0at = model->BSIM4v0at
				  + model->BSIM4v0lat * Inv_L
				  + model->BSIM4v0wat * Inv_W
				  + model->BSIM4v0pat * Inv_LW;
		  pParam->BSIM4v0a0 = model->BSIM4v0a0
				  + model->BSIM4v0la0 * Inv_L
				  + model->BSIM4v0wa0 * Inv_W
				  + model->BSIM4v0pa0 * Inv_LW; 
				  
		  pParam->BSIM4v0ags = model->BSIM4v0ags
				  + model->BSIM4v0lags * Inv_L
				  + model->BSIM4v0wags * Inv_W
				  + model->BSIM4v0pags * Inv_LW;
				  
		  pParam->BSIM4v0a1 = model->BSIM4v0a1
				  + model->BSIM4v0la1 * Inv_L
				  + model->BSIM4v0wa1 * Inv_W
				  + model->BSIM4v0pa1 * Inv_LW;
		  pParam->BSIM4v0a2 = model->BSIM4v0a2
				  + model->BSIM4v0la2 * Inv_L
				  + model->BSIM4v0wa2 * Inv_W
				  + model->BSIM4v0pa2 * Inv_LW;
		  pParam->BSIM4v0keta = model->BSIM4v0keta
				    + model->BSIM4v0lketa * Inv_L
				    + model->BSIM4v0wketa * Inv_W
				    + model->BSIM4v0pketa * Inv_LW;
		  pParam->BSIM4v0nsub = model->BSIM4v0nsub
				    + model->BSIM4v0lnsub * Inv_L
				    + model->BSIM4v0wnsub * Inv_W
				    + model->BSIM4v0pnsub * Inv_LW;
		  pParam->BSIM4v0ndep = model->BSIM4v0ndep
				    + model->BSIM4v0lndep * Inv_L
				    + model->BSIM4v0wndep * Inv_W
				    + model->BSIM4v0pndep * Inv_LW;
                  pParam->BSIM4v0nsd = model->BSIM4v0nsd
                                   + model->BSIM4v0lnsd * Inv_L
                                   + model->BSIM4v0wnsd * Inv_W
                                   + model->BSIM4v0pnsd * Inv_LW;
                  pParam->BSIM4v0phin = model->BSIM4v0phin
                                    + model->BSIM4v0lphin * Inv_L
                                    + model->BSIM4v0wphin * Inv_W
                                    + model->BSIM4v0pphin * Inv_LW;
		  pParam->BSIM4v0ngate = model->BSIM4v0ngate
				     + model->BSIM4v0lngate * Inv_L
				     + model->BSIM4v0wngate * Inv_W
				     + model->BSIM4v0pngate * Inv_LW;
		  pParam->BSIM4v0gamma1 = model->BSIM4v0gamma1
				      + model->BSIM4v0lgamma1 * Inv_L
				      + model->BSIM4v0wgamma1 * Inv_W
				      + model->BSIM4v0pgamma1 * Inv_LW;
		  pParam->BSIM4v0gamma2 = model->BSIM4v0gamma2
				      + model->BSIM4v0lgamma2 * Inv_L
				      + model->BSIM4v0wgamma2 * Inv_W
				      + model->BSIM4v0pgamma2 * Inv_LW;
		  pParam->BSIM4v0vbx = model->BSIM4v0vbx
				   + model->BSIM4v0lvbx * Inv_L
				   + model->BSIM4v0wvbx * Inv_W
				   + model->BSIM4v0pvbx * Inv_LW;
		  pParam->BSIM4v0vbm = model->BSIM4v0vbm
				   + model->BSIM4v0lvbm * Inv_L
				   + model->BSIM4v0wvbm * Inv_W
				   + model->BSIM4v0pvbm * Inv_LW;
		  pParam->BSIM4v0xt = model->BSIM4v0xt
				   + model->BSIM4v0lxt * Inv_L
				   + model->BSIM4v0wxt * Inv_W
				   + model->BSIM4v0pxt * Inv_LW;
                  pParam->BSIM4v0vfb = model->BSIM4v0vfb
                                   + model->BSIM4v0lvfb * Inv_L
                                   + model->BSIM4v0wvfb * Inv_W
                                   + model->BSIM4v0pvfb * Inv_LW;
		  pParam->BSIM4v0k1 = model->BSIM4v0k1
				  + model->BSIM4v0lk1 * Inv_L
				  + model->BSIM4v0wk1 * Inv_W
				  + model->BSIM4v0pk1 * Inv_LW;
		  pParam->BSIM4v0kt1 = model->BSIM4v0kt1
				   + model->BSIM4v0lkt1 * Inv_L
				   + model->BSIM4v0wkt1 * Inv_W
				   + model->BSIM4v0pkt1 * Inv_LW;
		  pParam->BSIM4v0kt1l = model->BSIM4v0kt1l
				    + model->BSIM4v0lkt1l * Inv_L
				    + model->BSIM4v0wkt1l * Inv_W
				    + model->BSIM4v0pkt1l * Inv_LW;
		  pParam->BSIM4v0k2 = model->BSIM4v0k2
				  + model->BSIM4v0lk2 * Inv_L
				  + model->BSIM4v0wk2 * Inv_W
				  + model->BSIM4v0pk2 * Inv_LW;
		  pParam->BSIM4v0kt2 = model->BSIM4v0kt2
				   + model->BSIM4v0lkt2 * Inv_L
				   + model->BSIM4v0wkt2 * Inv_W
				   + model->BSIM4v0pkt2 * Inv_LW;
		  pParam->BSIM4v0k3 = model->BSIM4v0k3
				  + model->BSIM4v0lk3 * Inv_L
				  + model->BSIM4v0wk3 * Inv_W
				  + model->BSIM4v0pk3 * Inv_LW;
		  pParam->BSIM4v0k3b = model->BSIM4v0k3b
				   + model->BSIM4v0lk3b * Inv_L
				   + model->BSIM4v0wk3b * Inv_W
				   + model->BSIM4v0pk3b * Inv_LW;
		  pParam->BSIM4v0w0 = model->BSIM4v0w0
				  + model->BSIM4v0lw0 * Inv_L
				  + model->BSIM4v0ww0 * Inv_W
				  + model->BSIM4v0pw0 * Inv_LW;
		  pParam->BSIM4v0lpe0 = model->BSIM4v0lpe0
				    + model->BSIM4v0llpe0 * Inv_L
 				    + model->BSIM4v0wlpe0 * Inv_W
				    + model->BSIM4v0plpe0 * Inv_LW;
                  pParam->BSIM4v0lpeb = model->BSIM4v0lpeb
                                    + model->BSIM4v0llpeb * Inv_L
                                    + model->BSIM4v0wlpeb * Inv_W
                                    + model->BSIM4v0plpeb * Inv_LW;
                  pParam->BSIM4v0dvtp0 = model->BSIM4v0dvtp0
                                     + model->BSIM4v0ldvtp0 * Inv_L
                                     + model->BSIM4v0wdvtp0 * Inv_W
                                     + model->BSIM4v0pdvtp0 * Inv_LW;
                  pParam->BSIM4v0dvtp1 = model->BSIM4v0dvtp1
                                     + model->BSIM4v0ldvtp1 * Inv_L
                                     + model->BSIM4v0wdvtp1 * Inv_W
                                     + model->BSIM4v0pdvtp1 * Inv_LW;
		  pParam->BSIM4v0dvt0 = model->BSIM4v0dvt0
				    + model->BSIM4v0ldvt0 * Inv_L
				    + model->BSIM4v0wdvt0 * Inv_W
				    + model->BSIM4v0pdvt0 * Inv_LW;
		  pParam->BSIM4v0dvt1 = model->BSIM4v0dvt1
				    + model->BSIM4v0ldvt1 * Inv_L
				    + model->BSIM4v0wdvt1 * Inv_W
				    + model->BSIM4v0pdvt1 * Inv_LW;
		  pParam->BSIM4v0dvt2 = model->BSIM4v0dvt2
				    + model->BSIM4v0ldvt2 * Inv_L
				    + model->BSIM4v0wdvt2 * Inv_W
				    + model->BSIM4v0pdvt2 * Inv_LW;
		  pParam->BSIM4v0dvt0w = model->BSIM4v0dvt0w
				    + model->BSIM4v0ldvt0w * Inv_L
				    + model->BSIM4v0wdvt0w * Inv_W
				    + model->BSIM4v0pdvt0w * Inv_LW;
		  pParam->BSIM4v0dvt1w = model->BSIM4v0dvt1w
				    + model->BSIM4v0ldvt1w * Inv_L
				    + model->BSIM4v0wdvt1w * Inv_W
				    + model->BSIM4v0pdvt1w * Inv_LW;
		  pParam->BSIM4v0dvt2w = model->BSIM4v0dvt2w
				    + model->BSIM4v0ldvt2w * Inv_L
				    + model->BSIM4v0wdvt2w * Inv_W
				    + model->BSIM4v0pdvt2w * Inv_LW;
		  pParam->BSIM4v0drout = model->BSIM4v0drout
				     + model->BSIM4v0ldrout * Inv_L
				     + model->BSIM4v0wdrout * Inv_W
				     + model->BSIM4v0pdrout * Inv_LW;
		  pParam->BSIM4v0dsub = model->BSIM4v0dsub
				    + model->BSIM4v0ldsub * Inv_L
				    + model->BSIM4v0wdsub * Inv_W
				    + model->BSIM4v0pdsub * Inv_LW;
		  pParam->BSIM4v0vth0 = model->BSIM4v0vth0
				    + model->BSIM4v0lvth0 * Inv_L
				    + model->BSIM4v0wvth0 * Inv_W
				    + model->BSIM4v0pvth0 * Inv_LW;
		  pParam->BSIM4v0ua = model->BSIM4v0ua
				  + model->BSIM4v0lua * Inv_L
				  + model->BSIM4v0wua * Inv_W
				  + model->BSIM4v0pua * Inv_LW;
		  pParam->BSIM4v0ua1 = model->BSIM4v0ua1
				   + model->BSIM4v0lua1 * Inv_L
				   + model->BSIM4v0wua1 * Inv_W
				   + model->BSIM4v0pua1 * Inv_LW;
		  pParam->BSIM4v0ub = model->BSIM4v0ub
				  + model->BSIM4v0lub * Inv_L
				  + model->BSIM4v0wub * Inv_W
				  + model->BSIM4v0pub * Inv_LW;
		  pParam->BSIM4v0ub1 = model->BSIM4v0ub1
				   + model->BSIM4v0lub1 * Inv_L
				   + model->BSIM4v0wub1 * Inv_W
				   + model->BSIM4v0pub1 * Inv_LW;
		  pParam->BSIM4v0uc = model->BSIM4v0uc
				  + model->BSIM4v0luc * Inv_L
				  + model->BSIM4v0wuc * Inv_W
				  + model->BSIM4v0puc * Inv_LW;
		  pParam->BSIM4v0uc1 = model->BSIM4v0uc1
				   + model->BSIM4v0luc1 * Inv_L
				   + model->BSIM4v0wuc1 * Inv_W
				   + model->BSIM4v0puc1 * Inv_LW;
                  pParam->BSIM4v0eu = model->BSIM4v0eu
                                  + model->BSIM4v0leu * Inv_L
                                  + model->BSIM4v0weu * Inv_W
                                  + model->BSIM4v0peu * Inv_LW;
		  pParam->BSIM4v0u0 = model->BSIM4v0u0
				  + model->BSIM4v0lu0 * Inv_L
				  + model->BSIM4v0wu0 * Inv_W
				  + model->BSIM4v0pu0 * Inv_LW;
		  pParam->BSIM4v0ute = model->BSIM4v0ute
				   + model->BSIM4v0lute * Inv_L
				   + model->BSIM4v0wute * Inv_W
				   + model->BSIM4v0pute * Inv_LW;
		  pParam->BSIM4v0voff = model->BSIM4v0voff
				    + model->BSIM4v0lvoff * Inv_L
				    + model->BSIM4v0wvoff * Inv_W
				    + model->BSIM4v0pvoff * Inv_LW;
                  pParam->BSIM4v0minv = model->BSIM4v0minv
                                    + model->BSIM4v0lminv * Inv_L
                                    + model->BSIM4v0wminv * Inv_W
                                    + model->BSIM4v0pminv * Inv_LW;
                  pParam->BSIM4v0fprout = model->BSIM4v0fprout
                                     + model->BSIM4v0lfprout * Inv_L
                                     + model->BSIM4v0wfprout * Inv_W
                                     + model->BSIM4v0pfprout * Inv_LW;
                  pParam->BSIM4v0pdits = model->BSIM4v0pdits
                                     + model->BSIM4v0lpdits * Inv_L
                                     + model->BSIM4v0wpdits * Inv_W
                                     + model->BSIM4v0ppdits * Inv_LW;
                  pParam->BSIM4v0pditsd = model->BSIM4v0pditsd
                                      + model->BSIM4v0lpditsd * Inv_L
                                      + model->BSIM4v0wpditsd * Inv_W
                                      + model->BSIM4v0ppditsd * Inv_LW;
		  pParam->BSIM4v0delta = model->BSIM4v0delta
				     + model->BSIM4v0ldelta * Inv_L
				     + model->BSIM4v0wdelta * Inv_W
				     + model->BSIM4v0pdelta * Inv_LW;
		  pParam->BSIM4v0rdsw = model->BSIM4v0rdsw
				    + model->BSIM4v0lrdsw * Inv_L
				    + model->BSIM4v0wrdsw * Inv_W
				    + model->BSIM4v0prdsw * Inv_LW;
                  pParam->BSIM4v0rdw = model->BSIM4v0rdw
                                    + model->BSIM4v0lrdw * Inv_L
                                    + model->BSIM4v0wrdw * Inv_W
                                    + model->BSIM4v0prdw * Inv_LW;
                  pParam->BSIM4v0rsw = model->BSIM4v0rsw
                                    + model->BSIM4v0lrsw * Inv_L
                                    + model->BSIM4v0wrsw * Inv_W
                                    + model->BSIM4v0prsw * Inv_LW;
		  pParam->BSIM4v0prwg = model->BSIM4v0prwg
				    + model->BSIM4v0lprwg * Inv_L
				    + model->BSIM4v0wprwg * Inv_W
				    + model->BSIM4v0pprwg * Inv_LW;
		  pParam->BSIM4v0prwb = model->BSIM4v0prwb
				    + model->BSIM4v0lprwb * Inv_L
				    + model->BSIM4v0wprwb * Inv_W
				    + model->BSIM4v0pprwb * Inv_LW;
		  pParam->BSIM4v0prt = model->BSIM4v0prt
				    + model->BSIM4v0lprt * Inv_L
				    + model->BSIM4v0wprt * Inv_W
				    + model->BSIM4v0pprt * Inv_LW;
		  pParam->BSIM4v0eta0 = model->BSIM4v0eta0
				    + model->BSIM4v0leta0 * Inv_L
				    + model->BSIM4v0weta0 * Inv_W
				    + model->BSIM4v0peta0 * Inv_LW;
		  pParam->BSIM4v0etab = model->BSIM4v0etab
				    + model->BSIM4v0letab * Inv_L
				    + model->BSIM4v0wetab * Inv_W
				    + model->BSIM4v0petab * Inv_LW;
		  pParam->BSIM4v0pclm = model->BSIM4v0pclm
				    + model->BSIM4v0lpclm * Inv_L
				    + model->BSIM4v0wpclm * Inv_W
				    + model->BSIM4v0ppclm * Inv_LW;
		  pParam->BSIM4v0pdibl1 = model->BSIM4v0pdibl1
				      + model->BSIM4v0lpdibl1 * Inv_L
				      + model->BSIM4v0wpdibl1 * Inv_W
				      + model->BSIM4v0ppdibl1 * Inv_LW;
		  pParam->BSIM4v0pdibl2 = model->BSIM4v0pdibl2
				      + model->BSIM4v0lpdibl2 * Inv_L
				      + model->BSIM4v0wpdibl2 * Inv_W
				      + model->BSIM4v0ppdibl2 * Inv_LW;
		  pParam->BSIM4v0pdiblb = model->BSIM4v0pdiblb
				      + model->BSIM4v0lpdiblb * Inv_L
				      + model->BSIM4v0wpdiblb * Inv_W
				      + model->BSIM4v0ppdiblb * Inv_LW;
		  pParam->BSIM4v0pscbe1 = model->BSIM4v0pscbe1
				      + model->BSIM4v0lpscbe1 * Inv_L
				      + model->BSIM4v0wpscbe1 * Inv_W
				      + model->BSIM4v0ppscbe1 * Inv_LW;
		  pParam->BSIM4v0pscbe2 = model->BSIM4v0pscbe2
				      + model->BSIM4v0lpscbe2 * Inv_L
				      + model->BSIM4v0wpscbe2 * Inv_W
				      + model->BSIM4v0ppscbe2 * Inv_LW;
		  pParam->BSIM4v0pvag = model->BSIM4v0pvag
				    + model->BSIM4v0lpvag * Inv_L
				    + model->BSIM4v0wpvag * Inv_W
				    + model->BSIM4v0ppvag * Inv_LW;
		  pParam->BSIM4v0wr = model->BSIM4v0wr
				  + model->BSIM4v0lwr * Inv_L
				  + model->BSIM4v0wwr * Inv_W
				  + model->BSIM4v0pwr * Inv_LW;
		  pParam->BSIM4v0dwg = model->BSIM4v0dwg
				   + model->BSIM4v0ldwg * Inv_L
				   + model->BSIM4v0wdwg * Inv_W
				   + model->BSIM4v0pdwg * Inv_LW;
		  pParam->BSIM4v0dwb = model->BSIM4v0dwb
				   + model->BSIM4v0ldwb * Inv_L
				   + model->BSIM4v0wdwb * Inv_W
				   + model->BSIM4v0pdwb * Inv_LW;
		  pParam->BSIM4v0b0 = model->BSIM4v0b0
				  + model->BSIM4v0lb0 * Inv_L
				  + model->BSIM4v0wb0 * Inv_W
				  + model->BSIM4v0pb0 * Inv_LW;
		  pParam->BSIM4v0b1 = model->BSIM4v0b1
				  + model->BSIM4v0lb1 * Inv_L
				  + model->BSIM4v0wb1 * Inv_W
				  + model->BSIM4v0pb1 * Inv_LW;
		  pParam->BSIM4v0alpha0 = model->BSIM4v0alpha0
				      + model->BSIM4v0lalpha0 * Inv_L
				      + model->BSIM4v0walpha0 * Inv_W
				      + model->BSIM4v0palpha0 * Inv_LW;
                  pParam->BSIM4v0alpha1 = model->BSIM4v0alpha1
                                      + model->BSIM4v0lalpha1 * Inv_L
                                      + model->BSIM4v0walpha1 * Inv_W
                                      + model->BSIM4v0palpha1 * Inv_LW;
		  pParam->BSIM4v0beta0 = model->BSIM4v0beta0
				     + model->BSIM4v0lbeta0 * Inv_L
				     + model->BSIM4v0wbeta0 * Inv_W
				     + model->BSIM4v0pbeta0 * Inv_LW;
                  pParam->BSIM4v0agidl = model->BSIM4v0agidl
                                     + model->BSIM4v0lagidl * Inv_L
                                     + model->BSIM4v0wagidl * Inv_W
                                     + model->BSIM4v0pagidl * Inv_LW;
                  pParam->BSIM4v0bgidl = model->BSIM4v0bgidl
                                     + model->BSIM4v0lbgidl * Inv_L
                                     + model->BSIM4v0wbgidl * Inv_W
                                     + model->BSIM4v0pbgidl * Inv_LW;
                  pParam->BSIM4v0cgidl = model->BSIM4v0cgidl
                                     + model->BSIM4v0lcgidl * Inv_L
                                     + model->BSIM4v0wcgidl * Inv_W
                                     + model->BSIM4v0pcgidl * Inv_LW;
                  pParam->BSIM4v0egidl = model->BSIM4v0egidl
                                     + model->BSIM4v0legidl * Inv_L
                                     + model->BSIM4v0wegidl * Inv_W
                                     + model->BSIM4v0pegidl * Inv_LW;
                  pParam->BSIM4v0aigc = model->BSIM4v0aigc
                                     + model->BSIM4v0laigc * Inv_L
                                     + model->BSIM4v0waigc * Inv_W
                                     + model->BSIM4v0paigc * Inv_LW;
                  pParam->BSIM4v0bigc = model->BSIM4v0bigc
                                     + model->BSIM4v0lbigc * Inv_L
                                     + model->BSIM4v0wbigc * Inv_W
                                     + model->BSIM4v0pbigc * Inv_LW;
                  pParam->BSIM4v0cigc = model->BSIM4v0cigc
                                     + model->BSIM4v0lcigc * Inv_L
                                     + model->BSIM4v0wcigc * Inv_W
                                     + model->BSIM4v0pcigc * Inv_LW;
                  pParam->BSIM4v0aigsd = model->BSIM4v0aigsd
                                     + model->BSIM4v0laigsd * Inv_L
                                     + model->BSIM4v0waigsd * Inv_W
                                     + model->BSIM4v0paigsd * Inv_LW;
                  pParam->BSIM4v0bigsd = model->BSIM4v0bigsd
                                     + model->BSIM4v0lbigsd * Inv_L
                                     + model->BSIM4v0wbigsd * Inv_W
                                     + model->BSIM4v0pbigsd * Inv_LW;
                  pParam->BSIM4v0cigsd = model->BSIM4v0cigsd
                                     + model->BSIM4v0lcigsd * Inv_L
                                     + model->BSIM4v0wcigsd * Inv_W
                                     + model->BSIM4v0pcigsd * Inv_LW;
                  pParam->BSIM4v0aigbacc = model->BSIM4v0aigbacc
                                       + model->BSIM4v0laigbacc * Inv_L
                                       + model->BSIM4v0waigbacc * Inv_W
                                       + model->BSIM4v0paigbacc * Inv_LW;
                  pParam->BSIM4v0bigbacc = model->BSIM4v0bigbacc
                                       + model->BSIM4v0lbigbacc * Inv_L
                                       + model->BSIM4v0wbigbacc * Inv_W
                                       + model->BSIM4v0pbigbacc * Inv_LW;
                  pParam->BSIM4v0cigbacc = model->BSIM4v0cigbacc
                                       + model->BSIM4v0lcigbacc * Inv_L
                                       + model->BSIM4v0wcigbacc * Inv_W
                                       + model->BSIM4v0pcigbacc * Inv_LW;
                  pParam->BSIM4v0aigbinv = model->BSIM4v0aigbinv
                                       + model->BSIM4v0laigbinv * Inv_L
                                       + model->BSIM4v0waigbinv * Inv_W
                                       + model->BSIM4v0paigbinv * Inv_LW;
                  pParam->BSIM4v0bigbinv = model->BSIM4v0bigbinv
                                       + model->BSIM4v0lbigbinv * Inv_L
                                       + model->BSIM4v0wbigbinv * Inv_W
                                       + model->BSIM4v0pbigbinv * Inv_LW;
                  pParam->BSIM4v0cigbinv = model->BSIM4v0cigbinv
                                       + model->BSIM4v0lcigbinv * Inv_L
                                       + model->BSIM4v0wcigbinv * Inv_W
                                       + model->BSIM4v0pcigbinv * Inv_LW;
                  pParam->BSIM4v0nigc = model->BSIM4v0nigc
                                       + model->BSIM4v0lnigc * Inv_L
                                       + model->BSIM4v0wnigc * Inv_W
                                       + model->BSIM4v0pnigc * Inv_LW;
                  pParam->BSIM4v0nigbacc = model->BSIM4v0nigbacc
                                       + model->BSIM4v0lnigbacc * Inv_L
                                       + model->BSIM4v0wnigbacc * Inv_W
                                       + model->BSIM4v0pnigbacc * Inv_LW;
                  pParam->BSIM4v0nigbinv = model->BSIM4v0nigbinv
                                       + model->BSIM4v0lnigbinv * Inv_L
                                       + model->BSIM4v0wnigbinv * Inv_W
                                       + model->BSIM4v0pnigbinv * Inv_LW;
                  pParam->BSIM4v0ntox = model->BSIM4v0ntox
                                    + model->BSIM4v0lntox * Inv_L
                                    + model->BSIM4v0wntox * Inv_W
                                    + model->BSIM4v0pntox * Inv_LW;
                  pParam->BSIM4v0eigbinv = model->BSIM4v0eigbinv
                                       + model->BSIM4v0leigbinv * Inv_L
                                       + model->BSIM4v0weigbinv * Inv_W
                                       + model->BSIM4v0peigbinv * Inv_LW;
                  pParam->BSIM4v0pigcd = model->BSIM4v0pigcd
                                     + model->BSIM4v0lpigcd * Inv_L
                                     + model->BSIM4v0wpigcd * Inv_W
                                     + model->BSIM4v0ppigcd * Inv_LW;
                  pParam->BSIM4v0poxedge = model->BSIM4v0poxedge
                                       + model->BSIM4v0lpoxedge * Inv_L
                                       + model->BSIM4v0wpoxedge * Inv_W
                                       + model->BSIM4v0ppoxedge * Inv_LW;
                  pParam->BSIM4v0xrcrg1 = model->BSIM4v0xrcrg1
                                      + model->BSIM4v0lxrcrg1 * Inv_L
                                      + model->BSIM4v0wxrcrg1 * Inv_W
                                      + model->BSIM4v0pxrcrg1 * Inv_LW;
                  pParam->BSIM4v0xrcrg2 = model->BSIM4v0xrcrg2
                                      + model->BSIM4v0lxrcrg2 * Inv_L
                                      + model->BSIM4v0wxrcrg2 * Inv_W
                                      + model->BSIM4v0pxrcrg2 * Inv_LW;

		  pParam->BSIM4v0cgsl = model->BSIM4v0cgsl
				    + model->BSIM4v0lcgsl * Inv_L
				    + model->BSIM4v0wcgsl * Inv_W
				    + model->BSIM4v0pcgsl * Inv_LW;
		  pParam->BSIM4v0cgdl = model->BSIM4v0cgdl
				    + model->BSIM4v0lcgdl * Inv_L
				    + model->BSIM4v0wcgdl * Inv_W
				    + model->BSIM4v0pcgdl * Inv_LW;
		  pParam->BSIM4v0ckappas = model->BSIM4v0ckappas
				       + model->BSIM4v0lckappas * Inv_L
				       + model->BSIM4v0wckappas * Inv_W
 				       + model->BSIM4v0pckappas * Inv_LW;
                  pParam->BSIM4v0ckappad = model->BSIM4v0ckappad
                                       + model->BSIM4v0lckappad * Inv_L
                                       + model->BSIM4v0wckappad * Inv_W
                                       + model->BSIM4v0pckappad * Inv_LW;
		  pParam->BSIM4v0cf = model->BSIM4v0cf
				  + model->BSIM4v0lcf * Inv_L
				  + model->BSIM4v0wcf * Inv_W
				  + model->BSIM4v0pcf * Inv_LW;
		  pParam->BSIM4v0clc = model->BSIM4v0clc
				   + model->BSIM4v0lclc * Inv_L
				   + model->BSIM4v0wclc * Inv_W
				   + model->BSIM4v0pclc * Inv_LW;
		  pParam->BSIM4v0cle = model->BSIM4v0cle
				   + model->BSIM4v0lcle * Inv_L
				   + model->BSIM4v0wcle * Inv_W
				   + model->BSIM4v0pcle * Inv_LW;
		  pParam->BSIM4v0vfbcv = model->BSIM4v0vfbcv
				     + model->BSIM4v0lvfbcv * Inv_L
				     + model->BSIM4v0wvfbcv * Inv_W
				     + model->BSIM4v0pvfbcv * Inv_LW;
                  pParam->BSIM4v0acde = model->BSIM4v0acde
                                    + model->BSIM4v0lacde * Inv_L
                                    + model->BSIM4v0wacde * Inv_W
                                    + model->BSIM4v0pacde * Inv_LW;
                  pParam->BSIM4v0moin = model->BSIM4v0moin
                                    + model->BSIM4v0lmoin * Inv_L
                                    + model->BSIM4v0wmoin * Inv_W
                                    + model->BSIM4v0pmoin * Inv_LW;
                  pParam->BSIM4v0noff = model->BSIM4v0noff
                                    + model->BSIM4v0lnoff * Inv_L
                                    + model->BSIM4v0wnoff * Inv_W
                                    + model->BSIM4v0pnoff * Inv_LW;
                  pParam->BSIM4v0voffcv = model->BSIM4v0voffcv
                                      + model->BSIM4v0lvoffcv * Inv_L
                                      + model->BSIM4v0wvoffcv * Inv_W
                                      + model->BSIM4v0pvoffcv * Inv_LW;

                  pParam->BSIM4v0abulkCVfactor = 1.0 + pow((pParam->BSIM4v0clc
					     / pParam->BSIM4v0leffCV),
					     pParam->BSIM4v0cle);

	          T0 = (TRatio - 1.0);
	          pParam->BSIM4v0ua = pParam->BSIM4v0ua + pParam->BSIM4v0ua1 * T0;
	          pParam->BSIM4v0ub = pParam->BSIM4v0ub + pParam->BSIM4v0ub1 * T0;
	          pParam->BSIM4v0uc = pParam->BSIM4v0uc + pParam->BSIM4v0uc1 * T0;
                  if (pParam->BSIM4v0u0 > 1.0) 
                      pParam->BSIM4v0u0 = pParam->BSIM4v0u0 / 1.0e4;

                  pParam->BSIM4v0u0temp = pParam->BSIM4v0u0
				      * pow(TRatio, pParam->BSIM4v0ute); 
                  pParam->BSIM4v0vsattemp = pParam->BSIM4v0vsat - pParam->BSIM4v0at 
			                * T0;
                  if (pParam->BSIM4v0eu < 0.0)
                  {   pParam->BSIM4v0eu = 0.0;
		      printf("Warning: eu has been negative; reset to 0.0.\n");
		  }


		  PowWeffWr = pow(pParam->BSIM4v0weffCJ * 1.0e6, pParam->BSIM4v0wr) * here->BSIM4v0nf;
		  /* External Rd(V) */
		  T1 = pParam->BSIM4v0rdw + pParam->BSIM4v0prt * T0;
		  if (T1 < 0.0)
		  {   T1 = 0.0;
		      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
		  }
                  T2 = model->BSIM4v0rdwmin + pParam->BSIM4v0prt * T0;
		  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
		  pParam->BSIM4v0rd0 = T1 / PowWeffWr;
                  pParam->BSIM4v0rdwmin = T2 / PowWeffWr;


		  /* External Rs(V) */
		  T1 = pParam->BSIM4v0rsw + pParam->BSIM4v0prt * T0;
                  if (T1 < 0.0)
                  {   T1 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  T2 = model->BSIM4v0rswmin + pParam->BSIM4v0prt * T0;
                  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v0rs0 = T1 / PowWeffWr;
                  pParam->BSIM4v0rswmin = T2 / PowWeffWr;

		  /* Internal Rds(V) in IV */
	          pParam->BSIM4v0rds0 = (pParam->BSIM4v0rdsw + pParam->BSIM4v0prt * T0)
				    * here->BSIM4v0nf / PowWeffWr;
		  pParam->BSIM4v0rdswmin = (model->BSIM4v0rdswmin + pParam->BSIM4v0prt * T0)
				       * here->BSIM4v0nf / PowWeffWr;

                  pParam->BSIM4v0cgdo = (model->BSIM4v0cgdo + pParam->BSIM4v0cf)
				    * pParam->BSIM4v0weffCV;
                  pParam->BSIM4v0cgso = (model->BSIM4v0cgso + pParam->BSIM4v0cf)
				    * pParam->BSIM4v0weffCV;
                  pParam->BSIM4v0cgbo = model->BSIM4v0cgbo * pParam->BSIM4v0leffCV * here->BSIM4v0nf;

                  if (!model->BSIM4v0ndepGiven && model->BSIM4v0gamma1Given)
                  {   T0 = pParam->BSIM4v0gamma1 * model->BSIM4v0coxe;
                      pParam->BSIM4v0ndep = 3.01248e22 * T0 * T0;
                  }

		  pParam->BSIM4v0phi = Vtm0 * log(pParam->BSIM4v0ndep / ni)
				   + pParam->BSIM4v0phin + 0.4;

	          pParam->BSIM4v0sqrtPhi = sqrt(pParam->BSIM4v0phi);
	          pParam->BSIM4v0phis3 = pParam->BSIM4v0sqrtPhi * pParam->BSIM4v0phi;

                  pParam->BSIM4v0Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM4v0ndep * 1.0e6))
                                     * pParam->BSIM4v0sqrtPhi; 
                  pParam->BSIM4v0sqrtXdep0 = sqrt(pParam->BSIM4v0Xdep0);
                  pParam->BSIM4v0litl = sqrt(3.0 * pParam->BSIM4v0xj
				    * model->BSIM4v0toxe);
                  pParam->BSIM4v0vbi = Vtm0 * log(pParam->BSIM4v0nsd
			           * pParam->BSIM4v0ndep / (ni * ni));

		  if (pParam->BSIM4v0ngate > 0.0)
                  {   pParam->BSIM4v0vfbsd = Vtm0 * log(pParam->BSIM4v0ngate
                                         / pParam->BSIM4v0nsd);
		  }
		  else
		      pParam->BSIM4v0vfbsd = 0.0;

                  pParam->BSIM4v0cdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM4v0ndep * 1.0e6 / 2.0
				     / pParam->BSIM4v0phi);

                  pParam->BSIM4v0ToxRatio = exp(pParam->BSIM4v0ntox
					* log(model->BSIM4v0toxref / model->BSIM4v0toxe))
					/ model->BSIM4v0toxe / model->BSIM4v0toxe;
                  pParam->BSIM4v0ToxRatioEdge = exp(pParam->BSIM4v0ntox
                                            * log(model->BSIM4v0toxref
                                            / (model->BSIM4v0toxe * pParam->BSIM4v0poxedge)))
                                            / model->BSIM4v0toxe / model->BSIM4v0toxe
                                            / pParam->BSIM4v0poxedge / pParam->BSIM4v0poxedge;
                  pParam->BSIM4v0Aechvb = (model->BSIM4v0type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v0Bechvb = (model->BSIM4v0type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v0AechvbEdge = pParam->BSIM4v0Aechvb * pParam->BSIM4v0weff
					  * pParam->BSIM4v0dlcig * pParam->BSIM4v0ToxRatioEdge;
                  pParam->BSIM4v0BechvbEdge = -pParam->BSIM4v0Bechvb
					  * model->BSIM4v0toxe * pParam->BSIM4v0poxedge;
                  pParam->BSIM4v0Aechvb *= pParam->BSIM4v0weff * pParam->BSIM4v0leff
				       * pParam->BSIM4v0ToxRatio;
                  pParam->BSIM4v0Bechvb *= -model->BSIM4v0toxe;


                  pParam->BSIM4v0mstar = 0.5 + atan(pParam->BSIM4v0minv) / PI;
                  pParam->BSIM4v0voffcbn =  pParam->BSIM4v0voff + model->BSIM4v0voffl / pParam->BSIM4v0leff;

                  pParam->BSIM4v0ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4v0ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v0acde *= pow((pParam->BSIM4v0ndep / 2.0e16), -0.25);


                  if (model->BSIM4v0k1Given || model->BSIM4v0k2Given)
	          {   if (!model->BSIM4v0k1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v0k1 = 0.53;
                      }
                      if (!model->BSIM4v0k2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v0k2 = -0.0186;
                      }
                      if (model->BSIM4v0nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v0xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v0vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v0gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v0gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM4v0vbxGiven)
                          pParam->BSIM4v0vbx = pParam->BSIM4v0phi - 7.7348e-4 
                                           * pParam->BSIM4v0ndep
					   * pParam->BSIM4v0xt * pParam->BSIM4v0xt;
	              if (pParam->BSIM4v0vbx > 0.0)
		          pParam->BSIM4v0vbx = -pParam->BSIM4v0vbx;
	              if (pParam->BSIM4v0vbm > 0.0)
                          pParam->BSIM4v0vbm = -pParam->BSIM4v0vbm;
           
                      if (!model->BSIM4v0gamma1Given)
                          pParam->BSIM4v0gamma1 = 5.753e-12
					      * sqrt(pParam->BSIM4v0ndep)
                                              / model->BSIM4v0coxe;
                      if (!model->BSIM4v0gamma2Given)
                          pParam->BSIM4v0gamma2 = 5.753e-12
					      * sqrt(pParam->BSIM4v0nsub)
                                              / model->BSIM4v0coxe;

                      T0 = pParam->BSIM4v0gamma1 - pParam->BSIM4v0gamma2;
                      T1 = sqrt(pParam->BSIM4v0phi - pParam->BSIM4v0vbx)
			 - pParam->BSIM4v0sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v0phi * (pParam->BSIM4v0phi
			 - pParam->BSIM4v0vbm)) - pParam->BSIM4v0phi;
                      pParam->BSIM4v0k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v0vbm);
                      pParam->BSIM4v0k1 = pParam->BSIM4v0gamma2 - 2.0
				      * pParam->BSIM4v0k2 * sqrt(pParam->BSIM4v0phi
				      - pParam->BSIM4v0vbm);
                  }
 
		  if (pParam->BSIM4v0k2 < 0.0)
		  {   T0 = 0.5 * pParam->BSIM4v0k1 / pParam->BSIM4v0k2;
                      pParam->BSIM4v0vbsc = 0.9 * (pParam->BSIM4v0phi - T0 * T0);
		      if (pParam->BSIM4v0vbsc > -3.0)
		          pParam->BSIM4v0vbsc = -3.0;
		      else if (pParam->BSIM4v0vbsc < -30.0)
		          pParam->BSIM4v0vbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM4v0vbsc = -30.0;
		  }
		  if (pParam->BSIM4v0vbsc > pParam->BSIM4v0vbm)
		      pParam->BSIM4v0vbsc = pParam->BSIM4v0vbm;

                  if (!model->BSIM4v0vfbGiven)
                  {   if (model->BSIM4v0vth0Given)
                      {   pParam->BSIM4v0vfb = model->BSIM4v0type * pParam->BSIM4v0vth0
                                           - pParam->BSIM4v0phi - pParam->BSIM4v0k1
                                           * pParam->BSIM4v0sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4v0vfb = -1.0;
                      }
                  }
                  if (!model->BSIM4v0vth0Given)
                  {   pParam->BSIM4v0vth0 = model->BSIM4v0type * (pParam->BSIM4v0vfb
                                        + pParam->BSIM4v0phi + pParam->BSIM4v0k1
                                        * pParam->BSIM4v0sqrtPhi);
                  }

                  pParam->BSIM4v0k1ox = pParam->BSIM4v0k1 * model->BSIM4v0toxe
                                    / model->BSIM4v0toxm;
                  pParam->BSIM4v0k2ox = pParam->BSIM4v0k2 * model->BSIM4v0toxe
                                    / model->BSIM4v0toxm;

		  T3 = model->BSIM4v0type * pParam->BSIM4v0vth0
		     - pParam->BSIM4v0vfb - pParam->BSIM4v0phi;
		  T4 = T3 + T3;
		  T5 = 2.5 * T3;
                  pParam->BSIM4v0vtfbphi1 = (model->BSIM4v0type == NMOS) ? T4 : T5; 
		  if (pParam->BSIM4v0vtfbphi1 < 0.0)
		      pParam->BSIM4v0vtfbphi1 = 0.0;

                  pParam->BSIM4v0vtfbphi2 = 4.0 * T3;
                  if (pParam->BSIM4v0vtfbphi2 < 0.0)
                      pParam->BSIM4v0vtfbphi2 = 0.0;

                  tmp = sqrt(EPSSI / (model->BSIM4v0epsrox * EPS0)
                      * model->BSIM4v0toxe * pParam->BSIM4v0Xdep0);
          	  T0 = pParam->BSIM4v0dsub * pParam->BSIM4v0leff / tmp;
                  if (T0 < EXP_THRESHOLD)
          	  {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
              	      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v0theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v0theta0vb0 = 1.0 / (MAX_EXP - 2.0);

 	          T0 = pParam->BSIM4v0drout * pParam->BSIM4v0leff / tmp;
        	  if (T0 < EXP_THRESHOLD)
       	          {   T1 = exp(T0);
              	      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v0thetaRout = pParam->BSIM4v0pdibl1 * T5
                                         + pParam->BSIM4v0pdibl2;

                  tmp = sqrt(pParam->BSIM4v0Xdep0);
                  tmp1 = pParam->BSIM4v0vbi - pParam->BSIM4v0phi;
                  tmp2 = model->BSIM4v0factor1 * tmp;

                  T0 = pParam->BSIM4v0dvt1w * pParam->BSIM4v0weff
                     * pParam->BSIM4v0leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v0dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v0dvt1 * pParam->BSIM4v0leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  }
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v0dvt0 * T9 * tmp1;

                  T4 = model->BSIM4v0toxe * pParam->BSIM4v0phi
                     / (pParam->BSIM4v0weff + pParam->BSIM4v0w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v0lpe0 / pParam->BSIM4v0leff);
                  T5 = pParam->BSIM4v0k1ox * (T0 - 1.0) * pParam->BSIM4v0sqrtPhi
                     + (pParam->BSIM4v0kt1 + pParam->BSIM4v0kt1l / pParam->BSIM4v0leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM4v0type * pParam->BSIM4v0vth0
                       - T8 - T9 + pParam->BSIM4v0k3 * T4 + T5;
                  pParam->BSIM4v0vfbzb = tmp3 - pParam->BSIM4v0phi - pParam->BSIM4v0k1
                                     * pParam->BSIM4v0sqrtPhi; /* End of vfbzb */
              } /* End of SizeNotFound */

              here->BSIM4v0cgso = pParam->BSIM4v0cgso;
              here->BSIM4v0cgdo = pParam->BSIM4v0cgdo;

              if (here->BSIM4v0rbodyMod)
              {   if (here->BSIM4v0rbdb < 1.0e-3)
                      here->BSIM4v0grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v0grbdb = model->BSIM4v0gbmin + 1.0 / here->BSIM4v0rbdb;
                  if (here->BSIM4v0rbpb < 1.0e-3)
                      here->BSIM4v0grbpb = 1.0e3;
                  else
                      here->BSIM4v0grbpb = model->BSIM4v0gbmin + 1.0 / here->BSIM4v0rbpb;
                  if (here->BSIM4v0rbps < 1.0e-3)
                      here->BSIM4v0grbps = 1.0e3;
                  else
                      here->BSIM4v0grbps = model->BSIM4v0gbmin + 1.0 / here->BSIM4v0rbps;
                  if (here->BSIM4v0rbsb < 1.0e-3)
                      here->BSIM4v0grbsb = 1.0e3;
                  else
                      here->BSIM4v0grbsb = model->BSIM4v0gbmin + 1.0 / here->BSIM4v0rbsb;
                  if (here->BSIM4v0rbpd < 1.0e-3)
                      here->BSIM4v0grbpd = 1.0e3;
                  else
                      here->BSIM4v0grbpd = model->BSIM4v0gbmin + 1.0 / here->BSIM4v0rbpd;
              }


              /* 
               * Process geomertry dependent parasitics
	       */

              here->BSIM4v0grgeltd = model->BSIM4v0rshg * (model->BSIM4v0xgw
                      + pParam->BSIM4v0weffCJ / 3.0 / model->BSIM4v0ngcon) /
                      (model->BSIM4v0ngcon * here->BSIM4v0nf *
                      (here->BSIM4v0l - model->BSIM4v0xgl));
              if (here->BSIM4v0grgeltd > 0.0)
                  here->BSIM4v0grgeltd = 1.0 / here->BSIM4v0grgeltd;
              else
              {   here->BSIM4v0grgeltd = 1.0e3; /* mho */
		  if (here->BSIM4v0rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

	      DMCGeff = model->BSIM4v0dmcg - model->BSIM4v0dmcgt;
              DMCIeff = model->BSIM4v0dmci;
              DMDGeff = model->BSIM4v0dmdg - model->BSIM4v0dmcgt;

	      if (here->BSIM4v0sourcePerimeterGiven)
	      {   if (model->BSIM4v0perMod == 0)
	              here->BSIM4v0Pseff = here->BSIM4v0sourcePerimeter;
		  else
		      here->BSIM4v0Pseff = here->BSIM4v0sourcePerimeter 
				       - pParam->BSIM4v0weffCJ * here->BSIM4v0nf;
	      }
	      else
	          BSIM4v0PAeffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0min, 
                                    pParam->BSIM4v0weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &(here->BSIM4v0Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4v0drainPerimeterGiven)
              {   if (model->BSIM4v0perMod == 0)
                      here->BSIM4v0Pdeff = here->BSIM4v0drainPerimeter;
                  else
                      here->BSIM4v0Pdeff = here->BSIM4v0drainPerimeter 
				       - pParam->BSIM4v0weffCJ * here->BSIM4v0nf;
              }
              else
                  BSIM4v0PAeffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0min,
                                    pParam->BSIM4v0weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &(here->BSIM4v0Pdeff), &dumAs, &dumAd);

              if (here->BSIM4v0sourceAreaGiven)
                  here->BSIM4v0Aseff = here->BSIM4v0sourceArea;
              else
                  BSIM4v0PAeffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0min,
                                    pParam->BSIM4v0weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &(here->BSIM4v0Aseff), &dumAd);

              if (here->BSIM4v0drainAreaGiven)
                  here->BSIM4v0Adeff = here->BSIM4v0drainArea;
              else
                  BSIM4v0PAeffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0min,
                                    pParam->BSIM4v0weffCJ, DMCGeff, DMCIeff, DMDGeff,
				    &dumPs, &dumPd, &dumAs, &(here->BSIM4v0Adeff));

	      /* Processing S/D resistance and conductance below */
              if (here->BSIM4v0rgeoMod == 0)
                  here->BSIM4v0sourceConductance = 0.0;
              else if (here->BSIM4v0sourceSquaresGiven)
		  here->BSIM4v0sourceConductance = model->BSIM4v0sheetResistance
                                               * here->BSIM4v0sourceSquares;
	      else	
                  BSIM4v0RdseffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0rgeoMod, here->BSIM4v0min,
                                     pParam->BSIM4v0weffCJ, model->BSIM4v0sheetResistance,
				     DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v0sourceConductance));

              if (here->BSIM4v0rgeoMod == 0)
                  here->BSIM4v0drainConductance = 0.0;
              else if (here->BSIM4v0drainSquaresGiven)
                  here->BSIM4v0drainConductance = model->BSIM4v0sheetResistance
                                              * here->BSIM4v0drainSquares;
              else
                  BSIM4v0RdseffGeo(here->BSIM4v0nf, here->BSIM4v0geoMod, here->BSIM4v0rgeoMod, here->BSIM4v0min,
                                     pParam->BSIM4v0weffCJ, model->BSIM4v0sheetResistance, 
				     DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v0drainConductance));

              if (here->BSIM4v0drainConductance > 0.0)
                  here->BSIM4v0drainConductance = 1.0 / here->BSIM4v0drainConductance;
	      else
                  here->BSIM4v0drainConductance = 0.0;
                  
              if (here->BSIM4v0sourceConductance > 0.0) 
                  here->BSIM4v0sourceConductance = 1.0 / here->BSIM4v0sourceConductance;
	      else
                  here->BSIM4v0sourceConductance = 0.0;


	      if (((here->BSIM4v0rgeoMod != 0) || (model->BSIM4v0rdsMod != 0)
                  || (model->BSIM4v0tnoiMod != 0)) && (here->BSIM4v0sourceConductance == 0.0))
	      {   here->BSIM4v0sourceConductance = 1.0e3; /* mho */
		  printf("Warning: Source conductance reset to 1.0e3 mho.\n");
	      }
              if (((here->BSIM4v0rgeoMod != 0) || (model->BSIM4v0rdsMod != 0)
                  || (model->BSIM4v0tnoiMod != 0)) && (here->BSIM4v0drainConductance == 0.0))
              {   here->BSIM4v0drainConductance = 1.0e3; /* mho */
                  printf("Warning: Drain conductance reset to 1.0e3 mho.\n");
              } /* End of Rsd processing */


              Nvtms = model->BSIM4v0vtm * model->BSIM4v0SjctEmissionCoeff;
              if ((here->BSIM4v0Aseff <= 0.0) && (here->BSIM4v0Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4v0Aseff * model->BSIM4v0SjctTempSatCurDensity
				   + here->BSIM4v0Pseff * model->BSIM4v0SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v0weffCJ * here->BSIM4v0nf
                                   * model->BSIM4v0SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v0dioMod)
                  {   case 0:
			  if ((model->BSIM4v0bvs / Nvtms) > EXP_THRESHOLD)
			      here->BSIM4v0XExpBVS = model->BSIM4v0xjbvs * MIN_EXP;
			  else
	                      here->BSIM4v0XExpBVS = model->BSIM4v0xjbvs * exp(-model->BSIM4v0bvs / Nvtms);	
		          break;
                      case 1:
                          BSIM4v0DioIjthVjmEval(Nvtms, model->BSIM4v0ijthsfwd, SourceSatCurrent, 
			                      0.0, &(here->BSIM4v0vjsmFwd));
                          here->BSIM4v0IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v0vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v0bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v0XExpBVS = model->BSIM4v0xjbvs * MIN_EXP;
			      tmp = MIN_EXP;
			  }
                          else
			  {   here->BSIM4v0XExpBVS = exp(-model->BSIM4v0bvs / Nvtms);
			      tmp = here->BSIM4v0XExpBVS;
		              here->BSIM4v0XExpBVS *= model->BSIM4v0xjbvs;	
			  }

                          BSIM4v0DioIjthVjmEval(Nvtms, model->BSIM4v0ijthsfwd, SourceSatCurrent, 
                               		      here->BSIM4v0XExpBVS, &(here->BSIM4v0vjsmFwd));
		          T0 = exp(here->BSIM4v0vjsmFwd / Nvtms);
                          here->BSIM4v0IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v0XExpBVS / T0
			  		      + here->BSIM4v0XExpBVS - 1.0);
		          here->BSIM4v0SslpFwd = SourceSatCurrent
					       * (T0 + here->BSIM4v0XExpBVS / T0) / Nvtms;

			  T2 = model->BSIM4v0ijthsrev / SourceSatCurrent;
			  if (T2 < 1.0)
			  {   T2 = 10.0;
			      fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
			  } 
                          here->BSIM4v0vjsmRev = -model->BSIM4v0bvs
					     - Nvtms * log((T2 - 1.0) / model->BSIM4v0xjbvs);
			  T1 = model->BSIM4v0xjbvs * exp(-(model->BSIM4v0bvs
			     + here->BSIM4v0vjsmRev) / Nvtms);
			  here->BSIM4v0IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v0SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v0dioMod);
                  }
              }

              Nvtmd = model->BSIM4v0vtm * model->BSIM4v0DjctEmissionCoeff;
	      if ((here->BSIM4v0Adeff <= 0.0) && (here->BSIM4v0Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4v0Adeff * model->BSIM4v0DjctTempSatCurDensity
				  + here->BSIM4v0Pdeff * model->BSIM4v0DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v0weffCJ * here->BSIM4v0nf
                                  * model->BSIM4v0DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v0dioMod)
                  {   case 0:
                          if ((model->BSIM4v0bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v0XExpBVD = model->BSIM4v0xjbvd * MIN_EXP;
                          else
                          here->BSIM4v0XExpBVD = model->BSIM4v0xjbvd * exp(-model->BSIM4v0bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v0DioIjthVjmEval(Nvtmd, model->BSIM4v0ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v0vjdmFwd));
                          here->BSIM4v0IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v0vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v0bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v0XExpBVD = model->BSIM4v0xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v0XExpBVD = exp(-model->BSIM4v0bvd / Nvtmd);
                              tmp = here->BSIM4v0XExpBVD;
                              here->BSIM4v0XExpBVD *= model->BSIM4v0xjbvd;
                          }

                          BSIM4v0DioIjthVjmEval(Nvtmd, model->BSIM4v0ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v0XExpBVD, &(here->BSIM4v0vjdmFwd));
                          T0 = exp(here->BSIM4v0vjdmFwd / Nvtmd);
                          here->BSIM4v0IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v0XExpBVD / T0
                                              + here->BSIM4v0XExpBVD - 1.0);
                          here->BSIM4v0DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v0XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v0ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0) 
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4v0vjdmRev = -model->BSIM4v0bvd
                                             - Nvtms * log((T2 - 1.0) / model->BSIM4v0xjbvd);
                          T1 = model->BSIM4v0xjbvd * exp(-(model->BSIM4v0bvd
                             + here->BSIM4v0vjdmRev) / Nvtmd);
                          here->BSIM4v0IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v0DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v0dioMod);
                  }
              }

              if (BSIM4v0checkModel(model, here, ckt))
              {   IFuid namarray[2];
                  namarray[0] = model->BSIM4v0modName;
                  namarray[1] = here->BSIM4v0name;
                  (*(SPfrontEnd->IFerror)) (ERR_FATAL, "Fatal error(s) detected during BSIM4.0.0 parameter checking for %s in model %s", namarray);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
