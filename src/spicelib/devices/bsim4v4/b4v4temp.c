/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/
/* ngspice multirevision code extension covering 4.2.1 & 4.3.0 & 4.4.0 */
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

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
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


static int
BSIM4v4DioIjthVjmEval(
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
BSIM4v4temp(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v4model *model = (BSIM4v4model*) inModel;
BSIM4v4instance *here;
struct bsim4SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Lnew=0.0, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0=0.0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
double T10;
double Inv_saref, Inv_sbref, Inv_sa, Inv_sb, rho, Ldrn, dvth0_lod;
double W_tmp, Inv_ODeff, OD_offset, dk2_lod, deta0_lod;

int Size_Not_Found, i;

    /*  loop through all the BSIM4v4 device models */
    for (; model != NULL; model = model->BSIM4v4nextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v4SbulkJctPotential < 0.1)  
         {   model->BSIM4v4SbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
         }
         if (model->BSIM4v4SsidewallJctPotential < 0.1)
         {   model->BSIM4v4SsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
         }
         if (model->BSIM4v4SGatesidewallJctPotential < 0.1)
         {   model->BSIM4v4SGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
         }

         if (model->BSIM4v4DbulkJctPotential < 0.1) 
         {   model->BSIM4v4DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v4DsidewallJctPotential < 0.1)
         {   model->BSIM4v4DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v4DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v4DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4v4toxeGiven) && (model->BSIM4v4toxpGiven) && (model->BSIM4v4dtoxGiven)
             && (model->BSIM4v4toxe != (model->BSIM4v4toxp + model->BSIM4v4dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
         else if ((model->BSIM4v4toxeGiven) && (!model->BSIM4v4toxpGiven))
             model->BSIM4v4toxp = model->BSIM4v4toxe - model->BSIM4v4dtox;
         else if ((!model->BSIM4v4toxeGiven) && (model->BSIM4v4toxpGiven))
             model->BSIM4v4toxe = model->BSIM4v4toxp + model->BSIM4v4dtox;

         model->BSIM4v4coxe = model->BSIM4v4epsrox * EPS0 / model->BSIM4v4toxe;
         model->BSIM4v4coxp = model->BSIM4v4epsrox * EPS0 / model->BSIM4v4toxp;

         if (!model->BSIM4v4cgdoGiven)
         {   if (model->BSIM4v4dlcGiven && (model->BSIM4v4dlc > 0.0))
                 model->BSIM4v4cgdo = model->BSIM4v4dlc * model->BSIM4v4coxe
                                  - model->BSIM4v4cgdl ;
             else
                 model->BSIM4v4cgdo = 0.6 * model->BSIM4v4xj * model->BSIM4v4coxe;
         }
         if (!model->BSIM4v4cgsoGiven)
         {   if (model->BSIM4v4dlcGiven && (model->BSIM4v4dlc > 0.0))
                 model->BSIM4v4cgso = model->BSIM4v4dlc * model->BSIM4v4coxe
                                  - model->BSIM4v4cgsl ;
             else
                 model->BSIM4v4cgso = 0.6 * model->BSIM4v4xj * model->BSIM4v4coxe;
         }
         if (!model->BSIM4v4cgboGiven)
             model->BSIM4v4cgbo = 2.0 * model->BSIM4v4dwc * model->BSIM4v4coxe;
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM4v4tnom;
         TRatio = Temp / Tnom;

         model->BSIM4v4vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v4factor1 = sqrt(EPSSI / (model->BSIM4v4epsrox * EPS0)
                             * model->BSIM4v4toxe);

         switch (model->BSIM4v4intVersion) {
           case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
             Vtm0 = KboQ * Tnom;
             break;
           case BSIM4v40:
             Vtm0 = model->BSIM4v4vtm0 = KboQ * Tnom;
             break;
           default: break;
         }
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4v4vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v4vtm;
             T1 = log(Temp / Tnom);
             T2 = T0 + model->BSIM4v4SjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v4SjctEmissionCoeff);
             model->BSIM4v4SjctTempSatCurDensity = model->BSIM4v4SjctSatCurDensity
                                               * T3;
             model->BSIM4v4SjctSidewallTempSatCurDensity
                         = model->BSIM4v4SjctSidewallSatCurDensity * T3;
             model->BSIM4v4SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v4SjctGateSidewallSatCurDensity * T3;

             T2 = T0 + model->BSIM4v4DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v4DjctEmissionCoeff);
             model->BSIM4v4DjctTempSatCurDensity = model->BSIM4v4DjctSatCurDensity
                                               * T3;
             model->BSIM4v4DjctSidewallTempSatCurDensity
                         = model->BSIM4v4DjctSidewallSatCurDensity * T3;
             model->BSIM4v4DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v4DjctGateSidewallSatCurDensity * T3;
         }
         else
         {   model->BSIM4v4SjctTempSatCurDensity = model->BSIM4v4SjctSatCurDensity;
             model->BSIM4v4SjctSidewallTempSatCurDensity
                        = model->BSIM4v4SjctSidewallSatCurDensity;
             model->BSIM4v4SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v4SjctGateSidewallSatCurDensity;
             model->BSIM4v4DjctTempSatCurDensity = model->BSIM4v4DjctSatCurDensity;
             model->BSIM4v4DjctSidewallTempSatCurDensity
                        = model->BSIM4v4DjctSidewallSatCurDensity;
             model->BSIM4v4DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v4DjctGateSidewallSatCurDensity;
         }

         if (model->BSIM4v4SjctTempSatCurDensity < 0.0)
             model->BSIM4v4SjctTempSatCurDensity = 0.0;
         if (model->BSIM4v4SjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v4SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v4SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v4SjctGateSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v4DjctTempSatCurDensity < 0.0)
             model->BSIM4v4DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v4DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v4DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v4DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v4DjctGateSidewallTempSatCurDensity = 0.0;

         /* Temperature dependence of D/B and S/B diode capacitance begins */
         delTemp = ckt->CKTtemp - model->BSIM4v4tnom;
         T0 = model->BSIM4v4tcj * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v4SunitAreaTempJctCap = model->BSIM4v4SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4v4DunitAreaTempJctCap = model->BSIM4v4DunitAreaJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v4SunitAreaJctCap > 0.0)
             {   model->BSIM4v4SunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
             if (model->BSIM4v4DunitAreaJctCap > 0.0)
             {   model->BSIM4v4DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
         }
         T0 = model->BSIM4v4tcjsw * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v4SunitLengthSidewallTempJctCap = model->BSIM4v4SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4v4DunitLengthSidewallTempJctCap = model->BSIM4v4DunitLengthSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v4SunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v4SunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
             }
             if (model->BSIM4v4DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v4DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }        
         }
         T0 = model->BSIM4v4tcjswg * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v4SunitLengthGateSidewallTempJctCap = model->BSIM4v4SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4v4DunitLengthGateSidewallTempJctCap = model->BSIM4v4DunitLengthGateSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v4SunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v4SunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
             }
             if (model->BSIM4v4DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v4DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
         }

         model->BSIM4v4PhiBS = model->BSIM4v4SbulkJctPotential
                           - model->BSIM4v4tpb * delTemp;
         if (model->BSIM4v4PhiBS < 0.01)
         {   model->BSIM4v4PhiBS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
         }
         model->BSIM4v4PhiBD = model->BSIM4v4DbulkJctPotential
                           - model->BSIM4v4tpb * delTemp;
         if (model->BSIM4v4PhiBD < 0.01)
         {   model->BSIM4v4PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v4PhiBSWS = model->BSIM4v4SsidewallJctPotential
                             - model->BSIM4v4tpbsw * delTemp;
         if (model->BSIM4v4PhiBSWS <= 0.01)
         {   model->BSIM4v4PhiBSWS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
         }
         model->BSIM4v4PhiBSWD = model->BSIM4v4DsidewallJctPotential
                             - model->BSIM4v4tpbsw * delTemp;
         if (model->BSIM4v4PhiBSWD <= 0.01)
         {   model->BSIM4v4PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

         model->BSIM4v4PhiBSWGS = model->BSIM4v4SGatesidewallJctPotential
                              - model->BSIM4v4tpbswg * delTemp;
         if (model->BSIM4v4PhiBSWGS <= 0.01)
         {   model->BSIM4v4PhiBSWGS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
         }
         model->BSIM4v4PhiBSWGD = model->BSIM4v4DGatesidewallJctPotential
                              - model->BSIM4v4tpbswg * delTemp;
         if (model->BSIM4v4PhiBSWGD <= 0.01)
         {   model->BSIM4v4PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v4ijthdfwd <= 0.0)
         {   model->BSIM4v4ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v4ijthdfwd);
         }
         if (model->BSIM4v4ijthsfwd <= 0.0)
         {   model->BSIM4v4ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v4ijthsfwd);
         }
         if (model->BSIM4v4ijthdrev <= 0.0)
         {   model->BSIM4v4ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v4ijthdrev);
         }
         if (model->BSIM4v4ijthsrev <= 0.0)
         {   model->BSIM4v4ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v4ijthsrev);
         }

         if ((model->BSIM4v4xjbvd <= 0.0) && (model->BSIM4v4dioMod == 2))
         {   model->BSIM4v4xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v4xjbvd);
         }
         else if ((model->BSIM4v4xjbvd < 0.0) && (model->BSIM4v4dioMod == 0))
         {   model->BSIM4v4xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v4xjbvd);
         }

         if (model->BSIM4v4bvd <= 0.0)
         {   model->BSIM4v4bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v4bvd);
         }

         if ((model->BSIM4v4xjbvs <= 0.0) && (model->BSIM4v4dioMod == 2))
         {   model->BSIM4v4xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v4xjbvs);
         }
         else if ((model->BSIM4v4xjbvs < 0.0) && (model->BSIM4v4dioMod == 0))
         {   model->BSIM4v4xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v4xjbvs);
         }

         if (model->BSIM4v4bvs <= 0.0)
         {   model->BSIM4v4bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v4bvs);
         }


         /* loop through all the instances of the model */
         for (here = model->BSIM4v4instances; here != NULL;
              here = here->BSIM4v4nextInstance) 
           {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM4v4l == pSizeDependParamKnot->Length)
                      && (here->BSIM4v4w == pSizeDependParamKnot->Width)
                      && (here->BSIM4v4nf == pSizeDependParamKnot->NFinger))
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
              Ldrn = here->BSIM4v4l;

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim4SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v4l;
                  pParam->Width = here->BSIM4v4w;
                  pParam->NFinger = here->BSIM4v4nf;
                  Lnew = here->BSIM4v4l  + model->BSIM4v4xl ;
                  Wnew = here->BSIM4v4w / here->BSIM4v4nf + model->BSIM4v4xw;

                  T0 = pow(Lnew, model->BSIM4v4Lln);
                  T1 = pow(Wnew, model->BSIM4v4Lwn);
                  tmp1 = model->BSIM4v4Ll / T0 + model->BSIM4v4Lw / T1
                       + model->BSIM4v4Lwl / (T0 * T1);
                  pParam->BSIM4v4dl = model->BSIM4v4Lint + tmp1;
                  tmp2 = model->BSIM4v4Llc / T0 + model->BSIM4v4Lwc / T1
                       + model->BSIM4v4Lwlc / (T0 * T1);
                  pParam->BSIM4v4dlc = model->BSIM4v4dlc + tmp2;
                  switch (model->BSIM4v4intVersion) {
                    case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
                      pParam->BSIM4v4dlcig = model->BSIM4v4dlcig + tmp2;
                      break;
                    case BSIM4v40:
                      pParam->BSIM4v4dlcig = model->BSIM4v4dlcig;
                      break;
                    default: break;
                  }

                  T2 = pow(Lnew, model->BSIM4v4Wln);
                  T3 = pow(Wnew, model->BSIM4v4Wwn);
                  tmp1 = model->BSIM4v4Wl / T2 + model->BSIM4v4Ww / T3
                       + model->BSIM4v4Wwl / (T2 * T3);
                  pParam->BSIM4v4dw = model->BSIM4v4Wint + tmp1;
                  tmp2 = model->BSIM4v4Wlc / T2 + model->BSIM4v4Wwc / T3
                       + model->BSIM4v4Wwlc / (T2 * T3); 
                  pParam->BSIM4v4dwc = model->BSIM4v4dwc + tmp2;
                  pParam->BSIM4v4dwj = model->BSIM4v4dwj + tmp2;

                  pParam->BSIM4v4leff = Lnew - 2.0 * pParam->BSIM4v4dl;
                  if (pParam->BSIM4v4leff <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v4modName;
                      namarray[1] = here->BSIM4v4name;
                      SPfrontEnd->IFerror (ERR_FATAL,
                      "BSIM4v4: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v4weff = Wnew - 2.0 * pParam->BSIM4v4dw;
                  if (pParam->BSIM4v4weff <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v4modName;
                      namarray[1] = here->BSIM4v4name;
                      SPfrontEnd->IFerror (ERR_FATAL,
                      "BSIM4v4: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v4leffCV = Lnew - 2.0 * pParam->BSIM4v4dlc;
                  if (pParam->BSIM4v4leffCV <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v4modName;
                      namarray[1] = here->BSIM4v4name;
                      SPfrontEnd->IFerror (ERR_FATAL,
                      "BSIM4v4: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v4weffCV = Wnew - 2.0 * pParam->BSIM4v4dwc;
                  if (pParam->BSIM4v4weffCV <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v4modName;
                      namarray[1] = here->BSIM4v4name;
                      SPfrontEnd->IFerror (ERR_FATAL,
                      "BSIM4v4: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v4weffCJ = Wnew - 2.0 * pParam->BSIM4v4dwj;
                  if (pParam->BSIM4v4weffCJ <= 0.0)
                  {   IFuid namarray[2];
                      namarray[0] = model->BSIM4v4modName;
                      namarray[1] = here->BSIM4v4name;
                      SPfrontEnd->IFerror (ERR_FATAL,
                      "BSIM4v4: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       namarray);
                      return(E_BADPARM);
                  }


                  if (model->BSIM4v4binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM4v4leff;
                      Inv_W = 1.0e-6 / pParam->BSIM4v4weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM4v4leff
                             * pParam->BSIM4v4weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM4v4leff;
                      Inv_W = 1.0 / pParam->BSIM4v4weff;
                      Inv_LW = 1.0 / (pParam->BSIM4v4leff
                             * pParam->BSIM4v4weff);
                  }
                  pParam->BSIM4v4cdsc = model->BSIM4v4cdsc
                                    + model->BSIM4v4lcdsc * Inv_L
                                    + model->BSIM4v4wcdsc * Inv_W
                                    + model->BSIM4v4pcdsc * Inv_LW;
                  pParam->BSIM4v4cdscb = model->BSIM4v4cdscb
                                     + model->BSIM4v4lcdscb * Inv_L
                                     + model->BSIM4v4wcdscb * Inv_W
                                     + model->BSIM4v4pcdscb * Inv_LW; 
                                     
                      pParam->BSIM4v4cdscd = model->BSIM4v4cdscd
                                     + model->BSIM4v4lcdscd * Inv_L
                                     + model->BSIM4v4wcdscd * Inv_W
                                     + model->BSIM4v4pcdscd * Inv_LW; 
                                     
                  pParam->BSIM4v4cit = model->BSIM4v4cit
                                   + model->BSIM4v4lcit * Inv_L
                                   + model->BSIM4v4wcit * Inv_W
                                   + model->BSIM4v4pcit * Inv_LW;
                  pParam->BSIM4v4nfactor = model->BSIM4v4nfactor
                                       + model->BSIM4v4lnfactor * Inv_L
                                       + model->BSIM4v4wnfactor * Inv_W
                                       + model->BSIM4v4pnfactor * Inv_LW;
                  pParam->BSIM4v4xj = model->BSIM4v4xj
                                  + model->BSIM4v4lxj * Inv_L
                                  + model->BSIM4v4wxj * Inv_W
                                  + model->BSIM4v4pxj * Inv_LW;
                  pParam->BSIM4v4vsat = model->BSIM4v4vsat
                                    + model->BSIM4v4lvsat * Inv_L
                                    + model->BSIM4v4wvsat * Inv_W
                                    + model->BSIM4v4pvsat * Inv_LW;
                  pParam->BSIM4v4at = model->BSIM4v4at
                                  + model->BSIM4v4lat * Inv_L
                                  + model->BSIM4v4wat * Inv_W
                                  + model->BSIM4v4pat * Inv_LW;
                  pParam->BSIM4v4a0 = model->BSIM4v4a0
                                  + model->BSIM4v4la0 * Inv_L
                                  + model->BSIM4v4wa0 * Inv_W
                                  + model->BSIM4v4pa0 * Inv_LW; 
                                  
                  pParam->BSIM4v4ags = model->BSIM4v4ags
                                  + model->BSIM4v4lags * Inv_L
                                  + model->BSIM4v4wags * Inv_W
                                  + model->BSIM4v4pags * Inv_LW;
                                  
                  pParam->BSIM4v4a1 = model->BSIM4v4a1
                                  + model->BSIM4v4la1 * Inv_L
                                  + model->BSIM4v4wa1 * Inv_W
                                  + model->BSIM4v4pa1 * Inv_LW;
                  pParam->BSIM4v4a2 = model->BSIM4v4a2
                                  + model->BSIM4v4la2 * Inv_L
                                  + model->BSIM4v4wa2 * Inv_W
                                  + model->BSIM4v4pa2 * Inv_LW;
                  pParam->BSIM4v4keta = model->BSIM4v4keta
                                    + model->BSIM4v4lketa * Inv_L
                                    + model->BSIM4v4wketa * Inv_W
                                    + model->BSIM4v4pketa * Inv_LW;
                  pParam->BSIM4v4nsub = model->BSIM4v4nsub
                                    + model->BSIM4v4lnsub * Inv_L
                                    + model->BSIM4v4wnsub * Inv_W
                                    + model->BSIM4v4pnsub * Inv_LW;
                  pParam->BSIM4v4ndep = model->BSIM4v4ndep
                                    + model->BSIM4v4lndep * Inv_L
                                    + model->BSIM4v4wndep * Inv_W
                                    + model->BSIM4v4pndep * Inv_LW;
                  pParam->BSIM4v4nsd = model->BSIM4v4nsd
                                   + model->BSIM4v4lnsd * Inv_L
                                   + model->BSIM4v4wnsd * Inv_W
                                   + model->BSIM4v4pnsd * Inv_LW;
                  pParam->BSIM4v4phin = model->BSIM4v4phin
                                    + model->BSIM4v4lphin * Inv_L
                                    + model->BSIM4v4wphin * Inv_W
                                    + model->BSIM4v4pphin * Inv_LW;
                  pParam->BSIM4v4ngate = model->BSIM4v4ngate
                                     + model->BSIM4v4lngate * Inv_L
                                     + model->BSIM4v4wngate * Inv_W
                                     + model->BSIM4v4pngate * Inv_LW;
                  pParam->BSIM4v4gamma1 = model->BSIM4v4gamma1
                                      + model->BSIM4v4lgamma1 * Inv_L
                                      + model->BSIM4v4wgamma1 * Inv_W
                                      + model->BSIM4v4pgamma1 * Inv_LW;
                  pParam->BSIM4v4gamma2 = model->BSIM4v4gamma2
                                      + model->BSIM4v4lgamma2 * Inv_L
                                      + model->BSIM4v4wgamma2 * Inv_W
                                      + model->BSIM4v4pgamma2 * Inv_LW;
                  pParam->BSIM4v4vbx = model->BSIM4v4vbx
                                   + model->BSIM4v4lvbx * Inv_L
                                   + model->BSIM4v4wvbx * Inv_W
                                   + model->BSIM4v4pvbx * Inv_LW;
                  pParam->BSIM4v4vbm = model->BSIM4v4vbm
                                   + model->BSIM4v4lvbm * Inv_L
                                   + model->BSIM4v4wvbm * Inv_W
                                   + model->BSIM4v4pvbm * Inv_LW;
                  pParam->BSIM4v4xt = model->BSIM4v4xt
                                   + model->BSIM4v4lxt * Inv_L
                                   + model->BSIM4v4wxt * Inv_W
                                   + model->BSIM4v4pxt * Inv_LW;
                  pParam->BSIM4v4vfb = model->BSIM4v4vfb
                                   + model->BSIM4v4lvfb * Inv_L
                                   + model->BSIM4v4wvfb * Inv_W
                                   + model->BSIM4v4pvfb * Inv_LW;
                  pParam->BSIM4v4k1 = model->BSIM4v4k1
                                  + model->BSIM4v4lk1 * Inv_L
                                  + model->BSIM4v4wk1 * Inv_W
                                  + model->BSIM4v4pk1 * Inv_LW;
                  pParam->BSIM4v4kt1 = model->BSIM4v4kt1
                                   + model->BSIM4v4lkt1 * Inv_L
                                   + model->BSIM4v4wkt1 * Inv_W
                                   + model->BSIM4v4pkt1 * Inv_LW;
                  pParam->BSIM4v4kt1l = model->BSIM4v4kt1l
                                    + model->BSIM4v4lkt1l * Inv_L
                                    + model->BSIM4v4wkt1l * Inv_W
                                    + model->BSIM4v4pkt1l * Inv_LW;
                  pParam->BSIM4v4k2 = model->BSIM4v4k2
                                  + model->BSIM4v4lk2 * Inv_L
                                  + model->BSIM4v4wk2 * Inv_W
                                  + model->BSIM4v4pk2 * Inv_LW;
                  pParam->BSIM4v4kt2 = model->BSIM4v4kt2
                                   + model->BSIM4v4lkt2 * Inv_L
                                   + model->BSIM4v4wkt2 * Inv_W
                                   + model->BSIM4v4pkt2 * Inv_LW;
                  pParam->BSIM4v4k3 = model->BSIM4v4k3
                                  + model->BSIM4v4lk3 * Inv_L
                                  + model->BSIM4v4wk3 * Inv_W
                                  + model->BSIM4v4pk3 * Inv_LW;
                  pParam->BSIM4v4k3b = model->BSIM4v4k3b
                                   + model->BSIM4v4lk3b * Inv_L
                                   + model->BSIM4v4wk3b * Inv_W
                                   + model->BSIM4v4pk3b * Inv_LW;
                  pParam->BSIM4v4w0 = model->BSIM4v4w0
                                  + model->BSIM4v4lw0 * Inv_L
                                  + model->BSIM4v4ww0 * Inv_W
                                  + model->BSIM4v4pw0 * Inv_LW;
                  pParam->BSIM4v4lpe0 = model->BSIM4v4lpe0
                                    + model->BSIM4v4llpe0 * Inv_L
                                     + model->BSIM4v4wlpe0 * Inv_W
                                    + model->BSIM4v4plpe0 * Inv_LW;
                  pParam->BSIM4v4lpeb = model->BSIM4v4lpeb
                                    + model->BSIM4v4llpeb * Inv_L
                                    + model->BSIM4v4wlpeb * Inv_W
                                    + model->BSIM4v4plpeb * Inv_LW;
                  pParam->BSIM4v4dvtp0 = model->BSIM4v4dvtp0
                                     + model->BSIM4v4ldvtp0 * Inv_L
                                     + model->BSIM4v4wdvtp0 * Inv_W
                                     + model->BSIM4v4pdvtp0 * Inv_LW;
                  pParam->BSIM4v4dvtp1 = model->BSIM4v4dvtp1
                                     + model->BSIM4v4ldvtp1 * Inv_L
                                     + model->BSIM4v4wdvtp1 * Inv_W
                                     + model->BSIM4v4pdvtp1 * Inv_LW;
                  pParam->BSIM4v4dvt0 = model->BSIM4v4dvt0
                                    + model->BSIM4v4ldvt0 * Inv_L
                                    + model->BSIM4v4wdvt0 * Inv_W
                                    + model->BSIM4v4pdvt0 * Inv_LW;
                  pParam->BSIM4v4dvt1 = model->BSIM4v4dvt1
                                    + model->BSIM4v4ldvt1 * Inv_L
                                    + model->BSIM4v4wdvt1 * Inv_W
                                    + model->BSIM4v4pdvt1 * Inv_LW;
                  pParam->BSIM4v4dvt2 = model->BSIM4v4dvt2
                                    + model->BSIM4v4ldvt2 * Inv_L
                                    + model->BSIM4v4wdvt2 * Inv_W
                                    + model->BSIM4v4pdvt2 * Inv_LW;
                  pParam->BSIM4v4dvt0w = model->BSIM4v4dvt0w
                                    + model->BSIM4v4ldvt0w * Inv_L
                                    + model->BSIM4v4wdvt0w * Inv_W
                                    + model->BSIM4v4pdvt0w * Inv_LW;
                  pParam->BSIM4v4dvt1w = model->BSIM4v4dvt1w
                                    + model->BSIM4v4ldvt1w * Inv_L
                                    + model->BSIM4v4wdvt1w * Inv_W
                                    + model->BSIM4v4pdvt1w * Inv_LW;
                  pParam->BSIM4v4dvt2w = model->BSIM4v4dvt2w
                                    + model->BSIM4v4ldvt2w * Inv_L
                                    + model->BSIM4v4wdvt2w * Inv_W
                                    + model->BSIM4v4pdvt2w * Inv_LW;
                  pParam->BSIM4v4drout = model->BSIM4v4drout
                                     + model->BSIM4v4ldrout * Inv_L
                                     + model->BSIM4v4wdrout * Inv_W
                                     + model->BSIM4v4pdrout * Inv_LW;
                  pParam->BSIM4v4dsub = model->BSIM4v4dsub
                                    + model->BSIM4v4ldsub * Inv_L
                                    + model->BSIM4v4wdsub * Inv_W
                                    + model->BSIM4v4pdsub * Inv_LW;
                  pParam->BSIM4v4vth0 = model->BSIM4v4vth0
                                    + model->BSIM4v4lvth0 * Inv_L
                                    + model->BSIM4v4wvth0 * Inv_W
                                    + model->BSIM4v4pvth0 * Inv_LW;
                  pParam->BSIM4v4ua = model->BSIM4v4ua
                                  + model->BSIM4v4lua * Inv_L
                                  + model->BSIM4v4wua * Inv_W
                                  + model->BSIM4v4pua * Inv_LW;
                  pParam->BSIM4v4ua1 = model->BSIM4v4ua1
                                   + model->BSIM4v4lua1 * Inv_L
                                   + model->BSIM4v4wua1 * Inv_W
                                   + model->BSIM4v4pua1 * Inv_LW;
                  pParam->BSIM4v4ub = model->BSIM4v4ub
                                  + model->BSIM4v4lub * Inv_L
                                  + model->BSIM4v4wub * Inv_W
                                  + model->BSIM4v4pub * Inv_LW;
                  pParam->BSIM4v4ub1 = model->BSIM4v4ub1
                                   + model->BSIM4v4lub1 * Inv_L
                                   + model->BSIM4v4wub1 * Inv_W
                                   + model->BSIM4v4pub1 * Inv_LW;
                  pParam->BSIM4v4uc = model->BSIM4v4uc
                                  + model->BSIM4v4luc * Inv_L
                                  + model->BSIM4v4wuc * Inv_W
                                  + model->BSIM4v4puc * Inv_LW;
                  pParam->BSIM4v4uc1 = model->BSIM4v4uc1
                                   + model->BSIM4v4luc1 * Inv_L
                                   + model->BSIM4v4wuc1 * Inv_W
                                   + model->BSIM4v4puc1 * Inv_LW;
                  pParam->BSIM4v4eu = model->BSIM4v4eu
                                  + model->BSIM4v4leu * Inv_L
                                  + model->BSIM4v4weu * Inv_W
                                  + model->BSIM4v4peu * Inv_LW;
                  pParam->BSIM4v4u0 = model->BSIM4v4u0
                                  + model->BSIM4v4lu0 * Inv_L
                                  + model->BSIM4v4wu0 * Inv_W
                                  + model->BSIM4v4pu0 * Inv_LW;
                  pParam->BSIM4v4ute = model->BSIM4v4ute
                                   + model->BSIM4v4lute * Inv_L
                                   + model->BSIM4v4wute * Inv_W
                                   + model->BSIM4v4pute * Inv_LW;
                  pParam->BSIM4v4voff = model->BSIM4v4voff
                                    + model->BSIM4v4lvoff * Inv_L
                                    + model->BSIM4v4wvoff * Inv_W
                                    + model->BSIM4v4pvoff * Inv_LW;
                  pParam->BSIM4v4minv = model->BSIM4v4minv
                                    + model->BSIM4v4lminv * Inv_L
                                    + model->BSIM4v4wminv * Inv_W
                                    + model->BSIM4v4pminv * Inv_LW;
                  pParam->BSIM4v4fprout = model->BSIM4v4fprout
                                     + model->BSIM4v4lfprout * Inv_L
                                     + model->BSIM4v4wfprout * Inv_W
                                     + model->BSIM4v4pfprout * Inv_LW;
                  pParam->BSIM4v4pdits = model->BSIM4v4pdits
                                     + model->BSIM4v4lpdits * Inv_L
                                     + model->BSIM4v4wpdits * Inv_W
                                     + model->BSIM4v4ppdits * Inv_LW;
                  pParam->BSIM4v4pditsd = model->BSIM4v4pditsd
                                      + model->BSIM4v4lpditsd * Inv_L
                                      + model->BSIM4v4wpditsd * Inv_W
                                      + model->BSIM4v4ppditsd * Inv_LW;
                  pParam->BSIM4v4delta = model->BSIM4v4delta
                                     + model->BSIM4v4ldelta * Inv_L
                                     + model->BSIM4v4wdelta * Inv_W
                                     + model->BSIM4v4pdelta * Inv_LW;
                  pParam->BSIM4v4rdsw = model->BSIM4v4rdsw
                                    + model->BSIM4v4lrdsw * Inv_L
                                    + model->BSIM4v4wrdsw * Inv_W
                                    + model->BSIM4v4prdsw * Inv_LW;
                  pParam->BSIM4v4rdw = model->BSIM4v4rdw
                                    + model->BSIM4v4lrdw * Inv_L
                                    + model->BSIM4v4wrdw * Inv_W
                                    + model->BSIM4v4prdw * Inv_LW;
                  pParam->BSIM4v4rsw = model->BSIM4v4rsw
                                    + model->BSIM4v4lrsw * Inv_L
                                    + model->BSIM4v4wrsw * Inv_W
                                    + model->BSIM4v4prsw * Inv_LW;
                  pParam->BSIM4v4prwg = model->BSIM4v4prwg
                                    + model->BSIM4v4lprwg * Inv_L
                                    + model->BSIM4v4wprwg * Inv_W
                                    + model->BSIM4v4pprwg * Inv_LW;
                  pParam->BSIM4v4prwb = model->BSIM4v4prwb
                                    + model->BSIM4v4lprwb * Inv_L
                                    + model->BSIM4v4wprwb * Inv_W
                                    + model->BSIM4v4pprwb * Inv_LW;
                  pParam->BSIM4v4prt = model->BSIM4v4prt
                                    + model->BSIM4v4lprt * Inv_L
                                    + model->BSIM4v4wprt * Inv_W
                                    + model->BSIM4v4pprt * Inv_LW;
                  pParam->BSIM4v4eta0 = model->BSIM4v4eta0
                                    + model->BSIM4v4leta0 * Inv_L
                                    + model->BSIM4v4weta0 * Inv_W
                                    + model->BSIM4v4peta0 * Inv_LW;
                  pParam->BSIM4v4etab = model->BSIM4v4etab
                                    + model->BSIM4v4letab * Inv_L
                                    + model->BSIM4v4wetab * Inv_W
                                    + model->BSIM4v4petab * Inv_LW;
                  pParam->BSIM4v4pclm = model->BSIM4v4pclm
                                    + model->BSIM4v4lpclm * Inv_L
                                    + model->BSIM4v4wpclm * Inv_W
                                    + model->BSIM4v4ppclm * Inv_LW;
                  pParam->BSIM4v4pdibl1 = model->BSIM4v4pdibl1
                                      + model->BSIM4v4lpdibl1 * Inv_L
                                      + model->BSIM4v4wpdibl1 * Inv_W
                                      + model->BSIM4v4ppdibl1 * Inv_LW;
                  pParam->BSIM4v4pdibl2 = model->BSIM4v4pdibl2
                                      + model->BSIM4v4lpdibl2 * Inv_L
                                      + model->BSIM4v4wpdibl2 * Inv_W
                                      + model->BSIM4v4ppdibl2 * Inv_LW;
                  pParam->BSIM4v4pdiblb = model->BSIM4v4pdiblb
                                      + model->BSIM4v4lpdiblb * Inv_L
                                      + model->BSIM4v4wpdiblb * Inv_W
                                      + model->BSIM4v4ppdiblb * Inv_LW;
                  pParam->BSIM4v4pscbe1 = model->BSIM4v4pscbe1
                                      + model->BSIM4v4lpscbe1 * Inv_L
                                      + model->BSIM4v4wpscbe1 * Inv_W
                                      + model->BSIM4v4ppscbe1 * Inv_LW;
                  pParam->BSIM4v4pscbe2 = model->BSIM4v4pscbe2
                                      + model->BSIM4v4lpscbe2 * Inv_L
                                      + model->BSIM4v4wpscbe2 * Inv_W
                                      + model->BSIM4v4ppscbe2 * Inv_LW;
                  pParam->BSIM4v4pvag = model->BSIM4v4pvag
                                    + model->BSIM4v4lpvag * Inv_L
                                    + model->BSIM4v4wpvag * Inv_W
                                    + model->BSIM4v4ppvag * Inv_LW;
                  pParam->BSIM4v4wr = model->BSIM4v4wr
                                  + model->BSIM4v4lwr * Inv_L
                                  + model->BSIM4v4wwr * Inv_W
                                  + model->BSIM4v4pwr * Inv_LW;
                  pParam->BSIM4v4dwg = model->BSIM4v4dwg
                                   + model->BSIM4v4ldwg * Inv_L
                                   + model->BSIM4v4wdwg * Inv_W
                                   + model->BSIM4v4pdwg * Inv_LW;
                  pParam->BSIM4v4dwb = model->BSIM4v4dwb
                                   + model->BSIM4v4ldwb * Inv_L
                                   + model->BSIM4v4wdwb * Inv_W
                                   + model->BSIM4v4pdwb * Inv_LW;
                  pParam->BSIM4v4b0 = model->BSIM4v4b0
                                  + model->BSIM4v4lb0 * Inv_L
                                  + model->BSIM4v4wb0 * Inv_W
                                  + model->BSIM4v4pb0 * Inv_LW;
                  pParam->BSIM4v4b1 = model->BSIM4v4b1
                                  + model->BSIM4v4lb1 * Inv_L
                                  + model->BSIM4v4wb1 * Inv_W
                                  + model->BSIM4v4pb1 * Inv_LW;
                  pParam->BSIM4v4alpha0 = model->BSIM4v4alpha0
                                      + model->BSIM4v4lalpha0 * Inv_L
                                      + model->BSIM4v4walpha0 * Inv_W
                                      + model->BSIM4v4palpha0 * Inv_LW;
                  pParam->BSIM4v4alpha1 = model->BSIM4v4alpha1
                                      + model->BSIM4v4lalpha1 * Inv_L
                                      + model->BSIM4v4walpha1 * Inv_W
                                      + model->BSIM4v4palpha1 * Inv_LW;
                  pParam->BSIM4v4beta0 = model->BSIM4v4beta0
                                     + model->BSIM4v4lbeta0 * Inv_L
                                     + model->BSIM4v4wbeta0 * Inv_W
                                     + model->BSIM4v4pbeta0 * Inv_LW;
                  pParam->BSIM4v4agidl = model->BSIM4v4agidl
                                     + model->BSIM4v4lagidl * Inv_L
                                     + model->BSIM4v4wagidl * Inv_W
                                     + model->BSIM4v4pagidl * Inv_LW;
                  pParam->BSIM4v4bgidl = model->BSIM4v4bgidl
                                     + model->BSIM4v4lbgidl * Inv_L
                                     + model->BSIM4v4wbgidl * Inv_W
                                     + model->BSIM4v4pbgidl * Inv_LW;
                  pParam->BSIM4v4cgidl = model->BSIM4v4cgidl
                                     + model->BSIM4v4lcgidl * Inv_L
                                     + model->BSIM4v4wcgidl * Inv_W
                                     + model->BSIM4v4pcgidl * Inv_LW;
                  pParam->BSIM4v4egidl = model->BSIM4v4egidl
                                     + model->BSIM4v4legidl * Inv_L
                                     + model->BSIM4v4wegidl * Inv_W
                                     + model->BSIM4v4pegidl * Inv_LW;
                  pParam->BSIM4v4aigc = model->BSIM4v4aigc
                                     + model->BSIM4v4laigc * Inv_L
                                     + model->BSIM4v4waigc * Inv_W
                                     + model->BSIM4v4paigc * Inv_LW;
                  pParam->BSIM4v4bigc = model->BSIM4v4bigc
                                     + model->BSIM4v4lbigc * Inv_L
                                     + model->BSIM4v4wbigc * Inv_W
                                     + model->BSIM4v4pbigc * Inv_LW;
                  pParam->BSIM4v4cigc = model->BSIM4v4cigc
                                     + model->BSIM4v4lcigc * Inv_L
                                     + model->BSIM4v4wcigc * Inv_W
                                     + model->BSIM4v4pcigc * Inv_LW;
                  pParam->BSIM4v4aigsd = model->BSIM4v4aigsd
                                     + model->BSIM4v4laigsd * Inv_L
                                     + model->BSIM4v4waigsd * Inv_W
                                     + model->BSIM4v4paigsd * Inv_LW;
                  pParam->BSIM4v4bigsd = model->BSIM4v4bigsd
                                     + model->BSIM4v4lbigsd * Inv_L
                                     + model->BSIM4v4wbigsd * Inv_W
                                     + model->BSIM4v4pbigsd * Inv_LW;
                  pParam->BSIM4v4cigsd = model->BSIM4v4cigsd
                                     + model->BSIM4v4lcigsd * Inv_L
                                     + model->BSIM4v4wcigsd * Inv_W
                                     + model->BSIM4v4pcigsd * Inv_LW;
                  pParam->BSIM4v4aigbacc = model->BSIM4v4aigbacc
                                       + model->BSIM4v4laigbacc * Inv_L
                                       + model->BSIM4v4waigbacc * Inv_W
                                       + model->BSIM4v4paigbacc * Inv_LW;
                  pParam->BSIM4v4bigbacc = model->BSIM4v4bigbacc
                                       + model->BSIM4v4lbigbacc * Inv_L
                                       + model->BSIM4v4wbigbacc * Inv_W
                                       + model->BSIM4v4pbigbacc * Inv_LW;
                  pParam->BSIM4v4cigbacc = model->BSIM4v4cigbacc
                                       + model->BSIM4v4lcigbacc * Inv_L
                                       + model->BSIM4v4wcigbacc * Inv_W
                                       + model->BSIM4v4pcigbacc * Inv_LW;
                  pParam->BSIM4v4aigbinv = model->BSIM4v4aigbinv
                                       + model->BSIM4v4laigbinv * Inv_L
                                       + model->BSIM4v4waigbinv * Inv_W
                                       + model->BSIM4v4paigbinv * Inv_LW;
                  pParam->BSIM4v4bigbinv = model->BSIM4v4bigbinv
                                       + model->BSIM4v4lbigbinv * Inv_L
                                       + model->BSIM4v4wbigbinv * Inv_W
                                       + model->BSIM4v4pbigbinv * Inv_LW;
                  pParam->BSIM4v4cigbinv = model->BSIM4v4cigbinv
                                       + model->BSIM4v4lcigbinv * Inv_L
                                       + model->BSIM4v4wcigbinv * Inv_W
                                       + model->BSIM4v4pcigbinv * Inv_LW;
                  pParam->BSIM4v4nigc = model->BSIM4v4nigc
                                       + model->BSIM4v4lnigc * Inv_L
                                       + model->BSIM4v4wnigc * Inv_W
                                       + model->BSIM4v4pnigc * Inv_LW;
                  pParam->BSIM4v4nigbacc = model->BSIM4v4nigbacc
                                       + model->BSIM4v4lnigbacc * Inv_L
                                       + model->BSIM4v4wnigbacc * Inv_W
                                       + model->BSIM4v4pnigbacc * Inv_LW;
                  pParam->BSIM4v4nigbinv = model->BSIM4v4nigbinv
                                       + model->BSIM4v4lnigbinv * Inv_L
                                       + model->BSIM4v4wnigbinv * Inv_W
                                       + model->BSIM4v4pnigbinv * Inv_LW;
                  pParam->BSIM4v4ntox = model->BSIM4v4ntox
                                    + model->BSIM4v4lntox * Inv_L
                                    + model->BSIM4v4wntox * Inv_W
                                    + model->BSIM4v4pntox * Inv_LW;
                  pParam->BSIM4v4eigbinv = model->BSIM4v4eigbinv
                                       + model->BSIM4v4leigbinv * Inv_L
                                       + model->BSIM4v4weigbinv * Inv_W
                                       + model->BSIM4v4peigbinv * Inv_LW;
                  pParam->BSIM4v4pigcd = model->BSIM4v4pigcd
                                     + model->BSIM4v4lpigcd * Inv_L
                                     + model->BSIM4v4wpigcd * Inv_W
                                     + model->BSIM4v4ppigcd * Inv_LW;
                  pParam->BSIM4v4poxedge = model->BSIM4v4poxedge
                                       + model->BSIM4v4lpoxedge * Inv_L
                                       + model->BSIM4v4wpoxedge * Inv_W
                                       + model->BSIM4v4ppoxedge * Inv_LW;
                  pParam->BSIM4v4xrcrg1 = model->BSIM4v4xrcrg1
                                      + model->BSIM4v4lxrcrg1 * Inv_L
                                      + model->BSIM4v4wxrcrg1 * Inv_W
                                      + model->BSIM4v4pxrcrg1 * Inv_LW;
                  pParam->BSIM4v4xrcrg2 = model->BSIM4v4xrcrg2
                                      + model->BSIM4v4lxrcrg2 * Inv_L
                                      + model->BSIM4v4wxrcrg2 * Inv_W
                                      + model->BSIM4v4pxrcrg2 * Inv_LW;
                  switch (model->BSIM4v4intVersion) {
                    case BSIM4vOLD: case BSIM4v21:
                      break;
                    case BSIM4v30:
                      pParam->BSIM4v4lambda = model->BSIM4v4lambda
                                          + model->BSIM4v4llambda * Inv_L
                                          + model->BSIM4v4wlambda * Inv_W
                                          + model->BSIM4v4plambda * Inv_LW;
                      pParam->BSIM4v4vtl = model->BSIM4v4vtl
                                          + model->BSIM4v4lvtl * Inv_L
                                          + model->BSIM4v4wvtl * Inv_W
                                          + model->BSIM4v4pvtl * Inv_LW;
                      pParam->BSIM4v4xn = model->BSIM4v4xn
                                          + model->BSIM4v4lxn * Inv_L
                                          + model->BSIM4v4wxn * Inv_W
                                          + model->BSIM4v4pxn * Inv_LW;
                      break;
                    case BSIM4v40:
                      pParam->BSIM4v4lambda = model->BSIM4v4lambda
                                          + model->BSIM4v4llambda * Inv_L
                                          + model->BSIM4v4wlambda * Inv_W
                                          + model->BSIM4v4plambda * Inv_LW;
                      pParam->BSIM4v4vtl = model->BSIM4v4vtl
                                          + model->BSIM4v4lvtl * Inv_L
                                          + model->BSIM4v4wvtl * Inv_W
                                          + model->BSIM4v4pvtl * Inv_LW;
                      pParam->BSIM4v4xn = model->BSIM4v4xn
                                          + model->BSIM4v4lxn * Inv_L
                                          + model->BSIM4v4wxn * Inv_W
                                          + model->BSIM4v4pxn * Inv_LW;
                      pParam->BSIM4v4vfbsdoff = model->BSIM4v4vfbsdoff
                                          + model->BSIM4v4lvfbsdoff * Inv_L
                                          + model->BSIM4v4wvfbsdoff * Inv_W
                                          + model->BSIM4v4pvfbsdoff * Inv_LW;
                      break;
                    default: break;
                  }

                  pParam->BSIM4v4cgsl = model->BSIM4v4cgsl
                                    + model->BSIM4v4lcgsl * Inv_L
                                    + model->BSIM4v4wcgsl * Inv_W
                                    + model->BSIM4v4pcgsl * Inv_LW;
                  pParam->BSIM4v4cgdl = model->BSIM4v4cgdl
                                    + model->BSIM4v4lcgdl * Inv_L
                                    + model->BSIM4v4wcgdl * Inv_W
                                    + model->BSIM4v4pcgdl * Inv_LW;
                  pParam->BSIM4v4ckappas = model->BSIM4v4ckappas
                                       + model->BSIM4v4lckappas * Inv_L
                                       + model->BSIM4v4wckappas * Inv_W
                                        + model->BSIM4v4pckappas * Inv_LW;
                  pParam->BSIM4v4ckappad = model->BSIM4v4ckappad
                                       + model->BSIM4v4lckappad * Inv_L
                                       + model->BSIM4v4wckappad * Inv_W
                                       + model->BSIM4v4pckappad * Inv_LW;
                  pParam->BSIM4v4cf = model->BSIM4v4cf
                                  + model->BSIM4v4lcf * Inv_L
                                  + model->BSIM4v4wcf * Inv_W
                                  + model->BSIM4v4pcf * Inv_LW;
                  pParam->BSIM4v4clc = model->BSIM4v4clc
                                   + model->BSIM4v4lclc * Inv_L
                                   + model->BSIM4v4wclc * Inv_W
                                   + model->BSIM4v4pclc * Inv_LW;
                  pParam->BSIM4v4cle = model->BSIM4v4cle
                                   + model->BSIM4v4lcle * Inv_L
                                   + model->BSIM4v4wcle * Inv_W
                                   + model->BSIM4v4pcle * Inv_LW;
                  pParam->BSIM4v4vfbcv = model->BSIM4v4vfbcv
                                     + model->BSIM4v4lvfbcv * Inv_L
                                     + model->BSIM4v4wvfbcv * Inv_W
                                     + model->BSIM4v4pvfbcv * Inv_LW;
                  pParam->BSIM4v4acde = model->BSIM4v4acde
                                    + model->BSIM4v4lacde * Inv_L
                                    + model->BSIM4v4wacde * Inv_W
                                    + model->BSIM4v4pacde * Inv_LW;
                  pParam->BSIM4v4moin = model->BSIM4v4moin
                                    + model->BSIM4v4lmoin * Inv_L
                                    + model->BSIM4v4wmoin * Inv_W
                                    + model->BSIM4v4pmoin * Inv_LW;
                  pParam->BSIM4v4noff = model->BSIM4v4noff
                                    + model->BSIM4v4lnoff * Inv_L
                                    + model->BSIM4v4wnoff * Inv_W
                                    + model->BSIM4v4pnoff * Inv_LW;
                  pParam->BSIM4v4voffcv = model->BSIM4v4voffcv
                                      + model->BSIM4v4lvoffcv * Inv_L
                                      + model->BSIM4v4wvoffcv * Inv_W
                                      + model->BSIM4v4pvoffcv * Inv_LW;

                  pParam->BSIM4v4abulkCVfactor = 1.0 + pow((pParam->BSIM4v4clc
                                             / pParam->BSIM4v4leffCV),
                                             pParam->BSIM4v4cle);

                  T0 = (TRatio - 1.0);

                  switch (model->BSIM4v4intVersion) {
                    case BSIM4vOLD: case BSIM4v21:
                      pParam->BSIM4v4ua = pParam->BSIM4v4ua + pParam->BSIM4v4ua1 * T0;
                      pParam->BSIM4v4ub = pParam->BSIM4v4ub + pParam->BSIM4v4ub1 * T0;
                      pParam->BSIM4v4uc = pParam->BSIM4v4uc + pParam->BSIM4v4uc1 * T0;
                      if (pParam->BSIM4v4u0 > 1.0) 
                          pParam->BSIM4v4u0 = pParam->BSIM4v4u0 / 1.0e4;

                     
                      pParam->BSIM4v4u0temp = pParam->BSIM4v4u0
                                          * pow(TRatio, pParam->BSIM4v4ute); 
                      pParam->BSIM4v4vsattemp = pParam->BSIM4v4vsat - pParam->BSIM4v4at 
                                            * T0;
                      if (pParam->BSIM4v4eu < 0.0)
                      {   pParam->BSIM4v4eu = 0.0;
                          printf("Warning: eu has been negative; reset to 0.0.\n");
                      }

                     
                      PowWeffWr = pow(pParam->BSIM4v4weffCJ * 1.0e6, pParam->BSIM4v4wr) * here->BSIM4v4nf;
                      /* External Rd(V) */
                      T1 = pParam->BSIM4v4rdw + pParam->BSIM4v4prt * T0;
                      if (T1 < 0.0)
                      {   T1 = 0.0;
                          printf("Warning: Rdw at current temperature is negative; set to 0.\n");
                      }
                      T2 = model->BSIM4v4rdwmin + pParam->BSIM4v4prt * T0;
                      if (T2 < 0.0)
                      {   T2 = 0.0;
                          printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                      }
                      pParam->BSIM4v4rd0 = T1 / PowWeffWr;
                      pParam->BSIM4v4rdwmin = T2 / PowWeffWr;

                                         
                      /* External Rs(V) */
                      T1 = pParam->BSIM4v4rsw + pParam->BSIM4v4prt * T0;
                      if (T1 < 0.0)
                      {   T1 = 0.0;
                          printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                      }
                      T2 = model->BSIM4v4rswmin + pParam->BSIM4v4prt * T0;
                      if (T2 < 0.0)
                      {   T2 = 0.0;
                          printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                      }
                      pParam->BSIM4v4rs0 = T1 / PowWeffWr;
                      pParam->BSIM4v4rswmin = T2 / PowWeffWr;

                     
                      /* Internal Rds(V) in IV */
                      pParam->BSIM4v4rds0 = (pParam->BSIM4v4rdsw + pParam->BSIM4v4prt * T0)
                                        * here->BSIM4v4nf / PowWeffWr;
                      pParam->BSIM4v4rdswmin = (model->BSIM4v4rdswmin + pParam->BSIM4v4prt * T0)
                                       * here->BSIM4v4nf / PowWeffWr;
                      break;
                    case BSIM4v30: case BSIM4v40:
                      PowWeffWr = pow(pParam->BSIM4v4weffCJ * 1.0e6, pParam->BSIM4v4wr) * here->BSIM4v4nf;
                      
                      T1 = T2 = T3 = T4 = 0.0;
                      if(model->BSIM4v4tempMod == 0) {
                              pParam->BSIM4v4ua = pParam->BSIM4v4ua + pParam->BSIM4v4ua1 * T0;
                              pParam->BSIM4v4ub = pParam->BSIM4v4ub + pParam->BSIM4v4ub1 * T0;
                              pParam->BSIM4v4uc = pParam->BSIM4v4uc + pParam->BSIM4v4uc1 * T0;
                              pParam->BSIM4v4vsattemp = pParam->BSIM4v4vsat - pParam->BSIM4v4at * T0;
                              T10 = pParam->BSIM4v4prt * T0;
                         if(model->BSIM4v4rdsMod) {
                              /* External Rd(V) */
                              T1 = pParam->BSIM4v4rdw + T10;
                              T2 = model->BSIM4v4rdwmin + T10;
                              /* External Rs(V) */
                              T3 = pParam->BSIM4v4rsw + T10;
                              T4 = model->BSIM4v4rswmin + T10;
                         }
                              /* Internal Rds(V) in IV */
                              pParam->BSIM4v4rds0 = (pParam->BSIM4v4rdsw + T10)
                                                * here->BSIM4v4nf / PowWeffWr;
                              pParam->BSIM4v4rdswmin = (model->BSIM4v4rdswmin + T10)
                                                   * here->BSIM4v4nf / PowWeffWr;
                      } else { /* tempMod = 1 */
                              pParam->BSIM4v4ua = pParam->BSIM4v4ua * (1.0 + pParam->BSIM4v4ua1 * delTemp) ;
                              pParam->BSIM4v4ub = pParam->BSIM4v4ub * (1.0 + pParam->BSIM4v4ub1 * delTemp);
                              pParam->BSIM4v4uc = pParam->BSIM4v4uc * (1.0 + pParam->BSIM4v4uc1 * delTemp);
                              pParam->BSIM4v4vsattemp = pParam->BSIM4v4vsat * (1.0 - pParam->BSIM4v4at * delTemp);
                              T10 = 1.0 + pParam->BSIM4v4prt * delTemp;
                         if(model->BSIM4v4rdsMod) {
                              /* External Rd(V) */
                              T1 = pParam->BSIM4v4rdw * T10;
                              T2 = model->BSIM4v4rdwmin * T10;
                              /* External Rs(V) */
                              T3 = pParam->BSIM4v4rsw * T10;
                              T4 = model->BSIM4v4rswmin * T10;
                         }
                              /* Internal Rds(V) in IV */
                              pParam->BSIM4v4rds0 = pParam->BSIM4v4rdsw * T10 * here->BSIM4v4nf / PowWeffWr;
                              pParam->BSIM4v4rdswmin = model->BSIM4v4rdswmin * T10 * here->BSIM4v4nf / PowWeffWr;
                      }
                      if (T1 < 0.0)
                      {   T1 = 0.0;
                          printf("Warning: Rdw at current temperature is negative; set to 0.\n");
                      }
                      if (T2 < 0.0)
                      {   T2 = 0.0;
                          printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                      }
                      pParam->BSIM4v4rd0 = T1 / PowWeffWr;
                      pParam->BSIM4v4rdwmin = T2 / PowWeffWr;
                      if (T3 < 0.0)
                      {   T3 = 0.0;
                          printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                      }
                      if (T4 < 0.0)
                      {   T4 = 0.0;
                          printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                      }
                      pParam->BSIM4v4rs0 = T3 / PowWeffWr;
                      pParam->BSIM4v4rswmin = T4 / PowWeffWr;
                      
                      if (pParam->BSIM4v4u0 > 1.0) 
                          pParam->BSIM4v4u0 = pParam->BSIM4v4u0 / 1.0e4;
                      
                      pParam->BSIM4v4u0temp = pParam->BSIM4v4u0
                                          * pow(TRatio, pParam->BSIM4v4ute); 
                      if (pParam->BSIM4v4eu < 0.0)
                      {   pParam->BSIM4v4eu = 0.0;
                          printf("Warning: eu has been negative; reset to 0.0.\n");
                      }

                      /* Source End Velocity Limit  */
                      if((model->BSIM4v4vtlGiven) && (model->BSIM4v4vtl > 0.0) )
                      {  
                        if(model->BSIM4v4lc < 0.0) pParam->BSIM4v4lc = 0.0;
                        else   pParam->BSIM4v4lc = model->BSIM4v4lc ;
                        T0 = pParam->BSIM4v4leff / (pParam->BSIM4v4xn * pParam->BSIM4v4leff + pParam->BSIM4v4lc);
                        pParam->BSIM4v4tfactor = (1.0 - T0) / (1.0 + T0 );
                      }
                      break;
                    default: break;
                  }

                  pParam->BSIM4v4cgdo = (model->BSIM4v4cgdo + pParam->BSIM4v4cf)
                                    * pParam->BSIM4v4weffCV;
                  pParam->BSIM4v4cgso = (model->BSIM4v4cgso + pParam->BSIM4v4cf)
                                    * pParam->BSIM4v4weffCV;
                  pParam->BSIM4v4cgbo = model->BSIM4v4cgbo * pParam->BSIM4v4leffCV * here->BSIM4v4nf;

                  if (!model->BSIM4v4ndepGiven && model->BSIM4v4gamma1Given)
                  {   T0 = pParam->BSIM4v4gamma1 * model->BSIM4v4coxe;
                      pParam->BSIM4v4ndep = 3.01248e22 * T0 * T0;
                  }

                  pParam->BSIM4v4phi = Vtm0 * log(pParam->BSIM4v4ndep / ni)
                                   + pParam->BSIM4v4phin + 0.4;

                  pParam->BSIM4v4sqrtPhi = sqrt(pParam->BSIM4v4phi);
                  pParam->BSIM4v4phis3 = pParam->BSIM4v4sqrtPhi * pParam->BSIM4v4phi;

                  pParam->BSIM4v4Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM4v4ndep * 1.0e6))
                                     * pParam->BSIM4v4sqrtPhi; 
                  pParam->BSIM4v4sqrtXdep0 = sqrt(pParam->BSIM4v4Xdep0);
                  pParam->BSIM4v4litl = sqrt(3.0 * pParam->BSIM4v4xj
                                    * model->BSIM4v4toxe);
                  pParam->BSIM4v4vbi = Vtm0 * log(pParam->BSIM4v4nsd
                                   * pParam->BSIM4v4ndep / (ni * ni));

                  if (pParam->BSIM4v4ngate > 0.0)
                  {   pParam->BSIM4v4vfbsd = Vtm0 * log(pParam->BSIM4v4ngate
                                         / pParam->BSIM4v4nsd);
                  }
                  else
                      pParam->BSIM4v4vfbsd = 0.0;

                  pParam->BSIM4v4cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM4v4ndep * 1.0e6 / 2.0
                                     / pParam->BSIM4v4phi);

                  pParam->BSIM4v4ToxRatio = exp(pParam->BSIM4v4ntox
                                        * log(model->BSIM4v4toxref / model->BSIM4v4toxe))
                                        / model->BSIM4v4toxe / model->BSIM4v4toxe;
                  pParam->BSIM4v4ToxRatioEdge = exp(pParam->BSIM4v4ntox
                                            * log(model->BSIM4v4toxref
                                            / (model->BSIM4v4toxe * pParam->BSIM4v4poxedge)))
                                            / model->BSIM4v4toxe / model->BSIM4v4toxe
                                            / pParam->BSIM4v4poxedge / pParam->BSIM4v4poxedge;
                  pParam->BSIM4v4Aechvb = (model->BSIM4v4type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v4Bechvb = (model->BSIM4v4type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v4AechvbEdge = pParam->BSIM4v4Aechvb * pParam->BSIM4v4weff
                                          * pParam->BSIM4v4dlcig * pParam->BSIM4v4ToxRatioEdge;
                  pParam->BSIM4v4BechvbEdge = -pParam->BSIM4v4Bechvb
                                          * model->BSIM4v4toxe * pParam->BSIM4v4poxedge;
                  pParam->BSIM4v4Aechvb *= pParam->BSIM4v4weff * pParam->BSIM4v4leff
                                       * pParam->BSIM4v4ToxRatio;
                  pParam->BSIM4v4Bechvb *= -model->BSIM4v4toxe;


                  pParam->BSIM4v4mstar = 0.5 + atan(pParam->BSIM4v4minv) / PI;
                  pParam->BSIM4v4voffcbn =  pParam->BSIM4v4voff + model->BSIM4v4voffl / pParam->BSIM4v4leff;

                  pParam->BSIM4v4ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4v4ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v4acde *= pow((pParam->BSIM4v4ndep / 2.0e16), -0.25);


                  if (model->BSIM4v4k1Given || model->BSIM4v4k2Given)
                  {   if (!model->BSIM4v4k1Given)
                      {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v4k1 = 0.53;
                      }
                      if (!model->BSIM4v4k2Given)
                      {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v4k2 = -0.0186;
                      }
                      if (model->BSIM4v4nsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v4xtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v4vbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v4gamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM4v4gamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
                  {   if (!model->BSIM4v4vbxGiven)
                          pParam->BSIM4v4vbx = pParam->BSIM4v4phi - 7.7348e-4 
                                           * pParam->BSIM4v4ndep
                                           * pParam->BSIM4v4xt * pParam->BSIM4v4xt;
                      if (pParam->BSIM4v4vbx > 0.0)
                          pParam->BSIM4v4vbx = -pParam->BSIM4v4vbx;
                      if (pParam->BSIM4v4vbm > 0.0)
                          pParam->BSIM4v4vbm = -pParam->BSIM4v4vbm;
           
                      if (!model->BSIM4v4gamma1Given)
                          pParam->BSIM4v4gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM4v4ndep)
                                              / model->BSIM4v4coxe;
                      if (!model->BSIM4v4gamma2Given)
                          pParam->BSIM4v4gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM4v4nsub)
                                              / model->BSIM4v4coxe;

                      T0 = pParam->BSIM4v4gamma1 - pParam->BSIM4v4gamma2;
                      T1 = sqrt(pParam->BSIM4v4phi - pParam->BSIM4v4vbx)
                         - pParam->BSIM4v4sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v4phi * (pParam->BSIM4v4phi
                         - pParam->BSIM4v4vbm)) - pParam->BSIM4v4phi;
                      pParam->BSIM4v4k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v4vbm);
                      pParam->BSIM4v4k1 = pParam->BSIM4v4gamma2 - 2.0
                                      * pParam->BSIM4v4k2 * sqrt(pParam->BSIM4v4phi
                                      - pParam->BSIM4v4vbm);
                  }
 
                  if (pParam->BSIM4v4k2 < 0.0)
                  {   T0 = 0.5 * pParam->BSIM4v4k1 / pParam->BSIM4v4k2;
                      pParam->BSIM4v4vbsc = 0.9 * (pParam->BSIM4v4phi - T0 * T0);
                      if (pParam->BSIM4v4vbsc > -3.0)
                          pParam->BSIM4v4vbsc = -3.0;
                      else if (pParam->BSIM4v4vbsc < -30.0)
                          pParam->BSIM4v4vbsc = -30.0;
                  }
                  else
                  {   pParam->BSIM4v4vbsc = -30.0;
                  }
                  if (pParam->BSIM4v4vbsc > pParam->BSIM4v4vbm)
                      pParam->BSIM4v4vbsc = pParam->BSIM4v4vbm;

                  if (!model->BSIM4v4vfbGiven)
                  {   if (model->BSIM4v4vth0Given)
                      {   pParam->BSIM4v4vfb = model->BSIM4v4type * pParam->BSIM4v4vth0
                                           - pParam->BSIM4v4phi - pParam->BSIM4v4k1
                                           * pParam->BSIM4v4sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4v4vfb = -1.0;
                      }
                  }
                   if (!model->BSIM4v4vth0Given)
                  {   pParam->BSIM4v4vth0 = model->BSIM4v4type * (pParam->BSIM4v4vfb
                                        + pParam->BSIM4v4phi + pParam->BSIM4v4k1
                                        * pParam->BSIM4v4sqrtPhi);
                  }

                  pParam->BSIM4v4k1ox = pParam->BSIM4v4k1 * model->BSIM4v4toxe
                                    / model->BSIM4v4toxm;
                  pParam->BSIM4v4k2ox = pParam->BSIM4v4k2 * model->BSIM4v4toxe
                                    / model->BSIM4v4toxm;

                  T3 = model->BSIM4v4type * pParam->BSIM4v4vth0
                     - pParam->BSIM4v4vfb - pParam->BSIM4v4phi;
                  T4 = T3 + T3;
                  T5 = 2.5 * T3;
                  pParam->BSIM4v4vtfbphi1 = (model->BSIM4v4type == NMOS) ? T4 : T5; 
                  if (pParam->BSIM4v4vtfbphi1 < 0.0)
                      pParam->BSIM4v4vtfbphi1 = 0.0;

                  pParam->BSIM4v4vtfbphi2 = 4.0 * T3;
                  if (pParam->BSIM4v4vtfbphi2 < 0.0)
                      pParam->BSIM4v4vtfbphi2 = 0.0;

                  tmp = sqrt(EPSSI / (model->BSIM4v4epsrox * EPS0)
                      * model->BSIM4v4toxe * pParam->BSIM4v4Xdep0);
                    T0 = pParam->BSIM4v4dsub * pParam->BSIM4v4leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                    {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                            T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v4theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v4theta0vb0 = 1.0 / (MAX_EXP - 2.0);

                   T0 = pParam->BSIM4v4drout * pParam->BSIM4v4leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                         {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v4thetaRout = pParam->BSIM4v4pdibl1 * T5
                                         + pParam->BSIM4v4pdibl2;

                  tmp = sqrt(pParam->BSIM4v4Xdep0);
                  tmp1 = pParam->BSIM4v4vbi - pParam->BSIM4v4phi;
                  tmp2 = model->BSIM4v4factor1 * tmp;

                  T0 = pParam->BSIM4v4dvt1w * pParam->BSIM4v4weff
                     * pParam->BSIM4v4leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v4dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v4dvt1 * pParam->BSIM4v4leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  } 
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v4dvt0 * T9 * tmp1;

                  T4 = model->BSIM4v4toxe * pParam->BSIM4v4phi
                     / (pParam->BSIM4v4weff + pParam->BSIM4v4w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v4lpe0 / pParam->BSIM4v4leff);
                  T5 = pParam->BSIM4v4k1ox * (T0 - 1.0) * pParam->BSIM4v4sqrtPhi
                     + (pParam->BSIM4v4kt1 + pParam->BSIM4v4kt1l / pParam->BSIM4v4leff)
                     * (TRatio - 1.0);

                  tmp3 = model->BSIM4v4type * pParam->BSIM4v4vth0
                       - T8 - T9 + pParam->BSIM4v4k3 * T4 + T5;
                  pParam->BSIM4v4vfbzb = tmp3 - pParam->BSIM4v4phi - pParam->BSIM4v4k1
                                     * pParam->BSIM4v4sqrtPhi; /* End of vfbzb */

                  /* stress effect */
                  switch (model->BSIM4v4intVersion) {
                    case BSIM4vOLD: case BSIM4v21:
                      break;
                    case BSIM4v30: case BSIM4v40:
                      T0 = pow(Lnew, model->BSIM4v4llodku0);
                      W_tmp = Wnew + model->BSIM4v4wlod;
                      T1 = pow(W_tmp, model->BSIM4v4wlodku0);
                      tmp1 = model->BSIM4v4lku0 / T0 + model->BSIM4v4wku0 / T1
                             + model->BSIM4v4pku0 / (T0 * T1);
                      pParam->BSIM4v4ku0 = 1.0 + tmp1;
                      
                      T0 = pow(Lnew, model->BSIM4v4llodvth);
                      T1 = pow(W_tmp, model->BSIM4v4wlodvth);
                      tmp1 = model->BSIM4v4lkvth0 / T0 + model->BSIM4v4wkvth0 / T1
                           + model->BSIM4v4pkvth0 / (T0 * T1);
                      pParam->BSIM4v4kvth0 = 1.0 + tmp1;
                      pParam->BSIM4v4kvth0 = sqrt(pParam->BSIM4v4kvth0*pParam->BSIM4v4kvth0 + DELTA);
                      
                      T0 = (TRatio - 1.0);
                      pParam->BSIM4v4ku0temp = pParam->BSIM4v4ku0 * (1.0 + model->BSIM4v4tku0 *T0) + DELTA;
                      
                      Inv_saref = 1.0/(model->BSIM4v4saref + 0.5*Ldrn);
                      Inv_sbref = 1.0/(model->BSIM4v4sbref + 0.5*Ldrn);
                      pParam->BSIM4v4inv_od_ref = Inv_saref + Inv_sbref;
                      pParam->BSIM4v4rho_ref = model->BSIM4v4ku0 / pParam->BSIM4v4ku0temp * pParam->BSIM4v4inv_od_ref;
                      break;
                    default: break;
                  }

              } /* End of SizeNotFound */

              /*  stress effect */
              switch (model->BSIM4v4intVersion) {
                case BSIM4vOLD: case BSIM4v21:
                  break;
                case BSIM4v30: case BSIM4v40:
                  if( (here->BSIM4v4sa > 0.0) && (here->BSIM4v4sb > 0.0) && 
                     ((here->BSIM4v4nf == 1.0) || ((here->BSIM4v4nf > 1.0) && (here->BSIM4v4sd > 0.0))) )
                  {   Inv_sa = 0;
                      Inv_sb = 0;

                      if (model->BSIM4v4wlod < 0.0)
                      {   fprintf(stderr, "Warning: WLOD = %g is less than 0. Set to 0.0\n",model->BSIM4v4wlod);
                          model->BSIM4v4wlod = 0.0;
                      }
                      if (model->BSIM4v4kvsat < -1.0 )
                      {   fprintf(stderr, "Warning: KVSAT = %g is too small; Reset to -1.0.\n",model->BSIM4v4kvsat);
                          model->BSIM4v4kvsat = -1.0;
                      }
                      if (model->BSIM4v4kvsat > 1.0)
                      {   fprintf(stderr, "Warning: KVSAT = %g is too big; Reset to 1.0.\n",model->BSIM4v4kvsat);
                          model->BSIM4v4kvsat = 1.0;
                      }

                      for(i = 0; i < here->BSIM4v4nf; i++){
                        T0 = 1.0 / here->BSIM4v4nf / (here->BSIM4v4sa + 0.5*Ldrn + i * (here->BSIM4v4sd +Ldrn));
                        T1 = 1.0 / here->BSIM4v4nf / (here->BSIM4v4sb + 0.5*Ldrn + i * (here->BSIM4v4sd +Ldrn));
                        Inv_sa += T0;
                        Inv_sb += T1;
                      }
                      Inv_ODeff = Inv_sa + Inv_sb; 
                      rho = model->BSIM4v4ku0 / pParam->BSIM4v4ku0temp * Inv_ODeff;
                      T0 = (1.0 + rho)/(1.0 + pParam->BSIM4v4rho_ref);
                      here->BSIM4v4u0temp = pParam->BSIM4v4u0temp * T0;
                  
                      T1 = (1.0 + model->BSIM4v4kvsat * rho)/(1.0 + model->BSIM4v4kvsat * pParam->BSIM4v4rho_ref);
                      here->BSIM4v4vsattemp = pParam->BSIM4v4vsattemp * T1;
                  
                      OD_offset = Inv_ODeff - pParam->BSIM4v4inv_od_ref;
                      dvth0_lod = model->BSIM4v4kvth0 / pParam->BSIM4v4kvth0 * OD_offset;
                      dk2_lod = model->BSIM4v4stk2 / pow(pParam->BSIM4v4kvth0, model->BSIM4v4lodk2) *
                                       OD_offset;
                      deta0_lod = model->BSIM4v4steta0 / pow(pParam->BSIM4v4kvth0, model->BSIM4v4lodeta0) *
                                         OD_offset;
                      here->BSIM4v4vth0 = pParam->BSIM4v4vth0 + dvth0_lod;
                  
                      if (!model->BSIM4v4vfbGiven && !model->BSIM4v4vth0Given)
                           here->BSIM4v4vfb = -1.0;
                      else  
                           here->BSIM4v4vfb = pParam->BSIM4v4vfb + model->BSIM4v4type * dvth0_lod;
                      here->BSIM4v4vfbzb = pParam->BSIM4v4vfbzb + model->BSIM4v4type * dvth0_lod;
                  
                      T3 = model->BSIM4v4type * here->BSIM4v4vth0
                         - here->BSIM4v4vfb - pParam->BSIM4v4phi;
                      T4 = T3 + T3;
                      T5 = 2.5 * T3;
                      here->BSIM4v4vtfbphi1 = (model->BSIM4v4type == NMOS) ? T4 : T5;
                      if (here->BSIM4v4vtfbphi1 < 0.0)
                          here->BSIM4v4vtfbphi1 = 0.0;
                  
                      here->BSIM4v4vtfbphi2 = 4.0 * T3;
                      if (here->BSIM4v4vtfbphi2 < 0.0)
                          here->BSIM4v4vtfbphi2 = 0.0;
                      
                      here->BSIM4v4k2 = pParam->BSIM4v4k2 + dk2_lod;
                      if (here->BSIM4v4k2 < 0.0)
                      {   T0 = 0.5 * pParam->BSIM4v4k1 / here->BSIM4v4k2;
                          here->BSIM4v4vbsc = 0.9 * (pParam->BSIM4v4phi - T0 * T0);
                          if (here->BSIM4v4vbsc > -3.0)
                              here->BSIM4v4vbsc = -3.0;
                          else if (here->BSIM4v4vbsc < -30.0)
                              here->BSIM4v4vbsc = -30.0;
                      }
                      else
                          here->BSIM4v4vbsc = -30.0;
                      if (here->BSIM4v4vbsc > pParam->BSIM4v4vbm)
                          here->BSIM4v4vbsc = pParam->BSIM4v4vbm;
                      here->BSIM4v4k2ox = here->BSIM4v4k2 * model->BSIM4v4toxe
                                        / model->BSIM4v4toxm;
                  
                      here->BSIM4v4eta0 = pParam->BSIM4v4eta0 + deta0_lod;
                  } else {
                          here->BSIM4v4u0temp = pParam->BSIM4v4u0temp;
                          here->BSIM4v4vth0 = pParam->BSIM4v4vth0;
                          here->BSIM4v4vsattemp = pParam->BSIM4v4vsattemp;
                          here->BSIM4v4vfb = pParam->BSIM4v4vfb;
                          here->BSIM4v4vfbzb = pParam->BSIM4v4vfbzb;
                          here->BSIM4v4vtfbphi1 = pParam->BSIM4v4vtfbphi1;
                          here->BSIM4v4vtfbphi2 = pParam->BSIM4v4vtfbphi2;
                          here->BSIM4v4k2 = pParam->BSIM4v4k2;
                          here->BSIM4v4vbsc = pParam->BSIM4v4vbsc;
                          here->BSIM4v4k2ox = pParam->BSIM4v4k2ox;
                          here->BSIM4v4eta0 = pParam->BSIM4v4eta0;
                  }
                  break;
                default: break;
              }
                   
              here->BSIM4v4cgso = pParam->BSIM4v4cgso;
              here->BSIM4v4cgdo = pParam->BSIM4v4cgdo;
              
              if (here->BSIM4v4rbodyMod)
              {   if (here->BSIM4v4rbdb < 1.0e-3)
                      here->BSIM4v4grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v4grbdb = model->BSIM4v4gbmin + 1.0 / here->BSIM4v4rbdb;
                  if (here->BSIM4v4rbpb < 1.0e-3)
                      here->BSIM4v4grbpb = 1.0e3;
                  else
                      here->BSIM4v4grbpb = model->BSIM4v4gbmin + 1.0 / here->BSIM4v4rbpb;
                  if (here->BSIM4v4rbps < 1.0e-3)
                      here->BSIM4v4grbps = 1.0e3;
                  else
                      here->BSIM4v4grbps = model->BSIM4v4gbmin + 1.0 / here->BSIM4v4rbps;
                  if (here->BSIM4v4rbsb < 1.0e-3)
                      here->BSIM4v4grbsb = 1.0e3;
                  else
                      here->BSIM4v4grbsb = model->BSIM4v4gbmin + 1.0 / here->BSIM4v4rbsb;
                  if (here->BSIM4v4rbpd < 1.0e-3)
                      here->BSIM4v4grbpd = 1.0e3;
                  else
                      here->BSIM4v4grbpd = model->BSIM4v4gbmin + 1.0 / here->BSIM4v4rbpd;
              }


              /* 
               * Process geomertry dependent parasitics
               */

              here->BSIM4v4grgeltd = model->BSIM4v4rshg * (model->BSIM4v4xgw
                      + pParam->BSIM4v4weffCJ / 3.0 / model->BSIM4v4ngcon) /
                      (model->BSIM4v4ngcon * here->BSIM4v4nf *
                      (Lnew - model->BSIM4v4xgl));
              if (here->BSIM4v4grgeltd > 0.0)
                  here->BSIM4v4grgeltd = 1.0 / here->BSIM4v4grgeltd;
              else
              {   here->BSIM4v4grgeltd = 1.0e3; /* mho */
                  if (here->BSIM4v4rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

              DMCGeff = model->BSIM4v4dmcg - model->BSIM4v4dmcgt;
              DMCIeff = model->BSIM4v4dmci;
              DMDGeff = model->BSIM4v4dmdg - model->BSIM4v4dmcgt;

              if (here->BSIM4v4sourcePerimeterGiven)
              {   if (model->BSIM4v4perMod == 0)
                      here->BSIM4v4Pseff = here->BSIM4v4sourcePerimeter;
                  else
                      here->BSIM4v4Pseff = here->BSIM4v4sourcePerimeter 
                                       - pParam->BSIM4v4weffCJ * here->BSIM4v4nf;
              }
              else
                  BSIM4v4PAeffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod, here->BSIM4v4min, 
                                    pParam->BSIM4v4weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &(here->BSIM4v4Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4v4drainPerimeterGiven)
              {   if (model->BSIM4v4perMod == 0)
                      here->BSIM4v4Pdeff = here->BSIM4v4drainPerimeter;
                  else
                      here->BSIM4v4Pdeff = here->BSIM4v4drainPerimeter 
                                       - pParam->BSIM4v4weffCJ * here->BSIM4v4nf;
              }
              else
                  BSIM4v4PAeffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod, here->BSIM4v4min,
                                    pParam->BSIM4v4weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &(here->BSIM4v4Pdeff), &dumAs, &dumAd);

              if (here->BSIM4v4sourceAreaGiven)
                  here->BSIM4v4Aseff = here->BSIM4v4sourceArea;
              else
                  BSIM4v4PAeffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod, here->BSIM4v4min,
                                    pParam->BSIM4v4weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &(here->BSIM4v4Aseff), &dumAd);

              if (here->BSIM4v4drainAreaGiven)
                  here->BSIM4v4Adeff = here->BSIM4v4drainArea;
              else
                  BSIM4v4PAeffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod, here->BSIM4v4min,
                                    pParam->BSIM4v4weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &dumAs, &(here->BSIM4v4Adeff));

              /* Processing S/D resistance and conductance below */
              if(here->BSIM4v4sNodePrime != here->BSIM4v4sNode)
              {
                 here->BSIM4v4sourceConductance = 0.0;
                 if(here->BSIM4v4sourceSquaresGiven)
                 {
                    here->BSIM4v4sourceConductance = model->BSIM4v4sheetResistance
                                               * here->BSIM4v4sourceSquares;
                 } else if (here->BSIM4v4rgeoMod > 0)
                 {
                    BSIM4v4RdseffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod,
                      here->BSIM4v4rgeoMod, here->BSIM4v4min,
                      pParam->BSIM4v4weffCJ, model->BSIM4v4sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v4sourceConductance));
                 } else
                 {
                    here->BSIM4v4sourceConductance = 0.0;
                 }

                 if (here->BSIM4v4sourceConductance > 0.0)
                     here->BSIM4v4sourceConductance = 1.0
                                            / here->BSIM4v4sourceConductance;
                 else
                 {
                     here->BSIM4v4sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4v4sourceConductance = 0.0;
              }

              if(here->BSIM4v4dNodePrime != here->BSIM4v4dNode)
              {
                 here->BSIM4v4drainConductance = 0.0;
                 if(here->BSIM4v4drainSquaresGiven)
                 {
                    here->BSIM4v4drainConductance = model->BSIM4v4sheetResistance
                                              * here->BSIM4v4drainSquares;
                 } else if (here->BSIM4v4rgeoMod > 0)
                 {
                    BSIM4v4RdseffGeo(here->BSIM4v4nf, here->BSIM4v4geoMod,
                      here->BSIM4v4rgeoMod, here->BSIM4v4min,
                      pParam->BSIM4v4weffCJ, model->BSIM4v4sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v4drainConductance));
                 } else
                 {
                    here->BSIM4v4drainConductance = 0.0;
                 }

                 if (here->BSIM4v4drainConductance > 0.0)
                     here->BSIM4v4drainConductance = 1.0
                                           / here->BSIM4v4drainConductance;
                 else
                 {
                     here->BSIM4v4drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4v4drainConductance = 0.0;
              }
           
               /* End of Rsd processing */


              Nvtms = model->BSIM4v4vtm * model->BSIM4v4SjctEmissionCoeff;
              if ((here->BSIM4v4Aseff <= 0.0) && (here->BSIM4v4Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4v4Aseff * model->BSIM4v4SjctTempSatCurDensity
                                   + here->BSIM4v4Pseff * model->BSIM4v4SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v4weffCJ * here->BSIM4v4nf
                                   * model->BSIM4v4SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v4dioMod)
                  {   case 0:
                          if ((model->BSIM4v4bvs / Nvtms) > EXP_THRESHOLD)
                              here->BSIM4v4XExpBVS = model->BSIM4v4xjbvs * MIN_EXP;
                          else
                              here->BSIM4v4XExpBVS = model->BSIM4v4xjbvs * exp(-model->BSIM4v4bvs / Nvtms);        
                          break;
                      case 1:
                          BSIM4v4DioIjthVjmEval(Nvtms, model->BSIM4v4ijthsfwd, SourceSatCurrent, 
                                              0.0, &(here->BSIM4v4vjsmFwd));
                          here->BSIM4v4IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v4vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v4bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v4XExpBVS = model->BSIM4v4xjbvs * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v4XExpBVS = exp(-model->BSIM4v4bvs / Nvtms);
                              tmp = here->BSIM4v4XExpBVS;
                              here->BSIM4v4XExpBVS *= model->BSIM4v4xjbvs;        
                          }

                          BSIM4v4DioIjthVjmEval(Nvtms, model->BSIM4v4ijthsfwd, SourceSatCurrent, 
                                                     here->BSIM4v4XExpBVS, &(here->BSIM4v4vjsmFwd));
                          T0 = exp(here->BSIM4v4vjsmFwd / Nvtms);
                          here->BSIM4v4IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v4XExpBVS / T0
                                                + here->BSIM4v4XExpBVS - 1.0);
                          here->BSIM4v4SslpFwd = SourceSatCurrent
                                               * (T0 + here->BSIM4v4XExpBVS / T0) / Nvtms;

                          T2 = model->BSIM4v4ijthsrev / SourceSatCurrent;
                          if (T2 < 1.0)
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
                          } 
                          here->BSIM4v4vjsmRev = -model->BSIM4v4bvs
                                             - Nvtms * log((T2 - 1.0) / model->BSIM4v4xjbvs);
                          T1 = model->BSIM4v4xjbvs * exp(-(model->BSIM4v4bvs
                             + here->BSIM4v4vjsmRev) / Nvtms);
                          here->BSIM4v4IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v4SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v4dioMod);
                  }
              }

              Nvtmd = model->BSIM4v4vtm * model->BSIM4v4DjctEmissionCoeff;
              if ((here->BSIM4v4Adeff <= 0.0) && (here->BSIM4v4Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4v4Adeff * model->BSIM4v4DjctTempSatCurDensity
                                  + here->BSIM4v4Pdeff * model->BSIM4v4DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v4weffCJ * here->BSIM4v4nf
                                  * model->BSIM4v4DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v4dioMod)
                  {   case 0:
                          if ((model->BSIM4v4bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v4XExpBVD = model->BSIM4v4xjbvd * MIN_EXP;
                          else
                          here->BSIM4v4XExpBVD = model->BSIM4v4xjbvd * exp(-model->BSIM4v4bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v4DioIjthVjmEval(Nvtmd, model->BSIM4v4ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v4vjdmFwd));
                          here->BSIM4v4IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v4vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v4bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v4XExpBVD = model->BSIM4v4xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v4XExpBVD = exp(-model->BSIM4v4bvd / Nvtmd);
                              tmp = here->BSIM4v4XExpBVD;
                              here->BSIM4v4XExpBVD *= model->BSIM4v4xjbvd;
                          }

                          BSIM4v4DioIjthVjmEval(Nvtmd, model->BSIM4v4ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v4XExpBVD, &(here->BSIM4v4vjdmFwd));
                          T0 = exp(here->BSIM4v4vjdmFwd / Nvtmd);
                          here->BSIM4v4IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v4XExpBVD / T0
                                              + here->BSIM4v4XExpBVD - 1.0);
                          here->BSIM4v4DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v4XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v4ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0) 
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          switch (model->BSIM4v4intVersion) {
                            case BSIM4vOLD: case BSIM4v21:
                              here->BSIM4v4vjdmRev = -model->BSIM4v4bvd
                                               - Nvtms * log((T2 - 1.0) / model->BSIM4v4xjbvd);
                              break;
                            case BSIM4v30: case BSIM4v40:
                              here->BSIM4v4vjdmRev = -model->BSIM4v4bvd
                                               - Nvtmd * log((T2 - 1.0) / model->BSIM4v4xjbvd); /* bugfix */
                              break;
                            default: break;
                          }
                          T1 = model->BSIM4v4xjbvd * exp(-(model->BSIM4v4bvd
                             + here->BSIM4v4vjdmRev) / Nvtmd);
                          here->BSIM4v4IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v4DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v4dioMod);
                  }
              }

              switch (model->BSIM4v4intVersion) {
                case BSIM4vOLD: case BSIM4v21: case BSIM4v30:
                  break;
                case BSIM4v40:
                  /* GEDL current reverse bias */
                  T0 = (TRatio - 1.0);
                  model->BSIM4v4njtstemp = model->BSIM4v4njts * (1.0 + model->BSIM4v4tnjts * T0);
                  model->BSIM4v4njtsswtemp = model->BSIM4v4njtssw * (1.0 + model->BSIM4v4tnjtssw * T0);
                  model->BSIM4v4njtsswgtemp = model->BSIM4v4njtsswg * (1.0 + model->BSIM4v4tnjtsswg * T0);
                  T7 = Eg0 / model->BSIM4v4vtm * T0;
                  T9 = model->BSIM4v4xtss * T7;
                  DEXP(T9, T1);
                  T9 = model->BSIM4v4xtsd * T7;
                  DEXP(T9, T2);
                  T9 = model->BSIM4v4xtssws * T7;
                  DEXP(T9, T3);
                  T9 = model->BSIM4v4xtsswd * T7;
                  DEXP(T9, T4);
                  T9 = model->BSIM4v4xtsswgs * T7;
                  DEXP(T9, T5);
                  T9 = model->BSIM4v4xtsswgd * T7;
                  DEXP(T9, T6);

                  T10 = pParam->BSIM4v4weffCJ * here->BSIM4v4nf;
                  here->BSIM4v4SjctTempRevSatCur = T1 * here->BSIM4v4Aseff * model->BSIM4v4jtss;
                  here->BSIM4v4DjctTempRevSatCur = T2 * here->BSIM4v4Adeff * model->BSIM4v4jtsd;
                  here->BSIM4v4SswTempRevSatCur = T3 * here->BSIM4v4Pseff * model->BSIM4v4jtssws;
                  here->BSIM4v4DswTempRevSatCur = T4 * here->BSIM4v4Pdeff * model->BSIM4v4jtsswd;
                  here->BSIM4v4SswgTempRevSatCur = T5 * T10 * model->BSIM4v4jtsswgs;
                  here->BSIM4v4DswgTempRevSatCur = T6 * T10 * model->BSIM4v4jtsswgd;
                  break;
                default: break;
              }

              if (BSIM4v4checkModel(model, here, ckt))
              {   IFuid namarray[2];
                  namarray[0] = model->BSIM4v4modName;
                  namarray[1] = here->BSIM4v4name;
                  switch (model->BSIM4v4intVersion) {
                    case BSIM4vOLD: case BSIM4v21:
                      SPfrontEnd->IFerror (ERR_FATAL, "Fatal error(s) detected during BSIM4v4.2.1 parameter checking for %s in model %s", namarray);
                      break;
                    case BSIM4v30:
                      SPfrontEnd->IFerror (ERR_FATAL, "Fatal error(s) detected during BSIM4v4.3.0 parameter checking for %s in model %s", namarray);
                      break;
                    case BSIM4v40:
                      SPfrontEnd->IFerror (ERR_FATAL, "Fatal error(s) detected during BSIM4v4.4.0 parameter checking for %s in model %s", namarray);
                      break;
                    default: break;
                  }
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
