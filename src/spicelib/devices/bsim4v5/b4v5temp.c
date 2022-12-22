/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4temp.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
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
BSIM4v5DioIjthVjmEval(
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
BSIM4v5temp(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v5model *model = (BSIM4v5model*) inModel;
BSIM4v5instance *here;
struct bsim4v5SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp, tmp1, tmp2, Eg, Eg0, ni;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Lnew=0.0, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
double T10;
double Inv_saref, Inv_sbref, Inv_sa, Inv_sb, rho, Ldrn, dvth0_lod;
double W_tmp, Inv_ODeff, OD_offset, dk2_lod, deta0_lod;
double lnl, lnw, lnnf, rbpbx, rbpby, rbsbx, rbsby, rbdbx, rbdby,bodymode;
double kvsat, wlod, sceff, Wdrn;

int Size_Not_Found, i;

    /*  loop through all the BSIM4v5 device models */
    for (; model != NULL; model = BSIM4v5nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v5SbulkJctPotential < 0.1)
         {   model->BSIM4v5SbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
         }
         if (model->BSIM4v5SsidewallJctPotential < 0.1)
         {   model->BSIM4v5SsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
         }
         if (model->BSIM4v5SGatesidewallJctPotential < 0.1)
         {   model->BSIM4v5SGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
         }

         if (model->BSIM4v5DbulkJctPotential < 0.1)
         {   model->BSIM4v5DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v5DsidewallJctPotential < 0.1)
         {   model->BSIM4v5DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v5DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v5DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if ((model->BSIM4v5toxeGiven) && (model->BSIM4v5toxpGiven) && (model->BSIM4v5dtoxGiven)
             && (model->BSIM4v5toxe != (model->BSIM4v5toxp + model->BSIM4v5dtox)))
             printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
         else if ((model->BSIM4v5toxeGiven) && (!model->BSIM4v5toxpGiven))
             model->BSIM4v5toxp = model->BSIM4v5toxe - model->BSIM4v5dtox;
         else if ((!model->BSIM4v5toxeGiven) && (model->BSIM4v5toxpGiven))
             model->BSIM4v5toxe = model->BSIM4v5toxp + model->BSIM4v5dtox;

         model->BSIM4v5coxe = model->BSIM4v5epsrox * EPS0 / model->BSIM4v5toxe;
         model->BSIM4v5coxp = model->BSIM4v5epsrox * EPS0 / model->BSIM4v5toxp;

         if (!model->BSIM4v5cgdoGiven)
         {   if (model->BSIM4v5dlcGiven && (model->BSIM4v5dlc > 0.0))
                 model->BSIM4v5cgdo = model->BSIM4v5dlc * model->BSIM4v5coxe
                                  - model->BSIM4v5cgdl ;
             else
                 model->BSIM4v5cgdo = 0.6 * model->BSIM4v5xj * model->BSIM4v5coxe;
         }
         if (!model->BSIM4v5cgsoGiven)
         {   if (model->BSIM4v5dlcGiven && (model->BSIM4v5dlc > 0.0))
                 model->BSIM4v5cgso = model->BSIM4v5dlc * model->BSIM4v5coxe
                                  - model->BSIM4v5cgsl ;
             else
                 model->BSIM4v5cgso = 0.6 * model->BSIM4v5xj * model->BSIM4v5coxe;
         }
         if (!model->BSIM4v5cgboGiven)
             model->BSIM4v5cgbo = 2.0 * model->BSIM4v5dwc * model->BSIM4v5coxe;

         struct bsim4v5SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim4v5SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM4v5tnom;
         TRatio = Temp / Tnom;

         model->BSIM4v5vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v5factor1 = sqrt(EPSSI / (model->BSIM4v5epsrox * EPS0)
                             * model->BSIM4v5toxe);

         Vtm0 = model->BSIM4v5vtm0 = KboQ * Tnom;
         Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
         ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
            * exp(21.5565981 - Eg0 / (2.0 * Vtm0));

         model->BSIM4v5vtm = KboQ * Temp;
         Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v5vtm;
             T1 = log(Temp / Tnom);
             T2 = T0 + model->BSIM4v5SjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v5SjctEmissionCoeff);
             model->BSIM4v5SjctTempSatCurDensity = model->BSIM4v5SjctSatCurDensity
                                               * T3;
             model->BSIM4v5SjctSidewallTempSatCurDensity
                         = model->BSIM4v5SjctSidewallSatCurDensity * T3;
             model->BSIM4v5SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v5SjctGateSidewallSatCurDensity * T3;

             T2 = T0 + model->BSIM4v5DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v5DjctEmissionCoeff);
             model->BSIM4v5DjctTempSatCurDensity = model->BSIM4v5DjctSatCurDensity
                                               * T3;
             model->BSIM4v5DjctSidewallTempSatCurDensity
                         = model->BSIM4v5DjctSidewallSatCurDensity * T3;
             model->BSIM4v5DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v5DjctGateSidewallSatCurDensity * T3;
         }
         else
         {   model->BSIM4v5SjctTempSatCurDensity = model->BSIM4v5SjctSatCurDensity;
             model->BSIM4v5SjctSidewallTempSatCurDensity
                        = model->BSIM4v5SjctSidewallSatCurDensity;
             model->BSIM4v5SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v5SjctGateSidewallSatCurDensity;
             model->BSIM4v5DjctTempSatCurDensity = model->BSIM4v5DjctSatCurDensity;
             model->BSIM4v5DjctSidewallTempSatCurDensity
                        = model->BSIM4v5DjctSidewallSatCurDensity;
             model->BSIM4v5DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v5DjctGateSidewallSatCurDensity;
         }

         if (model->BSIM4v5SjctTempSatCurDensity < 0.0)
             model->BSIM4v5SjctTempSatCurDensity = 0.0;
         if (model->BSIM4v5SjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v5SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v5SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v5SjctGateSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v5DjctTempSatCurDensity < 0.0)
             model->BSIM4v5DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v5DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v5DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v5DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v5DjctGateSidewallTempSatCurDensity = 0.0;

         /* Temperature dependence of D/B and S/B diode capacitance begins */
         delTemp = ckt->CKTtemp - model->BSIM4v5tnom;
         T0 = model->BSIM4v5tcj * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v5SunitAreaTempJctCap = model->BSIM4v5SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4v5DunitAreaTempJctCap = model->BSIM4v5DunitAreaJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v5SunitAreaJctCap > 0.0)
             {   model->BSIM4v5SunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
             if (model->BSIM4v5DunitAreaJctCap > 0.0)
             {   model->BSIM4v5DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
         }
         T0 = model->BSIM4v5tcjsw * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v5SunitLengthSidewallTempJctCap = model->BSIM4v5SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4v5DunitLengthSidewallTempJctCap = model->BSIM4v5DunitLengthSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v5SunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v5SunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
             }
             if (model->BSIM4v5DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v5DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }
         }
         T0 = model->BSIM4v5tcjswg * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v5SunitLengthGateSidewallTempJctCap = model->BSIM4v5SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4v5DunitLengthGateSidewallTempJctCap = model->BSIM4v5DunitLengthGateSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v5SunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v5SunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
             }
             if (model->BSIM4v5DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v5DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
         }

         model->BSIM4v5PhiBS = model->BSIM4v5SbulkJctPotential
                           - model->BSIM4v5tpb * delTemp;
         if (model->BSIM4v5PhiBS < 0.01)
         {   model->BSIM4v5PhiBS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
         }
         model->BSIM4v5PhiBD = model->BSIM4v5DbulkJctPotential
                           - model->BSIM4v5tpb * delTemp;
         if (model->BSIM4v5PhiBD < 0.01)
         {   model->BSIM4v5PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v5PhiBSWS = model->BSIM4v5SsidewallJctPotential
                             - model->BSIM4v5tpbsw * delTemp;
         if (model->BSIM4v5PhiBSWS <= 0.01)
         {   model->BSIM4v5PhiBSWS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
         }
         model->BSIM4v5PhiBSWD = model->BSIM4v5DsidewallJctPotential
                             - model->BSIM4v5tpbsw * delTemp;
         if (model->BSIM4v5PhiBSWD <= 0.01)
         {   model->BSIM4v5PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

         model->BSIM4v5PhiBSWGS = model->BSIM4v5SGatesidewallJctPotential
                              - model->BSIM4v5tpbswg * delTemp;
         if (model->BSIM4v5PhiBSWGS <= 0.01)
         {   model->BSIM4v5PhiBSWGS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
         }
         model->BSIM4v5PhiBSWGD = model->BSIM4v5DGatesidewallJctPotential
                              - model->BSIM4v5tpbswg * delTemp;
         if (model->BSIM4v5PhiBSWGD <= 0.01)
         {   model->BSIM4v5PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v5ijthdfwd <= 0.0)
         {   model->BSIM4v5ijthdfwd = 0.1;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v5ijthdfwd);
         }
         if (model->BSIM4v5ijthsfwd <= 0.0)
         {   model->BSIM4v5ijthsfwd = 0.1;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v5ijthsfwd);
         }
         if (model->BSIM4v5ijthdrev <= 0.0)
         {   model->BSIM4v5ijthdrev = 0.1;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v5ijthdrev);
         }
         if (model->BSIM4v5ijthsrev <= 0.0)
         {   model->BSIM4v5ijthsrev = 0.1;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v5ijthsrev);
         }

         if ((model->BSIM4v5xjbvd <= 0.0) && (model->BSIM4v5dioMod == 2))
         {   model->BSIM4v5xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v5xjbvd);
         }
         else if ((model->BSIM4v5xjbvd < 0.0) && (model->BSIM4v5dioMod == 0))
         {   model->BSIM4v5xjbvd = 1.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v5xjbvd);
         }

         if (model->BSIM4v5bvd <= 0.0)
         {   model->BSIM4v5bvd = 10.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v5bvd);
         }

         if ((model->BSIM4v5xjbvs <= 0.0) && (model->BSIM4v5dioMod == 2))
         {   model->BSIM4v5xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v5xjbvs);
         }
         else if ((model->BSIM4v5xjbvs < 0.0) && (model->BSIM4v5dioMod == 0))
         {   model->BSIM4v5xjbvs = 1.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v5xjbvs);
         }

         if (model->BSIM4v5bvs <= 0.0)
         {   model->BSIM4v5bvs = 10.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v5bvs);
         }


         /* loop through all the instances of the model */
         for (here = BSIM4v5instances(model); here != NULL;
              here = BSIM4v5nextInstance(here))
            {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM4v5l == pSizeDependParamKnot->Length)
                      && (here->BSIM4v5w == pSizeDependParamKnot->Width)
                      && (here->BSIM4v5nf == pSizeDependParamKnot->NFinger))
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
              Ldrn = here->BSIM4v5l;
              Wdrn = here->BSIM4v5w / here->BSIM4v5nf;

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim4v5SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v5l;
                  pParam->Width = here->BSIM4v5w;
                  pParam->NFinger = here->BSIM4v5nf;
                  Lnew = here->BSIM4v5l  + model->BSIM4v5xl ;
                  Wnew = here->BSIM4v5w / here->BSIM4v5nf + model->BSIM4v5xw;

                  T0 = pow(Lnew, model->BSIM4v5Lln);
                  T1 = pow(Wnew, model->BSIM4v5Lwn);
                  tmp1 = model->BSIM4v5Ll / T0 + model->BSIM4v5Lw / T1
                       + model->BSIM4v5Lwl / (T0 * T1);
                  pParam->BSIM4v5dl = model->BSIM4v5Lint + tmp1;
                  tmp2 = model->BSIM4v5Llc / T0 + model->BSIM4v5Lwc / T1
                       + model->BSIM4v5Lwlc / (T0 * T1);
                  pParam->BSIM4v5dlc = model->BSIM4v5dlc + tmp2;

                  T2 = pow(Lnew, model->BSIM4v5Wln);
                  T3 = pow(Wnew, model->BSIM4v5Wwn);
                  tmp1 = model->BSIM4v5Wl / T2 + model->BSIM4v5Ww / T3
                       + model->BSIM4v5Wwl / (T2 * T3);
                  pParam->BSIM4v5dw = model->BSIM4v5Wint + tmp1;
                  tmp2 = model->BSIM4v5Wlc / T2 + model->BSIM4v5Wwc / T3
                       + model->BSIM4v5Wwlc / (T2 * T3);
                  pParam->BSIM4v5dwc = model->BSIM4v5dwc + tmp2;
                  pParam->BSIM4v5dwj = model->BSIM4v5dwj + tmp2;

                  pParam->BSIM4v5leff = Lnew - 2.0 * pParam->BSIM4v5dl;
                  if (pParam->BSIM4v5leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v5: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM4v5modName, here->BSIM4v5name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v5weff = Wnew - 2.0 * pParam->BSIM4v5dw;
                  if (pParam->BSIM4v5weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v5: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM4v5modName, here->BSIM4v5name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v5leffCV = Lnew - 2.0 * pParam->BSIM4v5dlc;
                  if (pParam->BSIM4v5leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v5: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM4v5modName, here->BSIM4v5name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v5weffCV = Wnew - 2.0 * pParam->BSIM4v5dwc;
                  if (pParam->BSIM4v5weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v5: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM4v5modName, here->BSIM4v5name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v5weffCJ = Wnew - 2.0 * pParam->BSIM4v5dwj;
                  if (pParam->BSIM4v5weffCJ <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v5: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       model->BSIM4v5modName, here->BSIM4v5name);
                      return(E_BADPARM);
                  }


                  if (model->BSIM4v5binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM4v5leff;
                      Inv_W = 1.0e-6 / pParam->BSIM4v5weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM4v5leff
                             * pParam->BSIM4v5weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM4v5leff;
                      Inv_W = 1.0 / pParam->BSIM4v5weff;
                      Inv_LW = 1.0 / (pParam->BSIM4v5leff
                             * pParam->BSIM4v5weff);
                  }
                  pParam->BSIM4v5cdsc = model->BSIM4v5cdsc
                                    + model->BSIM4v5lcdsc * Inv_L
                                    + model->BSIM4v5wcdsc * Inv_W
                                    + model->BSIM4v5pcdsc * Inv_LW;
                  pParam->BSIM4v5cdscb = model->BSIM4v5cdscb
                                     + model->BSIM4v5lcdscb * Inv_L
                                     + model->BSIM4v5wcdscb * Inv_W
                                     + model->BSIM4v5pcdscb * Inv_LW;

                      pParam->BSIM4v5cdscd = model->BSIM4v5cdscd
                                     + model->BSIM4v5lcdscd * Inv_L
                                     + model->BSIM4v5wcdscd * Inv_W
                                     + model->BSIM4v5pcdscd * Inv_LW;

                  pParam->BSIM4v5cit = model->BSIM4v5cit
                                   + model->BSIM4v5lcit * Inv_L
                                   + model->BSIM4v5wcit * Inv_W
                                   + model->BSIM4v5pcit * Inv_LW;
                  pParam->BSIM4v5nfactor = model->BSIM4v5nfactor
                                       + model->BSIM4v5lnfactor * Inv_L
                                       + model->BSIM4v5wnfactor * Inv_W
                                       + model->BSIM4v5pnfactor * Inv_LW;
                  pParam->BSIM4v5xj = model->BSIM4v5xj
                                  + model->BSIM4v5lxj * Inv_L
                                  + model->BSIM4v5wxj * Inv_W
                                  + model->BSIM4v5pxj * Inv_LW;
                  pParam->BSIM4v5vsat = model->BSIM4v5vsat
                                    + model->BSIM4v5lvsat * Inv_L
                                    + model->BSIM4v5wvsat * Inv_W
                                    + model->BSIM4v5pvsat * Inv_LW;
                  pParam->BSIM4v5at = model->BSIM4v5at
                                  + model->BSIM4v5lat * Inv_L
                                  + model->BSIM4v5wat * Inv_W
                                  + model->BSIM4v5pat * Inv_LW;
                  pParam->BSIM4v5a0 = model->BSIM4v5a0
                                  + model->BSIM4v5la0 * Inv_L
                                  + model->BSIM4v5wa0 * Inv_W
                                  + model->BSIM4v5pa0 * Inv_LW;

                  pParam->BSIM4v5ags = model->BSIM4v5ags
                                  + model->BSIM4v5lags * Inv_L
                                  + model->BSIM4v5wags * Inv_W
                                  + model->BSIM4v5pags * Inv_LW;

                  pParam->BSIM4v5a1 = model->BSIM4v5a1
                                  + model->BSIM4v5la1 * Inv_L
                                  + model->BSIM4v5wa1 * Inv_W
                                  + model->BSIM4v5pa1 * Inv_LW;
                  pParam->BSIM4v5a2 = model->BSIM4v5a2
                                  + model->BSIM4v5la2 * Inv_L
                                  + model->BSIM4v5wa2 * Inv_W
                                  + model->BSIM4v5pa2 * Inv_LW;
                  pParam->BSIM4v5keta = model->BSIM4v5keta
                                    + model->BSIM4v5lketa * Inv_L
                                    + model->BSIM4v5wketa * Inv_W
                                    + model->BSIM4v5pketa * Inv_LW;
                  pParam->BSIM4v5nsub = model->BSIM4v5nsub
                                    + model->BSIM4v5lnsub * Inv_L
                                    + model->BSIM4v5wnsub * Inv_W
                                    + model->BSIM4v5pnsub * Inv_LW;
                  pParam->BSIM4v5ndep = model->BSIM4v5ndep
                                    + model->BSIM4v5lndep * Inv_L
                                    + model->BSIM4v5wndep * Inv_W
                                    + model->BSIM4v5pndep * Inv_LW;
                  pParam->BSIM4v5nsd = model->BSIM4v5nsd
                                   + model->BSIM4v5lnsd * Inv_L
                                   + model->BSIM4v5wnsd * Inv_W
                                   + model->BSIM4v5pnsd * Inv_LW;
                  pParam->BSIM4v5phin = model->BSIM4v5phin
                                    + model->BSIM4v5lphin * Inv_L
                                    + model->BSIM4v5wphin * Inv_W
                                    + model->BSIM4v5pphin * Inv_LW;
                  pParam->BSIM4v5ngate = model->BSIM4v5ngate
                                     + model->BSIM4v5lngate * Inv_L
                                     + model->BSIM4v5wngate * Inv_W
                                     + model->BSIM4v5pngate * Inv_LW;
                  pParam->BSIM4v5gamma1 = model->BSIM4v5gamma1
                                      + model->BSIM4v5lgamma1 * Inv_L
                                      + model->BSIM4v5wgamma1 * Inv_W
                                      + model->BSIM4v5pgamma1 * Inv_LW;
                  pParam->BSIM4v5gamma2 = model->BSIM4v5gamma2
                                      + model->BSIM4v5lgamma2 * Inv_L
                                      + model->BSIM4v5wgamma2 * Inv_W
                                      + model->BSIM4v5pgamma2 * Inv_LW;
                  pParam->BSIM4v5vbx = model->BSIM4v5vbx
                                   + model->BSIM4v5lvbx * Inv_L
                                   + model->BSIM4v5wvbx * Inv_W
                                   + model->BSIM4v5pvbx * Inv_LW;
                  pParam->BSIM4v5vbm = model->BSIM4v5vbm
                                   + model->BSIM4v5lvbm * Inv_L
                                   + model->BSIM4v5wvbm * Inv_W
                                   + model->BSIM4v5pvbm * Inv_LW;
                  pParam->BSIM4v5xt = model->BSIM4v5xt
                                   + model->BSIM4v5lxt * Inv_L
                                   + model->BSIM4v5wxt * Inv_W
                                   + model->BSIM4v5pxt * Inv_LW;
                  pParam->BSIM4v5vfb = model->BSIM4v5vfb
                                   + model->BSIM4v5lvfb * Inv_L
                                   + model->BSIM4v5wvfb * Inv_W
                                   + model->BSIM4v5pvfb * Inv_LW;
                  pParam->BSIM4v5k1 = model->BSIM4v5k1
                                  + model->BSIM4v5lk1 * Inv_L
                                  + model->BSIM4v5wk1 * Inv_W
                                  + model->BSIM4v5pk1 * Inv_LW;
                  pParam->BSIM4v5kt1 = model->BSIM4v5kt1
                                   + model->BSIM4v5lkt1 * Inv_L
                                   + model->BSIM4v5wkt1 * Inv_W
                                   + model->BSIM4v5pkt1 * Inv_LW;
                  pParam->BSIM4v5kt1l = model->BSIM4v5kt1l
                                    + model->BSIM4v5lkt1l * Inv_L
                                    + model->BSIM4v5wkt1l * Inv_W
                                    + model->BSIM4v5pkt1l * Inv_LW;
                  pParam->BSIM4v5k2 = model->BSIM4v5k2
                                  + model->BSIM4v5lk2 * Inv_L
                                  + model->BSIM4v5wk2 * Inv_W
                                  + model->BSIM4v5pk2 * Inv_LW;
                  pParam->BSIM4v5kt2 = model->BSIM4v5kt2
                                   + model->BSIM4v5lkt2 * Inv_L
                                   + model->BSIM4v5wkt2 * Inv_W
                                   + model->BSIM4v5pkt2 * Inv_LW;
                  pParam->BSIM4v5k3 = model->BSIM4v5k3
                                  + model->BSIM4v5lk3 * Inv_L
                                  + model->BSIM4v5wk3 * Inv_W
                                  + model->BSIM4v5pk3 * Inv_LW;
                  pParam->BSIM4v5k3b = model->BSIM4v5k3b
                                   + model->BSIM4v5lk3b * Inv_L
                                   + model->BSIM4v5wk3b * Inv_W
                                   + model->BSIM4v5pk3b * Inv_LW;
                  pParam->BSIM4v5w0 = model->BSIM4v5w0
                                  + model->BSIM4v5lw0 * Inv_L
                                  + model->BSIM4v5ww0 * Inv_W
                                  + model->BSIM4v5pw0 * Inv_LW;
                  pParam->BSIM4v5lpe0 = model->BSIM4v5lpe0
                                    + model->BSIM4v5llpe0 * Inv_L
                                     + model->BSIM4v5wlpe0 * Inv_W
                                    + model->BSIM4v5plpe0 * Inv_LW;
                  pParam->BSIM4v5lpeb = model->BSIM4v5lpeb
                                    + model->BSIM4v5llpeb * Inv_L
                                    + model->BSIM4v5wlpeb * Inv_W
                                    + model->BSIM4v5plpeb * Inv_LW;
                  pParam->BSIM4v5dvtp0 = model->BSIM4v5dvtp0
                                     + model->BSIM4v5ldvtp0 * Inv_L
                                     + model->BSIM4v5wdvtp0 * Inv_W
                                     + model->BSIM4v5pdvtp0 * Inv_LW;
                  pParam->BSIM4v5dvtp1 = model->BSIM4v5dvtp1
                                     + model->BSIM4v5ldvtp1 * Inv_L
                                     + model->BSIM4v5wdvtp1 * Inv_W
                                     + model->BSIM4v5pdvtp1 * Inv_LW;
                  pParam->BSIM4v5dvt0 = model->BSIM4v5dvt0
                                    + model->BSIM4v5ldvt0 * Inv_L
                                    + model->BSIM4v5wdvt0 * Inv_W
                                    + model->BSIM4v5pdvt0 * Inv_LW;
                  pParam->BSIM4v5dvt1 = model->BSIM4v5dvt1
                                    + model->BSIM4v5ldvt1 * Inv_L
                                    + model->BSIM4v5wdvt1 * Inv_W
                                    + model->BSIM4v5pdvt1 * Inv_LW;
                  pParam->BSIM4v5dvt2 = model->BSIM4v5dvt2
                                    + model->BSIM4v5ldvt2 * Inv_L
                                    + model->BSIM4v5wdvt2 * Inv_W
                                    + model->BSIM4v5pdvt2 * Inv_LW;
                  pParam->BSIM4v5dvt0w = model->BSIM4v5dvt0w
                                    + model->BSIM4v5ldvt0w * Inv_L
                                    + model->BSIM4v5wdvt0w * Inv_W
                                    + model->BSIM4v5pdvt0w * Inv_LW;
                  pParam->BSIM4v5dvt1w = model->BSIM4v5dvt1w
                                    + model->BSIM4v5ldvt1w * Inv_L
                                    + model->BSIM4v5wdvt1w * Inv_W
                                    + model->BSIM4v5pdvt1w * Inv_LW;
                  pParam->BSIM4v5dvt2w = model->BSIM4v5dvt2w
                                    + model->BSIM4v5ldvt2w * Inv_L
                                    + model->BSIM4v5wdvt2w * Inv_W
                                    + model->BSIM4v5pdvt2w * Inv_LW;
                  pParam->BSIM4v5drout = model->BSIM4v5drout
                                     + model->BSIM4v5ldrout * Inv_L
                                     + model->BSIM4v5wdrout * Inv_W
                                     + model->BSIM4v5pdrout * Inv_LW;
                  pParam->BSIM4v5dsub = model->BSIM4v5dsub
                                    + model->BSIM4v5ldsub * Inv_L
                                    + model->BSIM4v5wdsub * Inv_W
                                    + model->BSIM4v5pdsub * Inv_LW;
                  pParam->BSIM4v5vth0 = model->BSIM4v5vth0
                                    + model->BSIM4v5lvth0 * Inv_L
                                    + model->BSIM4v5wvth0 * Inv_W
                                    + model->BSIM4v5pvth0 * Inv_LW;
                  pParam->BSIM4v5ua = model->BSIM4v5ua
                                  + model->BSIM4v5lua * Inv_L
                                  + model->BSIM4v5wua * Inv_W
                                  + model->BSIM4v5pua * Inv_LW;
                  pParam->BSIM4v5ua1 = model->BSIM4v5ua1
                                   + model->BSIM4v5lua1 * Inv_L
                                   + model->BSIM4v5wua1 * Inv_W
                                   + model->BSIM4v5pua1 * Inv_LW;
                  pParam->BSIM4v5ub = model->BSIM4v5ub
                                  + model->BSIM4v5lub * Inv_L
                                  + model->BSIM4v5wub * Inv_W
                                  + model->BSIM4v5pub * Inv_LW;
                  pParam->BSIM4v5ub1 = model->BSIM4v5ub1
                                   + model->BSIM4v5lub1 * Inv_L
                                   + model->BSIM4v5wub1 * Inv_W
                                   + model->BSIM4v5pub1 * Inv_LW;
                  pParam->BSIM4v5uc = model->BSIM4v5uc
                                  + model->BSIM4v5luc * Inv_L
                                  + model->BSIM4v5wuc * Inv_W
                                  + model->BSIM4v5puc * Inv_LW;
                  pParam->BSIM4v5uc1 = model->BSIM4v5uc1
                                   + model->BSIM4v5luc1 * Inv_L
                                   + model->BSIM4v5wuc1 * Inv_W
                                   + model->BSIM4v5puc1 * Inv_LW;
                  pParam->BSIM4v5ud = model->BSIM4v5ud
                                  + model->BSIM4v5lud * Inv_L
                                  + model->BSIM4v5wud * Inv_W
                                  + model->BSIM4v5pud * Inv_LW;
                  pParam->BSIM4v5ud1 = model->BSIM4v5ud1
                                  + model->BSIM4v5lud1 * Inv_L
                                  + model->BSIM4v5wud1 * Inv_W
                                  + model->BSIM4v5pud1 * Inv_LW;
                  pParam->BSIM4v5up = model->BSIM4v5up
                                  + model->BSIM4v5lup * Inv_L
                                  + model->BSIM4v5wup * Inv_W
                                  + model->BSIM4v5pup * Inv_LW;
                  pParam->BSIM4v5lp = model->BSIM4v5lp
                                  + model->BSIM4v5llp * Inv_L
                                  + model->BSIM4v5wlp * Inv_W
                                  + model->BSIM4v5plp * Inv_LW;
                  pParam->BSIM4v5eu = model->BSIM4v5eu
                                  + model->BSIM4v5leu * Inv_L
                                  + model->BSIM4v5weu * Inv_W
                                  + model->BSIM4v5peu * Inv_LW;
                  pParam->BSIM4v5u0 = model->BSIM4v5u0
                                  + model->BSIM4v5lu0 * Inv_L
                                  + model->BSIM4v5wu0 * Inv_W
                                  + model->BSIM4v5pu0 * Inv_LW;
                  pParam->BSIM4v5ute = model->BSIM4v5ute
                                   + model->BSIM4v5lute * Inv_L
                                   + model->BSIM4v5wute * Inv_W
                                   + model->BSIM4v5pute * Inv_LW;
                  pParam->BSIM4v5voff = model->BSIM4v5voff
                                    + model->BSIM4v5lvoff * Inv_L
                                    + model->BSIM4v5wvoff * Inv_W
                                    + model->BSIM4v5pvoff * Inv_LW;
                  pParam->BSIM4v5tvoff = model->BSIM4v5tvoff
                                    + model->BSIM4v5ltvoff * Inv_L
                                    + model->BSIM4v5wtvoff * Inv_W
                                    + model->BSIM4v5ptvoff * Inv_LW;
                  pParam->BSIM4v5minv = model->BSIM4v5minv
                                    + model->BSIM4v5lminv * Inv_L
                                    + model->BSIM4v5wminv * Inv_W
                                    + model->BSIM4v5pminv * Inv_LW;
                  pParam->BSIM4v5fprout = model->BSIM4v5fprout
                                     + model->BSIM4v5lfprout * Inv_L
                                     + model->BSIM4v5wfprout * Inv_W
                                     + model->BSIM4v5pfprout * Inv_LW;
                  pParam->BSIM4v5pdits = model->BSIM4v5pdits
                                     + model->BSIM4v5lpdits * Inv_L
                                     + model->BSIM4v5wpdits * Inv_W
                                     + model->BSIM4v5ppdits * Inv_LW;
                  pParam->BSIM4v5pditsd = model->BSIM4v5pditsd
                                      + model->BSIM4v5lpditsd * Inv_L
                                      + model->BSIM4v5wpditsd * Inv_W
                                      + model->BSIM4v5ppditsd * Inv_LW;
                  pParam->BSIM4v5delta = model->BSIM4v5delta
                                     + model->BSIM4v5ldelta * Inv_L
                                     + model->BSIM4v5wdelta * Inv_W
                                     + model->BSIM4v5pdelta * Inv_LW;
                  pParam->BSIM4v5rdsw = model->BSIM4v5rdsw
                                    + model->BSIM4v5lrdsw * Inv_L
                                    + model->BSIM4v5wrdsw * Inv_W
                                    + model->BSIM4v5prdsw * Inv_LW;
                  pParam->BSIM4v5rdw = model->BSIM4v5rdw
                                    + model->BSIM4v5lrdw * Inv_L
                                    + model->BSIM4v5wrdw * Inv_W
                                    + model->BSIM4v5prdw * Inv_LW;
                  pParam->BSIM4v5rsw = model->BSIM4v5rsw
                                    + model->BSIM4v5lrsw * Inv_L
                                    + model->BSIM4v5wrsw * Inv_W
                                    + model->BSIM4v5prsw * Inv_LW;
                  pParam->BSIM4v5prwg = model->BSIM4v5prwg
                                    + model->BSIM4v5lprwg * Inv_L
                                    + model->BSIM4v5wprwg * Inv_W
                                    + model->BSIM4v5pprwg * Inv_LW;
                  pParam->BSIM4v5prwb = model->BSIM4v5prwb
                                    + model->BSIM4v5lprwb * Inv_L
                                    + model->BSIM4v5wprwb * Inv_W
                                    + model->BSIM4v5pprwb * Inv_LW;
                  pParam->BSIM4v5prt = model->BSIM4v5prt
                                    + model->BSIM4v5lprt * Inv_L
                                    + model->BSIM4v5wprt * Inv_W
                                    + model->BSIM4v5pprt * Inv_LW;
                  pParam->BSIM4v5eta0 = model->BSIM4v5eta0
                                    + model->BSIM4v5leta0 * Inv_L
                                    + model->BSIM4v5weta0 * Inv_W
                                    + model->BSIM4v5peta0 * Inv_LW;
                  pParam->BSIM4v5etab = model->BSIM4v5etab
                                    + model->BSIM4v5letab * Inv_L
                                    + model->BSIM4v5wetab * Inv_W
                                    + model->BSIM4v5petab * Inv_LW;
                  pParam->BSIM4v5pclm = model->BSIM4v5pclm
                                    + model->BSIM4v5lpclm * Inv_L
                                    + model->BSIM4v5wpclm * Inv_W
                                    + model->BSIM4v5ppclm * Inv_LW;
                  pParam->BSIM4v5pdibl1 = model->BSIM4v5pdibl1
                                      + model->BSIM4v5lpdibl1 * Inv_L
                                      + model->BSIM4v5wpdibl1 * Inv_W
                                      + model->BSIM4v5ppdibl1 * Inv_LW;
                  pParam->BSIM4v5pdibl2 = model->BSIM4v5pdibl2
                                      + model->BSIM4v5lpdibl2 * Inv_L
                                      + model->BSIM4v5wpdibl2 * Inv_W
                                      + model->BSIM4v5ppdibl2 * Inv_LW;
                  pParam->BSIM4v5pdiblb = model->BSIM4v5pdiblb
                                      + model->BSIM4v5lpdiblb * Inv_L
                                      + model->BSIM4v5wpdiblb * Inv_W
                                      + model->BSIM4v5ppdiblb * Inv_LW;
                  pParam->BSIM4v5pscbe1 = model->BSIM4v5pscbe1
                                      + model->BSIM4v5lpscbe1 * Inv_L
                                      + model->BSIM4v5wpscbe1 * Inv_W
                                      + model->BSIM4v5ppscbe1 * Inv_LW;
                  pParam->BSIM4v5pscbe2 = model->BSIM4v5pscbe2
                                      + model->BSIM4v5lpscbe2 * Inv_L
                                      + model->BSIM4v5wpscbe2 * Inv_W
                                      + model->BSIM4v5ppscbe2 * Inv_LW;
                  pParam->BSIM4v5pvag = model->BSIM4v5pvag
                                    + model->BSIM4v5lpvag * Inv_L
                                    + model->BSIM4v5wpvag * Inv_W
                                    + model->BSIM4v5ppvag * Inv_LW;
                  pParam->BSIM4v5wr = model->BSIM4v5wr
                                  + model->BSIM4v5lwr * Inv_L
                                  + model->BSIM4v5wwr * Inv_W
                                  + model->BSIM4v5pwr * Inv_LW;
                  pParam->BSIM4v5dwg = model->BSIM4v5dwg
                                   + model->BSIM4v5ldwg * Inv_L
                                   + model->BSIM4v5wdwg * Inv_W
                                   + model->BSIM4v5pdwg * Inv_LW;
                  pParam->BSIM4v5dwb = model->BSIM4v5dwb
                                   + model->BSIM4v5ldwb * Inv_L
                                   + model->BSIM4v5wdwb * Inv_W
                                   + model->BSIM4v5pdwb * Inv_LW;
                  pParam->BSIM4v5b0 = model->BSIM4v5b0
                                  + model->BSIM4v5lb0 * Inv_L
                                  + model->BSIM4v5wb0 * Inv_W
                                  + model->BSIM4v5pb0 * Inv_LW;
                  pParam->BSIM4v5b1 = model->BSIM4v5b1
                                  + model->BSIM4v5lb1 * Inv_L
                                  + model->BSIM4v5wb1 * Inv_W
                                  + model->BSIM4v5pb1 * Inv_LW;
                  pParam->BSIM4v5alpha0 = model->BSIM4v5alpha0
                                      + model->BSIM4v5lalpha0 * Inv_L
                                      + model->BSIM4v5walpha0 * Inv_W
                                      + model->BSIM4v5palpha0 * Inv_LW;
                  pParam->BSIM4v5alpha1 = model->BSIM4v5alpha1
                                      + model->BSIM4v5lalpha1 * Inv_L
                                      + model->BSIM4v5walpha1 * Inv_W
                                      + model->BSIM4v5palpha1 * Inv_LW;
                  pParam->BSIM4v5beta0 = model->BSIM4v5beta0
                                     + model->BSIM4v5lbeta0 * Inv_L
                                     + model->BSIM4v5wbeta0 * Inv_W
                                     + model->BSIM4v5pbeta0 * Inv_LW;
                  pParam->BSIM4v5agidl = model->BSIM4v5agidl
                                     + model->BSIM4v5lagidl * Inv_L
                                     + model->BSIM4v5wagidl * Inv_W
                                     + model->BSIM4v5pagidl * Inv_LW;
                  pParam->BSIM4v5bgidl = model->BSIM4v5bgidl
                                     + model->BSIM4v5lbgidl * Inv_L
                                     + model->BSIM4v5wbgidl * Inv_W
                                     + model->BSIM4v5pbgidl * Inv_LW;
                  pParam->BSIM4v5cgidl = model->BSIM4v5cgidl
                                     + model->BSIM4v5lcgidl * Inv_L
                                     + model->BSIM4v5wcgidl * Inv_W
                                     + model->BSIM4v5pcgidl * Inv_LW;
                  pParam->BSIM4v5egidl = model->BSIM4v5egidl
                                     + model->BSIM4v5legidl * Inv_L
                                     + model->BSIM4v5wegidl * Inv_W
                                     + model->BSIM4v5pegidl * Inv_LW;
                  pParam->BSIM4v5aigc = model->BSIM4v5aigc
                                     + model->BSIM4v5laigc * Inv_L
                                     + model->BSIM4v5waigc * Inv_W
                                     + model->BSIM4v5paigc * Inv_LW;
                  pParam->BSIM4v5bigc = model->BSIM4v5bigc
                                     + model->BSIM4v5lbigc * Inv_L
                                     + model->BSIM4v5wbigc * Inv_W
                                     + model->BSIM4v5pbigc * Inv_LW;
                  pParam->BSIM4v5cigc = model->BSIM4v5cigc
                                     + model->BSIM4v5lcigc * Inv_L
                                     + model->BSIM4v5wcigc * Inv_W
                                     + model->BSIM4v5pcigc * Inv_LW;
                  pParam->BSIM4v5aigsd = model->BSIM4v5aigsd
                                     + model->BSIM4v5laigsd * Inv_L
                                     + model->BSIM4v5waigsd * Inv_W
                                     + model->BSIM4v5paigsd * Inv_LW;
                  pParam->BSIM4v5bigsd = model->BSIM4v5bigsd
                                     + model->BSIM4v5lbigsd * Inv_L
                                     + model->BSIM4v5wbigsd * Inv_W
                                     + model->BSIM4v5pbigsd * Inv_LW;
                  pParam->BSIM4v5cigsd = model->BSIM4v5cigsd
                                     + model->BSIM4v5lcigsd * Inv_L
                                     + model->BSIM4v5wcigsd * Inv_W
                                     + model->BSIM4v5pcigsd * Inv_LW;
                  pParam->BSIM4v5aigbacc = model->BSIM4v5aigbacc
                                       + model->BSIM4v5laigbacc * Inv_L
                                       + model->BSIM4v5waigbacc * Inv_W
                                       + model->BSIM4v5paigbacc * Inv_LW;
                  pParam->BSIM4v5bigbacc = model->BSIM4v5bigbacc
                                       + model->BSIM4v5lbigbacc * Inv_L
                                       + model->BSIM4v5wbigbacc * Inv_W
                                       + model->BSIM4v5pbigbacc * Inv_LW;
                  pParam->BSIM4v5cigbacc = model->BSIM4v5cigbacc
                                       + model->BSIM4v5lcigbacc * Inv_L
                                       + model->BSIM4v5wcigbacc * Inv_W
                                       + model->BSIM4v5pcigbacc * Inv_LW;
                  pParam->BSIM4v5aigbinv = model->BSIM4v5aigbinv
                                       + model->BSIM4v5laigbinv * Inv_L
                                       + model->BSIM4v5waigbinv * Inv_W
                                       + model->BSIM4v5paigbinv * Inv_LW;
                  pParam->BSIM4v5bigbinv = model->BSIM4v5bigbinv
                                       + model->BSIM4v5lbigbinv * Inv_L
                                       + model->BSIM4v5wbigbinv * Inv_W
                                       + model->BSIM4v5pbigbinv * Inv_LW;
                  pParam->BSIM4v5cigbinv = model->BSIM4v5cigbinv
                                       + model->BSIM4v5lcigbinv * Inv_L
                                       + model->BSIM4v5wcigbinv * Inv_W
                                       + model->BSIM4v5pcigbinv * Inv_LW;
                  pParam->BSIM4v5nigc = model->BSIM4v5nigc
                                       + model->BSIM4v5lnigc * Inv_L
                                       + model->BSIM4v5wnigc * Inv_W
                                       + model->BSIM4v5pnigc * Inv_LW;
                  pParam->BSIM4v5nigbacc = model->BSIM4v5nigbacc
                                       + model->BSIM4v5lnigbacc * Inv_L
                                       + model->BSIM4v5wnigbacc * Inv_W
                                       + model->BSIM4v5pnigbacc * Inv_LW;
                  pParam->BSIM4v5nigbinv = model->BSIM4v5nigbinv
                                       + model->BSIM4v5lnigbinv * Inv_L
                                       + model->BSIM4v5wnigbinv * Inv_W
                                       + model->BSIM4v5pnigbinv * Inv_LW;
                  pParam->BSIM4v5ntox = model->BSIM4v5ntox
                                    + model->BSIM4v5lntox * Inv_L
                                    + model->BSIM4v5wntox * Inv_W
                                    + model->BSIM4v5pntox * Inv_LW;
                  pParam->BSIM4v5eigbinv = model->BSIM4v5eigbinv
                                       + model->BSIM4v5leigbinv * Inv_L
                                       + model->BSIM4v5weigbinv * Inv_W
                                       + model->BSIM4v5peigbinv * Inv_LW;
                  pParam->BSIM4v5pigcd = model->BSIM4v5pigcd
                                     + model->BSIM4v5lpigcd * Inv_L
                                     + model->BSIM4v5wpigcd * Inv_W
                                     + model->BSIM4v5ppigcd * Inv_LW;
                  pParam->BSIM4v5poxedge = model->BSIM4v5poxedge
                                       + model->BSIM4v5lpoxedge * Inv_L
                                       + model->BSIM4v5wpoxedge * Inv_W
                                       + model->BSIM4v5ppoxedge * Inv_LW;
                  pParam->BSIM4v5xrcrg1 = model->BSIM4v5xrcrg1
                                      + model->BSIM4v5lxrcrg1 * Inv_L
                                      + model->BSIM4v5wxrcrg1 * Inv_W
                                      + model->BSIM4v5pxrcrg1 * Inv_LW;
                  pParam->BSIM4v5xrcrg2 = model->BSIM4v5xrcrg2
                                      + model->BSIM4v5lxrcrg2 * Inv_L
                                      + model->BSIM4v5wxrcrg2 * Inv_W
                                      + model->BSIM4v5pxrcrg2 * Inv_LW;
                  pParam->BSIM4v5lambda = model->BSIM4v5lambda
                                      + model->BSIM4v5llambda * Inv_L
                                      + model->BSIM4v5wlambda * Inv_W
                                      + model->BSIM4v5plambda * Inv_LW;
                  pParam->BSIM4v5vtl = model->BSIM4v5vtl
                                      + model->BSIM4v5lvtl * Inv_L
                                      + model->BSIM4v5wvtl * Inv_W
                                      + model->BSIM4v5pvtl * Inv_LW;
                  pParam->BSIM4v5xn = model->BSIM4v5xn
                                      + model->BSIM4v5lxn * Inv_L
                                      + model->BSIM4v5wxn * Inv_W
                                      + model->BSIM4v5pxn * Inv_LW;
                  pParam->BSIM4v5vfbsdoff = model->BSIM4v5vfbsdoff
                                      + model->BSIM4v5lvfbsdoff * Inv_L
                                      + model->BSIM4v5wvfbsdoff * Inv_W
                                      + model->BSIM4v5pvfbsdoff * Inv_LW;
                  pParam->BSIM4v5tvfbsdoff = model->BSIM4v5tvfbsdoff
                                      + model->BSIM4v5ltvfbsdoff * Inv_L
                                      + model->BSIM4v5wtvfbsdoff * Inv_W
                                      + model->BSIM4v5ptvfbsdoff * Inv_LW;

                  pParam->BSIM4v5cgsl = model->BSIM4v5cgsl
                                    + model->BSIM4v5lcgsl * Inv_L
                                    + model->BSIM4v5wcgsl * Inv_W
                                    + model->BSIM4v5pcgsl * Inv_LW;
                  pParam->BSIM4v5cgdl = model->BSIM4v5cgdl
                                    + model->BSIM4v5lcgdl * Inv_L
                                    + model->BSIM4v5wcgdl * Inv_W
                                    + model->BSIM4v5pcgdl * Inv_LW;
                  pParam->BSIM4v5ckappas = model->BSIM4v5ckappas
                                       + model->BSIM4v5lckappas * Inv_L
                                       + model->BSIM4v5wckappas * Inv_W
                                        + model->BSIM4v5pckappas * Inv_LW;
                  pParam->BSIM4v5ckappad = model->BSIM4v5ckappad
                                       + model->BSIM4v5lckappad * Inv_L
                                       + model->BSIM4v5wckappad * Inv_W
                                       + model->BSIM4v5pckappad * Inv_LW;
                  pParam->BSIM4v5cf = model->BSIM4v5cf
                                  + model->BSIM4v5lcf * Inv_L
                                  + model->BSIM4v5wcf * Inv_W
                                  + model->BSIM4v5pcf * Inv_LW;
                  pParam->BSIM4v5clc = model->BSIM4v5clc
                                   + model->BSIM4v5lclc * Inv_L
                                   + model->BSIM4v5wclc * Inv_W
                                   + model->BSIM4v5pclc * Inv_LW;
                  pParam->BSIM4v5cle = model->BSIM4v5cle
                                   + model->BSIM4v5lcle * Inv_L
                                   + model->BSIM4v5wcle * Inv_W
                                   + model->BSIM4v5pcle * Inv_LW;
                  pParam->BSIM4v5vfbcv = model->BSIM4v5vfbcv
                                     + model->BSIM4v5lvfbcv * Inv_L
                                     + model->BSIM4v5wvfbcv * Inv_W
                                     + model->BSIM4v5pvfbcv * Inv_LW;
                  pParam->BSIM4v5acde = model->BSIM4v5acde
                                    + model->BSIM4v5lacde * Inv_L
                                    + model->BSIM4v5wacde * Inv_W
                                    + model->BSIM4v5pacde * Inv_LW;
                  pParam->BSIM4v5moin = model->BSIM4v5moin
                                    + model->BSIM4v5lmoin * Inv_L
                                    + model->BSIM4v5wmoin * Inv_W
                                    + model->BSIM4v5pmoin * Inv_LW;
                  pParam->BSIM4v5noff = model->BSIM4v5noff
                                    + model->BSIM4v5lnoff * Inv_L
                                    + model->BSIM4v5wnoff * Inv_W
                                    + model->BSIM4v5pnoff * Inv_LW;
                  pParam->BSIM4v5voffcv = model->BSIM4v5voffcv
                                      + model->BSIM4v5lvoffcv * Inv_L
                                      + model->BSIM4v5wvoffcv * Inv_W
                                      + model->BSIM4v5pvoffcv * Inv_LW;
                  pParam->BSIM4v5kvth0we = model->BSIM4v5kvth0we
                                      + model->BSIM4v5lkvth0we * Inv_L
                                      + model->BSIM4v5wkvth0we * Inv_W
                                      + model->BSIM4v5pkvth0we * Inv_LW;
                  pParam->BSIM4v5k2we = model->BSIM4v5k2we
                                      + model->BSIM4v5lk2we * Inv_L
                                      + model->BSIM4v5wk2we * Inv_W
                                      + model->BSIM4v5pk2we * Inv_LW;
                  pParam->BSIM4v5ku0we = model->BSIM4v5ku0we
                                      + model->BSIM4v5lku0we * Inv_L
                                      + model->BSIM4v5wku0we * Inv_W
                                      + model->BSIM4v5pku0we * Inv_LW;

                  pParam->BSIM4v5abulkCVfactor = 1.0 + pow((pParam->BSIM4v5clc
                                             / pParam->BSIM4v5leffCV),
                                             pParam->BSIM4v5cle);

                  T0 = (TRatio - 1.0);

                  PowWeffWr = pow(pParam->BSIM4v5weffCJ * 1.0e6, pParam->BSIM4v5wr) * here->BSIM4v5nf;

                  T1 = T2 = T3 = T4 = 0.0;
                  if(model->BSIM4v5tempMod == 0) {
                          pParam->BSIM4v5ua = pParam->BSIM4v5ua + pParam->BSIM4v5ua1 * T0;
                          pParam->BSIM4v5ub = pParam->BSIM4v5ub + pParam->BSIM4v5ub1 * T0;
                          pParam->BSIM4v5uc = pParam->BSIM4v5uc + pParam->BSIM4v5uc1 * T0;
                          pParam->BSIM4v5ud = pParam->BSIM4v5ud + pParam->BSIM4v5ud1 * T0;
                          pParam->BSIM4v5vsattemp = pParam->BSIM4v5vsat - pParam->BSIM4v5at * T0;
                          T10 = pParam->BSIM4v5prt * T0;
                     if(model->BSIM4v5rdsMod) {
                          /* External Rd(V) */
                          T1 = pParam->BSIM4v5rdw + T10;
                          T2 = model->BSIM4v5rdwmin + T10;
                          /* External Rs(V) */
                          T3 = pParam->BSIM4v5rsw + T10;
                          T4 = model->BSIM4v5rswmin + T10;
                     }
                          /* Internal Rds(V) in IV */
                          pParam->BSIM4v5rds0 = (pParam->BSIM4v5rdsw + T10)
                                            * here->BSIM4v5nf / PowWeffWr;
                          pParam->BSIM4v5rdswmin = (model->BSIM4v5rdswmin + T10)
                                               * here->BSIM4v5nf / PowWeffWr;
                  } else { /* tempMod = 1, 2 */
                          pParam->BSIM4v5ua = pParam->BSIM4v5ua * (1.0 + pParam->BSIM4v5ua1 * delTemp) ;
                          pParam->BSIM4v5ub = pParam->BSIM4v5ub * (1.0 + pParam->BSIM4v5ub1 * delTemp);
                          pParam->BSIM4v5uc = pParam->BSIM4v5uc * (1.0 + pParam->BSIM4v5uc1 * delTemp);
                          pParam->BSIM4v5ud = pParam->BSIM4v5ud * (1.0 + pParam->BSIM4v5ud1 * delTemp);
                          pParam->BSIM4v5vsattemp = pParam->BSIM4v5vsat * (1.0 - pParam->BSIM4v5at * delTemp);
                          T10 = 1.0 + pParam->BSIM4v5prt * delTemp;
                     if(model->BSIM4v5rdsMod) {
                          /* External Rd(V) */
                          T1 = pParam->BSIM4v5rdw * T10;
                          T2 = model->BSIM4v5rdwmin * T10;
                          /* External Rs(V) */
                          T3 = pParam->BSIM4v5rsw * T10;
                          T4 = model->BSIM4v5rswmin * T10;
                     }
                          /* Internal Rds(V) in IV */
                          pParam->BSIM4v5rds0 = pParam->BSIM4v5rdsw * T10 * here->BSIM4v5nf / PowWeffWr;
                          pParam->BSIM4v5rdswmin = model->BSIM4v5rdswmin * T10 * here->BSIM4v5nf / PowWeffWr;
                  }
                  if (T1 < 0.0)
                  {   T1 = 0.0;
                      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
                  }
                  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v5rd0 = T1 / PowWeffWr;
                  pParam->BSIM4v5rdwmin = T2 / PowWeffWr;
                  if (T3 < 0.0)
                  {   T3 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  if (T4 < 0.0)
                  {   T4 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v5rs0 = T3 / PowWeffWr;
                  pParam->BSIM4v5rswmin = T4 / PowWeffWr;

                  if (pParam->BSIM4v5u0 > 1.0)
                      pParam->BSIM4v5u0 = pParam->BSIM4v5u0 / 1.0e4;

                  /* mobility channel length dependence */
                  T5 = 1.0 - pParam->BSIM4v5up * exp( - pParam->BSIM4v5leff / pParam->BSIM4v5lp);
                  pParam->BSIM4v5u0temp = pParam->BSIM4v5u0 * T5
                                      * pow(TRatio, pParam->BSIM4v5ute);
                  if (pParam->BSIM4v5eu < 0.0)
                  {   pParam->BSIM4v5eu = 0.0;
                      printf("Warning: eu has been negative; reset to 0.0.\n");
                  }

                  pParam->BSIM4v5vfbsdoff = pParam->BSIM4v5vfbsdoff * (1.0 + pParam->BSIM4v5tvfbsdoff * delTemp);
                  pParam->BSIM4v5voff = pParam->BSIM4v5voff * (1.0 + pParam->BSIM4v5tvoff * delTemp);

                /* Source End Velocity Limit  */
                        if((model->BSIM4v5vtlGiven) && (model->BSIM4v5vtl > 0.0) )
                      {
                     if(model->BSIM4v5lc < 0.0) pParam->BSIM4v5lc = 0.0;
                     else   pParam->BSIM4v5lc = model->BSIM4v5lc ;
                     T0 = pParam->BSIM4v5leff / (pParam->BSIM4v5xn * pParam->BSIM4v5leff + pParam->BSIM4v5lc);
                     pParam->BSIM4v5tfactor = (1.0 - T0) / (1.0 + T0 );
                       }

                  pParam->BSIM4v5cgdo = (model->BSIM4v5cgdo + pParam->BSIM4v5cf)
                                    * pParam->BSIM4v5weffCV;
                  pParam->BSIM4v5cgso = (model->BSIM4v5cgso + pParam->BSIM4v5cf)
                                    * pParam->BSIM4v5weffCV;
                  pParam->BSIM4v5cgbo = model->BSIM4v5cgbo * pParam->BSIM4v5leffCV * here->BSIM4v5nf;

                  if (!model->BSIM4v5ndepGiven && model->BSIM4v5gamma1Given)
                  {   T0 = pParam->BSIM4v5gamma1 * model->BSIM4v5coxe;
                      pParam->BSIM4v5ndep = 3.01248e22 * T0 * T0;
                  }

                  pParam->BSIM4v5phi = Vtm0 * log(pParam->BSIM4v5ndep / ni)
                                   + pParam->BSIM4v5phin + 0.4;

                  pParam->BSIM4v5sqrtPhi = sqrt(pParam->BSIM4v5phi);
                  pParam->BSIM4v5phis3 = pParam->BSIM4v5sqrtPhi * pParam->BSIM4v5phi;

                  pParam->BSIM4v5Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
                                     * pParam->BSIM4v5ndep * 1.0e6))
                                     * pParam->BSIM4v5sqrtPhi;
                  pParam->BSIM4v5sqrtXdep0 = sqrt(pParam->BSIM4v5Xdep0);
                  pParam->BSIM4v5litl = sqrt(3.0 * pParam->BSIM4v5xj
                                    * model->BSIM4v5toxe);
                  pParam->BSIM4v5vbi = Vtm0 * log(pParam->BSIM4v5nsd
                                   * pParam->BSIM4v5ndep / (ni * ni));

                  if (pParam->BSIM4v5ngate > 0.0)
                  {   pParam->BSIM4v5vfbsd = Vtm0 * log(pParam->BSIM4v5ngate
                                         / pParam->BSIM4v5nsd);
                  }
                  else
                      pParam->BSIM4v5vfbsd = 0.0;

                  pParam->BSIM4v5cdep0 = sqrt(Charge_q * EPSSI
                                     * pParam->BSIM4v5ndep * 1.0e6 / 2.0
                                     / pParam->BSIM4v5phi);

                  pParam->BSIM4v5ToxRatio = exp(pParam->BSIM4v5ntox
                                        * log(model->BSIM4v5toxref / model->BSIM4v5toxe))
                                        / model->BSIM4v5toxe / model->BSIM4v5toxe;
                  pParam->BSIM4v5ToxRatioEdge = exp(pParam->BSIM4v5ntox
                                            * log(model->BSIM4v5toxref
                                            / (model->BSIM4v5toxe * pParam->BSIM4v5poxedge)))
                                            / model->BSIM4v5toxe / model->BSIM4v5toxe
                                            / pParam->BSIM4v5poxedge / pParam->BSIM4v5poxedge;
                  pParam->BSIM4v5Aechvb = (model->BSIM4v5type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v5Bechvb = (model->BSIM4v5type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v5AechvbEdge = pParam->BSIM4v5Aechvb * pParam->BSIM4v5weff
                                          * model->BSIM4v5dlcig * pParam->BSIM4v5ToxRatioEdge;
                  pParam->BSIM4v5BechvbEdge = -pParam->BSIM4v5Bechvb
                                          * model->BSIM4v5toxe * pParam->BSIM4v5poxedge;
                  pParam->BSIM4v5Aechvb *= pParam->BSIM4v5weff * pParam->BSIM4v5leff
                                       * pParam->BSIM4v5ToxRatio;
                  pParam->BSIM4v5Bechvb *= -model->BSIM4v5toxe;


                  pParam->BSIM4v5mstar = 0.5 + atan(pParam->BSIM4v5minv) / PI;
                  pParam->BSIM4v5voffcbn =  pParam->BSIM4v5voff + model->BSIM4v5voffl / pParam->BSIM4v5leff;

                  pParam->BSIM4v5ldeb = sqrt(EPSSI * Vtm0 / (Charge_q
                                    * pParam->BSIM4v5ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v5acde *= pow((pParam->BSIM4v5ndep / 2.0e16), -0.25);


                  if (model->BSIM4v5k1Given || model->BSIM4v5k2Given)
                  {   if (!model->BSIM4v5k1Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v5k1 = 0.53;
                      }
                      if (!model->BSIM4v5k2Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v5k2 = -0.0186;
                      }
                      if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) { /* don't print in sensitivity */
                          if (model->BSIM4v5nsubGiven)
                              fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v5xtGiven)
                              fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v5vbxGiven)
                              fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v5gamma1Given)
                              fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v5gamma2Given)
                              fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                      }
                  }
                  else
                  {   if (!model->BSIM4v5vbxGiven)
                          pParam->BSIM4v5vbx = pParam->BSIM4v5phi - 7.7348e-4
                                           * pParam->BSIM4v5ndep
                                           * pParam->BSIM4v5xt * pParam->BSIM4v5xt;
                      if (pParam->BSIM4v5vbx > 0.0)
                          pParam->BSIM4v5vbx = -pParam->BSIM4v5vbx;
                      if (pParam->BSIM4v5vbm > 0.0)
                          pParam->BSIM4v5vbm = -pParam->BSIM4v5vbm;

                      if (!model->BSIM4v5gamma1Given)
                          pParam->BSIM4v5gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM4v5ndep)
                                              / model->BSIM4v5coxe;
                      if (!model->BSIM4v5gamma2Given)
                          pParam->BSIM4v5gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM4v5nsub)
                                              / model->BSIM4v5coxe;

                      T0 = pParam->BSIM4v5gamma1 - pParam->BSIM4v5gamma2;
                      T1 = sqrt(pParam->BSIM4v5phi - pParam->BSIM4v5vbx)
                         - pParam->BSIM4v5sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v5phi * (pParam->BSIM4v5phi
                         - pParam->BSIM4v5vbm)) - pParam->BSIM4v5phi;
                      pParam->BSIM4v5k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v5vbm);
                      pParam->BSIM4v5k1 = pParam->BSIM4v5gamma2 - 2.0
                                      * pParam->BSIM4v5k2 * sqrt(pParam->BSIM4v5phi
                                      - pParam->BSIM4v5vbm);
                  }

                  if (!model->BSIM4v5vfbGiven)
                  {   if (model->BSIM4v5vth0Given)
                      {   pParam->BSIM4v5vfb = model->BSIM4v5type * pParam->BSIM4v5vth0
                                           - pParam->BSIM4v5phi - pParam->BSIM4v5k1
                                           * pParam->BSIM4v5sqrtPhi;
                      }
                      else
                      {   pParam->BSIM4v5vfb = -1.0;
                      }
                  }
                   if (!model->BSIM4v5vth0Given)
                  {   pParam->BSIM4v5vth0 = model->BSIM4v5type * (pParam->BSIM4v5vfb
                                        + pParam->BSIM4v5phi + pParam->BSIM4v5k1
                                        * pParam->BSIM4v5sqrtPhi);
                  }

                  pParam->BSIM4v5k1ox = pParam->BSIM4v5k1 * model->BSIM4v5toxe
                                    / model->BSIM4v5toxm;

                  tmp = sqrt(EPSSI / (model->BSIM4v5epsrox * EPS0)
                      * model->BSIM4v5toxe * pParam->BSIM4v5Xdep0);
                    T0 = pParam->BSIM4v5dsub * pParam->BSIM4v5leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                    {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                            T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v5theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v5theta0vb0 = 1.0 / (MAX_EXP - 2.0);

                   T0 = pParam->BSIM4v5drout * pParam->BSIM4v5leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                         {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v5thetaRout = pParam->BSIM4v5pdibl1 * T5
                                         + pParam->BSIM4v5pdibl2;

                  tmp = sqrt(pParam->BSIM4v5Xdep0);
                  tmp1 = pParam->BSIM4v5vbi - pParam->BSIM4v5phi;
                  tmp2 = model->BSIM4v5factor1 * tmp;

                  T0 = pParam->BSIM4v5dvt1w * pParam->BSIM4v5weff
                     * pParam->BSIM4v5leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v5dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v5dvt1 * pParam->BSIM4v5leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  }
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v5dvt0 * T9 * tmp1;

                  T4 = model->BSIM4v5toxe * pParam->BSIM4v5phi
                     / (pParam->BSIM4v5weff + pParam->BSIM4v5w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v5lpe0 / pParam->BSIM4v5leff);
                  if((model->BSIM4v5tempMod == 1) || (model->BSIM4v5tempMod == 0))
                          T3 = (pParam->BSIM4v5kt1 + pParam->BSIM4v5kt1l / pParam->BSIM4v5leff)
                                     * (TRatio - 1.0);
                  if(model->BSIM4v5tempMod == 2)
                        T3 = - pParam->BSIM4v5kt1 * (TRatio - 1.0);

                  T5 = pParam->BSIM4v5k1ox * (T0 - 1.0) * pParam->BSIM4v5sqrtPhi
                     + T3;
                  pParam->BSIM4v5vfbzbfactor = - T8 - T9 + pParam->BSIM4v5k3 * T4 + T5
                                             - pParam->BSIM4v5phi - pParam->BSIM4v5k1 * pParam->BSIM4v5sqrtPhi;

                  /* stress effect */

                      wlod = model->BSIM4v5wlod;
                      if (model->BSIM4v5wlod < 0.0)
                  {   fprintf(stderr, "Warning: WLOD = %g is less than 0. 0.0 is used\n",model->BSIM4v5wlod);
                             wlod = 0.0;
                  }
                  T0 = pow(Lnew, model->BSIM4v5llodku0);
                  W_tmp = Wnew + wlod;
                  T1 = pow(W_tmp, model->BSIM4v5wlodku0);
                  tmp1 = model->BSIM4v5lku0 / T0 + model->BSIM4v5wku0 / T1
                         + model->BSIM4v5pku0 / (T0 * T1);
                  pParam->BSIM4v5ku0 = 1.0 + tmp1;

                  T0 = pow(Lnew, model->BSIM4v5llodvth);
                  T1 = pow(W_tmp, model->BSIM4v5wlodvth);
                  tmp1 = model->BSIM4v5lkvth0 / T0 + model->BSIM4v5wkvth0 / T1
                       + model->BSIM4v5pkvth0 / (T0 * T1);
                  pParam->BSIM4v5kvth0 = 1.0 + tmp1;
                  pParam->BSIM4v5kvth0 = sqrt(pParam->BSIM4v5kvth0*pParam->BSIM4v5kvth0 + DELTA);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM4v5ku0temp = pParam->BSIM4v5ku0 * (1.0 + model->BSIM4v5tku0 *T0) + DELTA;

                  Inv_saref = 1.0/(model->BSIM4v5saref + 0.5*Ldrn);
                  Inv_sbref = 1.0/(model->BSIM4v5sbref + 0.5*Ldrn);
                  pParam->BSIM4v5inv_od_ref = Inv_saref + Inv_sbref;
                  pParam->BSIM4v5rho_ref = model->BSIM4v5ku0 / pParam->BSIM4v5ku0temp * pParam->BSIM4v5inv_od_ref;

              } /* End of SizeNotFound */

              /*  stress effect */
              if( (here->BSIM4v5sa > 0.0) && (here->BSIM4v5sb > 0.0) &&
                        ((here->BSIM4v5nf == 1.0) || ((here->BSIM4v5nf > 1.0) && (here->BSIM4v5sd > 0.0))) )
              {          Inv_sa = 0;
                        Inv_sb = 0;

                         kvsat = model->BSIM4v5kvsat;
                  if (model->BSIM4v5kvsat < -1.0 )
                  {   fprintf(stderr, "Warning: KVSAT = %g is too small; -1.0 is used.\n",model->BSIM4v5kvsat);
                             kvsat = -1.0;
                      }
                      if (model->BSIM4v5kvsat > 1.0)
                      {   fprintf(stderr, "Warning: KVSAT = %g is too big; 1.0 is used.\n",model->BSIM4v5kvsat);
                         kvsat = 1.0;
                      }

                        for(i = 0; i < here->BSIM4v5nf; i++){
                           T0 = 1.0 / here->BSIM4v5nf / (here->BSIM4v5sa + 0.5*Ldrn + i * (here->BSIM4v5sd +Ldrn));
                            T1 = 1.0 / here->BSIM4v5nf / (here->BSIM4v5sb + 0.5*Ldrn + i * (here->BSIM4v5sd +Ldrn));
                           Inv_sa += T0;
                            Inv_sb += T1;
                  }
                  Inv_ODeff = Inv_sa + Inv_sb;
                  rho = model->BSIM4v5ku0 / pParam->BSIM4v5ku0temp * Inv_ODeff;
                  T0 = (1.0 + rho)/(1.0 + pParam->BSIM4v5rho_ref);
                  here->BSIM4v5u0temp = pParam->BSIM4v5u0temp * T0;

                  T1 = (1.0 + kvsat * rho)/(1.0 + kvsat * pParam->BSIM4v5rho_ref);
                  here->BSIM4v5vsattemp = pParam->BSIM4v5vsattemp * T1;

                  OD_offset = Inv_ODeff - pParam->BSIM4v5inv_od_ref;
                  dvth0_lod = model->BSIM4v5kvth0 / pParam->BSIM4v5kvth0 * OD_offset;
                  dk2_lod = model->BSIM4v5stk2 / pow(pParam->BSIM4v5kvth0, model->BSIM4v5lodk2) *
                                   OD_offset;
                  deta0_lod = model->BSIM4v5steta0 / pow(pParam->BSIM4v5kvth0, model->BSIM4v5lodeta0) *
                                     OD_offset;
                  here->BSIM4v5vth0 = pParam->BSIM4v5vth0 + dvth0_lod;

                  here->BSIM4v5eta0 = pParam->BSIM4v5eta0 + deta0_lod;
                  here->BSIM4v5k2 = pParam->BSIM4v5k2 + dk2_lod;
               } else {
                      here->BSIM4v5u0temp = pParam->BSIM4v5u0temp;
                      here->BSIM4v5vth0 = pParam->BSIM4v5vth0;
                      here->BSIM4v5vsattemp = pParam->BSIM4v5vsattemp;
                      here->BSIM4v5eta0 = pParam->BSIM4v5eta0;
                      here->BSIM4v5k2 = pParam->BSIM4v5k2;
              }

              /*  Well Proximity Effect  */
              if (model->BSIM4v5wpemod)
              { if( (!here->BSIM4v5scaGiven) && (!here->BSIM4v5scbGiven) && (!here->BSIM4v5sccGiven) )
                {   if((here->BSIM4v5scGiven) && (here->BSIM4v5sc > 0.0) )
                          {   T1 = here->BSIM4v5sc + Wdrn;
                        T2 = 1.0 / model->BSIM4v5scref;
                        here->BSIM4v5sca = model->BSIM4v5scref * model->BSIM4v5scref
                                        / (here->BSIM4v5sc * T1);
                        here->BSIM4v5scb = ( (0.1 * here->BSIM4v5sc + 0.01 * model->BSIM4v5scref)
                                        * exp(-10.0 * here->BSIM4v5sc * T2)
                                        - (0.1 * T1 + 0.01 * model->BSIM4v5scref)
                                        * exp(-10.0 * T1 * T2) ) / Wdrn;
                        here->BSIM4v5scc = ( (0.05 * here->BSIM4v5sc + 0.0025 * model->BSIM4v5scref)
                                        * exp(-20.0 * here->BSIM4v5sc * T2)
                                        - (0.05 * T1 + 0.0025 * model->BSIM4v5scref)
                                        * exp(-20.0 * T1 * T2) ) / Wdrn;
                    } else {
                      //fprintf(stderr, "Warning: No WPE as none of SCA, SCB, SCC, SC is given and/or SC not positive.\n");
                    }
                }
                sceff = here->BSIM4v5sca + model->BSIM4v5web * here->BSIM4v5scb
                      + model->BSIM4v5wec * here->BSIM4v5scc;
                here->BSIM4v5vth0 += pParam->BSIM4v5kvth0we * sceff;
                here->BSIM4v5k2 +=  pParam->BSIM4v5k2we * sceff;
                  T3 =  1.0 + pParam->BSIM4v5ku0we * sceff;
                if (T3 <= 0.0)
                {
                        fprintf(stderr, "Warning: ku0we = %g is negatively too high. Negative mobility! \n", T3);
                        T3 = 0.0;
                }
                here->BSIM4v5u0temp *= T3;
              }

            /* adding delvto  */
            here->BSIM4v5vth0 += here->BSIM4v5delvto;
            here->BSIM4v5vfb = pParam->BSIM4v5vfb + model->BSIM4v5type * here->BSIM4v5delvto;

            /* low field mobility multiplier */
            here->BSIM4v5u0temp = pParam->BSIM4v5u0temp * here->BSIM4v5mulu0;

            /* Instance variables calculation  */
            T3 = model->BSIM4v5type * here->BSIM4v5vth0
               - here->BSIM4v5vfb - pParam->BSIM4v5phi;
            T4 = T3 + T3;
            T5 = 2.5 * T3;
            here->BSIM4v5vtfbphi1 = (model->BSIM4v5type == NMOS) ? T4 : T5;
            if (here->BSIM4v5vtfbphi1 < 0.0)
                here->BSIM4v5vtfbphi1 = 0.0;

            here->BSIM4v5vtfbphi2 = 4.0 * T3;
            if (here->BSIM4v5vtfbphi2 < 0.0)
                here->BSIM4v5vtfbphi2 = 0.0;

            if (here->BSIM4v5k2 < 0.0)
            {   T0 = 0.5 * pParam->BSIM4v5k1 / here->BSIM4v5k2;
                here->BSIM4v5vbsc = 0.9 * (pParam->BSIM4v5phi - T0 * T0);
                if (here->BSIM4v5vbsc > -3.0)
                    here->BSIM4v5vbsc = -3.0;
                else if (here->BSIM4v5vbsc < -30.0)
                    here->BSIM4v5vbsc = -30.0;
            }
            else
                here->BSIM4v5vbsc = -30.0;
            if (here->BSIM4v5vbsc > pParam->BSIM4v5vbm)
                here->BSIM4v5vbsc = pParam->BSIM4v5vbm;
            here->BSIM4v5k2ox = here->BSIM4v5k2 * model->BSIM4v5toxe
                              / model->BSIM4v5toxm;

            here->BSIM4v5vfbzb = pParam->BSIM4v5vfbzbfactor
                                +  model->BSIM4v5type * here->BSIM4v5vth0 ;

              here->BSIM4v5cgso = pParam->BSIM4v5cgso;
              here->BSIM4v5cgdo = pParam->BSIM4v5cgdo;

              lnl = log(pParam->BSIM4v5leff * 1.0e6);
              lnw = log(pParam->BSIM4v5weff * 1.0e6);
              lnnf = log(here->BSIM4v5nf);

              bodymode = 5;
              if( ( !model->BSIM4v5rbps0Given) ||
                  ( !model->BSIM4v5rbpd0Given) )
                bodymode = 1;
              else
                if( (!model->BSIM4v5rbsbx0Given && !model->BSIM4v5rbsby0Given) ||
                      (!model->BSIM4v5rbdbx0Given && !model->BSIM4v5rbdby0Given) )
                  bodymode = 3;

              if(here->BSIM4v5rbodyMod == 2)
                {
                  if (bodymode == 5)
                    {
                      rbsbx =  exp( log(model->BSIM4v5rbsbx0) + model->BSIM4v5rbsdbxl * lnl +
                                    model->BSIM4v5rbsdbxw * lnw + model->BSIM4v5rbsdbxnf * lnnf );
                      rbsby =  exp( log(model->BSIM4v5rbsby0) + model->BSIM4v5rbsdbyl * lnl +
                                    model->BSIM4v5rbsdbyw * lnw + model->BSIM4v5rbsdbynf * lnnf );
                      here->BSIM4v5rbsb = rbsbx * rbsby / (rbsbx + rbsby);


                      rbdbx =  exp( log(model->BSIM4v5rbdbx0) + model->BSIM4v5rbsdbxl * lnl +
                                    model->BSIM4v5rbsdbxw * lnw + model->BSIM4v5rbsdbxnf * lnnf );
                      rbdby =  exp( log(model->BSIM4v5rbdby0) + model->BSIM4v5rbsdbyl * lnl +
                                    model->BSIM4v5rbsdbyw * lnw + model->BSIM4v5rbsdbynf * lnnf );
                      here->BSIM4v5rbdb = rbdbx * rbdby / (rbdbx + rbdby);
                    }

                  if ((bodymode == 3)|| (bodymode == 5))
                    {
                      here->BSIM4v5rbps = exp( log(model->BSIM4v5rbps0) + model->BSIM4v5rbpsl * lnl +
                                             model->BSIM4v5rbpsw * lnw + model->BSIM4v5rbpsnf * lnnf );
                      here->BSIM4v5rbpd = exp( log(model->BSIM4v5rbpd0) + model->BSIM4v5rbpdl * lnl +
                                             model->BSIM4v5rbpdw * lnw + model->BSIM4v5rbpdnf * lnnf );
                    }

                  rbpbx =  exp( log(model->BSIM4v5rbpbx0) + model->BSIM4v5rbpbxl * lnl +
                                model->BSIM4v5rbpbxw * lnw + model->BSIM4v5rbpbxnf * lnnf );
                  rbpby =  exp( log(model->BSIM4v5rbpby0) + model->BSIM4v5rbpbyl * lnl +
                                model->BSIM4v5rbpbyw * lnw + model->BSIM4v5rbpbynf * lnnf );
                  here->BSIM4v5rbpb = rbpbx*rbpby/(rbpbx + rbpby);
                }


              if ((here->BSIM4v5rbodyMod == 1 ) || ((here->BSIM4v5rbodyMod == 2 ) && (bodymode == 5)) )
              {   if (here->BSIM4v5rbdb < 1.0e-3)
                      here->BSIM4v5grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v5grbdb = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbdb;
                  if (here->BSIM4v5rbpb < 1.0e-3)
                      here->BSIM4v5grbpb = 1.0e3;
                  else
                      here->BSIM4v5grbpb = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbpb;
                  if (here->BSIM4v5rbps < 1.0e-3)
                      here->BSIM4v5grbps = 1.0e3;
                  else
                      here->BSIM4v5grbps = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbps;
                  if (here->BSIM4v5rbsb < 1.0e-3)
                      here->BSIM4v5grbsb = 1.0e3;
                  else
                      here->BSIM4v5grbsb = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbsb;
                  if (here->BSIM4v5rbpd < 1.0e-3)
                      here->BSIM4v5grbpd = 1.0e3;
                  else
                      here->BSIM4v5grbpd = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbpd;

              }

              if((here->BSIM4v5rbodyMod == 2) && (bodymode == 3))
              {
                      here->BSIM4v5grbdb = here->BSIM4v5grbsb = model->BSIM4v5gbmin;
                  if (here->BSIM4v5rbpb < 1.0e-3)
                      here->BSIM4v5grbpb = 1.0e3;
                  else
                      here->BSIM4v5grbpb = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbpb;
                  if (here->BSIM4v5rbps < 1.0e-3)
                      here->BSIM4v5grbps = 1.0e3;
                  else
                      here->BSIM4v5grbps = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbps;
                  if (here->BSIM4v5rbpd < 1.0e-3)
                      here->BSIM4v5grbpd = 1.0e3;
                  else
                      here->BSIM4v5grbpd = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbpd;
              }

              if((here->BSIM4v5rbodyMod == 2) && (bodymode == 1))
              {
                      here->BSIM4v5grbdb = here->BSIM4v5grbsb = model->BSIM4v5gbmin;
                      here->BSIM4v5grbps = here->BSIM4v5grbpd = 1.0e3;
                  if (here->BSIM4v5rbpb < 1.0e-3)
                      here->BSIM4v5grbpb = 1.0e3;
                  else
                      here->BSIM4v5grbpb = model->BSIM4v5gbmin + 1.0 / here->BSIM4v5rbpb;
              }


              /*
               * Process geomertry dependent parasitics
               */

              here->BSIM4v5grgeltd = model->BSIM4v5rshg * (here->BSIM4v5xgw
                      + pParam->BSIM4v5weffCJ / 3.0 / here->BSIM4v5ngcon) /
                      (here->BSIM4v5ngcon * here->BSIM4v5nf *
                      (Lnew - model->BSIM4v5xgl));
              if (here->BSIM4v5grgeltd > 0.0)
                  here->BSIM4v5grgeltd = 1.0 / here->BSIM4v5grgeltd;
              else
              {   here->BSIM4v5grgeltd = 1.0e3; /* mho */
                  if (here->BSIM4v5rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

              DMCGeff = model->BSIM4v5dmcg - model->BSIM4v5dmcgt;
              DMCIeff = model->BSIM4v5dmci;
              DMDGeff = model->BSIM4v5dmdg - model->BSIM4v5dmcgt;

              if (here->BSIM4v5sourcePerimeterGiven)
              {   if (model->BSIM4v5perMod == 0)
                      here->BSIM4v5Pseff = here->BSIM4v5sourcePerimeter;
                  else
                      here->BSIM4v5Pseff = here->BSIM4v5sourcePerimeter
                                       - pParam->BSIM4v5weffCJ * here->BSIM4v5nf;
              }
              else
                  BSIM4v5PAeffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod, here->BSIM4v5min,
                                    pParam->BSIM4v5weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &(here->BSIM4v5Pseff), &dumPd, &dumAs, &dumAd);

              if (here->BSIM4v5drainPerimeterGiven)
              {   if (model->BSIM4v5perMod == 0)
                      here->BSIM4v5Pdeff = here->BSIM4v5drainPerimeter;
                  else
                      here->BSIM4v5Pdeff = here->BSIM4v5drainPerimeter
                                       - pParam->BSIM4v5weffCJ * here->BSIM4v5nf;
              }
              else
                  BSIM4v5PAeffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod, here->BSIM4v5min,
                                    pParam->BSIM4v5weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &(here->BSIM4v5Pdeff), &dumAs, &dumAd);

              if (here->BSIM4v5sourceAreaGiven)
                  here->BSIM4v5Aseff = here->BSIM4v5sourceArea;
              else
                  BSIM4v5PAeffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod, here->BSIM4v5min,
                                    pParam->BSIM4v5weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &(here->BSIM4v5Aseff), &dumAd);

              if (here->BSIM4v5drainAreaGiven)
                  here->BSIM4v5Adeff = here->BSIM4v5drainArea;
              else
                  BSIM4v5PAeffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod, here->BSIM4v5min,
                                    pParam->BSIM4v5weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &dumAs, &(here->BSIM4v5Adeff));

              /* Processing S/D resistance and conductance below */
              if(here->BSIM4v5sNodePrime != here->BSIM4v5sNode)
              {
                 here->BSIM4v5sourceConductance = 0.0;
                 if(here->BSIM4v5sourceSquaresGiven)
                 {
                    here->BSIM4v5sourceConductance = model->BSIM4v5sheetResistance
                                               * here->BSIM4v5sourceSquares;
                 } else if (here->BSIM4v5rgeoMod > 0)
                 {
                    BSIM4v5RdseffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod,
                      here->BSIM4v5rgeoMod, here->BSIM4v5min,
                      pParam->BSIM4v5weffCJ, model->BSIM4v5sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v5sourceConductance));
                 } else
                 {
                    here->BSIM4v5sourceConductance = 0.0;
                 }

                 if (here->BSIM4v5sourceConductance > 0.0)
                     here->BSIM4v5sourceConductance = 1.0
                                            / here->BSIM4v5sourceConductance;
                 else
                 {
                     here->BSIM4v5sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4v5sourceConductance = 0.0;
              }

              if(here->BSIM4v5dNodePrime != here->BSIM4v5dNode)
              {
                 here->BSIM4v5drainConductance = 0.0;
                 if(here->BSIM4v5drainSquaresGiven)
                 {
                    here->BSIM4v5drainConductance = model->BSIM4v5sheetResistance
                                              * here->BSIM4v5drainSquares;
                 } else if (here->BSIM4v5rgeoMod > 0)
                 {
                    BSIM4v5RdseffGeo(here->BSIM4v5nf, here->BSIM4v5geoMod,
                      here->BSIM4v5rgeoMod, here->BSIM4v5min,
                      pParam->BSIM4v5weffCJ, model->BSIM4v5sheetResistance,
                  DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v5drainConductance));
                 } else
                 {
                    here->BSIM4v5drainConductance = 0.0;
                 }

                 if (here->BSIM4v5drainConductance > 0.0)
                     here->BSIM4v5drainConductance = 1.0
                                           / here->BSIM4v5drainConductance;
                 else
                 {
                     here->BSIM4v5drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4v5drainConductance = 0.0;
              }

               /* End of Rsd processing */


              Nvtms = model->BSIM4v5vtm * model->BSIM4v5SjctEmissionCoeff;
              if ((here->BSIM4v5Aseff <= 0.0) && (here->BSIM4v5Pseff <= 0.0))
              {   SourceSatCurrent = 1.0e-14;
              }
              else
              {   SourceSatCurrent = here->BSIM4v5Aseff * model->BSIM4v5SjctTempSatCurDensity
                                   + here->BSIM4v5Pseff * model->BSIM4v5SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v5weffCJ * here->BSIM4v5nf
                                   * model->BSIM4v5SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v5dioMod)
                  {   case 0:
                          if ((model->BSIM4v5bvs / Nvtms) > EXP_THRESHOLD)
                              here->BSIM4v5XExpBVS = model->BSIM4v5xjbvs * MIN_EXP;
                          else
                              here->BSIM4v5XExpBVS = model->BSIM4v5xjbvs * exp(-model->BSIM4v5bvs / Nvtms);
                          break;
                      case 1:
                          BSIM4v5DioIjthVjmEval(Nvtms, model->BSIM4v5ijthsfwd, SourceSatCurrent,
                                              0.0, &(here->BSIM4v5vjsmFwd));
                          here->BSIM4v5IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v5vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v5bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v5XExpBVS = model->BSIM4v5xjbvs * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v5XExpBVS = exp(-model->BSIM4v5bvs / Nvtms);
                              tmp = here->BSIM4v5XExpBVS;
                              here->BSIM4v5XExpBVS *= model->BSIM4v5xjbvs;
                          }

                          BSIM4v5DioIjthVjmEval(Nvtms, model->BSIM4v5ijthsfwd, SourceSatCurrent,
                                                     here->BSIM4v5XExpBVS, &(here->BSIM4v5vjsmFwd));
                          T0 = exp(here->BSIM4v5vjsmFwd / Nvtms);
                          here->BSIM4v5IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v5XExpBVS / T0
                                                + here->BSIM4v5XExpBVS - 1.0);
                          here->BSIM4v5SslpFwd = SourceSatCurrent
                                               * (T0 + here->BSIM4v5XExpBVS / T0) / Nvtms;

                          T2 = model->BSIM4v5ijthsrev / SourceSatCurrent;
                          if (T2 < 1.0)
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
                          }
                          here->BSIM4v5vjsmRev = -model->BSIM4v5bvs
                                             - Nvtms * log((T2 - 1.0) / model->BSIM4v5xjbvs);
                          T1 = model->BSIM4v5xjbvs * exp(-(model->BSIM4v5bvs
                             + here->BSIM4v5vjsmRev) / Nvtms);
                          here->BSIM4v5IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v5SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v5dioMod);
                  }
              }

              Nvtmd = model->BSIM4v5vtm * model->BSIM4v5DjctEmissionCoeff;
              if ((here->BSIM4v5Adeff <= 0.0) && (here->BSIM4v5Pdeff <= 0.0))
              {   DrainSatCurrent = 1.0e-14;
              }
              else
              {   DrainSatCurrent = here->BSIM4v5Adeff * model->BSIM4v5DjctTempSatCurDensity
                                  + here->BSIM4v5Pdeff * model->BSIM4v5DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v5weffCJ * here->BSIM4v5nf
                                  * model->BSIM4v5DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v5dioMod)
                  {   case 0:
                          if ((model->BSIM4v5bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v5XExpBVD = model->BSIM4v5xjbvd * MIN_EXP;
                          else
                          here->BSIM4v5XExpBVD = model->BSIM4v5xjbvd * exp(-model->BSIM4v5bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v5DioIjthVjmEval(Nvtmd, model->BSIM4v5ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v5vjdmFwd));
                          here->BSIM4v5IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v5vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v5bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v5XExpBVD = model->BSIM4v5xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v5XExpBVD = exp(-model->BSIM4v5bvd / Nvtmd);
                              tmp = here->BSIM4v5XExpBVD;
                              here->BSIM4v5XExpBVD *= model->BSIM4v5xjbvd;
                          }

                          BSIM4v5DioIjthVjmEval(Nvtmd, model->BSIM4v5ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v5XExpBVD, &(here->BSIM4v5vjdmFwd));
                          T0 = exp(here->BSIM4v5vjdmFwd / Nvtmd);
                          here->BSIM4v5IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v5XExpBVD / T0
                                              + here->BSIM4v5XExpBVD - 1.0);
                          here->BSIM4v5DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v5XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v5ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0)
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4v5vjdmRev = -model->BSIM4v5bvd
                                             - Nvtmd * log((T2 - 1.0) / model->BSIM4v5xjbvd); /* bugfix */
                          T1 = model->BSIM4v5xjbvd * exp(-(model->BSIM4v5bvd
                             + here->BSIM4v5vjdmRev) / Nvtmd);
                          here->BSIM4v5IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v5DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v5dioMod);
                  }
              }

                /* GEDL current reverse bias */
                T0 = (TRatio - 1.0);
                model->BSIM4v5njtstemp = model->BSIM4v5njts * (1.0 + model->BSIM4v5tnjts * T0);
                model->BSIM4v5njtsswtemp = model->BSIM4v5njtssw * (1.0 + model->BSIM4v5tnjtssw * T0);
                model->BSIM4v5njtsswgtemp = model->BSIM4v5njtsswg * (1.0 + model->BSIM4v5tnjtsswg * T0);
                T7 = Eg0 / model->BSIM4v5vtm * T0;
                T9 = model->BSIM4v5xtss * T7;
                DEXP(T9, T1);
                T9 = model->BSIM4v5xtsd * T7;
                DEXP(T9, T2);
                T9 = model->BSIM4v5xtssws * T7;
                DEXP(T9, T3);
                T9 = model->BSIM4v5xtsswd * T7;
                DEXP(T9, T4);
                T9 = model->BSIM4v5xtsswgs * T7;
                DEXP(T9, T5);
                T9 = model->BSIM4v5xtsswgd * T7;
                DEXP(T9, T6);

                T10 = pParam->BSIM4v5weffCJ * here->BSIM4v5nf;
                here->BSIM4v5SjctTempRevSatCur = T1 * here->BSIM4v5Aseff * model->BSIM4v5jtss;
                here->BSIM4v5DjctTempRevSatCur = T2 * here->BSIM4v5Adeff * model->BSIM4v5jtsd;
                here->BSIM4v5SswTempRevSatCur = T3 * here->BSIM4v5Pseff * model->BSIM4v5jtssws;
                here->BSIM4v5DswTempRevSatCur = T4 * here->BSIM4v5Pdeff * model->BSIM4v5jtsswd;
                here->BSIM4v5SswgTempRevSatCur = T5 * T10 * model->BSIM4v5jtsswgs;
                here->BSIM4v5DswgTempRevSatCur = T6 * T10 * model->BSIM4v5jtsswgd;


              if (BSIM4v5checkModel(model, here, ckt))
              {
                  SPfrontEnd->IFerrorf (ERR_FATAL,
                      "detected during BSIM4v5.5.0 parameter checking for \n    model %s of device instance %s\n", model->BSIM4v5modName, here->BSIM4v5name);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
