/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/
/**** BSIM4.6.5 Update ngspice 09/22/2009 ****/
/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4temp.c of BSIM4.6.3.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
  * Authors: 2008- Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006.
 * Modified by Mohan Dunga, Wenwei Yang, 05/18/2007.
 * Modified by Wenwei Yang, 07/31/2008.
 **********/


#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
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
BSIM4v6DioIjthVjmEval(double Nvtm, double Ijth, double Isb, double XExpBV, double *Vjm)
{
double Tb, Tc, EVjmovNv;

       Tc = XExpBV;
       Tb = 1.0 + Ijth / Isb - Tc;
       EVjmovNv = 0.5 * (Tb + sqrt(Tb * Tb + 4.0 * Tc));
       *Vjm = Nvtm * log(EVjmovNv);

return 0;
}


int
BSIM4v6temp(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v6model *model = (BSIM4v6model*) inModel;
BSIM4v6instance *here;
struct bsim4v6SizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam=NULL;
double tmp, tmp1, tmp2, tmp3, Eg, Eg0, ni, epssub;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Lnew=0.0, Wnew;
double delTemp, Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
double dumPs, dumPd, dumAs, dumAd, PowWeffWr;
double DMCGeff, DMCIeff, DMDGeff;
double Nvtms, Nvtmd, SourceSatCurrent, DrainSatCurrent;
double T10, T11;
double Inv_saref, Inv_sbref, Inv_sa, Inv_sb, rho, Ldrn, dvth0_lod;
double W_tmp, Inv_ODeff, OD_offset, dk2_lod, deta0_lod;
double lnl, lnw, lnnf, rbpbx, rbpby, rbsbx, rbsby, rbdbx, rbdby,bodymode;
double kvsat, wlod, sceff, Wdrn;
double V0, lt1, ltw, Theta0, Delt_vth, Vth_NarrowW, Lpe_Vb, Vth;
double n, n0, Vgsteff, Vgs_eff, niter, toxpf, toxpi, Tcen, toxe, epsrox, vddeot;
double vtfbphi2eot, phieot, TempRatioeot, Vtm0eot, Vtmeot,vbieot;

int Size_Not_Found, i;

    /*  loop through all the BSIM4v6 device models */
    for (; model != NULL; model = BSIM4v6nextModel(model))
    {    Temp = ckt->CKTtemp;
         if (model->BSIM4v6SbulkJctPotential < 0.1)
         {   model->BSIM4v6SbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbs is less than 0.1. Pbs is set to 0.1.\n");
         }
         if (model->BSIM4v6SsidewallJctPotential < 0.1)
         {   model->BSIM4v6SsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbsws is less than 0.1. Pbsws is set to 0.1.\n");
         }
         if (model->BSIM4v6SGatesidewallJctPotential < 0.1)
         {   model->BSIM4v6SGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgs is less than 0.1. Pbswgs is set to 0.1.\n");
         }

         if (model->BSIM4v6DbulkJctPotential < 0.1)
         {   model->BSIM4v6DbulkJctPotential = 0.1;
             fprintf(stderr, "Given pbd is less than 0.1. Pbd is set to 0.1.\n");
         }
         if (model->BSIM4v6DsidewallJctPotential < 0.1)
         {   model->BSIM4v6DsidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswd is less than 0.1. Pbswd is set to 0.1.\n");
         }
         if (model->BSIM4v6DGatesidewallJctPotential < 0.1)
         {   model->BSIM4v6DGatesidewallJctPotential = 0.1;
             fprintf(stderr, "Given pbswgd is less than 0.1. Pbswgd is set to 0.1.\n");
         }

         if(model->BSIM4v6mtrlMod == 0)
           {
             if ((model->BSIM4v6toxeGiven) && (model->BSIM4v6toxpGiven) && (model->BSIM4v6dtoxGiven)
                 && (model->BSIM4v6toxe != (model->BSIM4v6toxp + model->BSIM4v6dtox)))
               printf("Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored.\n");
             else if ((model->BSIM4v6toxeGiven) && (!model->BSIM4v6toxpGiven))
               model->BSIM4v6toxp = model->BSIM4v6toxe - model->BSIM4v6dtox;
             else if ((!model->BSIM4v6toxeGiven) && (model->BSIM4v6toxpGiven))
               model->BSIM4v6toxe = model->BSIM4v6toxp + model->BSIM4v6dtox;
           }

         if(model->BSIM4v6mtrlMod)
           {
             epsrox = 3.9;
             toxe = model->BSIM4v6eot;
             epssub = EPS0 * model->BSIM4v6epsrsub;
           }
         else
           {
             epsrox = model->BSIM4v6epsrox;
             toxe = model->BSIM4v6toxe;
             epssub = EPSSI;
           }


         model->BSIM4v6coxe = epsrox * EPS0 / toxe;
         if(model->BSIM4v6mtrlMod == 0)
           model->BSIM4v6coxp = model->BSIM4v6epsrox * EPS0 / model->BSIM4v6toxp;

         if (!model->BSIM4v6cgdoGiven)
         {   if (model->BSIM4v6dlcGiven && (model->BSIM4v6dlc > 0.0))
                 model->BSIM4v6cgdo = model->BSIM4v6dlc * model->BSIM4v6coxe
                                  - model->BSIM4v6cgdl ;
             else
                 model->BSIM4v6cgdo = 0.6 * model->BSIM4v6xj * model->BSIM4v6coxe;
         }
         if (!model->BSIM4v6cgsoGiven)
         {   if (model->BSIM4v6dlcGiven && (model->BSIM4v6dlc > 0.0))
                 model->BSIM4v6cgso = model->BSIM4v6dlc * model->BSIM4v6coxe
                                  - model->BSIM4v6cgsl ;
             else
                 model->BSIM4v6cgso = 0.6 * model->BSIM4v6xj * model->BSIM4v6coxe;
         }
         if (!model->BSIM4v6cgboGiven)
             model->BSIM4v6cgbo = 2.0 * model->BSIM4v6dwc * model->BSIM4v6coxe;

         struct bsim4v6SizeDependParam *p = model->pSizeDependParamKnot;
         while (p) {
             struct bsim4v6SizeDependParam *next_p = p->pNext;
             FREE(p);
             p = next_p;
         }
         model->pSizeDependParamKnot = NULL;
         pLastKnot = NULL;

         Tnom = model->BSIM4v6tnom;
         TRatio = Temp / Tnom;

         model->BSIM4v6vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
         model->BSIM4v6factor1 = sqrt(epssub / (epsrox * EPS0)* toxe);

         Vtm0 = model->BSIM4v6vtm0 = KboQ * Tnom;

         if(model->BSIM4v6mtrlMod==0)
         {
             Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
             ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15)
                 * exp(21.5565981 - Eg0 / (2.0 * Vtm0));
         }
         else
         {
           Eg0 = model->BSIM4v6bg0sub - model->BSIM4v6tbgasub * Tnom * Tnom
                                      / (Tnom + model->BSIM4v6tbgbsub);
           T0 =  model->BSIM4v6bg0sub - model->BSIM4v6tbgasub * 90090.0225
                                      / (300.15 + model->BSIM4v6tbgbsub);
           ni = model->BSIM4v6ni0sub * (Tnom / 300.15) * sqrt(Tnom / 300.15)
                 * exp((T0 - Eg0) / (2.0 * Vtm0));
         }

         model->BSIM4v6Eg0 = Eg0;
         model->BSIM4v6vtm = KboQ * Temp;
         if(model->BSIM4v6mtrlMod == 0)
           Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
         else
           Eg = model->BSIM4v6bg0sub - model->BSIM4v6tbgasub * Temp * Temp
                                      / (Temp + model->BSIM4v6tbgbsub);
         if (Temp != Tnom)
         {   T0 = Eg0 / Vtm0 - Eg / model->BSIM4v6vtm;
             T1 = log(Temp / Tnom);
             T2 = T0 + model->BSIM4v6SjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v6SjctEmissionCoeff);
             model->BSIM4v6SjctTempSatCurDensity = model->BSIM4v6SjctSatCurDensity
                                               * T3;
             model->BSIM4v6SjctSidewallTempSatCurDensity
                         = model->BSIM4v6SjctSidewallSatCurDensity * T3;
             model->BSIM4v6SjctGateSidewallTempSatCurDensity
                         = model->BSIM4v6SjctGateSidewallSatCurDensity * T3;

             T2 = T0 + model->BSIM4v6DjctTempExponent * T1;
             T3 = exp(T2 / model->BSIM4v6DjctEmissionCoeff);
             model->BSIM4v6DjctTempSatCurDensity = model->BSIM4v6DjctSatCurDensity
                                               * T3;
             model->BSIM4v6DjctSidewallTempSatCurDensity
                         = model->BSIM4v6DjctSidewallSatCurDensity * T3;
             model->BSIM4v6DjctGateSidewallTempSatCurDensity
                         = model->BSIM4v6DjctGateSidewallSatCurDensity * T3;
         }
         else
         {   model->BSIM4v6SjctTempSatCurDensity = model->BSIM4v6SjctSatCurDensity;
             model->BSIM4v6SjctSidewallTempSatCurDensity
                        = model->BSIM4v6SjctSidewallSatCurDensity;
             model->BSIM4v6SjctGateSidewallTempSatCurDensity
                        = model->BSIM4v6SjctGateSidewallSatCurDensity;
             model->BSIM4v6DjctTempSatCurDensity = model->BSIM4v6DjctSatCurDensity;
             model->BSIM4v6DjctSidewallTempSatCurDensity
                        = model->BSIM4v6DjctSidewallSatCurDensity;
             model->BSIM4v6DjctGateSidewallTempSatCurDensity
                        = model->BSIM4v6DjctGateSidewallSatCurDensity;
         }

         if (model->BSIM4v6SjctTempSatCurDensity < 0.0)
             model->BSIM4v6SjctTempSatCurDensity = 0.0;
         if (model->BSIM4v6SjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v6SjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v6SjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v6SjctGateSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v6DjctTempSatCurDensity < 0.0)
             model->BSIM4v6DjctTempSatCurDensity = 0.0;
         if (model->BSIM4v6DjctSidewallTempSatCurDensity < 0.0)
             model->BSIM4v6DjctSidewallTempSatCurDensity = 0.0;
         if (model->BSIM4v6DjctGateSidewallTempSatCurDensity < 0.0)
             model->BSIM4v6DjctGateSidewallTempSatCurDensity = 0.0;

         /* Temperature dependence of D/B and S/B diode capacitance begins */
         delTemp = ckt->CKTtemp - model->BSIM4v6tnom;
         T0 = model->BSIM4v6tcj * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v6SunitAreaTempJctCap = model->BSIM4v6SunitAreaJctCap *(1.0 + T0); /*bug_fix -JX */
             model->BSIM4v6DunitAreaTempJctCap = model->BSIM4v6DunitAreaJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v6SunitAreaJctCap > 0.0)
             {   model->BSIM4v6SunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjs to be negative. Cjs is clamped to zero.\n");
             }
             if (model->BSIM4v6DunitAreaJctCap > 0.0)
             {   model->BSIM4v6DunitAreaTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjd to be negative. Cjd is clamped to zero.\n");
             }
         }
         T0 = model->BSIM4v6tcjsw * delTemp;
                   if (model->BSIM4v6SunitLengthSidewallJctCap < 0.0)/*4.6.2*/
                      {model->BSIM4v6SunitLengthSidewallJctCap = 0.0;
                           fprintf(stderr, "CJSWS is negative. Cjsws is clamped to zero.\n");}
                  if (model->BSIM4v6DunitLengthSidewallJctCap < 0.0)
                      {model->BSIM4v6DunitLengthSidewallJctCap = 0.0;
                           fprintf(stderr, "CJSWD is negative. Cjswd is clamped to zero.\n");}
         if (T0 >= -1.0)
         {   model->BSIM4v6SunitLengthSidewallTempJctCap = model->BSIM4v6SunitLengthSidewallJctCap *(1.0 + T0);
             model->BSIM4v6DunitLengthSidewallTempJctCap = model->BSIM4v6DunitLengthSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v6SunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v6SunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero.\n");
             }
             if (model->BSIM4v6DunitLengthSidewallJctCap > 0.0)
             {   model->BSIM4v6DunitLengthSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero.\n");
             }
         }
         T0 = model->BSIM4v6tcjswg * delTemp;
         if (T0 >= -1.0)
         {   model->BSIM4v6SunitLengthGateSidewallTempJctCap = model->BSIM4v6SunitLengthGateSidewallJctCap *(1.0 + T0);
             model->BSIM4v6DunitLengthGateSidewallTempJctCap = model->BSIM4v6DunitLengthGateSidewallJctCap *(1.0 + T0);
         }
         else
         {   if (model->BSIM4v6SunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v6SunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.\n");
             }
             if (model->BSIM4v6DunitLengthGateSidewallJctCap > 0.0)
             {   model->BSIM4v6DunitLengthGateSidewallTempJctCap = 0.0;
                 fprintf(stderr, "Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.\n");
             }
         }

         model->BSIM4v6PhiBS = model->BSIM4v6SbulkJctPotential
                           - model->BSIM4v6tpb * delTemp;
         if (model->BSIM4v6PhiBS < 0.01)
         {   model->BSIM4v6PhiBS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01.\n");
         }
         model->BSIM4v6PhiBD = model->BSIM4v6DbulkJctPotential
                           - model->BSIM4v6tpb * delTemp;
         if (model->BSIM4v6PhiBD < 0.01)
         {   model->BSIM4v6PhiBD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01.\n");
         }

         model->BSIM4v6PhiBSWS = model->BSIM4v6SsidewallJctPotential
                             - model->BSIM4v6tpbsw * delTemp;
         if (model->BSIM4v6PhiBSWS <= 0.01)
         {   model->BSIM4v6PhiBSWS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.\n");
         }
         model->BSIM4v6PhiBSWD = model->BSIM4v6DsidewallJctPotential
                             - model->BSIM4v6tpbsw * delTemp;
         if (model->BSIM4v6PhiBSWD <= 0.01)
         {   model->BSIM4v6PhiBSWD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.\n");
         }

         model->BSIM4v6PhiBSWGS = model->BSIM4v6SGatesidewallJctPotential
                              - model->BSIM4v6tpbswg * delTemp;
         if (model->BSIM4v6PhiBSWGS <= 0.01)
         {   model->BSIM4v6PhiBSWGS = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.\n");
         }
         model->BSIM4v6PhiBSWGD = model->BSIM4v6DGatesidewallJctPotential
                              - model->BSIM4v6tpbswg * delTemp;
         if (model->BSIM4v6PhiBSWGD <= 0.01)
         {   model->BSIM4v6PhiBSWGD = 0.01;
             fprintf(stderr, "Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.\n");
         } /* End of junction capacitance */


         if (model->BSIM4v6ijthdfwd <= 0.0)
         {   model->BSIM4v6ijthdfwd = 0.0;
             fprintf(stderr, "Ijthdfwd reset to %g.\n", model->BSIM4v6ijthdfwd);
         }
         if (model->BSIM4v6ijthsfwd <= 0.0)
         {   model->BSIM4v6ijthsfwd = 0.0;
             fprintf(stderr, "Ijthsfwd reset to %g.\n", model->BSIM4v6ijthsfwd);
         }
         if (model->BSIM4v6ijthdrev <= 0.0)
         {   model->BSIM4v6ijthdrev = 0.0;
             fprintf(stderr, "Ijthdrev reset to %g.\n", model->BSIM4v6ijthdrev);
         }
         if (model->BSIM4v6ijthsrev <= 0.0)
         {   model->BSIM4v6ijthsrev = 0.0;
             fprintf(stderr, "Ijthsrev reset to %g.\n", model->BSIM4v6ijthsrev);
         }

         if ((model->BSIM4v6xjbvd <= 0.0) && (model->BSIM4v6dioMod == 2))
         {   model->BSIM4v6xjbvd = 0.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v6xjbvd);
         }
         else if ((model->BSIM4v6xjbvd < 0.0) && (model->BSIM4v6dioMod == 0))
         {   model->BSIM4v6xjbvd = 0.0;
             fprintf(stderr, "Xjbvd reset to %g.\n", model->BSIM4v6xjbvd);
         }

         if (model->BSIM4v6bvd <= 0.0)   /*4.6.2*/
         {   model->BSIM4v6bvd = 0.0;
             fprintf(stderr, "BVD reset to %g.\n", model->BSIM4v6bvd);
         }

         if ((model->BSIM4v6xjbvs <= 0.0) && (model->BSIM4v6dioMod == 2))
         {   model->BSIM4v6xjbvs = 0.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v6xjbvs);
         }
         else if ((model->BSIM4v6xjbvs < 0.0) && (model->BSIM4v6dioMod == 0))
         {   model->BSIM4v6xjbvs = 0.0;
             fprintf(stderr, "Xjbvs reset to %g.\n", model->BSIM4v6xjbvs);
         }

         if (model->BSIM4v6bvs <= 0.0)
         {   model->BSIM4v6bvs = 0.0;
             fprintf(stderr, "BVS reset to %g.\n", model->BSIM4v6bvs);
         }


         /* loop through all the instances of the model */
         for (here = BSIM4v6instances(model); here != NULL;
              here = BSIM4v6nextInstance(here))
         {
              pSizeDependParamKnot = model->pSizeDependParamKnot;
              Size_Not_Found = 1;
              while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
              {   if ((here->BSIM4v6l == pSizeDependParamKnot->Length)
                      && (here->BSIM4v6w == pSizeDependParamKnot->Width)
                      && (here->BSIM4v6nf == pSizeDependParamKnot->NFinger))
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
              Ldrn = here->BSIM4v6l;
              Wdrn = here->BSIM4v6w / here->BSIM4v6nf;

              if (Size_Not_Found)
              {   pParam = TMALLOC(struct bsim4v6SizeDependParam, 1);
                  if (pLastKnot == NULL)
                      model->pSizeDependParamKnot = pParam;
                  else
                      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

                  pParam->Length = here->BSIM4v6l;
                  pParam->Width = here->BSIM4v6w;
                  pParam->NFinger = here->BSIM4v6nf;
                  Lnew = here->BSIM4v6l  + model->BSIM4v6xl ;
                  Wnew = here->BSIM4v6w / here->BSIM4v6nf + model->BSIM4v6xw;

                  T0 = pow(Lnew, model->BSIM4v6Lln);
                  T1 = pow(Wnew, model->BSIM4v6Lwn);
                  tmp1 = model->BSIM4v6Ll / T0 + model->BSIM4v6Lw / T1
                       + model->BSIM4v6Lwl / (T0 * T1);
                  pParam->BSIM4v6dl = model->BSIM4v6Lint + tmp1;
                  tmp2 = model->BSIM4v6Llc / T0 + model->BSIM4v6Lwc / T1
                       + model->BSIM4v6Lwlc / (T0 * T1);
                  pParam->BSIM4v6dlc = model->BSIM4v6dlc + tmp2;

                  T2 = pow(Lnew, model->BSIM4v6Wln);
                  T3 = pow(Wnew, model->BSIM4v6Wwn);
                  tmp1 = model->BSIM4v6Wl / T2 + model->BSIM4v6Ww / T3
                       + model->BSIM4v6Wwl / (T2 * T3);
                  pParam->BSIM4v6dw = model->BSIM4v6Wint + tmp1;
                  tmp2 = model->BSIM4v6Wlc / T2 + model->BSIM4v6Wwc / T3
                       + model->BSIM4v6Wwlc / (T2 * T3);
                  pParam->BSIM4v6dwc = model->BSIM4v6dwc + tmp2;
                  pParam->BSIM4v6dwj = model->BSIM4v6dwj + tmp2;

                  pParam->BSIM4v6leff = Lnew - 2.0 * pParam->BSIM4v6dl;
                  if (pParam->BSIM4v6leff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v6: mosfet %s, model %s: Effective channel length <= 0",
                       model->BSIM4v6modName, here->BSIM4v6name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v6weff = Wnew - 2.0 * pParam->BSIM4v6dw;
                  if (pParam->BSIM4v6weff <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v6: mosfet %s, model %s: Effective channel width <= 0",
                       model->BSIM4v6modName, here->BSIM4v6name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v6leffCV = Lnew - 2.0 * pParam->BSIM4v6dlc;
                  if (pParam->BSIM4v6leffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v6: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       model->BSIM4v6modName, here->BSIM4v6name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v6weffCV = Wnew - 2.0 * pParam->BSIM4v6dwc;
                  if (pParam->BSIM4v6weffCV <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v6: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       model->BSIM4v6modName, here->BSIM4v6name);
                      return(E_BADPARM);
                  }

                  pParam->BSIM4v6weffCJ = Wnew - 2.0 * pParam->BSIM4v6dwj;
                  if (pParam->BSIM4v6weffCJ <= 0.0)
                  {
                      SPfrontEnd->IFerrorf (ERR_FATAL,
                      "BSIM4v6: mosfet %s, model %s: Effective channel width for S/D junctions <= 0",
                       model->BSIM4v6modName, here->BSIM4v6name);
                      return(E_BADPARM);
                  }


                  if (model->BSIM4v6binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM4v6leff;
                      Inv_W = 1.0e-6 / pParam->BSIM4v6weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM4v6leff
                             * pParam->BSIM4v6weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM4v6leff;
                      Inv_W = 1.0 / pParam->BSIM4v6weff;
                      Inv_LW = 1.0 / (pParam->BSIM4v6leff
                             * pParam->BSIM4v6weff);
                  }
                  pParam->BSIM4v6cdsc = model->BSIM4v6cdsc
                                    + model->BSIM4v6lcdsc * Inv_L
                                    + model->BSIM4v6wcdsc * Inv_W
                                    + model->BSIM4v6pcdsc * Inv_LW;
                  pParam->BSIM4v6cdscb = model->BSIM4v6cdscb
                                     + model->BSIM4v6lcdscb * Inv_L
                                     + model->BSIM4v6wcdscb * Inv_W
                                     + model->BSIM4v6pcdscb * Inv_LW;

                      pParam->BSIM4v6cdscd = model->BSIM4v6cdscd
                                     + model->BSIM4v6lcdscd * Inv_L
                                     + model->BSIM4v6wcdscd * Inv_W
                                     + model->BSIM4v6pcdscd * Inv_LW;

                  pParam->BSIM4v6cit = model->BSIM4v6cit
                                   + model->BSIM4v6lcit * Inv_L
                                   + model->BSIM4v6wcit * Inv_W
                                   + model->BSIM4v6pcit * Inv_LW;
                  pParam->BSIM4v6nfactor = model->BSIM4v6nfactor
                                       + model->BSIM4v6lnfactor * Inv_L
                                       + model->BSIM4v6wnfactor * Inv_W
                                       + model->BSIM4v6pnfactor * Inv_LW;
                  pParam->BSIM4v6xj = model->BSIM4v6xj
                                  + model->BSIM4v6lxj * Inv_L
                                  + model->BSIM4v6wxj * Inv_W
                                  + model->BSIM4v6pxj * Inv_LW;
                  pParam->BSIM4v6vsat = model->BSIM4v6vsat
                                    + model->BSIM4v6lvsat * Inv_L
                                    + model->BSIM4v6wvsat * Inv_W
                                    + model->BSIM4v6pvsat * Inv_LW;
                  pParam->BSIM4v6at = model->BSIM4v6at
                                  + model->BSIM4v6lat * Inv_L
                                  + model->BSIM4v6wat * Inv_W
                                  + model->BSIM4v6pat * Inv_LW;
                  pParam->BSIM4v6a0 = model->BSIM4v6a0
                                  + model->BSIM4v6la0 * Inv_L
                                  + model->BSIM4v6wa0 * Inv_W
                                  + model->BSIM4v6pa0 * Inv_LW;

                  pParam->BSIM4v6ags = model->BSIM4v6ags
                                  + model->BSIM4v6lags * Inv_L
                                  + model->BSIM4v6wags * Inv_W
                                  + model->BSIM4v6pags * Inv_LW;

                  pParam->BSIM4v6a1 = model->BSIM4v6a1
                                  + model->BSIM4v6la1 * Inv_L
                                  + model->BSIM4v6wa1 * Inv_W
                                  + model->BSIM4v6pa1 * Inv_LW;
                  pParam->BSIM4v6a2 = model->BSIM4v6a2
                                  + model->BSIM4v6la2 * Inv_L
                                  + model->BSIM4v6wa2 * Inv_W
                                  + model->BSIM4v6pa2 * Inv_LW;
                  pParam->BSIM4v6keta = model->BSIM4v6keta
                                    + model->BSIM4v6lketa * Inv_L
                                    + model->BSIM4v6wketa * Inv_W
                                    + model->BSIM4v6pketa * Inv_LW;
                  pParam->BSIM4v6nsub = model->BSIM4v6nsub
                                    + model->BSIM4v6lnsub * Inv_L
                                    + model->BSIM4v6wnsub * Inv_W
                                    + model->BSIM4v6pnsub * Inv_LW;
                  pParam->BSIM4v6ndep = model->BSIM4v6ndep
                                    + model->BSIM4v6lndep * Inv_L
                                    + model->BSIM4v6wndep * Inv_W
                                    + model->BSIM4v6pndep * Inv_LW;
                  pParam->BSIM4v6nsd = model->BSIM4v6nsd
                                   + model->BSIM4v6lnsd * Inv_L
                                   + model->BSIM4v6wnsd * Inv_W
                                   + model->BSIM4v6pnsd * Inv_LW;
                  pParam->BSIM4v6phin = model->BSIM4v6phin
                                    + model->BSIM4v6lphin * Inv_L
                                    + model->BSIM4v6wphin * Inv_W
                                    + model->BSIM4v6pphin * Inv_LW;
                  pParam->BSIM4v6ngate = model->BSIM4v6ngate
                                     + model->BSIM4v6lngate * Inv_L
                                     + model->BSIM4v6wngate * Inv_W
                                     + model->BSIM4v6pngate * Inv_LW;
                  pParam->BSIM4v6gamma1 = model->BSIM4v6gamma1
                                      + model->BSIM4v6lgamma1 * Inv_L
                                      + model->BSIM4v6wgamma1 * Inv_W
                                      + model->BSIM4v6pgamma1 * Inv_LW;
                  pParam->BSIM4v6gamma2 = model->BSIM4v6gamma2
                                      + model->BSIM4v6lgamma2 * Inv_L
                                      + model->BSIM4v6wgamma2 * Inv_W
                                      + model->BSIM4v6pgamma2 * Inv_LW;
                  pParam->BSIM4v6vbx = model->BSIM4v6vbx
                                   + model->BSIM4v6lvbx * Inv_L
                                   + model->BSIM4v6wvbx * Inv_W
                                   + model->BSIM4v6pvbx * Inv_LW;
                  pParam->BSIM4v6vbm = model->BSIM4v6vbm
                                   + model->BSIM4v6lvbm * Inv_L
                                   + model->BSIM4v6wvbm * Inv_W
                                   + model->BSIM4v6pvbm * Inv_LW;
                  pParam->BSIM4v6xt = model->BSIM4v6xt
                                   + model->BSIM4v6lxt * Inv_L
                                   + model->BSIM4v6wxt * Inv_W
                                   + model->BSIM4v6pxt * Inv_LW;
                  pParam->BSIM4v6vfb = model->BSIM4v6vfb
                                   + model->BSIM4v6lvfb * Inv_L
                                   + model->BSIM4v6wvfb * Inv_W
                                   + model->BSIM4v6pvfb * Inv_LW;
                  pParam->BSIM4v6k1 = model->BSIM4v6k1
                                  + model->BSIM4v6lk1 * Inv_L
                                  + model->BSIM4v6wk1 * Inv_W
                                  + model->BSIM4v6pk1 * Inv_LW;
                  pParam->BSIM4v6kt1 = model->BSIM4v6kt1
                                   + model->BSIM4v6lkt1 * Inv_L
                                   + model->BSIM4v6wkt1 * Inv_W
                                   + model->BSIM4v6pkt1 * Inv_LW;
                  pParam->BSIM4v6kt1l = model->BSIM4v6kt1l
                                    + model->BSIM4v6lkt1l * Inv_L
                                    + model->BSIM4v6wkt1l * Inv_W
                                    + model->BSIM4v6pkt1l * Inv_LW;
                  pParam->BSIM4v6k2 = model->BSIM4v6k2
                                  + model->BSIM4v6lk2 * Inv_L
                                  + model->BSIM4v6wk2 * Inv_W
                                  + model->BSIM4v6pk2 * Inv_LW;
                  pParam->BSIM4v6kt2 = model->BSIM4v6kt2
                                   + model->BSIM4v6lkt2 * Inv_L
                                   + model->BSIM4v6wkt2 * Inv_W
                                   + model->BSIM4v6pkt2 * Inv_LW;
                  pParam->BSIM4v6k3 = model->BSIM4v6k3
                                  + model->BSIM4v6lk3 * Inv_L
                                  + model->BSIM4v6wk3 * Inv_W
                                  + model->BSIM4v6pk3 * Inv_LW;
                  pParam->BSIM4v6k3b = model->BSIM4v6k3b
                                   + model->BSIM4v6lk3b * Inv_L
                                   + model->BSIM4v6wk3b * Inv_W
                                   + model->BSIM4v6pk3b * Inv_LW;
                  pParam->BSIM4v6w0 = model->BSIM4v6w0
                                  + model->BSIM4v6lw0 * Inv_L
                                  + model->BSIM4v6ww0 * Inv_W
                                  + model->BSIM4v6pw0 * Inv_LW;
                  pParam->BSIM4v6lpe0 = model->BSIM4v6lpe0
                                    + model->BSIM4v6llpe0 * Inv_L
                                     + model->BSIM4v6wlpe0 * Inv_W
                                    + model->BSIM4v6plpe0 * Inv_LW;
                  pParam->BSIM4v6lpeb = model->BSIM4v6lpeb
                                    + model->BSIM4v6llpeb * Inv_L
                                    + model->BSIM4v6wlpeb * Inv_W
                                    + model->BSIM4v6plpeb * Inv_LW;
                  pParam->BSIM4v6dvtp0 = model->BSIM4v6dvtp0
                                     + model->BSIM4v6ldvtp0 * Inv_L
                                     + model->BSIM4v6wdvtp0 * Inv_W
                                     + model->BSIM4v6pdvtp0 * Inv_LW;
                  pParam->BSIM4v6dvtp1 = model->BSIM4v6dvtp1
                                     + model->BSIM4v6ldvtp1 * Inv_L
                                     + model->BSIM4v6wdvtp1 * Inv_W
                                     + model->BSIM4v6pdvtp1 * Inv_LW;
                  pParam->BSIM4v6dvt0 = model->BSIM4v6dvt0
                                    + model->BSIM4v6ldvt0 * Inv_L
                                    + model->BSIM4v6wdvt0 * Inv_W
                                    + model->BSIM4v6pdvt0 * Inv_LW;
                  pParam->BSIM4v6dvt1 = model->BSIM4v6dvt1
                                    + model->BSIM4v6ldvt1 * Inv_L
                                    + model->BSIM4v6wdvt1 * Inv_W
                                    + model->BSIM4v6pdvt1 * Inv_LW;
                  pParam->BSIM4v6dvt2 = model->BSIM4v6dvt2
                                    + model->BSIM4v6ldvt2 * Inv_L
                                    + model->BSIM4v6wdvt2 * Inv_W
                                    + model->BSIM4v6pdvt2 * Inv_LW;
                  pParam->BSIM4v6dvt0w = model->BSIM4v6dvt0w
                                    + model->BSIM4v6ldvt0w * Inv_L
                                    + model->BSIM4v6wdvt0w * Inv_W
                                    + model->BSIM4v6pdvt0w * Inv_LW;
                  pParam->BSIM4v6dvt1w = model->BSIM4v6dvt1w
                                    + model->BSIM4v6ldvt1w * Inv_L
                                    + model->BSIM4v6wdvt1w * Inv_W
                                    + model->BSIM4v6pdvt1w * Inv_LW;
                  pParam->BSIM4v6dvt2w = model->BSIM4v6dvt2w
                                    + model->BSIM4v6ldvt2w * Inv_L
                                    + model->BSIM4v6wdvt2w * Inv_W
                                    + model->BSIM4v6pdvt2w * Inv_LW;
                  pParam->BSIM4v6drout = model->BSIM4v6drout
                                     + model->BSIM4v6ldrout * Inv_L
                                     + model->BSIM4v6wdrout * Inv_W
                                     + model->BSIM4v6pdrout * Inv_LW;
                  pParam->BSIM4v6dsub = model->BSIM4v6dsub
                                    + model->BSIM4v6ldsub * Inv_L
                                    + model->BSIM4v6wdsub * Inv_W
                                    + model->BSIM4v6pdsub * Inv_LW;
                  pParam->BSIM4v6vth0 = model->BSIM4v6vth0
                                    + model->BSIM4v6lvth0 * Inv_L
                                    + model->BSIM4v6wvth0 * Inv_W
                                    + model->BSIM4v6pvth0 * Inv_LW;
                  pParam->BSIM4v6ua = model->BSIM4v6ua
                                  + model->BSIM4v6lua * Inv_L
                                  + model->BSIM4v6wua * Inv_W
                                  + model->BSIM4v6pua * Inv_LW;
                  pParam->BSIM4v6ua1 = model->BSIM4v6ua1
                                   + model->BSIM4v6lua1 * Inv_L
                                   + model->BSIM4v6wua1 * Inv_W
                                   + model->BSIM4v6pua1 * Inv_LW;
                  pParam->BSIM4v6ub = model->BSIM4v6ub
                                  + model->BSIM4v6lub * Inv_L
                                  + model->BSIM4v6wub * Inv_W
                                  + model->BSIM4v6pub * Inv_LW;
                  pParam->BSIM4v6ub1 = model->BSIM4v6ub1
                                   + model->BSIM4v6lub1 * Inv_L
                                   + model->BSIM4v6wub1 * Inv_W
                                   + model->BSIM4v6pub1 * Inv_LW;
                  pParam->BSIM4v6uc = model->BSIM4v6uc
                                  + model->BSIM4v6luc * Inv_L
                                  + model->BSIM4v6wuc * Inv_W
                                  + model->BSIM4v6puc * Inv_LW;
                  pParam->BSIM4v6uc1 = model->BSIM4v6uc1
                                   + model->BSIM4v6luc1 * Inv_L
                                   + model->BSIM4v6wuc1 * Inv_W
                                   + model->BSIM4v6puc1 * Inv_LW;
                  pParam->BSIM4v6ud = model->BSIM4v6ud
                                  + model->BSIM4v6lud * Inv_L
                                  + model->BSIM4v6wud * Inv_W
                                  + model->BSIM4v6pud * Inv_LW;
                  pParam->BSIM4v6ud1 = model->BSIM4v6ud1
                                  + model->BSIM4v6lud1 * Inv_L
                                  + model->BSIM4v6wud1 * Inv_W
                                  + model->BSIM4v6pud1 * Inv_LW;
                  pParam->BSIM4v6up = model->BSIM4v6up
                                  + model->BSIM4v6lup * Inv_L
                                  + model->BSIM4v6wup * Inv_W
                                  + model->BSIM4v6pup * Inv_LW;
                  pParam->BSIM4v6lp = model->BSIM4v6lp
                                  + model->BSIM4v6llp * Inv_L
                                  + model->BSIM4v6wlp * Inv_W
                                  + model->BSIM4v6plp * Inv_LW;
                  pParam->BSIM4v6eu = model->BSIM4v6eu
                                  + model->BSIM4v6leu * Inv_L
                                  + model->BSIM4v6weu * Inv_W
                                  + model->BSIM4v6peu * Inv_LW;
                  pParam->BSIM4v6u0 = model->BSIM4v6u0
                                  + model->BSIM4v6lu0 * Inv_L
                                  + model->BSIM4v6wu0 * Inv_W
                                  + model->BSIM4v6pu0 * Inv_LW;
                  pParam->BSIM4v6ute = model->BSIM4v6ute
                                   + model->BSIM4v6lute * Inv_L
                                   + model->BSIM4v6wute * Inv_W
                                   + model->BSIM4v6pute * Inv_LW;
                /*high k mobility*/
                 pParam->BSIM4v6ucs = model->BSIM4v6ucs
                                  + model->BSIM4v6lucs * Inv_L
                                  + model->BSIM4v6wucs * Inv_W
                                  + model->BSIM4v6pucs * Inv_LW;
                  pParam->BSIM4v6ucste = model->BSIM4v6ucste
                           + model->BSIM4v6lucste * Inv_L
                                   + model->BSIM4v6wucste * Inv_W
                                   + model->BSIM4v6pucste * Inv_LW;

                  pParam->BSIM4v6voff = model->BSIM4v6voff
                                    + model->BSIM4v6lvoff * Inv_L
                                    + model->BSIM4v6wvoff * Inv_W
                                    + model->BSIM4v6pvoff * Inv_LW;
                  pParam->BSIM4v6tvoff = model->BSIM4v6tvoff
                                    + model->BSIM4v6ltvoff * Inv_L
                                    + model->BSIM4v6wtvoff * Inv_W
                                    + model->BSIM4v6ptvoff * Inv_LW;
                  pParam->BSIM4v6minv = model->BSIM4v6minv
                                    + model->BSIM4v6lminv * Inv_L
                                    + model->BSIM4v6wminv * Inv_W
                                    + model->BSIM4v6pminv * Inv_LW;
                  pParam->BSIM4v6minvcv = model->BSIM4v6minvcv
                                    + model->BSIM4v6lminvcv * Inv_L
                                    + model->BSIM4v6wminvcv * Inv_W
                                    + model->BSIM4v6pminvcv * Inv_LW;
                  pParam->BSIM4v6fprout = model->BSIM4v6fprout
                                     + model->BSIM4v6lfprout * Inv_L
                                     + model->BSIM4v6wfprout * Inv_W
                                     + model->BSIM4v6pfprout * Inv_LW;
                  pParam->BSIM4v6pdits = model->BSIM4v6pdits
                                     + model->BSIM4v6lpdits * Inv_L
                                     + model->BSIM4v6wpdits * Inv_W
                                     + model->BSIM4v6ppdits * Inv_LW;
                  pParam->BSIM4v6pditsd = model->BSIM4v6pditsd
                                      + model->BSIM4v6lpditsd * Inv_L
                                      + model->BSIM4v6wpditsd * Inv_W
                                      + model->BSIM4v6ppditsd * Inv_LW;
                  pParam->BSIM4v6delta = model->BSIM4v6delta
                                     + model->BSIM4v6ldelta * Inv_L
                                     + model->BSIM4v6wdelta * Inv_W
                                     + model->BSIM4v6pdelta * Inv_LW;
                  pParam->BSIM4v6rdsw = model->BSIM4v6rdsw
                                    + model->BSIM4v6lrdsw * Inv_L
                                    + model->BSIM4v6wrdsw * Inv_W
                                    + model->BSIM4v6prdsw * Inv_LW;
                  pParam->BSIM4v6rdw = model->BSIM4v6rdw
                                    + model->BSIM4v6lrdw * Inv_L
                                    + model->BSIM4v6wrdw * Inv_W
                                    + model->BSIM4v6prdw * Inv_LW;
                  pParam->BSIM4v6rsw = model->BSIM4v6rsw
                                    + model->BSIM4v6lrsw * Inv_L
                                    + model->BSIM4v6wrsw * Inv_W
                                    + model->BSIM4v6prsw * Inv_LW;
                  pParam->BSIM4v6prwg = model->BSIM4v6prwg
                                    + model->BSIM4v6lprwg * Inv_L
                                    + model->BSIM4v6wprwg * Inv_W
                                    + model->BSIM4v6pprwg * Inv_LW;
                  pParam->BSIM4v6prwb = model->BSIM4v6prwb
                                    + model->BSIM4v6lprwb * Inv_L
                                    + model->BSIM4v6wprwb * Inv_W
                                    + model->BSIM4v6pprwb * Inv_LW;
                  pParam->BSIM4v6prt = model->BSIM4v6prt
                                    + model->BSIM4v6lprt * Inv_L
                                    + model->BSIM4v6wprt * Inv_W
                                    + model->BSIM4v6pprt * Inv_LW;
                  pParam->BSIM4v6eta0 = model->BSIM4v6eta0
                                    + model->BSIM4v6leta0 * Inv_L
                                    + model->BSIM4v6weta0 * Inv_W
                                    + model->BSIM4v6peta0 * Inv_LW;
                  pParam->BSIM4v6etab = model->BSIM4v6etab
                                    + model->BSIM4v6letab * Inv_L
                                    + model->BSIM4v6wetab * Inv_W
                                    + model->BSIM4v6petab * Inv_LW;
                  pParam->BSIM4v6pclm = model->BSIM4v6pclm
                                    + model->BSIM4v6lpclm * Inv_L
                                    + model->BSIM4v6wpclm * Inv_W
                                    + model->BSIM4v6ppclm * Inv_LW;
                  pParam->BSIM4v6pdibl1 = model->BSIM4v6pdibl1
                                      + model->BSIM4v6lpdibl1 * Inv_L
                                      + model->BSIM4v6wpdibl1 * Inv_W
                                      + model->BSIM4v6ppdibl1 * Inv_LW;
                  pParam->BSIM4v6pdibl2 = model->BSIM4v6pdibl2
                                      + model->BSIM4v6lpdibl2 * Inv_L
                                      + model->BSIM4v6wpdibl2 * Inv_W
                                      + model->BSIM4v6ppdibl2 * Inv_LW;
                  pParam->BSIM4v6pdiblb = model->BSIM4v6pdiblb
                                      + model->BSIM4v6lpdiblb * Inv_L
                                      + model->BSIM4v6wpdiblb * Inv_W
                                      + model->BSIM4v6ppdiblb * Inv_LW;
                  pParam->BSIM4v6pscbe1 = model->BSIM4v6pscbe1
                                      + model->BSIM4v6lpscbe1 * Inv_L
                                      + model->BSIM4v6wpscbe1 * Inv_W
                                      + model->BSIM4v6ppscbe1 * Inv_LW;
                  pParam->BSIM4v6pscbe2 = model->BSIM4v6pscbe2
                                      + model->BSIM4v6lpscbe2 * Inv_L
                                      + model->BSIM4v6wpscbe2 * Inv_W
                                      + model->BSIM4v6ppscbe2 * Inv_LW;
                  pParam->BSIM4v6pvag = model->BSIM4v6pvag
                                    + model->BSIM4v6lpvag * Inv_L
                                    + model->BSIM4v6wpvag * Inv_W
                                    + model->BSIM4v6ppvag * Inv_LW;
                  pParam->BSIM4v6wr = model->BSIM4v6wr
                                  + model->BSIM4v6lwr * Inv_L
                                  + model->BSIM4v6wwr * Inv_W
                                  + model->BSIM4v6pwr * Inv_LW;
                  pParam->BSIM4v6dwg = model->BSIM4v6dwg
                                   + model->BSIM4v6ldwg * Inv_L
                                   + model->BSIM4v6wdwg * Inv_W
                                   + model->BSIM4v6pdwg * Inv_LW;
                  pParam->BSIM4v6dwb = model->BSIM4v6dwb
                                   + model->BSIM4v6ldwb * Inv_L
                                   + model->BSIM4v6wdwb * Inv_W
                                   + model->BSIM4v6pdwb * Inv_LW;
                  pParam->BSIM4v6b0 = model->BSIM4v6b0
                                  + model->BSIM4v6lb0 * Inv_L
                                  + model->BSIM4v6wb0 * Inv_W
                                  + model->BSIM4v6pb0 * Inv_LW;
                  pParam->BSIM4v6b1 = model->BSIM4v6b1
                                  + model->BSIM4v6lb1 * Inv_L
                                  + model->BSIM4v6wb1 * Inv_W
                                  + model->BSIM4v6pb1 * Inv_LW;
                  pParam->BSIM4v6alpha0 = model->BSIM4v6alpha0
                                      + model->BSIM4v6lalpha0 * Inv_L
                                      + model->BSIM4v6walpha0 * Inv_W
                                      + model->BSIM4v6palpha0 * Inv_LW;
                  pParam->BSIM4v6alpha1 = model->BSIM4v6alpha1
                                      + model->BSIM4v6lalpha1 * Inv_L
                                      + model->BSIM4v6walpha1 * Inv_W
                                      + model->BSIM4v6palpha1 * Inv_LW;
                  pParam->BSIM4v6beta0 = model->BSIM4v6beta0
                                     + model->BSIM4v6lbeta0 * Inv_L
                                     + model->BSIM4v6wbeta0 * Inv_W
                                     + model->BSIM4v6pbeta0 * Inv_LW;
                  pParam->BSIM4v6agidl = model->BSIM4v6agidl
                                     + model->BSIM4v6lagidl * Inv_L
                                     + model->BSIM4v6wagidl * Inv_W
                                     + model->BSIM4v6pagidl * Inv_LW;
                  pParam->BSIM4v6bgidl = model->BSIM4v6bgidl
                                     + model->BSIM4v6lbgidl * Inv_L
                                     + model->BSIM4v6wbgidl * Inv_W
                                     + model->BSIM4v6pbgidl * Inv_LW;
                  pParam->BSIM4v6cgidl = model->BSIM4v6cgidl
                                     + model->BSIM4v6lcgidl * Inv_L
                                     + model->BSIM4v6wcgidl * Inv_W
                                     + model->BSIM4v6pcgidl * Inv_LW;
                  pParam->BSIM4v6egidl = model->BSIM4v6egidl
                                     + model->BSIM4v6legidl * Inv_L
                                     + model->BSIM4v6wegidl * Inv_W
                                     + model->BSIM4v6pegidl * Inv_LW;
                  pParam->BSIM4v6agisl = model->BSIM4v6agisl
                                     + model->BSIM4v6lagisl * Inv_L
                                     + model->BSIM4v6wagisl * Inv_W
                                     + model->BSIM4v6pagisl * Inv_LW;
                  pParam->BSIM4v6bgisl = model->BSIM4v6bgisl
                                     + model->BSIM4v6lbgisl * Inv_L
                                     + model->BSIM4v6wbgisl * Inv_W
                                     + model->BSIM4v6pbgisl * Inv_LW;
                  pParam->BSIM4v6cgisl = model->BSIM4v6cgisl
                                     + model->BSIM4v6lcgisl * Inv_L
                                     + model->BSIM4v6wcgisl * Inv_W
                                     + model->BSIM4v6pcgisl * Inv_LW;
                  pParam->BSIM4v6egisl = model->BSIM4v6egisl
                                     + model->BSIM4v6legisl * Inv_L
                                     + model->BSIM4v6wegisl * Inv_W
                                     + model->BSIM4v6pegisl * Inv_LW;
                  pParam->BSIM4v6aigc = model->BSIM4v6aigc
                                     + model->BSIM4v6laigc * Inv_L
                                     + model->BSIM4v6waigc * Inv_W
                                     + model->BSIM4v6paigc * Inv_LW;
                  pParam->BSIM4v6bigc = model->BSIM4v6bigc
                                     + model->BSIM4v6lbigc * Inv_L
                                     + model->BSIM4v6wbigc * Inv_W
                                     + model->BSIM4v6pbigc * Inv_LW;
                  pParam->BSIM4v6cigc = model->BSIM4v6cigc
                                     + model->BSIM4v6lcigc * Inv_L
                                     + model->BSIM4v6wcigc * Inv_W
                                     + model->BSIM4v6pcigc * Inv_LW;
                  pParam->BSIM4v6aigs = model->BSIM4v6aigs
                                     + model->BSIM4v6laigs * Inv_L
                                     + model->BSIM4v6waigs * Inv_W
                                     + model->BSIM4v6paigs * Inv_LW;
                  pParam->BSIM4v6bigs = model->BSIM4v6bigs
                                     + model->BSIM4v6lbigs * Inv_L
                                     + model->BSIM4v6wbigs * Inv_W
                                     + model->BSIM4v6pbigs * Inv_LW;
                  pParam->BSIM4v6cigs = model->BSIM4v6cigs
                                     + model->BSIM4v6lcigs * Inv_L
                                     + model->BSIM4v6wcigs * Inv_W
                                     + model->BSIM4v6pcigs * Inv_LW;
                  pParam->BSIM4v6aigd = model->BSIM4v6aigd
                                     + model->BSIM4v6laigd * Inv_L
                                     + model->BSIM4v6waigd * Inv_W
                                     + model->BSIM4v6paigd * Inv_LW;
                  pParam->BSIM4v6bigd = model->BSIM4v6bigd
                                     + model->BSIM4v6lbigd * Inv_L
                                     + model->BSIM4v6wbigd * Inv_W
                                     + model->BSIM4v6pbigd * Inv_LW;
                  pParam->BSIM4v6cigd = model->BSIM4v6cigd
                                     + model->BSIM4v6lcigd * Inv_L
                                     + model->BSIM4v6wcigd * Inv_W
                                     + model->BSIM4v6pcigd * Inv_LW;
                  pParam->BSIM4v6aigbacc = model->BSIM4v6aigbacc
                                       + model->BSIM4v6laigbacc * Inv_L
                                       + model->BSIM4v6waigbacc * Inv_W
                                       + model->BSIM4v6paigbacc * Inv_LW;
                  pParam->BSIM4v6bigbacc = model->BSIM4v6bigbacc
                                       + model->BSIM4v6lbigbacc * Inv_L
                                       + model->BSIM4v6wbigbacc * Inv_W
                                       + model->BSIM4v6pbigbacc * Inv_LW;
                  pParam->BSIM4v6cigbacc = model->BSIM4v6cigbacc
                                       + model->BSIM4v6lcigbacc * Inv_L
                                       + model->BSIM4v6wcigbacc * Inv_W
                                       + model->BSIM4v6pcigbacc * Inv_LW;
                  pParam->BSIM4v6aigbinv = model->BSIM4v6aigbinv
                                       + model->BSIM4v6laigbinv * Inv_L
                                       + model->BSIM4v6waigbinv * Inv_W
                                       + model->BSIM4v6paigbinv * Inv_LW;
                  pParam->BSIM4v6bigbinv = model->BSIM4v6bigbinv
                                       + model->BSIM4v6lbigbinv * Inv_L
                                       + model->BSIM4v6wbigbinv * Inv_W
                                       + model->BSIM4v6pbigbinv * Inv_LW;
                  pParam->BSIM4v6cigbinv = model->BSIM4v6cigbinv
                                       + model->BSIM4v6lcigbinv * Inv_L
                                       + model->BSIM4v6wcigbinv * Inv_W
                                       + model->BSIM4v6pcigbinv * Inv_LW;
                  pParam->BSIM4v6nigc = model->BSIM4v6nigc
                                       + model->BSIM4v6lnigc * Inv_L
                                       + model->BSIM4v6wnigc * Inv_W
                                       + model->BSIM4v6pnigc * Inv_LW;
                  pParam->BSIM4v6nigbacc = model->BSIM4v6nigbacc
                                       + model->BSIM4v6lnigbacc * Inv_L
                                       + model->BSIM4v6wnigbacc * Inv_W
                                       + model->BSIM4v6pnigbacc * Inv_LW;
                  pParam->BSIM4v6nigbinv = model->BSIM4v6nigbinv
                                       + model->BSIM4v6lnigbinv * Inv_L
                                       + model->BSIM4v6wnigbinv * Inv_W
                                       + model->BSIM4v6pnigbinv * Inv_LW;
                  pParam->BSIM4v6ntox = model->BSIM4v6ntox
                                    + model->BSIM4v6lntox * Inv_L
                                    + model->BSIM4v6wntox * Inv_W
                                    + model->BSIM4v6pntox * Inv_LW;
                  pParam->BSIM4v6eigbinv = model->BSIM4v6eigbinv
                                       + model->BSIM4v6leigbinv * Inv_L
                                       + model->BSIM4v6weigbinv * Inv_W
                                       + model->BSIM4v6peigbinv * Inv_LW;
                  pParam->BSIM4v6pigcd = model->BSIM4v6pigcd
                                     + model->BSIM4v6lpigcd * Inv_L
                                     + model->BSIM4v6wpigcd * Inv_W
                                     + model->BSIM4v6ppigcd * Inv_LW;
                  pParam->BSIM4v6poxedge = model->BSIM4v6poxedge
                                       + model->BSIM4v6lpoxedge * Inv_L
                                       + model->BSIM4v6wpoxedge * Inv_W
                                       + model->BSIM4v6ppoxedge * Inv_LW;
                  pParam->BSIM4v6xrcrg1 = model->BSIM4v6xrcrg1
                                      + model->BSIM4v6lxrcrg1 * Inv_L
                                      + model->BSIM4v6wxrcrg1 * Inv_W
                                      + model->BSIM4v6pxrcrg1 * Inv_LW;
                  pParam->BSIM4v6xrcrg2 = model->BSIM4v6xrcrg2
                                      + model->BSIM4v6lxrcrg2 * Inv_L
                                      + model->BSIM4v6wxrcrg2 * Inv_W
                                      + model->BSIM4v6pxrcrg2 * Inv_LW;
                  pParam->BSIM4v6lambda = model->BSIM4v6lambda
                                      + model->BSIM4v6llambda * Inv_L
                                      + model->BSIM4v6wlambda * Inv_W
                                      + model->BSIM4v6plambda * Inv_LW;
                  pParam->BSIM4v6vtl = model->BSIM4v6vtl
                                      + model->BSIM4v6lvtl * Inv_L
                                      + model->BSIM4v6wvtl * Inv_W
                                      + model->BSIM4v6pvtl * Inv_LW;
                  pParam->BSIM4v6xn = model->BSIM4v6xn
                                      + model->BSIM4v6lxn * Inv_L
                                      + model->BSIM4v6wxn * Inv_W
                                      + model->BSIM4v6pxn * Inv_LW;
                  pParam->BSIM4v6vfbsdoff = model->BSIM4v6vfbsdoff
                                      + model->BSIM4v6lvfbsdoff * Inv_L
                                      + model->BSIM4v6wvfbsdoff * Inv_W
                                      + model->BSIM4v6pvfbsdoff * Inv_LW;
                  pParam->BSIM4v6tvfbsdoff = model->BSIM4v6tvfbsdoff
                                      + model->BSIM4v6ltvfbsdoff * Inv_L
                                      + model->BSIM4v6wtvfbsdoff * Inv_W
                                      + model->BSIM4v6ptvfbsdoff * Inv_LW;

                  pParam->BSIM4v6cgsl = model->BSIM4v6cgsl
                                    + model->BSIM4v6lcgsl * Inv_L
                                    + model->BSIM4v6wcgsl * Inv_W
                                    + model->BSIM4v6pcgsl * Inv_LW;
                  pParam->BSIM4v6cgdl = model->BSIM4v6cgdl
                                    + model->BSIM4v6lcgdl * Inv_L
                                    + model->BSIM4v6wcgdl * Inv_W
                                    + model->BSIM4v6pcgdl * Inv_LW;
                  pParam->BSIM4v6ckappas = model->BSIM4v6ckappas
                                       + model->BSIM4v6lckappas * Inv_L
                                       + model->BSIM4v6wckappas * Inv_W
                                        + model->BSIM4v6pckappas * Inv_LW;
                  pParam->BSIM4v6ckappad = model->BSIM4v6ckappad
                                       + model->BSIM4v6lckappad * Inv_L
                                       + model->BSIM4v6wckappad * Inv_W
                                       + model->BSIM4v6pckappad * Inv_LW;
                  pParam->BSIM4v6cf = model->BSIM4v6cf
                                  + model->BSIM4v6lcf * Inv_L
                                  + model->BSIM4v6wcf * Inv_W
                                  + model->BSIM4v6pcf * Inv_LW;
                  pParam->BSIM4v6clc = model->BSIM4v6clc
                                   + model->BSIM4v6lclc * Inv_L
                                   + model->BSIM4v6wclc * Inv_W
                                   + model->BSIM4v6pclc * Inv_LW;
                  pParam->BSIM4v6cle = model->BSIM4v6cle
                                   + model->BSIM4v6lcle * Inv_L
                                   + model->BSIM4v6wcle * Inv_W
                                   + model->BSIM4v6pcle * Inv_LW;
                  pParam->BSIM4v6vfbcv = model->BSIM4v6vfbcv
                                     + model->BSIM4v6lvfbcv * Inv_L
                                     + model->BSIM4v6wvfbcv * Inv_W
                                     + model->BSIM4v6pvfbcv * Inv_LW;
                  pParam->BSIM4v6acde = model->BSIM4v6acde
                                    + model->BSIM4v6lacde * Inv_L
                                    + model->BSIM4v6wacde * Inv_W
                                    + model->BSIM4v6pacde * Inv_LW;
                  pParam->BSIM4v6moin = model->BSIM4v6moin
                                    + model->BSIM4v6lmoin * Inv_L
                                    + model->BSIM4v6wmoin * Inv_W
                                    + model->BSIM4v6pmoin * Inv_LW;
                  pParam->BSIM4v6noff = model->BSIM4v6noff
                                    + model->BSIM4v6lnoff * Inv_L
                                    + model->BSIM4v6wnoff * Inv_W
                                    + model->BSIM4v6pnoff * Inv_LW;
                  pParam->BSIM4v6voffcv = model->BSIM4v6voffcv
                                      + model->BSIM4v6lvoffcv * Inv_L
                                      + model->BSIM4v6wvoffcv * Inv_W
                                      + model->BSIM4v6pvoffcv * Inv_LW;
                  pParam->BSIM4v6kvth0we = model->BSIM4v6kvth0we
                                      + model->BSIM4v6lkvth0we * Inv_L
                                      + model->BSIM4v6wkvth0we * Inv_W
                                      + model->BSIM4v6pkvth0we * Inv_LW;
                  pParam->BSIM4v6k2we = model->BSIM4v6k2we
                                      + model->BSIM4v6lk2we * Inv_L
                                      + model->BSIM4v6wk2we * Inv_W
                                      + model->BSIM4v6pk2we * Inv_LW;
                  pParam->BSIM4v6ku0we = model->BSIM4v6ku0we
                                      + model->BSIM4v6lku0we * Inv_L
                                      + model->BSIM4v6wku0we * Inv_W
                                      + model->BSIM4v6pku0we * Inv_LW;

                  pParam->BSIM4v6abulkCVfactor = 1.0 + pow((pParam->BSIM4v6clc
                                             / pParam->BSIM4v6leffCV),
                                             pParam->BSIM4v6cle);

                  T0 = (TRatio - 1.0);

                  PowWeffWr = pow(pParam->BSIM4v6weffCJ * 1.0e6, pParam->BSIM4v6wr) * here->BSIM4v6nf;

                  T1 = T2 = T3 = T4 = 0.0;
                          pParam->BSIM4v6ucs = pParam->BSIM4v6ucs * pow(TRatio, pParam->BSIM4v6ucste);
                  if(model->BSIM4v6tempMod == 0) {
                          pParam->BSIM4v6ua = pParam->BSIM4v6ua + pParam->BSIM4v6ua1 * T0;
                          pParam->BSIM4v6ub = pParam->BSIM4v6ub + pParam->BSIM4v6ub1 * T0;
                          pParam->BSIM4v6uc = pParam->BSIM4v6uc + pParam->BSIM4v6uc1 * T0;
                          pParam->BSIM4v6ud = pParam->BSIM4v6ud + pParam->BSIM4v6ud1 * T0;
                          pParam->BSIM4v6vsattemp = pParam->BSIM4v6vsat - pParam->BSIM4v6at * T0;
                          T10 = pParam->BSIM4v6prt * T0;
                     if(model->BSIM4v6rdsMod) {
                          /* External Rd(V) */
                          T1 = pParam->BSIM4v6rdw + T10;
                          T2 = model->BSIM4v6rdwmin + T10;
                          /* External Rs(V) */
                          T3 = pParam->BSIM4v6rsw + T10;
                          T4 = model->BSIM4v6rswmin + T10;
                     }
                          /* Internal Rds(V) in IV */
                          pParam->BSIM4v6rds0 = (pParam->BSIM4v6rdsw + T10)
                                            * here->BSIM4v6nf / PowWeffWr;
                          pParam->BSIM4v6rdswmin = (model->BSIM4v6rdswmin + T10)
                                               * here->BSIM4v6nf / PowWeffWr;
                  } else {
                        if (model->BSIM4v6tempMod == 3)
                          {pParam->BSIM4v6ua = pParam->BSIM4v6ua * pow(TRatio, pParam->BSIM4v6ua1) ;
                             pParam->BSIM4v6ub = pParam->BSIM4v6ub * pow(TRatio, pParam->BSIM4v6ub1);
                             pParam->BSIM4v6uc = pParam->BSIM4v6uc * pow(TRatio, pParam->BSIM4v6uc1);
                             pParam->BSIM4v6ud = pParam->BSIM4v6ud * pow(TRatio, pParam->BSIM4v6ud1);
                        }
                        else{  /* tempMod = 1, 2 */
                             pParam->BSIM4v6ua = pParam->BSIM4v6ua * (1.0 + pParam->BSIM4v6ua1 * delTemp) ;
                             pParam->BSIM4v6ub = pParam->BSIM4v6ub * (1.0 + pParam->BSIM4v6ub1 * delTemp);
                             pParam->BSIM4v6uc = pParam->BSIM4v6uc * (1.0 + pParam->BSIM4v6uc1 * delTemp);
                             pParam->BSIM4v6ud = pParam->BSIM4v6ud * (1.0 + pParam->BSIM4v6ud1 * delTemp);
                        }
                          pParam->BSIM4v6vsattemp = pParam->BSIM4v6vsat * (1.0 - pParam->BSIM4v6at * delTemp);
                          T10 = 1.0 + pParam->BSIM4v6prt * delTemp;
                     if(model->BSIM4v6rdsMod) {
                          /* External Rd(V) */
                          T1 = pParam->BSIM4v6rdw * T10;
                          T2 = model->BSIM4v6rdwmin * T10;
                          /* External Rs(V) */
                          T3 = pParam->BSIM4v6rsw * T10;
                          T4 = model->BSIM4v6rswmin * T10;
                     }
                          /* Internal Rds(V) in IV */
                          pParam->BSIM4v6rds0 = pParam->BSIM4v6rdsw * T10 * here->BSIM4v6nf / PowWeffWr;
                          pParam->BSIM4v6rdswmin = model->BSIM4v6rdswmin * T10 * here->BSIM4v6nf / PowWeffWr;
                  }
                  if (T1 < 0.0)
                  {   T1 = 0.0;
                      printf("Warning: Rdw at current temperature is negative; set to 0.\n");
                  }
                  if (T2 < 0.0)
                  {   T2 = 0.0;
                      printf("Warning: Rdwmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v6rd0 = T1 / PowWeffWr;
                  pParam->BSIM4v6rdwmin = T2 / PowWeffWr;
                  if (T3 < 0.0)
                  {   T3 = 0.0;
                      printf("Warning: Rsw at current temperature is negative; set to 0.\n");
                  }
                  if (T4 < 0.0)
                  {   T4 = 0.0;
                      printf("Warning: Rswmin at current temperature is negative; set to 0.\n");
                  }
                  pParam->BSIM4v6rs0 = T3 / PowWeffWr;
                  pParam->BSIM4v6rswmin = T4 / PowWeffWr;

                  if (pParam->BSIM4v6u0 > 1.0)
                      pParam->BSIM4v6u0 = pParam->BSIM4v6u0 / 1.0e4;

                  /* mobility channel length dependence */
                  T5 = 1.0 - pParam->BSIM4v6up * exp( - pParam->BSIM4v6leff / pParam->BSIM4v6lp);
                  pParam->BSIM4v6u0temp = pParam->BSIM4v6u0 * T5
                                      * pow(TRatio, pParam->BSIM4v6ute);
                  if (pParam->BSIM4v6eu < 0.0)
                  {   pParam->BSIM4v6eu = 0.0;
                      printf("Warning: eu has been negative; reset to 0.0.\n");
                  }
                  if (pParam->BSIM4v6ucs < 0.0)
                  {   pParam->BSIM4v6ucs = 0.0;
                      printf("Warning: ucs has been negative; reset to 0.0.\n");
                  }

                  pParam->BSIM4v6vfbsdoff = pParam->BSIM4v6vfbsdoff * (1.0 + pParam->BSIM4v6tvfbsdoff * delTemp);
                  pParam->BSIM4v6voff = pParam->BSIM4v6voff * (1.0 + pParam->BSIM4v6tvoff * delTemp);

                /* Source End Velocity Limit  */
                        if((model->BSIM4v6vtlGiven) && (model->BSIM4v6vtl > 0.0) )
                      {
                     if(model->BSIM4v6lc < 0.0) pParam->BSIM4v6lc = 0.0;
                     else   pParam->BSIM4v6lc = model->BSIM4v6lc ;
                     T0 = pParam->BSIM4v6leff / (pParam->BSIM4v6xn * pParam->BSIM4v6leff + pParam->BSIM4v6lc);
                     pParam->BSIM4v6tfactor = (1.0 - T0) / (1.0 + T0 );
                       }

                  pParam->BSIM4v6cgdo = (model->BSIM4v6cgdo + pParam->BSIM4v6cf)
                                    * pParam->BSIM4v6weffCV;
                  pParam->BSIM4v6cgso = (model->BSIM4v6cgso + pParam->BSIM4v6cf)
                                    * pParam->BSIM4v6weffCV;
                  pParam->BSIM4v6cgbo = model->BSIM4v6cgbo * pParam->BSIM4v6leffCV * here->BSIM4v6nf;

                  if (!model->BSIM4v6ndepGiven && model->BSIM4v6gamma1Given)
                  {   T0 = pParam->BSIM4v6gamma1 * model->BSIM4v6coxe;
                      pParam->BSIM4v6ndep = 3.01248e22 * T0 * T0;
                  }

                  pParam->BSIM4v6phi = Vtm0 * log(pParam->BSIM4v6ndep / ni)
                                   + pParam->BSIM4v6phin + 0.4;

                  pParam->BSIM4v6sqrtPhi = sqrt(pParam->BSIM4v6phi);
                  pParam->BSIM4v6phis3 = pParam->BSIM4v6sqrtPhi * pParam->BSIM4v6phi;

                  pParam->BSIM4v6Xdep0 = sqrt(2.0 * epssub / (Charge_q
                                     * pParam->BSIM4v6ndep * 1.0e6))
                                     * pParam->BSIM4v6sqrtPhi;
                  pParam->BSIM4v6sqrtXdep0 = sqrt(pParam->BSIM4v6Xdep0);

                  if(model->BSIM4v6mtrlMod == 0)
                    pParam->BSIM4v6litl = sqrt(3.0 * 3.9 / epsrox * pParam->BSIM4v6xj * toxe);
                  else
                    pParam->BSIM4v6litl = sqrt(model->BSIM4v6epsrsub/epsrox * pParam->BSIM4v6xj * toxe);

                  pParam->BSIM4v6vbi = Vtm0 * log(pParam->BSIM4v6nsd
                                   * pParam->BSIM4v6ndep / (ni * ni));

                  if (model->BSIM4v6mtrlMod == 0)
                  {
                    if (pParam->BSIM4v6ngate > 0.0)
                    {   pParam->BSIM4v6vfbsd = Vtm0 * log(pParam->BSIM4v6ngate
                                         / pParam->BSIM4v6nsd);
                     }
                    else
                      pParam->BSIM4v6vfbsd = 0.0;
                  }
                  else
                  {
                    T0 = Vtm0 * log(pParam->BSIM4v6nsd/ni);
                    T1 = 0.5 * Eg0;
                    if(T0 > T1)
                      T0 = T1;
                    T2 = model->BSIM4v6easub + T1 - model->BSIM4v6type * T0;
                    pParam->BSIM4v6vfbsd = model->BSIM4v6phig - T2;
                  }

                  pParam->BSIM4v6cdep0 = sqrt(Charge_q * epssub
                                     * pParam->BSIM4v6ndep * 1.0e6 / 2.0
                                     / pParam->BSIM4v6phi);

                  pParam->BSIM4v6ToxRatio = exp(pParam->BSIM4v6ntox
                                        * log(model->BSIM4v6toxref / toxe))
                                        / toxe / toxe;
                  pParam->BSIM4v6ToxRatioEdge = exp(pParam->BSIM4v6ntox
                                            * log(model->BSIM4v6toxref
                                            / (toxe * pParam->BSIM4v6poxedge)))
                                            / toxe / toxe
                                            / pParam->BSIM4v6poxedge / pParam->BSIM4v6poxedge;
                  pParam->BSIM4v6Aechvb = (model->BSIM4v6type == NMOS) ? 4.97232e-7 : 3.42537e-7;
                  pParam->BSIM4v6Bechvb = (model->BSIM4v6type == NMOS) ? 7.45669e11 : 1.16645e12;
                  pParam->BSIM4v6AechvbEdgeS = pParam->BSIM4v6Aechvb * pParam->BSIM4v6weff
                                          * model->BSIM4v6dlcig * pParam->BSIM4v6ToxRatioEdge;
                  pParam->BSIM4v6AechvbEdgeD = pParam->BSIM4v6Aechvb * pParam->BSIM4v6weff
                                          * model->BSIM4v6dlcigd * pParam->BSIM4v6ToxRatioEdge;
                  pParam->BSIM4v6BechvbEdge = -pParam->BSIM4v6Bechvb
                                          * toxe * pParam->BSIM4v6poxedge;
                  pParam->BSIM4v6Aechvb *= pParam->BSIM4v6weff * pParam->BSIM4v6leff
                                       * pParam->BSIM4v6ToxRatio;
                  pParam->BSIM4v6Bechvb *= -toxe;


                  pParam->BSIM4v6mstar = 0.5 + atan(pParam->BSIM4v6minv) / PI;
                  pParam->BSIM4v6mstarcv = 0.5 + atan(pParam->BSIM4v6minvcv) / PI;
                  pParam->BSIM4v6voffcbn =  pParam->BSIM4v6voff + model->BSIM4v6voffl / pParam->BSIM4v6leff;
                  pParam->BSIM4v6voffcbncv =  pParam->BSIM4v6voffcv + model->BSIM4v6voffcvl / pParam->BSIM4v6leff;

                  pParam->BSIM4v6ldeb = sqrt(epssub * Vtm0 / (Charge_q
                                    * pParam->BSIM4v6ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4v6acde *= pow((pParam->BSIM4v6ndep / 2.0e16), -0.25);


                  if (model->BSIM4v6k1Given || model->BSIM4v6k2Given)
                  {   if (!model->BSIM4v6k1Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM4v6k1 = 0.53;
                      }
                      if (!model->BSIM4v6k2Given)
                      {
                          if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) /* don't print in sensitivity */
                              fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM4v6k2 = -0.0186;
                      }
                      if ((!ckt->CKTcurJob) || (ckt->CKTcurJob->JOBtype < 9)) { /* don't print in sensitivity */
                          if (model->BSIM4v6nsubGiven)
                              fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v6xtGiven)
                              fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v6vbxGiven)
                              fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v6gamma1Given)
                              fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                          if (model->BSIM4v6gamma2Given)
                              fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                      }
                  }
                  else
                  {   if (!model->BSIM4v6vbxGiven)
                          pParam->BSIM4v6vbx = pParam->BSIM4v6phi - 7.7348e-4
                                           * pParam->BSIM4v6ndep
                                           * pParam->BSIM4v6xt * pParam->BSIM4v6xt;
                      if (pParam->BSIM4v6vbx > 0.0)
                          pParam->BSIM4v6vbx = -pParam->BSIM4v6vbx;
                      if (pParam->BSIM4v6vbm > 0.0)
                          pParam->BSIM4v6vbm = -pParam->BSIM4v6vbm;

                      if (!model->BSIM4v6gamma1Given)
                          pParam->BSIM4v6gamma1 = 5.753e-12
                                              * sqrt(pParam->BSIM4v6ndep)
                                              / model->BSIM4v6coxe;
                      if (!model->BSIM4v6gamma2Given)
                          pParam->BSIM4v6gamma2 = 5.753e-12
                                              * sqrt(pParam->BSIM4v6nsub)
                                              / model->BSIM4v6coxe;

                      T0 = pParam->BSIM4v6gamma1 - pParam->BSIM4v6gamma2;
                      T1 = sqrt(pParam->BSIM4v6phi - pParam->BSIM4v6vbx)
                         - pParam->BSIM4v6sqrtPhi;
                      T2 = sqrt(pParam->BSIM4v6phi * (pParam->BSIM4v6phi
                         - pParam->BSIM4v6vbm)) - pParam->BSIM4v6phi;
                      pParam->BSIM4v6k2 = T0 * T1 / (2.0 * T2 + pParam->BSIM4v6vbm);
                      pParam->BSIM4v6k1 = pParam->BSIM4v6gamma2 - 2.0
                                      * pParam->BSIM4v6k2 * sqrt(pParam->BSIM4v6phi
                                      - pParam->BSIM4v6vbm);
                  }

                  if (!model->BSIM4v6vfbGiven)
                  {
                    if (model->BSIM4v6vth0Given)
                      {   pParam->BSIM4v6vfb = model->BSIM4v6type * pParam->BSIM4v6vth0
                                           - pParam->BSIM4v6phi - pParam->BSIM4v6k1
                                           * pParam->BSIM4v6sqrtPhi;
                      }
                      else
                      {
                        if ((model->BSIM4v6mtrlMod) && (model->BSIM4v6phigGiven) &&
                            (model->BSIM4v6nsubGiven))
                          {
                            T0 = Vtm0 * log(pParam->BSIM4v6nsub/ni);
                            T1 = 0.5 * Eg0;
                            if(T0 > T1)
                              T0 = T1;
                            T2 = model->BSIM4v6easub + T1 + model->BSIM4v6type * T0;
                            pParam->BSIM4v6vfb = model->BSIM4v6phig - T2;
                          }
                        else
                          {
                            pParam->BSIM4v6vfb = -1.0;
                          }
                      }
                  }
                   if (!model->BSIM4v6vth0Given)
                  {   pParam->BSIM4v6vth0 = model->BSIM4v6type * (pParam->BSIM4v6vfb
                                        + pParam->BSIM4v6phi + pParam->BSIM4v6k1
                                        * pParam->BSIM4v6sqrtPhi);
                  }

                  pParam->BSIM4v6k1ox = pParam->BSIM4v6k1 * toxe
                                    / model->BSIM4v6toxm;

                  tmp = sqrt(epssub / (epsrox * EPS0) * toxe * pParam->BSIM4v6Xdep0);
                    T0 = pParam->BSIM4v6dsub * pParam->BSIM4v6leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                    {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                            T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      pParam->BSIM4v6theta0vb0 = T1 / T4;
                  }
                  else
                      pParam->BSIM4v6theta0vb0 = 1.0 / (MAX_EXP - 2.0);

                   T0 = pParam->BSIM4v6drout * pParam->BSIM4v6leff / tmp;
                  if (T0 < EXP_THRESHOLD)
                         {   T1 = exp(T0);
                            T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                  }
                  else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                  pParam->BSIM4v6thetaRout = pParam->BSIM4v6pdibl1 * T5
                                         + pParam->BSIM4v6pdibl2;

                  tmp = sqrt(pParam->BSIM4v6Xdep0);
                  tmp1 = pParam->BSIM4v6vbi - pParam->BSIM4v6phi;
                  tmp2 = model->BSIM4v6factor1 * tmp;

                  T0 = pParam->BSIM4v6dvt1w * pParam->BSIM4v6weff
                     * pParam->BSIM4v6leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T8 = T1 / T4;
                  }
                  else
                      T8 = 1.0 / (MAX_EXP - 2.0);
                  T0 = pParam->BSIM4v6dvt0w * T8;
                  T8 = T0 * tmp1;

                  T0 = pParam->BSIM4v6dvt1 * pParam->BSIM4v6leff / tmp2;
                  if (T0 < EXP_THRESHOLD)
                  {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T9 = T1 / T4;
                  }
                  else
                      T9 = 1.0 / (MAX_EXP - 2.0);
                  T9 = pParam->BSIM4v6dvt0 * T9 * tmp1;

                  T4 = toxe * pParam->BSIM4v6phi
                     / (pParam->BSIM4v6weff + pParam->BSIM4v6w0);

                  T0 = sqrt(1.0 + pParam->BSIM4v6lpe0 / pParam->BSIM4v6leff);
                  if((model->BSIM4v6tempMod == 1) || (model->BSIM4v6tempMod == 0))
                          T3 = (pParam->BSIM4v6kt1 + pParam->BSIM4v6kt1l / pParam->BSIM4v6leff)
                                     * (TRatio - 1.0);
                  if((model->BSIM4v6tempMod == 2)||(model->BSIM4v6tempMod == 3))
                        T3 = - pParam->BSIM4v6kt1 * (TRatio - 1.0);

                  T5 = pParam->BSIM4v6k1ox * (T0 - 1.0) * pParam->BSIM4v6sqrtPhi
                     + T3;
                  pParam->BSIM4v6vfbzbfactor = - T8 - T9 + pParam->BSIM4v6k3 * T4 + T5
                                             - pParam->BSIM4v6phi - pParam->BSIM4v6k1 * pParam->BSIM4v6sqrtPhi;

                  /* stress effect */

                      wlod = model->BSIM4v6wlod;
                      if (model->BSIM4v6wlod < 0.0)
                  {   fprintf(stderr, "Warning: WLOD = %g is less than 0. 0.0 is used\n",model->BSIM4v6wlod);
                             wlod = 0.0;
                  }
                  T0 = pow(Lnew, model->BSIM4v6llodku0);
                  W_tmp = Wnew + wlod;
                  T1 = pow(W_tmp, model->BSIM4v6wlodku0);
                  tmp1 = model->BSIM4v6lku0 / T0 + model->BSIM4v6wku0 / T1
                         + model->BSIM4v6pku0 / (T0 * T1);
                  pParam->BSIM4v6ku0 = 1.0 + tmp1;

                  T0 = pow(Lnew, model->BSIM4v6llodvth);
                  T1 = pow(W_tmp, model->BSIM4v6wlodvth);
                  tmp1 = model->BSIM4v6lkvth0 / T0 + model->BSIM4v6wkvth0 / T1
                       + model->BSIM4v6pkvth0 / (T0 * T1);
                  pParam->BSIM4v6kvth0 = 1.0 + tmp1;
                  pParam->BSIM4v6kvth0 = sqrt(pParam->BSIM4v6kvth0*pParam->BSIM4v6kvth0 + DELTA);

                  T0 = (TRatio - 1.0);
                  pParam->BSIM4v6ku0temp = pParam->BSIM4v6ku0 * (1.0 + model->BSIM4v6tku0 *T0) + DELTA;

                  Inv_saref = 1.0/(model->BSIM4v6saref + 0.5*Ldrn);
                  Inv_sbref = 1.0/(model->BSIM4v6sbref + 0.5*Ldrn);
                  pParam->BSIM4v6inv_od_ref = Inv_saref + Inv_sbref;
                  pParam->BSIM4v6rho_ref = model->BSIM4v6ku0 / pParam->BSIM4v6ku0temp * pParam->BSIM4v6inv_od_ref;

              } /* End of SizeNotFound */

              /*  stress effect */
              if( (here->BSIM4v6sa > 0.0) && (here->BSIM4v6sb > 0.0) &&
                        ((here->BSIM4v6nf == 1.0) || ((here->BSIM4v6nf > 1.0) && (here->BSIM4v6sd > 0.0))) )
              {          Inv_sa = 0;
                        Inv_sb = 0;

                         kvsat = model->BSIM4v6kvsat;
                  if (model->BSIM4v6kvsat < -1.0 )
                  {   fprintf(stderr, "Warning: KVSAT = %g is too small; -1.0 is used.\n",model->BSIM4v6kvsat);
                             kvsat = -1.0;
                      }
                      if (model->BSIM4v6kvsat > 1.0)
                      {   fprintf(stderr, "Warning: KVSAT = %g is too big; 1.0 is used.\n",model->BSIM4v6kvsat);
                         kvsat = 1.0;
                      }

                        for(i = 0; i < here->BSIM4v6nf; i++){
                           T0 = 1.0 / here->BSIM4v6nf / (here->BSIM4v6sa + 0.5*Ldrn + i * (here->BSIM4v6sd +Ldrn));
                            T1 = 1.0 / here->BSIM4v6nf / (here->BSIM4v6sb + 0.5*Ldrn + i * (here->BSIM4v6sd +Ldrn));
                           Inv_sa += T0;
                            Inv_sb += T1;
                  }
                  Inv_ODeff = Inv_sa + Inv_sb;
                  rho = model->BSIM4v6ku0 / pParam->BSIM4v6ku0temp * Inv_ODeff;
                  T0 = (1.0 + rho)/(1.0 + pParam->BSIM4v6rho_ref);
                  here->BSIM4v6u0temp = pParam->BSIM4v6u0temp * T0;

                  T1 = (1.0 + kvsat * rho)/(1.0 + kvsat * pParam->BSIM4v6rho_ref);
                  here->BSIM4v6vsattemp = pParam->BSIM4v6vsattemp * T1;

                  OD_offset = Inv_ODeff - pParam->BSIM4v6inv_od_ref;
                  dvth0_lod = model->BSIM4v6kvth0 / pParam->BSIM4v6kvth0 * OD_offset;
                  dk2_lod = model->BSIM4v6stk2 / pow(pParam->BSIM4v6kvth0, model->BSIM4v6lodk2) *
                                   OD_offset;
                  deta0_lod = model->BSIM4v6steta0 / pow(pParam->BSIM4v6kvth0, model->BSIM4v6lodeta0) *
                                     OD_offset;
                  here->BSIM4v6vth0 = pParam->BSIM4v6vth0 + dvth0_lod;

                  here->BSIM4v6eta0 = pParam->BSIM4v6eta0 + deta0_lod;
                  here->BSIM4v6k2 = pParam->BSIM4v6k2 + dk2_lod;
               } else {
                      here->BSIM4v6u0temp = pParam->BSIM4v6u0temp;
                      here->BSIM4v6vth0 = pParam->BSIM4v6vth0;
                      here->BSIM4v6vsattemp = pParam->BSIM4v6vsattemp;
                      here->BSIM4v6eta0 = pParam->BSIM4v6eta0;
                      here->BSIM4v6k2 = pParam->BSIM4v6k2;
              }

              /*  Well Proximity Effect  */
              if (model->BSIM4v6wpemod)
              { if( (!here->BSIM4v6scaGiven) && (!here->BSIM4v6scbGiven) && (!here->BSIM4v6sccGiven) )
                {   if((here->BSIM4v6scGiven) && (here->BSIM4v6sc > 0.0) )
                          {   T1 = here->BSIM4v6sc + Wdrn;
                        T2 = 1.0 / model->BSIM4v6scref;
                        here->BSIM4v6sca = model->BSIM4v6scref * model->BSIM4v6scref
                                        / (here->BSIM4v6sc * T1);
                        here->BSIM4v6scb = ( (0.1 * here->BSIM4v6sc + 0.01 * model->BSIM4v6scref)
                                        * exp(-10.0 * here->BSIM4v6sc * T2)
                                        - (0.1 * T1 + 0.01 * model->BSIM4v6scref)
                                        * exp(-10.0 * T1 * T2) ) / Wdrn;
                        here->BSIM4v6scc = ( (0.05 * here->BSIM4v6sc + 0.0025 * model->BSIM4v6scref)
                                        * exp(-20.0 * here->BSIM4v6sc * T2)
                                        - (0.05 * T1 + 0.0025 * model->BSIM4v6scref)
                                        * exp(-20.0 * T1 * T2) ) / Wdrn;
                    } else {
                        fprintf(stderr, "Warning: No WPE as none of SCA, SCB, SCC, SC is given and/or SC not positive.\n");
                    }
                }

                       if (here->BSIM4v6sca < 0.0)
                {
                    printf("Warning: SCA = %g is negative. Set to 0.0.\n", here->BSIM4v6sca);
                    here->BSIM4v6sca = 0.0;
                }
                if (here->BSIM4v6scb < 0.0)
                {
                    printf("Warning: SCB = %g is negative. Set to 0.0.\n", here->BSIM4v6scb);
                    here->BSIM4v6scb = 0.0;
                }
                if (here->BSIM4v6scc < 0.0)
                {
                    printf("Warning: SCC = %g is negative. Set to 0.0.\n", here->BSIM4v6scc);
                    here->BSIM4v6scc = 0.0;
                }
                if (here->BSIM4v6sc < 0.0)
                {
                    printf("Warning: SC = %g is negative. Set to 0.0.\n", here->BSIM4v6sc);
                    here->BSIM4v6sc = 0.0;
                }
                                /*4.6.2*/
                sceff = here->BSIM4v6sca + model->BSIM4v6web * here->BSIM4v6scb
                      + model->BSIM4v6wec * here->BSIM4v6scc;
                here->BSIM4v6vth0 += pParam->BSIM4v6kvth0we * sceff;
                here->BSIM4v6k2 +=  pParam->BSIM4v6k2we * sceff;
                  T3 =  1.0 + pParam->BSIM4v6ku0we * sceff;
                if (T3 <= 0.0)
                {         T3 = 0.0;
                        fprintf(stderr, "Warning: ku0we = %g is negatively too high. Negative mobility! \n", pParam->BSIM4v6ku0we);
                }
                here->BSIM4v6u0temp *= T3;
              }

            /* adding delvto  */
            here->BSIM4v6vth0 += here->BSIM4v6delvto;
            here->BSIM4v6vfb = pParam->BSIM4v6vfb + model->BSIM4v6type * here->BSIM4v6delvto;

            /* low field mobility multiplier */
            here->BSIM4v6u0temp = pParam->BSIM4v6u0temp * here->BSIM4v6mulu0;

            /* Instance variables calculation  */
            T3 = model->BSIM4v6type * here->BSIM4v6vth0
               - here->BSIM4v6vfb - pParam->BSIM4v6phi;
            T4 = T3 + T3;
            T5 = 2.5 * T3;
            here->BSIM4v6vtfbphi1 = (model->BSIM4v6type == NMOS) ? T4 : T5;
            if (here->BSIM4v6vtfbphi1 < 0.0)
                here->BSIM4v6vtfbphi1 = 0.0;

            here->BSIM4v6vtfbphi2 = 4.0 * T3;
            if (here->BSIM4v6vtfbphi2 < 0.0)
                here->BSIM4v6vtfbphi2 = 0.0;

            if (here->BSIM4v6k2 < 0.0)
            {   T0 = 0.5 * pParam->BSIM4v6k1 / here->BSIM4v6k2;
                here->BSIM4v6vbsc = 0.9 * (pParam->BSIM4v6phi - T0 * T0);
                if (here->BSIM4v6vbsc > -3.0)
                    here->BSIM4v6vbsc = -3.0;
                else if (here->BSIM4v6vbsc < -30.0)
                    here->BSIM4v6vbsc = -30.0;
            }
            else
                here->BSIM4v6vbsc = -30.0;
            if (here->BSIM4v6vbsc > pParam->BSIM4v6vbm)
                here->BSIM4v6vbsc = pParam->BSIM4v6vbm;
            here->BSIM4v6k2ox = here->BSIM4v6k2 * toxe
                              / model->BSIM4v6toxm;

            here->BSIM4v6vfbzb = pParam->BSIM4v6vfbzbfactor
                                +  model->BSIM4v6type * here->BSIM4v6vth0 ;

              here->BSIM4v6cgso = pParam->BSIM4v6cgso;
              here->BSIM4v6cgdo = pParam->BSIM4v6cgdo;

              lnl = log(pParam->BSIM4v6leff * 1.0e6);
              lnw = log(pParam->BSIM4v6weff * 1.0e6);
              lnnf = log(here->BSIM4v6nf);

              bodymode = 5;
              if( ( !model->BSIM4v6rbps0Given) ||
                  ( !model->BSIM4v6rbpd0Given) )
                bodymode = 1;
              else
                if( (!model->BSIM4v6rbsbx0Given && !model->BSIM4v6rbsby0Given) ||
                      (!model->BSIM4v6rbdbx0Given && !model->BSIM4v6rbdby0Given) )
                  bodymode = 3;

              if(here->BSIM4v6rbodyMod == 2)
                {
                  if (bodymode == 5)
                    {
                      rbsbx =  exp( log(model->BSIM4v6rbsbx0) + model->BSIM4v6rbsdbxl * lnl +
                                    model->BSIM4v6rbsdbxw * lnw + model->BSIM4v6rbsdbxnf * lnnf );
                      rbsby =  exp( log(model->BSIM4v6rbsby0) + model->BSIM4v6rbsdbyl * lnl +
                                    model->BSIM4v6rbsdbyw * lnw + model->BSIM4v6rbsdbynf * lnnf );
                      here->BSIM4v6rbsb = rbsbx * rbsby / (rbsbx + rbsby);


                      rbdbx =  exp( log(model->BSIM4v6rbdbx0) + model->BSIM4v6rbsdbxl * lnl +
                                    model->BSIM4v6rbsdbxw * lnw + model->BSIM4v6rbsdbxnf * lnnf );
                      rbdby =  exp( log(model->BSIM4v6rbdby0) + model->BSIM4v6rbsdbyl * lnl +
                                    model->BSIM4v6rbsdbyw * lnw + model->BSIM4v6rbsdbynf * lnnf );
                      here->BSIM4v6rbdb = rbdbx * rbdby / (rbdbx + rbdby);
                    }

                  if ((bodymode == 3)|| (bodymode == 5))
                    {
                      here->BSIM4v6rbps = exp( log(model->BSIM4v6rbps0) + model->BSIM4v6rbpsl * lnl +
                                             model->BSIM4v6rbpsw * lnw + model->BSIM4v6rbpsnf * lnnf );
                      here->BSIM4v6rbpd = exp( log(model->BSIM4v6rbpd0) + model->BSIM4v6rbpdl * lnl +
                                             model->BSIM4v6rbpdw * lnw + model->BSIM4v6rbpdnf * lnnf );
                    }

                  rbpbx =  exp( log(model->BSIM4v6rbpbx0) + model->BSIM4v6rbpbxl * lnl +
                                model->BSIM4v6rbpbxw * lnw + model->BSIM4v6rbpbxnf * lnnf );
                  rbpby =  exp( log(model->BSIM4v6rbpby0) + model->BSIM4v6rbpbyl * lnl +
                                model->BSIM4v6rbpbyw * lnw + model->BSIM4v6rbpbynf * lnnf );
                  here->BSIM4v6rbpb = rbpbx*rbpby/(rbpbx + rbpby);
                }


              if ((here->BSIM4v6rbodyMod == 1 ) || ((here->BSIM4v6rbodyMod == 2 ) && (bodymode == 5)) )
              {   if (here->BSIM4v6rbdb < 1.0e-3)
                      here->BSIM4v6grbdb = 1.0e3; /* in mho */
                  else
                      here->BSIM4v6grbdb = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbdb;
                  if (here->BSIM4v6rbpb < 1.0e-3)
                      here->BSIM4v6grbpb = 1.0e3;
                  else
                      here->BSIM4v6grbpb = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbpb;
                  if (here->BSIM4v6rbps < 1.0e-3)
                      here->BSIM4v6grbps = 1.0e3;
                  else
                      here->BSIM4v6grbps = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbps;
                  if (here->BSIM4v6rbsb < 1.0e-3)
                      here->BSIM4v6grbsb = 1.0e3;
                  else
                      here->BSIM4v6grbsb = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbsb;
                  if (here->BSIM4v6rbpd < 1.0e-3)
                      here->BSIM4v6grbpd = 1.0e3;
                  else
                      here->BSIM4v6grbpd = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbpd;

              }

              if((here->BSIM4v6rbodyMod == 2) && (bodymode == 3))
              {
                      here->BSIM4v6grbdb = here->BSIM4v6grbsb = model->BSIM4v6gbmin;
                  if (here->BSIM4v6rbpb < 1.0e-3)
                      here->BSIM4v6grbpb = 1.0e3;
                  else
                      here->BSIM4v6grbpb = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbpb;
                  if (here->BSIM4v6rbps < 1.0e-3)
                      here->BSIM4v6grbps = 1.0e3;
                  else
                      here->BSIM4v6grbps = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbps;
                  if (here->BSIM4v6rbpd < 1.0e-3)
                      here->BSIM4v6grbpd = 1.0e3;
                  else
                      here->BSIM4v6grbpd = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbpd;
              }

              if((here->BSIM4v6rbodyMod == 2) && (bodymode == 1))
              {
                      here->BSIM4v6grbdb = here->BSIM4v6grbsb = model->BSIM4v6gbmin;
                      here->BSIM4v6grbps = here->BSIM4v6grbpd = 1.0e3;
                  if (here->BSIM4v6rbpb < 1.0e-3)
                      here->BSIM4v6grbpb = 1.0e3;
                  else
                      here->BSIM4v6grbpb = model->BSIM4v6gbmin + 1.0 / here->BSIM4v6rbpb;
              }


              /*
               * Process geomertry dependent parasitics
               */

              here->BSIM4v6grgeltd = model->BSIM4v6rshg * (here->BSIM4v6xgw
                      + pParam->BSIM4v6weffCJ / 3.0 / here->BSIM4v6ngcon) /
                      (here->BSIM4v6ngcon * here->BSIM4v6nf *
                      (Lnew - model->BSIM4v6xgl));
              if (here->BSIM4v6grgeltd > 0.0)
                  here->BSIM4v6grgeltd = 1.0 / here->BSIM4v6grgeltd;
              else
              {   here->BSIM4v6grgeltd = 1.0e3; /* mho */
                  if (here->BSIM4v6rgateMod != 0)
                  printf("Warning: The gate conductance reset to 1.0e3 mho.\n");
              }

              DMCGeff = model->BSIM4v6dmcg - model->BSIM4v6dmcgt;
              DMCIeff = model->BSIM4v6dmci;
              DMDGeff = model->BSIM4v6dmdg - model->BSIM4v6dmcgt;

              if (here->BSIM4v6sourcePerimeterGiven)
              {
                if(here->BSIM4v6sourcePerimeter == 0.0)
                  here->BSIM4v6Pseff = 0.0;
                else if (here->BSIM4v6sourcePerimeter < 0.0)
                {
                  printf("Warning: Source Perimeter is specified as negative, it is set to zero.\n");
                  here->BSIM4v6Pseff = 0.0;
                }
                else
                {
                  if (model->BSIM4v6perMod == 0)
                    here->BSIM4v6Pseff = here->BSIM4v6sourcePerimeter;
                  else
                    here->BSIM4v6Pseff = here->BSIM4v6sourcePerimeter
                                       - pParam->BSIM4v6weffCJ * here->BSIM4v6nf;
                }
              }
              else
                  BSIM4v6PAeffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod, here->BSIM4v6min,
                                pParam->BSIM4v6weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                &(here->BSIM4v6Pseff), &dumPd, &dumAs, &dumAd);
              if (here->BSIM4v6Pseff < 0.0) /*4.6.2*/
                      here->BSIM4v6Pseff = 0.0;

              if (here->BSIM4v6drainPerimeterGiven)
              {
                if(here->BSIM4v6drainPerimeter == 0.0)
                  here->BSIM4v6Pdeff = 0.0;
                else if (here->BSIM4v6drainPerimeter < 0.0)
                {
                  printf("Warning: Drain Perimeter is specified as negative, it is set to zero\n");
                  here->BSIM4v6Pdeff = 0.0;
                }
                else
                {
                  if (model->BSIM4v6perMod == 0)
                    here->BSIM4v6Pdeff = here->BSIM4v6drainPerimeter;
                  else
                    here->BSIM4v6Pdeff = here->BSIM4v6drainPerimeter
                                       - pParam->BSIM4v6weffCJ * here->BSIM4v6nf;
                }
              }
              else
                  BSIM4v6PAeffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod, here->BSIM4v6min,
                                pParam->BSIM4v6weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                &dumPs, &(here->BSIM4v6Pdeff), &dumAs, &dumAd);
              if (here->BSIM4v6Pdeff < 0.0) /*4.6.2*/
                      here->BSIM4v6Pdeff = 0.0;

              if (here->BSIM4v6sourceAreaGiven)
                  here->BSIM4v6Aseff = here->BSIM4v6sourceArea;
              else
                  BSIM4v6PAeffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod, here->BSIM4v6min,
                                    pParam->BSIM4v6weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &(here->BSIM4v6Aseff), &dumAd);

              if (here->BSIM4v6drainAreaGiven)
                  here->BSIM4v6Adeff = here->BSIM4v6drainArea;
              else
                  BSIM4v6PAeffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod, here->BSIM4v6min,
                                    pParam->BSIM4v6weffCJ, DMCGeff, DMCIeff, DMDGeff,
                                    &dumPs, &dumPd, &dumAs, &(here->BSIM4v6Adeff));

              /* Processing S/D resistance and conductance below */
              if(here->BSIM4v6sNodePrime != here->BSIM4v6sNode)
              {
                 here->BSIM4v6sourceConductance = 0.0;
                 if(here->BSIM4v6sourceSquaresGiven)
                 {
                    here->BSIM4v6sourceConductance = model->BSIM4v6sheetResistance
                                               * here->BSIM4v6sourceSquares;
                 } else if (here->BSIM4v6rgeoMod > 0)
                 {
                    BSIM4v6RdseffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod,
                      here->BSIM4v6rgeoMod, here->BSIM4v6min,
                      pParam->BSIM4v6weffCJ, model->BSIM4v6sheetResistance,
                      DMCGeff, DMCIeff, DMDGeff, 1, &(here->BSIM4v6sourceConductance));
                 } else
                 {
                    here->BSIM4v6sourceConductance = 0.0;
                 }

                 if (here->BSIM4v6sourceConductance > 0.0)
                     here->BSIM4v6sourceConductance = 1.0
                                            / here->BSIM4v6sourceConductance;
                 else
                 {
                     here->BSIM4v6sourceConductance = 1.0e3; /* mho */
                     printf ("Warning: Source conductance reset to 1.0e3 mho.\n");
                 }
              } else
              {
                  here->BSIM4v6sourceConductance = 0.0;
              }

              if(here->BSIM4v6dNodePrime != here->BSIM4v6dNode)
              {
                 here->BSIM4v6drainConductance = 0.0;
                 if(here->BSIM4v6drainSquaresGiven)
                 {
                    here->BSIM4v6drainConductance = model->BSIM4v6sheetResistance
                                              * here->BSIM4v6drainSquares;
                 } else if (here->BSIM4v6rgeoMod > 0)
                 {
                    BSIM4v6RdseffGeo(here->BSIM4v6nf, here->BSIM4v6geoMod,
                      here->BSIM4v6rgeoMod, here->BSIM4v6min,
                      pParam->BSIM4v6weffCJ, model->BSIM4v6sheetResistance,
                      DMCGeff, DMCIeff, DMDGeff, 0, &(here->BSIM4v6drainConductance));
                 } else
                 {
                    here->BSIM4v6drainConductance = 0.0;
                 }

                 if (here->BSIM4v6drainConductance > 0.0)
                     here->BSIM4v6drainConductance = 1.0
                                           / here->BSIM4v6drainConductance;
                 else
                 {
                     here->BSIM4v6drainConductance = 1.0e3; /* mho */
                     printf ("Warning: Drain conductance reset to 1.0e3 mho.\n");
                  }
              } else
              {
                  here->BSIM4v6drainConductance = 0.0;
              }

               /* End of Rsd processing */


              Nvtms = model->BSIM4v6vtm * model->BSIM4v6SjctEmissionCoeff;
              if ((here->BSIM4v6Aseff <= 0.0) && (here->BSIM4v6Pseff <= 0.0))
              {   SourceSatCurrent = 0.0;
              }
              else
              {   SourceSatCurrent = here->BSIM4v6Aseff * model->BSIM4v6SjctTempSatCurDensity
                                   + here->BSIM4v6Pseff * model->BSIM4v6SjctSidewallTempSatCurDensity
                                   + pParam->BSIM4v6weffCJ * here->BSIM4v6nf
                                   * model->BSIM4v6SjctGateSidewallTempSatCurDensity;
              }
              if (SourceSatCurrent > 0.0)
              {   switch(model->BSIM4v6dioMod)
                  {   case 0:
                          if ((model->BSIM4v6bvs / Nvtms) > EXP_THRESHOLD)
                              here->BSIM4v6XExpBVS = model->BSIM4v6xjbvs * MIN_EXP;
                          else
                              here->BSIM4v6XExpBVS = model->BSIM4v6xjbvs * exp(-model->BSIM4v6bvs / Nvtms);
                          break;
                      case 1:
                          BSIM4v6DioIjthVjmEval(Nvtms, model->BSIM4v6ijthsfwd, SourceSatCurrent,
                                              0.0, &(here->BSIM4v6vjsmFwd));
                          here->BSIM4v6IVjsmFwd = SourceSatCurrent * exp(here->BSIM4v6vjsmFwd / Nvtms);
                          break;
                      case 2:
                          if ((model->BSIM4v6bvs / Nvtms) > EXP_THRESHOLD)
                          {   here->BSIM4v6XExpBVS = model->BSIM4v6xjbvs * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v6XExpBVS = exp(-model->BSIM4v6bvs / Nvtms);
                              tmp = here->BSIM4v6XExpBVS;
                              here->BSIM4v6XExpBVS *= model->BSIM4v6xjbvs;
                          }

                          BSIM4v6DioIjthVjmEval(Nvtms, model->BSIM4v6ijthsfwd, SourceSatCurrent,
                                                     here->BSIM4v6XExpBVS, &(here->BSIM4v6vjsmFwd));
                          T0 = exp(here->BSIM4v6vjsmFwd / Nvtms);
                          here->BSIM4v6IVjsmFwd = SourceSatCurrent * (T0 - here->BSIM4v6XExpBVS / T0
                                                + here->BSIM4v6XExpBVS - 1.0);
                          here->BSIM4v6SslpFwd = SourceSatCurrent
                                               * (T0 + here->BSIM4v6XExpBVS / T0) / Nvtms;

                          T2 = model->BSIM4v6ijthsrev / SourceSatCurrent;
                          if (T2 < 1.0)
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthsrev too small and set to 10 times IsbSat.\n");
                          }
                          here->BSIM4v6vjsmRev = -model->BSIM4v6bvs
                                             - Nvtms * log((T2 - 1.0) / model->BSIM4v6xjbvs);
                          T1 = model->BSIM4v6xjbvs * exp(-(model->BSIM4v6bvs
                             + here->BSIM4v6vjsmRev) / Nvtms);
                          here->BSIM4v6IVjsmRev = SourceSatCurrent * (1.0 + T1);
                          here->BSIM4v6SslpRev = -SourceSatCurrent * T1 / Nvtms;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v6dioMod);
                  }
              }

              Nvtmd = model->BSIM4v6vtm * model->BSIM4v6DjctEmissionCoeff;
              if ((here->BSIM4v6Adeff <= 0.0) && (here->BSIM4v6Pdeff <= 0.0))
              {   DrainSatCurrent = 0.0;
              }
              else
              {   DrainSatCurrent = here->BSIM4v6Adeff * model->BSIM4v6DjctTempSatCurDensity
                                  + here->BSIM4v6Pdeff * model->BSIM4v6DjctSidewallTempSatCurDensity
                                  + pParam->BSIM4v6weffCJ * here->BSIM4v6nf
                                  * model->BSIM4v6DjctGateSidewallTempSatCurDensity;
              }
              if (DrainSatCurrent > 0.0)
              {   switch(model->BSIM4v6dioMod)
                  {   case 0:
                          if ((model->BSIM4v6bvd / Nvtmd) > EXP_THRESHOLD)
                          here->BSIM4v6XExpBVD = model->BSIM4v6xjbvd * MIN_EXP;
                          else
                          here->BSIM4v6XExpBVD = model->BSIM4v6xjbvd * exp(-model->BSIM4v6bvd / Nvtmd);
                          break;
                      case 1:
                          BSIM4v6DioIjthVjmEval(Nvtmd, model->BSIM4v6ijthdfwd, DrainSatCurrent,
                                              0.0, &(here->BSIM4v6vjdmFwd));
                          here->BSIM4v6IVjdmFwd = DrainSatCurrent * exp(here->BSIM4v6vjdmFwd / Nvtmd);
                          break;
                      case 2:
                          if ((model->BSIM4v6bvd / Nvtmd) > EXP_THRESHOLD)
                          {   here->BSIM4v6XExpBVD = model->BSIM4v6xjbvd * MIN_EXP;
                              tmp = MIN_EXP;
                          }
                          else
                          {   here->BSIM4v6XExpBVD = exp(-model->BSIM4v6bvd / Nvtmd);
                              tmp = here->BSIM4v6XExpBVD;
                              here->BSIM4v6XExpBVD *= model->BSIM4v6xjbvd;
                          }

                          BSIM4v6DioIjthVjmEval(Nvtmd, model->BSIM4v6ijthdfwd, DrainSatCurrent,
                                              here->BSIM4v6XExpBVD, &(here->BSIM4v6vjdmFwd));
                          T0 = exp(here->BSIM4v6vjdmFwd / Nvtmd);
                          here->BSIM4v6IVjdmFwd = DrainSatCurrent * (T0 - here->BSIM4v6XExpBVD / T0
                                              + here->BSIM4v6XExpBVD - 1.0);
                          here->BSIM4v6DslpFwd = DrainSatCurrent
                                               * (T0 + here->BSIM4v6XExpBVD / T0) / Nvtmd;

                          T2 = model->BSIM4v6ijthdrev / DrainSatCurrent;
                          if (T2 < 1.0)
                          {   T2 = 10.0;
                              fprintf(stderr, "Warning: ijthdrev too small and set to 10 times IdbSat.\n");
                          }
                          here->BSIM4v6vjdmRev = -model->BSIM4v6bvd
                                             - Nvtmd * log((T2 - 1.0) / model->BSIM4v6xjbvd); /* bugfix */
                          T1 = model->BSIM4v6xjbvd * exp(-(model->BSIM4v6bvd
                             + here->BSIM4v6vjdmRev) / Nvtmd);
                          here->BSIM4v6IVjdmRev = DrainSatCurrent * (1.0 + T1);
                          here->BSIM4v6DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                          break;
                  default:
                          printf("Specified dioMod = %d not matched\n", model->BSIM4v6dioMod);
                  }
              }

                /* GEDL current reverse bias */
                T0 = (TRatio - 1.0);
                model->BSIM4v6njtsstemp = model->BSIM4v6njts * (1.0 + model->BSIM4v6tnjts * T0);
                model->BSIM4v6njtsswstemp = model->BSIM4v6njtssw * (1.0 + model->BSIM4v6tnjtssw * T0);
                model->BSIM4v6njtsswgstemp = model->BSIM4v6njtsswg * (1.0 + model->BSIM4v6tnjtsswg * T0);
                model->BSIM4v6njtsdtemp = model->BSIM4v6njtsd * (1.0 + model->BSIM4v6tnjtsd * T0);
                model->BSIM4v6njtsswdtemp = model->BSIM4v6njtsswd * (1.0 + model->BSIM4v6tnjtsswd * T0);
                model->BSIM4v6njtsswgdtemp = model->BSIM4v6njtsswgd * (1.0 + model->BSIM4v6tnjtsswgd * T0);
                T7 = Eg0 / model->BSIM4v6vtm * T0;
                T9 = model->BSIM4v6xtss * T7;
                DEXP(T9, T1);
                T9 = model->BSIM4v6xtsd * T7;
                DEXP(T9, T2);
                T9 = model->BSIM4v6xtssws * T7;
                DEXP(T9, T3);
                T9 = model->BSIM4v6xtsswd * T7;
                DEXP(T9, T4);
                T9 = model->BSIM4v6xtsswgs * T7;
                DEXP(T9, T5);
                T9 = model->BSIM4v6xtsswgd * T7;
                DEXP(T9, T6);
                                /*IBM TAT*/
                                if(model->BSIM4v6jtweff < 0.0)
                              {   model->BSIM4v6jtweff = 0.0;
                          fprintf(stderr, "TAT width dependence effect is negative. Jtweff is clamped to zero.\n");
                      }
                                T11 = sqrt(model->BSIM4v6jtweff / pParam->BSIM4v6weffCJ) + 1.0;

                T10 = pParam->BSIM4v6weffCJ * here->BSIM4v6nf;
                here->BSIM4v6SjctTempRevSatCur = T1 * here->BSIM4v6Aseff * model->BSIM4v6jtss;
                here->BSIM4v6DjctTempRevSatCur = T2 * here->BSIM4v6Adeff * model->BSIM4v6jtsd;
                here->BSIM4v6SswTempRevSatCur = T3 * here->BSIM4v6Pseff * model->BSIM4v6jtssws;
                here->BSIM4v6DswTempRevSatCur = T4 * here->BSIM4v6Pdeff * model->BSIM4v6jtsswd;
                here->BSIM4v6SswgTempRevSatCur = T5 * T10 * T11 * model->BSIM4v6jtsswgs;
                here->BSIM4v6DswgTempRevSatCur = T6 * T10 * T11 * model->BSIM4v6jtsswgd;

                /*high k*/
                /*Calculate VgsteffVth for mobMod=3*/
                if(model->BSIM4v6mobMod==3)
                {        /*Calculate n @ Vbs=Vds=0*/
            V0 = pParam->BSIM4v6vbi - pParam->BSIM4v6phi;
                    lt1 = model->BSIM4v6factor1* pParam->BSIM4v6sqrtXdep0;
                    ltw = lt1;
                    T0 = pParam->BSIM4v6dvt1 * pParam->BSIM4v6leff / lt1;
                    if (T0 < EXP_THRESHOLD)
                      {
                        T1 = exp(T0);
                        T2 = T1 - 1.0;
                        T3 = T2 * T2;
                        T4 = T3 + 2.0 * T1 * MIN_EXP;
                        Theta0 = T1 / T4;
                      }
                    else
                      Theta0 = 1.0 / (MAX_EXP - 2.0);

                     tmp1 = epssub / pParam->BSIM4v6Xdep0;
                    here->BSIM4v6nstar = model->BSIM4v6vtm / Charge_q *
                      (model->BSIM4v6coxe        + tmp1 + pParam->BSIM4v6cit);
                    tmp2 = pParam->BSIM4v6nfactor * tmp1;
                    tmp3 = (tmp2 + pParam->BSIM4v6cdsc * Theta0 + pParam->BSIM4v6cit) / model->BSIM4v6coxe;
                    if (tmp3 >= -0.5)
                      n0 = 1.0 + tmp3;
                    else
                      {
                        T0 = 1.0 / (3.0 + 8.0 * tmp3);
                        n0 = (1.0 + 3.0 * tmp3) * T0;
                      }

                 T0 = n0 * model->BSIM4v6vtm;
             T1 = pParam->BSIM4v6voffcbn;
             T2 = T1/T0;
                   if (T2 < -EXP_THRESHOLD)
          {   T3 = model->BSIM4v6coxe * MIN_EXP / pParam->BSIM4v6cdep0;
              T4 = pParam->BSIM4v6mstar + T3 * n0;
          }
          else if (T2 > EXP_THRESHOLD)
          {   T3 = model->BSIM4v6coxe * MAX_EXP / pParam->BSIM4v6cdep0;
              T4 = pParam->BSIM4v6mstar + T3 * n0;
          }
          else
          {  T3 = exp(T2)* model->BSIM4v6coxe / pParam->BSIM4v6cdep0;
                       T4 = pParam->BSIM4v6mstar + T3 * n0;

          }
                  pParam->BSIM4v6VgsteffVth = T0 * log(2.0)/T4;

                }

                if(model->BSIM4v6mtrlMod)
                  {
                    /* Calculate TOXP from EOT */
                    /* Calculate Vgs_eff @ Vgs = VDD with Poly Depletion Effect */
            Vtm0eot = KboQ * model->BSIM4v6tempeot;
                        Vtmeot  = Vtm0eot;
                        vbieot = Vtm0eot * log(pParam->BSIM4v6nsd
                                   * pParam->BSIM4v6ndep / (ni * ni));
                    phieot = Vtm0eot * log(pParam->BSIM4v6ndep / ni)
                                   + pParam->BSIM4v6phin + 0.4;
                    tmp2 = here->BSIM4v6vfb + phieot;
                    vddeot = model->BSIM4v6type * model->BSIM4v6vddeot;
                    T0 = model->BSIM4v6epsrgate * EPS0;
                    if ((pParam->BSIM4v6ngate > 1.0e18) && (pParam->BSIM4v6ngate < 1.0e25)
                        && (vddeot > tmp2) && (T0!=0))
                      {
                        T1 = 1.0e6 * CHARGE * T0 * pParam->BSIM4v6ngate /
                          (model->BSIM4v6coxe * model->BSIM4v6coxe);
                        T8 = vddeot - tmp2;
                        T4 = sqrt(1.0 + 2.0 * T8 / T1);
                        T2 = 2.0 * T8 / (T4 + 1.0);
                        T3 = 0.5 * T2 * T2 / T1;
                        T7 = 1.12 - T3 - 0.05;
                        T6 = sqrt(T7 * T7 + 0.224);
                        T5 = 1.12 - 0.5 * (T7 + T6);
                        Vgs_eff = vddeot - T5;
                      }
                    else
                      Vgs_eff = vddeot;

                    /* Calculate Vth @ Vds=Vbs=0 */

                    V0 = vbieot - phieot;
                    lt1 = model->BSIM4v6factor1* pParam->BSIM4v6sqrtXdep0;
                    ltw = lt1;
                    T0 = pParam->BSIM4v6dvt1 * model->BSIM4v6leffeot / lt1;
                    if (T0 < EXP_THRESHOLD)
                      {
                        T1 = exp(T0);
                        T2 = T1 - 1.0;
                        T3 = T2 * T2;
                        T4 = T3 + 2.0 * T1 * MIN_EXP;
                        Theta0 = T1 / T4;
                      }
                    else
                      Theta0 = 1.0 / (MAX_EXP - 2.0);
                    Delt_vth = pParam->BSIM4v6dvt0 * Theta0 * V0;
                    T0 = pParam->BSIM4v6dvt1w * model->BSIM4v6weffeot * model->BSIM4v6leffeot / ltw;
                    if (T0 < EXP_THRESHOLD)
                      {   T1 = exp(T0);
                      T2 = T1 - 1.0;
                      T3 = T2 * T2;
                      T4 = T3 + 2.0 * T1 * MIN_EXP;
                      T5 = T1 / T4;
                      }
                    else
                      T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
                    T2 = pParam->BSIM4v6dvt0w * T5 * V0;
                    TempRatioeot =  model->BSIM4v6tempeot / model->BSIM4v6tnom - 1.0;
                    T0 = sqrt(1.0 + pParam->BSIM4v6lpe0 / model->BSIM4v6leffeot);
                    T1 = pParam->BSIM4v6k1ox * (T0 - 1.0) * sqrt(phieot)
                      + (pParam->BSIM4v6kt1 + pParam->BSIM4v6kt1l / model->BSIM4v6leffeot) * TempRatioeot;
                    Vth_NarrowW = toxe * phieot
                      / (model->BSIM4v6weffeot + pParam->BSIM4v6w0);
                    Lpe_Vb = sqrt(1.0 + pParam->BSIM4v6lpeb / model->BSIM4v6leffeot);
                    Vth = model->BSIM4v6type * here->BSIM4v6vth0 +
                      (pParam->BSIM4v6k1ox - pParam->BSIM4v6k1)*sqrt(phieot)*Lpe_Vb
                      - Delt_vth - T2 + pParam->BSIM4v6k3 * Vth_NarrowW + T1;

                    /* Calculate n */
                    tmp1 = epssub / pParam->BSIM4v6Xdep0;
                    here->BSIM4v6nstar = Vtmeot / Charge_q *
                      (model->BSIM4v6coxe        + tmp1 + pParam->BSIM4v6cit);
                    tmp2 = pParam->BSIM4v6nfactor * tmp1;
                    tmp3 = (tmp2 + pParam->BSIM4v6cdsc * Theta0 + pParam->BSIM4v6cit) / model->BSIM4v6coxe;
                    if (tmp3 >= -0.5)
                      n = 1.0 + tmp3;
                    else
                      {
                        T0 = 1.0 / (3.0 + 8.0 * tmp3);
                        n = (1.0 + 3.0 * tmp3) * T0;
                      }

                    /* Vth correction for Pocket implant */
                    if (pParam->BSIM4v6dvtp0 > 0.0)
                      {
                        T3 = model->BSIM4v6leffeot + pParam->BSIM4v6dvtp0 * 2.0;
                        if (model->BSIM4v6tempMod < 2)
                          T4 = Vtmeot * log(model->BSIM4v6leffeot / T3);
                        else
                          T4 = Vtm0eot * log(model->BSIM4v6leffeot / T3);
                        Vth -= n * T4;
                      }
                    Vgsteff = Vgs_eff-Vth;
                    /* calculating Toxp */
                        T3 = model->BSIM4v6type * here->BSIM4v6vth0
               - here->BSIM4v6vfb - phieot;
            T4 = T3 + T3;
            T5 = 2.5 * T3;

            vtfbphi2eot = 4.0 * T3;
            if (vtfbphi2eot < 0.0)
                vtfbphi2eot = 0.0;


                    niter = 0;
                    toxpf = toxe;
                    do
                      {
                        toxpi = toxpf;
                        tmp2 = 2.0e8 * toxpf;
                        T0 = (Vgsteff + vtfbphi2eot) / tmp2;
                        T1 = 1.0 + exp(model->BSIM4v6bdos * 0.7 * log(T0));
                        Tcen = model->BSIM4v6ados * 1.9e-9 / T1;
                        toxpf = toxe - epsrox/model->BSIM4v6epsrsub * Tcen;
                        niter++;
                      } while ((niter<=4)&&(ABS(toxpf-toxpi)>1e-12));
                      model->BSIM4v6toxp = toxpf;
                      model->BSIM4v6coxp = epsrox * EPS0 / model->BSIM4v6toxp;
                      }

              if (BSIM4v6checkModel(model, here, ckt))
              {
                  SPfrontEnd->IFerrorf(ERR_FATAL,
                      "detected during BSIM4.6.5 parameter checking for \n    model %s of device instance %s\n", model->BSIM4v6modName, here->BSIM4v6name);
                  return(E_BADPARM);
              }
         } /* End instance */
    }
    return(OK);
}
