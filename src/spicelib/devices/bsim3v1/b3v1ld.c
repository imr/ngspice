/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3ld.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Mansun Chan  (1995)
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v1v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define Charge_q 1.60219e-19
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
#define DELTA_4 0.02


int
BSIM3v1load(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;
double SourceSatCurrent, DrainSatCurrent;
double ag0, qgd, qgs, qgb, von, cbhat, VgstNVt, ExpVgst = 0.0;
double cdrain, cdhat, cdreq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg, ceq, geq;
double czbd, czbdsw, czbdswg, czbs, czbssw, czbsswg, evbd, evbs, arg, sarg;
double delvbd, delvbs, delvds, delvgd, delvgs;
double Vfbeff, dVfbeff_dVg, dVfbeff_dVd, dVfbeff_dVb, V3, V4;
double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb, gcgdb, gcggb, gcgsb, gcsdb;
double gcsgb, gcssb, PhiB, PhiBSW, MJ, MJSW, PhiBSWG, MJSWG;
double vbd, vbs, vds, vgb, vgd, vgs, vgdo;
#ifndef PREDICTOR
double xfact;
#endif
double qgate = 0.0, qbulk = 0.0, qdrn = 0.0, qsrc, cqgate, cqbulk, cqdrn;
double Vds, Vgs, Vbs, Gmbs, FwdSum, RevSum;
double Vgs_eff, Vfb, dVfb_dVb, dVfb_dVd;
double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth, dVth_dVb, dVth_dVd;
double Vgst, dVgst_dVg, dVgst_dVb, dVgs_eff_dVg, Nvtm;
double n, dn_dVb, Vtm;
double ExpArg, V0;
double Denomi, dDenomi_dVg, dDenomi_dVd, dDenomi_dVb;
double ueff, dueff_dVg, dueff_dVd, dueff_dVb; 
double Esat, Vdsat;
double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb;
double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, Vasat, dAlphaz_dVg, dAlphaz_dVb; 
double dVasat_dVg, dVasat_dVb, dVasat_dVd, Va, dVa_dVd, dVa_dVg, dVa_dVb; 
double Vbseff, dVbseff_dVb, VbseffCV, dVbseffCV_dVb; 
double Arg1, One_Third_CoxWL, Two_Third_CoxWL, Alphaz, CoxWL; 
double T0, dT0_dVg, dT0_dVd, dT0_dVb;
double T1, dT1_dVg, dT1_dVd, dT1_dVb;
double T2, dT2_dVg, dT2_dVd, dT2_dVb;
double T3, dT3_dVg, dT3_dVd, dT3_dVb;
double T4;
double T5;
double T6;
double T7;
double T8;
double T9;
double T10;
double T11, T12;
double tmp, Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb;
double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb;
double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb;
double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb, Delt_vth, dDelt_vth_dVb;
double Theta0, dTheta0_dVb;
double TempRatio, tmp1, tmp2, tmp3, tmp4;
double DIBL_Sft, dDIBL_Sft_dVd, Lambda, dLambda_dVg;
#ifndef NOBYPASS
double tempv;
#endif
double a1;

double Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb; 
double Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb; 
double VdseffCV, dVdseffCV_dVg, dVdseffCV_dVd, dVdseffCV_dVb; 
double diffVds;
double dAbulk_dVg, dn_dVd ;
double beta, dbeta_dVg, dbeta_dVd, dbeta_dVb;
double gche, dgche_dVg, dgche_dVd, dgche_dVb;
double fgche1, dfgche1_dVg, dfgche1_dVd, dfgche1_dVb;
double fgche2, dfgche2_dVg, dfgche2_dVd, dfgche2_dVb;
double Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb;
double Idsa, dIdsa_dVg, dIdsa_dVd, dIdsa_dVb;
double Ids, Gm, Gds, Gmb;
double Isub, Gbd, Gbg, Gbb;
double VASCBE, dVASCBE_dVg, dVASCBE_dVd, dVASCBE_dVb;
double CoxWovL;
double Rds, dRds_dVg, dRds_dVb, WVCox, WVCoxRds;
double Vgst2Vtm, VdsatCV, dVdsatCV_dVg, dVdsatCV_dVb;
double Leff = 0.0, Weff, dWeff_dVg, dWeff_dVb;
double AbulkCV, dAbulkCV_dVb;
double qgdo, qgso, cgdo, cgso;

double qcheq, qdef, gqdef, cqdef, cqcheq, gtau_diff, gtau_drift;
double gcqdb,gcqsb,gcqgb,gcqbb;
double dxpart, sxpart;

double gbspsp, gbbdp, gbbsp, gbspg, gbspb, gbspdp; 
double gbdpdp, gbdpg, gbdpb, gbdpsp; 
double Cgg, Cgd, Cgb;
double Csg, Csd, Csb, Cbg, Cbd, Cbb;
double Cgg1, Cgb1, Cgd1, Cbg1, Cbb1, Cbd1, Qac0, Qsub0;
double dQac0_dVg, dQac0_dVd, dQac0_dVb, dQsub0_dVg, dQsub0_dVd, dQsub0_dVb;

double m = 1.0;
   
struct bsim3v1SizeDependParam *pParam;
int ByPass, Check, ChargeComputationNeeded = 0, error;

for (; model != NULL; model = BSIM3v1nextModel(model))
{    for (here = BSIM3v1instances(model); here != NULL; 
          here = BSIM3v1nextInstance(here))
     {
	  Check = 1;
          ByPass = 0;
	  pParam = here->pParam;
          if ((ckt->CKTmode & MODEINITSMSIG))
	  {   vbs = *(ckt->CKTstate0 + here->BSIM3v1vbs);
              vgs = *(ckt->CKTstate0 + here->BSIM3v1vgs);
              vds = *(ckt->CKTstate0 + here->BSIM3v1vds);
              qdef = *(ckt->CKTstate0 + here->BSIM3v1qcdump);
          }
	  else if ((ckt->CKTmode & MODEINITTRAN))
	  {   vbs = *(ckt->CKTstate1 + here->BSIM3v1vbs);
              vgs = *(ckt->CKTstate1 + here->BSIM3v1vgs);
              vds = *(ckt->CKTstate1 + here->BSIM3v1vds);
              qdef = *(ckt->CKTstate1 + here->BSIM3v1qcdump);
          }
	  else if ((ckt->CKTmode & MODEINITJCT) && !here->BSIM3v1off)
	  {   vds = model->BSIM3v1type * here->BSIM3v1icVDS;
              vgs = model->BSIM3v1type * here->BSIM3v1icVGS;
              vbs = model->BSIM3v1type * here->BSIM3v1icVBS;
              qdef = 0.0;

              if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) && 
                  ((ckt->CKTmode & (MODETRAN | MODEAC|MODEDCOP |
                   MODEDCTRANCURVE)) || (!(ckt->CKTmode & MODEUIC))))
	      {   vbs = 0.0;
                  vgs = pParam->BSIM3v1vth0 + 0.1;
                  vds = 0.1;
              }
          }
	  else if ((ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) && 
                  (here->BSIM3v1off)) 
          {    qdef = vbs = vgs = vds = 0.0;
	  }
          else
	  {
#ifndef PREDICTOR
               if ((ckt->CKTmode & MODEINITPRED))
	       {   xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
                   *(ckt->CKTstate0 + here->BSIM3v1vbs) = 
                         *(ckt->CKTstate1 + here->BSIM3v1vbs);
                   vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v1vbs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v1vbs)));
                   *(ckt->CKTstate0 + here->BSIM3v1vgs) = 
                         *(ckt->CKTstate1 + here->BSIM3v1vgs);
                   vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v1vgs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v1vgs)));
                   *(ckt->CKTstate0 + here->BSIM3v1vds) = 
                         *(ckt->CKTstate1 + here->BSIM3v1vds);
                   vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v1vds))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v1vds)));
                   *(ckt->CKTstate0 + here->BSIM3v1vbd) = 
                         *(ckt->CKTstate0 + here->BSIM3v1vbs)
                         - *(ckt->CKTstate0 + here->BSIM3v1vds);
                   qdef = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v1qcdump))
                           -(xfact * (*(ckt->CKTstate2 + here->BSIM3v1qcdump)));
               }
	       else
	       {
#endif /* PREDICTOR */
                   vbs = model->BSIM3v1type
                       * (*(ckt->CKTrhsOld + here->BSIM3v1bNode)
                       - *(ckt->CKTrhsOld + here->BSIM3v1sNodePrime));
                   vgs = model->BSIM3v1type
                       * (*(ckt->CKTrhsOld + here->BSIM3v1gNode) 
                       - *(ckt->CKTrhsOld + here->BSIM3v1sNodePrime));
                   vds = model->BSIM3v1type
                       * (*(ckt->CKTrhsOld + here->BSIM3v1dNodePrime)
                       - *(ckt->CKTrhsOld + here->BSIM3v1sNodePrime));
                   qdef = *(ckt->CKTrhsOld + here->BSIM3v1qNode);
#ifndef PREDICTOR
               }
#endif /* PREDICTOR */

               vbd = vbs - vds;
               vgd = vgs - vds;
               vgdo = *(ckt->CKTstate0 + here->BSIM3v1vgs)
                    - *(ckt->CKTstate0 + here->BSIM3v1vds);
               delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v1vbs);
               delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v1vbd);
               delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v1vgs);
               delvds = vds - *(ckt->CKTstate0 + here->BSIM3v1vds);
               delvgd = vgd - vgdo;

               if (here->BSIM3v1mode >= 0) 
	       {   cdhat = here->BSIM3v1cd - here->BSIM3v1gbd * delvbd 
                         + here->BSIM3v1gmbs * delvbs + here->BSIM3v1gm * delvgs
                         + here->BSIM3v1gds * delvds;
	       }
               else
	       {   cdhat = here->BSIM3v1cd - (here->BSIM3v1gbd - here->BSIM3v1gmbs)
                         * delvbd - here->BSIM3v1gm * delvgd 
                         + here->BSIM3v1gds * delvds;
               
	       }
               cbhat = here->BSIM3v1cbs + here->BSIM3v1cbd + here->BSIM3v1gbd
                     * delvbd + here->BSIM3v1gbs * delvbs;

#ifndef NOBYPASS
           /* following should be one big if connected by && all over
            * the place, but some C compilers can't handle that, so
            * we split it up here to let them digest it in stages
            */

               if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass))
               if ((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                   fabs(*(ckt->CKTstate0+here->BSIM3v1vbs))) + ckt->CKTvoltTol)))
               if ((fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                   fabs(*(ckt->CKTstate0+here->BSIM3v1vbd))) + ckt->CKTvoltTol)))
               if ((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                   fabs(*(ckt->CKTstate0+here->BSIM3v1vgs))) + ckt->CKTvoltTol)))
               if ((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                   fabs(*(ckt->CKTstate0+here->BSIM3v1vds))) + ckt->CKTvoltTol)))
               if ((fabs(cdhat - here->BSIM3v1cd) < ckt->CKTreltol 
                   * MAX(fabs(cdhat),fabs(here->BSIM3v1cd)) + ckt->CKTabstol))
	       {   tempv = MAX(fabs(cbhat),fabs(here->BSIM3v1cbs 
                         + here->BSIM3v1cbd)) + ckt->CKTabstol;
                   if ((fabs(cbhat - (here->BSIM3v1cbs + here->BSIM3v1cbd))
                       < ckt->CKTreltol * tempv))
		   {   /* bypass code */
                       vbs = *(ckt->CKTstate0 + here->BSIM3v1vbs);
                       vbd = *(ckt->CKTstate0 + here->BSIM3v1vbd);
                       vgs = *(ckt->CKTstate0 + here->BSIM3v1vgs);
                       vds = *(ckt->CKTstate0 + here->BSIM3v1vds);
                       qcheq = *(ckt->CKTstate0 + here->BSIM3v1qcheq);

                       vgd = vgs - vds;
                       vgb = vgs - vbs;

                       cdrain = here->BSIM3v1mode * (here->BSIM3v1cd
                              + here->BSIM3v1cbd); 
                       if ((ckt->CKTmode & (MODETRAN | MODEAC)) || 
                           ((ckt->CKTmode & MODETRANOP) && 
                           (ckt->CKTmode & MODEUIC)))
		       {   ByPass = 1;
                           goto line755;
                       }
		       else
		       {   goto line850;
		       }
                   }
               }

#endif /*NOBYPASS*/
               von = here->BSIM3v1von;
               if (*(ckt->CKTstate0 + here->BSIM3v1vds) >= 0.0)
	       {   vgs = DEVfetlim(vgs, *(ckt->CKTstate0+here->BSIM3v1vgs), von);
                   vds = vgs - vgd;
                   vds = DEVlimvds(vds, *(ckt->CKTstate0 + here->BSIM3v1vds));
                   vgd = vgs - vds;

               }
	       else
	       {   vgd = DEVfetlim(vgd, vgdo, von);
                   vds = vgs - vgd;
                   vds = -DEVlimvds(-vds, -(*(ckt->CKTstate0+here->BSIM3v1vds)));
                   vgs = vgd + vds;
               }

               if (vds >= 0.0)
	       {   vbs = DEVpnjlim(vbs, *(ckt->CKTstate0 + here->BSIM3v1vbs),
                                   CONSTvt0, model->BSIM3v1vcrit, &Check);
                   vbd = vbs - vds;

               }
	       else
	       {   vbd = DEVpnjlim(vbd, *(ckt->CKTstate0 + here->BSIM3v1vbd),
                                   CONSTvt0, model->BSIM3v1vcrit, &Check); 
                   vbs = vbd + vds;
               }
          }

          /* determine DC current and derivatives */
          vbd = vbs - vds;
          vgd = vgs - vds;
          vgb = vgs - vbs;
	  
	  m = here->BSIM3v1m;
		  
/* the following code has been changed for the calculation of S/B and D/B diodes*/

	  if ((here->BSIM3v1sourceArea <= 0.0) &&
	      (here->BSIM3v1sourcePerimeter <= 0.0))
	  {   SourceSatCurrent = 1.0e-14;
	  }
	  else
	  {   SourceSatCurrent = here->BSIM3v1sourceArea
			       * model->BSIM3v1jctTempSatCurDensity
			       + here->BSIM3v1sourcePerimeter
			       * model->BSIM3v1jctSidewallTempSatCurDensity;
	  }

	  Nvtm = model->BSIM3v1vtm * model->BSIM3v1jctEmissionCoeff;
	  if (SourceSatCurrent <= 0.0)
	  {   here->BSIM3v1gbs = ckt->CKTgmin;
              here->BSIM3v1cbs = here->BSIM3v1gbs * vbs;
          }
	  else if (vbs < 0.5)
	  {   evbs = exp(vbs / Nvtm);
              here->BSIM3v1gbs = SourceSatCurrent * evbs / Nvtm + ckt->CKTgmin;
              here->BSIM3v1cbs = SourceSatCurrent * (evbs - 1.0) 
                             + ckt->CKTgmin * vbs;
          }
	  else
	  {   evbs = exp(0.5 / Nvtm);
              T0 = SourceSatCurrent * evbs / Nvtm;
              here->BSIM3v1gbs = T0 + ckt->CKTgmin;
              here->BSIM3v1cbs = SourceSatCurrent * (evbs - 1.0) 
			     + T0 * (vbs - 0.5) + ckt->CKTgmin * vbs;
          }

	  if ((here->BSIM3v1drainArea <= 0.0) &&
	      (here->BSIM3v1drainPerimeter <= 0.0))
	  {   DrainSatCurrent = 1.0e-14;
	  }
	  else
	  {   DrainSatCurrent = here->BSIM3v1drainArea
			      * model->BSIM3v1jctTempSatCurDensity
			      + here->BSIM3v1drainPerimeter
			      * model->BSIM3v1jctSidewallTempSatCurDensity;
	  }

	  if (DrainSatCurrent <= 0.0)
	  {   here->BSIM3v1gbd = ckt->CKTgmin;
              here->BSIM3v1cbd = here->BSIM3v1gbd * vbd;
          }
	  else if (vbd < 0.5)
	  {   evbd = exp(vbd / Nvtm);
              here->BSIM3v1gbd = DrainSatCurrent * evbd / Nvtm + ckt->CKTgmin;
              here->BSIM3v1cbd = DrainSatCurrent * (evbd - 1.0) 
                             + ckt->CKTgmin * vbd;
          }
	  else
	  {   evbd = exp(0.5 / Nvtm);
              T0 = DrainSatCurrent * evbd / Nvtm;
              here->BSIM3v1gbd = T0 + ckt->CKTgmin;
              here->BSIM3v1cbd = DrainSatCurrent * (evbd - 1.0) 
			     + T0 * (vbd - 0.5) + ckt->CKTgmin * vbd;
          }
/* S/B and D/B diodes code change ends */

          if (vds >= 0.0)
	  {   /* normal mode */
              here->BSIM3v1mode = 1;
              Vds = vds;
              Vgs = vgs;
              Vbs = vbs;
          }
	  else
	  {   /* inverse mode */
              here->BSIM3v1mode = -1;
              Vds = -vds;
              Vgs = vgd;
              Vbs = vbd;
          }

          ChargeComputationNeeded =  
                 ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
                 ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
                 ? 1 : 0;
	  T0 = Vbs - pParam->BSIM3v1vbsc - 0.001;
	  T1 = sqrt(T0 * T0 - 0.004 * pParam->BSIM3v1vbsc);
	  Vbseff = pParam->BSIM3v1vbsc + 0.5 * (T0 + T1);
	  dVbseff_dVb = 0.5 * (1.0 + T0 / T1);
	  if (Vbseff < Vbs)
	  {   Vbseff = Vbs;
	  } /* Added to avoid the possible numerical problems due to computer accuracy. See comments for diffVds */

          if (Vbseff > 0.0)
	  {   T0 = pParam->BSIM3v1phi / (pParam->BSIM3v1phi + Vbseff);
              Phis = pParam->BSIM3v1phi * T0;
              dPhis_dVb = -T0 * T0;
              sqrtPhis = pParam->BSIM3v1phis3 / (pParam->BSIM3v1phi + 0.5 * Vbseff);
              dsqrtPhis_dVb = -0.5 * sqrtPhis * sqrtPhis / pParam->BSIM3v1phis3;
          }
	  else
	  {   Phis = pParam->BSIM3v1phi - Vbseff;
              dPhis_dVb = -1.0;
              sqrtPhis = sqrt(Phis);
              dsqrtPhis_dVb = -0.5 / sqrtPhis; 
          }
          Xdep = pParam->BSIM3v1Xdep0 * sqrtPhis / pParam->BSIM3v1sqrtPhi;
          dXdep_dVb = (pParam->BSIM3v1Xdep0 / pParam->BSIM3v1sqrtPhi)
		    * dsqrtPhis_dVb;

          Leff = pParam->BSIM3v1leff;
          Vtm = model->BSIM3v1vtm;
/* Vth Calculation */
          T3 = sqrt(Xdep);
          V0 = pParam->BSIM3v1vbi - pParam->BSIM3v1phi;

          T0 = pParam->BSIM3v1dvt2 * Vbseff;
          if (T0 >= - 0.5)
	  {   T1 = 1.0 + T0;
	      T2 = pParam->BSIM3v1dvt2;
	  }
	  else /* Added to avoid any discontinuity problems caused by dvt2 */ 
	  {   T4 = 1.0 / (3.0 + 8.0 * T0);
	      T1 = (1.0 + 3.0 * T0) * T4; 
	      T2 = pParam->BSIM3v1dvt2 * T4 * T4;
	  }
          lt1 = model->BSIM3v1factor1 * T3 * T1;
          dlt1_dVb = model->BSIM3v1factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

          T0 = pParam->BSIM3v1dvt2w * Vbseff;
          if (T0 >= - 0.5)
	  {   T1 = 1.0 + T0;
	      T2 = pParam->BSIM3v1dvt2w;
	  }
	  else /* Added to avoid any discontinuity problems caused by dvt2w */ 
	  {   T4 = 1.0 / (3.0 + 8.0 * T0);
	      T1 = (1.0 + 3.0 * T0) * T4; 
	      T2 = pParam->BSIM3v1dvt2w * T4 * T4;
	  }
          ltw = model->BSIM3v1factor1 * T3 * T1;
          dltw_dVb = model->BSIM3v1factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

          T0 = -0.5 * pParam->BSIM3v1dvt1 * Leff / lt1;
          if (T0 > -EXP_THRESHOLD)
          {   T1 = exp(T0);
              Theta0 = T1 * (1.0 + 2.0 * T1);
              dT1_dVb = -T0 / lt1 * T1 * dlt1_dVb;
              dTheta0_dVb = (1.0 + 4.0 * T1) * dT1_dVb;
          }
          else
          {   T1 = MIN_EXP;
              Theta0 = T1 * (1.0 + 2.0 * T1);
              dTheta0_dVb = 0.0;
          }

          here->BSIM3v1thetavth = pParam->BSIM3v1dvt0 * Theta0;
          Delt_vth = here->BSIM3v1thetavth * V0;
          dDelt_vth_dVb = pParam->BSIM3v1dvt0 * dTheta0_dVb * V0;

          T0 = -0.5 * pParam->BSIM3v1dvt1w * pParam->BSIM3v1weff * Leff / ltw;
          if (T0 > -EXP_THRESHOLD)
          {   T1 = exp(T0);
              T2 = T1 * (1.0 + 2.0 * T1);
              dT1_dVb = -T0 / ltw * T1 * dltw_dVb;
              dT2_dVb = (1.0 + 4.0 * T1) * dT1_dVb;
          }
          else
          {   T1 = MIN_EXP;
              T2 = T1 * (1.0 + 2.0 * T1);
              dT2_dVb = 0.0;
          }

          T0 = pParam->BSIM3v1dvt0w * T2;
          T2 = T0 * V0;
          dT2_dVb = pParam->BSIM3v1dvt0w * dT2_dVb * V0;

          TempRatio =  ckt->CKTtemp / model->BSIM3v1tnom - 1.0;
          T0 = sqrt(1.0 + pParam->BSIM3v1nlx / Leff);
          T1 = pParam->BSIM3v1k1 * (T0 - 1.0) * pParam->BSIM3v1sqrtPhi
             + (pParam->BSIM3v1kt1 + pParam->BSIM3v1kt1l / Leff
             + pParam->BSIM3v1kt2 * Vbseff) * TempRatio;
          tmp2 = model->BSIM3v1tox * pParam->BSIM3v1phi
	       / (pParam->BSIM3v1weff + pParam->BSIM3v1w0);

	  T3 = pParam->BSIM3v1eta0 + pParam->BSIM3v1etab * Vbseff;
	  if (T3 < 1.0e-4) /* avoid  discontinuity problems caused by etab */ 
	  {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
	      T3 = (2.0e-4 - T3) * T9;
	      T4 = T9 * T9;
	  }
	  else
	  {   T4 = 1.0;
	  }
	  dDIBL_Sft_dVd = T3 * pParam->BSIM3v1theta0vb0;
          DIBL_Sft = dDIBL_Sft_dVd * Vds;

          Vth = model->BSIM3v1type * pParam->BSIM3v1vth0 + pParam->BSIM3v1k1 
              * (sqrtPhis - pParam->BSIM3v1sqrtPhi) - pParam->BSIM3v1k2 
              * Vbseff - Delt_vth - T2 + (pParam->BSIM3v1k3 + pParam->BSIM3v1k3b
              * Vbseff) * tmp2 + T1 - DIBL_Sft;

          here->BSIM3v1von = Vth; 

          dVth_dVb = pParam->BSIM3v1k1 * dsqrtPhis_dVb - pParam->BSIM3v1k2
                   - dDelt_vth_dVb - dT2_dVb + pParam->BSIM3v1k3b * tmp2
                   - pParam->BSIM3v1etab * Vds * pParam->BSIM3v1theta0vb0 * T4
                   + pParam->BSIM3v1kt2 * TempRatio;
          dVth_dVd = -dDIBL_Sft_dVd; 

/* Calculate n */
          tmp2 = pParam->BSIM3v1nfactor * EPSSI / Xdep;
          tmp3 = pParam->BSIM3v1cdsc + pParam->BSIM3v1cdscb * Vbseff
               + pParam->BSIM3v1cdscd * Vds;
	  tmp4 = (tmp2 + tmp3 * Theta0 + pParam->BSIM3v1cit) / model->BSIM3v1cox;
	  if (tmp4 >= -0.5)
	  {   n = 1.0 + tmp4;
	      dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM3v1cdscb * Theta0) / model->BSIM3v1cox;
              dn_dVd = pParam->BSIM3v1cdscd * Theta0 / model->BSIM3v1cox;
	  }
	  else
	   /* avoid  discontinuity problems caused by tmp4 */ 
	  {   T0 = 1.0 / (3.0 + 8.0 * tmp4);
	      n = (1.0 + 3.0 * tmp4) * T0;
	      T0 *= T0;
	      dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM3v1cdscb * Theta0) / model->BSIM3v1cox * T0;
              dn_dVd = pParam->BSIM3v1cdscd * Theta0 / model->BSIM3v1cox * T0;
	  }

/* Poly Gate Si Depletion Effect */
	  T0 = pParam->BSIM3v1vfb + pParam->BSIM3v1phi;
          if ((pParam->BSIM3v1ngate > 1.e18) && (pParam->BSIM3v1ngate < 1.e25) 
               && (Vgs > T0))
	  /* added to avoid the problem caused by ngate */
          {   T1 = 1.0e6 * Charge_q * EPSSI * pParam->BSIM3v1ngate
                 / (model->BSIM3v1cox * model->BSIM3v1cox);
              T4 = sqrt(1.0 + 2.0 * (Vgs - T0) / T1);
              T2 = T1 * (T4 - 1.0);
              T3 = 0.5 * T2 * T2 / T1; /* T3 = Vpoly */
              T7 = 1.12 - T3 - 0.05;
              T6 = sqrt(T7 * T7 + 0.224);
              T5 = 1.12 - 0.5 * (T7 + T6);
              Vgs_eff = Vgs - T5;
              dVgs_eff_dVg = 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6); 
          }
	  else
	  {   Vgs_eff = Vgs;
              dVgs_eff_dVg = 1.0;
          }
          Vgst = Vgs_eff - Vth;

/* Effective Vgst (Vgsteff) Calculation */

          T10 = 2.0 * n * Vtm;
          VgstNVt = Vgst / T10;
          ExpArg = (2.0 * pParam->BSIM3v1voff - Vgst) / T10;

	  /* MCJ: Very small Vgst */
          if (VgstNVt > EXP_THRESHOLD)
	  {   Vgsteff = Vgst;
              dVgsteff_dVg = dVgs_eff_dVg;
              dVgsteff_dVd = -dVth_dVd;
              dVgsteff_dVb = -dVth_dVb;
	  }
	  else if (ExpArg > EXP_THRESHOLD)
	  {   T0 = (Vgst - pParam->BSIM3v1voff) / (n * Vtm);
	      ExpVgst = exp(T0);
	      Vgsteff = Vtm * pParam->BSIM3v1cdep0 / model->BSIM3v1cox * ExpVgst;
              dVgsteff_dVg = Vgsteff / (n * Vtm);
              dVgsteff_dVd = -dVgsteff_dVg * (dVth_dVd + T0 * Vtm * dn_dVd);
              dVgsteff_dVb = -dVgsteff_dVg * (dVth_dVb + T0 * Vtm * dn_dVb);
	      dVgsteff_dVg *= dVgs_eff_dVg;
	  }
	  else
	  {   ExpVgst = exp(VgstNVt);
              T1 = T10 * log(1.0 + ExpVgst);
              dT1_dVg = ExpVgst / (1.0 + ExpVgst);
              dT1_dVb = -dT1_dVg * (dVth_dVb + Vgst / n * dn_dVb)
		      + T1 / n * dn_dVb; 
	      dT1_dVd = -dT1_dVg * (dVth_dVd + Vgst / n * dn_dVd)
		      + T1 / n * dn_dVd;

	      dT2_dVg = -model->BSIM3v1cox / (Vtm * pParam->BSIM3v1cdep0)
		      * exp(ExpArg);
              T2 = 1.0 - T10 * dT2_dVg;
              dT2_dVd = -dT2_dVg * (dVth_dVd - 2.0 * Vtm * ExpArg * dn_dVd)
		      + (T2 - 1.0) / n * dn_dVd;
              dT2_dVb = -dT2_dVg * (dVth_dVb - 2.0 * Vtm * ExpArg * dn_dVb)
		      + (T2 - 1.0) / n * dn_dVb;

              Vgsteff = T1 / T2;
	      T3 = T2 * T2;
              dVgsteff_dVg = (T2 * dT1_dVg - T1 * dT2_dVg) / T3 * dVgs_eff_dVg;
              dVgsteff_dVd = (T2 * dT1_dVd - T1 * dT2_dVd) / T3;
              dVgsteff_dVb = (T2 * dT1_dVb - T1 * dT2_dVb) / T3;
	  }

/* Calculate Effective Channel Geometry */
          T9 = sqrtPhis - pParam->BSIM3v1sqrtPhi;
          Weff = pParam->BSIM3v1weff - 2.0 * (pParam->BSIM3v1dwg * Vgsteff 
               + pParam->BSIM3v1dwb * T9); 
          dWeff_dVg = -2.0 * pParam->BSIM3v1dwg;
          dWeff_dVb = -2.0 * pParam->BSIM3v1dwb * dsqrtPhis_dVb;

          if (Weff < 2.0e-8) /* to avoid the discontinuity problem due to Weff*/
	  {   T0 = 1.0 / (6.0e-8 - 2.0 * Weff);
	      Weff = 2.0e-8 * (4.0e-8 - Weff) * T0;
	      T0 *= T0 * 4.0e-16;
              dWeff_dVg *= T0;
	      dWeff_dVb *= T0;
          }

          T0 = pParam->BSIM3v1prwg * Vgsteff + pParam->BSIM3v1prwb * T9;
	  if (T0 >= -0.9)
	  {   Rds = pParam->BSIM3v1rds0 * (1.0 + T0);
	      dRds_dVg = pParam->BSIM3v1rds0 * pParam->BSIM3v1prwg;
              dRds_dVb = pParam->BSIM3v1rds0 * pParam->BSIM3v1prwb * dsqrtPhis_dVb;
	  }
	  else
           /* to avoid the discontinuity problem due to prwg and prwb*/
	  {   T1 = 1.0 / (17.0 + 20.0 * T0);
	      Rds = pParam->BSIM3v1rds0 * (0.8 + T0) * T1;
	      T1 *= T1;
	      dRds_dVg = pParam->BSIM3v1rds0 * pParam->BSIM3v1prwg * T1;
              dRds_dVb = pParam->BSIM3v1rds0 * pParam->BSIM3v1prwb * dsqrtPhis_dVb
		       * T1;
	  }
	  
/* Calculate Abulk */
          T1 = 0.5 * pParam->BSIM3v1k1 / sqrtPhis;
          dT1_dVb = -T1 / sqrtPhis * dsqrtPhis_dVb;

          T9 = sqrt(pParam->BSIM3v1xj * Xdep);
          tmp1 = Leff + 2.0 * T9;
          T5 = Leff / tmp1; 
          tmp2 = pParam->BSIM3v1a0 * T5;
          tmp3 = pParam->BSIM3v1weff + pParam->BSIM3v1b1; 
          tmp4 = pParam->BSIM3v1b0 / tmp3;
          T2 = tmp2 + tmp4;
          dT2_dVb = -T9 / tmp1 / Xdep * dXdep_dVb;
          T6 = T5 * T5;
          T7 = T5 * T6;

          Abulk0 = 1.0 + T1 * T2; 
          dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;

          T8 = pParam->BSIM3v1ags * pParam->BSIM3v1a0 * T7;
          dAbulk_dVg = -T1 * T8;
          Abulk = Abulk0 + dAbulk_dVg * Vgsteff; 
          dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb
		     + 3.0 * T1 * dT2_dVb);

          if (Abulk0 < 0.1) /* added to avoid the problems caused by Abulk0 */
	  {   T9 = 1.0 / (3.0 - 20.0 * Abulk0);
	      Abulk0 = (0.2 - Abulk0) * T9;
	      dAbulk0_dVb *= T9 * T9;
	  }

          if (Abulk < 0.1)
          /* added to avoid the problems caused by Abulk */
	  {   T9 = 1.0 / (3.0 - 20.0 * Abulk);
	      Abulk = (0.2 - Abulk) * T9;
	      dAbulk_dVb *= T9 * T9;
	  }

          T2 = pParam->BSIM3v1keta * Vbseff;
	  if (T2 >= -0.9)
	  {   T0 = 1.0 / (1.0 + T2);
              dT0_dVb = -pParam->BSIM3v1keta * T0 * T0;
	  }
	  else
          /* added to avoid the problems caused by Keta */
	  {   T1 = 1.0 / (0.8 + T2);
	      T0 = (17.0 + 20.0 * T2) * T1;
              dT0_dVb = -pParam->BSIM3v1keta * T1 * T1;
	  }
	  dAbulk_dVg *= T0;
	  dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
	  dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
	  Abulk *= T0;
	  Abulk0 *= T0;


/* Mobility calculation */
          if (model->BSIM3v1mobMod == 1)
	  {   T0 = Vgsteff + Vth + Vth;
              T2 = pParam->BSIM3v1ua + pParam->BSIM3v1uc * Vbseff;
              T3 = T0 / model->BSIM3v1tox;
              T5 = T3 * (T2 + pParam->BSIM3v1ub * T3);
              dDenomi_dVg = (T2 + 2.0 * pParam->BSIM3v1ub * T3) / model->BSIM3v1tox;
              dDenomi_dVd = dDenomi_dVg * 2.0 * dVth_dVd;
              dDenomi_dVb = dDenomi_dVg * 2.0 * dVth_dVb + pParam->BSIM3v1uc * T3;
          }
	  else if (model->BSIM3v1mobMod == 2)
	  {   T5 = Vgsteff / model->BSIM3v1tox * (pParam->BSIM3v1ua
		 + pParam->BSIM3v1uc * Vbseff + pParam->BSIM3v1ub * Vgsteff
                 / model->BSIM3v1tox);
              dDenomi_dVg = (pParam->BSIM3v1ua + pParam->BSIM3v1uc * Vbseff
		          + 2.0 * pParam->BSIM3v1ub * Vgsteff / model->BSIM3v1tox)
		          / model->BSIM3v1tox;
              dDenomi_dVd = 0.0;
              dDenomi_dVb = Vgsteff * pParam->BSIM3v1uc / model->BSIM3v1tox; 
          }
	  else
	  {   T0 = Vgsteff + Vth + Vth;
              T2 = 1.0 + pParam->BSIM3v1uc * Vbseff;
              T3 = T0 / model->BSIM3v1tox;
              T4 = T3 * (pParam->BSIM3v1ua + pParam->BSIM3v1ub * T3);
	      T5 = T4 * T2;
              dDenomi_dVg = (pParam->BSIM3v1ua + 2.0 * pParam->BSIM3v1ub * T3) * T2
		          / model->BSIM3v1tox;
              dDenomi_dVd = dDenomi_dVg * 2.0 * dVth_dVd;
              dDenomi_dVb = dDenomi_dVg * 2.0 * dVth_dVb + pParam->BSIM3v1uc * T4;
          }

	  if (T5 >= -0.8)
	  {   Denomi = 1.0 + T5;
	  }
	  else /* Added to avoid the discontinuity problem caused by ua and ub*/ 
	  {   T9 = 1.0 / (7.0 + 10.0 * T5);
	      Denomi = (0.6 + T5) * T9;
	      T9 *= T9;
              dDenomi_dVg *= T9;
              dDenomi_dVd *= T9;
              dDenomi_dVb *= T9;
	  }

          here->BSIM3v1ueff = ueff = pParam->BSIM3v1u0temp / Denomi;
	  T9 = -ueff / Denomi;
          dueff_dVg = T9 * dDenomi_dVg;
          dueff_dVd = T9 * dDenomi_dVd;
          dueff_dVb = T9 * dDenomi_dVb;

/* Saturation Drain Voltage  Vdsat */
          WVCox = Weff * pParam->BSIM3v1vsattemp * model->BSIM3v1cox;
          WVCoxRds = WVCox * Rds; 

          Esat = 2.0 * pParam->BSIM3v1vsattemp / ueff;
          EsatL = Esat * Leff;
          T0 = -EsatL /ueff;
          dEsatL_dVg = T0 * dueff_dVg;
          dEsatL_dVd = T0 * dueff_dVd;
          dEsatL_dVb = T0 * dueff_dVb;
  
	  /* Sqrt() */
          a1 = pParam->BSIM3v1a1;
	  if (a1 == 0.0)
	  {   Lambda = pParam->BSIM3v1a2;
	      dLambda_dVg = 0.0;
	  }
	  else if (a1 > 0.0)
/* Added to avoid the discontinuity problem
   caused by a1 and a2 (Lambda) */
	  {   T0 = 1.0 - pParam->BSIM3v1a2;
	      T1 = T0 - pParam->BSIM3v1a1 * Vgsteff - 0.0001;
	      T2 = sqrt(T1 * T1 + 0.0004 * T0);
	      Lambda = pParam->BSIM3v1a2 + T0 - 0.5 * (T1 + T2);
	      dLambda_dVg = 0.5 * pParam->BSIM3v1a1 * (1.0 + T1 / T2);
	  }
	  else
	  {   T1 = pParam->BSIM3v1a2 + pParam->BSIM3v1a1 * Vgsteff - 0.0001;
	      T2 = sqrt(T1 * T1 + 0.0004 * pParam->BSIM3v1a2);
	      Lambda = 0.5 * (T1 + T2);
	      dLambda_dVg = 0.5 * pParam->BSIM3v1a1 * (1.0 + T1 / T2);
	  }

          Vgst2Vtm = Vgsteff + 2.0 * Vtm;
          if (Rds > 0)
          {   tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff;
              tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff;
          }
          else
          {   tmp2 = dWeff_dVg / Weff;
              tmp3 = dWeff_dVb / Weff;
	  }
          if ((Rds == 0.0) && (Lambda == 1.0))
          {   T0 = 1.0 / (Abulk * EsatL + Vgst2Vtm);
              tmp1 = 0.0;
	      T1 = T0 * T0;
	      T2 = Vgst2Vtm * T0;
              T3 = EsatL * Vgst2Vtm;
              Vdsat = T3 * T0;
                           
              dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 1.0) * T1;
              dT0_dVd = -(Abulk * dEsatL_dVd) * T1; 
              dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1;   

              dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
              dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
              dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;   
          }
          else
          {   tmp1 = dLambda_dVg / (Lambda * Lambda);
              T9 = Abulk * WVCoxRds;
	      T8 = Abulk * T9;
	      T7 = Vgst2Vtm * T9;
              T6 = Vgst2Vtm * WVCoxRds;
              T0 = 2.0 * Abulk * (T9 - 1.0 + 1.0 / Lambda); 
              dT0_dVg = 2.0 * (T8 * tmp2 - Abulk * tmp1
		      + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbulk_dVg);
             
              dT0_dVb = 2.0 * (T8 * (2.0 / Abulk * dAbulk_dVb + tmp3)
		      + (1.0 / Lambda - 1.0) * dAbulk_dVb);
	      dT0_dVd = 0.0; 
              T1 = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abulk * EsatL + 3.0 * T7;
             
              dT1_dVg = (2.0 / Lambda - 1.0) - 2.0 * Vgst2Vtm * tmp1
		      + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 3.0 * (T9
		      + T7 * tmp2 + T6 * dAbulk_dVg);
              dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
	              + 3.0 * (T6 * dAbulk_dVb + T7 * tmp3);
              dT1_dVd = Abulk * dEsatL_dVd;

              T2 = Vgst2Vtm * (EsatL + 2.0 * T6);
              dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg
		      + T6 * (4.0 + 2.0 * Vgst2Vtm * tmp2);
              dT2_dVb = Vgst2Vtm * (dEsatL_dVb + 2.0 * T6 * tmp3);
              dT2_dVd = Vgst2Vtm * dEsatL_dVd;

              T3 = sqrt(T1 * T1 - 2.0 * T0 * T2);
              Vdsat = (T1 - T3) / T0;

              dT3_dVg = (T1 * dT1_dVg - 2.0 * (T0 * dT2_dVg + T2 * dT0_dVg))
	              / T3;
              dT3_dVd = (T1 * dT1_dVd - 2.0 * (T0 * dT2_dVd + T2 * dT0_dVd))
		      / T3;
              dT3_dVb = (T1 * dT1_dVb - 2.0 * (T0 * dT2_dVb + T2 * dT0_dVb))
		      / T3;

              dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2
			 - T0 * dT2_dVg) / T3 - Vdsat * dT0_dVg) / T0;
              dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2
			 - T0 * dT2_dVb) / T3 - Vdsat * dT0_dVb) / T0;
              dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) / T3) / T0;
          }
          here->BSIM3v1vdsat = Vdsat;

/* Effective Vds (Vdseff) Calculation */
          T1 = Vdsat - Vds - pParam->BSIM3v1delta;
          dT1_dVg = dVdsat_dVg;
          dT1_dVd = dVdsat_dVd - 1.0;
          dT1_dVb = dVdsat_dVb;

          T2 = sqrt(T1 * T1 + 4.0 * pParam->BSIM3v1delta * Vdsat);
	  T0 = T1 / T2;
	  T3 = 2.0 * pParam->BSIM3v1delta / T2;
          dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
          dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
          dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

          Vdseff = Vdsat - 0.5 * (T1 + T2);
          dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg); 
          dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd); 
          dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb); 

/* Calculate VAsat */
          tmp4 = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm;
          T9 = WVCoxRds * Vgsteff;
	  T8 = T9 / Vgst2Vtm;
          T0 = EsatL + Vdsat + 2.0 * T9 * tmp4;
         
          T7 = 2.0 * WVCoxRds * tmp4;
          dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (1.0 + tmp2 * Vgsteff)
		  - T8 * (Abulk * dVdsat_dVg - Abulk * Vdsat / Vgst2Vtm
		  + Vdsat * dAbulk_dVg);   
		  
          dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3 * Vgsteff
		  - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
          dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd;

          T9 = WVCoxRds * Abulk; 
          T1 = 2.0 / Lambda - 1.0 + T9; 
          dT1_dVg = -2.0 * tmp1 +  WVCoxRds * (Abulk * tmp2 + dAbulk_dVg);
          dT1_dVb = dAbulk_dVb * WVCoxRds + T9 * tmp3;

          Vasat = T0 / T1;
          dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) / T1;
          dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) / T1;
          dVasat_dVd = dT0_dVd / T1;

	  if (Vdseff > Vds)
	      Vdseff = Vds; /* This code is added to fixed the problem
			       caused by computer precision when
			       Vds is very close to Vdseff. */
          diffVds = Vds - Vdseff;
/* Calculate VACLM */
          if ((pParam->BSIM3v1pclm > 0.0) && (diffVds > 1.0e-10))
	  {   T0 = 1.0 / (pParam->BSIM3v1pclm * Abulk * pParam->BSIM3v1litl);
              dT0_dVb = -T0 / Abulk * dAbulk_dVb;
              dT0_dVg = -T0 / Abulk * dAbulk_dVg; 
              
	      T2 = Vgsteff / EsatL;
              T1 = Leff * (Abulk + T2); 
              dT1_dVg = Leff * ((1.0 - T2 * dEsatL_dVg) / EsatL + dAbulk_dVg);
              dT1_dVb = Leff * (dAbulk_dVb - T2 * dEsatL_dVb / EsatL);
              dT1_dVd = -T2 * dEsatL_dVd / Esat;

	      T9 = T0 * T1;
              VACLM = T9 * diffVds;
              dVACLM_dVg = T0 * dT1_dVg * diffVds - T9 * dVdseff_dVg
                         + T1 * diffVds * dT0_dVg;
              dVACLM_dVb = (dT0_dVb * T1 + T0 * dT1_dVb) * diffVds
			 - T9 * dVdseff_dVb;
              dVACLM_dVd = T0 * dT1_dVd * diffVds + T9 * (1.0 - dVdseff_dVd);
          }
	  else
	  {   VACLM = MAX_EXP;
              dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = 0.0;
          }

/* Calculate VADIBL */
          if (pParam->BSIM3v1thetaRout > 0.0)
	  {   T8 = Abulk * Vdsat;
	      T0 = Vgst2Vtm * T8;
              dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8
		      + Vgst2Vtm * Vdsat * dAbulk_dVg;
              dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
              dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;

              T1 = Vgst2Vtm + T8;
              dT1_dVg = 1.0 + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
              dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
              dT1_dVd = Abulk * dVdsat_dVd;

	      T9 = T1 * T1;
	      T2 = pParam->BSIM3v1thetaRout;
              VADIBL = (Vgst2Vtm - T0 / T1) / T2;
              dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2;
              dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2;
              dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2;

	      T7 = pParam->BSIM3v1pdiblb * Vbseff;
	      if (T7 >= -0.9)
	      {   T3 = 1.0 / (1.0 + T7);
                  VADIBL *= T3;
                  dVADIBL_dVg *= T3;
                  dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam->BSIM3v1pdiblb)
			      * T3;
                  dVADIBL_dVd *= T3;
	      }
	      else
/* Added to avoid the discontinuity problem caused by pdiblcb */
	      {   T4 = 1.0 / (0.8 + T7);
		  T3 = (17.0 + 20.0 * T7) * T4;
                  dVADIBL_dVg *= T3;
                  dVADIBL_dVb = dVADIBL_dVb * T3
			      - VADIBL * pParam->BSIM3v1pdiblb * T4 * T4;
                  dVADIBL_dVd *= T3;
                  VADIBL *= T3;
	      }
          }
	  else
	  {   VADIBL = MAX_EXP;
              dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = 0.0;
          }

/* Calculate VA */
          
	  T8 = pParam->BSIM3v1pvag / EsatL;
	  T9 = T8 * Vgsteff;
	  if (T9 > -0.9)
	  {   T0 = 1.0 + T9;
              dT0_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL);
              dT0_dVb = -T9 * dEsatL_dVb / EsatL;
              dT0_dVd = -T9 * dEsatL_dVd / EsatL;
	  }
	  else /* Added to avoid the discontinuity problems caused by pvag */
	  {   T1 = 1.0 / (17.0 + 20.0 * T9);
	      T0 = (0.8 + T9) * T1;
	      T1 *= T1;
              dT0_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T1;

              T9 *= T1 / EsatL;
              dT0_dVb = -T9 * dEsatL_dVb;
              dT0_dVd = -T9 * dEsatL_dVd;
	  }
        
          tmp1 = VACLM * VACLM;
          tmp2 = VADIBL * VADIBL;
          tmp3 = VACLM + VADIBL;

          T1 = VACLM * VADIBL / tmp3;
          tmp3 *= tmp3;
          dT1_dVg = (tmp1 * dVADIBL_dVg + tmp2 * dVACLM_dVg) / tmp3;
          dT1_dVd = (tmp1 * dVADIBL_dVd + tmp2 * dVACLM_dVd) / tmp3;
          dT1_dVb = (tmp1 * dVADIBL_dVb + tmp2 * dVACLM_dVb) / tmp3;

          Va = Vasat + T0 * T1;
          dVa_dVg = dVasat_dVg + T1 * dT0_dVg + T0 * dT1_dVg;
          dVa_dVd = dVasat_dVd + T1 * dT0_dVd + T0 * dT1_dVd;
          dVa_dVb = dVasat_dVb + T1 * dT0_dVb + T0 * dT1_dVb;

/* Calculate VASCBE */
	  if (pParam->BSIM3v1pscbe2 > 0.0)
	  {   if (diffVds > pParam->BSIM3v1pscbe1 * pParam->BSIM3v1litl
		  / EXP_THRESHOLD)
	      {   T0 =  pParam->BSIM3v1pscbe1 * pParam->BSIM3v1litl / diffVds;
	          VASCBE = Leff * exp(T0) / pParam->BSIM3v1pscbe2;
                  T1 = T0 * VASCBE / diffVds;
                  dVASCBE_dVg = T1 * dVdseff_dVg;
                  dVASCBE_dVd = -T1 * (1.0 - dVdseff_dVd);
                  dVASCBE_dVb = T1 * dVdseff_dVb;
              }
	      else
	      {   VASCBE = MAX_EXP * Leff/pParam->BSIM3v1pscbe2;
                  dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0;
              }
	  }
	  else
	  {   VASCBE = MAX_EXP;
              dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0;
	  }

/* Calculate Ids */
          CoxWovL = model->BSIM3v1cox * Weff / Leff;
          beta = ueff * CoxWovL;
          dbeta_dVg = CoxWovL * dueff_dVg + beta * dWeff_dVg / Weff;
          dbeta_dVd = CoxWovL * dueff_dVd;
          dbeta_dVb = CoxWovL * dueff_dVb + beta * dWeff_dVb / Weff;

          T0 = 1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm;
          dT0_dVg = -0.5 * (Abulk * dVdseff_dVg 
		  - Abulk * Vdseff / Vgst2Vtm + Vdseff * dAbulk_dVg) / Vgst2Vtm;
          dT0_dVd = -0.5 * Abulk * dVdseff_dVd / Vgst2Vtm;
          dT0_dVb = -0.5 * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff)
                  / Vgst2Vtm;

          fgche1 = Vgsteff * T0;
          dfgche1_dVg = Vgsteff * dT0_dVg + T0; 
          dfgche1_dVd = Vgsteff * dT0_dVd; 
          dfgche1_dVb = Vgsteff * dT0_dVb; 

          T9 = Vdseff / EsatL;
          fgche2 = 1.0 + T9;
          dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) / EsatL;
          dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) / EsatL;
          dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) / EsatL;
 
          gche = beta * fgche1 / fgche2;
          dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg
		    - gche * dfgche2_dVg) / fgche2;
          dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd
		    - gche * dfgche2_dVd) / fgche2;
          dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb
		    - gche * dfgche2_dVb) / fgche2;

          T0 = 1.0 + gche * Rds;
          T9 = Vdseff / T0;
          Idl = gche * T9;

          dIdl_dVg = (gche * dVdseff_dVg + T9 * dgche_dVg) / T0
                   - Idl * gche / T0 * dRds_dVg ; 

          dIdl_dVd = (gche * dVdseff_dVd + T9 * dgche_dVd) / T0; 
          dIdl_dVb = (gche * dVdseff_dVb + T9 * dgche_dVb 
                   - Idl * dRds_dVb * gche) / T0; 

          T9 =  diffVds / Va;
          T0 =  1.0 + T9;
          Idsa = Idl * T0;
          dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVa_dVg) / Va;
          dIdsa_dVd = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd
		    - T9 * dVa_dVd) / Va;
          dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVa_dVb) / Va;

          T9 = diffVds / VASCBE;
          T0 = 1.0 + T9;
          Ids = Idsa * T0;

          Gm = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) / VASCBE;
          Gds = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd
	      - T9 * dVASCBE_dVd) / VASCBE;
          Gmb = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb
	      + T9 * dVASCBE_dVb) / VASCBE;

          Gds += Gm * dVgsteff_dVd;
	  Gmb += Gm * dVgsteff_dVb;
	  Gm *= dVgsteff_dVg;
	  Gmb *= dVbseff_dVb;

/* calculate substrate current Isub */
          if ((pParam->BSIM3v1alpha0 <= 0.0) || (pParam->BSIM3v1beta0 <= 0.0))
	  {   Isub = Gbd = Gbb = Gbg = 0.0;
          }
	  else
	  {   T2 = pParam->BSIM3v1alpha0 / Leff;
	      if (diffVds > pParam->BSIM3v1beta0 / EXP_THRESHOLD)
	      {   T0 = -pParam->BSIM3v1beta0 / diffVds;
		  T1 = T2 * diffVds * exp(T0);
		  T3 = T1 / diffVds * (T0 - 1.0);
                  dT1_dVg = T3 * dVdseff_dVg;
                  dT1_dVd = -T3 * (1.0 - dVdseff_dVd);
                  dT1_dVb = T3 * dVdseff_dVb;
              }
	      else
	      {   T3 = T2 * MIN_EXP;
		  T1 = T3 * diffVds;
                  dT1_dVg = -T3 * dVdseff_dVg;
                  dT1_dVd = T3 * (1.0 - dVdseff_dVd);
                  dT1_dVb = -T3 * dVdseff_dVb;
              }
              Isub = T1 * Idsa;
              Gbg = T1 * dIdsa_dVg + Idsa * dT1_dVg;
              Gbd = T1 * dIdsa_dVd + Idsa * dT1_dVd;
              Gbb = T1 * dIdsa_dVb + Idsa * dT1_dVb;

              Gbd += Gbg * dVgsteff_dVd;
	      Gbb += Gbg * dVgsteff_dVb;
	      Gbg *= dVgsteff_dVg;
	      Gbb *= dVbseff_dVb; /* bug fixing */
          }
         
          cdrain = Ids;
          here->BSIM3v1gds = Gds;
          here->BSIM3v1gm = Gm;
          here->BSIM3v1gmbs = Gmb;
                   
          here->BSIM3v1gbbs = Gbb;
          here->BSIM3v1gbgs = Gbg;
          here->BSIM3v1gbds = Gbd;

          here->BSIM3v1csub = Isub - (Gbb * Vbseff + Gbd * Vds + Gbg * Vgs);

/* Calculate Qinv for Noise analysis */

          T1 = Vgsteff * (1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm);
          here->BSIM3v1qinv = -model->BSIM3v1cox * Weff * Leff * T1;

          if ((model->BSIM3v1xpart < 0) || (!ChargeComputationNeeded))
	  {   qgate  = qdrn = qsrc = qbulk = 0.0;
              here->BSIM3v1cggb = here->BSIM3v1cgsb = here->BSIM3v1cgdb = 0.0;
              here->BSIM3v1cdgb = here->BSIM3v1cdsb = here->BSIM3v1cddb = 0.0;
              here->BSIM3v1cbgb = here->BSIM3v1cbsb = here->BSIM3v1cbdb = 0.0;
              here->BSIM3v1cqdb = here->BSIM3v1cqsb = here->BSIM3v1cqgb 
                              = here->BSIM3v1cqbb = 0.0;
              here->BSIM3v1gtau = 0.0;
              goto finished;
          }
	  else if (model->BSIM3v1capMod == 0)
	  {
              if (Vbseff < 0.0)
	      {   Vbseff = Vbs;
                  dVbseff_dVb = 1.0;
              }
	      else
	      {   Vbseff = pParam->BSIM3v1phi - Phis;
                  dVbseff_dVb = -dPhis_dVb;
              }

         Vfb = pParam->BSIM3v1vfbcv;
         Vth = Vfb + pParam->BSIM3v1phi + pParam->BSIM3v1k1 * sqrtPhis; 
         Vgst = Vgs_eff - Vth;
         dVth_dVb = pParam->BSIM3v1k1 * dsqrtPhis_dVb; 
         dVgst_dVb = -dVth_dVb;
         dVgst_dVg = dVgs_eff_dVg; 

         CoxWL = model->BSIM3v1cox * pParam->BSIM3v1weffCV
                 * pParam->BSIM3v1leffCV;
         Arg1 = Vgs_eff - Vbseff - Vfb;
                if (Arg1 <= 0.0)
		{   qgate = CoxWL * Arg1;
                    qbulk = -qgate;
                    qdrn = 0.0;

                    here->BSIM3v1cggb = CoxWL * dVgs_eff_dVg;
                    here->BSIM3v1cgdb = 0.0;
                    here->BSIM3v1cgsb = CoxWL * (dVbseff_dVb
				    - dVgs_eff_dVg);

                    here->BSIM3v1cdgb = 0.0;
                    here->BSIM3v1cddb = 0.0;
                    here->BSIM3v1cdsb = 0.0;

                    here->BSIM3v1cbgb = -CoxWL * dVgs_eff_dVg;
                    here->BSIM3v1cbdb = 0.0;
                    here->BSIM3v1cbsb = -here->BSIM3v1cgsb;

                }
		else if (Vgst <= 0.0)
		{   T1 = 0.5 * pParam->BSIM3v1k1;
	            T2 = sqrt(T1 * T1 + Arg1);
	            qgate = CoxWL * pParam->BSIM3v1k1 * (T2 - T1);
                    qbulk = -qgate;
                    qdrn = 0.0;

	            T0 = CoxWL * T1 / T2;
	            here->BSIM3v1cggb = T0 * dVgs_eff_dVg;
	            here->BSIM3v1cgdb = 0.0;
                    here->BSIM3v1cgsb = T0 * (dVbseff_dVb
				    - dVgs_eff_dVg);
   
                    here->BSIM3v1cdgb = 0.0;
                    here->BSIM3v1cddb = 0.0;
                    here->BSIM3v1cdsb = 0.0;

                    here->BSIM3v1cbgb = -here->BSIM3v1cggb;
                    here->BSIM3v1cbdb = 0.0;
                    here->BSIM3v1cbsb = -here->BSIM3v1cgsb;
                }
		else
		{   One_Third_CoxWL = CoxWL / 3.0;
                    Two_Third_CoxWL = 2.0 * One_Third_CoxWL;

                  AbulkCV = Abulk0 * pParam->BSIM3v1abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM3v1abulkCVfactor * dAbulk0_dVb;
	          Vdsat = Vgst / AbulkCV;
	          dVdsat_dVg = dVgs_eff_dVg / AbulkCV;
	          dVdsat_dVb = - (Vdsat * dAbulkCV_dVb + dVth_dVb)/ AbulkCV; 

                    if (model->BSIM3v1xpart > 0.5)
		    {   /* 0/100 Charge petition model */
		        if (Vdsat <= Vds)
		        {   /* saturation region */
	                    T1 = Vdsat / 3.0;
	                    qgate = CoxWL * (Vgs_eff - Vfb
			          - pParam->BSIM3v1phi - T1);
	                    T2 = -Two_Third_CoxWL * Vgst;
	                    qbulk = -(qgate + T2);
	                    qdrn = 0.0;

	                    here->BSIM3v1cggb = One_Third_CoxWL * (3.0
					    - dVdsat_dVg) * dVgs_eff_dVg;
	                    T2 = -One_Third_CoxWL * dVdsat_dVb;
	                    here->BSIM3v1cgsb = -(here->BSIM3v1cggb + T2);
        	            here->BSIM3v1cgdb = 0.0;
       
                            here->BSIM3v1cdgb = 0.0;
                            here->BSIM3v1cddb = 0.0;
                            here->BSIM3v1cdsb = 0.0;

	                    here->BSIM3v1cbgb = -(here->BSIM3v1cggb
					    - Two_Third_CoxWL * dVgs_eff_dVg);
	                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                    here->BSIM3v1cbsb = -(here->BSIM3v1cbgb + T3);
                            here->BSIM3v1cbdb = 0.0;
	                }
		        else
		        {   /* linear region */
			    Alphaz = Vgst / Vdsat;
	                    T1 = 2.0 * Vdsat - Vds;
	                    T2 = Vds / (3.0 * T1);
	                    T3 = T2 * Vds;
	                    T9 = 0.25 * CoxWL;
	                    T4 = T9 * Alphaz;
	                    T7 = 2.0 * Vds - T1 - 3.0 * T3;
	                    T8 = T3 - T1 - 2.0 * Vds;
	                    qgate = CoxWL * (Vgs_eff - Vfb 
			          - pParam->BSIM3v1phi - 0.5 * (Vds - T3));
	                    T10 = T4 * T8;
	                    qdrn = T4 * T7;
	                    qbulk = -(qgate + qdrn + T10);
  
                            T5 = T3 / T1;
                            here->BSIM3v1cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					    * dVgs_eff_dVg;
                            T11 = -CoxWL * T5 * dVdsat_dVb;
                            here->BSIM3v1cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                            here->BSIM3v1cgsb = -(here->BSIM3v1cggb + T11
                                            + here->BSIM3v1cgdb);
                            T6 = 1.0 / Vdsat;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);
                            T7 = T9 * T7;
                            T8 = T9 * T8;
                            T9 = 2.0 * T4 * (1.0 - 3.0 * T5);
                            here->BSIM3v1cdgb = (T7 * dAlphaz_dVg - T9
					    * dVdsat_dVg) * dVgs_eff_dVg;
                            T12 = T7 * dAlphaz_dVb - T9 * dVdsat_dVb;
                            here->BSIM3v1cddb = T4 * (3.0 - 6.0 * T2 - 3.0 * T5);
                            here->BSIM3v1cdsb = -(here->BSIM3v1cdgb + T12
                                            + here->BSIM3v1cddb);

                            T9 = 2.0 * T4 * (1.0 + T5);
                            T10 = (T8 * dAlphaz_dVg - T9 * dVdsat_dVg)
				* dVgs_eff_dVg;
                            T11 = T8 * dAlphaz_dVb - T9 * dVdsat_dVb;
                            T12 = T4 * (2.0 * T2 + T5 - 1.0); 
                            T0 = -(T10 + T11 + T12);

                            here->BSIM3v1cbgb = -(here->BSIM3v1cggb
					    + here->BSIM3v1cdgb + T10);
                            here->BSIM3v1cbdb = -(here->BSIM3v1cgdb 
					    + here->BSIM3v1cddb + T12);
                            here->BSIM3v1cbsb = -(here->BSIM3v1cgsb
					    + here->BSIM3v1cdsb + T0);
                        }
                    }
		    else if (model->BSIM3v1xpart < 0.5)
		    {   /* 40/60 Charge petition model */
		        if (Vds >= Vdsat)
		        {   /* saturation region */
	                    T1 = Vdsat / 3.0;
	                    qgate = CoxWL * (Vgs_eff - Vfb
			          - pParam->BSIM3v1phi - T1);
	                    T2 = -Two_Third_CoxWL * Vgst;
	                    qbulk = -(qgate + T2);
	                    qdrn = 0.4 * T2;

	                    here->BSIM3v1cggb = One_Third_CoxWL * (3.0 
					    - dVdsat_dVg) * dVgs_eff_dVg;
	                    T2 = -One_Third_CoxWL * dVdsat_dVb;
	                    here->BSIM3v1cgsb = -(here->BSIM3v1cggb + T2);
        	            here->BSIM3v1cgdb = 0.0;
       
			    T3 = 0.4 * Two_Third_CoxWL;
                            here->BSIM3v1cdgb = -T3 * dVgs_eff_dVg;
                            here->BSIM3v1cddb = 0.0;
			    T4 = T3 * dVth_dVb;
                            here->BSIM3v1cdsb = -(T4 + here->BSIM3v1cdgb);

	                    here->BSIM3v1cbgb = -(here->BSIM3v1cggb 
					    - Two_Third_CoxWL * dVgs_eff_dVg);
	                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                    here->BSIM3v1cbsb = -(here->BSIM3v1cbgb + T3);
                            here->BSIM3v1cbdb = 0.0;
	                }
		        else
		        {   /* linear region  */
			    Alphaz = Vgst / Vdsat;
			    T1 = 2.0 * Vdsat - Vds;
			    T2 = Vds / (3.0 * T1);
			    T3 = T2 * Vds;
			    T9 = 0.25 * CoxWL;
			    T4 = T9 * Alphaz;
			    qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM3v1phi
				  - 0.5 * (Vds - T3));

			    T5 = T3 / T1;
                            here->BSIM3v1cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					    * dVgs_eff_dVg;
                            tmp = -CoxWL * T5 * dVdsat_dVb;
                            here->BSIM3v1cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                            here->BSIM3v1cgsb = -(here->BSIM3v1cggb 
					+ here->BSIM3v1cgdb + tmp);

			    T6 = 1.0 / Vdsat;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);

			    T6 = 8.0 * Vdsat * Vdsat - 6.0 * Vdsat * Vds
			       + 1.2 * Vds * Vds;
			    T8 = T2 / T1;
			    T7 = Vds - T1 - T8 * T6;
			    qdrn = T4 * T7;
			    T7 *= T9;
			    tmp = T8 / T1;
			    tmp1 = T4 * (2.0 - 4.0 * tmp * T6
				 + T8 * (16.0 * Vdsat - 6.0 * Vds));

                            here->BSIM3v1cdgb = (T7 * dAlphaz_dVg - tmp1
					    * dVdsat_dVg) * dVgs_eff_dVg;
                            T10 = T7 * dAlphaz_dVb - tmp1 * dVdsat_dVb;
                            here->BSIM3v1cddb = T4 * (2.0 - (1.0 / (3.0 * T1
					    * T1) + 2.0 * tmp) * T6 + T8
					    * (6.0 * Vdsat - 2.4 * Vds));
                            here->BSIM3v1cdsb = -(here->BSIM3v1cdgb 
					    + T10 + here->BSIM3v1cddb);

			    T7 = 2.0 * (T1 + T3);
			    qbulk = -(qgate - T4 * T7);
			    T7 *= T9;
			    T0 = 4.0 * T4 * (1.0 - T5);
			    T12 = (-T7 * dAlphaz_dVg - here->BSIM3v1cdgb
				- T0 * dVdsat_dVg) * dVgs_eff_dVg;
			    T11 = -T7 * dAlphaz_dVb - T10 - T0 * dVdsat_dVb;
			    T10 = -4.0 * T4 * (T2 - 0.5 + 0.5 * T5) 
				- here->BSIM3v1cddb;
                            tmp = -(T10 + T11 + T12);

                            here->BSIM3v1cbgb = -(here->BSIM3v1cggb 
					    + here->BSIM3v1cdgb + T12);
                            here->BSIM3v1cbdb = -(here->BSIM3v1cgdb
					    + here->BSIM3v1cddb + T11);
                            here->BSIM3v1cbsb = -(here->BSIM3v1cgsb
					    + here->BSIM3v1cdsb + tmp);
                        }
                    }
		    else
		    {   /* 50/50 partitioning */
		        if (Vds >= Vdsat)
		        {   /* saturation region */
	                    T1 = Vdsat / 3.0;
	                    qgate = CoxWL * (Vgs_eff - Vfb
			          - pParam->BSIM3v1phi - T1);
	                    T2 = -Two_Third_CoxWL * Vgst;
	                    qbulk = -(qgate + T2);
	                    qdrn = 0.5 * T2;

	                    here->BSIM3v1cggb = One_Third_CoxWL * (3.0
					    - dVdsat_dVg) * dVgs_eff_dVg;
	                    T2 = -One_Third_CoxWL * dVdsat_dVb;
	                    here->BSIM3v1cgsb = -(here->BSIM3v1cggb + T2);
        	            here->BSIM3v1cgdb = 0.0;
       
                            here->BSIM3v1cdgb = -One_Third_CoxWL * dVgs_eff_dVg;
                            here->BSIM3v1cddb = 0.0;
			    T4 = One_Third_CoxWL * dVth_dVb;
                            here->BSIM3v1cdsb = -(T4 + here->BSIM3v1cdgb);

	                    here->BSIM3v1cbgb = -(here->BSIM3v1cggb 
					    - Two_Third_CoxWL * dVgs_eff_dVg);
	                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                    here->BSIM3v1cbsb = -(here->BSIM3v1cbgb + T3);
                            here->BSIM3v1cbdb = 0.0;
	                }
		        else
		        {   /* linear region */
			    Alphaz = Vgst / Vdsat;
			    T1 = 2.0 * Vdsat - Vds;
			    T2 = Vds / (3.0 * T1);
			    T3 = T2 * Vds;
			    T9 = 0.25 * CoxWL;
			    T4 = T9 * Alphaz;
			    qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM3v1phi
				  - 0.5 * (Vds - T3));

			    T5 = T3 / T1;
                            here->BSIM3v1cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					    * dVgs_eff_dVg;
                            tmp = -CoxWL * T5 * dVdsat_dVb;
                            here->BSIM3v1cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                            here->BSIM3v1cgsb = -(here->BSIM3v1cggb 
					+ here->BSIM3v1cgdb + tmp);

			    T6 = 1.0 / Vdsat;
                            dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                            dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);

			    T7 = T1 + T3;
			    qdrn = -T4 * T7;
			    qbulk = - (qgate + qdrn + qdrn);
			    T7 *= T9;
			    T0 = T4 * (2.0 * T5 - 2.0);

                            here->BSIM3v1cdgb = (T0 * dVdsat_dVg - T7
					    * dAlphaz_dVg) * dVgs_eff_dVg;
			    T12 = T0 * dVdsat_dVb - T7 * dAlphaz_dVb;
                            here->BSIM3v1cddb = T4 * (1.0 - 2.0 * T2 - T5);
                            here->BSIM3v1cdsb = -(here->BSIM3v1cdgb + T12
                                        + here->BSIM3v1cddb);

                            here->BSIM3v1cbgb = -(here->BSIM3v1cggb
					+ 2.0 * here->BSIM3v1cdgb);
                            here->BSIM3v1cbdb = -(here->BSIM3v1cgdb
					+ 2.0 * here->BSIM3v1cddb);
                            here->BSIM3v1cbsb = -(here->BSIM3v1cgsb
					+ 2.0 * here->BSIM3v1cdsb);
                        }
	            }
		 }

          } 
	  else
	  {   if (Vbseff < 0.0)
	      {   VbseffCV = Vbseff;
                  dVbseffCV_dVb = 1.0;
              }
	      else
	      {   VbseffCV = pParam->BSIM3v1phi - Phis;
                  dVbseffCV_dVb = -dPhis_dVb;
              }

              CoxWL = model->BSIM3v1cox * pParam->BSIM3v1weffCV
		    * pParam->BSIM3v1leffCV;
	      Vfb = Vth - pParam->BSIM3v1phi - pParam->BSIM3v1k1 * sqrtPhis;

	      dVfb_dVb = dVth_dVb - pParam->BSIM3v1k1 * dsqrtPhis_dVb;
              dVfb_dVd = dVth_dVd;

              if ((VgstNVt > -EXP_THRESHOLD) && (VgstNVt < EXP_THRESHOLD))
	      {   ExpVgst *= ExpVgst;
                  Vgsteff = n * Vtm * log(1.0 + ExpVgst);
                  dVgsteff_dVg = ExpVgst / (1.0 + ExpVgst);
                  dVgsteff_dVd = -dVgsteff_dVg * (dVth_dVd + (Vgs_eff - Vth)
			       / n * dn_dVd) + Vgsteff / n * dn_dVd;
                  dVgsteff_dVb = -dVgsteff_dVg * (dVth_dVb + (Vgs_eff - Vth)
			       / n * dn_dVb) + Vgsteff / n * dn_dVb;
	          dVgsteff_dVg *= dVgs_eff_dVg;
              }

	      if (model->BSIM3v1capMod == 1)
	      {   Arg1 = Vgs_eff - VbseffCV - Vfb - Vgsteff;

                  if (Arg1 <= 0.0)
	          {   qgate = CoxWL * Arg1;
                      Cgg = CoxWL * (dVgs_eff_dVg - dVgsteff_dVg);
                      Cgd = -CoxWL * (dVfb_dVd + dVgsteff_dVd);
                      Cgb = -CoxWL * (dVfb_dVb + dVbseffCV_dVb + dVgsteff_dVb);
                  }
	          else
	          {   T0 = 0.5 * pParam->BSIM3v1k1;
		      T1 = sqrt(T0 * T0 + Arg1);
                      T2 = CoxWL * T0 / T1;
		      
                      qgate = CoxWL * pParam->BSIM3v1k1 * (T1 - T0);

                      Cgg = T2 * (dVgs_eff_dVg - dVgsteff_dVg);
                      Cgd = -T2 * (dVfb_dVd + dVgsteff_dVd);
                      Cgb = -T2 * (dVfb_dVb + dVbseffCV_dVb + dVgsteff_dVb);
                  }
	          qbulk = -qgate;
	          Cbg = -Cgg;
	          Cbd = -Cgd;
	          Cbb = -Cgb;

                  One_Third_CoxWL = CoxWL / 3.0;
                  Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
                  AbulkCV = Abulk0 * pParam->BSIM3v1abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM3v1abulkCVfactor * dAbulk0_dVb;
	          VdsatCV = Vgsteff / AbulkCV;
                  if (VdsatCV < Vds)
	          {   dVdsatCV_dVg = 1.0 / AbulkCV;
	              dVdsatCV_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;
		      T0 = Vgsteff - VdsatCV / 3.0;
                      dT0_dVg = 1.0 - dVdsatCV_dVg / 3.0;
                      dT0_dVb = -dVdsatCV_dVb / 3.0;
                      qgate += CoxWL * T0;
                      Cgg1 = CoxWL * dT0_dVg; 
                      Cgb1 = CoxWL * dT0_dVb + Cgg1 * dVgsteff_dVb;
                      Cgd1 = Cgg1 * dVgsteff_dVd;
		      Cgg1 *= dVgsteff_dVg;
		      Cgg += Cgg1;
		      Cgb += Cgb1;
		      Cgd += Cgd1;

		      T0 = VdsatCV - Vgsteff;
		      dT0_dVg = dVdsatCV_dVg - 1.0;
		      dT0_dVb = dVdsatCV_dVb;
                      qbulk += One_Third_CoxWL * T0;
                      Cbg1 = One_Third_CoxWL * dT0_dVg;
		      Cbb1 = One_Third_CoxWL * dT0_dVb + Cbg1 * dVgsteff_dVb;
		      Cbd1 = Cbg1 * dVgsteff_dVd;
		      Cbg1 *= dVgsteff_dVg;
		      Cbg += Cbg1;
		      Cbb += Cbb1;
                      Cbd += Cbd1;

                      if (model->BSIM3v1xpart > 0.5)
		          T0 = -Two_Third_CoxWL;
                      else if (model->BSIM3v1xpart < 0.5)
		          T0 = -0.4 * CoxWL;
                      else
		          T0 = -One_Third_CoxWL;

                      qsrc = T0 * Vgsteff;
                      Csg = T0 * dVgsteff_dVg;
                      Csb = T0 * dVgsteff_dVb;
                      Csd = T0 * dVgsteff_dVd;
		      Cgb *= dVbseff_dVb;
		      Cbb *= dVbseff_dVb;
		      Csb *= dVbseff_dVb;
                  }
	          else
	          {   T0 = AbulkCV * Vds;
                      T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.e-20);
		      T2 = Vds / T1;
		      T3 = T0 * T2;
                      dT3_dVg = -12.0 * T2 * T2 * AbulkCV;
                      dT3_dVd = 6.0 * T0 * (4.0 * Vgsteff - T0) / T1 / T1 - 0.5;
                      dT3_dVb = 12.0 * T2 * T2 * dAbulkCV_dVb * Vgsteff;

                      qgate += CoxWL * (Vgsteff - 0.5 * Vds + T3);
                      Cgg1 = CoxWL * (1.0 + dT3_dVg);
                      Cgb1 = CoxWL * dT3_dVb + Cgg1 * dVgsteff_dVb;
                      Cgd1 = CoxWL * dT3_dVd + Cgg1 * dVgsteff_dVd;
		      Cgg1 *= dVgsteff_dVg;
		      Cgg += Cgg1;
		      Cgb += Cgb1;
		      Cgd += Cgd1;

                      qbulk += CoxWL * (1.0 - AbulkCV) * (0.5 * Vds - T3);
		      Cbg1 = -CoxWL * ((1.0 - AbulkCV) * dT3_dVg);
		      Cbb1 = -CoxWL * ((1.0 - AbulkCV) * dT3_dVb
		           + (0.5 * Vds - T3) * dAbulkCV_dVb)
		           + Cbg1 * dVgsteff_dVb;
		      Cbd1 = -CoxWL * (1.0 - AbulkCV) * dT3_dVd
		           + Cbg1 * dVgsteff_dVd;
		      Cbg1 *= dVgsteff_dVg;
		      Cbg += Cbg1;
		      Cbb += Cbb1;
		      Cbd += Cbd1;

                      if (model->BSIM3v1xpart > 0.5)
		      {   /* 0/100 Charge petition model */
		          T1 = T1 + T1;
                          qsrc = -CoxWL * (0.5 * Vgsteff + 0.25 * T0
			       - T0 * T0 / T1);
		          Csg = -CoxWL * (0.5 + 24.0 * T0 * Vds / T1 / T1
			      * AbulkCV);
                          Csb = -CoxWL * (0.25 * Vds * dAbulkCV_dVb
			      - 12.0 * T0 * Vds / T1 / T1 * (4.0 * Vgsteff - T0)
			      * dAbulkCV_dVb) + Csg * dVgsteff_dVb;
		          Csd = -CoxWL * (0.25 * AbulkCV - 12.0 * AbulkCV * T0
			      / T1 / T1 * (4.0 * Vgsteff - T0))
			      + Csg * dVgsteff_dVd;
		          Csg *= dVgsteff_dVg;
                      }
		      else if (model->BSIM3v1xpart < 0.5)
		      {   /* 40/60 Charge petition model */
		          T1 = T1 / 12.0;
		          T2 = 0.5 * CoxWL / (T1 * T1);
		          T3 = Vgsteff * (2.0 * T0 * T0 / 3.0 + Vgsteff
			     * (Vgsteff - 4.0 * T0 / 3.0))
			     - 2.0 * T0 * T0 * T0 / 15.0;
		          qsrc = -T2 * T3;
		          T4 = 4.0 / 3.0 * Vgsteff * (Vgsteff - T0)
			     + 0.4 * T0 * T0;
		          Csg = -2.0 * qsrc / T1 - T2 * (Vgsteff * (3.0
			      * Vgsteff - 8.0 * T0 / 3.0)
			      + 2.0 * T0 * T0 / 3.0);
		          Csb = (qsrc / T1 * Vds + T2 * T4 * Vds) * dAbulkCV_dVb
		              + Csg * dVgsteff_dVb;
		          Csd = (qsrc / T1 + T2 * T4) * AbulkCV
			      + Csg * dVgsteff_dVd;
		          Csg *= dVgsteff_dVg;
                      }
		      else
		      {   /* 50/50 Charge petition model */
                          qsrc = -0.5 * (qgate + qbulk);
                          Csg = -0.5 * (Cgg1 + Cbg1);
                          Csb = -0.5 * (Cgb1 + Cbb1); 
                          Csd = -0.5 * (Cgd1 + Cbd1); 
                      }
		      Cgb *= dVbseff_dVb;
		      Cbb *= dVbseff_dVb;
		      Csb *= dVbseff_dVb;
                  }
                  qdrn = -(qgate + qbulk + qsrc);
                  here->BSIM3v1cggb = Cgg;
	          here->BSIM3v1cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM3v1cgdb = Cgd;
                  here->BSIM3v1cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM3v1cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM3v1cddb = -(Cgd + Cbd + Csd);
                  here->BSIM3v1cbgb = Cbg;
	          here->BSIM3v1cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM3v1cbdb = Cbd;
	      }
	      else if (model->BSIM3v1capMod == 2)

	      {   V3 = Vfb - Vgs_eff + VbseffCV - DELTA_3;
		  if (Vfb <= 0.0)
		  {   T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfb);
		      T2 = -DELTA_3 / T0;
		  }
		  else
		  {   T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfb);
		      T2 = DELTA_3 / T0;
		  }

		  T1 = 0.5 * (1.0 + V3 / T0);
		  Vfbeff = Vfb - 0.5 * (V3 + T0);
		  dVfbeff_dVd = (1.0 - T1 - T2) * dVfb_dVd;
		  dVfbeff_dVg = T1 * dVgs_eff_dVg;
		  dVfbeff_dVb = (1.0 - T1 - T2) * dVfb_dVb
			          - T1 * dVbseffCV_dVb;
		  Qac0 = CoxWL * (Vfbeff - Vfb);
		  dQac0_dVg = CoxWL * dVfbeff_dVg;
		  dQac0_dVd = CoxWL * (dVfbeff_dVd - dVfb_dVd);
		  dQac0_dVb = CoxWL * (dVfbeff_dVb - dVfb_dVb);

                  T0 = 0.5 * pParam->BSIM3v1k1;
		  T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
                  if (pParam->BSIM3v1k1 == 0.0)
                  {   T1 = 0.0;
                      T2 = 0.0;
                  }
		  else if (T3 < 0.0)
		  {   T1 = T0 + T3 / pParam->BSIM3v1k1;
                      T2 = CoxWL;
		  }
		  else
		  {   T1 = sqrt(T0 * T0 + T3);
                      T2 = CoxWL * T0 / T1;
		  }

		  Qsub0 = CoxWL * pParam->BSIM3v1k1 * (T1 - T0);

                  dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
                  dQsub0_dVd = -T2 * (dVfbeff_dVd + dVgsteff_dVd);
                  dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb 
                             + dVgsteff_dVb);

                  One_Third_CoxWL = CoxWL / 3.0;
                  Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
                  AbulkCV = Abulk0 * pParam->BSIM3v1abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM3v1abulkCVfactor * dAbulk0_dVb;
	          VdsatCV = Vgsteff / AbulkCV;

		  V4 = VdsatCV - Vds - DELTA_4;
		  T0 = sqrt(V4 * V4 + 4.0 * DELTA_4 * VdsatCV);
		  VdseffCV = VdsatCV - 0.5 * (V4 + T0);
		  T1 = 0.5 * (1.0 + V4 / T0);
		  T2 = DELTA_4 / T0;
		  T3 = (1.0 - T1 - T2) / AbulkCV;
		  dVdseffCV_dVg = T3;
		  dVdseffCV_dVd = T1;
		  dVdseffCV_dVb = -T3 * VdsatCV * dAbulkCV_dVb;

	          T0 = AbulkCV * VdseffCV;
                  T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1e-20);
		  T2 = VdseffCV / T1;
		  T3 = T0 * T2;

                  T4 = (1.0 - 12.0 * T2 * T2 * AbulkCV);
                  T5 = (6.0 * T0 * (4.0 * Vgsteff - T0) / (T1 * T1) - 0.5);
                  T6 = 12.0 * T2 * T2 * Vgsteff;

                  qgate = CoxWL * (Vgsteff - 0.5 * VdseffCV + T3);
                  Cgg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
                  Cgd1 = CoxWL * T5 * dVdseffCV_dVd + Cgg1 * dVgsteff_dVd;
                  Cgb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
		       + Cgg1 * dVgsteff_dVb;
		  Cgg1 *= dVgsteff_dVg;

		  T7 = 1.0 - AbulkCV;
                  qbulk = CoxWL * T7 * (0.5 * VdseffCV - T3);
		  T4 = -T7 * (T4 - 1.0);
		  T5 = -T7 * T5;
		  T6 = -(T7 * T6 + (0.5 * VdseffCV - T3));
                  Cbg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
                  Cbd1 = CoxWL * T5 * dVdseffCV_dVd + Cbg1 * dVgsteff_dVd;
                  Cbb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
		       + Cbg1 * dVgsteff_dVb;
		  Cbg1 *= dVgsteff_dVg;

                  if (model->BSIM3v1xpart > 0.5)
		  {   /* 0/100 Charge petition model */
		      T1 = T1 + T1;
                      qsrc = -CoxWL * (0.5 * Vgsteff + 0.25 * T0
			   - T0 * T0 / T1);
		      T7 = (4.0 * Vgsteff - T0) / (T1 * T1);
		      T4 = -(0.5 + 24.0 * T0 * T0 / (T1 * T1));
		      T5 = -(0.25 * AbulkCV - 12.0 * AbulkCV * T0 * T7);
                      T6 = -(0.25 * VdseffCV - 12.0 * T0 * VdseffCV * T7);
                      Csg = CoxWL * (T4 + T5 * dVdseffCV_dVg);
                      Csd = CoxWL * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd;
                      Csb = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
			  + Csg * dVgsteff_dVb;
		      Csg *= dVgsteff_dVg;
                  }
		  else if (model->BSIM3v1xpart < 0.5)
		  {   /* 40/60 Charge petition model */
		      T1 = T1 / 12.0;
		      T2 = 0.5 * CoxWL / (T1 * T1);
		      T3 = Vgsteff * (2.0 * T0 * T0 / 3.0 + Vgsteff
			 * (Vgsteff - 4.0 * T0 / 3.0))
			 - 2.0 * T0 * T0 * T0 / 15.0;
		      qsrc = -T2 * T3;
		      T7 = 4.0 / 3.0 * Vgsteff * (Vgsteff - T0)
			 + 0.4 * T0 * T0;
		      T4 = -2.0 * qsrc / T1 - T2 * (Vgsteff * (3.0
			 * Vgsteff - 8.0 * T0 / 3.0)
			 + 2.0 * T0 * T0 / 3.0);
		      T5 = (qsrc / T1 + T2 * T7) * AbulkCV;
		      T6 = (qsrc / T1 * VdseffCV + T2 * T7 * VdseffCV);
                      Csg = (T4 + T5 * dVdseffCV_dVg);
                      Csd = T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd;
                      Csb = (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
			  + Csg * dVgsteff_dVb;
		      Csg *= dVgsteff_dVg;
                  }
		  else
		  {   /* 50/50 Charge petition model */
                      qsrc = -0.5 * (qgate + qbulk);
                      Csg = -0.5 * (Cgg1 + Cbg1);
                      Csb = -0.5 * (Cgb1 + Cbb1); 
                      Csd = -0.5 * (Cgd1 + Cbd1); 
                  }

		  qgate += Qac0 + Qsub0;
		  qbulk -= (Qac0 + Qsub0);
                  qdrn = -(qgate + qbulk + qsrc);

		  Cgg = dQac0_dVg + dQsub0_dVg + Cgg1;
		  Cgd = dQac0_dVd + dQsub0_dVd + Cgd1;
		  Cgb = dQac0_dVb + dQsub0_dVb + Cgb1;

		  Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
		  Cbd = Cbd1 - dQac0_dVd - dQsub0_dVd;
		  Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;

		  Cgb *= dVbseff_dVb;
		  Cbb *= dVbseff_dVb;
		  Csb *= dVbseff_dVb;

                  here->BSIM3v1cggb = Cgg;
	          here->BSIM3v1cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM3v1cgdb = Cgd;
                  here->BSIM3v1cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM3v1cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM3v1cddb = -(Cgd + Cbd + Csd);
                  here->BSIM3v1cbgb = Cbg;
	          here->BSIM3v1cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM3v1cbdb = Cbd;

	      } 
/* Non-quasi-static Model */

              if (here->BSIM3v1nqsMod)
              {   
                  qcheq = -qbulk - qgate;
                  qbulk = qgate = qdrn = qsrc = 0.0;

                  here->BSIM3v1cqgb = -(here->BSIM3v1cggb + here->BSIM3v1cbgb);
                  here->BSIM3v1cqdb = -(here->BSIM3v1cgdb + here->BSIM3v1cbdb);
                  here->BSIM3v1cqsb = -(here->BSIM3v1cgsb + here->BSIM3v1cbsb);
                  here->BSIM3v1cqbb =  here->BSIM3v1cggb + here->BSIM3v1cgdb 
                                  + here->BSIM3v1cgsb + here->BSIM3v1cbgb 
                                  + here->BSIM3v1cbdb + here->BSIM3v1cbsb;

                  here->BSIM3v1cggb = here->BSIM3v1cgsb = here->BSIM3v1cgdb = 0.0;
                  here->BSIM3v1cdgb = here->BSIM3v1cdsb = here->BSIM3v1cddb = 0.0;
                  here->BSIM3v1cbgb = here->BSIM3v1cbsb = here->BSIM3v1cbdb = 0.0;

	          T0 = pParam->BSIM3v1leffCV * pParam->BSIM3v1leffCV;
                  here->BSIM3v1tconst = pParam->BSIM3v1u0temp * pParam->BSIM3v1elm
				    / CoxWL / T0;

                  if (qcheq == 0.0)
		      here->BSIM3v1tconst = 0.0;
                  else if (qcheq < 0.0)
		      here->BSIM3v1tconst = -here->BSIM3v1tconst;

                  gtau_drift = fabs(here->BSIM3v1tconst * qcheq);
                  gtau_diff = 16.0 * pParam->BSIM3v1u0temp * model->BSIM3v1vtm / T0;
 
                  here->BSIM3v1gtau =  gtau_drift + gtau_diff;

                  *(ckt->CKTstate0 + here->BSIM3v1qcheq) = qcheq;
 
                  if (ckt->CKTmode & MODEINITTRAN)
                     *(ckt->CKTstate1 + here->BSIM3v1qcheq) =
                                   *(ckt->CKTstate0 + here->BSIM3v1qcheq);
              
                  error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v1qcheq);
                  if (error) return (error);
              }
              else
              {   here->BSIM3v1cqgb = here->BSIM3v1cqdb = here->BSIM3v1cqsb 
                                  = here->BSIM3v1cqbb = 0.0;
                  here->BSIM3v1gtau = 0.0; 
              }
          }

finished: /* returning Values to Calling Routine */
          /*
           *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
           */
          here->BSIM3v1cd = here->BSIM3v1mode * cdrain - here->BSIM3v1cbd;
          if (ChargeComputationNeeded)
	  {   /*  charge storage elements
               *  bulk-drain and bulk-source depletion capacitances
               *  czbd : zero bias drain junction capacitance
               *  czbs : zero bias source junction capacitance
               *  czbdsw: zero bias drain junction sidewall capacitance
			  along field oxide
               *  czbssw: zero bias source junction sidewall capacitance
			  along field oxide
	       *  czbdswg: zero bias drain junction sidewall capacitance
			   along gate side
	       *  czbsswg: zero bias source junction sidewall capacitance
			   along gate side
               */

              czbd = model->BSIM3v1unitAreaJctCap * here->BSIM3v1drainArea;
              czbs = model->BSIM3v1unitAreaJctCap * here->BSIM3v1sourceArea;
              if (here->BSIM3v1drainPerimeter < pParam->BSIM3v1weff)
              { 
              czbdswg = model->BSIM3v1unitLengthGateSidewallJctCap 
		     * here->BSIM3v1drainPerimeter;
              czbdsw = 0.0;
              }
              else
              {
              czbdsw = model->BSIM3v1unitLengthSidewallJctCap 
		     * (here->BSIM3v1drainPerimeter - pParam->BSIM3v1weff);
	      czbdswg = model->BSIM3v1unitLengthGateSidewallJctCap
		      *  pParam->BSIM3v1weff;
              }
              if (here->BSIM3v1sourcePerimeter < pParam->BSIM3v1weff)
              {
              czbssw = 0.0; 
	      czbsswg = model->BSIM3v1unitLengthGateSidewallJctCap
		        * here->BSIM3v1sourcePerimeter;
              }
              else
              {
              czbssw = model->BSIM3v1unitLengthSidewallJctCap 
		     * (here->BSIM3v1sourcePerimeter - pParam->BSIM3v1weff);
	      czbsswg = model->BSIM3v1unitLengthGateSidewallJctCap
		      *  pParam->BSIM3v1weff;
              }
              PhiB = model->BSIM3v1bulkJctPotential;
              PhiBSW = model->BSIM3v1sidewallJctPotential;
	      PhiBSWG = model->BSIM3v1GatesidewallJctPotential;
              MJ = model->BSIM3v1bulkJctBotGradingCoeff;
              MJSW = model->BSIM3v1bulkJctSideGradingCoeff;
	      MJSWG = model->BSIM3v1bulkJctGateSideGradingCoeff;

              /* Source Bulk Junction */
	      if (vbs == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM3v1qbs) = 0.0;
                  here->BSIM3v1capbs = czbs + czbssw + czbsswg;
	      }
	      else if (vbs < 0.0)
	      {   if (czbs > 0.0)
		  {   arg = 1.0 - vbs / PhiB;
		      if (MJ == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJ * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbs) = PhiB * czbs 
					* (1.0 - arg * sarg) / (1.0 - MJ);
		      here->BSIM3v1capbs = czbs * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM3v1qbs) = 0.0;
		      here->BSIM3v1capbs = 0.0;
		  }
		  if (czbssw > 0.0)
		  {   arg = 1.0 - vbs / PhiBSW;
		      if (MJSW == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSW * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbs) += PhiBSW * czbssw
				    * (1.0 - arg * sarg) / (1.0 - MJSW);
                      here->BSIM3v1capbs += czbssw * sarg;
		  }
		  if (czbsswg > 0.0)
		  {   arg = 1.0 - vbs / PhiBSWG;
		      if (MJSWG == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWG * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbs) += PhiBSWG * czbsswg
				    * (1.0 - arg * sarg) / (1.0 - MJSWG);
                      here->BSIM3v1capbs += czbsswg * sarg;
		  }

              }
	      else
	      {   *(ckt->CKTstate0+here->BSIM3v1qbs) = vbs * (czbs + czbssw
			      + czbsswg) + vbs * vbs * (czbs * MJ * 0.5 / PhiB 
                              + czbssw * MJSW * 0.5 / PhiBSW
			      + czbsswg * MJSWG * 0.5 / PhiBSWG);
                  here->BSIM3v1capbs = czbs + czbssw + czbsswg + vbs * (czbs * MJ / 
			                           PhiB + czbssw * MJSW / PhiBSW + czbsswg * MJSWG / PhiBSWG);
              }

              /* Drain Bulk Junction */
	      if (vbd == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM3v1qbd) = 0.0;
                  here->BSIM3v1capbd = czbd + czbdsw + czbdswg;
	      }
	      else if (vbd < 0.0)
	      {   if (czbd > 0.0)
		  {   arg = 1.0 - vbd / PhiB;
		      if (MJ == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJ * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbd) = PhiB * czbd 
			           * (1.0 - arg * sarg) / (1.0 - MJ);
                      here->BSIM3v1capbd = czbd * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM3v1qbd) = 0.0;
                      here->BSIM3v1capbd = 0.0;
		  }
		  if (czbdsw > 0.0)
		  {   arg = 1.0 - vbd / PhiBSW;
		      if (MJSW == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSW * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbd) += PhiBSW * czbdsw 
			               * (1.0 - arg * sarg) / (1.0 - MJSW);
                      here->BSIM3v1capbd += czbdsw * sarg;
		  }
		  if (czbdswg > 0.0)
		  {   arg = 1.0 - vbd / PhiBSWG;
		      if (MJSWG == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWG * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v1qbd) += PhiBSWG * czbdswg
				    * (1.0 - arg * sarg) / (1.0 - MJSWG);
                      here->BSIM3v1capbd += czbdswg * sarg;
		  }
              }
	      else
	      {   *(ckt->CKTstate0+here->BSIM3v1qbd) = vbd * (czbd + czbdsw
			      + czbdswg) + vbd * vbd * (czbd * MJ * 0.5 / PhiB 
                              + czbdsw * MJSW * 0.5 / PhiBSW
			      + czbdswg * MJSWG * 0.5 / PhiBSWG);
                  here->BSIM3v1capbd = czbd + czbdsw + czbdswg + vbd * (czbd * MJ / 
			                           PhiB + czbdsw * MJSW / PhiBSW + czbdswg * MJSWG / PhiBSWG);
              }
          }

          /*
           *  check convergence
           */
          if ((here->BSIM3v1off == 0) || (!(ckt->CKTmode & MODEINITFIX)))
	  {   if (Check == 1)
	      {   ckt->CKTnoncon++;
#ifndef NEWCONV
              } 
	      else
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(here->BSIM3v1cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - here->BSIM3v1cd) >= tol)
		  {   ckt->CKTnoncon++;
                  }
		  else
		  {   tol = ckt->CKTreltol * MAX(fabs(cbhat),
			    fabs(here->BSIM3v1cbs + here->BSIM3v1cbd))
			    + ckt->CKTabstol;
                      if (fabs(cbhat - (here->BSIM3v1cbs + here->BSIM3v1cbd))
			  > tol)
		      {   ckt->CKTnoncon++;
                      }
                  }
#endif /* NEWCONV */
              }
          }
          *(ckt->CKTstate0 + here->BSIM3v1vbs) = vbs;
          *(ckt->CKTstate0 + here->BSIM3v1vbd) = vbd;
          *(ckt->CKTstate0 + here->BSIM3v1vgs) = vgs;
          *(ckt->CKTstate0 + here->BSIM3v1vds) = vds;

          /* bulk and channel charge plus overlaps */

          if (!ChargeComputationNeeded)
              goto line850; 
#ifndef NOBYPASS
line755:
#endif
          ag0 = ckt->CKTag[0];

	  if (model->BSIM3v1capMod == 0)
	  {
	      cgdo = pParam->BSIM3v1cgdo;
	      qgdo = pParam->BSIM3v1cgdo * vgd;
	      cgso = pParam->BSIM3v1cgso;
	      qgso = pParam->BSIM3v1cgso * vgs;
	  }
	  else if (model->BSIM3v1capMod == 1)
	  {   if (vgd < 0.0)
	      {   T1 = sqrt(1.0 - 4.0 * vgd / pParam->BSIM3v1ckappa);
	          cgdo = pParam->BSIM3v1cgdo + pParam->BSIM3v1weffCV
		       * pParam->BSIM3v1cgdl / T1;
	          qgdo = pParam->BSIM3v1cgdo * vgd - pParam->BSIM3v1weffCV * 0.5
		       * pParam->BSIM3v1cgdl * pParam->BSIM3v1ckappa * (T1 - 1.0);
	      }
	      else
	      {   cgdo = pParam->BSIM3v1cgdo + pParam->BSIM3v1weffCV
		       * pParam->BSIM3v1cgdl;
	          qgdo = (pParam->BSIM3v1weffCV * pParam->BSIM3v1cgdl
		       + pParam->BSIM3v1cgdo) * vgd;
	      }

	      if (vgs < 0.0)
	      {   T1 = sqrt(1.0 - 4.0 * vgs / pParam->BSIM3v1ckappa);
	          cgso = pParam->BSIM3v1cgso + pParam->BSIM3v1weffCV
		       * pParam->BSIM3v1cgsl / T1;
	          qgso = pParam->BSIM3v1cgso * vgs - pParam->BSIM3v1weffCV * 0.5
		       * pParam->BSIM3v1cgsl * pParam->BSIM3v1ckappa * (T1 - 1.0);
	      }
	      else
	      {   cgso = pParam->BSIM3v1cgso + pParam->BSIM3v1weffCV
		       * pParam->BSIM3v1cgsl;
	          qgso = (pParam->BSIM3v1weffCV * pParam->BSIM3v1cgsl
		       + pParam->BSIM3v1cgso) * vgs;
	      }
	  }
	  else
	  {   T0 = vgd + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);

	      T3 = pParam->BSIM3v1weffCV * pParam->BSIM3v1cgdl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM3v1ckappa);
	      cgdo = pParam->BSIM3v1cgdo + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgdo = (pParam->BSIM3v1cgdo + T3) * vgd - T3 * (T2
		   + 0.5 * pParam->BSIM3v1ckappa * (T4 - 1.0));

	      T0 = vgs + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);
	      T3 = pParam->BSIM3v1weffCV * pParam->BSIM3v1cgsl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM3v1ckappa);
	      cgso = pParam->BSIM3v1cgso + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgso = (pParam->BSIM3v1cgso + T3) * vgs - T3 * (T2
		   + 0.5 * pParam->BSIM3v1ckappa * (T4 - 1.0));
	  }

          if (here->BSIM3v1mode > 0)
	  {   gcdgb = (here->BSIM3v1cdgb - cgdo) * ag0;
              gcddb = (here->BSIM3v1cddb + here->BSIM3v1capbd + cgdo) * ag0;
              gcdsb = here->BSIM3v1cdsb * ag0;
              gcsgb = -(here->BSIM3v1cggb + here->BSIM3v1cbgb + here->BSIM3v1cdgb
	            + cgso) * ag0;
              gcsdb = -(here->BSIM3v1cgdb + here->BSIM3v1cbdb + here->BSIM3v1cddb)
		    * ag0;
              gcssb = (here->BSIM3v1capbs + cgso - (here->BSIM3v1cgsb
		    + here->BSIM3v1cbsb + here->BSIM3v1cdsb)) * ag0;
              gcggb = (here->BSIM3v1cggb + cgdo + cgso + pParam->BSIM3v1cgbo ) * ag0;
              gcgdb = (here->BSIM3v1cgdb - cgdo) * ag0;
              gcgsb = (here->BSIM3v1cgsb - cgso) * ag0;
              gcbgb = (here->BSIM3v1cbgb - pParam->BSIM3v1cgbo) * ag0;
              gcbdb = (here->BSIM3v1cbdb - here->BSIM3v1capbd) * ag0;
              gcbsb = (here->BSIM3v1cbsb - here->BSIM3v1capbs) * ag0;
 
              gcqgb = here->BSIM3v1cqgb * ag0;
              gcqdb = here->BSIM3v1cqdb * ag0;
              gcqsb = here->BSIM3v1cqsb * ag0;
              gcqbb = here->BSIM3v1cqbb * ag0;

              T0 = here->BSIM3v1tconst * qdef;
              here->BSIM3v1gtg = T0 * here->BSIM3v1cqgb;
              here->BSIM3v1gtb = T0 * here->BSIM3v1cqbb;
              here->BSIM3v1gtd = T0 * here->BSIM3v1cqdb;
              here->BSIM3v1gts = T0 * here->BSIM3v1cqsb;
 
              sxpart = 0.6;
              dxpart = 0.4;

              /* compute total terminal charge */
              qgd = qgdo;
              qgs = qgso;
              qgb = pParam->BSIM3v1cgbo * vgb;
              qgate += qgd + qgs + qgb;
              qbulk -= qgb;
              qdrn -= qgd;
              qsrc = -(qgate + qbulk + qdrn);
	  }
	  else
	  {   gcsgb = (here->BSIM3v1cdgb - cgso) * ag0;
              gcsdb = here->BSIM3v1cdsb * ag0;
              gcssb = (here->BSIM3v1cddb + here->BSIM3v1capbs + cgso) * ag0;

              gcdgb = -(here->BSIM3v1cggb + here->BSIM3v1cbgb + here->BSIM3v1cdgb
		    + cgdo) * ag0;
              gcdsb = -(here->BSIM3v1cgdb + here->BSIM3v1cbdb + here->BSIM3v1cddb)
		    * ag0;
              gcddb = (here->BSIM3v1capbd + cgdo - (here->BSIM3v1cgsb
		    + here->BSIM3v1cbsb + here->BSIM3v1cdsb)) * ag0;
              gcggb = (here->BSIM3v1cggb + cgdo + cgso + pParam->BSIM3v1cgbo ) * ag0;
              gcgdb = (here->BSIM3v1cgsb - cgdo) * ag0;
              gcgsb = (here->BSIM3v1cgdb - cgso) * ag0;
              gcbgb = (here->BSIM3v1cbgb - pParam->BSIM3v1cgbo) * ag0;
              gcbdb = (here->BSIM3v1cbsb - here->BSIM3v1capbd) * ag0;
              gcbsb = (here->BSIM3v1cbdb - here->BSIM3v1capbs) * ag0;
 
              gcqgb = here->BSIM3v1cqgb * ag0;
              gcqdb = here->BSIM3v1cqsb * ag0;
              gcqsb = here->BSIM3v1cqdb * ag0;
              gcqbb = here->BSIM3v1cqbb * ag0;

              T0 = here->BSIM3v1tconst * qdef;
              here->BSIM3v1gtg = T0 * here->BSIM3v1cqgb;
              here->BSIM3v1gtb = T0 * here->BSIM3v1cqbb;
              here->BSIM3v1gtd = T0 * here->BSIM3v1cqdb;
              here->BSIM3v1gts = T0 * here->BSIM3v1cqsb;
 
              dxpart = 0.6;
              sxpart = 0.4;

              /* compute total terminal charge */
              qgd = qgdo;
              qgs = qgso;
              qgb = pParam->BSIM3v1cgbo * vgb;
              qgate += qgd + qgs + qgb;
              qbulk -= qgb;
              qsrc = qdrn - qgs;
              qdrn = -(qgate + qbulk + qsrc);
          }

          here->BSIM3v1cgdo = cgdo;
          here->BSIM3v1cgso = cgso;

/* added by Mansun 11/1/93 */


          if (here->BSIM3v1nqsMod)
          {   *(ckt->CKTstate0 + here->BSIM3v1qcdump) = qdef;

              if (ckt->CKTmode & MODEINITTRAN)
	             *(ckt->CKTstate1 + here->BSIM3v1qcdump) =
                              *(ckt->CKTstate0 + here->BSIM3v1qcdump);

              error = NIintegrate(ckt, &gqdef, &cqdef, 1.0, here->BSIM3v1qcdump);
              if (error) return (error);
          }
	  else
	  {   gqdef = cqdef = 0.0;
	  }

          if (ByPass) goto line860;

/* End added by Mansun 11/1/93 */
             
          *(ckt->CKTstate0 + here->BSIM3v1qg) = qgate;
          *(ckt->CKTstate0 + here->BSIM3v1qd) = qdrn
		    - *(ckt->CKTstate0 + here->BSIM3v1qbd);
          *(ckt->CKTstate0 + here->BSIM3v1qb) = qbulk
		    + *(ckt->CKTstate0 + here->BSIM3v1qbd)
		    + *(ckt->CKTstate0 + here->BSIM3v1qbs); 

          /* store small signal parameters */
          if (ckt->CKTmode & MODEINITSMSIG)
	  {   goto line1000;
          }
          if (!ChargeComputationNeeded)
              goto line850;
       
          if (ckt->CKTmode & MODEINITTRAN)
	  {   *(ckt->CKTstate1 + here->BSIM3v1qb) =
                    *(ckt->CKTstate0 + here->BSIM3v1qb);
              *(ckt->CKTstate1 + here->BSIM3v1qg) =
                    *(ckt->CKTstate0 + here->BSIM3v1qg);
              *(ckt->CKTstate1 + here->BSIM3v1qd) =
                    *(ckt->CKTstate0 + here->BSIM3v1qd);
          }
       
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v1qb);
          if (error)
	      return(error);
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v1qg);
          if (error)
	      return(error);
          error = NIintegrate(ckt,&geq, &ceq, 0.0, here->BSIM3v1qd);
          if (error)
	      return(error);
      
          goto line860;

line850:
          /* initialize to zero charge conductance and current */
          ceqqg = ceqqb = ceqqd = 0.0;

          cqcheq = cqdef = 0.0;

          gcdgb = gcddb = gcdsb = 0.0;
          gcsgb = gcsdb = gcssb = 0.0;
          gcggb = gcgdb = gcgsb = 0.0;
          gcbgb = gcbdb = gcbsb = 0.0;

          gcqgb = gcqdb = gcqsb = gcqbb = 0.0;
          here->BSIM3v1gtg = here->BSIM3v1gtd = here->BSIM3v1gts = here->BSIM3v1gtb = 0.0;
          gqdef = 0.0;
          sxpart = (1.0 - (dxpart = (here->BSIM3v1mode > 0) ? 0.4 : 0.6));
          if (here->BSIM3v1nqsMod)
              here->BSIM3v1gtau = 16.0 * pParam->BSIM3v1u0temp * model->BSIM3v1vtm 
                              / Leff / Leff;
	  else
              here->BSIM3v1gtau = 0.0;

          goto line900;
            
line860:
          /* evaluate equivalent charge current */

          cqgate = *(ckt->CKTstate0 + here->BSIM3v1cqg);
          cqbulk = *(ckt->CKTstate0 + here->BSIM3v1cqb);
          cqdrn = *(ckt->CKTstate0 + here->BSIM3v1cqd);

          ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs
                  + (here->BSIM3v1gtg * vgb - here->BSIM3v1gtd * vbd - here->BSIM3v1gts * vbs);
          ceqqb = cqbulk - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs;
          ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs
                  - dxpart * (here->BSIM3v1gtg * vgb - here->BSIM3v1gtd * vbd 
                  - here->BSIM3v1gts * vbs);
 
	     cqcheq = *(ckt->CKTstate0 + here->BSIM3v1cqcheq)
                 - (gcqgb * vgb - gcqdb * vbd  - gcqsb * vbs)
                 + (here->BSIM3v1gtg * vgb - here->BSIM3v1gtd * vbd  - here->BSIM3v1gts * vbs);

          if (ckt->CKTmode & MODEINITTRAN)
	  {   *(ckt->CKTstate1 + here->BSIM3v1cqb) =  
                    *(ckt->CKTstate0 + here->BSIM3v1cqb);
              *(ckt->CKTstate1 + here->BSIM3v1cqg) =  
                    *(ckt->CKTstate0 + here->BSIM3v1cqg);
              *(ckt->CKTstate1 + here->BSIM3v1cqd) =  
                    *(ckt->CKTstate0 + here->BSIM3v1cqd);
              *(ckt->CKTstate1 + here->BSIM3v1cqcheq) =  
                    *(ckt->CKTstate0 + here->BSIM3v1cqcheq);
              *(ckt->CKTstate1 + here->BSIM3v1cqcdump) =  
                    *(ckt->CKTstate0 + here->BSIM3v1cqcdump);
          }

          /*
           *  load current vector
           */
line900:

          if (here->BSIM3v1mode >= 0)
	  {   Gm = here->BSIM3v1gm;
              Gmbs = here->BSIM3v1gmbs;
              FwdSum = Gm + Gmbs;
              RevSum = 0.0;
              cdreq = model->BSIM3v1type * (cdrain - here->BSIM3v1gds * vds
		    - Gm * vgs - Gmbs * vbs);
              ceqbs = -here->BSIM3v1csub;
              ceqbd = 0.0;

              gbspsp = -here->BSIM3v1gbds - here->BSIM3v1gbgs - here->BSIM3v1gbbs;
              gbbdp = -here->BSIM3v1gbds;
              gbbsp = here->BSIM3v1gbds + here->BSIM3v1gbgs + here->BSIM3v1gbbs;
              gbspg = here->BSIM3v1gbgs;
              gbspb = here->BSIM3v1gbbs;
              gbspdp = here->BSIM3v1gbds;
              gbdpdp = 0.0;
              gbdpg = 0.0;
              gbdpb = 0.0;
              gbdpsp = 0.0;
          }
	  else
	  {   Gm = -here->BSIM3v1gm;
              Gmbs = -here->BSIM3v1gmbs;
              FwdSum = 0.0;
              RevSum = -(Gm + Gmbs);
              cdreq = -model->BSIM3v1type * (cdrain + here->BSIM3v1gds * vds
                    + Gm * vgd + Gmbs * vbd);
              ceqbs = 0.0;
              ceqbd = -here->BSIM3v1csub;

              gbspsp = 0.0;
              gbbdp = here->BSIM3v1gbds + here->BSIM3v1gbgs + here->BSIM3v1gbbs;
              gbbsp = -here->BSIM3v1gbds;
              gbspg = 0.0;
              gbspb = 0.0;
              gbspdp = 0.0;
              gbdpdp = -here->BSIM3v1gbds - here->BSIM3v1gbgs - here->BSIM3v1gbbs;
              gbdpg = here->BSIM3v1gbgs;
              gbdpb = here->BSIM3v1gbbs;
              gbdpsp = here->BSIM3v1gbds;
          }

	   if (model->BSIM3v1type > 0)
	   {   ceqbs += (here->BSIM3v1cbs - (here->BSIM3v1gbs  - ckt->CKTgmin) * vbs);
               ceqbd += (here->BSIM3v1cbd - (here->BSIM3v1gbd  - ckt->CKTgmin) * vbd);
               /*
               ceqqg = ceqqg;
               ceqqb = ceqqb;
               ceqqd = ceqqd;
               cqcheq = cqcheq;
               */
	   }
	   else
	   {   ceqbs = -ceqbs - (here->BSIM3v1cbs - (here->BSIM3v1gbs
		     - ckt->CKTgmin) * vbs); 
               ceqbd = -ceqbd - (here->BSIM3v1cbd - (here->BSIM3v1gbd
		     - ckt->CKTgmin) * vbd);
               ceqqg = -ceqqg;
               ceqqb = -ceqqb;
               ceqqd = -ceqqd;
               cqcheq = -cqcheq;
	   }

           (*(ckt->CKTrhs + here->BSIM3v1gNode) -= m * ceqqg);
           (*(ckt->CKTrhs + here->BSIM3v1bNode) -= m * (ceqbs + ceqbd + ceqqb));
           (*(ckt->CKTrhs + here->BSIM3v1dNodePrime) += m * (ceqbd - cdreq - ceqqd));
           (*(ckt->CKTrhs + here->BSIM3v1sNodePrime) += m * (cdreq + ceqbs + ceqqg
						  + ceqqb + ceqqd));

           *(ckt->CKTrhs + here->BSIM3v1qNode) += m * (cqcheq - cqdef);

           /*
            *  load y matrix
            */

           (*(here->BSIM3v1DdPtr) += m * here->BSIM3v1drainConductance);
           (*(here->BSIM3v1GgPtr) += m * (gcggb - here->BSIM3v1gtg));
           (*(here->BSIM3v1SsPtr) += m * here->BSIM3v1sourceConductance);
           (*(here->BSIM3v1BbPtr) += m * ((here->BSIM3v1gbd + here->BSIM3v1gbs
                               - gcbgb - gcbdb - gcbsb) - here->BSIM3v1gbbs));
           (*(here->BSIM3v1DPdpPtr) += m * ((here->BSIM3v1drainConductance
                                 + here->BSIM3v1gds + here->BSIM3v1gbd
                                 + RevSum + gcddb) + dxpart * here->BSIM3v1gtd + gbdpdp));
           (*(here->BSIM3v1SPspPtr) += m * ((here->BSIM3v1sourceConductance
                                 + here->BSIM3v1gds + here->BSIM3v1gbs
                                 + FwdSum + gcssb) + sxpart * here->BSIM3v1gts + gbspsp));
           (*(here->BSIM3v1DdpPtr) -= m * here->BSIM3v1drainConductance);
           (*(here->BSIM3v1GbPtr) -= m * (gcggb + gcgdb + gcgsb + here->BSIM3v1gtb));
           (*(here->BSIM3v1GdpPtr) += m * (gcgdb - here->BSIM3v1gtd));
           (*(here->BSIM3v1GspPtr) += m * (gcgsb - here->BSIM3v1gts));
           (*(here->BSIM3v1SspPtr) -= m * here->BSIM3v1sourceConductance);
           (*(here->BSIM3v1BgPtr) += m * (gcbgb - here->BSIM3v1gbgs));
           (*(here->BSIM3v1BdpPtr) += m * (gcbdb - here->BSIM3v1gbd + gbbdp));
           (*(here->BSIM3v1BspPtr) += m * (gcbsb - here->BSIM3v1gbs + gbbsp));
           (*(here->BSIM3v1DPdPtr) -= m * here->BSIM3v1drainConductance);
           (*(here->BSIM3v1DPgPtr) += m * ((Gm + gcdgb) + dxpart *
	                                  here->BSIM3v1gtg + gbdpg));
           (*(here->BSIM3v1DPbPtr) -= m * ((here->BSIM3v1gbd - Gmbs + gcdgb + gcddb
                                          + gcdsb - dxpart * here->BSIM3v1gtb) - gbdpb));
           (*(here->BSIM3v1DPspPtr) -= m * ((here->BSIM3v1gds + FwdSum - gcdsb
				          - dxpart * here->BSIM3v1gts) - gbdpsp));
           (*(here->BSIM3v1SPgPtr) += m * (gcsgb - Gm + sxpart *
	                                   here->BSIM3v1gtg + gbspg));
           (*(here->BSIM3v1SPsPtr) -= m * here->BSIM3v1sourceConductance);
           (*(here->BSIM3v1SPbPtr) -= m * ((here->BSIM3v1gbs + Gmbs + gcsgb + gcsdb
                                            + gcssb - sxpart * here->BSIM3v1gtb) - gbspb));
           (*(here->BSIM3v1SPdpPtr) -= m * ((here->BSIM3v1gds + RevSum - gcsdb
                                             - sxpart * here->BSIM3v1gtd - here->BSIM3v1gbd) - gbspdp));
 
           *(here->BSIM3v1QqPtr) += m * ((gqdef + here->BSIM3v1gtau));
 
           *(here->BSIM3v1DPqPtr) += m * ((dxpart * here->BSIM3v1gtau));
           *(here->BSIM3v1SPqPtr) += m * ((sxpart * here->BSIM3v1gtau));
           *(here->BSIM3v1GqPtr) -= m * here->BSIM3v1gtau;
 
           *(here->BSIM3v1QgPtr) += m * (-gcqgb + here->BSIM3v1gtg);
           *(here->BSIM3v1QdpPtr) += m * (-gcqdb + here->BSIM3v1gtd);
           *(here->BSIM3v1QspPtr) += m * (-gcqsb + here->BSIM3v1gts);
           *(here->BSIM3v1QbPtr) += m * (-gcqbb + here->BSIM3v1gtb);

line1000:  ;

     }  /* End of Mosfet Instance */
}   /* End of Model Instance */

return(OK);
}



