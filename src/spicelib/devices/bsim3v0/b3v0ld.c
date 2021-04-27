/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: b3ld.c          1/3/92
Modified by Mansun Chan  (1995)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
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
BSIM3v0load(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;
double SourceSatCurrent, DrainSatCurrent;
double ag0, qgd, qgs, qgb, von, cbhat = 0.0, VgstNVt, ExpVgst = 0.0;
double cdrain, cdhat = 0.0, cdreq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg, ceq, geq;
double czbd, czbdsw, czbs, czbssw, evbd, evbs, arg, sarg;
double delvbd, delvbs, delvds, delvgd, delvgs;
double Vfbeff, dVfbeff_dVg, dVfbeff_dVd, dVfbeff_dVb, V3, V4;
double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb, gcgdb, gcggb, gcgsb, gcsdb;
#ifndef NEWCONV
double tol;
#endif
double gcsgb, gcssb, PhiB, PhiBSW, MJ, MJSW;
double vbd, vbs, vds, vgb, vgd, vgs, vgdo;
#ifndef PREDICTOR
double xfact;
#endif
double qgate = 0.0, qbulk = 0.0, qdrn = 0.0, qsrc, cqgate, cqbulk, cqdrn;
double Vds, Vgs, Vbs, Gmbs, FwdSum, RevSum;
double Vgs_eff, Vfb, dVfb_dVb, dVfb_dVd;
double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth, dVth_dVb, dVth_dVd;
double Vgst, dVgs_eff_dVg;
double n, dn_dVb, Vtm;
double ExpArg;
double Denomi, dDenomi_dVg, dDenomi_dVd, dDenomi_dVb;
double ueff, dueff_dVg, dueff_dVd, dueff_dVb; 
double Esat, Vdsat;
double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb;
double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, Vasat;
double dVasat_dVg, dVasat_dVb, dVasat_dVd, Va, dVa_dVd, dVa_dVg, dVa_dVb; 
double Vbseff, dVbseff_dVb, VbseffCV, dVbseffCV_dVb; 
double Arg1, One_Third_CoxWL, Two_Third_CoxWL, CoxWL; 
double T0, dT0_dVg, dT0_dVd, dT0_dVb;
double T1, dT1_dVg, dT1_dVd, dT1_dVb;
double T2, dT2_dVg, dT2_dVd, dT2_dVb;
double T3, dT3_dVg, dT3_dVd, dT3_dVb;
double T4, dT4_dVg, dT4_dVd, dT4_dVb;
double T5;
double T6;
double T7;
double T8;
double T10;
double Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb;
double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb;
double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb;
double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb, Delt_vth, dDelt_vth_dVb;
double Theta0, dTheta0_dVb;
double TempRatio, tmp1, tmp2, tmp3, tmp4;
double DIBL_Sft, dDIBL_Sft_dVd, Pmos_factor;
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
   
struct bsim3v0SizeDependParam *pParam;
int ByPass, Check, ChargeComputationNeeded = 0, error;
double m = 1.0;

for (; model != NULL; model = BSIM3v0nextModel(model))
{    for (here = BSIM3v0instances(model); here != NULL; 
          here = BSIM3v0nextInstance(here))
     {
          Check = 1;
          ByPass = 0;
	  pParam = here->pParam;
          if ((ckt->CKTmode & MODEINITSMSIG))
	  {   vbs = *(ckt->CKTstate0 + here->BSIM3v0vbs);
              vgs = *(ckt->CKTstate0 + here->BSIM3v0vgs);
              vds = *(ckt->CKTstate0 + here->BSIM3v0vds);
              qdef = *(ckt->CKTstate0 + here->BSIM3v0qcdump);
          }
	  else if ((ckt->CKTmode & MODEINITTRAN))
	  {   vbs = *(ckt->CKTstate1 + here->BSIM3v0vbs);
              vgs = *(ckt->CKTstate1 + here->BSIM3v0vgs);
              vds = *(ckt->CKTstate1 + here->BSIM3v0vds);
              qdef = *(ckt->CKTstate1 + here->BSIM3v0qcdump);
          }
	  else if ((ckt->CKTmode & MODEINITJCT) && !here->BSIM3v0off)
	  {   vds = model->BSIM3v0type * here->BSIM3v0icVDS;
              vgs = model->BSIM3v0type * here->BSIM3v0icVGS;
              vbs = model->BSIM3v0type * here->BSIM3v0icVBS;
              qdef = 0.0;

              if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) && 
                  ((ckt->CKTmode & (MODETRAN | MODEAC|MODEDCOP |
                   MODEDCTRANCURVE)) || (!(ckt->CKTmode & MODEUIC))))
	      {   vbs = 0.0;
                  vgs = pParam->BSIM3v0vth0 + 0.1;
                  vds = 0.1;
              }
          }
	  else if ((ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) && 
                  (here->BSIM3v0off)) 
          {    qdef = vbs = vgs = vds = 0.0;
	  }
          else
	  {
#ifndef PREDICTOR
               if ((ckt->CKTmode & MODEINITPRED))
	       {   xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
                   *(ckt->CKTstate0 + here->BSIM3v0vbs) = 
                         *(ckt->CKTstate1 + here->BSIM3v0vbs);
                   vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v0vbs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v0vbs)));
                   *(ckt->CKTstate0 + here->BSIM3v0vgs) = 
                         *(ckt->CKTstate1 + here->BSIM3v0vgs);
                   vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v0vgs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v0vgs)));
                   *(ckt->CKTstate0 + here->BSIM3v0vds) = 
                         *(ckt->CKTstate1 + here->BSIM3v0vds);
                   vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v0vds))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM3v0vds)));
                   *(ckt->CKTstate0 + here->BSIM3v0vbd) = 
                         *(ckt->CKTstate0 + here->BSIM3v0vbs)
                         - *(ckt->CKTstate0 + here->BSIM3v0vds);
                   qdef = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM3v0qcdump))
                           -(xfact * (*(ckt->CKTstate2 + here->BSIM3v0qcdump)));
               }
	       else
	       {
#endif /* PREDICTOR */
                   vbs = model->BSIM3v0type
                       * (*(ckt->CKTrhsOld + here->BSIM3v0bNode)
                       - *(ckt->CKTrhsOld + here->BSIM3v0sNodePrime));
                   vgs = model->BSIM3v0type
                       * (*(ckt->CKTrhsOld + here->BSIM3v0gNode) 
                       - *(ckt->CKTrhsOld + here->BSIM3v0sNodePrime));
                   vds = model->BSIM3v0type
                       * (*(ckt->CKTrhsOld + here->BSIM3v0dNodePrime)
                       - *(ckt->CKTrhsOld + here->BSIM3v0sNodePrime));
                   qdef = *(ckt->CKTrhsOld + here->BSIM3v0qNode);
#ifndef PREDICTOR
               }
#endif /* PREDICTOR */

               vbd = vbs - vds;
               vgd = vgs - vds;
               vgdo = *(ckt->CKTstate0 + here->BSIM3v0vgs)
                    - *(ckt->CKTstate0 + here->BSIM3v0vds);
               delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v0vbs);
               delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v0vbd);
               delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v0vgs);
               delvds = vds - *(ckt->CKTstate0 + here->BSIM3v0vds);
               delvgd = vgd - vgdo;

               if (here->BSIM3v0mode >= 0) 
	       {   cdhat = here->BSIM3v0cd - here->BSIM3v0gbd * delvbd 
                         + here->BSIM3v0gmbs * delvbs + here->BSIM3v0gm * delvgs
                         + here->BSIM3v0gds * delvds;
	       }
               else
	       {   cdhat = here->BSIM3v0cd - (here->BSIM3v0gbd - here->BSIM3v0gmbs)
                         * delvbd - here->BSIM3v0gm * delvgd 
                         + here->BSIM3v0gds * delvds;
               
	       }
               cbhat = here->BSIM3v0cbs + here->BSIM3v0cbd + here->BSIM3v0gbd
                     * delvbd + here->BSIM3v0gbs * delvbs;

#ifndef NOBYPASS
           /* following should be one big if connected by && all over
            * the place, but some C compilers can't handle that, so
            * we split it up here to let them digest it in stages
            */

               if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass))
               if ((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                   fabs(*(ckt->CKTstate0+here->BSIM3v0vbs))) + ckt->CKTvoltTol)))
               if ((fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                   fabs(*(ckt->CKTstate0+here->BSIM3v0vbd))) + ckt->CKTvoltTol)))
               if ((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                   fabs(*(ckt->CKTstate0+here->BSIM3v0vgs))) + ckt->CKTvoltTol)))
               if ((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                   fabs(*(ckt->CKTstate0+here->BSIM3v0vds))) + ckt->CKTvoltTol)))
               if ((fabs(cdhat - here->BSIM3v0cd) < ckt->CKTreltol 
                   * MAX(fabs(cdhat),fabs(here->BSIM3v0cd)) + ckt->CKTabstol))
	       {   tempv = MAX(fabs(cbhat),fabs(here->BSIM3v0cbs 
                         + here->BSIM3v0cbd)) + ckt->CKTabstol;
                   if ((fabs(cbhat - (here->BSIM3v0cbs + here->BSIM3v0cbd))
                       < ckt->CKTreltol * tempv))
		   {   /* bypass code */
                       vbs = *(ckt->CKTstate0 + here->BSIM3v0vbs);
                       vbd = *(ckt->CKTstate0 + here->BSIM3v0vbd);
                       vgs = *(ckt->CKTstate0 + here->BSIM3v0vgs);
                       vds = *(ckt->CKTstate0 + here->BSIM3v0vds);
                       qcheq = *(ckt->CKTstate0 + here->BSIM3v0qcheq);

                       vgd = vgs - vds;
                       vgb = vgs - vbs;

                       cdrain = here->BSIM3v0mode * (here->BSIM3v0cd
                              + here->BSIM3v0cbd); 
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
               von = here->BSIM3v0von;
               if (*(ckt->CKTstate0 + here->BSIM3v0vds) >= 0.0)
	       {   vgs = DEVfetlim(vgs, *(ckt->CKTstate0+here->BSIM3v0vgs), von);
                   vds = vgs - vgd;
                   vds = DEVlimvds(vds, *(ckt->CKTstate0 + here->BSIM3v0vds));
                   vgd = vgs - vds;

               }
	       else
	       {   vgd = DEVfetlim(vgd, vgdo, von);
                   vds = vgs - vgd;
                   vds = -DEVlimvds(-vds, -(*(ckt->CKTstate0+here->BSIM3v0vds)));
                   vgs = vgd + vds;
               }

               if (vds >= 0.0)
	       {   vbs = DEVpnjlim(vbs, *(ckt->CKTstate0 + here->BSIM3v0vbs),
                                   CONSTvt0, model->BSIM3v0vcrit, &Check);
                   vbd = vbs - vds;

               }
	       else
	       {   vbd = DEVpnjlim(vbd, *(ckt->CKTstate0 + here->BSIM3v0vbd),
                                   CONSTvt0, model->BSIM3v0vcrit, &Check); 
                   vbs = vbd + vds;
               }
          }

          /* determine DC current and derivatives */
          vbd = vbs - vds;
          vgd = vgs - vds;
          vgb = vgs - vbs;

          m = here->BSIM3v0m;

          SourceSatCurrent = here->BSIM3v0sourceArea*model->BSIM3v0jctSatCurDensity;
	  if (here->BSIM3v0sourceArea <= 0.0)
              SourceSatCurrent = 1.0e-14;
          if (SourceSatCurrent <= 0.0)
	  {   here->BSIM3v0gbs = ckt->CKTgmin;
              here->BSIM3v0cbs = here->BSIM3v0gbs * vbs;
          }
	  else if (vbs <= -0.5)
	  {   here->BSIM3v0gbs = ckt->CKTgmin;
              here->BSIM3v0cbs = -SourceSatCurrent + here->BSIM3v0gbs * vbs;
          }
	  else if (vbs < 0.5)
	  {   evbs = exp(vbs / CONSTvt0);
              here->BSIM3v0gbs = SourceSatCurrent * evbs / CONSTvt0 
                             + ckt->CKTgmin;
              here->BSIM3v0cbs = SourceSatCurrent * (evbs - 1.0) 
                             + ckt->CKTgmin * vbs;
          }
	  else
	  {   evbs = exp(0.5 / CONSTvt0);
              here->BSIM3v0gbs = SourceSatCurrent * evbs / CONSTvt0 
                             + ckt->CKTgmin;
              here->BSIM3v0cbs = SourceSatCurrent * (evbs - 1.0) 
                             + ckt->CKTgmin * 0.5;
          }

          DrainSatCurrent = here->BSIM3v0drainArea * model->BSIM3v0jctSatCurDensity;
	  if (here->BSIM3v0drainArea <= 0.0)
              DrainSatCurrent = 1.0e-14;
          if (DrainSatCurrent <= 0.0)
	  {   here->BSIM3v0gbd = ckt->CKTgmin;
              here->BSIM3v0cbd = here->BSIM3v0gbd * vbd;
          }
	  else if (vbd <= -0.5)
	  {   here->BSIM3v0gbd = ckt->CKTgmin;
              here->BSIM3v0cbd = -DrainSatCurrent + here->BSIM3v0gbd * vbd;
          }
	  else if (vbd < 0.5)
	  {   evbd = exp(vbd/CONSTvt0);
              here->BSIM3v0gbd = DrainSatCurrent * evbd / CONSTvt0 
                             + ckt->CKTgmin;
              here->BSIM3v0cbd = DrainSatCurrent * (evbd - 1.0) 
                             + ckt->CKTgmin * vbd;
          }
	  else
	  {   evbd = exp(0.5/CONSTvt0);
              here->BSIM3v0gbd = DrainSatCurrent * evbd / CONSTvt0 
                             + ckt->CKTgmin;
              here->BSIM3v0cbd = DrainSatCurrent * (evbd - 1.0) 
                             + ckt->CKTgmin * 0.5;
          }

          if (vds >= 0.0)
	  {   /* normal mode */
              here->BSIM3v0mode = 1;
              Vds = vds;
              Vgs = vgs;
              Vbs = vbs;
          }
	  else
	  {   /* inverse mode */
              here->BSIM3v0mode = -1;
              Vds = -vds;
              Vgs = vgd;
              Vbs = vbd;
          }

          ChargeComputationNeeded =  
                 ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
                 ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
                 ? 1 : 0;

	  T0 = Vbs - pParam->BSIM3v0vbsc - 0.001;
	  T1 = sqrt(T0 * T0 - 0.004 * pParam->BSIM3v0vbsc);
	  Vbseff = pParam->BSIM3v0vbsc + 0.5 * (T0 + T1);
	  dVbseff_dVb = 0.5 * (1.0 + T0 / T1);

          if (Vbseff > 0.0)
	  {   T0 = pParam->BSIM3v0phi / (pParam->BSIM3v0phi + Vbseff);
              Phis = pParam->BSIM3v0phi * T0;
              dPhis_dVb = -T0 * T0;
              sqrtPhis = pParam->BSIM3v0phis3 / (pParam->BSIM3v0phi + 0.5 * Vbseff);
              dsqrtPhis_dVb = -0.5 * sqrtPhis * sqrtPhis / pParam->BSIM3v0phis3;
          }
	  else
	  {   Phis = pParam->BSIM3v0phi - Vbseff;
              dPhis_dVb = -1.0;
              sqrtPhis = sqrt(Phis);
              dsqrtPhis_dVb = -0.5 / sqrtPhis; 
          }
          Xdep = pParam->BSIM3v0Xdep0 * sqrtPhis / pParam->BSIM3v0sqrtPhi;
          dXdep_dVb = (pParam->BSIM3v0Xdep0 / pParam->BSIM3v0sqrtPhi)
		    * dsqrtPhis_dVb;

          Leff = pParam->BSIM3v0leff;
/* Vth Calculation */
          if ((T1 = 1.0 + pParam->BSIM3v0dvt2 * Vbseff) < 1.0e-10)
	      T1 = 1.0E-10;
          if ((T2 = 1.0 + pParam->BSIM3v0dvt2w * Vbseff) < 1.0E-10)
	      T2 = 1.0E-10;

          T3 = sqrt(Xdep);
          lt1 = model->BSIM3v0factor1 * T3 * T1;
          dlt1_dVb = model->BSIM3v0factor1 * (0.5 / T3 * T1 * dXdep_dVb
                   + T3 * pParam->BSIM3v0dvt2);

          ltw = model->BSIM3v0factor1 * T3 * T2;
          dltw_dVb = model->BSIM3v0factor1 * (0.5 / T3 * T2 * dXdep_dVb
                   + T3 * pParam->BSIM3v0dvt2w);

          T0 = -0.5 * pParam->BSIM3v0dvt1 * Leff / lt1;
          if (T0 > -EXP_THRESHOLD)
          {   T1 = exp(T0);
              dT1_dVb = -T0 / lt1 * T1 * dlt1_dVb;
          }
          else
          {   T1 = MIN_EXP;
              dT1_dVb = 0.0;
          }

          Theta0 = T1 * (1.0 + 2.0 * T1);
          dTheta0_dVb = (1.0 + 4.0 * T1) * dT1_dVb;

          here->BSIM3v0thetavth = pParam->BSIM3v0dvt0 * Theta0;
          T0 = pParam->BSIM3v0vbi - pParam->BSIM3v0phi;
          Delt_vth = here->BSIM3v0thetavth * T0;
          dDelt_vth_dVb = pParam->BSIM3v0dvt0 * dTheta0_dVb * T0;

          T0 = -0.5 * pParam->BSIM3v0dvt1w * pParam->BSIM3v0weff * Leff / ltw;
          if (T0 > -EXP_THRESHOLD)
          {   T1 = exp(T0);
              dT1_dVb = -T0 / ltw * T1 * dltw_dVb;
          }
          else
          {   T1 = MIN_EXP;
              dT1_dVb = 0.0;
          }

          T2 = T1 * (1.0 + 2.0 * T1);
          dT2_dVb = (1.0 + 4.0 * T1) * dT1_dVb;

          T0 = pParam->BSIM3v0dvt0w * T2;
          T1 = pParam->BSIM3v0vbi - pParam->BSIM3v0phi;
          T2 = T0 * T1;
          dT2_dVb = pParam->BSIM3v0dvt0w * dT2_dVb * T1;

          TempRatio =  ckt->CKTtemp / model->BSIM3v0tnom - 1.0;
          T0 = sqrt(1.0 + pParam->BSIM3v0nlx / Leff);
          T1 = pParam->BSIM3v0k1 * (T0 - 1.0) * pParam->BSIM3v0sqrtPhi
             + (pParam->BSIM3v0kt1 + pParam->BSIM3v0kt1l / Leff
             + pParam->BSIM3v0kt2 * Vbseff) * TempRatio;
          tmp2 = model->BSIM3v0tox / (pParam->BSIM3v0weff
	       + pParam->BSIM3v0w0) * pParam->BSIM3v0phi;

          dDIBL_Sft_dVd = (pParam->BSIM3v0eta0 + pParam->BSIM3v0etab
                        * Vbseff) * pParam->BSIM3v0theta0vb0;
          DIBL_Sft = dDIBL_Sft_dVd * Vds;

          Vth = model->BSIM3v0type * pParam->BSIM3v0vth0 + pParam->BSIM3v0k1 
              * (sqrtPhis - pParam->BSIM3v0sqrtPhi) - pParam->BSIM3v0k2 
              * Vbseff - Delt_vth - T2 + (pParam->BSIM3v0k3 + pParam->BSIM3v0k3b
              * Vbseff) * tmp2 + T1 - DIBL_Sft;

          here->BSIM3v0von = Vth; 

          dVth_dVb = pParam->BSIM3v0k1 * dsqrtPhis_dVb - pParam->BSIM3v0k2
                   - dDelt_vth_dVb - dT2_dVb + pParam->BSIM3v0k3b * tmp2
                   - pParam->BSIM3v0etab * Vds * pParam->BSIM3v0theta0vb0
                   + pParam->BSIM3v0kt2 * TempRatio;
          dVth_dVd = -dDIBL_Sft_dVd; 

/* Calculate n */
          tmp2 = pParam->BSIM3v0nfactor * EPSSI / Xdep;
          tmp3 = pParam->BSIM3v0cdsc + pParam->BSIM3v0cdscb * Vbseff
               + pParam->BSIM3v0cdscd * Vds;
	  n = 1.0 + (tmp2 + tmp3 * Theta0 + pParam->BSIM3v0cit) / model->BSIM3v0cox;
	  if (n > 1.0)
	  {   dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM3v0cdscb * Theta0) / model->BSIM3v0cox;  
              dn_dVd = pParam->BSIM3v0cdscd * Theta0 / model->BSIM3v0cox;            
	  }
	  else
	  {   n = 1.0;
	      dn_dVb = dn_dVd = 0.0;  
	  }

/* Poly Gate Si Depletion Effect */
	  T0 = pParam->BSIM3v0vfb + pParam->BSIM3v0phi;
          if (model->BSIM3v0ngateGiven && (Vgs > T0))
	  {   T1 = 1.0e6 * Charge_q * EPSSI * pParam->BSIM3v0ngate
                 / (model->BSIM3v0cox * model->BSIM3v0cox);
              T4 = sqrt(1.0 + 2.0 * (Vgs - T0) / T1);
              T2 = T1 * (T4 - 1.0);
              T3 = 0.5 * T2 * T2 / T1;

              if (T3 < 1.12)
	      {   Vgs_eff = T0 + T2;
                  dVgs_eff_dVg = 1.0 / T4;
              }
	      else
	      {   Vgs_eff = Vgs - 1.12;
                  dVgs_eff_dVg = 1.0;
              }
          }
	  else
	  {   Vgs_eff = Vgs;
              dVgs_eff_dVg = 1.0;
          }
          Vgst = Vgs_eff - Vth;

/* Effective Vgst (Vgsteff) Calculation */
          Vtm = model->BSIM3v0vtm;

          T10 = 2.0 * n * Vtm;
          VgstNVt = Vgst / T10;
          if (VgstNVt < -EXP_THRESHOLD) 
	  {   T1 = T10 * MIN_EXP; 
	      dT1_dVg = dT1_dVd = dT1_dVb = 0.0;
	  }
          else if (VgstNVt > EXP_THRESHOLD)
	  {   T1 = Vgst;
              dT1_dVg = dVgs_eff_dVg;
              dT1_dVd = -dVth_dVd;
              dT1_dVb = -dVth_dVb;
          }
	  else
	  {   ExpVgst = exp(VgstNVt);
              T1 = T10 * log(1.0 + ExpVgst);
              dT1_dVg = ExpVgst / (1.0 + ExpVgst);
              dT1_dVb = -dT1_dVg * (dVth_dVb + Vgst / n * dn_dVb)
		      + T1 / n * dn_dVb; 
	      dT1_dVd = -dT1_dVg * (dVth_dVd + Vgst / n * dn_dVd)
		      + T1 / n * dn_dVd;
	      dT1_dVg *= dVgs_eff_dVg;
          }

	  T2 = model->BSIM3v0tox / (pParam->BSIM3v0weff + pParam->BSIM3v0w0);
          ExpArg = (2.0 * pParam->BSIM3v0voff - Vgst) / T10;
          if (ExpArg < -EXP_THRESHOLD)
	  {   T2 = 1.0;
              dT2_dVg = dT2_dVd = dT2_dVb = 0.0;

          }
	  else if (ExpArg > EXP_THRESHOLD)
	  {   T2 = 1.0 + 2.0 * n * model->BSIM3v0cox / pParam->BSIM3v0cdep0
		 * MAX_EXP;
              dT2_dVg = dT2_dVd = dT2_dVb = 0.0;
          }
	  else
	  {   dT2_dVg = -model->BSIM3v0cox / Vtm / pParam->BSIM3v0cdep0
		      * exp(ExpArg);
              T2 = 1.0 - T10 * dT2_dVg;
              dT2_dVd = -dT2_dVg * (dVth_dVd - 2.0 * Vtm
		      * ExpArg * dn_dVd) + (T2 - 1.0) / n * dn_dVd;
              dT2_dVb = -dT2_dVg * (dVth_dVb - 2.0 * Vtm
		      * ExpArg * dn_dVb) + (T2 - 1.0) / n * dn_dVb;
	      dT2_dVg *= dVgs_eff_dVg;
          }

          Vgsteff = T1 / T2;
          dVgsteff_dVg = (T2 * dT1_dVg - T1 * dT2_dVg) / (T2 * T2);
          dVgsteff_dVd = (T2 * dT1_dVd - T1 * dT2_dVd) / (T2 * T2);
          dVgsteff_dVb = (T2 * dT1_dVb - T1 * dT2_dVb) / (T2 * T2);

/* Calculate Effective Channel Geometry */
          Weff = pParam->BSIM3v0weff - 2.0 * (pParam->BSIM3v0dwg * Vgsteff 
               + pParam->BSIM3v0dwb * (sqrtPhis - pParam->BSIM3v0sqrtPhi)); 
          dWeff_dVg = -2.0 * pParam->BSIM3v0dwg;
          dWeff_dVb = -2.0 * pParam->BSIM3v0dwb * dsqrtPhis_dVb;

          if (Weff < 1.0e-8)
	  {   Weff = 1.0e-8;
              dWeff_dVg = dWeff_dVb = 0;
          }

          Rds = pParam->BSIM3v0rds0 * (1.0 + pParam->BSIM3v0prwg * Vgsteff 
              + pParam->BSIM3v0prwb * (sqrtPhis - pParam->BSIM3v0sqrtPhi));
	  if (Rds > 0.0)
	  {   dRds_dVg = pParam->BSIM3v0rds0 * pParam->BSIM3v0prwg;
              dRds_dVb = pParam->BSIM3v0rds0 * pParam->BSIM3v0prwb * dsqrtPhis_dVb;
	  }
	  else
	  {   Rds = dRds_dVg = dRds_dVb = 0.0;
	  }
	  
          WVCox = Weff * pParam->BSIM3v0vsattemp * model->BSIM3v0cox;
          WVCoxRds = WVCox * Rds; 

/* Calculate Abulk */
          T0 = 1.0 / (1.0 + pParam->BSIM3v0keta * Vbseff);
          dT0_dVb = -pParam->BSIM3v0keta * T0 * T0;

          T1 = 0.5 * pParam->BSIM3v0k1 / sqrtPhis;
          dT1_dVb = -T1 / sqrtPhis * dsqrtPhis_dVb;

          tmp1 = Leff + 2.0 * sqrt(pParam->BSIM3v0xj * Xdep);
          T5 = Leff / tmp1; 
          tmp2 = pParam->BSIM3v0a0 *T5;
          tmp3 = pParam->BSIM3v0weff + pParam->BSIM3v0b1; 
          tmp4 = pParam->BSIM3v0b0 / tmp3;
          T2 = tmp2 + tmp4;
          dT2_dVb = -tmp2 / tmp1 * sqrt(pParam->BSIM3v0xj / Xdep) * dXdep_dVb;
          T6 = T5 * T5;
          T7 = T5 * T6;
          Abulk0 = T0 * (1.0 + T1 * T2); 
                   if (Abulk0 < 0.01)
                      Abulk0= 0.01; 
          T8 = pParam->BSIM3v0ags * pParam->BSIM3v0a0 * T7;
          dAbulk_dVg = -T1 * T0 * T8;
          Abulk = Abulk0 + dAbulk_dVg * Vgsteff; 
                   if (Abulk < 0.01)
                        Abulk= 0.01;
                      
          dAbulk0_dVb = T0 * T1 * dT2_dVb + T0 * T2 * dT1_dVb   
               	      + (1.0 + T1 * T2) * dT0_dVb;
          dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (T1 * (3.0 * T0 * dT2_dVb
		     / tmp2 + dT0_dVb) + T0 * dT1_dVb); 
/* Mobility calculation */

          if (model->BSIM3v0mobMod == 1)
	  {   T0 = Vgsteff + Vth + Vth;
              T2 = pParam->BSIM3v0ua + pParam->BSIM3v0uc * Vbseff;
              T3 = T0 / model->BSIM3v0tox;
              Denomi = 1.0 + T3 * (T2
                     + pParam->BSIM3v0ub * T3);
              T1 = T2 / model->BSIM3v0tox + 2.0 * pParam->BSIM3v0ub * T3
                 / model->BSIM3v0tox;
              dDenomi_dVg = T1;
              dDenomi_dVd = T1 * 2.0 * dVth_dVd;
              dDenomi_dVb = T1 * 2.0 * dVth_dVb + pParam->BSIM3v0uc * T3;
          }
	  else if (model->BSIM3v0mobMod == 2)
	  {   Denomi = 1.0 + Vgsteff / model->BSIM3v0tox * (pParam->BSIM3v0ua
		     + pParam->BSIM3v0uc * Vbseff + pParam->BSIM3v0ub * Vgsteff
                     / model->BSIM3v0tox);
              T1 = (pParam->BSIM3v0ua + pParam->BSIM3v0uc * Vbseff) / model->BSIM3v0tox
                 + 2.0 * pParam->BSIM3v0ub / (model->BSIM3v0tox * model->BSIM3v0tox)
		 * Vgsteff;
              dDenomi_dVg = T1;
              dDenomi_dVd = 0.0;
              dDenomi_dVb = Vgsteff * pParam->BSIM3v0uc / model->BSIM3v0tox; 
          }
	  else
	  {   T0 = Vgsteff + Vth + Vth;
              T2 = 1.0 + pParam->BSIM3v0uc * Vbseff;
              T3 = T0 / model->BSIM3v0tox;
              T4 = T3 * (pParam->BSIM3v0ua + pParam->BSIM3v0ub * T3);
              Denomi = 1.0 + T4 * T2;

              T1 = (pParam->BSIM3v0ua / model->BSIM3v0tox + 2.0 * pParam->BSIM3v0ub
		 * T3 / model->BSIM3v0tox) * T2;
              dDenomi_dVg = T1;
              dDenomi_dVd = T1 * 2.0 * dVth_dVd;
              dDenomi_dVb = T1 * 2.0 * dVth_dVb + pParam->BSIM3v0uc * T4;
          }

          here->BSIM3v0ueff = ueff = pParam->BSIM3v0u0temp / Denomi;
          dueff_dVg = -ueff / Denomi * dDenomi_dVg;
          dueff_dVd = -ueff / Denomi * dDenomi_dVd;
          dueff_dVb = -ueff / Denomi * dDenomi_dVb;


/* Saturation Drain Voltage  Vdsat */
          Esat = 2.0 * pParam->BSIM3v0vsattemp / ueff;
          EsatL = Esat * Leff;
          T0 = -EsatL /ueff;
          dEsatL_dVg = T0 * dueff_dVg;
          dEsatL_dVd = T0 * dueff_dVd;
          dEsatL_dVb = T0 * dueff_dVb;
  
          a1 = pParam->BSIM3v0a1;
          if ((Pmos_factor = a1 * Vgsteff + pParam->BSIM3v0a2) > 1.0)
          {   Pmos_factor = 1.0; 
              a1 = 0.0;
          }

          Vgst2Vtm = Vgsteff + 2.0 * Vtm;
          if ((Rds == 0.0) && (Pmos_factor == 1.0))
          {   T0 = 1.0 / (Abulk * EsatL + Vgst2Vtm);
              here->BSIM3v0vdsat = Vdsat = EsatL * Vgst2Vtm * T0;
                           
              dT0_dVg = -(Abulk * dEsatL_dVg +EsatL*dAbulk_dVg+ 1.0) * T0 * T0;
              dT0_dVd = -(Abulk * dEsatL_dVd) * T0 * T0; 
              dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T0 * T0;   

              dVdsat_dVg = EsatL * Vgst2Vtm * dT0_dVg + EsatL * T0
                         + Vgst2Vtm * T0 * dEsatL_dVg;
              dVdsat_dVd = EsatL * Vgst2Vtm * dT0_dVd
			 + Vgst2Vtm * T0 * dEsatL_dVd;
              dVdsat_dVb = EsatL * Vgst2Vtm * dT0_dVb
			 + Vgst2Vtm * T0 * dEsatL_dVb;   
          }
          else
          {   tmp1 = a1 / (Pmos_factor * Pmos_factor);
              if (Rds > 0)
              {   tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff;
                  tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff;
              }
              else
              {   tmp2 = dWeff_dVg / Weff;
                  tmp3 = dWeff_dVb / Weff;
              }
                          
              T0 = 2.0 * Abulk * (Abulk * WVCoxRds - 1.0 + 1.0 / Pmos_factor); 
              dT0_dVg = 2.0 * (Abulk * Abulk * WVCoxRds * tmp2 - Abulk * tmp1
		      + (2.0 * WVCoxRds * Abulk + 1.0 / Pmos_factor - 1.0)
		      * dAbulk_dVg);
             
              dT0_dVb = 2.0 * (Abulk * Abulk * WVCoxRds * (2.0 / Abulk
		      * dAbulk_dVb + tmp3) + (1.0 / Pmos_factor - 1.0)
		      * dAbulk_dVb);
		    dT0_dVd= 0.0; 
              T1 = Vgst2Vtm * (2.0 / Pmos_factor - 1.0) + Abulk 
                 * EsatL + 3.0 * Abulk * Vgst2Vtm * WVCoxRds;
             
              dT1_dVg = (2.0 / Pmos_factor - 1.0) - 2.0 * Vgst2Vtm * tmp1
		      + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 3.0 * (Abulk
		      * WVCoxRds + Abulk * Vgst2Vtm * WVCoxRds * tmp2 + Vgst2Vtm
		      * WVCoxRds * dAbulk_dVg);
              dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
	              + 3.0 * (Vgst2Vtm * WVCoxRds * dAbulk_dVb
		      + Abulk * Vgst2Vtm * WVCoxRds * tmp3);
              dT1_dVd = Abulk * dEsatL_dVd;

              T2 = Vgst2Vtm * (EsatL +  2.0 * Vgst2Vtm * WVCoxRds);
              dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + Vgst2Vtm * WVCoxRds
		      * (4.0 + 2.0 * Vgst2Vtm * tmp2);
              dT2_dVb = Vgst2Vtm * dEsatL_dVb + 2.0 * Vgst2Vtm * WVCoxRds
		      * Vgst2Vtm * tmp3;
              dT2_dVd = Vgst2Vtm * dEsatL_dVd;

              T3 = sqrt(T1 * T1 - 2.0 * T0 * T2);
              here->BSIM3v0vdsat = Vdsat = (T1 - T3) / T0;

              dT3_dVg = (T1 * dT1_dVg - 2.0 * (T0 * dT2_dVg + T2 * dT0_dVg))
	              / T3;
              dT3_dVd = (T1 * dT1_dVd - 2.0 * (T0 * dT2_dVd + T2 * dT0_dVd))
		      / T3;
              dT3_dVb = (T1 * dT1_dVb - 2.0 * (T0 * dT2_dVb + T2 * dT0_dVb))
		      / T3;

              T4 = T1 - T3;
              dT4_dVg = - dT1_dVg - dT3_dVg;
              dT4_dVd = - dT1_dVd - dT3_dVd;
              dT4_dVb = - dT1_dVb - dT3_dVb;

              dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2
			 - T0 * dT2_dVg) / T3 - Vdsat * dT0_dVg) / T0;
              dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2
			 - T0 * dT2_dVb) / T3 - Vdsat * dT0_dVb) / T0;
              dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) / T3) / T0;
          }

/* Effective Vds (Vdseff) Calculation */
          T1 = Vdsat - Vds - pParam->BSIM3v0delta;
          dT1_dVg = dVdsat_dVg;
          dT1_dVd = dVdsat_dVd - 1.0;
          dT1_dVb = dVdsat_dVb;

          T2 = sqrt(T1 * T1 + 4.0 * pParam->BSIM3v0delta * Vdsat);
	  T0 = T1 / T2;
	  T3 = 2.0 * pParam->BSIM3v0delta / T2;
          dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
          dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
          dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

          Vdseff = Vdsat - 0.5 * (T1 + T2);
          dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg); 
          dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd); 
          dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb); 

/* Calculate VAsat */
          tmp1 = a1 / (Pmos_factor * Pmos_factor);
	  if (Rds > 0)
	  {   tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff;
	      tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff;
	  }
	  else
	  {   tmp2 = dWeff_dVg / Weff;
	      tmp3 = dWeff_dVb / Weff;
	  }
          tmp4 = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm;
          T0 = EsatL + Vdsat + 2.0 * WVCoxRds * Vgsteff * tmp4;
         
          dT0_dVg = dEsatL_dVg + dVdsat_dVg + 2.0 * WVCoxRds * tmp4
		  * (1.0 + tmp2 * Vgsteff) - WVCoxRds * Vgsteff / Vgst2Vtm
		  * (Abulk * dVdsat_dVg - Abulk * Vdsat / Vgst2Vtm
		  + Vdsat * dAbulk_dVg);   
		  
          dT0_dVb = dEsatL_dVb + dVdsat_dVb + 2.0 * WVCoxRds * tmp4 * tmp3
		  * Vgsteff - WVCoxRds * Vgsteff / Vgst2Vtm * (dAbulk_dVb
		  * Vdsat + Abulk * dVdsat_dVb);
          dT0_dVd = dEsatL_dVd + dVdsat_dVd - WVCoxRds * Vgsteff / Vgst2Vtm
                  * Abulk * dVdsat_dVd;

          T1 = 2.0 / Pmos_factor - 1.0 + WVCoxRds * Abulk; 
          dT1_dVg = -2.0 * tmp1 +  WVCoxRds *(Abulk * tmp2+ dAbulk_dVg);
          dT1_dVb = dAbulk_dVb * WVCoxRds + Abulk * WVCoxRds * tmp3;

          Vasat = T0 / T1;
          dVasat_dVg = (dT0_dVg - T0 / T1 * dT1_dVg) / T1;
          dVasat_dVb = (dT0_dVb - T0 / T1 * dT1_dVb) / T1;
          dVasat_dVd = dT0_dVd / T1;

          diffVds = Vds - Vdseff;
/* Calculate VACLM */
          if (pParam->BSIM3v0pclm > 0.0)
	  {   T0 = 1.0 / (pParam->BSIM3v0pclm * Abulk * pParam->BSIM3v0litl);
              dT0_dVb = -T0 / Abulk * dAbulk_dVb;
              dT0_dVg= - T0/Abulk*dAbulk_dVg; 
              
	      T2 = Vgsteff / EsatL;
              T1 = Leff * (Abulk + T2); 
              dT1_dVg = Leff * ((1.0 - T2 * dEsatL_dVg) / EsatL+dAbulk_dVg);
              dT1_dVb = Leff * (dAbulk_dVb - T2 * dEsatL_dVb / EsatL);
              dT1_dVd = -T2 * dEsatL_dVd / Esat;

              VACLM = T0 * T1 * diffVds;
              dVACLM_dVg = T0 * dT1_dVg * diffVds - T0 * T1 * dVdseff_dVg
                         + T1 * diffVds * dT0_dVg;
              dVACLM_dVb = (dT0_dVb * T1 + T0 * dT1_dVb) * diffVds
			 - T0 * T1 * dVdseff_dVb;
              dVACLM_dVd = T0 * dT1_dVd * diffVds
			 + T0 * T1 * (1.0 - dVdseff_dVd);
          }
	  else
	  {   VACLM = MAX_EXP;
              dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = 0.0;
          }

/* Calculate VADIBL */
          if (pParam->BSIM3v0thetaRout > 0.0)
	  {   T0 = Vgst2Vtm * Abulk * Vdsat;
              dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + Abulk * Vdsat
		      + Vgst2Vtm * Vdsat * dAbulk_dVg;
              dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
              dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;

              T1 = Vgst2Vtm + Abulk * Vdsat;
              dT1_dVg = 1.0 + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
              dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
              dT1_dVd = Abulk * dVdsat_dVd;

	      T2 = pParam->BSIM3v0thetaRout * (1.0 + pParam->BSIM3v0pdiblb * Vbseff);
              VADIBL = (Vgst2Vtm - T0 / T1) / T2;
              dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / (T1 * T1)) / T2;
              dVADIBL_dVb = ((-dT0_dVb / T1 + T0 * dT1_dVb / (T1 * T1)) - VADIBL
			  * pParam->BSIM3v0thetaRout * pParam->BSIM3v0pdiblb) / T2;
              dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / (T1 * T1)) / T2;
          }
	  else
	  {   VADIBL = MAX_EXP;
              dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = 0.0;
          }

/* Calculate VA */
          T0 = 1.0 + pParam->BSIM3v0pvag * Vgsteff / EsatL;
          dT0_dVg = pParam->BSIM3v0pvag * (1.0 - Vgsteff * dEsatL_dVg
                  / EsatL) / EsatL;
          dT0_dVb = -pParam->BSIM3v0pvag * Vgsteff * dEsatL_dVb
                  / EsatL / EsatL;
          dT0_dVd = -pParam->BSIM3v0pvag * Vgsteff * dEsatL_dVd
                  / EsatL / EsatL;
        
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
          if ((diffVds) && ((T0 = pParam->BSIM3v0pscbe1 * pParam->BSIM3v0litl
	      / diffVds) > 0.0) && (T0 < EXP_THRESHOLD))
	  {   VASCBE = Leff * exp(T0) / pParam->BSIM3v0pscbe2;
              T1 = T0 * VASCBE / diffVds;
              dVASCBE_dVg = T1 * dVdseff_dVg;
              dVASCBE_dVd = -T1 * (1.0 - dVdseff_dVd);
              dVASCBE_dVb = T1 * dVdseff_dVb;
          }
	  else
	  {   VASCBE = MAX_EXP*Leff/pParam->BSIM3v0pscbe2;
              dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0;
          }

/* Calculate Ids */
          CoxWovL = model->BSIM3v0cox * Weff / Leff;
          beta = ueff * CoxWovL;
          dbeta_dVg = CoxWovL * dueff_dVg + beta * dWeff_dVg / Weff;
          dbeta_dVd = CoxWovL * dueff_dVd;
          dbeta_dVb = CoxWovL * dueff_dVb + beta * dWeff_dVb / Weff;

          T0 = 1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm;
          dT0_dVg = -0.5 * (Abulk * dVdseff_dVg 
		  - Abulk * Vdseff / Vgst2Vtm+Vdseff*dAbulk_dVg) / Vgst2Vtm;
          dT0_dVd = -0.5 * Abulk * dVdseff_dVd / Vgst2Vtm;
          dT0_dVb = -0.5 * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff)
                  / Vgst2Vtm;

          fgche1 = Vgsteff * T0;
          dfgche1_dVg = Vgsteff * dT0_dVg + T0; 
          dfgche1_dVd = Vgsteff * dT0_dVd; 
          dfgche1_dVb = Vgsteff * dT0_dVb; 

          fgche2 = 1.0 + Vdseff / EsatL;
          dfgche2_dVg = (dVdseff_dVg - Vdseff / EsatL * dEsatL_dVg) / EsatL;
          dfgche2_dVd = (dVdseff_dVd - Vdseff / EsatL * dEsatL_dVd) / EsatL;
          dfgche2_dVb = (dVdseff_dVb - Vdseff / EsatL * dEsatL_dVb) / EsatL;
 
          gche = beta * fgche1 / fgche2;
          dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg
		    - gche * dfgche2_dVg) / fgche2;
          dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd
		    - gche * dfgche2_dVd) / fgche2;
          dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb
		    - gche * dfgche2_dVb) / fgche2;

          T0 = 1.0 + gche * Rds;
          Idl = gche * Vdseff / T0;

          dIdl_dVg = (gche * dVdseff_dVg + Vdseff * dgche_dVg / T0) / T0
                   - Idl * gche / T0 * dRds_dVg ; 

          dIdl_dVd = (gche * dVdseff_dVd + Vdseff * dgche_dVd / T0) / T0; 
          dIdl_dVb = (gche * dVdseff_dVb + Vdseff * dgche_dVb / T0 
                   - Idl * dRds_dVb * gche) / T0; 

          T0 =  1.0 + diffVds / Va;
          Idsa = Idl * T0;
          dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + diffVds / Va
		    * dVa_dVg) / Va;
          dIdsa_dVd = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd - diffVds / Va
		    * dVa_dVd) / Va;
          dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + diffVds / Va
		    * dVa_dVb) / Va;

          T0 = 1.0 + diffVds / VASCBE;
          Ids = Idsa * T0;

          Gm = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + diffVds / VASCBE
	     * dVASCBE_dVg) / VASCBE;
          Gds = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - diffVds / VASCBE
	      * dVASCBE_dVd) / VASCBE;
          Gmb = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + diffVds / VASCBE
	      * dVASCBE_dVb) / VASCBE;

          Gds += Gm * dVgsteff_dVd;
	  Gmb += Gm * dVgsteff_dVb;
	  Gm *= dVgsteff_dVg;
	  Gmb *= dVbseff_dVb;

/* calculate substrate current Isub */
          if ((pParam->BSIM3v0alpha0 <= 0.0) || (pParam->BSIM3v0beta0 <= 0.0))
	  {   Isub = Gbd = Gbb = Gbg = 0.0;
          }
	  else
	  {   T2 = pParam->BSIM3v0alpha0 / Leff;
                     if (diffVds<0.0) {
                                      diffVds=0.0;
                                      Vdseff=Vds;
                                      } /* added to avoid the hardwrae problem
                                           when Vds=0 */

	      if ((diffVds != 0.0) && ((T0 = -pParam->BSIM3v0beta0 / diffVds)
		  > -EXP_THRESHOLD))
	      {   T1 = T2 * diffVds * exp(T0);
                  dT1_dVg = T1 / diffVds * (T0 - 1.0) * dVdseff_dVg;
                  dT1_dVd = T1 / diffVds * (1.0 - T0) * (1.0 - dVdseff_dVd);
                  dT1_dVb = T1 / diffVds * (T0 - 1.0) * dVdseff_dVb;
              }
	      else
	      {   T1 = T2 * diffVds * MIN_EXP;
                  dT1_dVg = -T2 * MIN_EXP * dVdseff_dVg;
                  dT1_dVd = T2 * MIN_EXP * (1.0 - dVdseff_dVd);
                  dT1_dVb = -T2 * MIN_EXP * dVdseff_dVb;
              }
              Isub = T1 * Idsa;
              Gbg = T1 * dIdsa_dVg + Idsa * dT1_dVg;
              Gbd = T1 * dIdsa_dVd + Idsa * dT1_dVd;
              Gbb = T1 * dIdsa_dVb + Idsa * dT1_dVb;

              Gbd += Gbg * dVgsteff_dVd;
	      Gbb += Gbg * dVgsteff_dVb;
	      Gbg *= dVgsteff_dVg;
	      Gbg *= dVbseff_dVb;
          }
         
          cdrain = Ids;
          here->BSIM3v0gds = Gds;
          here->BSIM3v0gm = Gm;
          here->BSIM3v0gmbs = Gmb;
                   
          here->BSIM3v0gbbs = Gbb;
          here->BSIM3v0gbgs = Gbg;
          here->BSIM3v0gbds = Gbd;

          here->BSIM3v0csub = Isub - (Gbb * Vbseff + Gbd * Vds + Gbg * Vgs);

/* Calculate Qinv for Noise analysis */

          T1 = Vgsteff * (1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm);
          here->BSIM3v0qinv = -model->BSIM3v0cox * Weff * Leff * T1;

          if ((model->BSIM3v0xpart < 0) || (!ChargeComputationNeeded))
	  {   qgate  = qdrn = qsrc = qbulk = 0.0;
              here->BSIM3v0cggb = here->BSIM3v0cgsb = here->BSIM3v0cgdb = 0.0;
              here->BSIM3v0cdgb = here->BSIM3v0cdsb = here->BSIM3v0cddb = 0.0;
              here->BSIM3v0cbgb = here->BSIM3v0cbsb = here->BSIM3v0cbdb = 0.0;
              here->BSIM3v0cqdb = here->BSIM3v0cqsb = here->BSIM3v0cqgb 
                          = here->BSIM3v0cqbb = 0.0;
              here->BSIM3v0gtau = 0.0;
              goto finished;
          }
	  else
	  {   if (Vbseff < 0.0)
	      {   VbseffCV = Vbseff;
                  dVbseffCV_dVb = 1.0;
              }
	      else
	      {   VbseffCV = pParam->BSIM3v0phi - Phis;
                  dVbseffCV_dVb = -dPhis_dVb;
              }

              CoxWL = model->BSIM3v0cox * pParam->BSIM3v0weffCV
		    * pParam->BSIM3v0leffCV;
	      Vfb = Vth - pParam->BSIM3v0phi - pParam->BSIM3v0k1 * sqrtPhis;

	      dVfb_dVb = dVth_dVb - pParam->BSIM3v0k1 * dsqrtPhis_dVb;
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

	      if (model->BSIM3v0capMod == 1)
	      {   Arg1 = Vgs_eff - VbseffCV - Vfb;

                  if (Arg1 <= 0.0)
	          {   qgate = CoxWL * (Arg1 - Vgsteff);
                      Cgg = CoxWL * (dVgs_eff_dVg - dVgsteff_dVg);
                      Cgd = -CoxWL * (dVfb_dVd + dVgsteff_dVd);
                      Cgb = -CoxWL * (dVfb_dVb + dVbseffCV_dVb + dVgsteff_dVb);
                  }
	          else
	          {   T0 = 0.5 * pParam->BSIM3v0k1;
                      T1 = sqrt(T0 * T0 + Arg1 - Vgsteff);
                      qgate = CoxWL * pParam->BSIM3v0k1 * (T1 - T0);

                      T2 = CoxWL * T0 / T1;
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
                  AbulkCV = Abulk0 * pParam->BSIM3v0abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM3v0abulkCVfactor * dAbulk0_dVb;
	          VdsatCV = Vgsteff / AbulkCV;
                  if (VdsatCV <= Vds)
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

                      if (model->BSIM3v0xpart > 0.5)
		          T0 = -Two_Third_CoxWL;
                      else if (model->BSIM3v0xpart < 0.5)
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
                      T1 = 12.0 * (Vgsteff - 0.5 * T0);
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

                      if (model->BSIM3v0xpart > 0.5)
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
		      else if (model->BSIM3v0xpart < 0.5)
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
                  here->BSIM3v0cggb = Cgg;
	          here->BSIM3v0cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM3v0cgdb = Cgd;
                  here->BSIM3v0cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM3v0cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM3v0cddb = -(Cgd + Cbd + Csd);
                  here->BSIM3v0cbgb = Cbg;
	          here->BSIM3v0cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM3v0cbdb = Cbd;
	      }
	      else
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

                  T0 = 0.5 * pParam->BSIM3v0k1;
		  T1 = sqrt(T0 * T0 + Vgs_eff - Vfbeff - VbseffCV - Vgsteff);

		  Qsub0 = CoxWL * pParam->BSIM3v0k1 * (T1 - T0);

                  T2 = CoxWL * T0 / T1;
                  dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
                  dQsub0_dVd = -T2 * (dVfbeff_dVd + dVgsteff_dVd);
                  dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb 
                               + dVgsteff_dVb);

                  One_Third_CoxWL = CoxWL / 3.0;
                  Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
                  AbulkCV = Abulk0 * pParam->BSIM3v0abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM3v0abulkCVfactor * dAbulk0_dVb;
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

                  if (model->BSIM3v0xpart > 0.5)
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
		  else if (model->BSIM3v0xpart < 0.5)
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

                  here->BSIM3v0cggb = Cgg;
	          here->BSIM3v0cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM3v0cgdb = Cgd;
                  here->BSIM3v0cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM3v0cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM3v0cddb = -(Cgd + Cbd + Csd);
                  here->BSIM3v0cbgb = Cbg;
	          here->BSIM3v0cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM3v0cbdb = Cbd;

	      }

/* Non-quasi-static Model */

              if (here->BSIM3v0nqsMod)
              {   
                  qcheq = -qbulk - qgate;
                  qbulk = qgate = qdrn = qsrc = 0.0;

                  here->BSIM3v0cqgb = -(here->BSIM3v0cggb + here->BSIM3v0cbgb);
                  here->BSIM3v0cqdb = -(here->BSIM3v0cgdb + here->BSIM3v0cbdb);
                  here->BSIM3v0cqsb = -(here->BSIM3v0cgsb + here->BSIM3v0cbsb);
                  here->BSIM3v0cqbb =  here->BSIM3v0cggb + here->BSIM3v0cgdb 
                                  + here->BSIM3v0cgsb + here->BSIM3v0cbgb 
                                  + here->BSIM3v0cbdb + here->BSIM3v0cbsb;

                  here->BSIM3v0cggb = here->BSIM3v0cgsb = here->BSIM3v0cgdb = 0.0;
                  here->BSIM3v0cdgb = here->BSIM3v0cdsb = here->BSIM3v0cddb = 0.0;
                  here->BSIM3v0cbgb = here->BSIM3v0cbsb = here->BSIM3v0cbdb = 0.0;

	          T0 = pParam->BSIM3v0leffCV * pParam->BSIM3v0leffCV;
                  here->BSIM3v0tconst = pParam->BSIM3v0u0temp * pParam->BSIM3v0elm
				    / CoxWL / T0;

                  if (qcheq == 0.0)
		      here->BSIM3v0tconst = 0.0;
                  else if (qcheq < 0.0)
		      here->BSIM3v0tconst = -here->BSIM3v0tconst;

                  gtau_drift = fabs(here->BSIM3v0tconst * qcheq);
                  gtau_diff = 16.0 * pParam->BSIM3v0u0temp * model->BSIM3v0vtm / T0;
 
                  here->BSIM3v0gtau =  gtau_drift + gtau_diff;

                  *(ckt->CKTstate0 + here->BSIM3v0qcheq) = qcheq;
 
                  if (ckt->CKTmode & MODEINITTRAN)
                     *(ckt->CKTstate1 + here->BSIM3v0qcheq) =
                                   *(ckt->CKTstate0 + here->BSIM3v0qcheq);
              
                  error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v0qcheq);
                  if (error) return (error);
              }
              else
              {   here->BSIM3v0cqgb = here->BSIM3v0cqdb = here->BSIM3v0cqsb 
                                  = here->BSIM3v0cqbb = 0.0;
                  here->BSIM3v0gtau = 0.0; 
              }
          }

finished: /* returning Values to Calling Routine */
          /*
           *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
           */
          here->BSIM3v0cd = here->BSIM3v0mode * cdrain - here->BSIM3v0cbd;
          if (ChargeComputationNeeded)
	  {   /*  charge storage elements
               *  bulk-drain and bulk-source depletion capacitances
               *  czbd : zero bias drain junction capacitance
               *  czbs : zero bias source junction capacitance
               *  czbdsw:zero bias drain junction sidewall capacitance
               *  czbssw:zero bias source junction sidewall capacitance
               */

              czbd = model->BSIM3v0unitAreaJctCap * here->BSIM3v0drainArea;
              czbs = model->BSIM3v0unitAreaJctCap * here->BSIM3v0sourceArea;
              czbdsw = model->BSIM3v0unitLengthSidewallJctCap 
		     * here->BSIM3v0drainPerimeter;
              czbssw = model->BSIM3v0unitLengthSidewallJctCap 
		     * here->BSIM3v0sourcePerimeter;
              PhiB = model->BSIM3v0bulkJctPotential;
              PhiBSW = model->BSIM3v0sidewallJctPotential;
              MJ = model->BSIM3v0bulkJctBotGradingCoeff;
              MJSW = model->BSIM3v0bulkJctSideGradingCoeff;

              /* Source Bulk Junction */
	      if (vbs == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM3v0qbs) = 0.0;
                  here->BSIM3v0capbs = czbs + czbssw;
	      }
	      else if (vbs < 0.0)
	      {   if (czbs > 0.0)
		  {   arg = 1.0 - vbs / PhiB;
		      if (MJ == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJ * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v0qbs) = PhiB * czbs 
					* (1.0 - arg * sarg) / (1.0 - MJ);
		      here->BSIM3v0capbs = czbs * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM3v0qbs) = 0.0;
		      here->BSIM3v0capbs = 0.0;
		  }
		  if (czbssw > 0.0)
		  {   arg = 1.0 - vbs / PhiBSW;
		      if (MJSW == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSW * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v0qbs) += PhiBSW * czbssw
				    * (1.0 - arg * sarg) / (1.0 - MJSW);
                      here->BSIM3v0capbs += czbssw * sarg;
		  }
              }
	      else
	      {   *(ckt->CKTstate0+here->BSIM3v0qbs) = vbs * (czbs + czbssw) 
			           + vbs * vbs * (czbs * MJ * 0.5 / PhiB 
                                   + czbssw * MJSW * 0.5 / PhiBSW);
                  here->BSIM3v0capbs = czbs + czbssw + vbs * (czbs * MJ /PhiB
		                   + czbssw * MJSW / PhiBSW );
              }

              /* Drain Bulk Junction */
	      if (vbd == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM3v0qbd) = 0.0;
                  here->BSIM3v0capbd = czbd + czbdsw;
	      }
	      else if (vbd < 0.0)
	      {   if (czbd > 0.0)
		  {   arg = 1.0 - vbd / PhiB;
		      if (MJ == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJ * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v0qbd) = PhiB * czbd 
			           * (1.0 - arg * sarg) / (1.0 - MJ);
                      here->BSIM3v0capbd = czbd * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM3v0qbd) = 0.0;
                      here->BSIM3v0capbd = 0.0;
		  }
		  if (czbdsw > 0.0)
		  {   arg = 1.0 - vbd / PhiBSW;
		      if (MJSW == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSW * log(arg));
                      *(ckt->CKTstate0 + here->BSIM3v0qbd) += PhiBSW * czbdsw 
			               * (1.0 - arg * sarg) / (1.0 - MJSW);
                      here->BSIM3v0capbd += czbdsw * sarg;
		  }
              }
	      else
	      {   *(ckt->CKTstate0+here->BSIM3v0qbd) = vbd * (czbd + czbdsw)
			      + vbd * vbd * (czbd * MJ * 0.5 / PhiB 
                              + czbdsw * MJSW * 0.5 / PhiBSW);
                  here->BSIM3v0capbd = czbd + czbdsw + vbd * (czbd * MJ / PhiB
			      + czbdsw * MJSW / PhiBSW );
              }
          }

          /*
           *  check convergence
           */
          if ((here->BSIM3v0off == 0) || (!(ckt->CKTmode & MODEINITFIX)))
	  {   if (Check == 1)
	      {   ckt->CKTnoncon++;
#ifndef NEWCONV
              } 
	      else
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(here->BSIM3v0cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - here->BSIM3v0cd) >= tol)
		  {   ckt->CKTnoncon++;
                  }
		  else
		  {   tol = ckt->CKTreltol * MAX(fabs(cbhat), 
			    fabs(here->BSIM3v0cbs + here->BSIM3v0cbd)) 
			    + ckt->CKTabstol;
                      if (fabs(cbhat - (here->BSIM3v0cbs + here->BSIM3v0cbd)) 
			  > tol)
		      {   ckt->CKTnoncon++;
                      }
                  }
#endif /* NEWCONV */
              }
          }
          *(ckt->CKTstate0 + here->BSIM3v0vbs) = vbs;
          *(ckt->CKTstate0 + here->BSIM3v0vbd) = vbd;
          *(ckt->CKTstate0 + here->BSIM3v0vgs) = vgs;
          *(ckt->CKTstate0 + here->BSIM3v0vds) = vds;

          /* bulk and channel charge plus overlaps */

          if (!ChargeComputationNeeded)
              goto line850; 
#ifndef NOBYPASS      
line755:
#endif
          ag0 = ckt->CKTag[0];

	  if (model->BSIM3v0capMod == 1)
	  {   if (vgd < 0.0)
	      {   T1 = sqrt(1.0 - 4.0 * vgd / pParam->BSIM3v0ckappa);
	          cgdo = pParam->BSIM3v0cgdo + pParam->BSIM3v0weffCV
		       * pParam->BSIM3v0cgdl / T1;
	          qgdo = pParam->BSIM3v0cgdo * vgd - pParam->BSIM3v0weffCV * 0.5
		       * pParam->BSIM3v0cgdl * pParam->BSIM3v0ckappa * (T1 - 1.0);
	      }
	      else
	      {   cgdo = pParam->BSIM3v0cgdo + pParam->BSIM3v0weffCV
		       * pParam->BSIM3v0cgdl;
	          qgdo = (pParam->BSIM3v0weffCV * pParam->BSIM3v0cgdl
		       + pParam->BSIM3v0cgdo) * vgd;
	      }

	      if (vgs < 0.0)
	      {   T1 = sqrt(1.0 - 4.0 * vgs / pParam->BSIM3v0ckappa);
	          cgso = pParam->BSIM3v0cgso + pParam->BSIM3v0weffCV
		       * pParam->BSIM3v0cgsl / T1;
	          qgso = pParam->BSIM3v0cgso * vgs - pParam->BSIM3v0weffCV * 0.5
		       * pParam->BSIM3v0cgsl * pParam->BSIM3v0ckappa * (T1 - 1.0);
	      }
	      else
	      {   cgso = pParam->BSIM3v0cgso + pParam->BSIM3v0weffCV
		       * pParam->BSIM3v0cgsl;
	          qgso = (pParam->BSIM3v0weffCV * pParam->BSIM3v0cgsl
		       + pParam->BSIM3v0cgso) * vgs;
	      }
	  }
	  else
	  {   T0 = vgd + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);

	      T3 = pParam->BSIM3v0weffCV * pParam->BSIM3v0cgdl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM3v0ckappa);
	      cgdo = pParam->BSIM3v0cgdo + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgdo = (pParam->BSIM3v0cgdo + T3) * vgd - T3 * (T2
		   + 0.5 * pParam->BSIM3v0ckappa * (T4 - 1.0));

	      T0 = vgs + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);
	      T3 = pParam->BSIM3v0weffCV * pParam->BSIM3v0cgsl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM3v0ckappa);
	      cgso = pParam->BSIM3v0cgso + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgso = (pParam->BSIM3v0cgso + T3) * vgs - T3 * (T2
		   + 0.5 * pParam->BSIM3v0ckappa * (T4 - 1.0));
	  }

          if (here->BSIM3v0mode > 0)
	  {   gcdgb = (here->BSIM3v0cdgb - cgdo) * ag0;
              gcddb = (here->BSIM3v0cddb + here->BSIM3v0capbd + cgdo) * ag0;
              gcdsb = here->BSIM3v0cdsb * ag0;
              gcsgb = -(here->BSIM3v0cggb + here->BSIM3v0cbgb + here->BSIM3v0cdgb
	            + cgso) * ag0;
              gcsdb = -(here->BSIM3v0cgdb + here->BSIM3v0cbdb + here->BSIM3v0cddb)
		    * ag0;
              gcssb = (here->BSIM3v0capbs + cgso - (here->BSIM3v0cgsb
		    + here->BSIM3v0cbsb + here->BSIM3v0cdsb)) * ag0;
              gcggb = (here->BSIM3v0cggb + cgdo + cgso + pParam->BSIM3v0cgbo ) * ag0;
              gcgdb = (here->BSIM3v0cgdb - cgdo) * ag0;
              gcgsb = (here->BSIM3v0cgsb - cgso) * ag0;
              gcbgb = (here->BSIM3v0cbgb - pParam->BSIM3v0cgbo) * ag0;
              gcbdb = (here->BSIM3v0cbdb - here->BSIM3v0capbd) * ag0;
              gcbsb = (here->BSIM3v0cbsb - here->BSIM3v0capbs) * ag0;
 
              gcqgb = here->BSIM3v0cqgb * ag0;
              gcqdb = here->BSIM3v0cqdb * ag0;
              gcqsb = here->BSIM3v0cqsb * ag0;
              gcqbb = here->BSIM3v0cqbb * ag0;

              T0 = here->BSIM3v0tconst * qdef;
              here->BSIM3v0gtg = T0 * here->BSIM3v0cqgb;
              here->BSIM3v0gtb = T0 * here->BSIM3v0cqbb;
              here->BSIM3v0gtd = T0 * here->BSIM3v0cqdb;
              here->BSIM3v0gts = T0 * here->BSIM3v0cqsb;
 
              sxpart = 0.6;
              dxpart = 0.4;

              /* compute total terminal charge */
              qgd = qgdo;
              qgs = qgso;
              qgb = pParam->BSIM3v0cgbo * vgb;
              qgate += qgd + qgs + qgb;
              qbulk -= qgb;
              qdrn -= qgd;
              qsrc = -(qgate + qbulk + qdrn);
	  }
	  else
	  {   gcsgb = (here->BSIM3v0cdgb - cgso) * ag0;
              gcsdb = here->BSIM3v0cdsb * ag0;
              gcssb = (here->BSIM3v0cddb + here->BSIM3v0capbs + cgso) * ag0;

              gcdgb = -(here->BSIM3v0cggb + here->BSIM3v0cbgb + here->BSIM3v0cdgb
		    + cgdo) * ag0;
              gcdsb = -(here->BSIM3v0cgdb + here->BSIM3v0cbdb + here->BSIM3v0cddb)
		    * ag0;
              gcddb = (here->BSIM3v0capbd + cgdo - (here->BSIM3v0cgsb
		    + here->BSIM3v0cbsb + here->BSIM3v0cdsb)) * ag0;
              gcggb = (here->BSIM3v0cggb + cgdo + cgso + pParam->BSIM3v0cgbo ) * ag0;
              gcgdb = (here->BSIM3v0cgsb - cgdo) * ag0;
              gcgsb = (here->BSIM3v0cgdb - cgso) * ag0;
              gcbgb = (here->BSIM3v0cbgb - pParam->BSIM3v0cgbo) * ag0;
              gcbdb = (here->BSIM3v0cbsb - here->BSIM3v0capbd) * ag0;
              gcbsb = (here->BSIM3v0cbdb - here->BSIM3v0capbs) * ag0;
 
              gcqgb = here->BSIM3v0cqgb * ag0;
              gcqdb = here->BSIM3v0cqsb * ag0;
              gcqsb = here->BSIM3v0cqdb * ag0;
              gcqbb = here->BSIM3v0cqbb * ag0;

              T0 = here->BSIM3v0tconst * qdef;
              here->BSIM3v0gtg = T0 * here->BSIM3v0cqgb;
              here->BSIM3v0gtb = T0 * here->BSIM3v0cqbb;
              here->BSIM3v0gtd = T0 * here->BSIM3v0cqdb;
              here->BSIM3v0gts = T0 * here->BSIM3v0cqsb;
 
              dxpart = 0.6;
              sxpart = 0.4;

              /* compute total terminal charge */
              qgd = qgdo;
              qgs = qgso;
              qgb = pParam->BSIM3v0cgbo * vgb;
              qgate += qgd + qgs + qgb;
              qbulk -= qgb;
              qsrc = qdrn - qgs;
              qdrn = -(qgate + qbulk + qsrc);
          }

          here->BSIM3v0cgdo = cgdo;
          here->BSIM3v0cgso = cgso;

/* added by Mansun 11/1/93 */


          if (here->BSIM3v0nqsMod)
          {   *(ckt->CKTstate0 + here->BSIM3v0qcdump) = qdef;

              if (ckt->CKTmode & MODEINITTRAN)
	             *(ckt->CKTstate1 + here->BSIM3v0qcdump) =
                              *(ckt->CKTstate0 + here->BSIM3v0qcdump);

              error = NIintegrate(ckt, &gqdef, &cqdef, 1.0, here->BSIM3v0qcdump);
              if (error) return (error);
          }
	  else
	  {   gqdef = cqdef = 0.0;
	  }

          if (ByPass) goto line860;

/* End added by Mansun 11/1/93 */
             
          *(ckt->CKTstate0 + here->BSIM3v0qg) = qgate;
          *(ckt->CKTstate0 + here->BSIM3v0qd) = qdrn
		    - *(ckt->CKTstate0 + here->BSIM3v0qbd);
          *(ckt->CKTstate0 + here->BSIM3v0qb) = qbulk
		    + *(ckt->CKTstate0 + here->BSIM3v0qbd)
		    + *(ckt->CKTstate0 + here->BSIM3v0qbs); 

          /* store small signal parameters */
          if (ckt->CKTmode & MODEINITSMSIG)
	  {   goto line1000;
          }
          if (!ChargeComputationNeeded)
              goto line850;
       
          if (ckt->CKTmode & MODEINITTRAN)
	  {   *(ckt->CKTstate1 + here->BSIM3v0qb) =
                    *(ckt->CKTstate0 + here->BSIM3v0qb);
              *(ckt->CKTstate1 + here->BSIM3v0qg) =
                    *(ckt->CKTstate0 + here->BSIM3v0qg);
              *(ckt->CKTstate1 + here->BSIM3v0qd) =
                    *(ckt->CKTstate0 + here->BSIM3v0qd);
          }
       
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v0qb);
          if (error)
	      return(error);
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM3v0qg);
          if (error)
	      return(error);
          error = NIintegrate(ckt,&geq, &ceq, 0.0, here->BSIM3v0qd);
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
          here->BSIM3v0gtg = here->BSIM3v0gtd = here->BSIM3v0gts = here->BSIM3v0gtb = 0.0;
          gqdef = 0.0;
          sxpart = (1.0 - (dxpart = (here->BSIM3v0mode > 0) ? 0.4 : 0.6));
          if (here->BSIM3v0nqsMod)
              here->BSIM3v0gtau = 16.0 * pParam->BSIM3v0u0temp * model->BSIM3v0vtm 
                              / Leff / Leff;
	  else
              here->BSIM3v0gtau = 0.0;

          goto line900;
            
line860:
          /* evaluate equivalent charge current */

          cqgate = *(ckt->CKTstate0 + here->BSIM3v0cqg);
          cqbulk = *(ckt->CKTstate0 + here->BSIM3v0cqb);
          cqdrn = *(ckt->CKTstate0 + here->BSIM3v0cqd);

          ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs
                  + (here->BSIM3v0gtg * vgb - here->BSIM3v0gtd * vbd - here->BSIM3v0gts * vbs);
          ceqqb = cqbulk - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs;
          ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs
                  - dxpart * (here->BSIM3v0gtg * vgb - here->BSIM3v0gtd * vbd 
                  - here->BSIM3v0gts * vbs);
 
	     cqcheq = *(ckt->CKTstate0 + here->BSIM3v0cqcheq)
                 - (gcqgb * vgb - gcqdb * vbd  - gcqsb * vbs)
                 + (here->BSIM3v0gtg * vgb - here->BSIM3v0gtd * vbd  - here->BSIM3v0gts * vbs);

          if (ckt->CKTmode & MODEINITTRAN)
	  {   *(ckt->CKTstate1 + here->BSIM3v0cqb) =  
                    *(ckt->CKTstate0 + here->BSIM3v0cqb);
              *(ckt->CKTstate1 + here->BSIM3v0cqg) =  
                    *(ckt->CKTstate0 + here->BSIM3v0cqg);
              *(ckt->CKTstate1 + here->BSIM3v0cqd) =  
                    *(ckt->CKTstate0 + here->BSIM3v0cqd);
              *(ckt->CKTstate1 + here->BSIM3v0cqcheq) =  
                    *(ckt->CKTstate0 + here->BSIM3v0cqcheq);
              *(ckt->CKTstate1 + here->BSIM3v0cqcdump) =  
                    *(ckt->CKTstate0 + here->BSIM3v0cqcdump);
          }

          /*
           *  load current vector
           */
line900:

          if (here->BSIM3v0mode >= 0)
	  {   Gm = here->BSIM3v0gm;
              Gmbs = here->BSIM3v0gmbs;
              FwdSum = Gm + Gmbs;
              RevSum = 0.0;
              cdreq = model->BSIM3v0type * (cdrain - here->BSIM3v0gds * vds
		    - Gm * vgs - Gmbs * vbs);
              ceqbs = -here->BSIM3v0csub;
              ceqbd = 0.0;

              gbspsp = -here->BSIM3v0gbds - here->BSIM3v0gbgs - here->BSIM3v0gbbs;
              gbbdp = -here->BSIM3v0gbds;
              gbbsp = here->BSIM3v0gbds + here->BSIM3v0gbgs + here->BSIM3v0gbbs;
              gbspg = here->BSIM3v0gbgs;
              gbspb = here->BSIM3v0gbbs;
              gbspdp = here->BSIM3v0gbds;
              gbdpdp = 0.0;
              gbdpg = 0.0;
              gbdpb = 0.0;
              gbdpsp = 0.0;
          }
	  else
	  {   Gm = -here->BSIM3v0gm;
              Gmbs = -here->BSIM3v0gmbs;
              FwdSum = 0.0;
              RevSum = -(Gm + Gmbs);
              cdreq = -model->BSIM3v0type * (cdrain + here->BSIM3v0gds * vds
                    + Gm * vgd + Gmbs * vbd);
              ceqbs = 0.0;
              ceqbd = -here->BSIM3v0csub;

              gbspsp = 0.0;
              gbbdp = here->BSIM3v0gbds + here->BSIM3v0gbgs + here->BSIM3v0gbbs;
              gbbsp = -here->BSIM3v0gbds;
              gbspg = 0.0;
              gbspb = 0.0;
              gbspdp = 0.0;
              gbdpdp = -here->BSIM3v0gbds - here->BSIM3v0gbgs - here->BSIM3v0gbbs;
              gbdpg = here->BSIM3v0gbgs;
              gbdpb = here->BSIM3v0gbbs;
              gbdpsp = here->BSIM3v0gbds;
          }

	   if (model->BSIM3v0type > 0)
	   {   ceqbs += (here->BSIM3v0cbs - (here->BSIM3v0gbs - ckt->CKTgmin) * vbs);
               ceqbd += (here->BSIM3v0cbd - (here->BSIM3v0gbd - ckt->CKTgmin) * vbd);
               /*
               ceqqg = ceqqg;
               ceqqb = ceqqb;
               ceqqd = ceqqd;
               cqcheq = cqcheq;
               */
	   }
	   else
	   {   ceqbs = -ceqbs - (here->BSIM3v0cbs - (here->BSIM3v0gbs
		     - ckt->CKTgmin) * vbs); 
               ceqbd = -ceqbd - (here->BSIM3v0cbd - (here->BSIM3v0gbd
		     - ckt->CKTgmin) * vbd);
               ceqqg = -ceqqg;
               ceqqb = -ceqqb;
               ceqqd = -ceqqd;
               cqcheq = -cqcheq;
	   }

           (*(ckt->CKTrhs + here->BSIM3v0gNode) -= m * ceqqg);
           (*(ckt->CKTrhs + here->BSIM3v0bNode) -= m * (ceqbs + ceqbd + ceqqb));
           (*(ckt->CKTrhs + here->BSIM3v0dNodePrime) += m * (ceqbd - cdreq - ceqqd));
           (*(ckt->CKTrhs + here->BSIM3v0sNodePrime) += m * (cdreq + ceqbs + ceqqg
						  + ceqqb + ceqqd));

           *(ckt->CKTrhs + here->BSIM3v0qNode) += m * (cqcheq - cqdef);

           /*
            *  load y matrix
            */

           (*(here->BSIM3v0DdPtr) += m * here->BSIM3v0drainConductance);
           (*(here->BSIM3v0GgPtr) += m * (gcggb - here->BSIM3v0gtg));
           (*(here->BSIM3v0SsPtr) += m * here->BSIM3v0sourceConductance);
           (*(here->BSIM3v0BbPtr) += m * ((here->BSIM3v0gbd + here->BSIM3v0gbs
                               - gcbgb - gcbdb - gcbsb) - here->BSIM3v0gbbs));
           (*(here->BSIM3v0DPdpPtr) += m * ((here->BSIM3v0drainConductance
                                 + here->BSIM3v0gds + here->BSIM3v0gbd
                                 + RevSum + gcddb) + dxpart * here->BSIM3v0gtd + gbdpdp));
           (*(here->BSIM3v0SPspPtr) += m * ((here->BSIM3v0sourceConductance
                                 + here->BSIM3v0gds + here->BSIM3v0gbs
                                 + FwdSum + gcssb) + sxpart * here->BSIM3v0gts + gbspsp));
           (*(here->BSIM3v0DdpPtr) -= m * here->BSIM3v0drainConductance);
           (*(here->BSIM3v0GbPtr) -= m * (gcggb + gcgdb + gcgsb + here->BSIM3v0gtb));
           (*(here->BSIM3v0GdpPtr) += m * (gcgdb - here->BSIM3v0gtd));
           (*(here->BSIM3v0GspPtr) += m * (gcgsb - here->BSIM3v0gts));
           (*(here->BSIM3v0SspPtr) -= m * here->BSIM3v0sourceConductance);
           (*(here->BSIM3v0BgPtr) += m * (gcbgb - here->BSIM3v0gbgs));
           (*(here->BSIM3v0BdpPtr) += m * (gcbdb - here->BSIM3v0gbd + gbbdp));
           (*(here->BSIM3v0BspPtr) += m * (gcbsb - here->BSIM3v0gbs + gbbsp));
           (*(here->BSIM3v0DPdPtr) -= m * here->BSIM3v0drainConductance);
           (*(here->BSIM3v0DPgPtr) += m * ((Gm + gcdgb) + dxpart * here->BSIM3v0gtg + gbdpg));
           (*(here->BSIM3v0DPbPtr) -= m * ((here->BSIM3v0gbd - Gmbs + gcdgb + gcddb
                                + gcdsb - dxpart * here->BSIM3v0gtb) - gbdpb));
           (*(here->BSIM3v0DPspPtr) -= m * ((here->BSIM3v0gds + FwdSum - gcdsb
				 - 0.5 * here->BSIM3v0gts) - gbdpsp));
           (*(here->BSIM3v0SPgPtr) += m * (gcsgb - Gm + sxpart * here->BSIM3v0gtg + gbspg));
           (*(here->BSIM3v0SPsPtr) -= m * here->BSIM3v0sourceConductance);
           (*(here->BSIM3v0SPbPtr) -= m * ((here->BSIM3v0gbs + Gmbs + gcsgb + gcsdb
                                + gcssb - sxpart * here->BSIM3v0gtg) - gbspb));
           (*(here->BSIM3v0SPdpPtr) -= m * ((here->BSIM3v0gds + RevSum - gcsdb
                                 - sxpart * here->BSIM3v0gtd - here->BSIM3v0gbd) - gbspdp));
 
           *(here->BSIM3v0QqPtr) += m * (gqdef + here->BSIM3v0gtau);
 
           *(here->BSIM3v0DPqPtr) += m * (dxpart * here->BSIM3v0gtau);
           *(here->BSIM3v0SPqPtr) += m * (sxpart * here->BSIM3v0gtau);
           *(here->BSIM3v0GqPtr) -= m * here->BSIM3v0gtau;
 
           *(here->BSIM3v0QgPtr) += m * (-gcqgb + here->BSIM3v0gtg);
           *(here->BSIM3v0QdpPtr) += m * (-gcqdb + here->BSIM3v0gtd);
           *(here->BSIM3v0QspPtr) += m * (-gcqsb + here->BSIM3v0gts);
           *(here->BSIM3v0QbPtr) += m * (-gcqbb + here->BSIM3v0gtb);

line1000:  ;

     }  /* End of Mosfet Instance */
}   /* End of Model Instance */

return(OK);
}





