/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4ld.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 02/06/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
/*
#define MAX_EXP 2.688117142e+43
#define MIN_EXP 3.720075976e-44
#define EXP_THRESHOLD 100.0
*/
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPSSI 1.03594e-10
#define Charge_q 1.60219e-19
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
#define DELTA_4 0.02
#define MM  3  /* smooth coeff */
#define DEXP(A,B,C) {                                                         \
        if (A > EXP_THRESHOLD) {                                              \
            B = MAX_EXP*(1.0+(A)-EXP_THRESHOLD);                              \
            C = MAX_EXP;                                                      \
        } else if (A < -EXP_THRESHOLD)  {                                     \
            B = MIN_EXP;                                                      \
            C = 0;                                                            \
        } else   {                                                            \
            B = exp(A);                                                       \
            C = B;                                                            \
        }                                                                     \
    }

#ifdef USE_OMP
int BSIM4v5LoadOMP(BSIM4v5instance *here, CKTcircuit *ckt);
void BSIM4v5LoadRhsMat(GENmodel *inModel, CKTcircuit *ckt);
#endif

int BSIM4v5polyDepletion(double phi, double ngate,double coxe, double Vgs, double *Vgs_eff, double *dVgs_eff_dVg);

int
BSIM4v5load(
GENmodel *inModel,
CKTcircuit *ckt)
{
#ifdef USE_OMP
    int idx;
    BSIM4v5model *model = (BSIM4v5model*)inModel;
    int error = 0;
    BSIM4v5instance **InstArray;
    InstArray = model->BSIM4v5InstanceArray;

#pragma omp parallel for
    for (idx = 0; idx < model->BSIM4v5InstCount; idx++) {
        BSIM4v5instance *here = InstArray[idx];
        int local_error = BSIM4v5LoadOMP(here, ckt);
        if (local_error)
            error = local_error;
    }

    BSIM4v5LoadRhsMat(inModel, ckt);

    return error;
}


int BSIM4v5LoadOMP(BSIM4v5instance *here, CKTcircuit *ckt) {
BSIM4v5model *model = BSIM4v5modPtr(here);
#else
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;
#endif

double ceqgstot, dgstot_dvd, dgstot_dvg, dgstot_dvs, dgstot_dvb;
double ceqgdtot, dgdtot_dvd, dgdtot_dvg, dgdtot_dvs, dgdtot_dvb;
double gstot, gstotd, gstotg, gstots, gstotb, gspr, Rs, Rd;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double vgs_eff, vgd_eff, dvgs_eff_dvg, dvgd_eff_dvg;
double dRs_dvg, dRd_dvg, dRs_dvb, dRd_dvb;
double dT0_dvg, dT1_dvb, dT3_dvg, dT3_dvb;
double vses, vdes, vdedo, delvses, delvded, delvdes;
double Isestot=0.0, cseshat=0.0, Idedtot=0.0, cdedhat=0.0;
#ifndef NEWCONV
double tol0, tol1, tol2, tol3, tol4, tol5, tol6;
#endif

double geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb, ceqgcrg;
double vges, vgms, vgedo, vgmdo, vged, vgmd, delvged, delvgmd;
double delvges, delvgms, vgmb;
double gcgmgmb=0.0, gcgmdb=0.0, gcgmsb=0.0, gcdgmb, gcsgmb;
double gcgmbb=0.0, gcbgmb, qgmb, qgmid=0.0, ceqqgmid;

double vbd, vbs, vds, vgb, vgd, vgs, vgdo;
#ifndef PREDICTOR
double xfact;
#endif
double vdbs, vdbd, vsbs, vsbdo, vsbd;
double delvdbs, delvdbd, delvsbs;
double delvbd_jct, delvbs_jct, vbs_jct, vbd_jct;

double SourceSatCurrent, DrainSatCurrent;
double ag0, qgb, von, cbhat=0.0, VgstNVt, ExpVgst;
double ceqqb, ceqqd, ceqqg, ceqqjd=0.0, ceqqjs=0.0, ceq, geq;
double cdrain, cdhat=0.0, ceqdrn, ceqbd, ceqbs, ceqjd, ceqjs, gjbd, gjbs;
double czbd, czbdsw, czbdswg, czbs, czbssw, czbsswg, evbd, evbs, arg, sarg;
double delvbd, delvbs, delvds, delvgd, delvgs;
double Vfbeff, dVfbeff_dVg, dVfbeff_dVb, V3, V4;
double gcbdb, gcbgb, gcbsb, gcddb, gcdgb, gcdsb, gcgdb, gcggb, gcgsb, gcsdb;
double gcgbb, gcdbb, gcsbb, gcbbb;
double gcdbdb, gcsbsb;
double gcsgb, gcssb, MJD, MJSWD, MJSWGD, MJS, MJSWS, MJSWGS;
double qgate=0.0, qbulk=0.0, qdrn=0.0, qsrc, cqgate, cqbody, cqdrn;
double Vdb, Vds, Vgs, Vbs, Gmbs, FwdSum, RevSum;
double Igidl, Ggidld, Ggidlg, Ggidlb;
double Voxacc=0.0, dVoxacc_dVg=0.0, dVoxacc_dVb=0.0;
double Voxdepinv=0.0, dVoxdepinv_dVg=0.0, dVoxdepinv_dVd=0.0, dVoxdepinv_dVb=0.0;
double VxNVt=0.0, ExpVxNVt, Vaux=0.0, dVaux_dVg=0.0, dVaux_dVd=0.0, dVaux_dVb=0.0;
double Igc, dIgc_dVg, dIgc_dVd, dIgc_dVb;
double Igcs, dIgcs_dVg, dIgcs_dVd, dIgcs_dVb;
double Igcd, dIgcd_dVg, dIgcd_dVd, dIgcd_dVb;
double Igs, dIgs_dVg, dIgs_dVs, Igd, dIgd_dVg, dIgd_dVd;
double Igbacc, dIgbacc_dVg, dIgbacc_dVb;
double Igbinv, dIgbinv_dVg, dIgbinv_dVd, dIgbinv_dVb;
double Pigcd, dPigcd_dVg, dPigcd_dVd, dPigcd_dVb;
double Istoteq, gIstotg, gIstotd, gIstots, gIstotb;
double Idtoteq, gIdtotg, gIdtotd, gIdtots, gIdtotb;
double Ibtoteq, gIbtotg, gIbtotd, gIbtots, gIbtotb;
double Igtoteq, gIgtotg, gIgtotd, gIgtots, gIgtotb;
double Igstot=0.0, cgshat=0.0, Igdtot=0.0, cgdhat=0.0, Igbtot=0.0, cgbhat=0.0;
double Vgs_eff, Vfb=0.0, Vth_NarrowW;
double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth, dVth_dVb, dVth_dVd;
double Vgst, dVgst_dVg, dVgst_dVb, dVgs_eff_dVg, Nvtms, Nvtmd;
double Vtm, Vtm0;
double n, dn_dVb, dn_dVd, voffcv, noff, dnoff_dVd, dnoff_dVb;
double V0, CoxWLcen, QovCox, LINK;
double DeltaPhi, dDeltaPhi_dVg, VgDP, dVgDP_dVg;
double Cox, Tox, Tcen, dTcen_dVg, dTcen_dVd, dTcen_dVb;
double Ccen, Coxeff, dCoxeff_dVd, dCoxeff_dVg, dCoxeff_dVb;
double Denomi, dDenomi_dVg, dDenomi_dVd, dDenomi_dVb;
double ueff, dueff_dVg, dueff_dVd, dueff_dVb; 
double Esat, Vdsat;
double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb;
double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, Vasat, dAlphaz_dVg, dAlphaz_dVb; 
double dVasat_dVg, dVasat_dVb, dVasat_dVd, Va, dVa_dVd, dVa_dVg, dVa_dVb; 
double Vbseff, dVbseff_dVb, VbseffCV, dVbseffCV_dVb; 
double Arg1, One_Third_CoxWL, Two_Third_CoxWL, Alphaz, CoxWL; 
double T0=0.0, dT0_dVg, dT0_dVd, dT0_dVb;
double T1, dT1_dVg, dT1_dVd, dT1_dVb;
double T2, dT2_dVg, dT2_dVd, dT2_dVb;
double T3, dT3_dVg, dT3_dVd, dT3_dVb;
double T4, dT4_dVd, dT4_dVb;
double T5, dT5_dVg, dT5_dVd, dT5_dVb;
double T6, dT6_dVg, dT6_dVd, dT6_dVb;
double T7, dT7_dVg, dT7_dVd, dT7_dVb;
double T8, dT8_dVg, dT8_dVd, dT8_dVb;
double T9, dT9_dVg, dT9_dVd, dT9_dVb;
double T10, dT10_dVg, dT10_dVb, dT10_dVd; 
double T11, T12, T13, T14;
double tmp, Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb;
double Cclm, dCclm_dVg, dCclm_dVd, dCclm_dVb;
double FP, dFP_dVg, PvagTerm, dPvagTerm_dVg, dPvagTerm_dVd, dPvagTerm_dVb;
double VADITS, dVADITS_dVg, dVADITS_dVd;
double Lpe_Vb, dDITS_Sft_dVb, dDITS_Sft_dVd;
double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb;
double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb;
double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb, Delt_vth, dDelt_vth_dVb;
double Theta0, dTheta0_dVb;
double TempRatio, tmp1, tmp2, tmp3, tmp4;
double DIBL_Sft, dDIBL_Sft_dVd, Lambda, dLambda_dVg;
double Idtot, Ibtot, a1, ScalingFactor;

double Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb; 
double Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb; 
double VdseffCV, dVdseffCV_dVg, dVdseffCV_dVd, dVdseffCV_dVb; 
double diffVds, dAbulk_dVg;
double beta, dbeta_dVg, dbeta_dVd, dbeta_dVb;
double gche, dgche_dVg, dgche_dVd, dgche_dVb;
double fgche1, dfgche1_dVg, dfgche1_dVd, dfgche1_dVb;
double fgche2, dfgche2_dVg, dfgche2_dVd, dfgche2_dVb;
double Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb;
double Idsa, dIdsa_dVg, dIdsa_dVd, dIdsa_dVb;
double Ids, Gm, Gds, Gmb, devbs_dvb, devbd_dvb;
double Isub, Gbd, Gbg, Gbb;
double VASCBE, dVASCBE_dVg, dVASCBE_dVd, dVASCBE_dVb;
double CoxeffWovL;
double Rds, dRds_dVg, dRds_dVb, WVCox, WVCoxRds;
double Vgst2Vtm, VdsatCV;
double Leff, Weff, dWeff_dVg, dWeff_dVb;
double AbulkCV, dAbulkCV_dVb;
double qcheq, qdef, gqdef=0.0, cqdef=0.0, cqcheq=0.0;
double gcqdb=0.0, gcqsb=0.0, gcqgb=0.0, gcqbb=0.0;
double dxpart, sxpart, ggtg, ggtd, ggts, ggtb;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb, gbspdp; 
double gbdpdp, gbdpg, gbdpb, gbdpsp; 
double qgdo, qgso, cgdo, cgso;
double Cgg, Cgd, Cgb, Cdg, Cdd, Cds;
double Csg, Csd, Css, Csb, Cbg, Cbd, Cbb;
double Cgg1, Cgb1, Cgd1, Cbg1, Cbb1, Cbd1, Qac0, Qsub0;
double dQac0_dVg, dQac0_dVb, dQsub0_dVg, dQsub0_dVd, dQsub0_dVb;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;
double Igisl, Ggislg, Ggislb, Ggisls;
double Nvtmrs, Nvtmrssw, Nvtmrsswg;

double vs, Fsevl, dvs_dVg, dvs_dVd, dvs_dVb, dFsevl_dVg, dFsevl_dVd, dFsevl_dVb;
double vgdx, vgsx;
struct bsim4v5SizeDependParam *pParam;
int ByPass, ChargeComputationNeeded, error, Check, Check1, Check2;

 double m;

ScalingFactor = 1.0e-9;
ChargeComputationNeeded =  
                 ((ckt->CKTmode & (MODEDCTRANCURVE | MODEAC | MODETRAN | MODEINITSMSIG)) ||
                 ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
                 ? 1 : 0;

#ifndef USE_OMP
for (; model != NULL; model = BSIM4v5nextModel(model))
{    for (here = BSIM4v5instances(model); here != NULL; 
          here = BSIM4v5nextInstance(here))
     {
#endif

          Check = Check1 = Check2 = 1;
          ByPass = 0;
	  pParam = here->pParam;

          if ((ckt->CKTmode & MODEINITSMSIG))
	  {   vds = *(ckt->CKTstate0 + here->BSIM4v5vds);
              vgs = *(ckt->CKTstate0 + here->BSIM4v5vgs);
              vbs = *(ckt->CKTstate0 + here->BSIM4v5vbs);
              vges = *(ckt->CKTstate0 + here->BSIM4v5vges);
              vgms = *(ckt->CKTstate0 + here->BSIM4v5vgms);
              vdbs = *(ckt->CKTstate0 + here->BSIM4v5vdbs);
              vsbs = *(ckt->CKTstate0 + here->BSIM4v5vsbs);
              vses = *(ckt->CKTstate0 + here->BSIM4v5vses);
              vdes = *(ckt->CKTstate0 + here->BSIM4v5vdes);

              qdef = *(ckt->CKTstate0 + here->BSIM4v5qdef);
          }
	  else if ((ckt->CKTmode & MODEINITTRAN))
	  {   vds = *(ckt->CKTstate1 + here->BSIM4v5vds);
              vgs = *(ckt->CKTstate1 + here->BSIM4v5vgs);
              vbs = *(ckt->CKTstate1 + here->BSIM4v5vbs);
              vges = *(ckt->CKTstate1 + here->BSIM4v5vges);
              vgms = *(ckt->CKTstate1 + here->BSIM4v5vgms);
              vdbs = *(ckt->CKTstate1 + here->BSIM4v5vdbs);
              vsbs = *(ckt->CKTstate1 + here->BSIM4v5vsbs);
              vses = *(ckt->CKTstate1 + here->BSIM4v5vses);
              vdes = *(ckt->CKTstate1 + here->BSIM4v5vdes);

              qdef = *(ckt->CKTstate1 + here->BSIM4v5qdef);
          }
	  else if ((ckt->CKTmode & MODEINITJCT) && !here->BSIM4v5off)
	  {   vds = model->BSIM4v5type * here->BSIM4v5icVDS;
              vgs = vges = vgms = model->BSIM4v5type * here->BSIM4v5icVGS;
              vbs = vdbs = vsbs = model->BSIM4v5type * here->BSIM4v5icVBS;
	      if (vds > 0.0)
	      {   vdes = vds + 0.01;
		  vses = -0.01;
	      }
	      else if (vds < 0.0)
              {   vdes = vds - 0.01;
                  vses = 0.01;
              }
	      else
	          vdes = vses = 0.0;

              qdef = 0.0;

              if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) &&
                  ((ckt->CKTmode & (MODETRAN | MODEAC|MODEDCOP |
                   MODEDCTRANCURVE)) || (!(ckt->CKTmode & MODEUIC))))
	      {   vds = 0.1;
		  vdes = 0.11;
	  	  vses = -0.01;
                  vgs = vges = vgms = model->BSIM4v5type 
                                    * here->BSIM4v5vth0 + 0.1;
                  vbs = vdbs = vsbs = 0.0;
              }
          }
	  else if ((ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) && 
                  (here->BSIM4v5off)) 
          {   vds = vgs = vbs = vges = vgms = 0.0;
              vdbs = vsbs = vdes = vses = qdef = 0.0;
	  }
          else
	  {
#ifndef PREDICTOR
               if ((ckt->CKTmode & MODEINITPRED))
	       {   xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
                   *(ckt->CKTstate0 + here->BSIM4v5vds) = 
                         *(ckt->CKTstate1 + here->BSIM4v5vds);
                   vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vds))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vds)));
                   *(ckt->CKTstate0 + here->BSIM4v5vgs) = 
                         *(ckt->CKTstate1 + here->BSIM4v5vgs);
                   vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vgs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vgs)));
                   *(ckt->CKTstate0 + here->BSIM4v5vges) =
                         *(ckt->CKTstate1 + here->BSIM4v5vges);
                   vges = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vges))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vges)));
                   *(ckt->CKTstate0 + here->BSIM4v5vgms) =
                         *(ckt->CKTstate1 + here->BSIM4v5vgms);
                   vgms = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vgms))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vgms)));
                   *(ckt->CKTstate0 + here->BSIM4v5vbs) = 
                         *(ckt->CKTstate1 + here->BSIM4v5vbs);
                   vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vbs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vbs)));
                   *(ckt->CKTstate0 + here->BSIM4v5vbd) = 
                         *(ckt->CKTstate0 + here->BSIM4v5vbs)
                         - *(ckt->CKTstate0 + here->BSIM4v5vds);
                   *(ckt->CKTstate0 + here->BSIM4v5vdbs) =
                         *(ckt->CKTstate1 + here->BSIM4v5vdbs);
                   vdbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vdbs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vdbs)));
                   *(ckt->CKTstate0 + here->BSIM4v5vdbd) =
                         *(ckt->CKTstate0 + here->BSIM4v5vdbs)
                         - *(ckt->CKTstate0 + here->BSIM4v5vds);
                   *(ckt->CKTstate0 + here->BSIM4v5vsbs) =
                         *(ckt->CKTstate1 + here->BSIM4v5vsbs);
                   vsbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vsbs))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vsbs)));
                   *(ckt->CKTstate0 + here->BSIM4v5vses) =
                         *(ckt->CKTstate1 + here->BSIM4v5vses);
                   vses = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vses))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vses)));
                   *(ckt->CKTstate0 + here->BSIM4v5vdes) =
                         *(ckt->CKTstate1 + here->BSIM4v5vdes);
                   vdes = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5vdes))
                         - (xfact * (*(ckt->CKTstate2 + here->BSIM4v5vdes)));

                   *(ckt->CKTstate0 + here->BSIM4v5qdef) =
                         *(ckt->CKTstate1 + here->BSIM4v5qdef);
                   qdef = (1.0 + xfact)* (*(ckt->CKTstate1 + here->BSIM4v5qdef))
                        -(xfact * (*(ckt->CKTstate2 + here->BSIM4v5qdef)));
               }
	       else
	       {
#endif /* PREDICTOR */
                   vds = model->BSIM4v5type
                       * (*(ckt->CKTrhsOld + here->BSIM4v5dNodePrime)
                       - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vgs = model->BSIM4v5type
                       * (*(ckt->CKTrhsOld + here->BSIM4v5gNodePrime) 
                       - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vbs = model->BSIM4v5type
                       * (*(ckt->CKTrhsOld + here->BSIM4v5bNodePrime)
                       - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vges = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5gNodeExt)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vgms = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5gNodeMid)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vdbs = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5dbNode)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vsbs = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5sbNode)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vses = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5sNode)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   vdes = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5dNode)
                        - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
                   qdef = model->BSIM4v5type
                        * (*(ckt->CKTrhsOld + here->BSIM4v5qNode));
#ifndef PREDICTOR
               }
#endif /* PREDICTOR */

               vgdo = *(ckt->CKTstate0 + here->BSIM4v5vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v5vds);
	       vgedo = *(ckt->CKTstate0 + here->BSIM4v5vges)
                     - *(ckt->CKTstate0 + here->BSIM4v5vds);
	       vgmdo = *(ckt->CKTstate0 + here->BSIM4v5vgms)
                     - *(ckt->CKTstate0 + here->BSIM4v5vds);

               vbd = vbs - vds;
               vdbd = vdbs - vds;
               vgd = vgs - vds;
  	       vged = vges - vds;
  	       vgmd = vgms - vds;

               delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v5vbd);
               delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v5vdbd);
               delvgd = vgd - vgdo;
	       delvged = vged - vgedo;
	       delvgmd = vgmd - vgmdo;

               delvds = vds - *(ckt->CKTstate0 + here->BSIM4v5vds);
               delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v5vgs);
	       delvges = vges - *(ckt->CKTstate0 + here->BSIM4v5vges);
	       delvgms = vgms - *(ckt->CKTstate0 + here->BSIM4v5vgms);
               delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v5vbs);
               delvdbs = vdbs - *(ckt->CKTstate0 + here->BSIM4v5vdbs);
               delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v5vsbs);

               delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v5vses));
               vdedo = *(ckt->CKTstate0 + here->BSIM4v5vdes)
                     - *(ckt->CKTstate0 + here->BSIM4v5vds);
	       delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v5vdes);
               delvded = vdes - vds - vdedo;

               delvbd_jct = (!here->BSIM4v5rbodyMod) ? delvbd : delvdbd;
               delvbs_jct = (!here->BSIM4v5rbodyMod) ? delvbs : delvsbs;
               if (here->BSIM4v5mode >= 0)
               {   Idtot = here->BSIM4v5cd + here->BSIM4v5csub - here->BSIM4v5cbd
			 + here->BSIM4v5Igidl;
                   cdhat = Idtot - here->BSIM4v5gbd * delvbd_jct
                         + (here->BSIM4v5gmbs + here->BSIM4v5gbbs + here->BSIM4v5ggidlb) * delvbs
                         + (here->BSIM4v5gm + here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs 
                         + (here->BSIM4v5gds + here->BSIM4v5gbds + here->BSIM4v5ggidld) * delvds;
                   Ibtot = here->BSIM4v5cbs + here->BSIM4v5cbd 
			 - here->BSIM4v5Igidl - here->BSIM4v5Igisl - here->BSIM4v5csub;
                   cbhat = Ibtot + here->BSIM4v5gbd * delvbd_jct
                         + here->BSIM4v5gbs * delvbs_jct - (here->BSIM4v5gbbs + here->BSIM4v5ggidlb)
                         * delvbs - (here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs
			 - (here->BSIM4v5gbds + here->BSIM4v5ggidld - here->BSIM4v5ggisls) * delvds 
			 - here->BSIM4v5ggislg * delvgd - here->BSIM4v5ggislb* delvbd;

		   Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcs;
		   cgshat = Igstot + (here->BSIM4v5gIgsg + here->BSIM4v5gIgcsg) * delvgs
			  + here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbs;

		   Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcd;
		   cgdhat = Igdtot + here->BSIM4v5gIgdg * delvgd + here->BSIM4v5gIgcdg * delvgs
                          + here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbs;

 		   Igbtot = here->BSIM4v5Igb;
		   cgbhat = here->BSIM4v5Igb + here->BSIM4v5gIgbg * delvgs + here->BSIM4v5gIgbd
			  * delvds + here->BSIM4v5gIgbb * delvbs;
               }
               else
               {   Idtot = here->BSIM4v5cd + here->BSIM4v5cbd - here->BSIM4v5Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4v5gbd * delvbd_jct + here->BSIM4v5gmbs 
                         * delvbd + here->BSIM4v5gm * delvgd 
                         - (here->BSIM4v5gds + here->BSIM4v5ggidls) * delvds 
                         - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs;
                   Ibtot = here->BSIM4v5cbs + here->BSIM4v5cbd 
			 - here->BSIM4v5Igidl - here->BSIM4v5Igisl - here->BSIM4v5csub;
                   cbhat = Ibtot + here->BSIM4v5gbs * delvbs_jct + here->BSIM4v5gbd 
                         * delvbd_jct - (here->BSIM4v5gbbs + here->BSIM4v5ggislb) * delvbd
                         - (here->BSIM4v5gbgs + here->BSIM4v5ggislg) * delvgd
			 + (here->BSIM4v5gbds + here->BSIM4v5ggisld - here->BSIM4v5ggidls) * delvds
			 - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs; 

                   Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcd;
                   cgshat = Igstot + here->BSIM4v5gIgsg * delvgs + here->BSIM4v5gIgcdg * delvgd
                          - here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbd;

                   Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcs;
                   cgdhat = Igdtot + (here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg) * delvgd
                          - here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbd;

                   Igbtot = here->BSIM4v5Igb;
                   cgbhat = here->BSIM4v5Igb + here->BSIM4v5gIgbg * delvgd - here->BSIM4v5gIgbd
                          * delvds + here->BSIM4v5gIgbb * delvbd;
               }

               Isestot = here->BSIM4v5gstot * (*(ckt->CKTstate0 + here->BSIM4v5vses));
               cseshat = Isestot + here->BSIM4v5gstot * delvses
                       + here->BSIM4v5gstotd * delvds + here->BSIM4v5gstotg * delvgs
                       + here->BSIM4v5gstotb * delvbs;

               Idedtot = here->BSIM4v5gdtot * vdedo;
               cdedhat = Idedtot + here->BSIM4v5gdtot * delvded
                       + here->BSIM4v5gdtotd * delvds + here->BSIM4v5gdtotg * delvgs
                       + here->BSIM4v5gdtotb * delvbs;


#ifndef NOBYPASS
               /* Following should be one IF statement, but some C compilers 
                * can't handle that all at once, so we split it into several
                * successive IF's */

               if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass))
               if ((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                   fabs(*(ckt->CKTstate0 + here->BSIM4v5vds))) + ckt->CKTvoltTol)))
               if ((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                   fabs(*(ckt->CKTstate0 + here->BSIM4v5vgs))) + ckt->CKTvoltTol)))
               if ((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                   fabs(*(ckt->CKTstate0 + here->BSIM4v5vbs))) + ckt->CKTvoltTol)))
               if ((fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                   fabs(*(ckt->CKTstate0 + here->BSIM4v5vbd))) + ckt->CKTvoltTol)))
	       if ((here->BSIM4v5rgateMod == 0) || (here->BSIM4v5rgateMod == 1) 
      		   || (fabs(delvges) < (ckt->CKTreltol * MAX(fabs(vges),
		   fabs(*(ckt->CKTstate0 + here->BSIM4v5vges))) + ckt->CKTvoltTol)))
	       if ((here->BSIM4v5rgateMod != 3) || (fabs(delvgms) < (ckt->CKTreltol
                   * MAX(fabs(vgms), fabs(*(ckt->CKTstate0 + here->BSIM4v5vgms)))
                   + ckt->CKTvoltTol)))
               if ((!here->BSIM4v5rbodyMod) || (fabs(delvdbs) < (ckt->CKTreltol
                   * MAX(fabs(vdbs), fabs(*(ckt->CKTstate0 + here->BSIM4v5vdbs)))
                   + ckt->CKTvoltTol)))
               if ((!here->BSIM4v5rbodyMod) || (fabs(delvdbd) < (ckt->CKTreltol
                   * MAX(fabs(vdbd), fabs(*(ckt->CKTstate0 + here->BSIM4v5vdbd)))
                   + ckt->CKTvoltTol)))
               if ((!here->BSIM4v5rbodyMod) || (fabs(delvsbs) < (ckt->CKTreltol
                   * MAX(fabs(vsbs), fabs(*(ckt->CKTstate0 + here->BSIM4v5vsbs)))
                   + ckt->CKTvoltTol)))
               if ((!model->BSIM4v5rdsMod) || (fabs(delvses) < (ckt->CKTreltol
                   * MAX(fabs(vses), fabs(*(ckt->CKTstate0 + here->BSIM4v5vses)))
                   + ckt->CKTvoltTol)))
               if ((!model->BSIM4v5rdsMod) || (fabs(delvdes) < (ckt->CKTreltol
                   * MAX(fabs(vdes), fabs(*(ckt->CKTstate0 + here->BSIM4v5vdes)))
                   + ckt->CKTvoltTol)))
               if ((fabs(cdhat - Idtot) < ckt->CKTreltol
                   * MAX(fabs(cdhat), fabs(Idtot)) + ckt->CKTabstol))
               if ((fabs(cbhat - Ibtot) < ckt->CKTreltol
                   * MAX(fabs(cbhat), fabs(Ibtot)) + ckt->CKTabstol))
               if ((!model->BSIM4v5igcMod) || ((fabs(cgshat - Igstot) < ckt->CKTreltol
                   * MAX(fabs(cgshat), fabs(Igstot)) + ckt->CKTabstol)))
               if ((!model->BSIM4v5igcMod) || ((fabs(cgdhat - Igdtot) < ckt->CKTreltol
                   * MAX(fabs(cgdhat), fabs(Igdtot)) + ckt->CKTabstol)))
               if ((!model->BSIM4v5igbMod) || ((fabs(cgbhat - Igbtot) < ckt->CKTreltol
                   * MAX(fabs(cgbhat), fabs(Igbtot)) + ckt->CKTabstol)))
               if ((!model->BSIM4v5rdsMod) || ((fabs(cseshat - Isestot) < ckt->CKTreltol
                   * MAX(fabs(cseshat), fabs(Isestot)) + ckt->CKTabstol)))
               if ((!model->BSIM4v5rdsMod) || ((fabs(cdedhat - Idedtot) < ckt->CKTreltol
                   * MAX(fabs(cdedhat), fabs(Idedtot)) + ckt->CKTabstol)))
               {   vds = *(ckt->CKTstate0 + here->BSIM4v5vds);
                   vgs = *(ckt->CKTstate0 + here->BSIM4v5vgs);
                   vbs = *(ckt->CKTstate0 + here->BSIM4v5vbs);
		   vges = *(ckt->CKTstate0 + here->BSIM4v5vges);
		   vgms = *(ckt->CKTstate0 + here->BSIM4v5vgms);

                   vbd = *(ckt->CKTstate0 + here->BSIM4v5vbd);
                   vdbs = *(ckt->CKTstate0 + here->BSIM4v5vdbs);
                   vdbd = *(ckt->CKTstate0 + here->BSIM4v5vdbd);
                   vsbs = *(ckt->CKTstate0 + here->BSIM4v5vsbs);
                   vses = *(ckt->CKTstate0 + here->BSIM4v5vses);
                   vdes = *(ckt->CKTstate0 + here->BSIM4v5vdes);

                   vgd = vgs - vds;
                   vgb = vgs - vbs;
		   vged = vges - vds;
		   vgmd = vgms - vds;
		   vgmb = vgms - vbs;

                   vbs_jct = (!here->BSIM4v5rbodyMod) ? vbs : vsbs;
                   vbd_jct = (!here->BSIM4v5rbodyMod) ? vbd : vdbd;

/*** qdef should not be kept fixed even if vgs, vds & vbs has converged 
****               qdef = *(ckt->CKTstate0 + here->BSIM4v5qdef);  
***/
                   cdrain = here->BSIM4v5cd;

                   if ((ckt->CKTmode & (MODETRAN | MODEAC)) || 
                       ((ckt->CKTmode & MODETRANOP) && 
                       (ckt->CKTmode & MODEUIC)))
		   {   ByPass = 1;

                       qgate = here->BSIM4v5qgate;
                       qbulk = here->BSIM4v5qbulk;
                       qdrn = here->BSIM4v5qdrn;
		       cgdo = here->BSIM4v5cgdo;
		       qgdo = here->BSIM4v5qgdo;
                       cgso = here->BSIM4v5cgso;
                       qgso = here->BSIM4v5qgso;

		       goto line755;
                    }
		    else
		       goto line850;
               }
#endif /*NOBYPASS*/

               von = here->BSIM4v5von;
               if (*(ckt->CKTstate0 + here->BSIM4v5vds) >= 0.0)
	       {   vgs = DEVfetlim(vgs, *(ckt->CKTstate0 + here->BSIM4v5vgs), von);
                   vds = vgs - vgd;
                   vds = DEVlimvds(vds, *(ckt->CKTstate0 + here->BSIM4v5vds));
                   vgd = vgs - vds;
                   if (here->BSIM4v5rgateMod == 3)
                   {   vges = DEVfetlim(vges, *(ckt->CKTstate0 + here->BSIM4v5vges), von);
                       vgms = DEVfetlim(vgms, *(ckt->CKTstate0 + here->BSIM4v5vgms), von);
                       vged = vges - vds;
                       vgmd = vgms - vds;
                   }
                   else if ((here->BSIM4v5rgateMod == 1) || (here->BSIM4v5rgateMod == 2))
                   {   vges = DEVfetlim(vges, *(ckt->CKTstate0 + here->BSIM4v5vges), von);
                       vged = vges - vds;
                   }

                   if (model->BSIM4v5rdsMod)
		   {   vdes = DEVlimvds(vdes, *(ckt->CKTstate0 + here->BSIM4v5vdes));
		       vses = -DEVlimvds(-vses, -(*(ckt->CKTstate0 + here->BSIM4v5vses)));
		   }

               }
	       else
	       {   vgd = DEVfetlim(vgd, vgdo, von);
                   vds = vgs - vgd;
                   vds = -DEVlimvds(-vds, -(*(ckt->CKTstate0 + here->BSIM4v5vds)));
                   vgs = vgd + vds;

		   if (here->BSIM4v5rgateMod == 3)
                   {   vged = DEVfetlim(vged, vgedo, von);
                       vges = vged + vds;
                       vgmd = DEVfetlim(vgmd, vgmdo, von);
                       vgms = vgmd + vds;
                   }
                   if ((here->BSIM4v5rgateMod == 1) || (here->BSIM4v5rgateMod == 2))
                   {   vged = DEVfetlim(vged, vgedo, von);
                       vges = vged + vds;
                   }

                   if (model->BSIM4v5rdsMod)
                   {   vdes = -DEVlimvds(-vdes, -(*(ckt->CKTstate0 + here->BSIM4v5vdes)));
                       vses = DEVlimvds(vses, *(ckt->CKTstate0 + here->BSIM4v5vses));
                   }
               }

               if (vds >= 0.0)
	       {   vbs = DEVpnjlim(vbs, *(ckt->CKTstate0 + here->BSIM4v5vbs),
                                   CONSTvt0, model->BSIM4v5vcrit, &Check);
                   vbd = vbs - vds;
                   if (here->BSIM4v5rbodyMod)
                   {   vdbs = DEVpnjlim(vdbs, *(ckt->CKTstate0 + here->BSIM4v5vdbs),
                                        CONSTvt0, model->BSIM4v5vcrit, &Check1);
                       vdbd = vdbs - vds;
                       vsbs = DEVpnjlim(vsbs, *(ckt->CKTstate0 + here->BSIM4v5vsbs),
                                        CONSTvt0, model->BSIM4v5vcrit, &Check2);
                       if ((Check1 == 0) && (Check2 == 0))
                           Check = 0;
                       else 
                           Check = 1;
                   }
               }
	       else
               {   vbd = DEVpnjlim(vbd, *(ckt->CKTstate0 + here->BSIM4v5vbd),
                                   CONSTvt0, model->BSIM4v5vcrit, &Check); 
                   vbs = vbd + vds;
                   if (here->BSIM4v5rbodyMod)
                   {   vdbd = DEVpnjlim(vdbd, *(ckt->CKTstate0 + here->BSIM4v5vdbd),
                                        CONSTvt0, model->BSIM4v5vcrit, &Check1);
                       vdbs = vdbd + vds;
                       vsbdo = *(ckt->CKTstate0 + here->BSIM4v5vsbs)
                             - *(ckt->CKTstate0 + here->BSIM4v5vds);
                       vsbd = vsbs - vds;
                       vsbd = DEVpnjlim(vsbd, vsbdo, CONSTvt0, model->BSIM4v5vcrit, &Check2);
                       vsbs = vsbd + vds;
                       if ((Check1 == 0) && (Check2 == 0))
                           Check = 0;
                       else
                           Check = 1;
                   }
               }
          }

          /* Calculate DC currents and their derivatives */
          vbd = vbs - vds;
          vgd = vgs - vds;
          vgb = vgs - vbs;
	  vged = vges - vds;
	  vgmd = vgms - vds;
	  vgmb = vgms - vbs;
          vdbd = vdbs - vds;

          vbs_jct = (!here->BSIM4v5rbodyMod) ? vbs : vsbs;
          vbd_jct = (!here->BSIM4v5rbodyMod) ? vbd : vdbd;

          /* Source/drain junction diode DC model begins */
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

	  if (SourceSatCurrent <= 0.0)
	  {   here->BSIM4v5gbs = ckt->CKTgmin;
              here->BSIM4v5cbs = here->BSIM4v5gbs * vbs_jct;
          }
          else
	  {   switch(model->BSIM4v5dioMod)
              {   case 0:
                      evbs = exp(vbs_jct / Nvtms);
                      T1 = model->BSIM4v5xjbvs * exp(-(model->BSIM4v5bvs + vbs_jct) / Nvtms);
		      /* WDLiu: Magic T1 in this form; different from BSIM4v5 beta. */
		      here->BSIM4v5gbs = SourceSatCurrent * (evbs + T1) / Nvtms + ckt->CKTgmin;
		      here->BSIM4v5cbs = SourceSatCurrent * (evbs + here->BSIM4v5XExpBVS
				     - T1 - 1.0) + ckt->CKTgmin * vbs_jct;
		      break;
                  case 1:
		      T2 = vbs_jct / Nvtms;
	              if (T2 < -EXP_THRESHOLD)
		      {   here->BSIM4v5gbs = ckt->CKTgmin;
                          here->BSIM4v5cbs = SourceSatCurrent * (MIN_EXP - 1.0)
                                         + ckt->CKTgmin * vbs_jct;
                      }
		      else if (vbs_jct <= here->BSIM4v5vjsmFwd)
		      {   evbs = exp(T2);
			  here->BSIM4v5gbs = SourceSatCurrent * evbs / Nvtms + ckt->CKTgmin;
                          here->BSIM4v5cbs = SourceSatCurrent * (evbs - 1.0)
                                         + ckt->CKTgmin * vbs_jct;
	              }
		      else
		      {   T0 = here->BSIM4v5IVjsmFwd / Nvtms;
                          here->BSIM4v5gbs = T0 + ckt->CKTgmin;
                          here->BSIM4v5cbs = here->BSIM4v5IVjsmFwd - SourceSatCurrent + T0 
					 * (vbs_jct - here->BSIM4v5vjsmFwd) + ckt->CKTgmin * vbs_jct;
		      }	
                      break;
                  case 2:
                      if (vbs_jct < here->BSIM4v5vjsmRev)
                      {   T0 = vbs_jct / Nvtms;
                          if (T0 < -EXP_THRESHOLD)
                          {    evbs = MIN_EXP;
			       devbs_dvb = 0.0;
			  }
                          else
			  {    evbs = exp(T0);
                               devbs_dvb = evbs / Nvtms;
			  }

			  T1 = evbs - 1.0;
			  T2 = here->BSIM4v5IVjsmRev + here->BSIM4v5SslpRev
			     * (vbs_jct - here->BSIM4v5vjsmRev);
			  here->BSIM4v5gbs = devbs_dvb * T2 + T1 * here->BSIM4v5SslpRev + ckt->CKTgmin;
                          here->BSIM4v5cbs = T1 * T2 + ckt->CKTgmin * vbs_jct;
                      }         
                      else if (vbs_jct <= here->BSIM4v5vjsmFwd)
                      {   T0 = vbs_jct / Nvtms;
                          if (T0 < -EXP_THRESHOLD)
                          {    evbs = MIN_EXP;
                               devbs_dvb = 0.0;
                          }
                          else
                          {    evbs = exp(T0);
                               devbs_dvb = evbs / Nvtms;
                          }

			  T1 = (model->BSIM4v5bvs + vbs_jct) / Nvtms;
                          if (T1 > EXP_THRESHOLD)
                          {   T2 = MIN_EXP;
			      T3 = 0.0;
			  }
                          else
			  {   T2 = exp(-T1);
			      T3 = -T2 /Nvtms;
			  }
                          here->BSIM4v5gbs = SourceSatCurrent * (devbs_dvb - model->BSIM4v5xjbvs * T3)
					 + ckt->CKTgmin;
			  here->BSIM4v5cbs = SourceSatCurrent * (evbs + here->BSIM4v5XExpBVS - 1.0
				         - model->BSIM4v5xjbvs * T2) + ckt->CKTgmin * vbs_jct;
                      }
		      else
		      {   here->BSIM4v5gbs = here->BSIM4v5SslpFwd + ckt->CKTgmin;
                          here->BSIM4v5cbs = here->BSIM4v5IVjsmFwd + here->BSIM4v5SslpFwd * (vbs_jct
					 - here->BSIM4v5vjsmFwd) + ckt->CKTgmin * vbs_jct;
		      }
                      break;
                  default: break;
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

	  if (DrainSatCurrent <= 0.0)
	  {   here->BSIM4v5gbd = ckt->CKTgmin;
              here->BSIM4v5cbd = here->BSIM4v5gbd * vbd_jct;
          }
          else
          {   switch(model->BSIM4v5dioMod)
              {   case 0:
                      evbd = exp(vbd_jct / Nvtmd);
                      T1 = model->BSIM4v5xjbvd * exp(-(model->BSIM4v5bvd + vbd_jct) / Nvtmd);
                      /* WDLiu: Magic T1 in this form; different from BSIM4v5 beta. */
                      here->BSIM4v5gbd = DrainSatCurrent * (evbd + T1) / Nvtmd + ckt->CKTgmin;
                      here->BSIM4v5cbd = DrainSatCurrent * (evbd + here->BSIM4v5XExpBVD
                                     - T1 - 1.0) + ckt->CKTgmin * vbd_jct;
                      break;
                  case 1:
		      T2 = vbd_jct / Nvtmd;
                      if (T2 < -EXP_THRESHOLD)
                      {   here->BSIM4v5gbd = ckt->CKTgmin;
                          here->BSIM4v5cbd = DrainSatCurrent * (MIN_EXP - 1.0)
                                         + ckt->CKTgmin * vbd_jct;
                      }
                      else if (vbd_jct <= here->BSIM4v5vjdmFwd)
                      {   evbd = exp(T2);
                          here->BSIM4v5gbd = DrainSatCurrent * evbd / Nvtmd + ckt->CKTgmin;
                          here->BSIM4v5cbd = DrainSatCurrent * (evbd - 1.0)
                                         + ckt->CKTgmin * vbd_jct;
                      }
                      else
                      {   T0 = here->BSIM4v5IVjdmFwd / Nvtmd;
                          here->BSIM4v5gbd = T0 + ckt->CKTgmin;
                          here->BSIM4v5cbd = here->BSIM4v5IVjdmFwd - DrainSatCurrent + T0
                                         * (vbd_jct - here->BSIM4v5vjdmFwd) + ckt->CKTgmin * vbd_jct;
                      }
                      break;
                  case 2:
                      if (vbd_jct < here->BSIM4v5vjdmRev)
                      {   T0 = vbd_jct / Nvtmd;
                          if (T0 < -EXP_THRESHOLD)
                          {    evbd = MIN_EXP;
                               devbd_dvb = 0.0;
                          }
                          else
                          {    evbd = exp(T0);
                               devbd_dvb = evbd / Nvtmd;
                          }

                          T1 = evbd - 1.0;
                          T2 = here->BSIM4v5IVjdmRev + here->BSIM4v5DslpRev
                             * (vbd_jct - here->BSIM4v5vjdmRev);
                          here->BSIM4v5gbd = devbd_dvb * T2 + T1 * here->BSIM4v5DslpRev + ckt->CKTgmin;
                          here->BSIM4v5cbd = T1 * T2 + ckt->CKTgmin * vbd_jct;
                      }
                      else if (vbd_jct <= here->BSIM4v5vjdmFwd)
                      {   T0 = vbd_jct / Nvtmd;
                          if (T0 < -EXP_THRESHOLD)
                          {    evbd = MIN_EXP;
                               devbd_dvb = 0.0;
                          }
                          else
                          {    evbd = exp(T0);
                               devbd_dvb = evbd / Nvtmd;
                          }

                          T1 = (model->BSIM4v5bvd + vbd_jct) / Nvtmd;
                          if (T1 > EXP_THRESHOLD)
                          {   T2 = MIN_EXP;
                              T3 = 0.0;
                          }
                          else
                          {   T2 = exp(-T1);
                              T3 = -T2 /Nvtmd;
                          }     
                          here->BSIM4v5gbd = DrainSatCurrent * (devbd_dvb - model->BSIM4v5xjbvd * T3)
                                         + ckt->CKTgmin;
                          here->BSIM4v5cbd = DrainSatCurrent * (evbd + here->BSIM4v5XExpBVD - 1.0
                                         - model->BSIM4v5xjbvd * T2) + ckt->CKTgmin * vbd_jct;
                      }
                      else
                      {   here->BSIM4v5gbd = here->BSIM4v5DslpFwd + ckt->CKTgmin;
                          here->BSIM4v5cbd = here->BSIM4v5IVjdmFwd + here->BSIM4v5DslpFwd * (vbd_jct
                                         - here->BSIM4v5vjdmFwd) + ckt->CKTgmin * vbd_jct;
                      }
                      break;
                  default: break;
              }
          } 

	   /* trap-assisted tunneling and recombination current for reverse bias  */
          Nvtmrssw = model->BSIM4v5vtm0 * model->BSIM4v5njtsswtemp;
          Nvtmrsswg = model->BSIM4v5vtm0 * model->BSIM4v5njtsswgtemp;
          Nvtmrs = model->BSIM4v5vtm0 * model->BSIM4v5njtstemp;

        if ((model->BSIM4v5vtss - vbs_jct) < (model->BSIM4v5vtss * 1e-3))
        { T9 = 1.0e3; 
          T0 = - vbs_jct / Nvtmrs * T9;
          DEXP(T0, T1, T10);
          dT1_dVb = T10 / Nvtmrs * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtss - vbs_jct);
          T0 = -vbs_jct / Nvtmrs * model->BSIM4v5vtss * T9;
          dT0_dVb = model->BSIM4v5vtss / Nvtmrs * (T9 + vbs_jct * T9 * T9) ;
          DEXP(T0, T1, T10);
          dT1_dVb = T10 * dT0_dVb;
        }

       if ((model->BSIM4v5vtsd - vbd_jct) < (model->BSIM4v5vtsd * 1e-3) )
        { T9 = 1.0e3;
          T0 = -vbd_jct / Nvtmrs * T9;
          DEXP(T0, T2, T10);
          dT2_dVb = T10 / Nvtmrs * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtsd - vbd_jct);
          T0 = -vbd_jct / Nvtmrs * model->BSIM4v5vtsd * T9;
          dT0_dVb = model->BSIM4v5vtsd / Nvtmrs * (T9 + vbd_jct * T9 * T9) ;
          DEXP(T0, T2, T10);
          dT2_dVb = T10 * dT0_dVb;
        }

        if ((model->BSIM4v5vtssws - vbs_jct) < (model->BSIM4v5vtssws * 1e-3) )
        { T9 = 1.0e3; 
          T0 = -vbs_jct / Nvtmrssw * T9;
          DEXP(T0, T3, T10);
          dT3_dVb = T10 / Nvtmrssw * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtssws - vbs_jct);
          T0 = -vbs_jct / Nvtmrssw * model->BSIM4v5vtssws * T9;
          dT0_dVb = model->BSIM4v5vtssws / Nvtmrssw * (T9 + vbs_jct * T9 * T9) ;
          DEXP(T0, T3, T10);
          dT3_dVb = T10 * dT0_dVb;
        }

        if ((model->BSIM4v5vtsswd - vbd_jct) < (model->BSIM4v5vtsswd * 1e-3) )
        { T9 = 1.0e3; 
          T0 = -vbd_jct / Nvtmrssw * T9;
          DEXP(T0, T4, T10);
          dT4_dVb = T10 / Nvtmrssw * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtsswd - vbd_jct);
          T0 = -vbd_jct / Nvtmrssw * model->BSIM4v5vtsswd * T9;
          dT0_dVb = model->BSIM4v5vtsswd / Nvtmrssw * (T9 + vbd_jct * T9 * T9) ;
          DEXP(T0, T4, T10);
          dT4_dVb = T10 * dT0_dVb;
        }

        if ((model->BSIM4v5vtsswgs - vbs_jct) < (model->BSIM4v5vtsswgs * 1e-3) )
        { T9 = 1.0e3; 
          T0 = -vbs_jct / Nvtmrsswg * T9;
          DEXP(T0, T5, T10);
          dT5_dVb = T10 / Nvtmrsswg * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtsswgs - vbs_jct);
          T0 = -vbs_jct / Nvtmrsswg * model->BSIM4v5vtsswgs * T9;
          dT0_dVb = model->BSIM4v5vtsswgs / Nvtmrsswg * (T9 + vbs_jct * T9 * T9) ;
          DEXP(T0, T5, T10);
          dT5_dVb = T10 * dT0_dVb;
        }

        if ((model->BSIM4v5vtsswgd - vbd_jct) < (model->BSIM4v5vtsswgd * 1e-3) )
        { T9 = 1.0e3; 
          T0 = -vbd_jct / Nvtmrsswg * T9;
          DEXP(T0, T6, T10);
          dT6_dVb = T10 / Nvtmrsswg * T9; 
        } else {
          T9 = 1.0 / (model->BSIM4v5vtsswgd - vbd_jct);
          T0 = -vbd_jct / Nvtmrsswg * model->BSIM4v5vtsswgd * T9;
          dT0_dVb = model->BSIM4v5vtsswgd / Nvtmrsswg * (T9 + vbd_jct * T9 * T9) ;
          DEXP(T0, T6, T10);
          dT6_dVb = T10 * dT0_dVb;
        }

	  here->BSIM4v5gbs += here->BSIM4v5SjctTempRevSatCur * dT1_dVb
	  			+ here->BSIM4v5SswTempRevSatCur * dT3_dVb
	  			+ here->BSIM4v5SswgTempRevSatCur * dT5_dVb; 
	  here->BSIM4v5cbs -= here->BSIM4v5SjctTempRevSatCur * (T1 - 1.0)
	  			+ here->BSIM4v5SswTempRevSatCur * (T3 - 1.0)
	  			+ here->BSIM4v5SswgTempRevSatCur * (T5 - 1.0); 
	  here->BSIM4v5gbd += here->BSIM4v5DjctTempRevSatCur * dT2_dVb
	  			+ here->BSIM4v5DswTempRevSatCur * dT4_dVb
	  			+ here->BSIM4v5DswgTempRevSatCur * dT6_dVb; 
	  here->BSIM4v5cbd -= here->BSIM4v5DjctTempRevSatCur * (T2 - 1.0) 
	  			+ here->BSIM4v5DswTempRevSatCur * (T4 - 1.0)
	  			+ here->BSIM4v5DswgTempRevSatCur * (T6 - 1.0); 

          /* End of diode DC model */

          if (vds >= 0.0)
	  {   here->BSIM4v5mode = 1;
              Vds = vds;
              Vgs = vgs;
              Vbs = vbs;
	      Vdb = vds - vbs;  /* WDLiu: for GIDL */
          }
	  else
	  {   here->BSIM4v5mode = -1;
              Vds = -vds;
              Vgs = vgd;
              Vbs = vbd;
	      Vdb = -vbs;
          }

	  T0 = Vbs - here->BSIM4v5vbsc - 0.001;
	  T1 = sqrt(T0 * T0 - 0.004 * here->BSIM4v5vbsc);
	  if (T0 >= 0.0)
	  {   Vbseff = here->BSIM4v5vbsc + 0.5 * (T0 + T1);
              dVbseff_dVb = 0.5 * (1.0 + T0 / T1);
	  }
	  else
	  {   T2 = -0.002 / (T1 - T0);
	      Vbseff = here->BSIM4v5vbsc * (1.0 + T2);
	      dVbseff_dVb = T2 * here->BSIM4v5vbsc / T1;
	  }

	/* JX: Correction to forward body bias  */
	  T9 = 0.95 * pParam->BSIM4v5phi;
	  T0 = T9 - Vbseff - 0.001;
	  T1 = sqrt(T0 * T0 + 0.004 * T9);
	  Vbseff = T9 - 0.5 * (T0 + T1);
          dVbseff_dVb *= 0.5 * (1.0 + T0 / T1);

          Phis = pParam->BSIM4v5phi - Vbseff;
          dPhis_dVb = -1.0;
          sqrtPhis = sqrt(Phis);
          dsqrtPhis_dVb = -0.5 / sqrtPhis; 

          Xdep = pParam->BSIM4v5Xdep0 * sqrtPhis / pParam->BSIM4v5sqrtPhi;
          dXdep_dVb = (pParam->BSIM4v5Xdep0 / pParam->BSIM4v5sqrtPhi)
		    * dsqrtPhis_dVb;

          Leff = pParam->BSIM4v5leff;
          Vtm = model->BSIM4v5vtm;
          Vtm0 = model->BSIM4v5vtm0;

          /* Vth Calculation */
          T3 = sqrt(Xdep);
          V0 = pParam->BSIM4v5vbi - pParam->BSIM4v5phi;

          T0 = pParam->BSIM4v5dvt2 * Vbseff;
          if (T0 >= - 0.5)
	  {   T1 = 1.0 + T0;
	      T2 = pParam->BSIM4v5dvt2;
	  }
	  else
	  {   T4 = 1.0 / (3.0 + 8.0 * T0);
	      T1 = (1.0 + 3.0 * T0) * T4; 
	      T2 = pParam->BSIM4v5dvt2 * T4 * T4;
	  }
          lt1 = model->BSIM4v5factor1 * T3 * T1;
          dlt1_dVb = model->BSIM4v5factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

          T0 = pParam->BSIM4v5dvt2w * Vbseff;
          if (T0 >= - 0.5)
	  {   T1 = 1.0 + T0;
	      T2 = pParam->BSIM4v5dvt2w;
	  }
	  else
	  {   T4 = 1.0 / (3.0 + 8.0 * T0);
	      T1 = (1.0 + 3.0 * T0) * T4; 
	      T2 = pParam->BSIM4v5dvt2w * T4 * T4;
	  }
          ltw = model->BSIM4v5factor1 * T3 * T1;
          dltw_dVb = model->BSIM4v5factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

          T0 = pParam->BSIM4v5dvt1 * Leff / lt1;
          if (T0 < EXP_THRESHOLD)
          {   T1 = exp(T0);
              T2 = T1 - 1.0;
              T3 = T2 * T2;
              T4 = T3 + 2.0 * T1 * MIN_EXP;
              Theta0 = T1 / T4;
              dT1_dVb = -T0 * T1 * dlt1_dVb / lt1;
              dTheta0_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
          }
          else
          {   Theta0 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
              dTheta0_dVb = 0.0;
          }
          here->BSIM4v5thetavth = pParam->BSIM4v5dvt0 * Theta0;
          Delt_vth = here->BSIM4v5thetavth * V0;
          dDelt_vth_dVb = pParam->BSIM4v5dvt0 * dTheta0_dVb * V0;

          T0 = pParam->BSIM4v5dvt1w * pParam->BSIM4v5weff * Leff / ltw;
          if (T0 < EXP_THRESHOLD)
          {   T1 = exp(T0);
              T2 = T1 - 1.0;
              T3 = T2 * T2;
              T4 = T3 + 2.0 * T1 * MIN_EXP;
              T5 = T1 / T4;
              dT1_dVb = -T0 * T1 * dltw_dVb / ltw; 
              dT5_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
          }
          else
          {   T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
              dT5_dVb = 0.0;
          }
          T0 = pParam->BSIM4v5dvt0w * T5;
          T2 = T0 * V0;
          dT2_dVb = pParam->BSIM4v5dvt0w * dT5_dVb * V0;

          TempRatio =  ckt->CKTtemp / model->BSIM4v5tnom - 1.0;
          T0 = sqrt(1.0 + pParam->BSIM4v5lpe0 / Leff);
          T1 = pParam->BSIM4v5k1ox * (T0 - 1.0) * pParam->BSIM4v5sqrtPhi
             + (pParam->BSIM4v5kt1 + pParam->BSIM4v5kt1l / Leff
             + pParam->BSIM4v5kt2 * Vbseff) * TempRatio;
          Vth_NarrowW = model->BSIM4v5toxe * pParam->BSIM4v5phi
	              / (pParam->BSIM4v5weff + pParam->BSIM4v5w0);

	  T3 = here->BSIM4v5eta0 + pParam->BSIM4v5etab * Vbseff;
	  if (T3 < 1.0e-4)
	  {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
	      T3 = (2.0e-4 - T3) * T9;
	      T4 = T9 * T9;
	  }
	  else
	  {   T4 = 1.0;
	  }
	  dDIBL_Sft_dVd = T3 * pParam->BSIM4v5theta0vb0;
          DIBL_Sft = dDIBL_Sft_dVd * Vds;

 	  Lpe_Vb = sqrt(1.0 + pParam->BSIM4v5lpeb / Leff);

          Vth = model->BSIM4v5type * here->BSIM4v5vth0 + (pParam->BSIM4v5k1ox * sqrtPhis
	      - pParam->BSIM4v5k1 * pParam->BSIM4v5sqrtPhi) * Lpe_Vb
              - here->BSIM4v5k2ox * Vbseff - Delt_vth - T2 + (pParam->BSIM4v5k3
              + pParam->BSIM4v5k3b * Vbseff) * Vth_NarrowW + T1 - DIBL_Sft;

          dVth_dVb = Lpe_Vb * pParam->BSIM4v5k1ox * dsqrtPhis_dVb - here->BSIM4v5k2ox
                   - dDelt_vth_dVb - dT2_dVb + pParam->BSIM4v5k3b * Vth_NarrowW
                   - pParam->BSIM4v5etab * Vds * pParam->BSIM4v5theta0vb0 * T4
                   + pParam->BSIM4v5kt2 * TempRatio;
          dVth_dVd = -dDIBL_Sft_dVd;


          /* Calculate n */
          tmp1 = EPSSI / Xdep;
	  here->BSIM4v5nstar = model->BSIM4v5vtm / Charge_q * (model->BSIM4v5coxe
			   + tmp1 + pParam->BSIM4v5cit);  
          tmp2 = pParam->BSIM4v5nfactor * tmp1;
          tmp3 = pParam->BSIM4v5cdsc + pParam->BSIM4v5cdscb * Vbseff
               + pParam->BSIM4v5cdscd * Vds;
	  tmp4 = (tmp2 + tmp3 * Theta0 + pParam->BSIM4v5cit) / model->BSIM4v5coxe;
	  if (tmp4 >= -0.5)
	  {   n = 1.0 + tmp4;
	      dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM4v5cdscb * Theta0) / model->BSIM4v5coxe;
              dn_dVd = pParam->BSIM4v5cdscd * Theta0 / model->BSIM4v5coxe;
	  }
	  else
	  {   T0 = 1.0 / (3.0 + 8.0 * tmp4);
	      n = (1.0 + 3.0 * tmp4) * T0;
	      T0 *= T0;
	      dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM4v5cdscb * Theta0) / model->BSIM4v5coxe * T0;
              dn_dVd = pParam->BSIM4v5cdscd * Theta0 / model->BSIM4v5coxe * T0;
	  }


          /* Vth correction for Pocket implant */
 	  if (pParam->BSIM4v5dvtp0 > 0.0)
          {   T0 = -pParam->BSIM4v5dvtp1 * Vds;
              if (T0 < -EXP_THRESHOLD)
              {   T2 = MIN_EXP;
                  dT2_dVd = 0.0;
              }
              else
              {   T2 = exp(T0);
                  dT2_dVd = -pParam->BSIM4v5dvtp1 * T2;
              }

              T3 = Leff + pParam->BSIM4v5dvtp0 * (1.0 + T2);
              dT3_dVd = pParam->BSIM4v5dvtp0 * dT2_dVd;
              if (model->BSIM4v5tempMod < 2)
              {
                T4 = Vtm * log(Leff / T3);
                dT4_dVd = -Vtm * dT3_dVd / T3;
              }
              else
              {
                T4 = model->BSIM4v5vtm0 * log(Leff / T3);
                dT4_dVd = -model->BSIM4v5vtm0 * dT3_dVd / T3;
              }
              dDITS_Sft_dVd = dn_dVd * T4 + n * dT4_dVd;
              dDITS_Sft_dVb = T4 * dn_dVb;

              Vth -= n * T4;
              dVth_dVd -= dDITS_Sft_dVd;
              dVth_dVb -= dDITS_Sft_dVb;
	  }
          here->BSIM4v5von = Vth;


          /* Poly Gate Si Depletion Effect */
	  T0 = here->BSIM4v5vfb + pParam->BSIM4v5phi;

    	  BSIM4v5polyDepletion(T0, pParam->BSIM4v5ngate, model->BSIM4v5coxe, vgs, &vgs_eff, &dvgs_eff_dvg);

    	  BSIM4v5polyDepletion(T0, pParam->BSIM4v5ngate, model->BSIM4v5coxe, vgd, &vgd_eff, &dvgd_eff_dvg);
    	  
    	  if(here->BSIM4v5mode>0) {
    	  	Vgs_eff = vgs_eff;
    	  	dVgs_eff_dVg = dvgs_eff_dvg;
    	  } else {
    	  	Vgs_eff = vgd_eff;
    	  	dVgs_eff_dVg = dvgd_eff_dvg;
    	  }
    	  here->BSIM4v5vgs_eff = vgs_eff;
    	  here->BSIM4v5vgd_eff = vgd_eff;
    	  here->BSIM4v5dvgs_eff_dvg = dvgs_eff_dvg;
    	  here->BSIM4v5dvgd_eff_dvg = dvgd_eff_dvg;


          Vgst = Vgs_eff - Vth;

	  /* Calculate Vgsteff */
	  T0 = n * Vtm;
	  T1 = pParam->BSIM4v5mstar * Vgst;
	  T2 = T1 / T0;
	  if (T2 > EXP_THRESHOLD)
	  {   T10 = T1;
	      dT10_dVg = pParam->BSIM4v5mstar * dVgs_eff_dVg;
              dT10_dVd = -dVth_dVd * pParam->BSIM4v5mstar;
              dT10_dVb = -dVth_dVb * pParam->BSIM4v5mstar;
	  }
	  else if (T2 < -EXP_THRESHOLD)
	  {   T10 = Vtm * log(1.0 + MIN_EXP);
              dT10_dVg = 0.0;
              dT10_dVd = T10 * dn_dVd;
              dT10_dVb = T10 * dn_dVb;
	      T10 *= n;
	  }
	  else
	  {   ExpVgst = exp(T2);
	      T3 = Vtm * log(1.0 + ExpVgst);
              T10 = n * T3;
              dT10_dVg = pParam->BSIM4v5mstar * ExpVgst / (1.0 + ExpVgst);
              dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n);
              dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n);
	      dT10_dVg *= dVgs_eff_dVg;
	  }

	  T1 = pParam->BSIM4v5voffcbn - (1.0 - pParam->BSIM4v5mstar) * Vgst;
	  T2 = T1 / T0;
          if (T2 < -EXP_THRESHOLD)
          {   T3 = model->BSIM4v5coxe * MIN_EXP / pParam->BSIM4v5cdep0;
	      T9 = pParam->BSIM4v5mstar + T3 * n;
              dT9_dVg = 0.0;
              dT9_dVd = dn_dVd * T3;
              dT9_dVb = dn_dVb * T3;
          }
          else if (T2 > EXP_THRESHOLD)
          {   T3 = model->BSIM4v5coxe * MAX_EXP / pParam->BSIM4v5cdep0;
              T9 = pParam->BSIM4v5mstar + T3 * n;
              dT9_dVg = 0.0;
              dT9_dVd = dn_dVd * T3;
              dT9_dVb = dn_dVb * T3;
          }
          else
          {   ExpVgst = exp(T2);
	      T3 = model->BSIM4v5coxe / pParam->BSIM4v5cdep0;
	      T4 = T3 * ExpVgst;
	      T5 = T1 * T4 / T0;
              T9 = pParam->BSIM4v5mstar + n * T4;
              dT9_dVg = T3 * (pParam->BSIM4v5mstar - 1.0) * ExpVgst / Vtm;
              dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb;
              dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd;
              dT9_dVg *= dVgs_eff_dVg;
          }

          here->BSIM4v5Vgsteff = Vgsteff = T10 / T9;
	  T11 = T9 * T9;
          dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11;
          dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11;
          dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11;

          /* Calculate Effective Channel Geometry */
          T9 = sqrtPhis - pParam->BSIM4v5sqrtPhi;
          Weff = pParam->BSIM4v5weff - 2.0 * (pParam->BSIM4v5dwg * Vgsteff 
               + pParam->BSIM4v5dwb * T9); 
          dWeff_dVg = -2.0 * pParam->BSIM4v5dwg;
          dWeff_dVb = -2.0 * pParam->BSIM4v5dwb * dsqrtPhis_dVb;

          if (Weff < 2.0e-8) /* to avoid the discontinuity problem due to Weff*/
	  {   T0 = 1.0 / (6.0e-8 - 2.0 * Weff);
	      Weff = 2.0e-8 * (4.0e-8 - Weff) * T0;
	      T0 *= T0 * 4.0e-16;
              dWeff_dVg *= T0;
	      dWeff_dVb *= T0;
          }

	  if (model->BSIM4v5rdsMod == 1)
	      Rds = dRds_dVg = dRds_dVb = 0.0;
          else
          {   T0 = 1.0 + pParam->BSIM4v5prwg * Vgsteff;
	      dT0_dVg = -pParam->BSIM4v5prwg / T0 / T0;
	      T1 = pParam->BSIM4v5prwb * T9;
	      dT1_dVb = pParam->BSIM4v5prwb * dsqrtPhis_dVb;

	      T2 = 1.0 / T0 + T1;
	      T3 = T2 + sqrt(T2 * T2 + 0.01); /* 0.01 = 4.0 * 0.05 * 0.05 */
	      dT3_dVg = 1.0 + T2 / (T3 - T2);
	      dT3_dVb = dT3_dVg * dT1_dVb;
	      dT3_dVg *= dT0_dVg;

	      T4 = pParam->BSIM4v5rds0 * 0.5;
	      Rds = pParam->BSIM4v5rdswmin + T3 * T4;
              dRds_dVg = T4 * dT3_dVg;
              dRds_dVb = T4 * dT3_dVb;

	      if (Rds > 0.0)
		  here->BSIM4v5grdsw = 1.0 / Rds; 
	      else
                  here->BSIM4v5grdsw = 0.0;
          }
	  
          /* Calculate Abulk */
	  T9 = 0.5 * pParam->BSIM4v5k1ox * Lpe_Vb / sqrtPhis;
          T1 = T9 + here->BSIM4v5k2ox - pParam->BSIM4v5k3b * Vth_NarrowW;
          dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;

          T9 = sqrt(pParam->BSIM4v5xj * Xdep);
          tmp1 = Leff + 2.0 * T9;
          T5 = Leff / tmp1; 
          tmp2 = pParam->BSIM4v5a0 * T5;
          tmp3 = pParam->BSIM4v5weff + pParam->BSIM4v5b1; 
          tmp4 = pParam->BSIM4v5b0 / tmp3;
          T2 = tmp2 + tmp4;
          dT2_dVb = -T9 / tmp1 / Xdep * dXdep_dVb;
          T6 = T5 * T5;
          T7 = T5 * T6;

          Abulk0 = 1.0 + T1 * T2; 
          dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;

          T8 = pParam->BSIM4v5ags * pParam->BSIM4v5a0 * T7;
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
	  {   T9 = 1.0 / (3.0 - 20.0 * Abulk);
	      Abulk = (0.2 - Abulk) * T9;
              T10 = T9 * T9;
	      dAbulk_dVb *= T10;
              dAbulk_dVg *= T10;
	  }
	  here->BSIM4v5Abulk = Abulk;

          T2 = pParam->BSIM4v5keta * Vbseff;
	  if (T2 >= -0.9)
	  {   T0 = 1.0 / (1.0 + T2);
              dT0_dVb = -pParam->BSIM4v5keta * T0 * T0;
	  }
	  else
	  {   T1 = 1.0 / (0.8 + T2);
	      T0 = (17.0 + 20.0 * T2) * T1;
              dT0_dVb = -pParam->BSIM4v5keta * T1 * T1;
	  }
	  dAbulk_dVg *= T0;
	  dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
	  dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
	  Abulk *= T0;
	  Abulk0 *= T0;

          /* Mobility calculation */
          if (model->BSIM4v5mobMod == 0)
          {   T0 = Vgsteff + Vth + Vth;
              T2 = pParam->BSIM4v5ua + pParam->BSIM4v5uc * Vbseff;
              T3 = T0 / model->BSIM4v5toxe;
              T6 = pParam->BSIM4v5ud / T3 / T3 * Vth * Vth;
              T5 = T3 * (T2 + pParam->BSIM4v5ub * T3) + T6;
              T7 = - 2.0 * T6 / T0;
              dDenomi_dVg = (T2 + 2.0 * pParam->BSIM4v5ub * T3) / model->BSIM4v5toxe + T7;
              dDenomi_dVd = dDenomi_dVg * 2.0 * dVth_dVd;
              dDenomi_dVb = dDenomi_dVg * 2.0 * dVth_dVb + pParam->BSIM4v5uc * T3;
          }
          else if (model->BSIM4v5mobMod == 1)
          {   T0 = Vgsteff + Vth + Vth;
              T2 = 1.0 + pParam->BSIM4v5uc * Vbseff;
              T3 = T0 / model->BSIM4v5toxe;
              T4 = T3 * (pParam->BSIM4v5ua + pParam->BSIM4v5ub * T3);
              T6 = pParam->BSIM4v5ud / T3 / T3 * Vth * Vth;
              T5 = T4 * T2 + T6;
              T7 = - 2.0 * T6 / T0;
              dDenomi_dVg = (pParam->BSIM4v5ua + 2.0 * pParam->BSIM4v5ub * T3) * T2
                          / model->BSIM4v5toxe + T7;
              dDenomi_dVd = dDenomi_dVg * 2.0 * dVth_dVd;
              dDenomi_dVb = dDenomi_dVg * 2.0 * dVth_dVb + pParam->BSIM4v5uc * T4;
          }
	  else
	  {   T0 = (Vgsteff + here->BSIM4v5vtfbphi1) / model->BSIM4v5toxe;
	      T1 = exp(pParam->BSIM4v5eu * log(T0));
	      dT1_dVg = T1 * pParam->BSIM4v5eu / T0 / model->BSIM4v5toxe;
	      T2 = pParam->BSIM4v5ua + pParam->BSIM4v5uc * Vbseff;
              T3 = T0 / model->BSIM4v5toxe;
              T6 = pParam->BSIM4v5ud / T3 / T3 * Vth * Vth;
	      T5 = T1 * T2 + T6;
              T7 = - 2.0 * T6 / T0;
 	      dDenomi_dVg = T2 * dT1_dVg + T7;
              dDenomi_dVd = 0.0;
              dDenomi_dVb = T1 * pParam->BSIM4v5uc;
	  }

	  if (T5 >= -0.8)
	  {   Denomi = 1.0 + T5;
	  }
	  else
	  {   T9 = 1.0 / (7.0 + 10.0 * T5);
	      Denomi = (0.6 + T5) * T9;
	      T9 *= T9;
              dDenomi_dVg *= T9;
              dDenomi_dVd *= T9;
              dDenomi_dVb *= T9;
	  }

          here->BSIM4v5ueff = ueff = here->BSIM4v5u0temp / Denomi;
	  T9 = -ueff / Denomi;
          dueff_dVg = T9 * dDenomi_dVg;
          dueff_dVd = T9 * dDenomi_dVd;
          dueff_dVb = T9 * dDenomi_dVb;

          /* Saturation Drain Voltage  Vdsat */
          WVCox = Weff * here->BSIM4v5vsattemp * model->BSIM4v5coxe;
          WVCoxRds = WVCox * Rds; 

          Esat = 2.0 * here->BSIM4v5vsattemp / ueff;
          here->BSIM4v5EsatL = EsatL = Esat * Leff;
          T0 = -EsatL /ueff;
          dEsatL_dVg = T0 * dueff_dVg;
          dEsatL_dVd = T0 * dueff_dVd;
          dEsatL_dVb = T0 * dueff_dVb;
  
	  /* Sqrt() */
          a1 = pParam->BSIM4v5a1;
	  if (a1 == 0.0)
	  {   Lambda = pParam->BSIM4v5a2;
	      dLambda_dVg = 0.0;
	  }
	  else if (a1 > 0.0)
	  {   T0 = 1.0 - pParam->BSIM4v5a2;
	      T1 = T0 - pParam->BSIM4v5a1 * Vgsteff - 0.0001;
	      T2 = sqrt(T1 * T1 + 0.0004 * T0);
	      Lambda = pParam->BSIM4v5a2 + T0 - 0.5 * (T1 + T2);
	      dLambda_dVg = 0.5 * pParam->BSIM4v5a1 * (1.0 + T1 / T2);
	  }
	  else
	  {   T1 = pParam->BSIM4v5a2 + pParam->BSIM4v5a1 * Vgsteff - 0.0001;
	      T2 = sqrt(T1 * T1 + 0.0004 * pParam->BSIM4v5a2);
	      Lambda = 0.5 * (T1 + T2);
	      dLambda_dVg = 0.5 * pParam->BSIM4v5a1 * (1.0 + T1 / T2);
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
          here->BSIM4v5vdsat = Vdsat;

          /* Calculate Vdseff */
          T1 = Vdsat - Vds - pParam->BSIM4v5delta;
          dT1_dVg = dVdsat_dVg;
          dT1_dVd = dVdsat_dVd - 1.0;
          dT1_dVb = dVdsat_dVb;

          T2 = sqrt(T1 * T1 + 4.0 * pParam->BSIM4v5delta * Vdsat);
	  T0 = T1 / T2;
   	  T9 = 2.0 * pParam->BSIM4v5delta;
	  T3 = T9 / T2;
          dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
          dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
          dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

	  if (T1 >= 0.0)
	  {   Vdseff = Vdsat - 0.5 * (T1 + T2);
	      dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg);
              dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd);
              dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb);
	  }
	  else
	  {   T4 = T9 / (T2 - T1);
	      T5 = 1.0 - T4;
	      T6 = Vdsat * T4 / (T2 - T1);
	      Vdseff = Vdsat * T5;
              dVdseff_dVg = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg);
              dVdseff_dVd = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd);
              dVdseff_dVb = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb);
	  }

          if (Vds == 0.0)
          {  Vdseff = 0.0;
             dVdseff_dVg = 0.0;
             dVdseff_dVb = 0.0; 
          }

          if (Vdseff > Vds)
              Vdseff = Vds;
          diffVds = Vds - Vdseff;
          here->BSIM4v5Vdseff = Vdseff;
          
          /* Velocity Overshoot */
        if((model->BSIM4v5lambdaGiven) && (model->BSIM4v5lambda > 0.0) )
        {  
          T1 =  Leff * ueff;
          T2 = pParam->BSIM4v5lambda / T1;
          T3 = -T2 / T1 * Leff;
          dT2_dVd = T3 * dueff_dVd;
          dT2_dVg = T3 * dueff_dVg;
          dT2_dVb = T3 * dueff_dVb;
          T5 = 1.0 / (Esat * pParam->BSIM4v5litl);
          T4 = -T5 / EsatL;
          dT5_dVg = dEsatL_dVg * T4;
          dT5_dVd = dEsatL_dVd * T4; 
          dT5_dVb = dEsatL_dVb * T4; 
          T6 = 1.0 + diffVds  * T5;
          dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
          dT6_dVd = dT5_dVd * diffVds + (1.0 - dVdseff_dVd) * T5;
          dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
          T7 = 2.0 / (T6 * T6 + 1.0);
          T8 = 1.0 - T7;
          T9 = T6 * T7 * T7;
          dT8_dVg = T9 * dT6_dVg;
          dT8_dVd = T9 * dT6_dVd;
          dT8_dVb = T9 * dT6_dVb;
          T10 = 1.0 + T2 * T8;
          dT10_dVg = dT2_dVg * T8 + T2 * dT8_dVg;
          dT10_dVd = dT2_dVd * T8 + T2 * dT8_dVd;
          dT10_dVb = dT2_dVb * T8 + T2 * dT8_dVb;
          if(T10 == 1.0)
                dT10_dVg = dT10_dVd = dT10_dVb = 0.0;

          dEsatL_dVg *= T10;
          dEsatL_dVg += EsatL * dT10_dVg;
          dEsatL_dVd *= T10;
          dEsatL_dVd += EsatL * dT10_dVd;
          dEsatL_dVb *= T10;
          dEsatL_dVb += EsatL * dT10_dVb;
          EsatL *= T10;
          here->BSIM4v5EsatL = EsatL;
        }

          /* Calculate Vasat */
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

	  /* Calculate Idl first */
          tmp1 = here->BSIM4v5vtfbphi2;
          tmp2 = 2.0e8 * model->BSIM4v5toxp;
          dT0_dVg = 1.0 / tmp2;
          T0 = (Vgsteff + tmp1) * dT0_dVg;

          tmp3 = exp(0.7 * log(T0));
          T1 = 1.0 + tmp3;
          T2 = 0.7 * tmp3 / T0;
          Tcen = 1.9e-9 / T1;
          dTcen_dVg = -Tcen * T2 * dT0_dVg / T1;

          Coxeff = EPSSI * model->BSIM4v5coxp
                 / (EPSSI + model->BSIM4v5coxp * Tcen);
          dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg / EPSSI;

          CoxeffWovL = Coxeff * Weff / Leff;
          beta = ueff * CoxeffWovL;
          T3 = ueff / Leff;
          dbeta_dVg = CoxeffWovL * dueff_dVg + T3
                    * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
          dbeta_dVd = CoxeffWovL * dueff_dVd;
          dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;

          here->BSIM4v5AbovVgst2Vtm = Abulk / Vgst2Vtm;
          T0 = 1.0 - 0.5 * Vdseff * here->BSIM4v5AbovVgst2Vtm;
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
          Idl = gche / T0;
          T1 = (1.0 - Idl * Rds) / T0;
          T2 = Idl * Idl;
          dIdl_dVg = T1 * dgche_dVg - T2 * dRds_dVg;
          dIdl_dVd = T1 * dgche_dVd;
          dIdl_dVb = T1 * dgche_dVb - T2 * dRds_dVb;

          /* Calculate degradation factor due to pocket implant */

	  if (pParam->BSIM4v5fprout <= 0.0)
	  {   FP = 1.0;
	      dFP_dVg = 0.0;
	  }
	  else
	  {   T9 = pParam->BSIM4v5fprout * sqrt(Leff) / Vgst2Vtm;
              FP = 1.0 / (1.0 + T9);
              dFP_dVg = FP * FP * T9 / Vgst2Vtm;
	  }

          /* Calculate VACLM */
          T8 = pParam->BSIM4v5pvag / EsatL;
          T9 = T8 * Vgsteff;
          if (T9 > -0.9)
          {   PvagTerm = 1.0 + T9;
              dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL);
              dPvagTerm_dVb = -T9 * dEsatL_dVb / EsatL;
              dPvagTerm_dVd = -T9 * dEsatL_dVd / EsatL;
          }
          else
          {   T4 = 1.0 / (17.0 + 20.0 * T9);
              PvagTerm = (0.8 + T9) * T4;
              T4 *= T4;
              dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T4;
              T9 *= T4 / EsatL;
              dPvagTerm_dVb = -T9 * dEsatL_dVb;
              dPvagTerm_dVd = -T9 * dEsatL_dVd;
          }

          if ((pParam->BSIM4v5pclm > MIN_EXP) && (diffVds > 1.0e-10))
	  {   T0 = 1.0 + Rds * Idl;
              dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
              dT0_dVd = Rds * dIdl_dVd;
              dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;

              T2 = Vdsat / Esat;
	      T1 = Leff + T2;
              dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg / Leff) / Esat;
              dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd / Leff) / Esat;
              dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb / Leff) / Esat;

              Cclm = FP * PvagTerm * T0 * T1 / (pParam->BSIM4v5pclm * pParam->BSIM4v5litl);
              dCclm_dVg = Cclm * (dFP_dVg / FP + dPvagTerm_dVg / PvagTerm
                        + dT0_dVg / T0 + dT1_dVg / T1);
              dCclm_dVb = Cclm * (dPvagTerm_dVb / PvagTerm + dT0_dVb / T0
                        + dT1_dVb / T1);
              dCclm_dVd = Cclm * (dPvagTerm_dVd / PvagTerm + dT0_dVd / T0
                        + dT1_dVd / T1);
              VACLM = Cclm * diffVds;

              dVACLM_dVg = dCclm_dVg * diffVds - dVdseff_dVg * Cclm;
              dVACLM_dVb = dCclm_dVb * diffVds - dVdseff_dVb * Cclm;
              dVACLM_dVd = dCclm_dVd * diffVds + (1.0 - dVdseff_dVd) * Cclm;
          }
          else
          {   VACLM = Cclm = MAX_EXP;
              dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = 0.0;
              dCclm_dVd = dCclm_dVg = dCclm_dVb = 0.0;
          }

          /* Calculate VADIBL */
          if (pParam->BSIM4v5thetaRout > MIN_EXP)
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
	      T2 = pParam->BSIM4v5thetaRout;
              VADIBL = (Vgst2Vtm - T0 / T1) / T2;
              dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2;
              dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2;
              dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2;

	      T7 = pParam->BSIM4v5pdiblb * Vbseff;
	      if (T7 >= -0.9)
	      {   T3 = 1.0 / (1.0 + T7);
                  VADIBL *= T3;
                  dVADIBL_dVg *= T3;
                  dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam->BSIM4v5pdiblb)
			      * T3;
                  dVADIBL_dVd *= T3;
	      }
	      else
	      {   T4 = 1.0 / (0.8 + T7);
		  T3 = (17.0 + 20.0 * T7) * T4;
                  dVADIBL_dVg *= T3;
                  dVADIBL_dVb = dVADIBL_dVb * T3
			      - VADIBL * pParam->BSIM4v5pdiblb * T4 * T4;
                  dVADIBL_dVd *= T3;
                  VADIBL *= T3;
	      }

              dVADIBL_dVg = dVADIBL_dVg * PvagTerm + VADIBL * dPvagTerm_dVg;
              dVADIBL_dVb = dVADIBL_dVb * PvagTerm + VADIBL * dPvagTerm_dVb;
              dVADIBL_dVd = dVADIBL_dVd * PvagTerm + VADIBL * dPvagTerm_dVd;
              VADIBL *= PvagTerm;
          }
	  else
	  {   VADIBL = MAX_EXP;
              dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = 0.0;
          }

          /* Calculate Va */
          Va = Vasat + VACLM;
          dVa_dVg = dVasat_dVg + dVACLM_dVg;
          dVa_dVb = dVasat_dVb + dVACLM_dVb;
          dVa_dVd = dVasat_dVd + dVACLM_dVd;

          /* Calculate VADITS */
          T0 = pParam->BSIM4v5pditsd * Vds;
          if (T0 > EXP_THRESHOLD)
          {   T1 = MAX_EXP;
              dT1_dVd = 0;
          }
	  else
          {   T1 = exp(T0);
              dT1_dVd = T1 * pParam->BSIM4v5pditsd;
          }

          if (pParam->BSIM4v5pdits > MIN_EXP)
          {   T2 = 1.0 + model->BSIM4v5pditsl * Leff;
              VADITS = (1.0 + T2 * T1) / pParam->BSIM4v5pdits;
              dVADITS_dVg = VADITS * dFP_dVg;
              dVADITS_dVd = FP * T2 * dT1_dVd / pParam->BSIM4v5pdits;
	      VADITS *= FP;
          }
	  else
          {   VADITS = MAX_EXP;
              dVADITS_dVg = dVADITS_dVd = 0;
          }

          /* Calculate VASCBE */
	  if (pParam->BSIM4v5pscbe2 > 0.0)
	  {   if (diffVds > pParam->BSIM4v5pscbe1 * pParam->BSIM4v5litl
		  / EXP_THRESHOLD)
	      {   T0 =  pParam->BSIM4v5pscbe1 * pParam->BSIM4v5litl / diffVds;
	          VASCBE = Leff * exp(T0) / pParam->BSIM4v5pscbe2;
                  T1 = T0 * VASCBE / diffVds;
                  dVASCBE_dVg = T1 * dVdseff_dVg;
                  dVASCBE_dVd = -T1 * (1.0 - dVdseff_dVd);
                  dVASCBE_dVb = T1 * dVdseff_dVb;
              }
	      else
	      {   VASCBE = MAX_EXP * Leff/pParam->BSIM4v5pscbe2;
                  dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0;
              }
	  }
	  else
	  {   VASCBE = MAX_EXP;
              dVASCBE_dVg = dVASCBE_dVd = dVASCBE_dVb = 0.0;
	  }

	  /* Add DIBL to Ids */
          T9 = diffVds / VADIBL;
          T0 = 1.0 + T9;
          Idsa = Idl * T0;
          dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVADIBL_dVg) / VADIBL;
          dIdsa_dVd = T0 * dIdl_dVd + Idl
                    * (1.0 - dVdseff_dVd - T9 * dVADIBL_dVd) / VADIBL;
          dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVADIBL_dVb) / VADIBL;

          /* Add DITS to Ids */
          T9 = diffVds / VADITS;
          T0 = 1.0 + T9;
          dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) / VADITS;
          dIdsa_dVd = T0 * dIdsa_dVd + Idsa 
		    * (1.0 - dVdseff_dVd - T9 * dVADITS_dVd) / VADITS;
          dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb / VADITS;
          Idsa *= T0;

          /* Add CLM to Ids */
          T0 = log(Va / Vasat);
          dT0_dVg = dVa_dVg / Va - dVasat_dVg / Vasat;
          dT0_dVb = dVa_dVb / Va - dVasat_dVb / Vasat;
          dT0_dVd = dVa_dVd / Va - dVasat_dVd / Vasat;
          T1 = T0 / Cclm;
          T9 = 1.0 + T1;
          dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) / Cclm;
          dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) / Cclm;
          dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) / Cclm;

          dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
          dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
          dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
          Idsa *= T9;

          /* Substrate current begins */
          tmp = pParam->BSIM4v5alpha0 + pParam->BSIM4v5alpha1 * Leff;
          if ((tmp <= 0.0) || (pParam->BSIM4v5beta0 <= 0.0))
          {   Isub = Gbd = Gbb = Gbg = 0.0;
          }
          else
          {   T2 = tmp / Leff;
              if (diffVds > pParam->BSIM4v5beta0 / EXP_THRESHOLD)
              {   T0 = -pParam->BSIM4v5beta0 / diffVds;
                  T1 = T2 * diffVds * exp(T0);
                  T3 = T1 / diffVds * (T0 - 1.0);
                  dT1_dVg = T3 * dVdseff_dVg;
                  dT1_dVd = T3 * (dVdseff_dVd - 1.0);
                  dT1_dVb = T3 * dVdseff_dVb;
              }
              else
              {   T3 = T2 * MIN_EXP;
                  T1 = T3 * diffVds;
                  dT1_dVg = -T3 * dVdseff_dVg;
                  dT1_dVd = T3 * (1.0 - dVdseff_dVd);
                  dT1_dVb = -T3 * dVdseff_dVb;
              }
              T4 = Idsa * Vdseff;
              Isub = T1 * T4;
              Gbg = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg)
                  + T4 * dT1_dVg;
              Gbd = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd)
                  + T4 * dT1_dVd;
              Gbb = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb)
                  + T4 * dT1_dVb;

              Gbd += Gbg * dVgsteff_dVd;
              Gbb += Gbg * dVgsteff_dVb;
              Gbg *= dVgsteff_dVg;
              Gbb *= dVbseff_dVb;
          }
          here->BSIM4v5csub = Isub;
          here->BSIM4v5gbbs = Gbb;
          here->BSIM4v5gbgs = Gbg;
          here->BSIM4v5gbds = Gbd;

          /* Add SCBE to Ids */
          T9 = diffVds / VASCBE;
          T0 = 1.0 + T9;
          Ids = Idsa * T0;

          Gm = T0 * dIdsa_dVg - Idsa 
	     * (dVdseff_dVg + T9 * dVASCBE_dVg) / VASCBE;
          Gds = T0 * dIdsa_dVd + Idsa 
	      * (1.0 - dVdseff_dVd - T9 * dVASCBE_dVd) / VASCBE;
          Gmb = T0 * dIdsa_dVb - Idsa
	      * (dVdseff_dVb + T9 * dVASCBE_dVb) / VASCBE;


	  tmp1 = Gds + Gm * dVgsteff_dVd;
	  tmp2 = Gmb + Gm * dVgsteff_dVb;
	  tmp3 = Gm;

          Gm = (Ids * dVdseff_dVg + Vdseff * tmp3) * dVgsteff_dVg;
          Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd)
	      + Vdseff * tmp1;
          Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb)
	      + Vdseff * tmp2) * dVbseff_dVb;

          cdrain = Ids * Vdseff;
          
          /* Source End Velocity Limit  */
        if((model->BSIM4v5vtlGiven) && (model->BSIM4v5vtl > 0.0) ) {
          T12 = 1.0 / Leff / CoxeffWovL;
          T11 = T12 / Vgsteff;
          T10 = -T11 / Vgsteff;
          vs = cdrain * T11; /* vs */
          dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
          dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
          dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
          T0 = 2 * MM;
          T1 = vs / (pParam->BSIM4v5vtl * pParam->BSIM4v5tfactor);
          if(T1 > 0.0)  
          {	T2 = 1.0 + exp(T0 * log(T1));
          	T3 = (T2 - 1.0) * T0 / vs; 
          	Fsevl = 1.0 / exp(log(T2)/ T0);
          	dT2_dVg = T3 * dvs_dVg;
          	dT2_dVd = T3 * dvs_dVd;
          	dT2_dVb = T3 * dvs_dVb;
          	T4 = -1.0 / T0 * Fsevl / T2;
          	dFsevl_dVg = T4 * dT2_dVg;
          	dFsevl_dVd = T4 * dT2_dVd;
          	dFsevl_dVb = T4 * dT2_dVb;
          } else {
          	Fsevl = 1.0;
          	dFsevl_dVg = 0.0;
          	dFsevl_dVd = 0.0;
          	dFsevl_dVb = 0.0;
          }
          Gm *=Fsevl;
          Gm += cdrain * dFsevl_dVg;
          Gmb *=Fsevl;
          Gmb += cdrain * dFsevl_dVb;
          Gds *=Fsevl;
          Gds += cdrain * dFsevl_dVd;

          cdrain *= Fsevl; 
        } 

          here->BSIM4v5gds = Gds;
          here->BSIM4v5gm = Gm;
          here->BSIM4v5gmbs = Gmb;
          here->BSIM4v5IdovVds = Ids;
          if( here->BSIM4v5IdovVds <= 1.0e-9) here->BSIM4v5IdovVds = 1.0e-9;

          /* Calculate Rg */
          if ((here->BSIM4v5rgateMod > 1) ||
              (here->BSIM4v5trnqsMod != 0) || (here->BSIM4v5acnqsMod != 0))
	  {   T9 = pParam->BSIM4v5xrcrg2 * model->BSIM4v5vtm;
              T0 = T9 * beta;
              dT0_dVd = (dbeta_dVd + dbeta_dVg * dVgsteff_dVd) * T9;
              dT0_dVb = (dbeta_dVb + dbeta_dVg * dVgsteff_dVb) * T9;
              dT0_dVg = dbeta_dVg * T9;

	      here->BSIM4v5gcrg = pParam->BSIM4v5xrcrg1 * ( T0 + Ids);
	      here->BSIM4v5gcrgd = pParam->BSIM4v5xrcrg1 * (dT0_dVd + tmp1);
              here->BSIM4v5gcrgb = pParam->BSIM4v5xrcrg1 * (dT0_dVb + tmp2)
	 		       * dVbseff_dVb;	
              here->BSIM4v5gcrgg = pParam->BSIM4v5xrcrg1 * (dT0_dVg + tmp3)
			       * dVgsteff_dVg;

	      if (here->BSIM4v5nf != 1.0)
	      {   here->BSIM4v5gcrg *= here->BSIM4v5nf; 
		  here->BSIM4v5gcrgg *= here->BSIM4v5nf;
		  here->BSIM4v5gcrgd *= here->BSIM4v5nf;
		  here->BSIM4v5gcrgb *= here->BSIM4v5nf;
	      }

              if (here->BSIM4v5rgateMod == 2)
              {   T10 = here->BSIM4v5grgeltd * here->BSIM4v5grgeltd;
		  T11 = here->BSIM4v5grgeltd + here->BSIM4v5gcrg;
		  here->BSIM4v5gcrg = here->BSIM4v5grgeltd * here->BSIM4v5gcrg / T11;
                  T12 = T10 / T11 / T11;
                  here->BSIM4v5gcrgg *= T12;
                  here->BSIM4v5gcrgd *= T12;
                  here->BSIM4v5gcrgb *= T12;
              }
              here->BSIM4v5gcrgs = -(here->BSIM4v5gcrgg + here->BSIM4v5gcrgd
			       + here->BSIM4v5gcrgb);
	  }


	  /* Calculate bias-dependent external S/D resistance */
          if (model->BSIM4v5rdsMod)
	  {   /* Rs(V) */
              T0 = vgs - pParam->BSIM4v5vfbsd;
              T1 = sqrt(T0 * T0 + 1.0e-4);
              vgs_eff = 0.5 * (T0 + T1);
              dvgs_eff_dvg = vgs_eff / T1;

              T0 = 1.0 + pParam->BSIM4v5prwg * vgs_eff;
              dT0_dvg = -pParam->BSIM4v5prwg / T0 / T0 * dvgs_eff_dvg;
              T1 = -pParam->BSIM4v5prwb * vbs;
              dT1_dvb = -pParam->BSIM4v5prwb;

              T2 = 1.0 / T0 + T1;
              T3 = T2 + sqrt(T2 * T2 + 0.01);
              dT3_dvg = T3 / (T3 - T2);
              dT3_dvb = dT3_dvg * dT1_dvb;
              dT3_dvg *= dT0_dvg;

              T4 = pParam->BSIM4v5rs0 * 0.5;
              Rs = pParam->BSIM4v5rswmin + T3 * T4;
              dRs_dvg = T4 * dT3_dvg;
              dRs_dvb = T4 * dT3_dvb;

              T0 = 1.0 + here->BSIM4v5sourceConductance * Rs;
              here->BSIM4v5gstot = here->BSIM4v5sourceConductance / T0;
              T0 = -here->BSIM4v5gstot * here->BSIM4v5gstot;
              dgstot_dvd = 0.0; /* place holder */
              dgstot_dvg = T0 * dRs_dvg;
              dgstot_dvb = T0 * dRs_dvb;
              dgstot_dvs = -(dgstot_dvg + dgstot_dvb + dgstot_dvd);

	      /* Rd(V) */
              T0 = vgd - pParam->BSIM4v5vfbsd;
              T1 = sqrt(T0 * T0 + 1.0e-4);
              vgd_eff = 0.5 * (T0 + T1);
              dvgd_eff_dvg = vgd_eff / T1;

              T0 = 1.0 + pParam->BSIM4v5prwg * vgd_eff;
              dT0_dvg = -pParam->BSIM4v5prwg / T0 / T0 * dvgd_eff_dvg;
              T1 = -pParam->BSIM4v5prwb * vbd;
              dT1_dvb = -pParam->BSIM4v5prwb;

              T2 = 1.0 / T0 + T1;
              T3 = T2 + sqrt(T2 * T2 + 0.01);
              dT3_dvg = T3 / (T3 - T2);
              dT3_dvb = dT3_dvg * dT1_dvb;
              dT3_dvg *= dT0_dvg;

              T4 = pParam->BSIM4v5rd0 * 0.5;
              Rd = pParam->BSIM4v5rdwmin + T3 * T4;
              dRd_dvg = T4 * dT3_dvg;
              dRd_dvb = T4 * dT3_dvb;

              T0 = 1.0 + here->BSIM4v5drainConductance * Rd;
              here->BSIM4v5gdtot = here->BSIM4v5drainConductance / T0;
              T0 = -here->BSIM4v5gdtot * here->BSIM4v5gdtot;
              dgdtot_dvs = 0.0;
              dgdtot_dvg = T0 * dRd_dvg;
              dgdtot_dvb = T0 * dRd_dvb;
              dgdtot_dvd = -(dgdtot_dvg + dgdtot_dvb + dgdtot_dvs);

              here->BSIM4v5gstotd = vses * dgstot_dvd;
              here->BSIM4v5gstotg = vses * dgstot_dvg;
              here->BSIM4v5gstots = vses * dgstot_dvs;
              here->BSIM4v5gstotb = vses * dgstot_dvb;

              T2 = vdes - vds;
              here->BSIM4v5gdtotd = T2 * dgdtot_dvd;
              here->BSIM4v5gdtotg = T2 * dgdtot_dvg;
              here->BSIM4v5gdtots = T2 * dgdtot_dvs;
              here->BSIM4v5gdtotb = T2 * dgdtot_dvb;
	  }
	  else /* WDLiu: for bypass */
	  {   here->BSIM4v5gstot = here->BSIM4v5gstotd = here->BSIM4v5gstotg = 0.0;
	      here->BSIM4v5gstots = here->BSIM4v5gstotb = 0.0;
	      here->BSIM4v5gdtot = here->BSIM4v5gdtotd = here->BSIM4v5gdtotg = 0.0;
              here->BSIM4v5gdtots = here->BSIM4v5gdtotb = 0.0;
	  }

          /* Calculate GIDL current */
          vgs_eff = here->BSIM4v5vgs_eff;
          dvgs_eff_dvg = here->BSIM4v5dvgs_eff_dvg;
          T0 = 3.0 * model->BSIM4v5toxe;

          T1 = (vds - vgs_eff - pParam->BSIM4v5egidl ) / T0;
          if ((pParam->BSIM4v5agidl <= 0.0) || (pParam->BSIM4v5bgidl <= 0.0)
              || (T1 <= 0.0) || (pParam->BSIM4v5cgidl <= 0.0) || (vbd > 0.0))
              Igidl = Ggidld = Ggidlg = Ggidlb = 0.0;
          else {
              dT1_dVd = 1.0 / T0;
              dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
              T2 = pParam->BSIM4v5bgidl / T1;
              if (T2 < 100.0)
              {   Igidl = pParam->BSIM4v5agidl * pParam->BSIM4v5weffCJ * T1 * exp(-T2);
                  T3 = Igidl * (1.0 + T2) / T1;
                  Ggidld = T3 * dT1_dVd;
                  Ggidlg = T3 * dT1_dVg;
              }
              else
              {   Igidl = pParam->BSIM4v5agidl * pParam->BSIM4v5weffCJ * 3.720075976e-44;
                  Ggidld = Igidl * dT1_dVd;
                  Ggidlg = Igidl * dT1_dVg;
                  Igidl *= T1;
              }
        
              T4 = vbd * vbd;
              T5 = -vbd * T4;
              T6 = pParam->BSIM4v5cgidl + T5;
              T7 = T5 / T6;
              T8 = 3.0 * pParam->BSIM4v5cgidl * T4 / T6 / T6;
              Ggidld = Ggidld * T7 + Igidl * T8;
              Ggidlg = Ggidlg * T7;
              Ggidlb = -Igidl * T8;
              Igidl *= T7;
          }
          here->BSIM4v5Igidl = Igidl;
          here->BSIM4v5ggidld = Ggidld;
          here->BSIM4v5ggidlg = Ggidlg;
          here->BSIM4v5ggidlb = Ggidlb;

        /* Calculate GISL current  */
          vgd_eff = here->BSIM4v5vgd_eff;
          dvgd_eff_dvg = here->BSIM4v5dvgd_eff_dvg;

          T1 = (-vds - vgd_eff - pParam->BSIM4v5egidl ) / T0;

          if ((pParam->BSIM4v5agidl <= 0.0) || (pParam->BSIM4v5bgidl <= 0.0)
              || (T1 <= 0.0) || (pParam->BSIM4v5cgidl <= 0.0) || (vbs > 0.0))
              Igisl = Ggisls = Ggislg = Ggislb = 0.0;
          else {
              dT1_dVd = 1.0 / T0;
              dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
              T2 = pParam->BSIM4v5bgidl / T1;
              if (T2 < 100.0) 
              {   Igisl = pParam->BSIM4v5agidl * pParam->BSIM4v5weffCJ * T1 * exp(-T2);
                  T3 = Igisl * (1.0 + T2) / T1;
                  Ggisls = T3 * dT1_dVd;
                  Ggislg = T3 * dT1_dVg;
              }
              else 
              {   Igisl = pParam->BSIM4v5agidl * pParam->BSIM4v5weffCJ * 3.720075976e-44;
                  Ggisls = Igisl * dT1_dVd;
                  Ggislg = Igisl * dT1_dVg;
                  Igisl *= T1;
              }
        
              T4 = vbs * vbs;
              T5 = -vbs * T4;
              T6 = pParam->BSIM4v5cgidl + T5;
              T7 = T5 / T6;
              T8 = 3.0 * pParam->BSIM4v5cgidl * T4 / T6 / T6;
              Ggisls = Ggisls * T7 + Igisl * T8;
              Ggislg = Ggislg * T7;
              Ggislb = -Igisl * T8;
              Igisl *= T7;
          }
          here->BSIM4v5Igisl = Igisl;
          here->BSIM4v5ggisls = Ggisls;
          here->BSIM4v5ggislg = Ggislg;
          here->BSIM4v5ggislb = Ggislb;


          /* Calculate gate tunneling current */
          if ((model->BSIM4v5igcMod != 0) || (model->BSIM4v5igbMod != 0))
          {   Vfb = here->BSIM4v5vfbzb;
              V3 = Vfb - Vgs_eff + Vbseff - DELTA_3;
              if (Vfb <= 0.0)
                  T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfb);
              else
                  T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfb);
              T1 = 0.5 * (1.0 + V3 / T0);
              Vfbeff = Vfb - 0.5 * (V3 + T0);
              dVfbeff_dVg = T1 * dVgs_eff_dVg;
              dVfbeff_dVb = -T1; /* WDLiu: -No surprise? No. -Good! */

              Voxacc = Vfb - Vfbeff;
              dVoxacc_dVg = -dVfbeff_dVg;
              dVoxacc_dVb = -dVfbeff_dVb;
              if (Voxacc < 0.0) /* WDLiu: Avoiding numerical instability. */
                  Voxacc = dVoxacc_dVg = dVoxacc_dVb = 0.0;

              T0 = 0.5 * pParam->BSIM4v5k1ox;
              T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff;
              if (pParam->BSIM4v5k1ox == 0.0)
                  Voxdepinv = dVoxdepinv_dVg = dVoxdepinv_dVd
			    = dVoxdepinv_dVb = 0.0;
              else if (T3 < 0.0)
              {   Voxdepinv = -T3;
                  dVoxdepinv_dVg = -dVgs_eff_dVg + dVfbeff_dVg
                                 + dVgsteff_dVg;
                  dVoxdepinv_dVd = dVgsteff_dVd;
                  dVoxdepinv_dVb = dVfbeff_dVb + 1.0 + dVgsteff_dVb;
	      }
              else
              {   T1 = sqrt(T0 * T0 + T3);
                  T2 = T0 / T1;
                  Voxdepinv = pParam->BSIM4v5k1ox * (T1 - T0);
                  dVoxdepinv_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg
				 - dVgsteff_dVg);
                  dVoxdepinv_dVd = -T2 * dVgsteff_dVd;
                  dVoxdepinv_dVb = -T2 * (dVfbeff_dVb + 1.0 + dVgsteff_dVb);
              }

              Voxdepinv += Vgsteff;
              dVoxdepinv_dVg += dVgsteff_dVg;
              dVoxdepinv_dVd += dVgsteff_dVd;
              dVoxdepinv_dVb += dVgsteff_dVb;
          }

          if(model->BSIM4v5tempMod < 2)
          	tmp = Vtm;
          else /* model->BSIM4v5tempMod = 2 */
          	tmp = Vtm0;
          if (model->BSIM4v5igcMod)
          {   T0 = tmp * pParam->BSIM4v5nigc;
	      if(model->BSIM4v5igcMod == 1) {
              	VxNVt = (Vgs_eff - model->BSIM4v5type * here->BSIM4v5vth0) / T0;
              	if (VxNVt > EXP_THRESHOLD)
              	{   Vaux = Vgs_eff - model->BSIM4v5type * here->BSIM4v5vth0;
              	    dVaux_dVg = dVgs_eff_dVg;
		    dVaux_dVd = 0.0;
                    dVaux_dVb = 0.0;
              	}
	      } else if (model->BSIM4v5igcMod == 2) {
                VxNVt = (Vgs_eff - here->BSIM4v5von) / T0;
                if (VxNVt > EXP_THRESHOLD)
                {   Vaux = Vgs_eff - here->BSIM4v5von;
                    dVaux_dVg = dVgs_eff_dVg;
                    dVaux_dVd = -dVth_dVd;
                    dVaux_dVb = -dVth_dVb;
                }
              } 
              if (VxNVt < -EXP_THRESHOLD)
              {   Vaux = T0 * log(1.0 + MIN_EXP);
                  dVaux_dVg = dVaux_dVd = dVaux_dVb = 0.0;
              }
              else if ((VxNVt >= -EXP_THRESHOLD) && (VxNVt <= EXP_THRESHOLD))
              {   ExpVxNVt = exp(VxNVt);
                  Vaux = T0 * log(1.0 + ExpVxNVt);
                  dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
		  if(model->BSIM4v5igcMod == 1) {
			dVaux_dVd = 0.0;
                  	dVaux_dVb = 0.0;
                  } else if (model->BSIM4v5igcMod == 2) {
                        dVaux_dVd = -dVgs_eff_dVg * dVth_dVd;
                        dVaux_dVb = -dVgs_eff_dVg * dVth_dVb;
		  }
		  dVaux_dVg *= dVgs_eff_dVg;
              }

              T2 = Vgs_eff * Vaux;
              dT2_dVg = dVgs_eff_dVg * Vaux + Vgs_eff * dVaux_dVg;
              dT2_dVd = Vgs_eff * dVaux_dVd;
              dT2_dVb = Vgs_eff * dVaux_dVb;

              T11 = pParam->BSIM4v5Aechvb;
              T12 = pParam->BSIM4v5Bechvb;
              T3 = pParam->BSIM4v5aigc * pParam->BSIM4v5cigc
                 - pParam->BSIM4v5bigc;
              T4 = pParam->BSIM4v5bigc * pParam->BSIM4v5cigc;
              T5 = T12 * (pParam->BSIM4v5aigc + T3 * Voxdepinv
                 - T4 * Voxdepinv * Voxdepinv);

              if (T5 > EXP_THRESHOLD)
              {   T6 = MAX_EXP;
                  dT6_dVg = dT6_dVd = dT6_dVb = 0.0;
              }
              else if (T5 < -EXP_THRESHOLD)
              {   T6 = MIN_EXP;
                  dT6_dVg = dT6_dVd = dT6_dVb = 0.0;
              }
              else
              {   T6 = exp(T5);
                  dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
                  dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
                  dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
                  dT6_dVg *= dVoxdepinv_dVg;
              }

              Igc = T11 * T2 * T6;
              dIgc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
              dIgc_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
              dIgc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);

              if (model->BSIM4v5pigcdGiven)
              {   Pigcd = pParam->BSIM4v5pigcd;
                  dPigcd_dVg = dPigcd_dVd = dPigcd_dVb = 0.0;
              }
              else
              {   T11 = pParam->BSIM4v5Bechvb * model->BSIM4v5toxe;
                  T12 = Vgsteff + 1.0e-20;
                  T13 = T11 / T12 / T12;
                  T14 = -T13 / T12;
                  Pigcd = T13 * (1.0 - 0.5 * Vdseff / T12); 
                  dPigcd_dVg = T14 * (2.0 + 0.5 * (dVdseff_dVg
                              - 3.0 * Vdseff / T12));
                  dPigcd_dVd = 0.5 * T14 * dVdseff_dVd;
                  dPigcd_dVb = 0.5 * T14 * dVdseff_dVb;
              }

              T7 = -Pigcd * Vdseff; /* bugfix */
              dT7_dVg = -Vdseff * dPigcd_dVg - Pigcd * dVdseff_dVg;
              dT7_dVd = -Vdseff * dPigcd_dVd - Pigcd * dVdseff_dVd + dT7_dVg * dVgsteff_dVd;
              dT7_dVb = -Vdseff * dPigcd_dVb - Pigcd * dVdseff_dVb + dT7_dVg * dVgsteff_dVb;
              dT7_dVg *= dVgsteff_dVg;
              dT7_dVb *= dVbseff_dVb;
              T8 = T7 * T7 + 2.0e-4;
              dT8_dVg = 2.0 * T7;
              dT8_dVd = dT8_dVg * dT7_dVd;
              dT8_dVb = dT8_dVg * dT7_dVb;
              dT8_dVg *= dT7_dVg;

              if (T7 > EXP_THRESHOLD)
              {   T9 = MAX_EXP;
                  dT9_dVg = dT9_dVd = dT9_dVb = 0.0;
              }
              else if (T7 < -EXP_THRESHOLD)
              {   T9 = MIN_EXP;
                  dT9_dVg = dT9_dVd = dT9_dVb = 0.0;
              }
              else
              {   T9 = exp(T7);
                  dT9_dVg = T9 * dT7_dVg;
                  dT9_dVd = T9 * dT7_dVd;
                  dT9_dVb = T9 * dT7_dVb;
              }

              T0 = T8 * T8;
              T1 = T9 - 1.0 + 1.0e-4;
              T10 = (T1 - T7) / T8;
              dT10_dVg = (dT9_dVg - dT7_dVg - T10 * dT8_dVg) / T8;
              dT10_dVd = (dT9_dVd - dT7_dVd - T10 * dT8_dVd) / T8;
              dT10_dVb = (dT9_dVb - dT7_dVb - T10 * dT8_dVb) / T8;

              Igcs = Igc * T10;
              dIgcs_dVg = dIgc_dVg * T10 + Igc * dT10_dVg;
              dIgcs_dVd = dIgc_dVd * T10 + Igc * dT10_dVd;
              dIgcs_dVb = dIgc_dVb * T10 + Igc * dT10_dVb;

              T1 = T9 - 1.0 - 1.0e-4;
              T10 = (T7 * T9 - T1) / T8;
              dT10_dVg = (dT7_dVg * T9 + (T7 - 1.0) * dT9_dVg
                       - T10 * dT8_dVg) / T8;
              dT10_dVd = (dT7_dVd * T9 + (T7 - 1.0) * dT9_dVd
                       - T10 * dT8_dVd) / T8;
              dT10_dVb = (dT7_dVb * T9 + (T7 - 1.0) * dT9_dVb
                       - T10 * dT8_dVb) / T8;
              Igcd = Igc * T10;
              dIgcd_dVg = dIgc_dVg * T10 + Igc * dT10_dVg;
              dIgcd_dVd = dIgc_dVd * T10 + Igc * dT10_dVd;
              dIgcd_dVb = dIgc_dVb * T10 + Igc * dT10_dVb;

              here->BSIM4v5Igcs = Igcs;
              here->BSIM4v5gIgcsg = dIgcs_dVg;
              here->BSIM4v5gIgcsd = dIgcs_dVd;
              here->BSIM4v5gIgcsb =  dIgcs_dVb * dVbseff_dVb;
              here->BSIM4v5Igcd = Igcd;
              here->BSIM4v5gIgcdg = dIgcd_dVg;
              here->BSIM4v5gIgcdd = dIgcd_dVd;
              here->BSIM4v5gIgcdb = dIgcd_dVb * dVbseff_dVb;

              T0 = vgs - (pParam->BSIM4v5vfbsd + pParam->BSIM4v5vfbsdoff);
              vgs_eff = sqrt(T0 * T0 + 1.0e-4);
              dvgs_eff_dvg = T0 / vgs_eff;

              T2 = vgs * vgs_eff;
              dT2_dVg = vgs * dvgs_eff_dvg + vgs_eff;
              T11 = pParam->BSIM4v5AechvbEdge;
              T12 = pParam->BSIM4v5BechvbEdge;
              T3 = pParam->BSIM4v5aigsd * pParam->BSIM4v5cigsd
                 - pParam->BSIM4v5bigsd;
              T4 = pParam->BSIM4v5bigsd * pParam->BSIM4v5cigsd;
              T5 = T12 * (pParam->BSIM4v5aigsd + T3 * vgs_eff
                 - T4 * vgs_eff * vgs_eff);
              if (T5 > EXP_THRESHOLD)
              {   T6 = MAX_EXP;
                  dT6_dVg = 0.0;
              }
              else if (T5 < -EXP_THRESHOLD)
              {   T6 = MIN_EXP;
                  dT6_dVg = 0.0;
              }
              else
              {   T6 = exp(T5);
                  dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgs_eff)
			  * dvgs_eff_dvg;
              }
              Igs = T11 * T2 * T6;
              dIgs_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
              dIgs_dVs = -dIgs_dVg;


              T0 = vgd - (pParam->BSIM4v5vfbsd + pParam->BSIM4v5vfbsdoff);
              vgd_eff = sqrt(T0 * T0 + 1.0e-4);
              dvgd_eff_dvg = T0 / vgd_eff;

              T2 = vgd * vgd_eff;
              dT2_dVg = vgd * dvgd_eff_dvg + vgd_eff;
              T5 = T12 * (pParam->BSIM4v5aigsd + T3 * vgd_eff
                 - T4 * vgd_eff * vgd_eff);
              if (T5 > EXP_THRESHOLD)
              {   T6 = MAX_EXP;
                  dT6_dVg = 0.0;
              }
              else if (T5 < -EXP_THRESHOLD)
              {   T6 = MIN_EXP;
                  dT6_dVg = 0.0;
              }
              else
              {   T6 = exp(T5);
                  dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgd_eff)
                          * dvgd_eff_dvg;
              }
              Igd = T11 * T2 * T6;
              dIgd_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
              dIgd_dVd = -dIgd_dVg;

              here->BSIM4v5Igs = Igs;
              here->BSIM4v5gIgsg = dIgs_dVg;
              here->BSIM4v5gIgss = dIgs_dVs;
              here->BSIM4v5Igd = Igd;
              here->BSIM4v5gIgdg = dIgd_dVg;
              here->BSIM4v5gIgdd = dIgd_dVd;
          }
          else
          {   here->BSIM4v5Igcs = here->BSIM4v5gIgcsg = here->BSIM4v5gIgcsd
 			      = here->BSIM4v5gIgcsb = 0.0;
              here->BSIM4v5Igcd = here->BSIM4v5gIgcdg = here->BSIM4v5gIgcdd
              		      = here->BSIM4v5gIgcdb = 0.0;
              here->BSIM4v5Igs = here->BSIM4v5gIgsg = here->BSIM4v5gIgss = 0.0;
              here->BSIM4v5Igd = here->BSIM4v5gIgdg = here->BSIM4v5gIgdd = 0.0;
          }

          if (model->BSIM4v5igbMod)
          {   T0 = tmp * pParam->BSIM4v5nigbacc;
	      T1 = -Vgs_eff + Vbseff + Vfb;
              VxNVt = T1 / T0;
              if (VxNVt > EXP_THRESHOLD)
              {   Vaux = T1;
                  dVaux_dVg = -dVgs_eff_dVg;
                  dVaux_dVb = 1.0;
              }
              else if (VxNVt < -EXP_THRESHOLD)
              {   Vaux = T0 * log(1.0 + MIN_EXP);
                  dVaux_dVg = dVaux_dVb = 0.0;
              }
              else
              {   ExpVxNVt = exp(VxNVt);
                  Vaux = T0 * log(1.0 + ExpVxNVt);
                  dVaux_dVb = ExpVxNVt / (1.0 + ExpVxNVt); 
                  dVaux_dVg = -dVaux_dVb * dVgs_eff_dVg;
              }

	      T2 = (Vgs_eff - Vbseff) * Vaux;
              dT2_dVg = dVgs_eff_dVg * Vaux + (Vgs_eff - Vbseff) * dVaux_dVg;
              dT2_dVb = -Vaux + (Vgs_eff - Vbseff) * dVaux_dVb;

              T11 = 4.97232e-7 * pParam->BSIM4v5weff
		  * pParam->BSIM4v5leff * pParam->BSIM4v5ToxRatio;
              T12 = -7.45669e11 * model->BSIM4v5toxe;
	      T3 = pParam->BSIM4v5aigbacc * pParam->BSIM4v5cigbacc
                 - pParam->BSIM4v5bigbacc;
              T4 = pParam->BSIM4v5bigbacc * pParam->BSIM4v5cigbacc;
	      T5 = T12 * (pParam->BSIM4v5aigbacc + T3 * Voxacc
		 - T4 * Voxacc * Voxacc);

              if (T5 > EXP_THRESHOLD)
              {   T6 = MAX_EXP;
                  dT6_dVg = dT6_dVb = 0.0;
              }
              else if (T5 < -EXP_THRESHOLD)
              {   T6 = MIN_EXP;
                  dT6_dVg = dT6_dVb = 0.0;
              }
              else
              {   T6 = exp(T5);
                  dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxacc);
                  dT6_dVb = dT6_dVg * dVoxacc_dVb;
                  dT6_dVg *= dVoxacc_dVg;
              }

              Igbacc = T11 * T2 * T6;
              dIgbacc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
              dIgbacc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);


              T0 = tmp * pParam->BSIM4v5nigbinv;
              T1 = Voxdepinv - pParam->BSIM4v5eigbinv;
              VxNVt = T1 / T0;
              if (VxNVt > EXP_THRESHOLD)
              {   Vaux = T1;
                  dVaux_dVg = dVoxdepinv_dVg;
                  dVaux_dVd = dVoxdepinv_dVd;
                  dVaux_dVb = dVoxdepinv_dVb;
              }
              else if (VxNVt < -EXP_THRESHOLD)
              {   Vaux = T0 * log(1.0 + MIN_EXP);
                  dVaux_dVg = dVaux_dVd = dVaux_dVb = 0.0;
              }
              else
              {   ExpVxNVt = exp(VxNVt);
                  Vaux = T0 * log(1.0 + ExpVxNVt);
		  dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
                  dVaux_dVd = dVaux_dVg * dVoxdepinv_dVd;
                  dVaux_dVb = dVaux_dVg * dVoxdepinv_dVb;
                  dVaux_dVg *= dVoxdepinv_dVg;
              }

              T2 = (Vgs_eff - Vbseff) * Vaux;
              dT2_dVg = dVgs_eff_dVg * Vaux + (Vgs_eff - Vbseff) * dVaux_dVg;
              dT2_dVd = (Vgs_eff - Vbseff) * dVaux_dVd;
              dT2_dVb = -Vaux + (Vgs_eff - Vbseff) * dVaux_dVb;

              T11 *= 0.75610;
              T12 *= 1.31724;
              T3 = pParam->BSIM4v5aigbinv * pParam->BSIM4v5cigbinv
                 - pParam->BSIM4v5bigbinv;
              T4 = pParam->BSIM4v5bigbinv * pParam->BSIM4v5cigbinv;
              T5 = T12 * (pParam->BSIM4v5aigbinv + T3 * Voxdepinv
                 - T4 * Voxdepinv * Voxdepinv);

              if (T5 > EXP_THRESHOLD)
              {   T6 = MAX_EXP;
                  dT6_dVg = dT6_dVd = dT6_dVb = 0.0;
              }
              else if (T5 < -EXP_THRESHOLD)
              {   T6 = MIN_EXP;
                  dT6_dVg = dT6_dVd = dT6_dVb = 0.0;
              }
              else
              {   T6 = exp(T5);
                  dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
                  dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
                  dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
                  dT6_dVg *= dVoxdepinv_dVg;
              }

              Igbinv = T11 * T2 * T6;
              dIgbinv_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
              dIgbinv_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
              dIgbinv_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);

              here->BSIM4v5Igb = Igbinv + Igbacc;
              here->BSIM4v5gIgbg = dIgbinv_dVg + dIgbacc_dVg;
              here->BSIM4v5gIgbd = dIgbinv_dVd;
              here->BSIM4v5gIgbb = (dIgbinv_dVb + dIgbacc_dVb) * dVbseff_dVb;
          }
          else
          {  here->BSIM4v5Igb = here->BSIM4v5gIgbg = here->BSIM4v5gIgbd
			    = here->BSIM4v5gIgbs = here->BSIM4v5gIgbb = 0.0;   
          } /* End of Gate current */

	  if (here->BSIM4v5nf != 1.0)
          {   cdrain *= here->BSIM4v5nf;
              here->BSIM4v5gds *= here->BSIM4v5nf;
              here->BSIM4v5gm *= here->BSIM4v5nf;
              here->BSIM4v5gmbs *= here->BSIM4v5nf;
	      here->BSIM4v5IdovVds *= here->BSIM4v5nf;
                   
              here->BSIM4v5gbbs *= here->BSIM4v5nf;
              here->BSIM4v5gbgs *= here->BSIM4v5nf;
              here->BSIM4v5gbds *= here->BSIM4v5nf;
              here->BSIM4v5csub *= here->BSIM4v5nf;

              here->BSIM4v5Igidl *= here->BSIM4v5nf;
              here->BSIM4v5ggidld *= here->BSIM4v5nf;
              here->BSIM4v5ggidlg *= here->BSIM4v5nf;
	      here->BSIM4v5ggidlb *= here->BSIM4v5nf;
	
              here->BSIM4v5Igisl *= here->BSIM4v5nf;
              here->BSIM4v5ggisls *= here->BSIM4v5nf;
              here->BSIM4v5ggislg *= here->BSIM4v5nf;
              here->BSIM4v5ggislb *= here->BSIM4v5nf;

              here->BSIM4v5Igcs *= here->BSIM4v5nf;
              here->BSIM4v5gIgcsg *= here->BSIM4v5nf;
              here->BSIM4v5gIgcsd *= here->BSIM4v5nf;
              here->BSIM4v5gIgcsb *= here->BSIM4v5nf;
              here->BSIM4v5Igcd *= here->BSIM4v5nf;
              here->BSIM4v5gIgcdg *= here->BSIM4v5nf;
              here->BSIM4v5gIgcdd *= here->BSIM4v5nf;
              here->BSIM4v5gIgcdb *= here->BSIM4v5nf;

              here->BSIM4v5Igs *= here->BSIM4v5nf;
              here->BSIM4v5gIgsg *= here->BSIM4v5nf;
              here->BSIM4v5gIgss *= here->BSIM4v5nf;
              here->BSIM4v5Igd *= here->BSIM4v5nf;
              here->BSIM4v5gIgdg *= here->BSIM4v5nf;
              here->BSIM4v5gIgdd *= here->BSIM4v5nf;

              here->BSIM4v5Igb *= here->BSIM4v5nf;
              here->BSIM4v5gIgbg *= here->BSIM4v5nf;
              here->BSIM4v5gIgbd *= here->BSIM4v5nf;
              here->BSIM4v5gIgbb *= here->BSIM4v5nf;
	  }

          here->BSIM4v5ggidls = -(here->BSIM4v5ggidld + here->BSIM4v5ggidlg
			    + here->BSIM4v5ggidlb);
          here->BSIM4v5ggisld = -(here->BSIM4v5ggisls + here->BSIM4v5ggislg
			    + here->BSIM4v5ggislb);
          here->BSIM4v5gIgbs = -(here->BSIM4v5gIgbg + here->BSIM4v5gIgbd
                           + here->BSIM4v5gIgbb);
          here->BSIM4v5gIgcss = -(here->BSIM4v5gIgcsg + here->BSIM4v5gIgcsd
                            + here->BSIM4v5gIgcsb);
          here->BSIM4v5gIgcds = -(here->BSIM4v5gIgcdg + here->BSIM4v5gIgcdd
                            + here->BSIM4v5gIgcdb);
	  here->BSIM4v5cd = cdrain;


          if (model->BSIM4v5tnoiMod == 0)
          {   Abulk = Abulk0 * pParam->BSIM4v5abulkCVfactor;
              Vdsat = Vgsteff / Abulk;
              T0 = Vdsat - Vds - DELTA_4;
              T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * Vdsat);
              if (T0 >= 0.0)
                  Vdseff = Vdsat - 0.5 * (T0 + T1);
              else
              {   T3 = (DELTA_4 + DELTA_4) / (T1 - T0);
                  T4 = 1.0 - T3;
                  T5 = Vdsat * T3 / (T1 - T0);
                  Vdseff = Vdsat * T4;
              }
              if (Vds == 0.0)
                  Vdseff = 0.0;

              T0 = Abulk * Vdseff;
              T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20);
              T2 = Vdseff / T1;
              T3 = T0 * T2;
              here->BSIM4v5qinv = Coxeff * pParam->BSIM4v5weffCV * here->BSIM4v5nf
                              * pParam->BSIM4v5leffCV
                              * (Vgsteff - 0.5 * T0 + Abulk * T3);
          }

	  /*
           *  BSIM4v5 C-V begins
	   */

          if ((model->BSIM4v5xpart < 0) || (!ChargeComputationNeeded))
	  {   qgate  = qdrn = qsrc = qbulk = 0.0;
              here->BSIM4v5cggb = here->BSIM4v5cgsb = here->BSIM4v5cgdb = 0.0;
              here->BSIM4v5cdgb = here->BSIM4v5cdsb = here->BSIM4v5cddb = 0.0;
              here->BSIM4v5cbgb = here->BSIM4v5cbsb = here->BSIM4v5cbdb = 0.0;
              here->BSIM4v5csgb = here->BSIM4v5cssb = here->BSIM4v5csdb = 0.0;
              here->BSIM4v5cgbb = here->BSIM4v5csbb = here->BSIM4v5cdbb = here->BSIM4v5cbbb = 0.0;
              here->BSIM4v5cqdb = here->BSIM4v5cqsb = here->BSIM4v5cqgb 
                              = here->BSIM4v5cqbb = 0.0;
              here->BSIM4v5gtau = 0.0;
              goto finished;
          }
	  else if (model->BSIM4v5capMod == 0)
	  {
              if (Vbseff < 0.0)
	      {   Vbseff = Vbs;
                  dVbseff_dVb = 1.0;
              }
	      else
	      {   Vbseff = pParam->BSIM4v5phi - Phis;
                  dVbseff_dVb = -dPhis_dVb;
              }

              Vfb = pParam->BSIM4v5vfbcv;
              Vth = Vfb + pParam->BSIM4v5phi + pParam->BSIM4v5k1ox * sqrtPhis; 
              Vgst = Vgs_eff - Vth;
              dVth_dVb = pParam->BSIM4v5k1ox * dsqrtPhis_dVb; 
              dVgst_dVb = -dVth_dVb;
              dVgst_dVg = dVgs_eff_dVg; 

              CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV
                    * pParam->BSIM4v5leffCV * here->BSIM4v5nf;
              Arg1 = Vgs_eff - Vbseff - Vfb;

              if (Arg1 <= 0.0)
	      {   qgate = CoxWL * Arg1;
                  qbulk = -qgate;
                  qdrn = 0.0;

                  here->BSIM4v5cggb = CoxWL * dVgs_eff_dVg;
                  here->BSIM4v5cgdb = 0.0;
                  here->BSIM4v5cgsb = CoxWL * (dVbseff_dVb - dVgs_eff_dVg);

                  here->BSIM4v5cdgb = 0.0;
                  here->BSIM4v5cddb = 0.0;
                  here->BSIM4v5cdsb = 0.0;

                  here->BSIM4v5cbgb = -CoxWL * dVgs_eff_dVg;
                  here->BSIM4v5cbdb = 0.0;
                  here->BSIM4v5cbsb = -here->BSIM4v5cgsb;
              } /* Arg1 <= 0.0, end of accumulation */
	      else if (Vgst <= 0.0)
	      {   T1 = 0.5 * pParam->BSIM4v5k1ox;
	          T2 = sqrt(T1 * T1 + Arg1);
	          qgate = CoxWL * pParam->BSIM4v5k1ox * (T2 - T1);
                  qbulk = -qgate;
                  qdrn = 0.0;

	          T0 = CoxWL * T1 / T2;
	          here->BSIM4v5cggb = T0 * dVgs_eff_dVg;
	          here->BSIM4v5cgdb = 0.0;
                  here->BSIM4v5cgsb = T0 * (dVbseff_dVb - dVgs_eff_dVg);
   
                  here->BSIM4v5cdgb = 0.0;
                  here->BSIM4v5cddb = 0.0;
                  here->BSIM4v5cdsb = 0.0;

                  here->BSIM4v5cbgb = -here->BSIM4v5cggb;
                  here->BSIM4v5cbdb = 0.0;
                  here->BSIM4v5cbsb = -here->BSIM4v5cgsb;
              } /* Vgst <= 0.0, end of depletion */
	      else
	      {   One_Third_CoxWL = CoxWL / 3.0;
                  Two_Third_CoxWL = 2.0 * One_Third_CoxWL;

                  AbulkCV = Abulk0 * pParam->BSIM4v5abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM4v5abulkCVfactor * dAbulk0_dVb;
	          Vdsat = Vgst / AbulkCV;
	          dVdsat_dVg = dVgs_eff_dVg / AbulkCV;
	          dVdsat_dVb = - (Vdsat * dAbulkCV_dVb + dVth_dVb)/ AbulkCV; 

                  if (model->BSIM4v5xpart > 0.5)
		  {   /* 0/100 Charge partition model */
		      if (Vdsat <= Vds)
		      {   /* saturation region */
	                  T1 = Vdsat / 3.0;
	                  qgate = CoxWL * (Vgs_eff - Vfb
			        - pParam->BSIM4v5phi - T1);
	                  T2 = -Two_Third_CoxWL * Vgst;
	                  qbulk = -(qgate + T2);
	                  qdrn = 0.0;

	                  here->BSIM4v5cggb = One_Third_CoxWL * (3.0
					  - dVdsat_dVg) * dVgs_eff_dVg;
	                  T2 = -One_Third_CoxWL * dVdsat_dVb;
	                  here->BSIM4v5cgsb = -(here->BSIM4v5cggb + T2);
                          here->BSIM4v5cgdb = 0.0;
       
                          here->BSIM4v5cdgb = 0.0;
                          here->BSIM4v5cddb = 0.0;
                          here->BSIM4v5cdsb = 0.0;

	                  here->BSIM4v5cbgb = -(here->BSIM4v5cggb
					  - Two_Third_CoxWL * dVgs_eff_dVg);
	                  T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                  here->BSIM4v5cbsb = -(here->BSIM4v5cbgb + T3);
                          here->BSIM4v5cbdb = 0.0;
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
			        - pParam->BSIM4v5phi - 0.5 * (Vds - T3));
	                  T10 = T4 * T8;
	                  qdrn = T4 * T7;
	                  qbulk = -(qgate + qdrn + T10);
  
                          T5 = T3 / T1;
                          here->BSIM4v5cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					  * dVgs_eff_dVg;
                          T11 = -CoxWL * T5 * dVdsat_dVb;
                          here->BSIM4v5cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                          here->BSIM4v5cgsb = -(here->BSIM4v5cggb + T11
                                          + here->BSIM4v5cgdb);
                          T6 = 1.0 / Vdsat;
                          dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                          dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);
                          T7 = T9 * T7;
                          T8 = T9 * T8;
                          T9 = 2.0 * T4 * (1.0 - 3.0 * T5);
                          here->BSIM4v5cdgb = (T7 * dAlphaz_dVg - T9
					  * dVdsat_dVg) * dVgs_eff_dVg;
                          T12 = T7 * dAlphaz_dVb - T9 * dVdsat_dVb;
                          here->BSIM4v5cddb = T4 * (3.0 - 6.0 * T2 - 3.0 * T5);
                          here->BSIM4v5cdsb = -(here->BSIM4v5cdgb + T12
                                          + here->BSIM4v5cddb);

                          T9 = 2.0 * T4 * (1.0 + T5);
                          T10 = (T8 * dAlphaz_dVg - T9 * dVdsat_dVg)
			      * dVgs_eff_dVg;
                          T11 = T8 * dAlphaz_dVb - T9 * dVdsat_dVb;
                          T12 = T4 * (2.0 * T2 + T5 - 1.0); 
                          T0 = -(T10 + T11 + T12);

                          here->BSIM4v5cbgb = -(here->BSIM4v5cggb
					  + here->BSIM4v5cdgb + T10);
                          here->BSIM4v5cbdb = -(here->BSIM4v5cgdb 
					  + here->BSIM4v5cddb + T12);
                          here->BSIM4v5cbsb = -(here->BSIM4v5cgsb
					  + here->BSIM4v5cdsb + T0);
                      }
                  }
		  else if (model->BSIM4v5xpart < 0.5)
		  {   /* 40/60 Charge partition model */
		      if (Vds >= Vdsat)
		      {   /* saturation region */
	                  T1 = Vdsat / 3.0;
	                  qgate = CoxWL * (Vgs_eff - Vfb
			        - pParam->BSIM4v5phi - T1);
	                  T2 = -Two_Third_CoxWL * Vgst;
	                  qbulk = -(qgate + T2);
	                  qdrn = 0.4 * T2;

	                  here->BSIM4v5cggb = One_Third_CoxWL * (3.0 
					  - dVdsat_dVg) * dVgs_eff_dVg;
	                  T2 = -One_Third_CoxWL * dVdsat_dVb;
	                  here->BSIM4v5cgsb = -(here->BSIM4v5cggb + T2);
        	          here->BSIM4v5cgdb = 0.0;
       
			  T3 = 0.4 * Two_Third_CoxWL;
                          here->BSIM4v5cdgb = -T3 * dVgs_eff_dVg;
                          here->BSIM4v5cddb = 0.0;
			  T4 = T3 * dVth_dVb;
                          here->BSIM4v5cdsb = -(T4 + here->BSIM4v5cdgb);

	                  here->BSIM4v5cbgb = -(here->BSIM4v5cggb 
					  - Two_Third_CoxWL * dVgs_eff_dVg);
	                  T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                  here->BSIM4v5cbsb = -(here->BSIM4v5cbgb + T3);
                          here->BSIM4v5cbdb = 0.0;
	              }
		      else
		      {   /* linear region  */
			  Alphaz = Vgst / Vdsat;
			  T1 = 2.0 * Vdsat - Vds;
			  T2 = Vds / (3.0 * T1);
			  T3 = T2 * Vds;
			  T9 = 0.25 * CoxWL;
			  T4 = T9 * Alphaz;
			  qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v5phi
				- 0.5 * (Vds - T3));

			  T5 = T3 / T1;
                          here->BSIM4v5cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					  * dVgs_eff_dVg;
                          tmp = -CoxWL * T5 * dVdsat_dVb;
                          here->BSIM4v5cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                          here->BSIM4v5cgsb = -(here->BSIM4v5cggb 
					  + here->BSIM4v5cgdb + tmp);

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

                          here->BSIM4v5cdgb = (T7 * dAlphaz_dVg - tmp1
					  * dVdsat_dVg) * dVgs_eff_dVg;
                          T10 = T7 * dAlphaz_dVb - tmp1 * dVdsat_dVb;
                          here->BSIM4v5cddb = T4 * (2.0 - (1.0 / (3.0 * T1
					  * T1) + 2.0 * tmp) * T6 + T8
					  * (6.0 * Vdsat - 2.4 * Vds));
                          here->BSIM4v5cdsb = -(here->BSIM4v5cdgb 
					  + T10 + here->BSIM4v5cddb);

			  T7 = 2.0 * (T1 + T3);
			  qbulk = -(qgate - T4 * T7);
			  T7 *= T9;
			  T0 = 4.0 * T4 * (1.0 - T5);
			  T12 = (-T7 * dAlphaz_dVg - here->BSIM4v5cdgb
			      - T0 * dVdsat_dVg) * dVgs_eff_dVg;
			  T11 = -T7 * dAlphaz_dVb - T10 - T0 * dVdsat_dVb;
			  T10 = -4.0 * T4 * (T2 - 0.5 + 0.5 * T5) 
			      - here->BSIM4v5cddb;
                          tmp = -(T10 + T11 + T12);

                          here->BSIM4v5cbgb = -(here->BSIM4v5cggb 
					  + here->BSIM4v5cdgb + T12);
                          here->BSIM4v5cbdb = -(here->BSIM4v5cgdb
					  + here->BSIM4v5cddb + T10);  
                          here->BSIM4v5cbsb = -(here->BSIM4v5cgsb
					  + here->BSIM4v5cdsb + tmp);
                      }
                  }
		  else
		  {   /* 50/50 partitioning */
		      if (Vds >= Vdsat)
		      {   /* saturation region */
	                  T1 = Vdsat / 3.0;
	                  qgate = CoxWL * (Vgs_eff - Vfb
			        - pParam->BSIM4v5phi - T1);
	                  T2 = -Two_Third_CoxWL * Vgst;
	                  qbulk = -(qgate + T2);
	                  qdrn = 0.5 * T2;

	                  here->BSIM4v5cggb = One_Third_CoxWL * (3.0
					  - dVdsat_dVg) * dVgs_eff_dVg;
	                  T2 = -One_Third_CoxWL * dVdsat_dVb;
	                  here->BSIM4v5cgsb = -(here->BSIM4v5cggb + T2);
        	          here->BSIM4v5cgdb = 0.0;
       
                          here->BSIM4v5cdgb = -One_Third_CoxWL * dVgs_eff_dVg;
                          here->BSIM4v5cddb = 0.0;
			  T4 = One_Third_CoxWL * dVth_dVb;
                          here->BSIM4v5cdsb = -(T4 + here->BSIM4v5cdgb);

	                  here->BSIM4v5cbgb = -(here->BSIM4v5cggb 
					  - Two_Third_CoxWL * dVgs_eff_dVg);
	                  T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
	                  here->BSIM4v5cbsb = -(here->BSIM4v5cbgb + T3);
                          here->BSIM4v5cbdb = 0.0;
	              }
		      else
		      {   /* linear region */
			  Alphaz = Vgst / Vdsat;
			  T1 = 2.0 * Vdsat - Vds;
			  T2 = Vds / (3.0 * T1);
			  T3 = T2 * Vds;
			  T9 = 0.25 * CoxWL;
			  T4 = T9 * Alphaz;
			  qgate = CoxWL * (Vgs_eff - Vfb - pParam->BSIM4v5phi
				- 0.5 * (Vds - T3));

			  T5 = T3 / T1;
                          here->BSIM4v5cggb = CoxWL * (1.0 - T5 * dVdsat_dVg)
					  * dVgs_eff_dVg;
                          tmp = -CoxWL * T5 * dVdsat_dVb;
                          here->BSIM4v5cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                          here->BSIM4v5cgsb = -(here->BSIM4v5cggb 
					  + here->BSIM4v5cgdb + tmp);

			  T6 = 1.0 / Vdsat;
                          dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                          dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);

			  T7 = T1 + T3;
			  qdrn = -T4 * T7;
			  qbulk = - (qgate + qdrn + qdrn);
			  T7 *= T9;
			  T0 = T4 * (2.0 * T5 - 2.0);

                          here->BSIM4v5cdgb = (T0 * dVdsat_dVg - T7
					  * dAlphaz_dVg) * dVgs_eff_dVg;
			  T12 = T0 * dVdsat_dVb - T7 * dAlphaz_dVb;
                          here->BSIM4v5cddb = T4 * (1.0 - 2.0 * T2 - T5);
                          here->BSIM4v5cdsb = -(here->BSIM4v5cdgb + T12
                                          + here->BSIM4v5cddb);

                          here->BSIM4v5cbgb = -(here->BSIM4v5cggb
					  + 2.0 * here->BSIM4v5cdgb);
                          here->BSIM4v5cbdb = -(here->BSIM4v5cgdb
					  + 2.0 * here->BSIM4v5cddb);
                          here->BSIM4v5cbsb = -(here->BSIM4v5cgsb
					  + 2.0 * here->BSIM4v5cdsb);
                      } /* end of linear region */
	          } /* end of 50/50 partition */
	      } /* end of inversion */
          } /* end of capMod=0 */ 
	  else
	  {   if (Vbseff < 0.0)
	      {   VbseffCV = Vbseff;
                  dVbseffCV_dVb = 1.0;
              }
	      else
	      {   VbseffCV = pParam->BSIM4v5phi - Phis;
                  dVbseffCV_dVb = -dPhis_dVb;
              }

              CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV
		    * pParam->BSIM4v5leffCV * here->BSIM4v5nf;

              /* Seperate VgsteffCV with noff and voffcv */
              noff = n * pParam->BSIM4v5noff;
              dnoff_dVd = pParam->BSIM4v5noff * dn_dVd;
              dnoff_dVb = pParam->BSIM4v5noff * dn_dVb;
              T0 = Vtm * noff;
              voffcv = pParam->BSIM4v5voffcv;
              VgstNVt = (Vgst - voffcv) / T0;

              if (VgstNVt > EXP_THRESHOLD)
              {   Vgsteff = Vgst - voffcv;
                  dVgsteff_dVg = dVgs_eff_dVg;
                  dVgsteff_dVd = -dVth_dVd;
                  dVgsteff_dVb = -dVth_dVb;
              }
              else if (VgstNVt < -EXP_THRESHOLD)
              {   Vgsteff = T0 * log(1.0 + MIN_EXP);
                  dVgsteff_dVg = 0.0;
                  dVgsteff_dVd = Vgsteff / noff;
                  dVgsteff_dVb = dVgsteff_dVd * dnoff_dVb;
                  dVgsteff_dVd *= dnoff_dVd;
              }
              else
              {   ExpVgst = exp(VgstNVt);
                  Vgsteff = T0 * log(1.0 + ExpVgst);
                  dVgsteff_dVg = ExpVgst / (1.0 + ExpVgst);
                  dVgsteff_dVd = -dVgsteff_dVg * (dVth_dVd + (Vgst - voffcv)
                               / noff * dnoff_dVd) + Vgsteff / noff * dnoff_dVd;
                  dVgsteff_dVb = -dVgsteff_dVg * (dVth_dVb + (Vgst - voffcv)
                               / noff * dnoff_dVb) + Vgsteff / noff * dnoff_dVb;
                  dVgsteff_dVg *= dVgs_eff_dVg;
              } /* End of VgsteffCV */


	      if (model->BSIM4v5capMod == 1)
	      {   Vfb = here->BSIM4v5vfbzb;
                  V3 = Vfb - Vgs_eff + VbseffCV - DELTA_3;
		  if (Vfb <= 0.0)
		      T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfb);
		  else
		      T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfb);

		  T1 = 0.5 * (1.0 + V3 / T0);
		  Vfbeff = Vfb - 0.5 * (V3 + T0);
		  dVfbeff_dVg = T1 * dVgs_eff_dVg;
		  dVfbeff_dVb = -T1 * dVbseffCV_dVb;
		  Qac0 = CoxWL * (Vfbeff - Vfb);
		  dQac0_dVg = CoxWL * dVfbeff_dVg;
		  dQac0_dVb = CoxWL * dVfbeff_dVb;

                  T0 = 0.5 * pParam->BSIM4v5k1ox;
		  T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
                  if (pParam->BSIM4v5k1ox == 0.0)
                  {   T1 = 0.0;
                      T2 = 0.0;
                  }
		  else if (T3 < 0.0)
		  {   T1 = T0 + T3 / pParam->BSIM4v5k1ox;
                      T2 = CoxWL;
		  }
		  else
		  {   T1 = sqrt(T0 * T0 + T3);
                      T2 = CoxWL * T0 / T1;
		  }

		  Qsub0 = CoxWL * pParam->BSIM4v5k1ox * (T1 - T0);

                  dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
                  dQsub0_dVd = -T2 * dVgsteff_dVd;
                  dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb 
                             + dVgsteff_dVb);

                  AbulkCV = Abulk0 * pParam->BSIM4v5abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM4v5abulkCVfactor * dAbulk0_dVb;
	          VdsatCV = Vgsteff / AbulkCV;

	 	  T0 = VdsatCV - Vds - DELTA_4;
		  dT0_dVg = 1.0 / AbulkCV;
		  dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV; 
		  T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * VdsatCV);
                  dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1;
                  dT1_dVd = -T0 / T1;
                  dT1_dVb = dT1_dVg * dT0_dVb;
		  dT1_dVg *= dT0_dVg;
		  if (T0 >= 0.0)
		  {   VdseffCV = VdsatCV - 0.5 * (T0 + T1);
                      dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg);
                      dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd);
                      dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb);
		  }
                  else
                  {   T3 = (DELTA_4 + DELTA_4) / (T1 - T0);
		      T4 = 1.0 - T3;
		      T5 = VdsatCV * T3 / (T1 - T0);
		      VdseffCV = VdsatCV * T4;
                      dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg);
                      dVdseffCV_dVd = T5 * (dT1_dVd + 1.0);
                      dVdseffCV_dVb = dT0_dVb * (1.0 - T5) + T5 * dT1_dVb;
                  }

                  if (Vds == 0.0)
                  {  VdseffCV = 0.0;
                     dVdseffCV_dVg = 0.0;
                     dVdseffCV_dVb = 0.0;
                  }

	          T0 = AbulkCV * VdseffCV;
                  T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20);
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

                  if (model->BSIM4v5xpart > 0.5)
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
		  else if (model->BSIM4v5xpart < 0.5)
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
		  Cgd = dQsub0_dVd + Cgd1;
		  Cgb = dQac0_dVb + dQsub0_dVb + Cgb1;

		  Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
		  Cbd = Cbd1 - dQsub0_dVd;
		  Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;

		  Cgb *= dVbseff_dVb;
		  Cbb *= dVbseff_dVb;
		  Csb *= dVbseff_dVb;

                  here->BSIM4v5cggb = Cgg;
	          here->BSIM4v5cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM4v5cgdb = Cgd;
                  here->BSIM4v5cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM4v5cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM4v5cddb = -(Cgd + Cbd + Csd);
                  here->BSIM4v5cbgb = Cbg;
	          here->BSIM4v5cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM4v5cbdb = Cbd;
	      } 

              /* Charge-Thickness capMod (CTM) begins */
	      else if (model->BSIM4v5capMod == 2)
	      {   V3 = here->BSIM4v5vfbzb - Vgs_eff + VbseffCV - DELTA_3;
		  if (here->BSIM4v5vfbzb <= 0.0)
		      T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * here->BSIM4v5vfbzb);
		  else
		      T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * here->BSIM4v5vfbzb);

		  T1 = 0.5 * (1.0 + V3 / T0);
		  Vfbeff = here->BSIM4v5vfbzb - 0.5 * (V3 + T0);
		  dVfbeff_dVg = T1 * dVgs_eff_dVg;
		  dVfbeff_dVb = -T1 * dVbseffCV_dVb;

                  Cox = model->BSIM4v5coxp;
                  Tox = 1.0e8 * model->BSIM4v5toxp;
                  T0 = (Vgs_eff - VbseffCV - here->BSIM4v5vfbzb) / Tox;
                  dT0_dVg = dVgs_eff_dVg / Tox;
                  dT0_dVb = -dVbseffCV_dVb / Tox;

                  tmp = T0 * pParam->BSIM4v5acde;
                  if ((-EXP_THRESHOLD < tmp) && (tmp < EXP_THRESHOLD))
                  {   Tcen = pParam->BSIM4v5ldeb * exp(tmp);
                      dTcen_dVg = pParam->BSIM4v5acde * Tcen;
                      dTcen_dVb = dTcen_dVg * dT0_dVb;
                      dTcen_dVg *= dT0_dVg;
                  }
                  else if (tmp <= -EXP_THRESHOLD)
                  {   Tcen = pParam->BSIM4v5ldeb * MIN_EXP;
                      dTcen_dVg = dTcen_dVb = 0.0;
                  }
                  else
                  {   Tcen = pParam->BSIM4v5ldeb * MAX_EXP;
                      dTcen_dVg = dTcen_dVb = 0.0;
                  }

                  LINK = 1.0e-3 * model->BSIM4v5toxp;
                  V3 = pParam->BSIM4v5ldeb - Tcen - LINK;
                  V4 = sqrt(V3 * V3 + 4.0 * LINK * pParam->BSIM4v5ldeb);
                  Tcen = pParam->BSIM4v5ldeb - 0.5 * (V3 + V4);
                  T1 = 0.5 * (1.0 + V3 / V4);
                  dTcen_dVg *= T1;
                  dTcen_dVb *= T1;

                  Ccen = EPSSI / Tcen;
                  T2 = Cox / (Cox + Ccen);
                  Coxeff = T2 * Ccen;
                  T3 = -Ccen / Tcen;
                  dCoxeff_dVg = T2 * T2 * T3;
                  dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
                  dCoxeff_dVg *= dTcen_dVg;
                  CoxWLcen = CoxWL * Coxeff / model->BSIM4v5coxe;

                  Qac0 = CoxWLcen * (Vfbeff - here->BSIM4v5vfbzb);
                  QovCox = Qac0 / Coxeff;
                  dQac0_dVg = CoxWLcen * dVfbeff_dVg
                            + QovCox * dCoxeff_dVg;
                  dQac0_dVb = CoxWLcen * dVfbeff_dVb 
			    + QovCox * dCoxeff_dVb;

                  T0 = 0.5 * pParam->BSIM4v5k1ox;
                  T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
                  if (pParam->BSIM4v5k1ox == 0.0)
                  {   T1 = 0.0;
                      T2 = 0.0;
                  }
                  else if (T3 < 0.0)
                  {   T1 = T0 + T3 / pParam->BSIM4v5k1ox;
                      T2 = CoxWLcen;
                  }
                  else
                  {   T1 = sqrt(T0 * T0 + T3);
                      T2 = CoxWLcen * T0 / T1;
                  }

                  Qsub0 = CoxWLcen * pParam->BSIM4v5k1ox * (T1 - T0);
                  QovCox = Qsub0 / Coxeff;
                  dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg)
                             + QovCox * dCoxeff_dVg;
                  dQsub0_dVd = -T2 * dVgsteff_dVd;
                  dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb)
                             + QovCox * dCoxeff_dVb;

		  /* Gate-bias dependent delta Phis begins */
		  if (pParam->BSIM4v5k1ox <= 0.0)
		  {   Denomi = 0.25 * pParam->BSIM4v5moin * Vtm;
                      T0 = 0.5 * pParam->BSIM4v5sqrtPhi;
		  }
		  else
		  {   Denomi = pParam->BSIM4v5moin * Vtm 
			     * pParam->BSIM4v5k1ox * pParam->BSIM4v5k1ox;
                      T0 = pParam->BSIM4v5k1ox * pParam->BSIM4v5sqrtPhi;
		  }
                  T1 = 2.0 * T0 + Vgsteff;

		  DeltaPhi = Vtm * log(1.0 + T1 * Vgsteff / Denomi);
		  dDeltaPhi_dVg = 2.0 * Vtm * (T1 -T0) / (Denomi + T1 * Vgsteff);
		  /* End of delta Phis */

		  /* VgDP = Vgsteff - DeltaPhi */
		  T0 = Vgsteff - DeltaPhi - 0.001;
		  dT0_dVg = 1.0 - dDeltaPhi_dVg;
		  T1 = sqrt(T0 * T0 + Vgsteff * 0.004);
		  VgDP = 0.5 * (T0 + T1);
		  dVgDP_dVg = 0.5 * (dT0_dVg + (T0 * dT0_dVg + 0.002) / T1);                  
                  
                  Tox += Tox; /* WDLiu: Tcen reevaluated below due to different Vgsteff */
                  T0 = (Vgsteff + here->BSIM4v5vtfbphi2) / Tox;
                  tmp = exp(0.7 * log(T0));
                  T1 = 1.0 + tmp;
                  T2 = 0.7 * tmp / (T0 * Tox);
                  Tcen = 1.9e-9 / T1;
                  dTcen_dVg = -Tcen * T2 / T1;
                  dTcen_dVd = dTcen_dVg * dVgsteff_dVd;
                  dTcen_dVb = dTcen_dVg * dVgsteff_dVb;
                  dTcen_dVg *= dVgsteff_dVg;

		  Ccen = EPSSI / Tcen;
		  T0 = Cox / (Cox + Ccen);
		  Coxeff = T0 * Ccen;
		  T1 = -Ccen / Tcen;
		  dCoxeff_dVg = T0 * T0 * T1;
		  dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd;
		  dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
		  dCoxeff_dVg *= dTcen_dVg;
		  CoxWLcen = CoxWL * Coxeff / model->BSIM4v5coxe;

                  AbulkCV = Abulk0 * pParam->BSIM4v5abulkCVfactor;
                  dAbulkCV_dVb = pParam->BSIM4v5abulkCVfactor * dAbulk0_dVb;
                  VdsatCV = VgDP / AbulkCV;

                  T0 = VdsatCV - Vds - DELTA_4;
                  dT0_dVg = dVgDP_dVg / AbulkCV;
                  dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;
                  T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * VdsatCV);
                  dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1;
                  dT1_dVd = -T0 / T1;
                  dT1_dVb = dT1_dVg * dT0_dVb;
                  dT1_dVg *= dT0_dVg;
                  if (T0 >= 0.0)
                  {   VdseffCV = VdsatCV - 0.5 * (T0 + T1);
                      dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg);
                      dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd);
                      dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb);
                  }
                  else
                  {   T3 = (DELTA_4 + DELTA_4) / (T1 - T0);
                      T4 = 1.0 - T3;
                      T5 = VdsatCV * T3 / (T1 - T0);
                      VdseffCV = VdsatCV * T4;
                      dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg);
                      dVdseffCV_dVd = T5 * (dT1_dVd + 1.0);
                      dVdseffCV_dVb = dT0_dVb * (1.0 - T5) + T5 * dT1_dVb;
                  }

                  if (Vds == 0.0)
                  {  VdseffCV = 0.0;
                     dVdseffCV_dVg = 0.0;
                     dVdseffCV_dVb = 0.0;
                  }

                  T0 = AbulkCV * VdseffCV;
		  T1 = VgDP;
                  T2 = 12.0 * (T1 - 0.5 * T0 + 1.0e-20);
                  T3 = T0 / T2;
                  T4 = 1.0 - 12.0 * T3 * T3;
                  T5 = AbulkCV * (6.0 * T0 * (4.0 * T1 - T0) / (T2 * T2) - 0.5);
		  T6 = T5 * VdseffCV / AbulkCV;

                  qgate = CoxWLcen * (T1 - T0 * (0.5 - T3));
		  QovCox = qgate / Coxeff;
		  Cgg1 = CoxWLcen * (T4 * dVgDP_dVg 
		       + T5 * dVdseffCV_dVg);
		  Cgd1 = CoxWLcen * T5 * dVdseffCV_dVd + Cgg1 
		       * dVgsteff_dVd + QovCox * dCoxeff_dVd;
		  Cgb1 = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) 
		       + Cgg1 * dVgsteff_dVb + QovCox * dCoxeff_dVb;
		  Cgg1 = Cgg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;


                  T7 = 1.0 - AbulkCV;
                  T8 = T2 * T2;
                  T9 = 12.0 * T7 * T0 * T0 / (T8 * AbulkCV);
                  T10 = T9 * dVgDP_dVg;
                  T11 = -T7 * T5 / AbulkCV;
                  T12 = -(T9 * T1 / AbulkCV + VdseffCV * (0.5 - T0 / T2));

		  qbulk = CoxWLcen * T7 * (0.5 * VdseffCV - T0 * VdseffCV / T2);
		  QovCox = qbulk / Coxeff;
		  Cbg1 = CoxWLcen * (T10 + T11 * dVdseffCV_dVg);
		  Cbd1 = CoxWLcen * T11 * dVdseffCV_dVd + Cbg1
		       * dVgsteff_dVd + QovCox * dCoxeff_dVd; 
		  Cbb1 = CoxWLcen * (T11 * dVdseffCV_dVb + T12 * dAbulkCV_dVb)
		       + Cbg1 * dVgsteff_dVb + QovCox * dCoxeff_dVb;
		  Cbg1 = Cbg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;

                  if (model->BSIM4v5xpart > 0.5)
		  {   /* 0/100 partition */
		      qsrc = -CoxWLcen * (T1 / 2.0 + T0 / 4.0 
			   - 0.5 * T0 * T0 / T2);
		      QovCox = qsrc / Coxeff;
		      T2 += T2;
		      T3 = T2 * T2;
		      T7 = -(0.25 - 12.0 * T0 * (4.0 * T1 - T0) / T3);
		      T4 = -(0.5 + 24.0 * T0 * T0 / T3) * dVgDP_dVg;
		      T5 = T7 * AbulkCV;
		      T6 = T7 * VdseffCV;

		      Csg = CoxWLcen * (T4 + T5 * dVdseffCV_dVg);
		      Csd = CoxWLcen * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd
			  + QovCox * dCoxeff_dVd;
		      Csb = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
			  + Csg * dVgsteff_dVb + QovCox * dCoxeff_dVb;
		      Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg;
                  }
		  else if (model->BSIM4v5xpart < 0.5)
		  {   /* 40/60 partition */
		      T2 = T2 / 12.0;
		      T3 = 0.5 * CoxWLcen / (T2 * T2);
                      T4 = T1 * (2.0 * T0 * T0 / 3.0 + T1 * (T1 - 4.0 
                         * T0 / 3.0)) - 2.0 * T0 * T0 * T0 / 15.0;
		      qsrc = -T3 * T4;
		      QovCox = qsrc / Coxeff;
		      T8 = 4.0 / 3.0 * T1 * (T1 - T0) + 0.4 * T0 * T0;
		      T5 = -2.0 * qsrc / T2 - T3 * (T1 * (3.0 * T1 - 8.0 
			 * T0 / 3.0) + 2.0 * T0 * T0 / 3.0);
		      T6 = AbulkCV * (qsrc / T2 + T3 * T8);
		      T7 = T6 * VdseffCV / AbulkCV; 

		      Csg = T5 * dVgDP_dVg + T6 * dVdseffCV_dVg; 
		      Csd = Csg * dVgsteff_dVd + T6 * dVdseffCV_dVd 
			  + QovCox * dCoxeff_dVd;
		      Csb = Csg * dVgsteff_dVb + T6 * dVdseffCV_dVb 
		          + T7 * dAbulkCV_dVb + QovCox * dCoxeff_dVb; 
		      Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg;
                  }
		  else
		  {   /* 50/50 partition */
                      qsrc = -0.5 * qgate;
                      Csg = -0.5 * Cgg1;
                      Csd = -0.5 * Cgd1; 
                      Csb = -0.5 * Cgb1; 
                  }

		  qgate += Qac0 + Qsub0 - qbulk;
		  qbulk -= (Qac0 + Qsub0);
                  qdrn = -(qgate + qbulk + qsrc);

		  Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
		  Cbd = Cbd1 - dQsub0_dVd;
		  Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;

                  Cgg = Cgg1 - Cbg;
                  Cgd = Cgd1 - Cbd;
                  Cgb = Cgb1 - Cbb;

		  Cgb *= dVbseff_dVb;
		  Cbb *= dVbseff_dVb;
		  Csb *= dVbseff_dVb;

                  here->BSIM4v5cggb = Cgg;
	          here->BSIM4v5cgsb = -(Cgg + Cgd + Cgb);
	          here->BSIM4v5cgdb = Cgd;
                  here->BSIM4v5cdgb = -(Cgg + Cbg + Csg);
	          here->BSIM4v5cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
			          + Csg + Csd + Csb);
	          here->BSIM4v5cddb = -(Cgd + Cbd + Csd);
                  here->BSIM4v5cbgb = Cbg;
	          here->BSIM4v5cbsb = -(Cbg + Cbd + Cbb);
	          here->BSIM4v5cbdb = Cbd;
	      }  /* End of CTM */
          }

          here->BSIM4v5csgb = - here->BSIM4v5cggb - here->BSIM4v5cdgb - here->BSIM4v5cbgb;
          here->BSIM4v5csdb = - here->BSIM4v5cgdb - here->BSIM4v5cddb - here->BSIM4v5cbdb;
          here->BSIM4v5cssb = - here->BSIM4v5cgsb - here->BSIM4v5cdsb - here->BSIM4v5cbsb;
          here->BSIM4v5cgbb = - here->BSIM4v5cgdb - here->BSIM4v5cggb - here->BSIM4v5cgsb;
          here->BSIM4v5cdbb = - here->BSIM4v5cddb - here->BSIM4v5cdgb - here->BSIM4v5cdsb;
          here->BSIM4v5cbbb = - here->BSIM4v5cbgb - here->BSIM4v5cbdb - here->BSIM4v5cbsb;
          here->BSIM4v5csbb = - here->BSIM4v5cgbb - here->BSIM4v5cdbb - here->BSIM4v5cbbb;
          here->BSIM4v5qgate = qgate;
          here->BSIM4v5qbulk = qbulk;
          here->BSIM4v5qdrn = qdrn;
          here->BSIM4v5qsrc = -(qgate + qbulk + qdrn);

          /* NQS begins */
          if ((here->BSIM4v5trnqsMod) || (here->BSIM4v5acnqsMod))
          {   here->BSIM4v5qchqs = qcheq = -(qbulk + qgate);
              here->BSIM4v5cqgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb);
              here->BSIM4v5cqdb = -(here->BSIM4v5cgdb + here->BSIM4v5cbdb);
              here->BSIM4v5cqsb = -(here->BSIM4v5cgsb + here->BSIM4v5cbsb);
              here->BSIM4v5cqbb = -(here->BSIM4v5cqgb + here->BSIM4v5cqdb
                              + here->BSIM4v5cqsb);

              CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV * here->BSIM4v5nf
                    * pParam->BSIM4v5leffCV;
              T1 = here->BSIM4v5gcrg / CoxWL; /* 1 / tau */
              here->BSIM4v5gtau = T1 * ScalingFactor;

	      if (here->BSIM4v5acnqsMod)
                  here->BSIM4v5taunet = 1.0 / T1;

              *(ckt->CKTstate0 + here->BSIM4v5qcheq) = qcheq;
              if (ckt->CKTmode & MODEINITTRAN)
                  *(ckt->CKTstate1 + here->BSIM4v5qcheq) =
                                   *(ckt->CKTstate0 + here->BSIM4v5qcheq);
	      if (here->BSIM4v5trnqsMod)
              {   error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qcheq);
                  if (error)
                      return(error);
	      }
          }


finished: 

	  /* Calculate junction C-V */
          if (ChargeComputationNeeded)
	  {   czbd = model->BSIM4v5DunitAreaTempJctCap * here->BSIM4v5Adeff; /* bug fix */
              czbs = model->BSIM4v5SunitAreaTempJctCap * here->BSIM4v5Aseff;
              czbdsw = model->BSIM4v5DunitLengthSidewallTempJctCap * here->BSIM4v5Pdeff;
              czbdswg = model->BSIM4v5DunitLengthGateSidewallTempJctCap
                      * pParam->BSIM4v5weffCJ * here->BSIM4v5nf;
              czbssw = model->BSIM4v5SunitLengthSidewallTempJctCap * here->BSIM4v5Pseff;
              czbsswg = model->BSIM4v5SunitLengthGateSidewallTempJctCap
                      * pParam->BSIM4v5weffCJ * here->BSIM4v5nf;

              MJS = model->BSIM4v5SbulkJctBotGradingCoeff;
              MJSWS = model->BSIM4v5SbulkJctSideGradingCoeff;
	      MJSWGS = model->BSIM4v5SbulkJctGateSideGradingCoeff;

              MJD = model->BSIM4v5DbulkJctBotGradingCoeff;
              MJSWD = model->BSIM4v5DbulkJctSideGradingCoeff;
              MJSWGD = model->BSIM4v5DbulkJctGateSideGradingCoeff;

              /* Source Bulk Junction */
	      if (vbs_jct == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM4v5qbs) = 0.0;
                  here->BSIM4v5capbs = czbs + czbssw + czbsswg;
	      }
	      else if (vbs_jct < 0.0)
	      {   if (czbs > 0.0)
		  {   arg = 1.0 - vbs_jct / model->BSIM4v5PhiBS;
		      if (MJS == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJS * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbs) = model->BSIM4v5PhiBS * czbs 
			               * (1.0 - arg * sarg) / (1.0 - MJS);
		      here->BSIM4v5capbs = czbs * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM4v5qbs) = 0.0;
		      here->BSIM4v5capbs = 0.0;
		  }
		  if (czbssw > 0.0)
		  {   arg = 1.0 - vbs_jct / model->BSIM4v5PhiBSWS;
		      if (MJSWS == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWS * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbs) += model->BSIM4v5PhiBSWS * czbssw
				       * (1.0 - arg * sarg) / (1.0 - MJSWS);
                      here->BSIM4v5capbs += czbssw * sarg;
		  }
		  if (czbsswg > 0.0)
		  {   arg = 1.0 - vbs_jct / model->BSIM4v5PhiBSWGS;
		      if (MJSWGS == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWGS * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbs) += model->BSIM4v5PhiBSWGS * czbsswg
				       * (1.0 - arg * sarg) / (1.0 - MJSWGS);
                      here->BSIM4v5capbs += czbsswg * sarg;
		  }

              }
	      else
	      {   T0 = czbs + czbssw + czbsswg;
                  T1 = vbs_jct * (czbs * MJS / model->BSIM4v5PhiBS + czbssw * MJSWS 
                     / model->BSIM4v5PhiBSWS + czbsswg * MJSWGS / model->BSIM4v5PhiBSWGS);    
                  *(ckt->CKTstate0 + here->BSIM4v5qbs) = vbs_jct * (T0 + 0.5 * T1);
                  here->BSIM4v5capbs = T0 + T1;
              }

              /* Drain Bulk Junction */
	      if (vbd_jct == 0.0)
	      {   *(ckt->CKTstate0 + here->BSIM4v5qbd) = 0.0;
                  here->BSIM4v5capbd = czbd + czbdsw + czbdswg;
	      }
	      else if (vbd_jct < 0.0)
	      {   if (czbd > 0.0)
		  {   arg = 1.0 - vbd_jct / model->BSIM4v5PhiBD;
		      if (MJD == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJD * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbd) = model->BSIM4v5PhiBD* czbd 
			               * (1.0 - arg * sarg) / (1.0 - MJD);
                      here->BSIM4v5capbd = czbd * sarg;
		  }
		  else
		  {   *(ckt->CKTstate0 + here->BSIM4v5qbd) = 0.0;
                      here->BSIM4v5capbd = 0.0;
		  }
		  if (czbdsw > 0.0)
		  {   arg = 1.0 - vbd_jct / model->BSIM4v5PhiBSWD;
		      if (MJSWD == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWD * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbd) += model->BSIM4v5PhiBSWD * czbdsw 
			               * (1.0 - arg * sarg) / (1.0 - MJSWD);
                      here->BSIM4v5capbd += czbdsw * sarg;
		  }
		  if (czbdswg > 0.0)
		  {   arg = 1.0 - vbd_jct / model->BSIM4v5PhiBSWGD;
		      if (MJSWGD == 0.5)
                          sarg = 1.0 / sqrt(arg);
		      else
                          sarg = exp(-MJSWGD * log(arg));
                      *(ckt->CKTstate0 + here->BSIM4v5qbd) += model->BSIM4v5PhiBSWGD * czbdswg
				       * (1.0 - arg * sarg) / (1.0 - MJSWGD);
                      here->BSIM4v5capbd += czbdswg * sarg;
		  }
              }
	      else
	      {   T0 = czbd + czbdsw + czbdswg;
                  T1 = vbd_jct * (czbd * MJD / model->BSIM4v5PhiBD + czbdsw * MJSWD
                     / model->BSIM4v5PhiBSWD + czbdswg * MJSWGD / model->BSIM4v5PhiBSWGD);
                  *(ckt->CKTstate0 + here->BSIM4v5qbd) = vbd_jct * (T0 + 0.5 * T1);
                  here->BSIM4v5capbd = T0 + T1; 
              }
          }


          /*
           *  check convergence
           */

          if ((here->BSIM4v5off == 0) || (!(ckt->CKTmode & MODEINITFIX)))
	  {   if (Check == 1)
	      {   ckt->CKTnoncon++;
#ifndef NEWCONV
              } 
	      else
              {   if (here->BSIM4v5mode >= 0)
                  {   Idtot = here->BSIM4v5cd + here->BSIM4v5csub
			    + here->BSIM4v5Igidl - here->BSIM4v5cbd;
                  }
                  else
                  {   Idtot = here->BSIM4v5cd + here->BSIM4v5cbd - here->BSIM4v5Igidl; /* bugfix */
                  }
                  tol0 = ckt->CKTreltol * MAX(fabs(cdhat), fabs(Idtot))
                       + ckt->CKTabstol;
		  tol1 = ckt->CKTreltol * MAX(fabs(cseshat), fabs(Isestot))
                       + ckt->CKTabstol;
                  tol2 = ckt->CKTreltol * MAX(fabs(cdedhat), fabs(Idedtot))
                       + ckt->CKTabstol;
                  tol3 = ckt->CKTreltol * MAX(fabs(cgshat), fabs(Igstot))
                       + ckt->CKTabstol;
                  tol4 = ckt->CKTreltol * MAX(fabs(cgdhat), fabs(Igdtot))
                       + ckt->CKTabstol;
                  tol5 = ckt->CKTreltol * MAX(fabs(cgbhat), fabs(Igbtot))
                       + ckt->CKTabstol;
                  if ((fabs(cdhat - Idtot) >= tol0) || (fabs(cseshat - Isestot) >= tol1)
		      || (fabs(cdedhat - Idedtot) >= tol2))
                  {   ckt->CKTnoncon++;
                  }
		  else if ((fabs(cgshat - Igstot) >= tol3) || (fabs(cgdhat - Igdtot) >= tol4)
		      || (fabs(cgbhat - Igbtot) >= tol5))
                  {   ckt->CKTnoncon++;
                  }
                  else
                  {   Ibtot = here->BSIM4v5cbs + here->BSIM4v5cbd
			    - here->BSIM4v5Igidl - here->BSIM4v5Igisl - here->BSIM4v5csub;
                      tol6 = ckt->CKTreltol * MAX(fabs(cbhat), fabs(Ibtot))
                          + ckt->CKTabstol;
                      if (fabs(cbhat - Ibtot) > tol6)
		      {   ckt->CKTnoncon++;
                      }
                  }
#endif /* NEWCONV */
              }
          }
          *(ckt->CKTstate0 + here->BSIM4v5vds) = vds;
          *(ckt->CKTstate0 + here->BSIM4v5vgs) = vgs;
          *(ckt->CKTstate0 + here->BSIM4v5vbs) = vbs;
          *(ckt->CKTstate0 + here->BSIM4v5vbd) = vbd;
	  *(ckt->CKTstate0 + here->BSIM4v5vges) = vges;
	  *(ckt->CKTstate0 + here->BSIM4v5vgms) = vgms;
          *(ckt->CKTstate0 + here->BSIM4v5vdbs) = vdbs;
          *(ckt->CKTstate0 + here->BSIM4v5vdbd) = vdbd;
          *(ckt->CKTstate0 + here->BSIM4v5vsbs) = vsbs;
          *(ckt->CKTstate0 + here->BSIM4v5vses) = vses;
          *(ckt->CKTstate0 + here->BSIM4v5vdes) = vdes;
          *(ckt->CKTstate0 + here->BSIM4v5qdef) = qdef;


          if (!ChargeComputationNeeded)
              goto line850; 

	  if (here->BSIM4v5rgateMod == 3) 
	  {   
	          vgdx = vgmd; 
	          vgsx = vgms;
	  }  
	  else  /* For rgateMod == 0, 1 and 2 */
	  {
	          vgdx = vgd;
	          vgsx = vgs;
	  }
	  if (model->BSIM4v5capMod == 0) 
	  {  
	          cgdo = pParam->BSIM4v5cgdo; 
	          qgdo = pParam->BSIM4v5cgdo * vgdx;
	          cgso = pParam->BSIM4v5cgso;
	          qgso = pParam->BSIM4v5cgso * vgsx;
	  }
	  else /* For both capMod == 1 and 2 */
	  {   T0 = vgdx + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);

	      T3 = pParam->BSIM4v5weffCV * pParam->BSIM4v5cgdl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM4v5ckappad); 
	      cgdo = pParam->BSIM4v5cgdo + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgdo = (pParam->BSIM4v5cgdo + T3) * vgdx - T3 * (T2
		   + 0.5 * pParam->BSIM4v5ckappad * (T4 - 1.0));

	      T0 = vgsx + DELTA_1;
	      T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
	      T2 = 0.5 * (T0 - T1);
	      T3 = pParam->BSIM4v5weffCV * pParam->BSIM4v5cgsl;
	      T4 = sqrt(1.0 - 4.0 * T2 / pParam->BSIM4v5ckappas);
	      cgso = pParam->BSIM4v5cgso + T3 - T3 * (1.0 - 1.0 / T4)
		   * (0.5 - 0.5 * T0 / T1);
	      qgso = (pParam->BSIM4v5cgso + T3) * vgsx - T3 * (T2
		   + 0.5 * pParam->BSIM4v5ckappas * (T4 - 1.0));
	  }

	  if (here->BSIM4v5nf != 1.0)
	  {   cgdo *= here->BSIM4v5nf;
	      cgso *= here->BSIM4v5nf;
              qgdo *= here->BSIM4v5nf;
              qgso *= here->BSIM4v5nf;
	  }	
          here->BSIM4v5cgdo = cgdo;
          here->BSIM4v5qgdo = qgdo;
          here->BSIM4v5cgso = cgso;
          here->BSIM4v5qgso = qgso;

#ifndef NOBYPASS
line755:
#endif
          ag0 = ckt->CKTag[0];
          if (here->BSIM4v5mode > 0)
          {   if (here->BSIM4v5trnqsMod == 0)
              {   qdrn -= qgdo;
		  if (here->BSIM4v5rgateMod == 3)
		  {   gcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * ag0;
		      gcgmdb = -cgdo * ag0;
                      gcgmsb = -cgso * ag0;
                      gcgmbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgmb = gcgmdb;
                      gcsgmb = gcgmsb;
                      gcbgmb = gcgmbb;

		      gcggb = here->BSIM4v5cggb * ag0;
                      gcgdb = here->BSIM4v5cgdb * ag0;
                      gcgsb = here->BSIM4v5cgsb * ag0;   
                      gcgbb = -(gcggb + gcgdb + gcgsb);

		      gcdgb = here->BSIM4v5cdgb * ag0;
		      gcsgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                            + here->BSIM4v5cdgb) * ag0;
                      gcbgb = here->BSIM4v5cbgb * ag0;

                      qgmb = pParam->BSIM4v5cgbo * vgmb;
                      qgmid = qgdo + qgso + qgmb;
                      qbulk -= qgmb;
                      qsrc = -(qgate + qgmid + qbulk + qdrn);
		  }
	  	  else
		  {   gcggb = (here->BSIM4v5cggb + cgdo + cgso
                            + pParam->BSIM4v5cgbo ) * ag0;
                      gcgdb = (here->BSIM4v5cgdb - cgdo) * ag0;
                      gcgsb = (here->BSIM4v5cgsb - cgso) * ag0;
                      gcgbb = -(gcggb + gcgdb + gcgsb);

		      gcdgb = (here->BSIM4v5cdgb - cgdo) * ag0;
                      gcsgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                            + here->BSIM4v5cdgb + cgso) * ag0;
                      gcbgb = (here->BSIM4v5cbgb - pParam->BSIM4v5cgbo) * ag0;

		      gcdgmb = gcsgmb = gcbgmb = 0.0;

                      qgb = pParam->BSIM4v5cgbo * vgb;
                      qgate += qgdo + qgso + qgb;
                      qbulk -= qgb;
                      qsrc = -(qgate + qbulk + qdrn);
		  }
                  gcddb = (here->BSIM4v5cddb + here->BSIM4v5capbd + cgdo) * ag0;
                  gcdsb = here->BSIM4v5cdsb * ag0;

                  gcsdb = -(here->BSIM4v5cgdb + here->BSIM4v5cbdb
                        + here->BSIM4v5cddb) * ag0;
                  gcssb = (here->BSIM4v5capbs + cgso - (here->BSIM4v5cgsb
                        + here->BSIM4v5cbsb + here->BSIM4v5cdsb)) * ag0;

                  if (!here->BSIM4v5rbodyMod)
                  {   gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                      gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                      gcbdb = (here->BSIM4v5cbdb - here->BSIM4v5capbd) * ag0;
                      gcbsb = (here->BSIM4v5cbsb - here->BSIM4v5capbs) * ag0;
                      gcdbdb = 0.0; gcsbsb = 0.0;
                  }
                  else
                  {   gcdbb  = -(here->BSIM4v5cddb + here->BSIM4v5cdgb 
                             + here->BSIM4v5cdsb) * ag0;
                      gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb)
			    + here->BSIM4v5capbs * ag0;
                      gcbdb = here->BSIM4v5cbdb * ag0;
                      gcbsb = here->BSIM4v5cbsb * ag0;

                      gcdbdb = -here->BSIM4v5capbd * ag0;
                      gcsbsb = -here->BSIM4v5capbs * ag0;
                  }
		  gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);

                  ggtg = ggtd = ggtb = ggts = 0.0;
		  sxpart = 0.6;
                  dxpart = 0.4;
		  ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0;
		  dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0;
              }
              else
              {   qcheq = here->BSIM4v5qchqs;
                  CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV * here->BSIM4v5nf
                        * pParam->BSIM4v5leffCV;
                  T0 = qdef * ScalingFactor / CoxWL;

                  ggtg = here->BSIM4v5gtg = T0 * here->BSIM4v5gcrgg;
                  ggtd = here->BSIM4v5gtd = T0 * here->BSIM4v5gcrgd;
                  ggts = here->BSIM4v5gts = T0 * here->BSIM4v5gcrgs;
                  ggtb = here->BSIM4v5gtb = T0 * here->BSIM4v5gcrgb;
		  gqdef = ScalingFactor * ag0;

                  gcqgb = here->BSIM4v5cqgb * ag0;
                  gcqdb = here->BSIM4v5cqdb * ag0;
                  gcqsb = here->BSIM4v5cqsb * ag0;
                  gcqbb = here->BSIM4v5cqbb * ag0;

                  if (fabs(qcheq) <= 1.0e-5 * CoxWL)
                  {   if (model->BSIM4v5xpart < 0.5)
                      {   dxpart = 0.4;
                      }
                      else if (model->BSIM4v5xpart > 0.5)
                      {   dxpart = 0.0;
                      }
                      else
                      {   dxpart = 0.5;
                      }
                      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
                                  = ddxpart_dVs = 0.0;
                  }
                  else
                  {   dxpart = qdrn / qcheq;
                      Cdd = here->BSIM4v5cddb;
                      Csd = -(here->BSIM4v5cgdb + here->BSIM4v5cddb
                          + here->BSIM4v5cbdb);
                      ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
                      Cdg = here->BSIM4v5cdgb;
                      Csg = -(here->BSIM4v5cggb + here->BSIM4v5cdgb
                          + here->BSIM4v5cbgb);
                      ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

                      Cds = here->BSIM4v5cdsb;
                      Css = -(here->BSIM4v5cgsb + here->BSIM4v5cdsb
                          + here->BSIM4v5cbsb);
                      ddxpart_dVs = (Cds - dxpart * (Cds + Css)) / qcheq;

                      ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                  }
                  sxpart = 1.0 - dxpart;
                  dsxpart_dVd = -ddxpart_dVd;
                  dsxpart_dVg = -ddxpart_dVg;
                  dsxpart_dVs = -ddxpart_dVs;
                  dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);

                  if (here->BSIM4v5rgateMod == 3)
                  {   gcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * ag0;
                      gcgmdb = -cgdo * ag0;
                      gcgmsb = -cgso * ag0;
                      gcgmbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgmb = gcgmdb;
                      gcsgmb = gcgmsb;
                      gcbgmb = gcgmbb;

                      gcdgb = gcsgb = gcbgb = 0.0;
		      gcggb = gcgdb = gcgsb = gcgbb = 0.0;

                      qgmb = pParam->BSIM4v5cgbo * vgmb;
                      qgmid = qgdo + qgso + qgmb;
		      qgate = 0.0;
                      qbulk = -qgmb;
		      qdrn = -qgdo;
                      qsrc = -(qgmid + qbulk + qdrn);
                  }
                  else
                  {   gcggb = (cgdo + cgso + pParam->BSIM4v5cgbo ) * ag0;
                      gcgdb = -cgdo * ag0;
                      gcgsb = -cgso * ag0;
		      gcgbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgb = gcgdb;
                      gcsgb = gcgsb;
                      gcbgb = gcgbb;
                      gcdgmb = gcsgmb = gcbgmb = 0.0;

                      qgb = pParam->BSIM4v5cgbo * vgb;
                      qgate = qgdo + qgso + qgb;
                      qbulk = -qgb;
		      qdrn = -qgdo;
                      qsrc = -(qgate + qbulk + qdrn);
                  }

                  gcddb = (here->BSIM4v5capbd + cgdo) * ag0;
                  gcdsb = gcsdb = 0.0;
                  gcssb = (here->BSIM4v5capbs + cgso) * ag0;

                  if (!here->BSIM4v5rbodyMod)
                  {   gcdbb = -(gcdgb + gcddb + gcdgmb);
                      gcsbb = -(gcsgb + gcssb + gcsgmb);
                      gcbdb = -here->BSIM4v5capbd * ag0;
                      gcbsb = -here->BSIM4v5capbs * ag0;
                      gcdbdb = 0.0; gcsbsb = 0.0;
                  }
                  else
                  {   gcdbb = gcsbb = gcbdb = gcbsb = 0.0;
                      gcdbdb = -here->BSIM4v5capbd * ag0;
                      gcsbsb = -here->BSIM4v5capbs * ag0;
                  }
                  gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
              }
          }
          else
          {   if (here->BSIM4v5trnqsMod == 0)
              {   qsrc = qdrn - qgso;
		  if (here->BSIM4v5rgateMod == 3)
		  {   gcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * ag0;
		      gcgmdb = -cgdo * ag0;
    		      gcgmsb = -cgso * ag0;
    		      gcgmbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgmb = gcgmdb;
                      gcsgmb = gcgmsb;
                      gcbgmb = gcgmbb;

                      gcggb = here->BSIM4v5cggb * ag0;
                      gcgdb = here->BSIM4v5cgsb * ag0;
                      gcgsb = here->BSIM4v5cgdb * ag0;
                      gcgbb = -(gcggb + gcgdb + gcgsb);

                      gcdgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                            + here->BSIM4v5cdgb) * ag0;
                      gcsgb = here->BSIM4v5cdgb * ag0;
                      gcbgb = here->BSIM4v5cbgb * ag0;

                      qgmb = pParam->BSIM4v5cgbo * vgmb;
                      qgmid = qgdo + qgso + qgmb;
                      qbulk -= qgmb;
                      qdrn = -(qgate + qgmid + qbulk + qsrc);
		  }
		  else
		  {   gcggb = (here->BSIM4v5cggb + cgdo + cgso
                            + pParam->BSIM4v5cgbo ) * ag0;
                      gcgdb = (here->BSIM4v5cgsb - cgdo) * ag0;
                      gcgsb = (here->BSIM4v5cgdb - cgso) * ag0;
                      gcgbb = -(gcggb + gcgdb + gcgsb);

                      gcdgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                            + here->BSIM4v5cdgb + cgdo) * ag0;
                      gcsgb = (here->BSIM4v5cdgb - cgso) * ag0;
                      gcbgb = (here->BSIM4v5cbgb - pParam->BSIM4v5cgbo) * ag0;

                      gcdgmb = gcsgmb = gcbgmb = 0.0;

                      qgb = pParam->BSIM4v5cgbo * vgb;
                      qgate += qgdo + qgso + qgb;
                      qbulk -= qgb;
                      qdrn = -(qgate + qbulk + qsrc);
 		  }
                  gcddb = (here->BSIM4v5capbd + cgdo - (here->BSIM4v5cgsb
                        + here->BSIM4v5cbsb + here->BSIM4v5cdsb)) * ag0;
                  gcdsb = -(here->BSIM4v5cgdb + here->BSIM4v5cbdb
                        + here->BSIM4v5cddb) * ag0;

                  gcsdb = here->BSIM4v5cdsb * ag0;
                  gcssb = (here->BSIM4v5cddb + here->BSIM4v5capbs + cgso) * ag0;

		  if (!here->BSIM4v5rbodyMod)
		  {   gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                      gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                      gcbdb = (here->BSIM4v5cbsb - here->BSIM4v5capbd) * ag0;
                      gcbsb = (here->BSIM4v5cbdb - here->BSIM4v5capbs) * ag0;
                      gcdbdb = 0.0; gcsbsb = 0.0;
                  }
                  else
                  {   gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb)
			    + here->BSIM4v5capbd * ag0;
                      gcsbb = -(here->BSIM4v5cddb + here->BSIM4v5cdgb
                            + here->BSIM4v5cdsb) * ag0;
                      gcbdb = here->BSIM4v5cbsb * ag0;
                      gcbsb = here->BSIM4v5cbdb * ag0;
                      gcdbdb = -here->BSIM4v5capbd * ag0;
		      gcsbsb = -here->BSIM4v5capbs * ag0;
                  }
		  gcbbb = -(gcbgb + gcbdb + gcbsb + gcbgmb);

                  ggtg = ggtd = ggtb = ggts = 0.0;
		  sxpart = 0.4;
                  dxpart = 0.6;
		  ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0;
		  dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0;
              }
              else
              {   qcheq = here->BSIM4v5qchqs;
                  CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV * here->BSIM4v5nf
                        * pParam->BSIM4v5leffCV;
                  T0 = qdef * ScalingFactor / CoxWL;
                  ggtg = here->BSIM4v5gtg = T0 * here->BSIM4v5gcrgg;
                  ggts = here->BSIM4v5gts = T0 * here->BSIM4v5gcrgd;
                  ggtd = here->BSIM4v5gtd = T0 * here->BSIM4v5gcrgs;
                  ggtb = here->BSIM4v5gtb = T0 * here->BSIM4v5gcrgb;
		  gqdef = ScalingFactor * ag0;

                  gcqgb = here->BSIM4v5cqgb * ag0;
                  gcqdb = here->BSIM4v5cqsb * ag0;
                  gcqsb = here->BSIM4v5cqdb * ag0;
                  gcqbb = here->BSIM4v5cqbb * ag0;

                  if (fabs(qcheq) <= 1.0e-5 * CoxWL)
                  {   if (model->BSIM4v5xpart < 0.5)
                      {   sxpart = 0.4;
                      }
                      else if (model->BSIM4v5xpart > 0.5)
                      {   sxpart = 0.0;
                      }
                      else
                      {   sxpart = 0.5;
                      }
                      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
                                  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   sxpart = qdrn / qcheq;
                      Css = here->BSIM4v5cddb;
                      Cds = -(here->BSIM4v5cgdb + here->BSIM4v5cddb
                          + here->BSIM4v5cbdb);
                      dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
                      Csg = here->BSIM4v5cdgb;
                      Cdg = -(here->BSIM4v5cggb + here->BSIM4v5cdgb
                          + here->BSIM4v5cbgb);
                      dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

                      Csd = here->BSIM4v5cdsb;
                      Cdd = -(here->BSIM4v5cgsb + here->BSIM4v5cdsb
                          + here->BSIM4v5cbsb);
                      dsxpart_dVd = (Csd - sxpart * (Csd + Cdd)) / qcheq;

                      dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                  }
                  dxpart = 1.0 - sxpart;
                  ddxpart_dVd = -dsxpart_dVd;
                  ddxpart_dVg = -dsxpart_dVg;
                  ddxpart_dVs = -dsxpart_dVs;
                  ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);

                  if (here->BSIM4v5rgateMod == 3)
                  {   gcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * ag0;
                      gcgmdb = -cgdo * ag0;
                      gcgmsb = -cgso * ag0;
                      gcgmbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgmb = gcgmdb;
                      gcsgmb = gcgmsb;
                      gcbgmb = gcgmbb;

                      gcdgb = gcsgb = gcbgb = 0.0;
                      gcggb = gcgdb = gcgsb = gcgbb = 0.0;

                      qgmb = pParam->BSIM4v5cgbo * vgmb;
                      qgmid = qgdo + qgso + qgmb;
                      qgate = 0.0;
                      qbulk = -qgmb;
                      qdrn = -qgdo;
                      qsrc = -qgso;
                  }
                  else
                  {   gcggb = (cgdo + cgso + pParam->BSIM4v5cgbo ) * ag0;
                      gcgdb = -cgdo * ag0;
                      gcgsb = -cgso * ag0;
                      gcgbb = -pParam->BSIM4v5cgbo * ag0;

                      gcdgb = gcgdb;
                      gcsgb = gcgsb;
                      gcbgb = gcgbb;
                      gcdgmb = gcsgmb = gcbgmb = 0.0;

                      qgb = pParam->BSIM4v5cgbo * vgb;
                      qgate = qgdo + qgso + qgb;
                      qbulk = -qgb;
                      qdrn = -qgdo;
                      qsrc = -qgso;
                  }

                  gcddb = (here->BSIM4v5capbd + cgdo) * ag0;
                  gcdsb = gcsdb = 0.0;
                  gcssb = (here->BSIM4v5capbs + cgso) * ag0;
                  if (!here->BSIM4v5rbodyMod)
                  {   gcdbb = -(gcdgb + gcddb + gcdgmb);
                      gcsbb = -(gcsgb + gcssb + gcsgmb);
                      gcbdb = -here->BSIM4v5capbd * ag0;
                      gcbsb = -here->BSIM4v5capbs * ag0;
                      gcdbdb = 0.0; gcsbsb = 0.0;
                  }
                  else
                  {   gcdbb = gcsbb = gcbdb = gcbsb = 0.0;
                      gcdbdb = -here->BSIM4v5capbd * ag0;
                      gcsbsb = -here->BSIM4v5capbs * ag0;
                  }
                  gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
              }
          }


          if (here->BSIM4v5trnqsMod)
          {   *(ckt->CKTstate0 + here->BSIM4v5qcdump) = qdef * ScalingFactor;
              if (ckt->CKTmode & MODEINITTRAN)
                  *(ckt->CKTstate1 + here->BSIM4v5qcdump) =
                                   *(ckt->CKTstate0 + here->BSIM4v5qcdump);
              error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qcdump);
              if (error)
                  return(error);
          }

          if (ByPass) goto line860;

          *(ckt->CKTstate0 + here->BSIM4v5qg) = qgate;
          *(ckt->CKTstate0 + here->BSIM4v5qd) = qdrn
                           - *(ckt->CKTstate0 + here->BSIM4v5qbd);
          *(ckt->CKTstate0 + here->BSIM4v5qs) = qsrc
                           - *(ckt->CKTstate0 + here->BSIM4v5qbs);
	  if (here->BSIM4v5rgateMod == 3)
	      *(ckt->CKTstate0 + here->BSIM4v5qgmid) = qgmid;

          if (!here->BSIM4v5rbodyMod)
          {   *(ckt->CKTstate0 + here->BSIM4v5qb) = qbulk
                               + *(ckt->CKTstate0 + here->BSIM4v5qbd)
                               + *(ckt->CKTstate0 + here->BSIM4v5qbs);
          }
          else
              *(ckt->CKTstate0 + here->BSIM4v5qb) = qbulk;


          /* Store small signal parameters */
          if (ckt->CKTmode & MODEINITSMSIG)
          {   goto line1000;
          }

          if (!ChargeComputationNeeded)
              goto line850;

          /* no integration, if dc sweep, but keep evaluating capacitances */
          if (ckt->CKTmode & MODEDCTRANCURVE)
              goto line850;

          if (ckt->CKTmode & MODEINITTRAN)
          {   *(ckt->CKTstate1 + here->BSIM4v5qb) =
                    *(ckt->CKTstate0 + here->BSIM4v5qb);
              *(ckt->CKTstate1 + here->BSIM4v5qg) =
                    *(ckt->CKTstate0 + here->BSIM4v5qg);
              *(ckt->CKTstate1 + here->BSIM4v5qd) =
                    *(ckt->CKTstate0 + here->BSIM4v5qd);
              if (here->BSIM4v5rgateMod == 3)
                  *(ckt->CKTstate1 + here->BSIM4v5qgmid) =
                        *(ckt->CKTstate0 + here->BSIM4v5qgmid);
              if (here->BSIM4v5rbodyMod)
              {   *(ckt->CKTstate1 + here->BSIM4v5qbs) =
                                   *(ckt->CKTstate0 + here->BSIM4v5qbs);
                  *(ckt->CKTstate1 + here->BSIM4v5qbd) =
                                   *(ckt->CKTstate0 + here->BSIM4v5qbd);
              }
          }

          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qb);
          if (error) 
              return(error);
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qg);
          if (error) 
              return(error);
          error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qd);
          if (error) 
              return(error);

          if (here->BSIM4v5rgateMod == 3)
          {   error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qgmid);
              if (error) return(error);
          }

          if (here->BSIM4v5rbodyMod)
          {   error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qbs);
              if (error) 
                  return(error);
              error = NIintegrate(ckt, &geq, &ceq, 0.0, here->BSIM4v5qbd);
              if (error) 
                  return(error);
          }

          goto line860;


line850:
          /* Zero gcap and ceqcap if (!ChargeComputationNeeded) */
          ceqqg = ceqqb = ceqqd = 0.0;
          ceqqjd = ceqqjs = 0.0;
          cqcheq = cqdef = 0.0;

          gcdgb = gcddb = gcdsb = gcdbb = 0.0;
          gcsgb = gcsdb = gcssb = gcsbb = 0.0;
          gcggb = gcgdb = gcgsb = gcgbb = 0.0;
          gcbdb = gcbgb = gcbsb = gcbbb = 0.0;

	  gcgmgmb = gcgmdb = gcgmsb = gcgmbb = 0.0;
	  gcdgmb = gcsgmb = gcbgmb = ceqqgmid = 0.0;
          gcdbdb = gcsbsb = 0.0;

	  gqdef = gcqgb = gcqdb = gcqsb = gcqbb = 0.0;
          ggtg = ggtd = ggtb = ggts = 0.0;
          sxpart = (1.0 - (dxpart = (here->BSIM4v5mode > 0) ? 0.4 : 0.6));
	  ddxpart_dVd = ddxpart_dVg = ddxpart_dVb = ddxpart_dVs = 0.0;
	  dsxpart_dVd = dsxpart_dVg = dsxpart_dVb = dsxpart_dVs = 0.0;

          if (here->BSIM4v5trnqsMod)
          {   CoxWL = model->BSIM4v5coxe * pParam->BSIM4v5weffCV * here->BSIM4v5nf
                    * pParam->BSIM4v5leffCV;
              T1 = here->BSIM4v5gcrg / CoxWL;
              here->BSIM4v5gtau = T1 * ScalingFactor;
          }
	  else
              here->BSIM4v5gtau = 0.0;

          goto line900;

            
line860:
          /* Calculate equivalent charge current */

          cqgate = *(ckt->CKTstate0 + here->BSIM4v5cqg);
          cqbody = *(ckt->CKTstate0 + here->BSIM4v5cqb);
          cqdrn = *(ckt->CKTstate0 + here->BSIM4v5cqd);

          ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
          ceqqd = cqdrn - gcdgb * vgb - gcdgmb * vgmb + (gcddb + gcdbdb)
	        * vbd - gcdbdb * vbd_jct + gcdsb * vbs;
	  ceqqb = cqbody - gcbgb * vgb - gcbgmb * vgmb
                + gcbdb * vbd + gcbsb * vbs;


          if (here->BSIM4v5rgateMod == 3)
              ceqqgmid = *(ckt->CKTstate0 + here->BSIM4v5cqgmid)
                       + gcgmdb * vbd + gcgmsb * vbs - gcgmgmb * vgmb;
	  else
 	      ceqqgmid = 0.0;	

          if (here->BSIM4v5rbodyMod)
          {   ceqqjs = *(ckt->CKTstate0 + here->BSIM4v5cqbs) + gcsbsb * vbs_jct;
              ceqqjd = *(ckt->CKTstate0 + here->BSIM4v5cqbd) + gcdbdb * vbd_jct; 
          }

          if (here->BSIM4v5trnqsMod)
          {   T0 = ggtg * vgb - ggtd * vbd - ggts * vbs;
              ceqqg += T0;
	      T1 = qdef * here->BSIM4v5gtau;
              ceqqd -= dxpart * T0 + T1 * (ddxpart_dVg * vgb - ddxpart_dVd
		     * vbd - ddxpart_dVs * vbs);
              cqdef = *(ckt->CKTstate0 + here->BSIM4v5cqcdump) - gqdef * qdef;
              cqcheq = *(ckt->CKTstate0 + here->BSIM4v5cqcheq)
                     - (gcqgb * vgb - gcqdb * vbd - gcqsb * vbs) + T0;
          }

          if (ckt->CKTmode & MODEINITTRAN)
          {   *(ckt->CKTstate1 + here->BSIM4v5cqb) =
                               *(ckt->CKTstate0 + here->BSIM4v5cqb);
              *(ckt->CKTstate1 + here->BSIM4v5cqg) =
                               *(ckt->CKTstate0 + here->BSIM4v5cqg);
              *(ckt->CKTstate1 + here->BSIM4v5cqd) =
                               *(ckt->CKTstate0 + here->BSIM4v5cqd);

              if (here->BSIM4v5rgateMod == 3)
                  *(ckt->CKTstate1 + here->BSIM4v5cqgmid) =
                                   *(ckt->CKTstate0 + here->BSIM4v5cqgmid);

              if (here->BSIM4v5rbodyMod)
              {   *(ckt->CKTstate1 + here->BSIM4v5cqbs) =
                                   *(ckt->CKTstate0 + here->BSIM4v5cqbs);
                  *(ckt->CKTstate1 + here->BSIM4v5cqbd) =
                                   *(ckt->CKTstate0 + here->BSIM4v5cqbd);
              }
          }


          /*
           *  Load current vector
           */

line900:
          if (here->BSIM4v5mode >= 0)
	  {   Gm = here->BSIM4v5gm;
              Gmbs = here->BSIM4v5gmbs;
              FwdSum = Gm + Gmbs;
              RevSum = 0.0;

              ceqdrn = model->BSIM4v5type * (cdrain - here->BSIM4v5gds * vds
	             - Gm * vgs - Gmbs * vbs);
              ceqbd = model->BSIM4v5type * (here->BSIM4v5csub + here->BSIM4v5Igidl
                    - (here->BSIM4v5gbds + here->BSIM4v5ggidld) * vds
		    - (here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * vgs
		    - (here->BSIM4v5gbbs + here->BSIM4v5ggidlb) * vbs);
              ceqbs = model->BSIM4v5type * (here->BSIM4v5Igisl + here->BSIM4v5ggisls * vds 
              	    - here->BSIM4v5ggislg * vgd - here->BSIM4v5ggislb * vbd);

              gbbdp = -(here->BSIM4v5gbds);
              gbbsp = here->BSIM4v5gbds + here->BSIM4v5gbgs + here->BSIM4v5gbbs; 
		    
              gbdpg = here->BSIM4v5gbgs;
              gbdpdp = here->BSIM4v5gbds;
              gbdpb = here->BSIM4v5gbbs;
              gbdpsp = -(gbdpg + gbdpdp + gbdpb);

              gbspg = 0.0;
              gbspdp = 0.0;
              gbspb = 0.0;
              gbspsp = 0.0;

              if (model->BSIM4v5igcMod)
	      {   gIstotg = here->BSIM4v5gIgsg + here->BSIM4v5gIgcsg;
		  gIstotd = here->BSIM4v5gIgcsd;
                  gIstots = here->BSIM4v5gIgss + here->BSIM4v5gIgcss;
                  gIstotb = here->BSIM4v5gIgcsb;
 	          Istoteq = model->BSIM4v5type * (here->BSIM4v5Igs + here->BSIM4v5Igcs
		  	  - gIstotg * vgs - here->BSIM4v5gIgcsd * vds
			  - here->BSIM4v5gIgcsb * vbs);

                  gIdtotg = here->BSIM4v5gIgdg + here->BSIM4v5gIgcdg;
                  gIdtotd = here->BSIM4v5gIgdd + here->BSIM4v5gIgcdd;
                  gIdtots = here->BSIM4v5gIgcds;
                  gIdtotb = here->BSIM4v5gIgcdb;
                  Idtoteq = model->BSIM4v5type * (here->BSIM4v5Igd + here->BSIM4v5Igcd
                          - here->BSIM4v5gIgdg * vgd - here->BSIM4v5gIgcdg * vgs
			  - here->BSIM4v5gIgcdd * vds - here->BSIM4v5gIgcdb * vbs);
	      }
	      else
	      {   gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0;
		  gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0;
	      }

              if (model->BSIM4v5igbMod)
              {   gIbtotg = here->BSIM4v5gIgbg;
                  gIbtotd = here->BSIM4v5gIgbd;
                  gIbtots = here->BSIM4v5gIgbs;
                  gIbtotb = here->BSIM4v5gIgbb;
                  Ibtoteq = model->BSIM4v5type * (here->BSIM4v5Igb
                          - here->BSIM4v5gIgbg * vgs - here->BSIM4v5gIgbd * vds
                          - here->BSIM4v5gIgbb * vbs);
              }
              else
                  gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0;

              if ((model->BSIM4v5igcMod != 0) || (model->BSIM4v5igbMod != 0))
              {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                  gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                  gIgtots = gIstots + gIdtots + gIbtots;
                  gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  Igtoteq = Istoteq + Idtoteq + Ibtoteq; 
	      }
	      else
	          gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0;


              if (here->BSIM4v5rgateMod == 2)
                  T0 = vges - vgs;
              else if (here->BSIM4v5rgateMod == 3)
                  T0 = vgms - vgs;
              if (here->BSIM4v5rgateMod > 1)
              {   gcrgd = here->BSIM4v5gcrgd * T0;
                  gcrgg = here->BSIM4v5gcrgg * T0;
                  gcrgs = here->BSIM4v5gcrgs * T0;
                  gcrgb = here->BSIM4v5gcrgb * T0;
                  ceqgcrg = -(gcrgd * vds + gcrgg * vgs
                          + gcrgb * vbs);
                  gcrgg -= here->BSIM4v5gcrg;
                  gcrg = here->BSIM4v5gcrg;
              }
              else
	          ceqgcrg = gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;
          }
	  else
	  {   Gm = -here->BSIM4v5gm;
              Gmbs = -here->BSIM4v5gmbs;
              FwdSum = 0.0;
              RevSum = -(Gm + Gmbs);

              ceqdrn = -model->BSIM4v5type * (cdrain + here->BSIM4v5gds * vds
                     + Gm * vgd + Gmbs * vbd);

              ceqbs = model->BSIM4v5type * (here->BSIM4v5csub + here->BSIM4v5Igisl 
                    + (here->BSIM4v5gbds + here->BSIM4v5ggisls) * vds
		    - (here->BSIM4v5gbgs + here->BSIM4v5ggislg) * vgd
                    - (here->BSIM4v5gbbs + here->BSIM4v5ggislb) * vbd);
              ceqbd = model->BSIM4v5type * (here->BSIM4v5Igidl - here->BSIM4v5ggidld * vds 
              		- here->BSIM4v5ggidlg * vgs - here->BSIM4v5ggidlb * vbs);

              gbbsp = -(here->BSIM4v5gbds);
              gbbdp = here->BSIM4v5gbds + here->BSIM4v5gbgs + here->BSIM4v5gbbs; 

              gbdpg = 0.0;
              gbdpsp = 0.0;
              gbdpb = 0.0;
              gbdpdp = 0.0;

              gbspg = here->BSIM4v5gbgs;
              gbspsp = here->BSIM4v5gbds;
              gbspb = here->BSIM4v5gbbs;
              gbspdp = -(gbspg + gbspsp + gbspb);

              if (model->BSIM4v5igcMod)
              {   gIstotg = here->BSIM4v5gIgsg + here->BSIM4v5gIgcdg;
                  gIstotd = here->BSIM4v5gIgcds;
                  gIstots = here->BSIM4v5gIgss + here->BSIM4v5gIgcdd;
                  gIstotb = here->BSIM4v5gIgcdb;
                  Istoteq = model->BSIM4v5type * (here->BSIM4v5Igs + here->BSIM4v5Igcd
                          - here->BSIM4v5gIgsg * vgs - here->BSIM4v5gIgcdg * vgd
			  + here->BSIM4v5gIgcdd * vds - here->BSIM4v5gIgcdb * vbd);

                  gIdtotg = here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg;
                  gIdtotd = here->BSIM4v5gIgdd + here->BSIM4v5gIgcss;
                  gIdtots = here->BSIM4v5gIgcsd;
                  gIdtotb = here->BSIM4v5gIgcsb;
                  Idtoteq = model->BSIM4v5type * (here->BSIM4v5Igd + here->BSIM4v5Igcs
                          - (here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg) * vgd
                          + here->BSIM4v5gIgcsd * vds - here->BSIM4v5gIgcsb * vbd);
              }
              else
              {   gIstotg = gIstotd = gIstots = gIstotb = Istoteq = 0.0;
                  gIdtotg = gIdtotd = gIdtots = gIdtotb = Idtoteq = 0.0;
              }

              if (model->BSIM4v5igbMod)
              {   gIbtotg = here->BSIM4v5gIgbg;
                  gIbtotd = here->BSIM4v5gIgbs;
                  gIbtots = here->BSIM4v5gIgbd;
                  gIbtotb = here->BSIM4v5gIgbb;
                  Ibtoteq = model->BSIM4v5type * (here->BSIM4v5Igb
                          - here->BSIM4v5gIgbg * vgd + here->BSIM4v5gIgbd * vds
                          - here->BSIM4v5gIgbb * vbd);
              }
              else
                  gIbtotg = gIbtotd = gIbtots = gIbtotb = Ibtoteq = 0.0;

              if ((model->BSIM4v5igcMod != 0) || (model->BSIM4v5igbMod != 0))
              {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                  gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                  gIgtots = gIstots + gIdtots + gIbtots;
                  gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  Igtoteq = Istoteq + Idtoteq + Ibtoteq;
              }
              else
                  gIgtotg = gIgtotd = gIgtots = gIgtotb = Igtoteq = 0.0;


              if (here->BSIM4v5rgateMod == 2)
                  T0 = vges - vgs;
              else if (here->BSIM4v5rgateMod == 3)
                  T0 = vgms - vgs;
              if (here->BSIM4v5rgateMod > 1)
              {   gcrgd = here->BSIM4v5gcrgs * T0;
                  gcrgg = here->BSIM4v5gcrgg * T0;
                  gcrgs = here->BSIM4v5gcrgd * T0;
                  gcrgb = here->BSIM4v5gcrgb * T0;
                  ceqgcrg = -(gcrgg * vgd - gcrgs * vds
                          + gcrgb * vbd);
                  gcrgg -= here->BSIM4v5gcrg;
                  gcrg = here->BSIM4v5gcrg;
              }
              else
                  ceqgcrg = gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;
          }

          if (model->BSIM4v5rdsMod == 1)
          {   ceqgstot = model->BSIM4v5type * (here->BSIM4v5gstotd * vds
                       + here->BSIM4v5gstotg * vgs + here->BSIM4v5gstotb * vbs);
              /* WDLiu: ceqgstot flowing away from sNodePrime */
              gstot = here->BSIM4v5gstot;
              gstotd = here->BSIM4v5gstotd;
              gstotg = here->BSIM4v5gstotg;
              gstots = here->BSIM4v5gstots - gstot;
              gstotb = here->BSIM4v5gstotb;

              ceqgdtot = -model->BSIM4v5type * (here->BSIM4v5gdtotd * vds
                       + here->BSIM4v5gdtotg * vgs + here->BSIM4v5gdtotb * vbs);
              /* WDLiu: ceqgdtot defined as flowing into dNodePrime */
              gdtot = here->BSIM4v5gdtot;
              gdtotd = here->BSIM4v5gdtotd - gdtot;
              gdtotg = here->BSIM4v5gdtotg;
              gdtots = here->BSIM4v5gdtots;
              gdtotb = here->BSIM4v5gdtotb;
          }
          else
          {   gstot = gstotd = gstotg = gstots = gstotb = ceqgstot = 0.0;
              gdtot = gdtotd = gdtotg = gdtots = gdtotb = ceqgdtot = 0.0;
          }

	   if (model->BSIM4v5type > 0)
           {   ceqjs = (here->BSIM4v5cbs - here->BSIM4v5gbs * vbs_jct);
               ceqjd = (here->BSIM4v5cbd - here->BSIM4v5gbd * vbd_jct);
           }
	   else
           {   ceqjs = -(here->BSIM4v5cbs - here->BSIM4v5gbs * vbs_jct); 
               ceqjd = -(here->BSIM4v5cbd - here->BSIM4v5gbd * vbd_jct);
               ceqqg = -ceqqg;
               ceqqd = -ceqqd;
               ceqqb = -ceqqb;
	       ceqgcrg = -ceqgcrg;

               if (here->BSIM4v5trnqsMod)
               {   cqdef = -cqdef;
                   cqcheq = -cqcheq;
	       }

               if (here->BSIM4v5rbodyMod)
               {   ceqqjs = -ceqqjs;
                   ceqqjd = -ceqqjd;
               }

	       if (here->BSIM4v5rgateMod == 3)
		   ceqqgmid = -ceqqgmid;
	   }


           /*
            *  Loading RHS
            */

   	       m = here->BSIM4v5m;

#ifdef USE_OMP
       here->BSIM4v5rhsdPrime = m * (ceqjd - ceqbd + ceqgdtot
                                                    - ceqdrn - ceqqd + Idtoteq);
       here->BSIM4v5rhsgPrime = m * (ceqqg - ceqgcrg + Igtoteq);

       if (here->BSIM4v5rgateMod == 2)
           here->BSIM4v5rhsgExt = m * ceqgcrg;
       else if (here->BSIM4v5rgateMod == 3)
               here->BSIM4v5grhsMid = m * (ceqqgmid + ceqgcrg);

       if (!here->BSIM4v5rbodyMod)
       {   here->BSIM4v5rhsbPrime = m * (ceqbd + ceqbs - ceqjd
                                                        - ceqjs - ceqqb + Ibtoteq);
           here->BSIM4v5rhssPrime = m * (ceqdrn - ceqbs + ceqjs
                              + ceqqg + ceqqb + ceqqd + ceqqgmid - ceqgstot + Istoteq);
        }
        else
        {   here->BSIM4v5rhsdb = m * (ceqjd + ceqqjd);
            here->BSIM4v5rhsbPrime = m * (ceqbd + ceqbs - ceqqb + Ibtoteq);
            here->BSIM4v5rhssb = m * (ceqjs + ceqqjs);
            here->BSIM4v5rhssPrime = m * (ceqdrn - ceqbs + ceqjs + ceqqd
                + ceqqg + ceqqb + ceqqjd + ceqqjs + ceqqgmid - ceqgstot + Istoteq);
        }

        if (model->BSIM4v5rdsMod)
        {   here->BSIM4v5rhsd = m * ceqgdtot;
            here->BSIM4v5rhss = m * ceqgstot;
        }

        if (here->BSIM4v5trnqsMod)
           here->BSIM4v5rhsq = m * (cqcheq - cqdef);
#else
           (*(ckt->CKTrhs + here->BSIM4v5dNodePrime) += m * (ceqjd - ceqbd + ceqgdtot
                                                    - ceqdrn - ceqqd + Idtoteq));
           (*(ckt->CKTrhs + here->BSIM4v5gNodePrime) -= m * (ceqqg - ceqgcrg + Igtoteq));

           if (here->BSIM4v5rgateMod == 2)
               (*(ckt->CKTrhs + here->BSIM4v5gNodeExt) -= m * ceqgcrg);
           else if (here->BSIM4v5rgateMod == 3)
	       (*(ckt->CKTrhs + here->BSIM4v5gNodeMid) -= m * (ceqqgmid + ceqgcrg));

           if (!here->BSIM4v5rbodyMod)
           {   (*(ckt->CKTrhs + here->BSIM4v5bNodePrime) += m * (ceqbd + ceqbs - ceqjd
                                                        - ceqjs - ceqqb + Ibtoteq));
               (*(ckt->CKTrhs + here->BSIM4v5sNodePrime) += m * (ceqdrn - ceqbs + ceqjs 
                              + ceqqg + ceqqb + ceqqd + ceqqgmid - ceqgstot + Istoteq));
           }
           else
           {   (*(ckt->CKTrhs + here->BSIM4v5dbNode) -= m * (ceqjd + ceqqjd));
               (*(ckt->CKTrhs + here->BSIM4v5bNodePrime) += m * (ceqbd + ceqbs - ceqqb + Ibtoteq));
               (*(ckt->CKTrhs + here->BSIM4v5sbNode) -= m * (ceqjs + ceqqjs));
               (*(ckt->CKTrhs + here->BSIM4v5sNodePrime) += m * (ceqdrn - ceqbs + ceqjs + ceqqd 
                + ceqqg + ceqqb + ceqqjd + ceqqjs + ceqqgmid - ceqgstot + Istoteq));
           }

           if (model->BSIM4v5rdsMod)
	   {   (*(ckt->CKTrhs + here->BSIM4v5dNode) -= m * ceqgdtot); 
	       (*(ckt->CKTrhs + here->BSIM4v5sNode) += m * ceqgstot);
	   }

           if (here->BSIM4v5trnqsMod)
               *(ckt->CKTrhs + here->BSIM4v5qNode) += m * (cqcheq - cqdef);
#endif


           /*
            *  Loading matrix
            */

	   if (!here->BSIM4v5rbodyMod)
           {   gjbd = here->BSIM4v5gbd;
               gjbs = here->BSIM4v5gbs;
           }
           else
               gjbd = gjbs = 0.0;

           if (!model->BSIM4v5rdsMod)
           {   gdpr = here->BSIM4v5drainConductance;
               gspr = here->BSIM4v5sourceConductance;
           }
           else
               gdpr = gspr = 0.0;

	   geltd = here->BSIM4v5grgeltd;

           T1 = qdef * here->BSIM4v5gtau;

#ifdef USE_OMP
       if (here->BSIM4v5rgateMod == 1)
       {   here->BSIM4v5_1 = m * geltd;
           here->BSIM4v5_2 = m * geltd;
           here->BSIM4v5_3 = m * geltd;
           here->BSIM4v5_4 = m * (gcggb + geltd - ggtg + gIgtotg);
           here->BSIM4v5_5 = m * (gcgdb - ggtd + gIgtotd);
           here->BSIM4v5_6 = m * (gcgsb - ggts + gIgtots);
           here->BSIM4v5_7 = m * (gcgbb - ggtb + gIgtotb);
       } /* WDLiu: gcrg already subtracted from all gcrgg below */
       else if (here->BSIM4v5rgateMod == 2)
       {   here->BSIM4v5_8 = m * gcrg;
           here->BSIM4v5_9 = m * gcrgg;
           here->BSIM4v5_10 = m * gcrgd;
           here->BSIM4v5_11 = m * gcrgs;
           here->BSIM4v5_12 = m * gcrgb;

           here->BSIM4v5_13 = m * gcrg;
           here->BSIM4v5_14 = m * (gcggb  - gcrgg - ggtg + gIgtotg);
           here->BSIM4v5_15 = m * (gcgdb - gcrgd - ggtd + gIgtotd);
           here->BSIM4v5_16 = m * (gcgsb - gcrgs - ggts + gIgtots);
           here->BSIM4v5_17 = m * (gcgbb - gcrgb - ggtb + gIgtotb);
       }
       else if (here->BSIM4v5rgateMod == 3)
       {   here->BSIM4v5_18 = m * geltd;
           here->BSIM4v5_19 = m * geltd;
           here->BSIM4v5_20 = m * geltd;
           here->BSIM4v5_21 = m * (geltd + gcrg + gcgmgmb);

           here->BSIM4v5_22 = m * (gcrgd + gcgmdb);
           here->BSIM4v5_23 = m * gcrgg;
           here->BSIM4v5_24 = m * (gcrgs + gcgmsb);
           here->BSIM4v5_25 = m * (gcrgb + gcgmbb);

           here->BSIM4v5_26 = m * gcdgmb;
           here->BSIM4v5_27 = m * gcrg;
           here->BSIM4v5_28 = m * gcsgmb;
           here->BSIM4v5_29 = m * gcbgmb;

           here->BSIM4v5_30 = m * (gcggb - gcrgg - ggtg + gIgtotg);
           here->BSIM4v5_31 = m * (gcgdb - gcrgd - ggtd + gIgtotd);
           here->BSIM4v5_32 = m * (gcgsb - gcrgs - ggts + gIgtots);
           here->BSIM4v5_33 = m * (gcgbb - gcrgb - ggtb + gIgtotb);
       }
       else
       {   here->BSIM4v5_34 = m * (gcggb - ggtg + gIgtotg);
           here->BSIM4v5_35 = m * (gcgdb - ggtd + gIgtotd);
           here->BSIM4v5_36 = m * (gcgsb - ggts + gIgtots);
           here->BSIM4v5_37 = m * (gcgbb - ggtb + gIgtotb);
       }

       if (model->BSIM4v5rdsMod)
       {   here->BSIM4v5_38 = m * gdtotg;
           here->BSIM4v5_39 = m * gdtots;
           here->BSIM4v5_40 = m * gdtotb;
           here->BSIM4v5_41 = m * gstotd;
           here->BSIM4v5_42 = m * gstotg;
           here->BSIM4v5_43 = m * gstotb;
       }

       here->BSIM4v5_44 = m * (gdpr + here->BSIM4v5gds + here->BSIM4v5gbd + T1 * ddxpart_dVd
                                   - gdtotd + RevSum + gcddb + gbdpdp + dxpart * ggtd - gIdtotd);
       here->BSIM4v5_45 = m * (gdpr + gdtot);
       here->BSIM4v5_46 = m * (Gm + gcdgb - gdtotg + gbdpg - gIdtotg
                                   + dxpart * ggtg + T1 * ddxpart_dVg);
       here->BSIM4v5_47 = m * (here->BSIM4v5gds + gdtots - dxpart * ggts + gIdtots
                                   - T1 * ddxpart_dVs + FwdSum - gcdsb - gbdpsp);
       here->BSIM4v5_48 = m * (gjbd + gdtotb - Gmbs - gcdbb - gbdpb + gIdtotb
                                   - T1 * ddxpart_dVb - dxpart * ggtb);

       here->BSIM4v5_49 = m * (gdpr - gdtotd);
       here->BSIM4v5_50 = m * (gdpr + gdtot);

       here->BSIM4v5_51 = m * (here->BSIM4v5gds + gstotd + RevSum - gcsdb - gbspdp
                                   - T1 * dsxpart_dVd - sxpart * ggtd + gIstotd);
       here->BSIM4v5_52 = m * (gcsgb - Gm - gstotg + gbspg + sxpart * ggtg
                                   + T1 * dsxpart_dVg - gIstotg);
       here->BSIM4v5_53 = m * (gspr + here->BSIM4v5gds + here->BSIM4v5gbs + T1 * dsxpart_dVs
                                   - gstots + FwdSum + gcssb + gbspsp + sxpart * ggts - gIstots);
       here->BSIM4v5_54 = m * (gspr + gstot);
       here->BSIM4v5_55 = m * (gjbs + gstotb + Gmbs - gcsbb - gbspb - sxpart * ggtb
                                   - T1 * dsxpart_dVb + gIstotb);

       here->BSIM4v5_56 = m * (gspr - gstots);
       here->BSIM4v5_57 = m * (gspr + gstot);

       here->BSIM4v5_58 = m * (gcbdb - gjbd + gbbdp - gIbtotd);
       here->BSIM4v5_59 = m * (gcbgb - here->BSIM4v5gbgs - gIbtotg);
       here->BSIM4v5_60 = m * (gcbsb - gjbs + gbbsp - gIbtots);
       here->BSIM4v5_61 = m * (gjbd + gjbs + gcbbb - here->BSIM4v5gbbs - gIbtotb);

       ggidld = here->BSIM4v5ggidld;
       ggidlg = here->BSIM4v5ggidlg;
       ggidlb = here->BSIM4v5ggidlb;
       ggislg = here->BSIM4v5ggislg;
       ggisls = here->BSIM4v5ggisls;
       ggislb = here->BSIM4v5ggislb;

       /* stamp gidl */
       here->BSIM4v5_62 = m * ggidld;
       here->BSIM4v5_63 = m * ggidlg;
       here->BSIM4v5_64 = m * (ggidlg + ggidld + ggidlb);
       here->BSIM4v5_65 = m * ggidlb;
       here->BSIM4v5_66 = m * ggidld;
       here->BSIM4v5_67 = m * ggidlg;
       here->BSIM4v5_68 = m * (ggidlg + ggidld + ggidlb);
       here->BSIM4v5_69 = m * ggidlb;
       /* stamp gisl */
       here->BSIM4v5_70 = m * (ggisls + ggislg + ggislb);
       here->BSIM4v5_71 = m * ggislg;
       here->BSIM4v5_72 = m * ggisls;
       here->BSIM4v5_73 = m * ggislb;
       here->BSIM4v5_74 = m * (ggislg + ggisls + ggislb);
       here->BSIM4v5_75 = m * ggislg;
       here->BSIM4v5_76 = m * ggisls;
       here->BSIM4v5_77 = m * ggislb;

       if (here->BSIM4v5rbodyMod)
       {   here->BSIM4v5_78 = m * (gcdbdb - here->BSIM4v5gbd);
           here->BSIM4v5_79 = m * (here->BSIM4v5gbs - gcsbsb);

           here->BSIM4v5_80 = m * (gcdbdb - here->BSIM4v5gbd);
           here->BSIM4v5_81 = m * (here->BSIM4v5gbd - gcdbdb
                          + here->BSIM4v5grbpd + here->BSIM4v5grbdb);
           here->BSIM4v5_82 = m * here->BSIM4v5grbpd;
           here->BSIM4v5_83 = m * here->BSIM4v5grbdb;

           here->BSIM4v5_84 = m * here->BSIM4v5grbpd;
           here->BSIM4v5_85 = m * here->BSIM4v5grbpb;
           here->BSIM4v5_86 = m * here->BSIM4v5grbps;
           here->BSIM4v5_87 = m * (here->BSIM4v5grbpd + here->BSIM4v5grbps
                          + here->BSIM4v5grbpb);
           /* WDLiu: (gcbbb - here->BSIM4v5gbbs) already added to BPbpPtr */

           here->BSIM4v5_88 = m * (gcsbsb - here->BSIM4v5gbs);
           here->BSIM4v5_89 = m * here->BSIM4v5grbps;
           here->BSIM4v5_90 = m * here->BSIM4v5grbsb;
           here->BSIM4v5_91 = m * (here->BSIM4v5gbs - gcsbsb
                          + here->BSIM4v5grbps + here->BSIM4v5grbsb);

           here->BSIM4v5_92 = m * here->BSIM4v5grbdb;
           here->BSIM4v5_93 = m * here->BSIM4v5grbpb;
           here->BSIM4v5_94 = m * here->BSIM4v5grbsb;
           here->BSIM4v5_95 = m * (here->BSIM4v5grbsb + here->BSIM4v5grbdb
                           + here->BSIM4v5grbpb);
       }

           if (here->BSIM4v5trnqsMod)
           {   here->BSIM4v5_96 = m * (gqdef + here->BSIM4v5gtau);
               here->BSIM4v5_97 = m * (ggtg - gcqgb);
               here->BSIM4v5_98 = m * (ggtd - gcqdb);
               here->BSIM4v5_99 = m * (ggts - gcqsb);
               here->BSIM4v5_100 = m * (ggtb - gcqbb);

               here->BSIM4v5_101 = m * dxpart * here->BSIM4v5gtau;
               here->BSIM4v5_102 = m * sxpart * here->BSIM4v5gtau;
               here->BSIM4v5_103 = m * here->BSIM4v5gtau;
           }
#else
           if (here->BSIM4v5rgateMod == 1)
           {   (*(here->BSIM4v5GEgePtr) += m * geltd);
               (*(here->BSIM4v5GPgePtr) -= m * geltd);
               (*(here->BSIM4v5GEgpPtr) -= m * geltd);
  	       (*(here->BSIM4v5GPgpPtr) += m * (gcggb + geltd - ggtg + gIgtotg));
  	       (*(here->BSIM4v5GPdpPtr) += m * (gcgdb - ggtd + gIgtotd));
   	       (*(here->BSIM4v5GPspPtr) += m * (gcgsb - ggts + gIgtots));
	       (*(here->BSIM4v5GPbpPtr) += m * (gcgbb - ggtb + gIgtotb));
           } /* WDLiu: gcrg already subtracted from all gcrgg below */
           else if (here->BSIM4v5rgateMod == 2)	
	   {   (*(here->BSIM4v5GEgePtr) += m * gcrg);
               (*(here->BSIM4v5GEgpPtr) += m * gcrgg);
               (*(here->BSIM4v5GEdpPtr) += m * gcrgd);
               (*(here->BSIM4v5GEspPtr) += m * gcrgs);
	       (*(here->BSIM4v5GEbpPtr) += m * gcrgb);	

               (*(here->BSIM4v5GPgePtr) -= m * gcrg);
  	       (*(here->BSIM4v5GPgpPtr) += m * (gcggb  - gcrgg - ggtg + gIgtotg));
	       (*(here->BSIM4v5GPdpPtr) += m * (gcgdb - gcrgd - ggtd + gIgtotd));
	       (*(here->BSIM4v5GPspPtr) += m * (gcgsb - gcrgs - ggts + gIgtots));
  	       (*(here->BSIM4v5GPbpPtr) += m * (gcgbb - gcrgb - ggtb + gIgtotb));
	   }
	   else if (here->BSIM4v5rgateMod == 3)
	   {   (*(here->BSIM4v5GEgePtr) += m * geltd);
               (*(here->BSIM4v5GEgmPtr) -= m * geltd);
               (*(here->BSIM4v5GMgePtr) -= m * geltd);
               (*(here->BSIM4v5GMgmPtr) += m * (geltd + gcrg + gcgmgmb));

               (*(here->BSIM4v5GMdpPtr) += m * (gcrgd + gcgmdb));
               (*(here->BSIM4v5GMgpPtr) += m * gcrgg);
               (*(here->BSIM4v5GMspPtr) += m * (gcrgs + gcgmsb));
               (*(here->BSIM4v5GMbpPtr) += m * (gcrgb + gcgmbb));

               (*(here->BSIM4v5DPgmPtr) += m * gcdgmb);
               (*(here->BSIM4v5GPgmPtr) -= m * gcrg);
               (*(here->BSIM4v5SPgmPtr) += m * gcsgmb);
               (*(here->BSIM4v5BPgmPtr) += m * gcbgmb);

               (*(here->BSIM4v5GPgpPtr) += m * (gcggb - gcrgg - ggtg + gIgtotg));
               (*(here->BSIM4v5GPdpPtr) += m * (gcgdb - gcrgd - ggtd + gIgtotd));
               (*(here->BSIM4v5GPspPtr) += m * (gcgsb - gcrgs - ggts + gIgtots));
               (*(here->BSIM4v5GPbpPtr) += m * (gcgbb - gcrgb - ggtb + gIgtotb));
	   }
 	   else
	   {   (*(here->BSIM4v5GPgpPtr) += m * (gcggb - ggtg + gIgtotg));
  	       (*(here->BSIM4v5GPdpPtr) += m * (gcgdb - ggtd + gIgtotd));
               (*(here->BSIM4v5GPspPtr) += m * (gcgsb - ggts + gIgtots));
	       (*(here->BSIM4v5GPbpPtr) += m * (gcgbb - ggtb + gIgtotb));
	   }

	   if (model->BSIM4v5rdsMod)
	   {   (*(here->BSIM4v5DgpPtr) += m * gdtotg);
	       (*(here->BSIM4v5DspPtr) += m * gdtots);
               (*(here->BSIM4v5DbpPtr) += m * gdtotb);
               (*(here->BSIM4v5SdpPtr) += m * gstotd);
               (*(here->BSIM4v5SgpPtr) += m * gstotg);
               (*(here->BSIM4v5SbpPtr) += m * gstotb);
	   }

           (*(here->BSIM4v5DPdpPtr) += m * (gdpr + here->BSIM4v5gds + here->BSIM4v5gbd + T1 * ddxpart_dVd
                                   - gdtotd + RevSum + gcddb + gbdpdp + dxpart * ggtd - gIdtotd));
           (*(here->BSIM4v5DPdPtr) -= m * (gdpr + gdtot));
           (*(here->BSIM4v5DPgpPtr) += m * (Gm + gcdgb - gdtotg + gbdpg - gIdtotg
				   + dxpart * ggtg + T1 * ddxpart_dVg));
           (*(here->BSIM4v5DPspPtr) -= m * (here->BSIM4v5gds + gdtots - dxpart * ggts + gIdtots
				   - T1 * ddxpart_dVs + FwdSum - gcdsb - gbdpsp));
           (*(here->BSIM4v5DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gcdbb - gbdpb + gIdtotb
				   - T1 * ddxpart_dVb - dxpart * ggtb));

           (*(here->BSIM4v5DdpPtr) -= m * (gdpr - gdtotd));
           (*(here->BSIM4v5DdPtr) += m * (gdpr + gdtot));

           (*(here->BSIM4v5SPdpPtr) -= m * (here->BSIM4v5gds + gstotd + RevSum - gcsdb - gbspdp
				   - T1 * dsxpart_dVd - sxpart * ggtd + gIstotd));
           (*(here->BSIM4v5SPgpPtr) += m * (gcsgb - Gm - gstotg + gbspg + sxpart * ggtg
				   + T1 * dsxpart_dVg - gIstotg));
           (*(here->BSIM4v5SPspPtr) += m * (gspr + here->BSIM4v5gds + here->BSIM4v5gbs + T1 * dsxpart_dVs
                                   - gstots + FwdSum + gcssb + gbspsp + sxpart * ggts - gIstots));
           (*(here->BSIM4v5SPsPtr) -= m * (gspr + gstot));
           (*(here->BSIM4v5SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gcsbb - gbspb - sxpart * ggtb
				   - T1 * dsxpart_dVb + gIstotb));

           (*(here->BSIM4v5SspPtr) -= m * (gspr - gstots));
           (*(here->BSIM4v5SsPtr) += m * (gspr + gstot));

           (*(here->BSIM4v5BPdpPtr) += m * (gcbdb - gjbd + gbbdp - gIbtotd));
           (*(here->BSIM4v5BPgpPtr) += m * (gcbgb - here->BSIM4v5gbgs - gIbtotg));
           (*(here->BSIM4v5BPspPtr) += m * (gcbsb - gjbs + gbbsp - gIbtots));
           (*(here->BSIM4v5BPbpPtr) += m * (gjbd + gjbs + gcbbb - here->BSIM4v5gbbs
				   - gIbtotb));

           ggidld = here->BSIM4v5ggidld;
           ggidlg = here->BSIM4v5ggidlg;
           ggidlb = here->BSIM4v5ggidlb;
           ggislg = here->BSIM4v5ggislg;
           ggisls = here->BSIM4v5ggisls;
           ggislb = here->BSIM4v5ggislb;

           /* stamp gidl */
           (*(here->BSIM4v5DPdpPtr) += m * ggidld);
           (*(here->BSIM4v5DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v5DPspPtr) -= m * (ggidlg + ggidld + ggidlb));
           (*(here->BSIM4v5DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v5BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v5BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v5BPspPtr) += m * (ggidlg + ggidld + ggidlb));
           (*(here->BSIM4v5BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v5SPdpPtr) -= m * (ggisls + ggislg + ggislb));
           (*(here->BSIM4v5SPgpPtr) += m * ggislg);
           (*(here->BSIM4v5SPspPtr) += m * ggisls);
           (*(here->BSIM4v5SPbpPtr) += m * ggislb);
           (*(here->BSIM4v5BPdpPtr) += m * (ggislg + ggisls + ggislb));
           (*(here->BSIM4v5BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v5BPspPtr) -= m * ggisls);
           (*(here->BSIM4v5BPbpPtr) -= m * ggislb);


           if (here->BSIM4v5rbodyMod)
           {   (*(here->BSIM4v5DPdbPtr) += m * (gcdbdb - here->BSIM4v5gbd));
               (*(here->BSIM4v5SPsbPtr) -= m * (here->BSIM4v5gbs - gcsbsb));

               (*(here->BSIM4v5DBdpPtr) += m * (gcdbdb - here->BSIM4v5gbd));
               (*(here->BSIM4v5DBdbPtr) += m * (here->BSIM4v5gbd - gcdbdb 
                                       + here->BSIM4v5grbpd + here->BSIM4v5grbdb));
               (*(here->BSIM4v5DBbpPtr) -= m * here->BSIM4v5grbpd);
               (*(here->BSIM4v5DBbPtr) -= m * here->BSIM4v5grbdb);

               (*(here->BSIM4v5BPdbPtr) -= m * here->BSIM4v5grbpd);
               (*(here->BSIM4v5BPbPtr) -= m * here->BSIM4v5grbpb);
               (*(here->BSIM4v5BPsbPtr) -= m * here->BSIM4v5grbps);
               (*(here->BSIM4v5BPbpPtr) += m * (here->BSIM4v5grbpd + here->BSIM4v5grbps 
                                       + here->BSIM4v5grbpb));
	       /* WDLiu: (gcbbb - here->BSIM4v5gbbs) already added to BPbpPtr */	

               (*(here->BSIM4v5SBspPtr) += m * (gcsbsb - here->BSIM4v5gbs));
               (*(here->BSIM4v5SBbpPtr) -= m * here->BSIM4v5grbps);
               (*(here->BSIM4v5SBbPtr) -= m * here->BSIM4v5grbsb);
               (*(here->BSIM4v5SBsbPtr) += m * (here->BSIM4v5gbs - gcsbsb 
                                       + here->BSIM4v5grbps + here->BSIM4v5grbsb));

               (*(here->BSIM4v5BdbPtr) -= m * here->BSIM4v5grbdb);
               (*(here->BSIM4v5BbpPtr) -= m * here->BSIM4v5grbpb);
               (*(here->BSIM4v5BsbPtr) -= m * here->BSIM4v5grbsb);
               (*(here->BSIM4v5BbPtr) += m * (here->BSIM4v5grbsb + here->BSIM4v5grbdb
                                     + here->BSIM4v5grbpb));
           }

           if (here->BSIM4v5trnqsMod)
           {   (*(here->BSIM4v5QqPtr) += m * (gqdef + here->BSIM4v5gtau));
               (*(here->BSIM4v5QgpPtr) += m * (ggtg - gcqgb));
               (*(here->BSIM4v5QdpPtr) += m * (ggtd - gcqdb));
               (*(here->BSIM4v5QspPtr) += m * (ggts - gcqsb));
               (*(here->BSIM4v5QbpPtr) += m * (ggtb - gcqbb));

               (*(here->BSIM4v5DPqPtr) += m * dxpart * here->BSIM4v5gtau);
               (*(here->BSIM4v5SPqPtr) += m * sxpart * here->BSIM4v5gtau);
               (*(here->BSIM4v5GPqPtr) -= m * here->BSIM4v5gtau);
           }
#endif

line1000:  ;

#ifndef USE_OMP
     }  /* End of MOSFET Instance */
}   /* End of Model Instance */
#endif

return(OK);
}


#ifdef USE_OMP
void BSIM4v5LoadRhsMat(GENmodel *inModel, CKTcircuit *ckt)
{
    int InstCount, idx;
    BSIM4v5instance **InstArray;
    BSIM4v5instance *here;
    BSIM4v5model *model = (BSIM4v5model*)inModel;

    InstArray = model->BSIM4v5InstanceArray;
    InstCount = model->BSIM4v5InstCount;

    for(idx = 0; idx < InstCount; idx++) {
       here = InstArray[idx];
       model = BSIM4v5modPtr(here);
        /* Update b for Ax = b */
           (*(ckt->CKTrhs + here->BSIM4v5dNodePrime) += here->BSIM4v5rhsdPrime);
           (*(ckt->CKTrhs + here->BSIM4v5gNodePrime) -= here->BSIM4v5rhsgPrime);

           if (here->BSIM4v5rgateMod == 2)
               (*(ckt->CKTrhs + here->BSIM4v5gNodeExt) -= here->BSIM4v5rhsgExt);
           else if (here->BSIM4v5rgateMod == 3)
               (*(ckt->CKTrhs + here->BSIM4v5gNodeMid) -= here->BSIM4v5grhsMid);

           if (!here->BSIM4v5rbodyMod)
           {   (*(ckt->CKTrhs + here->BSIM4v5bNodePrime) += here->BSIM4v5rhsbPrime);
               (*(ckt->CKTrhs + here->BSIM4v5sNodePrime) += here->BSIM4v5rhssPrime);
           }
           else
           {   (*(ckt->CKTrhs + here->BSIM4v5dbNode) -= here->BSIM4v5rhsdb);
               (*(ckt->CKTrhs + here->BSIM4v5bNodePrime) += here->BSIM4v5rhsbPrime);
               (*(ckt->CKTrhs + here->BSIM4v5sbNode) -= here->BSIM4v5rhssb);
               (*(ckt->CKTrhs + here->BSIM4v5sNodePrime) += here->BSIM4v5rhssPrime);
           }

           if (model->BSIM4v5rdsMod)
           {   (*(ckt->CKTrhs + here->BSIM4v5dNode) -= here->BSIM4v5rhsd);
               (*(ckt->CKTrhs + here->BSIM4v5sNode) += here->BSIM4v5rhss);
           }

           if (here->BSIM4v5trnqsMod)
               *(ckt->CKTrhs + here->BSIM4v5qNode) += here->BSIM4v5rhsq;


        /* Update A for Ax = b */
           if (here->BSIM4v5rgateMod == 1)
           {   (*(here->BSIM4v5GEgePtr) += here->BSIM4v5_1);
               (*(here->BSIM4v5GPgePtr) -= here->BSIM4v5_2);
               (*(here->BSIM4v5GEgpPtr) -= here->BSIM4v5_3);
               (*(here->BSIM4v5GPgpPtr) += here->BSIM4v5_4);
               (*(here->BSIM4v5GPdpPtr) += here->BSIM4v5_5);
               (*(here->BSIM4v5GPspPtr) += here->BSIM4v5_6);
               (*(here->BSIM4v5GPbpPtr) += here->BSIM4v5_7);
           }
           else if (here->BSIM4v5rgateMod == 2)
           {   (*(here->BSIM4v5GEgePtr) += here->BSIM4v5_8);
               (*(here->BSIM4v5GEgpPtr) += here->BSIM4v5_9);
               (*(here->BSIM4v5GEdpPtr) += here->BSIM4v5_10);
               (*(here->BSIM4v5GEspPtr) += here->BSIM4v5_11);
               (*(here->BSIM4v5GEbpPtr) += here->BSIM4v5_12);

               (*(here->BSIM4v5GPgePtr) -= here->BSIM4v5_13);
               (*(here->BSIM4v5GPgpPtr) += here->BSIM4v5_14);
               (*(here->BSIM4v5GPdpPtr) += here->BSIM4v5_15);
               (*(here->BSIM4v5GPspPtr) += here->BSIM4v5_16);
               (*(here->BSIM4v5GPbpPtr) += here->BSIM4v5_17);
           }
           else if (here->BSIM4v5rgateMod == 3)
           {   (*(here->BSIM4v5GEgePtr) += here->BSIM4v5_18);
               (*(here->BSIM4v5GEgmPtr) -= here->BSIM4v5_19);
               (*(here->BSIM4v5GMgePtr) -= here->BSIM4v5_20);
               (*(here->BSIM4v5GMgmPtr) += here->BSIM4v5_21);

               (*(here->BSIM4v5GMdpPtr) += here->BSIM4v5_22);
               (*(here->BSIM4v5GMgpPtr) += here->BSIM4v5_23);
               (*(here->BSIM4v5GMspPtr) += here->BSIM4v5_24);
               (*(here->BSIM4v5GMbpPtr) += here->BSIM4v5_25);

               (*(here->BSIM4v5DPgmPtr) += here->BSIM4v5_26);
               (*(here->BSIM4v5GPgmPtr) -= here->BSIM4v5_27);
               (*(here->BSIM4v5SPgmPtr) += here->BSIM4v5_28);
               (*(here->BSIM4v5BPgmPtr) += here->BSIM4v5_29);

               (*(here->BSIM4v5GPgpPtr) += here->BSIM4v5_30);
               (*(here->BSIM4v5GPdpPtr) += here->BSIM4v5_31);
               (*(here->BSIM4v5GPspPtr) += here->BSIM4v5_32);
               (*(here->BSIM4v5GPbpPtr) += here->BSIM4v5_33);
           }


            else
           {   (*(here->BSIM4v5GPgpPtr) += here->BSIM4v5_34);
               (*(here->BSIM4v5GPdpPtr) += here->BSIM4v5_35);
               (*(here->BSIM4v5GPspPtr) += here->BSIM4v5_36);
               (*(here->BSIM4v5GPbpPtr) += here->BSIM4v5_37);
           }


           if (model->BSIM4v5rdsMod)
           {   (*(here->BSIM4v5DgpPtr) += here->BSIM4v5_38);
               (*(here->BSIM4v5DspPtr) += here->BSIM4v5_39);
               (*(here->BSIM4v5DbpPtr) += here->BSIM4v5_40);
               (*(here->BSIM4v5SdpPtr) += here->BSIM4v5_41);
               (*(here->BSIM4v5SgpPtr) += here->BSIM4v5_42);
               (*(here->BSIM4v5SbpPtr) += here->BSIM4v5_43);
           }

           (*(here->BSIM4v5DPdpPtr) += here->BSIM4v5_44);
           (*(here->BSIM4v5DPdPtr) -= here->BSIM4v5_45);
           (*(here->BSIM4v5DPgpPtr) += here->BSIM4v5_46);
           (*(here->BSIM4v5DPspPtr) -= here->BSIM4v5_47);
           (*(here->BSIM4v5DPbpPtr) -= here->BSIM4v5_48);

           (*(here->BSIM4v5DdpPtr) -= here->BSIM4v5_49);
           (*(here->BSIM4v5DdPtr) += here->BSIM4v5_50);

           (*(here->BSIM4v5SPdpPtr) -= here->BSIM4v5_51);
           (*(here->BSIM4v5SPgpPtr) += here->BSIM4v5_52);
           (*(here->BSIM4v5SPspPtr) += here->BSIM4v5_53);
           (*(here->BSIM4v5SPsPtr) -= here->BSIM4v5_54);
           (*(here->BSIM4v5SPbpPtr) -= here->BSIM4v5_55);

           (*(here->BSIM4v5SspPtr) -= here->BSIM4v5_56);
           (*(here->BSIM4v5SsPtr) += here->BSIM4v5_57);

           (*(here->BSIM4v5BPdpPtr) += here->BSIM4v5_58);
           (*(here->BSIM4v5BPgpPtr) += here->BSIM4v5_59);
           (*(here->BSIM4v5BPspPtr) += here->BSIM4v5_60);
           (*(here->BSIM4v5BPbpPtr) += here->BSIM4v5_61);

           /* stamp gidl */
           (*(here->BSIM4v5DPdpPtr) += here->BSIM4v5_62);
           (*(here->BSIM4v5DPgpPtr) += here->BSIM4v5_63);
           (*(here->BSIM4v5DPspPtr) -= here->BSIM4v5_64);
           (*(here->BSIM4v5DPbpPtr) += here->BSIM4v5_65);
           (*(here->BSIM4v5BPdpPtr) -= here->BSIM4v5_66);
           (*(here->BSIM4v5BPgpPtr) -= here->BSIM4v5_67);
           (*(here->BSIM4v5BPspPtr) += here->BSIM4v5_68);
           (*(here->BSIM4v5BPbpPtr) -= here->BSIM4v5_69);
            /* stamp gisl */
           (*(here->BSIM4v5SPdpPtr) -= here->BSIM4v5_70);
           (*(here->BSIM4v5SPgpPtr) += here->BSIM4v5_71);
           (*(here->BSIM4v5SPspPtr) += here->BSIM4v5_72);
           (*(here->BSIM4v5SPbpPtr) += here->BSIM4v5_73);
           (*(here->BSIM4v5BPdpPtr) += here->BSIM4v5_74);
           (*(here->BSIM4v5BPgpPtr) -= here->BSIM4v5_75);
           (*(here->BSIM4v5BPspPtr) -= here->BSIM4v5_76);
           (*(here->BSIM4v5BPbpPtr) -= here->BSIM4v5_77);


           if (here->BSIM4v5rbodyMod)
           {   (*(here->BSIM4v5DPdbPtr) += here->BSIM4v5_78);
               (*(here->BSIM4v5SPsbPtr) -= here->BSIM4v5_79);

               (*(here->BSIM4v5DBdpPtr) += here->BSIM4v5_80);
               (*(here->BSIM4v5DBdbPtr) += here->BSIM4v5_81);
               (*(here->BSIM4v5DBbpPtr) -= here->BSIM4v5_82);
               (*(here->BSIM4v5DBbPtr) -= here->BSIM4v5_83);

               (*(here->BSIM4v5BPdbPtr) -= here->BSIM4v5_84);
               (*(here->BSIM4v5BPbPtr) -= here->BSIM4v5_85);
               (*(here->BSIM4v5BPsbPtr) -= here->BSIM4v5_86);
               (*(here->BSIM4v5BPbpPtr) += here->BSIM4v5_87);

               (*(here->BSIM4v5SBspPtr) += here->BSIM4v5_88);
               (*(here->BSIM4v5SBbpPtr) -= here->BSIM4v5_89);
               (*(here->BSIM4v5SBbPtr) -= here->BSIM4v5_90);
               (*(here->BSIM4v5SBsbPtr) += here->BSIM4v5_91);

               (*(here->BSIM4v5BdbPtr) -= here->BSIM4v5_92);
               (*(here->BSIM4v5BbpPtr) -= here->BSIM4v5_93);
               (*(here->BSIM4v5BsbPtr) -= here->BSIM4v5_94);
               (*(here->BSIM4v5BbPtr) += here->BSIM4v5_95);
           }

           if (here->BSIM4v5trnqsMod)
           {   (*(here->BSIM4v5QqPtr) += here->BSIM4v5_96);
               (*(here->BSIM4v5QgpPtr) += here->BSIM4v5_97);
               (*(here->BSIM4v5QdpPtr) += here->BSIM4v5_98);
               (*(here->BSIM4v5QspPtr) += here->BSIM4v5_99);
               (*(here->BSIM4v5QbpPtr) += here->BSIM4v5_100);

               (*(here->BSIM4v5DPqPtr) += here->BSIM4v5_101);
               (*(here->BSIM4v5SPqPtr) += here->BSIM4v5_102);
               (*(here->BSIM4v5GPqPtr) -= here->BSIM4v5_103);
           }
    }
}

#endif


/* function to compute poly depletion effect */
int BSIM4v5polyDepletion(
    double  phi,
    double  ngate,
    double  coxe,
    double  Vgs,
    double *Vgs_eff,
    double *dVgs_eff_dVg)
{
    double T1, T2, T3, T4, T5, T6, T7, T8;

    /* Poly Gate Si Depletion Effect */
    if ((ngate > 1.0e18) && 
        (ngate < 1.0e25) && (Vgs > phi)) {
        T1 = 1.0e6 * CHARGE * EPSSI * ngate / (coxe * coxe);
        T8 = Vgs - phi;
        T4 = sqrt(1.0 + 2.0 * T8 / T1);
        T2 = 2.0 * T8 / (T4 + 1.0);
        T3 = 0.5 * T2 * T2 / T1; /* T3 = Vpoly */
        T7 = 1.12 - T3 - 0.05;
        T6 = sqrt(T7 * T7 + 0.224);
        T5 = 1.12 - 0.5 * (T7 + T6);
        *Vgs_eff = Vgs - T5;
        *dVgs_eff_dVg = 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6); 
    }
    else {
        *Vgs_eff = Vgs;
        *dVgs_eff_dVg = 1.0;
    }
    return(0);
}
