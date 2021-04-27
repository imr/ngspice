/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdld.c          98/5/01
Modified by Pin Su, Weidong Liu and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su, Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 00/8/15
Modified by Pin Su 01/2/15
Modified by Pin Su and Hui Wan 02/3/5
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define Charge_q 1.60219e-19
#define KboQ 8.617087e-5  /*  Kb / q   */
#define Eg300 1.115   /*  energy gap at 300K  */
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
/* Original is 0.02, for matching IBM model, change to 0.08 */
#define DELTA_3_SOI 0.08
#define DELTA_4 0.02
#define DELT_Vbseff  0.005
#define DELTA_VFB  0.02
#define CONST_2OV3 0.6666666666

#define MAX_EXPL 2.688117142e+43
#define MIN_EXPL 3.720075976e-44
#define EXPL_THRESHOLD 100.0
#define DEXP(A,B,C) {                                                         \
        if (A > EXPL_THRESHOLD) {                                              \
            B = MAX_EXPL*(1.0+(A)-EXPL_THRESHOLD);                              \
            C = MAX_EXPL;                                                            \
        } else if (A < -EXPL_THRESHOLD)  {                                                \
            B = MIN_EXPL;                                                      \
            C = 0;                                                            \
        } else   {                                                            \
            B = exp(A);                                                       \
            C = B;                                                            \
        }                                                                     \
    }

#define FLOG(A)  fabs(A) + 1e-14


    /* B3SOIPDlimit(vnew,vold)
     *  limits the per-iteration change of any absolute voltage value
     */

static double
B3SOIPDlimit(double vnew, double vold, double limit, int *check)
{
    double T0, T1;

    if (isnan (vnew) || isnan (vold))
    {
	fprintf(stderr, "Alberto says:  YOU TURKEY!  The limiting function received NaN.\n");
	fprintf(stderr, "New prediction returns to 0.0!\n");
        vnew = 0.0;
        *check = 1;
    }
    T0 = vnew - vold;
    T1 = fabs(T0);
    if (T1 > limit) {
        if (T0 > 0.0) 
            vnew = vold + limit;
        else 
            vnew = vold - limit;
	*check = 1;
    }
    return vnew;
}



int
B3SOIPDload(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;
int selfheat;

double Gmin;
double ag0, qgd, qgs, von, cbhat, VgstNVt, ExpVgst = 0.0;
double cdhat, cdreq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg, ceq, geq;
double arg;
double delvbd, delvbs, delvds, delvgd, delvgs;
double Vfbeff, dVfbeff_dVg, dVfbeff_dVd, dVfbeff_dVb, V3, V4;
double PhiBSWG, MJSWG;
double gcgdb, gcggb, gcgsb, gcgeb, gcgT;
double gcsdb, gcsgb, gcssb, gcseb, gcsT;
double gcddb, gcdgb, gcdsb, gcdeb, gcdT;
double gcbdb, gcbgb, gcbsb, gcbeb, gcbT;
double gcedb, gcegb, gcesb, gceeb, gceT;
double gcTt, gTtg, gTtb, gTtdp, gTtt, gTtsp;
double vbd, vbs, vds, vgb, vgd, vgs, vgdo;
#ifndef PREDICTOR
double xfact;
#endif
double vg, vd, vs, vp, ve, vb;
double Vds, Vgs, Vbs, Gmbs, FwdSum, RevSum;
double Vgs_eff, Vfb, dVfb_dVb, dVfb_dVd, dVfb_dT;
double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth, dVth_dVb, dVth_dVd, dVth_dT;
double Vgst, dVgst_dVg, dVgst_dVb, dVgs_eff_dVg;
double n, dn_dVb, Vtm;
double ExpArg, V0;
double ueff, dueff_dVg, dueff_dVd, dueff_dVb, dueff_dT;
double Esat, Vdsat;
double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb, dEsatL_dT;
double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, dVdsat_dT, Vasat;
double dVasat_dVg, dVasat_dVb, dVasat_dVd, dVasat_dT;
double Va, dVa_dVd, dVa_dVg, dVa_dVb, dVa_dT;
double Vbseff, dVbseff_dVb;
double CoxWL;
double T0, dT0_dVg, dT0_dVd, dT0_dVb, dT0_dT;
double T1, dT1_dVg, dT1_dVd, dT1_dVb, dT1_dT;
double T2, dT2_dVg, dT2_dVd, dT2_dVb, dT2_dT;
double T3, dT3_dVg, dT3_dVd, dT3_dVb, dT3_dT = 0.0;
double T4, dT4_dVd, dT4_dVb, dT4_dT;
double T5, dT5_dVg, dT5_dVd, dT5_dVb, dT5_dT = 0.0;
double T6, dT6_dT = 0.0;
double T7, Vdsatii;
double T8;
double T9;
double T10, dT10_dVb;
double T11, T12;
double tmp, Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb;
double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb, dVACLM_dT;
double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb, dVADIBL_dT;
double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb;
double Delt_vth, dDelt_vth_dVb, dDelt_vth_dT;
double Theta0, dTheta0_dVb;
double TempRatio, tmp1, tmp2, tmp3, tmp4;
double DIBL_Sft, dDIBL_Sft_dVd, Lambda, dLambda_dVg;
double a1;
 
double Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb, dVgsteff_dT;
double Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb, dVdseff_dT;
double VdseffCV, dVdseffCV_dVg, dVdseffCV_dVd, dVdseffCV_dVb;
double diffVds;
double dAbulk_dVg, dn_dVd ;
double beta, dbeta_dVg, dbeta_dVd, dbeta_dVb, dbeta_dT;
double gche, dgche_dVg, dgche_dVd, dgche_dVb, dgche_dT;
double fgche1, dfgche1_dVg, dfgche1_dVd, dfgche1_dVb, dfgche1_dT;
double fgche2, dfgche2_dVg, dfgche2_dVd, dfgche2_dVb, dfgche2_dT;
double Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb, dIdl_dT;
double Ids, Gm, Gds, Gmb;
double CoxWovL;
double Rds, dRds_dVg, dRds_dVb, dRds_dT, WVCox, WVCoxRds;
double Vgst2Vtm, dVgst2Vtm_dT, VdsatCV, dVdsatCV_dVg, dVdsatCV_dVb;
double Leff, Weff, dWeff_dVg, dWeff_dVb;
double AbulkCV, dAbulkCV_dVb;
double qgdo, qgso, cgdo, cgso;
 
double dxpart, sxpart;
 
struct b3soipdSizeDependParam *pParam;
int ByPass, Check, ChargeComputationNeeded = 0, error;
 
double gbbsp, gbbdp, gbbg, gbbb, gbbp, gbbT;
double gddpsp, gddpdp, gddpg, gddpb, gddpT;
double gsspsp, gsspdp, gsspg, gsspb, gsspT;
double Gbpbs, Gbpps;
double ves, ved, veb, vge = 0.0, delves, vedo, delved;
double vps, vpd, Vps, delvps;
double Vbd, Ves, Vesfb, DeltVthtemp, dDeltVthtemp_dT;
double Vbp, dVbp_dVb;
double DeltVthw, dDeltVthw_dVb, dDeltVthw_dT;
double Gm0, Gds0, Gmb0, GmT0, Gmc, GmT;
double dDIBL_Sft_dVb;
double Idgidl, Gdgidld, Gdgidlg, Isgidl, Gsgidlg;
double Gjsd, Gjsb, GjsT, Gjdd, Gjdb, GjdT;
double Ibp, Iii, Giid, Giig, Giib, GiiT, Gcd, Gcb, GcT, ceqbody, ceqbodcon;
double gppb, gppp, gppT;
double delTemp, deldelTemp, Temp;
double ceqth, ceqqth;
double K1;
double qjs = 0.0, gcjsbs, gcjsT;
double qjd = 0.0, gcjdbs, gcjdds, gcjdT;
double qge;
double ceqqe;
double ni, Eg, Cbox, CboxWL;
double cjsbs;
double dVfbeff_dVrg;
double qinv, qgate = 0.0, qbody = 0.0, qdrn = 0.0, qsrc, qsub = 0.0;
double cqgate, cqbody, cqdrn, cqsub, cqtemp;
double Cgg, Cgd, Cgb;
double Csg, Csd, Csb, Cbg = 0.0, Cbd = 0.0, Cbb = 0.0;
double Cgg1, Cgb1, Cgd1, Cbg1, Cbb1, Cbd1, Csg1, Csd1, Csb1;
double Ibs1 ,dIbs1_dVb ,dIbs1_dT;
double Ibs2 ,dIbs2_dVb ,dIbs2_dT;
double Ibs3 ,dIbs3_dVb ,dIbs3_dVd, dIbs3_dT;
double Ibs4 ,dIbs4_dVb ,dIbs4_dT;
double Ibd1 ,dIbd1_dVb ,dIbd1_dVd ,dIbd1_dT;
double Ibd2 ,dIbd2_dVb ,dIbd2_dVd ,dIbd2_dT;
double Ibd3 ,dIbd3_dVb ,dIbd3_dVd ,dIbd3_dT;
double Ibd4 ,dIbd4_dVb ,dIbd4_dVd ,dIbd4_dT;
double WTsi, NVtm1, NVtm2;
double Ic  ,dIc_dVb ,dIc_dVd;
double Ibs;
double Ibd;
double Denomi ,dDenomi_dVg ,dDenomi_dVd ,dDenomi_dVb ,dDenomi_dT;
double Qsub0 = 0.0, dQsub0_dVg, dQsub0_dVb, dQsub0_dVd;
double Qac0 = 0.0, dQac0_dVb, dQac0_dVd;
double Qe1 , dQe1_dVb, dQe1_dVe, dQe1_dT;
double Ce1b ,Ce1e, Ce1T;
double dQac0_dVrg, dQsub0_dVrg; 

/*  for self-heating  */
double vbi, vfbb, phi, sqrtPhi, Xdep0, jbjt, jdif, jrec, jtun, u0temp, vsattemp;
double rds0, ua, ub, uc;
double dvbi_dT, dvfbb_dT, djbjt_dT, djdif_dT, djrec_dT, djtun_dT, du0temp_dT;
double dvsattemp_dT, drds0_dT, dua_dT, dub_dT, duc_dT, dni_dT, dVtm_dT;
double dVfbeff_dT, dQac0_dT, dQsub0_dT;
double CbT, CsT, CgT;


/* v2.0 release */
double Vbsh, dVbsh_dVb;
double sqrtPhisExt, dsqrtPhisExt_dVb;
double T13, T14;
double dT11_dVb, dT13_dVb, dT14_dVb;
double dVgst_dVd;
double Vdsatii0, dVdsatii0_dT;
double VgsStep, dVgsStep_dT, Ratio, dRatio_dVg, dRatio_dVb, dRatio_dVd, dRatio_dT, dTempRatio_dT;
double Vdiff, dVdiff_dVg, dVdiff_dVb, dVdiff_dVd, dVdiff_dT;
double dNVtm1_dT;
double NVtmf, NVtmr, dNVtmf_dT, dNVtmr_dT;
double TempRatioMinus1;
double Ahli, dAhli_dT;
double WsTsi, WdTsi;
double dPhiBSWG_dT, dcjsbs_dT, darg_dT = 0.0, ddT3_dVb_dT = 0.0;   
double dT7_dT, dT0_dT7, dT1_dT7, dT2_dT7;
double CoxWLb, CoxWLcenb;
double ExpVbsNVtm, dExpVbsNVtm_dVb, dExpVbsNVtm_dT;
double ExpVbdNVtm, dExpVbdNVtm_dVb, dExpVbdNVtm_dVd, dExpVbdNVtm_dT;
double Ien, dIen_dT, Iendif, dIendif_dT;
double Ibsdif, dIbsdif_dVb, dIbsdif_dT;
double Ibddif, dIbddif_dVb, dIbddif_dVd, dIbddif_dT;
double Ehlis, dEhlis_dVb, dEhlis_dT;
double EhlisFactor, dEhlisFactor_dVb, dEhlisFactor_dT;
double Ehlid, dEhlid_dVb, dEhlid_dVd, dEhlid_dT;
double EhlidFactor, dEhlidFactor_dVb, dEhlidFactor_dVd, dEhlidFactor_dT;
double E2ndFactor, dE2ndFactor_dVb, dE2ndFactor_dVd, dE2ndFactor_dT;
double dT10_dT, dT11_dT, DioMax;
double cjdbs, dcjdbs_dT;
double wdios, wdiod, wdiosCV, wdiodCV;

/* for capMod3 */
double Cox, Tox, Tcen, dTcen_dVg, dTcen_dVb, LINK, Ccen, Coxeff, dCoxeff_dVg, dCoxeff_dVb;
double CoxWLcen, QovCox, dQac0_dVg, DeltaPhi, dDeltaPhi_dVg, dDeltaPhi_dVd, dDeltaPhi_dVb;
double dTcen_dVd, dTcen_dT, dCoxeff_dVd, dCoxeff_dT, dCoxWLcenb_dT, qinoi, qbulk;
double T3zb, lt1zb, ltwzb, Theta0zb;
double Delt_vthzb, dDelt_vthzb_dT;
double DeltVthwzb, dDeltVthwzb_dT;
double DeltVthtempzb, dDeltVthtempzb_dT;
double Vthzb = 0.0, dVthzb_dT = 0.0, Vfbzb, dVfbzb_dT;


/* v2.2 release */
double Vgb, dVgb_dVg, dVgb_dVb, Vox, dVox_dVg, dVox_dVd, dVox_dVb;
double OxideRatio, Vaux, dVaux_dVg, dVaux_dVd, dVaux_dVb;
double Igb, dIgb_dVg, dIgb_dVd, dIgb_dVb;
double ceqgate;
double dT0_dVox, Voxeff, dVoxeff_dVox;
double dVox_dT = 0.0, dVaux_dT = 0.0, dIgb_dT;
double Voxacc, dVoxacc_dVg, dVoxacc_dVd, dVoxacc_dVb;
double Voxdepinv, dVoxdepinv_dVg, dVoxdepinv_dVb, dVoxdepinv_dVd;
double dVoxdepinv_dT = 0.0;
double Igb1, dIgb1_dVg, dIgb1_dVd, dIgb1_dVb, dIgb1_dT;
double Igb2, dIgb2_dVg, dIgb2_dVd, dIgb2_dVb, dIgb2_dT;
double gigs, gigd, gigb, gigg;
double gigT;

double m;

for (; model != NULL; model = B3SOIPDnextModel(model))
{    for (here = B3SOIPDinstances(model); here != NULL; 
          here = B3SOIPDnextInstance(here))
     {
          Check = 0;
          ByPass = 0;
          selfheat = (model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0);
	  pParam = here->pParam;


          if ((ckt->CKTmode & MODEINITSMSIG))
	  {   
              vs = *(ckt->CKTrhsOld + here->B3SOIPDsNodePrime);   
              if (!here->B3SOIPDvbsusrGiven) {
                 vbs = *(ckt->CKTstate0 + here->B3SOIPDvbs);
                 vb = *(ckt->CKTrhsOld + here->B3SOIPDbNode);
              }
              else {
                   vbs = here->B3SOIPDvbsusr;
                   vb = here->B3SOIPDvbsusr + vs;
              }
              vgs = *(ckt->CKTstate0 + here->B3SOIPDvgs);
              ves = *(ckt->CKTstate0 + here->B3SOIPDves);
              vps = *(ckt->CKTstate0 + here->B3SOIPDvps);
              vds = *(ckt->CKTstate0 + here->B3SOIPDvds);
              delTemp = *(ckt->CKTstate0 + here->B3SOIPDdeltemp);

              vg = *(ckt->CKTrhsOld + here->B3SOIPDgNode);
              vd = *(ckt->CKTrhsOld + here->B3SOIPDdNodePrime);
              vp = *(ckt->CKTrhsOld + here->B3SOIPDpNode);
              ve = *(ckt->CKTrhsOld + here->B3SOIPDeNode);  

          }
	  else if ((ckt->CKTmode & MODEINITTRAN))
	  {
              vs = *(ckt->CKTrhsOld + here->B3SOIPDsNodePrime);
              if (!here->B3SOIPDvbsusrGiven) {
                 vbs = *(ckt->CKTstate1 + here->B3SOIPDvbs);
                 vb = *(ckt->CKTrhsOld + here->B3SOIPDbNode);
              }
              else {
                   vbs = here->B3SOIPDvbsusr;
                   vb = here->B3SOIPDvbsusr + vs;
              }
              vgs = *(ckt->CKTstate1 + here->B3SOIPDvgs);
              ves = *(ckt->CKTstate1 + here->B3SOIPDves);
              vps = *(ckt->CKTstate1 + here->B3SOIPDvps);
              vds = *(ckt->CKTstate1 + here->B3SOIPDvds);
              delTemp = *(ckt->CKTstate1 + here->B3SOIPDdeltemp);

              vg = *(ckt->CKTrhsOld + here->B3SOIPDgNode);
              vd = *(ckt->CKTrhsOld + here->B3SOIPDdNodePrime);
              vp = *(ckt->CKTrhsOld + here->B3SOIPDpNode);
              ve = *(ckt->CKTrhsOld + here->B3SOIPDeNode);

          }
	  else if ((ckt->CKTmode & MODEINITJCT) && !here->B3SOIPDoff)
	  {   vds = model->B3SOIPDtype * here->B3SOIPDicVDS;
              vgs = model->B3SOIPDtype * here->B3SOIPDicVGS;
              ves = model->B3SOIPDtype * here->B3SOIPDicVES;
              vbs = model->B3SOIPDtype * here->B3SOIPDicVBS;
              vps = model->B3SOIPDtype * here->B3SOIPDicVPS;

	      vg = vd = vs = vp = ve = 0.0;


              delTemp = 0.0;
	      here->B3SOIPDphi = pParam->B3SOIPDphi;



	      if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) && 
	         ((ckt->CKTmode & (MODETRAN | MODEAC|MODEDCOP |
		   MODEDCTRANCURVE)) || (!(ckt->CKTmode & MODEUIC))))
	      {   vbs = 0.0;
		  vgs = model->B3SOIPDtype*0.1 + pParam->B3SOIPDvth0;
		  vds = 0.0;
		  ves = 0.0;
		  vps = 0.0;
	      }
	  }
	  else if ((ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) && 
		  (here->B3SOIPDoff)) 
	  {    delTemp = vps = vbs = vgs = vds = ves = 0.0;
               vg = vd = vs = vp = ve = 0.0;
	  }
	  else
	  {
#ifndef PREDICTOR
	       if ((ckt->CKTmode & MODEINITPRED))
	       {   xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
		   *(ckt->CKTstate0 + here->B3SOIPDvbs) = 
			 *(ckt->CKTstate1 + here->B3SOIPDvbs);
		   vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDvbs))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDvbs)));
		   *(ckt->CKTstate0 + here->B3SOIPDvgs) = 
			 *(ckt->CKTstate1 + here->B3SOIPDvgs);
		   vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDvgs))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDvgs)));
		   *(ckt->CKTstate0 + here->B3SOIPDves) = 
			 *(ckt->CKTstate1 + here->B3SOIPDves);
		   ves = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDves))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDves)));
		   *(ckt->CKTstate0 + here->B3SOIPDvps) = 
			 *(ckt->CKTstate1 + here->B3SOIPDvps);
		   vps = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDvps))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDvps)));
		   *(ckt->CKTstate0 + here->B3SOIPDvds) = 
			 *(ckt->CKTstate1 + here->B3SOIPDvds);
		   vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDvds))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDvds)));
		   *(ckt->CKTstate0 + here->B3SOIPDvbd) = 
			 *(ckt->CKTstate0 + here->B3SOIPDvbs)
			 - *(ckt->CKTstate0 + here->B3SOIPDvds);

                   *(ckt->CKTstate0 + here->B3SOIPDvg) = *(ckt->CKTstate1 + here->B3SOIPDvg);
                   *(ckt->CKTstate0 + here->B3SOIPDvd) = *(ckt->CKTstate1 + here->B3SOIPDvd);
                   *(ckt->CKTstate0 + here->B3SOIPDvs) = *(ckt->CKTstate1 + here->B3SOIPDvs);
                   *(ckt->CKTstate0 + here->B3SOIPDvp) = *(ckt->CKTstate1 + here->B3SOIPDvp);
                   *(ckt->CKTstate0 + here->B3SOIPDve) = *(ckt->CKTstate1 + here->B3SOIPDve);

                   /* Only predict ve */
                   ve = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIPDve))

                        - (xfact * (*(ckt->CKTstate2 + here->B3SOIPDve)));
                   /* Then update vg, vs, vb, vd, vp base on ve */
                   vs = ve - model->B3SOIPDtype * ves;
                   vg = model->B3SOIPDtype * vgs + vs;
                   vd = model->B3SOIPDtype * vds + vs;
                   vb = model->B3SOIPDtype * vbs + vs;
                   vp = model->B3SOIPDtype * vps + vs;

		   delTemp = (1.0 + xfact)* (*(ckt->CKTstate1 +
			 here->B3SOIPDdeltemp))-(xfact * (*(ckt->CKTstate2 +
			 here->B3SOIPDdeltemp)));

/* v2.2.3 bug fix */
                   *(ckt->CKTstate0 + here->B3SOIPDdeltemp) = 
                         *(ckt->CKTstate1 + here->B3SOIPDdeltemp);

		   if (selfheat)
		   {
		       here->B3SOIPDphi = 2.0 * here->B3SOIPDvtm
					* log(pParam->B3SOIPDnpeak /
					       here->B3SOIPDni); 
		   }

	       }
	       else
	       {
#endif /* PREDICTOR */

                   vg = B3SOIPDlimit(*(ckt->CKTrhsOld + here->B3SOIPDgNode),
                                 *(ckt->CKTstate0 + here->B3SOIPDvg), 3.0, &Check);
                   vd = B3SOIPDlimit(*(ckt->CKTrhsOld + here->B3SOIPDdNodePrime),
                                 *(ckt->CKTstate0 + here->B3SOIPDvd), 3.0, &Check);
                   vs = B3SOIPDlimit(*(ckt->CKTrhsOld + here->B3SOIPDsNodePrime),
                                 *(ckt->CKTstate0 + here->B3SOIPDvs), 3.0, &Check);
                   vp = B3SOIPDlimit(*(ckt->CKTrhsOld + here->B3SOIPDpNode),
                                 *(ckt->CKTstate0 + here->B3SOIPDvp), 3.0, &Check);
                   ve = B3SOIPDlimit(*(ckt->CKTrhsOld + here->B3SOIPDeNode),
                                 *(ckt->CKTstate0 + here->B3SOIPDve), 3.0, &Check);
                   delTemp = *(ckt->CKTrhsOld + here->B3SOIPDtempNode);

		   vbs = model->B3SOIPDtype * (*(ckt->CKTrhsOld+here->B3SOIPDbNode)
                                - *(ckt->CKTrhsOld+here->B3SOIPDsNodePrime));

		   vps = model->B3SOIPDtype * (vp - vs);
		   vgs = model->B3SOIPDtype * (vg - vs);
		   ves = model->B3SOIPDtype * (ve - vs);
		   vds = model->B3SOIPDtype * (vd - vs);


#ifndef PREDICTOR
	       }
#endif /* PREDICTOR */

	       vbd = vbs - vds;
	       vgd = vgs - vds;
               ved = ves - vds;
	       vgdo = *(ckt->CKTstate0 + here->B3SOIPDvgs)
		    - *(ckt->CKTstate0 + here->B3SOIPDvds);
	       vedo = *(ckt->CKTstate0 + here->B3SOIPDves)
		    - *(ckt->CKTstate0 + here->B3SOIPDvds);
	       delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIPDvbs);
	       delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIPDvbd);
	       delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIPDvgs);
	       delves = ves - *(ckt->CKTstate0 + here->B3SOIPDves);
	       delvps = vps - *(ckt->CKTstate0 + here->B3SOIPDvps);
	       deldelTemp = delTemp - *(ckt->CKTstate0 + here->B3SOIPDdeltemp);
	       delvds = vds - *(ckt->CKTstate0 + here->B3SOIPDvds);
	       delvgd = vgd - vgdo;
               delved = ved - vedo;

	       if (here->B3SOIPDmode >= 0) 
	       {   
                   cdhat = here->B3SOIPDcd + (here->B3SOIPDgm-here->B3SOIPDgjdg) * delvgs
                         + (here->B3SOIPDgds - here->B3SOIPDgjdd) * delvds
                         + (here->B3SOIPDgmbs - here->B3SOIPDgjdb) * delvbs
			 + (here->B3SOIPDgmT - here->B3SOIPDgjdT) * deldelTemp;
	       }
	       else
	       {   
                   cdhat = here->B3SOIPDcd + (here->B3SOIPDgm-here->B3SOIPDgjdg) * delvgd
                         - (here->B3SOIPDgds - here->B3SOIPDgjdd) * delvds
                         + (here->B3SOIPDgmbs - here->B3SOIPDgjdb) * delvbd
                         + (here->B3SOIPDgmT - here->B3SOIPDgjdT) * deldelTemp;

	       }
	       cbhat = here->B3SOIPDcb + here->B3SOIPDgbgs * delvgs
		     + here->B3SOIPDgbbs * delvbs 
                     + here->B3SOIPDgbds * delvds
                     + here->B3SOIPDgbps * delvps 
                     + here->B3SOIPDgbT * deldelTemp;

#ifndef NOBYPASS
	   /* following should be one big if connected by && all over
	    * the place, but some C compilers can't handle that, so
	    * we split it up here to let them digest it in stages
	    */

	       if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass) && Check == 0)
	       if ((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDvbs))) + ckt->CKTvoltTol))  )
	       if ((fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDvbd))) + ckt->CKTvoltTol))  )
	       if ((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDvgs))) + ckt->CKTvoltTol)))
	       if ((fabs(delves) < (ckt->CKTreltol * MAX(fabs(ves),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDves))) + ckt->CKTvoltTol)))
	       if ( (here->B3SOIPDbodyMod == 0) || (here->B3SOIPDbodyMod == 2) ||
                  (fabs(delvps) < (ckt->CKTreltol * MAX(fabs(vps),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDvps))) + ckt->CKTvoltTol)) )
	       if ( (here->B3SOIPDtempNode == 0)  ||
                  (fabs(deldelTemp) < (ckt->CKTreltol * MAX(fabs(delTemp),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDdeltemp)))
		   + ckt->CKTvoltTol*1e4)))
	       if ((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
		   fabs(*(ckt->CKTstate0+here->B3SOIPDvds))) + ckt->CKTvoltTol)))
	       if ((fabs(cdhat - here->B3SOIPDcd) < ckt->CKTreltol 
		   * MAX(fabs(cdhat),fabs(here->B3SOIPDcd)) + ckt->CKTabstol)) 
	       if ((fabs(cbhat - here->B3SOIPDcb) < ckt->CKTreltol 
		   * MAX(fabs(cbhat),fabs(here->B3SOIPDcb)) + ckt->CKTabstol) )
	       {   /* bypass code */
	           vbs = *(ckt->CKTstate0 + here->B3SOIPDvbs);
	           vbd = *(ckt->CKTstate0 + here->B3SOIPDvbd);
	           vgs = *(ckt->CKTstate0 + here->B3SOIPDvgs);
	           ves = *(ckt->CKTstate0 + here->B3SOIPDves);
	           vps = *(ckt->CKTstate0 + here->B3SOIPDvps);
	           vds = *(ckt->CKTstate0 + here->B3SOIPDvds);
	           delTemp = *(ckt->CKTstate0 + here->B3SOIPDdeltemp);

		   /*  calculate Vds for temperature conductance calculation
		       in bypass (used later when filling Temp node matrix)  */
		   Vds = here->B3SOIPDmode > 0 ? vds : -vds;

	           vgd = vgs - vds;
	           vgb = vgs - vbs;
                   veb = ves - vbs;

		   if ((ckt->CKTmode & (MODETRAN | MODEAC)) || 
		      ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
		   {   ByPass = 1;
		       goto line755;
		   }
		   else
		   {   goto line850;
		   }
	       }


#endif /*NOBYPASS*/
		       von = here->B3SOIPDvon;


                       if (*(ckt->CKTstate0 + here->B3SOIPDvds) >= 0.0)
                          T0 = *(ckt->CKTstate0 + here->B3SOIPDvbs);
                       else
                          T0 = *(ckt->CKTstate0 + here->B3SOIPDvbd);


		       if (vds >= 0.0) 
		       {   
		           vbs = B3SOIPDlimit(vbs, T0, 0.2, &Check);
			   vbd = vbs - vds;
                           vb = model->B3SOIPDtype * vbs + vs;
		       } else 
		       {   
		           vbd = B3SOIPDlimit(vbd, T0, 0.2, &Check);
			   vbs = vbd + vds;
                           vb = model->B3SOIPDtype * vbs + vd;
		       }

		       delTemp =B3SOIPDlimit(delTemp, *(ckt->CKTstate0 + here->B3SOIPDdeltemp),5.0,&Check);

                  }

/*  Calculate temperature dependent values for self-heating effect  */
		  Temp = delTemp + ckt->CKTtemp;
                  dTempRatio_dT = 1 / model->B3SOIPDtnom; 
                  TempRatio = Temp * dTempRatio_dT;

		  if (selfheat) {
		      Vtm = KboQ * Temp;

                      T0 = 1108.0 + Temp;
		      T5 = Temp * Temp;
		      Eg = 1.16 - 7.02e-4 * T5 / T0;
		      T1 = ((7.02e-4 * T5) - T0 * (14.04e-4 * Temp)) / T0 / T0;
                      /*  T1 = dEg / dT  */

                      T2 = 1.9230584e-4;  /*  T2 = 1 / 300.15^(3/2)  */
		      T5 = sqrt(Temp);
                      T3 = 1.45e10 * Temp * T5 * T2;
                      T4 = exp(21.5565981 - Eg / (2.0 * Vtm));
		      ni = T3 * T4;
                      dni_dT = 2.175e10 * T2 * T5 * T4 + T3 * T4 *
                               (-Vtm * T1 + Eg * KboQ) / (2.0 * Vtm * Vtm);

                      T0 = log(1.0e20 * pParam->B3SOIPDnpeak / (ni * ni));
		      vbi = Vtm * T0;
                      dvbi_dT = KboQ * T0 + Vtm * (-2.0 * dni_dT / ni);

		      if (pParam->B3SOIPDnsub > 0) {
                         T0 = log(pParam->B3SOIPDnpeak / pParam->B3SOIPDnsub);
		         vfbb = -model->B3SOIPDtype * Vtm * T0;
                         dvfbb_dT = -model->B3SOIPDtype * KboQ * T0;
                      } 
		      else {
                         T0 = log(-pParam->B3SOIPDnpeak * pParam->B3SOIPDnsub / ni / ni);
		         vfbb = -model->B3SOIPDtype * Vtm * T0;
                         dvfbb_dT = -model->B3SOIPDtype *
                                   (KboQ * T0 - Vtm * 2.0 * dni_dT / ni);
                      }

/*		      phi = 2.0 * Vtm * log(pParam->B3SOIPDnpeak / ni);  */
		      phi = here->B3SOIPDphi;
		      sqrtPhi = sqrt(phi);
		      Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				         * pParam->B3SOIPDnpeak * 1.0e6))
				         * sqrtPhi;
		      /*  Save the values below for phi calculation in B3SOIPDaccept()  */
		      here->B3SOIPDvtm = Vtm;
		      here->B3SOIPDni = ni;

                      T3 = TempRatio - 1.0;
                      T8 = 1/ model->B3SOIPDtnom;
                      T4 = Eg300 / Vtm * T3;
                      dT4_dT = Eg300 / Vtm / Vtm * (Vtm * T8 - T3 * KboQ);

                      T7 = model->B3SOIPDxbjt * T4 / pParam->B3SOIPDndiode;
                      dT7_dT = model->B3SOIPDxbjt * dT4_dT / pParam->B3SOIPDndiode;
                      DEXP(T7, T0, dT0_dT7);
                      dT0_dT = dT0_dT7 * dT7_dT;
 
                      if (model->B3SOIPDxbjt == model->B3SOIPDxdif) {
                         T1 = T0;
                         dT1_dT = dT0_dT;
                      } 
                      else {
                         T7 = model->B3SOIPDxdif * T4 / pParam->B3SOIPDndiode;
                         dT7_dT = model->B3SOIPDxdif * dT4_dT / pParam->B3SOIPDndiode;
                         DEXP(T7, T1, dT1_dT7);
                         dT1_dT = dT1_dT7 * dT7_dT;
                      }

                      T7 = model->B3SOIPDxrec * T4 / pParam->B3SOIPDnrecf0;
                      dT7_dT = model->B3SOIPDxrec * dT4_dT / pParam->B3SOIPDnrecf0;
                      DEXP(T7, T2, dT2_dT7);
                      dT2_dT = dT2_dT7 * dT7_dT;
 
                      /* high level injection */
                      Ahli = pParam->B3SOIPDahli * T0;
                      dAhli_dT = pParam->B3SOIPDahli * dT0_dT;
 
                      jbjt = pParam->B3SOIPDisbjt * T0;
                      jdif = pParam->B3SOIPDisdif * T1;
                      jrec = pParam->B3SOIPDisrec * T2;
                      djbjt_dT = pParam->B3SOIPDisbjt * dT0_dT;
                      djdif_dT = pParam->B3SOIPDisdif * dT1_dT;
                      djrec_dT = pParam->B3SOIPDisrec * dT2_dT;
 
                      T7 = model->B3SOIPDxtun * T3;
                      dT7_dT = model->B3SOIPDxtun * T8;
                      DEXP(T7, T0, dT0_dT7);
                      dT0_dT = dT0_dT7 * dT7_dT;
                      jtun = pParam->B3SOIPDistun * T0;
                      djtun_dT = pParam->B3SOIPDistun * dT0_dT;

		      u0temp = pParam->B3SOIPDu0 * pow(TempRatio, pParam->B3SOIPDute);
                      du0temp_dT = pParam->B3SOIPDu0 * pParam->B3SOIPDute *
                                   pow(TempRatio, pParam->B3SOIPDute - 1.0) * T8;

		      vsattemp = pParam->B3SOIPDvsat - pParam->B3SOIPDat * T3;
                      dvsattemp_dT = -pParam->B3SOIPDat * T8;

		      rds0 = (pParam->B3SOIPDrdsw + pParam->B3SOIPDprt
		          * T3) / pParam->B3SOIPDrds0denom;
                      drds0_dT = pParam->B3SOIPDprt / pParam->B3SOIPDrds0denom * T8;

		      ua = pParam->B3SOIPDuatemp + pParam->B3SOIPDua1 * T3;
		      ub = pParam->B3SOIPDubtemp + pParam->B3SOIPDub1 * T3;
		      uc = pParam->B3SOIPDuctemp + pParam->B3SOIPDuc1 * T3;
                      dua_dT = pParam->B3SOIPDua1 * T8;
                      dub_dT = pParam->B3SOIPDub1 * T8;
                      duc_dT = pParam->B3SOIPDuc1 * T8;
		  }
		  else {
                      vbi = pParam->B3SOIPDvbi;
                      vfbb = pParam->B3SOIPDvfbb;
                      phi = pParam->B3SOIPDphi;
                      sqrtPhi = pParam->B3SOIPDsqrtPhi;
                      Xdep0 = pParam->B3SOIPDXdep0;
                      jbjt = pParam->B3SOIPDjbjt;
                      jdif = pParam->B3SOIPDjdif;
                      jrec = pParam->B3SOIPDjrec;
                      jtun = pParam->B3SOIPDjtun;

                      /* v2.2.2 bug fix */
                      Ahli = pParam->B3SOIPDahli0; 

                      u0temp = pParam->B3SOIPDu0temp;
                      vsattemp = pParam->B3SOIPDvsattemp;
                      rds0 = pParam->B3SOIPDrds0;
                      ua = pParam->B3SOIPDua;
                      ub = pParam->B3SOIPDub;
                      uc = pParam->B3SOIPDuc;
                      dni_dT = dvbi_dT = dvfbb_dT = djbjt_dT = djdif_dT = 0.0;
                      djrec_dT = djtun_dT = du0temp_dT = dvsattemp_dT = 0.0;
                      drds0_dT = dua_dT = dub_dT = duc_dT = 0.0;
                      dAhli_dT = 0; 
		  }
		  
		  /* TempRatio used for Vth and mobility */
		  if (selfheat) {
		      TempRatioMinus1 = Temp / model->B3SOIPDtnom - 1.0;
		  }
		  else {
		      TempRatioMinus1 =  ckt->CKTtemp / model->B3SOIPDtnom - 1.0;
		  }

		  /* determine DC current and derivatives */
		  vbd = vbs - vds;
		  vgd = vgs - vds;
		  vgb = vgs - vbs;
		  ved = ves - vds;
		  veb = ves - vbs;
		  vge = vgs - ves;
		  vpd = vps - vds;


		  if (vds >= 0.0)
		  {   /* normal mode */
		      here->B3SOIPDmode = 1;
		      Vds = vds;
		      Vgs = vgs;
		      Vbs = vbs;
		      Vbd = vbd;
		      Ves = ves;
		      Vps = vps;

                      wdios = pParam->B3SOIPDwdios;
                      wdiod = pParam->B3SOIPDwdiod;
                      wdiosCV = pParam->B3SOIPDwdiosCV;
                      wdiodCV = pParam->B3SOIPDwdiodCV;

		  }
		  else
		  {   /* inverse mode */
		      here->B3SOIPDmode = -1;
		      Vds = -vds;
		      Vgs = vgd;
		      Vbs = vbd;
		      Vbd = vbs;
		      Ves = ved;
		      Vps = vpd;

                      wdios = pParam->B3SOIPDwdiod;
                      wdiod = pParam->B3SOIPDwdios;
                      wdiosCV = pParam->B3SOIPDwdiodCV;
                      wdiodCV = pParam->B3SOIPDwdiosCV;

		  }

		  Vesfb = Ves - vfbb;
		  Cbox = model->B3SOIPDcbox;
		  K1 = pParam->B3SOIPDk1eff;

		  ChargeComputationNeeded =  
			 ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
			 ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
				 ? 1 : 0;

                  if (here->B3SOIPDdebugMod <0)
                     ChargeComputationNeeded = 1;
                  



/* Poly Gate Si Depletion Effect */
		  T0 = pParam->B3SOIPDvfb + phi;
		  if ((pParam->B3SOIPDngate > 1.e18) && (pParam->B3SOIPDngate < 1.e25) 
		       && (Vgs > T0))
		  /* added to avoid the problem caused by ngate */
		  {   T1 = 1.0e6 * Charge_q * EPSSI * pParam->B3SOIPDngate
			 / (model->B3SOIPDcox * model->B3SOIPDcox);
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


		  Leff = pParam->B3SOIPDleff;

		  if (selfheat) {
		      Vtm = KboQ * Temp;
                      dVtm_dT = KboQ;
		  }
		  else {
		      Vtm = model->B3SOIPDvtm;
                      dVtm_dT = 0.0;
		  }

		  V0 = vbi - phi;

                  Vbp = Vbs - Vps;
                  dVbp_dVb = 1;


                  /* T2 is Vbs limited above Vbsc=-5 */
                  T0 = Vbs + 5 - 0.001;
                  T1 = sqrt(T0 * T0 - 0.004 * (-5));
                  T2 = (-5) + 0.5 * (T0 + T1);
                  dT2_dVb = 0.5 * (1.0 + T0 / T1);

                  /* Vbsh is T2 limited below 1.5 */
                  T0 = 1.5;
                  T1 = T0 - T2 - 0.002;
                  T3 = sqrt(T1 * T1 + 0.008 * T0);
                  Vbsh = T0 - 0.5 * (T1 + T3);
                  dVbsh_dVb = 0.5 * (1.0 + T1 / T3) * dT2_dVb;

                  /* Vbseff is Vbsh limited to 0.95*phi */
                  T0 = 0.95 * phi;
                  T1 = T0 - Vbsh - 0.002;
                  T2 = sqrt(T1 * T1 + 0.008 * T0);
                  Vbseff = T0 - 0.5 * (T1 + T2);
                  dVbseff_dVb = 0.5 * (1.0 + T1 / T2) * dVbsh_dVb;
                  here->B3SOIPDvbseff = Vbs;
            
                  /* Below all the variables refer to Vbseff */
                  if (dVbseff_dVb < 1e-20) {
                     dVbseff_dVb = 1e-20;
                     dVbsh_dVb *= 1e20;
                  }
                  else
                     dVbsh_dVb /= dVbseff_dVb;
 
                  Phis = phi - Vbseff;
                  dPhis_dVb = -1;
                  sqrtPhis = sqrt(Phis);
                  dsqrtPhis_dVb = -0.5 / sqrtPhis;

                  Xdep = Xdep0 * sqrtPhis / sqrtPhi;
                  dXdep_dVb = (Xdep0 / sqrtPhi) * dsqrtPhis_dVb;

/* Vth Calculation */
		  T3 = sqrt(Xdep);
                  
		  T0 = pParam->B3SOIPDdvt2 * Vbseff;
		  if (T0 >= - 0.5)
		  {   T1 = 1.0 + T0;
		      T2 = pParam->B3SOIPDdvt2 ;
		  }
		  else /* Added to avoid any discontinuity problems caused by dvt2 */ 
		  {   T4 = 1.0 / (3.0 + 8.0 * T0);
		      T1 = (1.0 + 3.0 * T0) * T4; 
		      T2 = pParam->B3SOIPDdvt2 * T4 * T4 ;
		  }
		  lt1 = model->B3SOIPDfactor1 * T3 * T1;
		  dlt1_dVb =model->B3SOIPDfactor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

		  T0 = pParam->B3SOIPDdvt2w * Vbseff;
		  if (T0 >= - 0.5)
		  {   T1 = 1.0 + T0;
		      T2 = pParam->B3SOIPDdvt2w ;
		  }
		  else /* Added to avoid any discontinuity problems caused by dvt2w */ 
		  {   T4 = 1.0 / (3.0 + 8.0 * T0);
		      T1 = (1.0 + 3.0 * T0) * T4; 
		      T2 = pParam->B3SOIPDdvt2w * T4 * T4 ;
		  }
		  ltw= model->B3SOIPDfactor1 * T3 * T1;
		  dltw_dVb=model->B3SOIPDfactor1*(0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

		  T0 = -0.5 * pParam->B3SOIPDdvt1 * Leff / lt1;
		  if (T0 > -EXPL_THRESHOLD)
		  {   T1 = exp(T0);
		      Theta0 = T1 * (1.0 + 2.0 * T1);
		      dT1_dVb = -T0 / lt1 * T1 * dlt1_dVb;
		      dTheta0_dVb = (1.0 + 4.0 * T1) * dT1_dVb;
		  }
		  else
		  {   T1 = MIN_EXPL;
		      Theta0 = T1 * (1.0 + 2.0 * T1);
		      dTheta0_dVb = 0.0;
		  }

		  here->B3SOIPDthetavth = pParam->B3SOIPDdvt0 * Theta0;
		  Delt_vth = here->B3SOIPDthetavth * V0;
		  dDelt_vth_dVb = pParam->B3SOIPDdvt0 * dTheta0_dVb * V0;
                  if (selfheat)  dDelt_vth_dT = here->B3SOIPDthetavth * dvbi_dT;
                  else  dDelt_vth_dT = 0.0;

		  T0 = -0.5 * pParam->B3SOIPDdvt1w * pParam->B3SOIPDweff * Leff / ltw;
		  if (T0 > -EXPL_THRESHOLD)
		  {   T1 = exp(T0);
		      T2 = T1 * (1.0 + 2.0 * T1);
		      dT1_dVb = -T0 / ltw * T1 * dltw_dVb;
		      dT2_dVb = (1.0 + 4.0 * T1) * dT1_dVb;
		  }
		  else
		  {   T1 = MIN_EXPL;
		      T2 = T1 * (1.0 + 2.0 * T1);
		      dT2_dVb = 0.0;
		  }

		  T0 = pParam->B3SOIPDdvt0w * T2;
		  DeltVthw = T0 * V0;
		  dDeltVthw_dVb = pParam->B3SOIPDdvt0w * dT2_dVb * V0;
                  if (selfheat)   dDeltVthw_dT = T0 * dvbi_dT;
                  else   dDeltVthw_dT = 0.0;

		  T0 = sqrt(1.0 + pParam->B3SOIPDnlx / Leff);
                  T1 = (pParam->B3SOIPDkt1 + pParam->B3SOIPDkt1l / Leff
                        + pParam->B3SOIPDkt2 * Vbseff);
                  DeltVthtemp = pParam->B3SOIPDk1eff * (T0 - 1.0) * sqrtPhi + T1 * TempRatioMinus1;
                  if (selfheat)
                     dDeltVthtemp_dT = T1 / model->B3SOIPDtnom;
                  else
                     dDeltVthtemp_dT = 0.0;

		  tmp2 = model->B3SOIPDtox * phi
		       / (pParam->B3SOIPDweff + pParam->B3SOIPDw0);

		  T3 = pParam->B3SOIPDeta0 + pParam->B3SOIPDetab * Vbseff;
		  if (T3 < 1.0e-4) /* avoid  discontinuity problems caused by etab */ 
		  {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
		      T3 = (2.0e-4 - T3) * T9;
		      T4 = T9 * T9 * pParam->B3SOIPDetab;
		      dT3_dVb = T4 ;
		  }
		  else
		  {   
		      dT3_dVb = pParam->B3SOIPDetab ;
		  }
		  DIBL_Sft = T3 * pParam->B3SOIPDtheta0vb0 * Vds;
		  dDIBL_Sft_dVd = pParam->B3SOIPDtheta0vb0 * T3;
		  dDIBL_Sft_dVb = pParam->B3SOIPDtheta0vb0 * Vds * dT3_dVb;

                  T9 =  2.2361 / sqrtPhi;
                  sqrtPhisExt = sqrtPhis - T9 * (Vbsh - Vbseff);
                  dsqrtPhisExt_dVb = dsqrtPhis_dVb - T9 * (dVbsh_dVb - 1);

                  Vth = model->B3SOIPDtype * pParam->B3SOIPDvth0 + pParam->B3SOIPDk1eff 
                      * (sqrtPhisExt - sqrtPhi) - pParam->B3SOIPDk2 
                      * Vbseff- Delt_vth - DeltVthw +(pParam->B3SOIPDk3 + pParam->B3SOIPDk3b
                      * Vbseff) * tmp2 + DeltVthtemp - DIBL_Sft;
                  here->B3SOIPDvon = Vth; 

                  T6 = pParam->B3SOIPDk3b * tmp2 - pParam->B3SOIPDk2 
                       + pParam->B3SOIPDkt2 * TempRatioMinus1;        
                  dVth_dVb = pParam->B3SOIPDk1eff * dsqrtPhisExt_dVb 
                           - dDelt_vth_dVb - dDeltVthw_dVb
                           + T6 - dDIBL_Sft_dVb;  
                  /*  this is actually dVth_dVbseff  */

                  dVth_dVd = -dDIBL_Sft_dVd;
                  if (selfheat)  
                     dVth_dT = dDeltVthtemp_dT - dDelt_vth_dT - dDeltVthw_dT;
                  else  dVth_dT = 0.0;

                  /* dVthzb_dT calculation */
                  if ((model->B3SOIPDcapMod == 3) && (selfheat == 1)) {
                     T3zb = sqrt(Xdep0);
                     ltwzb = lt1zb = model->B3SOIPDfactor1 * T3zb;

                     T0 = -0.5 * pParam->B3SOIPDdvt1 * Leff / lt1zb;
                     if (T0 > -EXPL_THRESHOLD)
                     {   T1 = exp(T0);
                         Theta0zb = T1 * (1.0 + 2.0 * T1);
                     }
                     else
                     {   T1 = MIN_EXPL;
                         Theta0zb = T1 * (1.0 + 2.0 * T1);
                     }
                     Delt_vthzb = pParam->B3SOIPDdvt0 * Theta0zb * V0;
                     dDelt_vthzb_dT = pParam->B3SOIPDdvt0 * Theta0zb * dvbi_dT;
      
                     T0 = -0.5 * pParam->B3SOIPDdvt1w * pParam->B3SOIPDweff * Leff / ltwzb;
                     if (T0 > -EXPL_THRESHOLD)
                     {   T1 = exp(T0);
                         T2 = T1 * (1.0 + 2.0 * T1);
                     }
                     else
                     {   T1 = MIN_EXPL;
                         T2 = T1 * (1.0 + 2.0 * T1);
                     }
                     T0 = pParam->B3SOIPDdvt0w * T2;
                     DeltVthwzb = T0 * V0;
                     dDeltVthwzb_dT = T0 * dvbi_dT;

                     T0 = sqrt(1.0 + pParam->B3SOIPDnlx / Leff);
                     T1 = (pParam->B3SOIPDkt1 + pParam->B3SOIPDkt1l / Leff);
                     DeltVthtempzb = pParam->B3SOIPDk1eff * (T0 - 1.0) * sqrtPhi
                                   + T1 * TempRatioMinus1;
                     dDeltVthtempzb_dT = T1 / model->B3SOIPDtnom;

                     Vthzb = model->B3SOIPDtype * pParam->B3SOIPDvth0 
                           - Delt_vthzb - DeltVthwzb + pParam->B3SOIPDk3 * tmp2
                           + DeltVthtempzb;
                     dVthzb_dT = dDeltVthtempzb_dT - dDelt_vthzb_dT - dDeltVthwzb_dT;
                  }

/* Calculate n */
		  T2 = pParam->B3SOIPDnfactor * EPSSI / Xdep;
		  dT2_dVb = - T2 / Xdep * dXdep_dVb;

		  T3 = pParam->B3SOIPDcdsc + pParam->B3SOIPDcdscb * Vbseff
		       + pParam->B3SOIPDcdscd * Vds;
		  dT3_dVb = pParam->B3SOIPDcdscb;
		  dT3_dVd = pParam->B3SOIPDcdscd;

		  T4 = (T2 + T3 * Theta0 + pParam->B3SOIPDcit) / model->B3SOIPDcox;
		  dT4_dVb = (dT2_dVb + Theta0 * dT3_dVb + dTheta0_dVb * T3)
                            / model->B3SOIPDcox;
		  dT4_dVd = Theta0 * dT3_dVd / model->B3SOIPDcox;

		  if (T4 >= -0.5)
		  {   n = 1.0 + T4;
		      dn_dVb = dT4_dVb;
		      dn_dVd = dT4_dVd;
		  }
		  else
		   /* avoid  discontinuity problems caused by T4 */ 
		  {   T0 = 1.0 / (3.0 + 8.0 * T4);
		      n = (1.0 + 3.0 * T4) * T0;
		      T0 *= T0;
		      dn_dVb = T0 * dT4_dVb;
		      dn_dVd = T0 * dT4_dVd;
		  }

/* Effective Vgst (Vgsteff) Calculation */

		  Vgst = Vgs_eff - Vth;
                  dVgst_dVg = dVgs_eff_dVg;
                  dVgst_dVd = -dVth_dVd;
                  dVgst_dVb = -dVth_dVb;

		  T10 = 2.0 * n * Vtm;
		  VgstNVt = Vgst / T10;
		  ExpArg = (2.0 * pParam->B3SOIPDvoff - Vgst) / T10;

		  /* MCJ: Very small Vgst */
		  if (VgstNVt > EXPL_THRESHOLD)
		  {   Vgsteff = Vgst;
                      /* T0 is dVgsteff_dVbseff */
                      T0 = -dVth_dVb;
		      dVgsteff_dVg = dVgs_eff_dVg;
		      dVgsteff_dVd = -dVth_dVd;
		      dVgsteff_dVb = T0 * dVbseff_dVb;
                      if (selfheat)
                         dVgsteff_dT  = -dVth_dT;
                      else
                         dVgsteff_dT = 0.0;
		  }
		  else if (ExpArg > EXPL_THRESHOLD)
		  {   T0 = (Vgst - pParam->B3SOIPDvoff) / (n * Vtm);
		      ExpVgst = exp(T0);
		      Vgsteff = Vtm * pParam->B3SOIPDcdep0 / model->B3SOIPDcox * ExpVgst;
		      T3 = Vgsteff / (n * Vtm) ;
                      /* T1 is dVgsteff_dVbseff */
		      T1  = -T3 * (dVth_dVb + T0 * Vtm * dn_dVb);
		      dVgsteff_dVg = T3 * dVgs_eff_dVg;
		      dVgsteff_dVd = -T3 * (dVth_dVd + T0 * Vtm * dn_dVd);
                      dVgsteff_dVb = T1 * dVbseff_dVb;
                      if (selfheat)
                         dVgsteff_dT = -T3 * (dVth_dT + T0 * dVtm_dT * n)
                                     + Vgsteff / Temp;
                      else
                         dVgsteff_dT = 0.0;
		  }
		  else
		  {   ExpVgst = exp(VgstNVt);
		      T1 = T10 * log(1.0 + ExpVgst);
		      dT1_dVg = ExpVgst / (1.0 + ExpVgst);
		      dT1_dVb = -dT1_dVg * (dVth_dVb + Vgst / n * dn_dVb)
			      + T1 / n * dn_dVb; 
		      dT1_dVd = -dT1_dVg * (dVth_dVd + Vgst / n * dn_dVd)
			      + T1 / n * dn_dVd;
                      T3 = (1.0 / Temp);
                      if (selfheat)
                         dT1_dT = -dT1_dVg * (dVth_dT + Vgst * T3) + T1 * T3;
                      else
                         dT1_dT = 0.0;

		      dT2_dVg = -model->B3SOIPDcox / (Vtm * pParam->B3SOIPDcdep0)
			      * exp(ExpArg);
		      T2 = 1.0 - T10 * dT2_dVg;
		      dT2_dVd = -dT2_dVg * (dVth_dVd - 2.0 * Vtm * ExpArg * dn_dVd)
			      + (T2 - 1.0) / n * dn_dVd;
		      dT2_dVb = -dT2_dVg * (dVth_dVb - 2.0 * Vtm * ExpArg * dn_dVb)
			      + (T2 - 1.0) / n * dn_dVb;
                      if (selfheat)
                         dT2_dT = -dT2_dVg * (dVth_dT - ExpArg * T10 * T3);
                      else
                         dT2_dT = 0.0;

		      Vgsteff = T1 / T2;
		      T3 = T2 * T2;
                      /*  T4 is dVgsteff_dVbseff  */
		      T4 = (T2 * dT1_dVb - T1 * dT2_dVb) / T3;
                      dVgsteff_dVb = T4 * dVbseff_dVb;
		      dVgsteff_dVg = (T2 * dT1_dVg - T1 * dT2_dVg) / T3 * dVgs_eff_dVg;
		      dVgsteff_dVd = (T2 * dT1_dVd - T1 * dT2_dVd) / T3;
                      if (selfheat)
                         dVgsteff_dT = (T2 * dT1_dT - T1 * dT2_dT) / T3;
                      else
                         dVgsteff_dT = 0.0;
		  }
		  Vgst2Vtm = Vgsteff + 2.0 * Vtm;
                  if (selfheat)  dVgst2Vtm_dT = 2.0 * dVtm_dT;  
                  else  dVgst2Vtm_dT = 0.0;
		  here->B3SOIPDVgsteff = Vgsteff; /* v2.2.3 bug fix */

/* Calculate Effective Channel Geometry */
		  T9 = sqrtPhis - sqrtPhi;
		  Weff = pParam->B3SOIPDweff - (2.0 - here->B3SOIPDnbc) * (pParam->B3SOIPDdwg * Vgsteff 
		       + pParam->B3SOIPDdwb * T9); 
		  dWeff_dVg = -(2.0 - here->B3SOIPDnbc) * pParam->B3SOIPDdwg;
		  dWeff_dVb = -(2.0 - here->B3SOIPDnbc) * pParam->B3SOIPDdwb * dsqrtPhis_dVb;

		  if (Weff < 2.0e-8) /* to avoid the discontinuity problem due to Weff*/
		  {   T0 = 1.0 / (6.0e-8 - 2.0 * Weff);
		      Weff = 2.0e-8 * (4.0e-8 - Weff) * T0;
		      T0 *= T0 * 4.0e-16;
		      dWeff_dVg *= T0;
		      dWeff_dVb *= T0;
		  }

		  T0 = pParam->B3SOIPDprwg * Vgsteff + pParam->B3SOIPDprwb * T9;
		  if (T0 >= -0.9)
		  {   Rds = rds0 * (1.0 + T0);
		      dRds_dVg = rds0 * pParam->B3SOIPDprwg;
		      dRds_dVb = rds0 * pParam->B3SOIPDprwb * dsqrtPhis_dVb;

                      if (selfheat && (Rds!=0.0))  dRds_dT = (1.0 + T0) * drds0_dT;
                      else  dRds_dT = 0.0;

		  }
		  else
		   /* to avoid the discontinuity problem due to prwg and prwb*/
		  {   T1 = 1.0 / (17.0 + 20.0 * T0);
		      Rds = rds0 * (0.8 + T0) * T1;
		      T1 *= T1;
		      dRds_dVg = rds0 * pParam->B3SOIPDprwg * T1;
		      dRds_dVb = rds0 * pParam->B3SOIPDprwb * dsqrtPhis_dVb
			       * T1;

                      if (selfheat && (Rds!=0.0))  dRds_dT = (0.8 + T0) * T1 * drds0_dT;
                      else  dRds_dT = 0.0;

		  }
		  here->B3SOIPDrds = Rds; /* v2.2.3 bug fix */

/* Calculate Abulk */
                  if (pParam->B3SOIPDa0 == 0.0) {

                     Abulk0 = Abulk = 1.0;

                     dAbulk0_dVb = dAbulk_dVg = dAbulk_dVb = 0.0;
                  }
                  else { 
                     T10 = pParam->B3SOIPDketa * Vbsh; 
                     if (T10 >= -0.9) {
                        T11 = 1.0 / (1.0 + T10);
                        dT11_dVb = -pParam->B3SOIPDketa * T11 * T11 * dVbsh_dVb;
                     }
                     else { /* added to avoid the problems caused by Keta */
                        T12 = 1.0 / (0.8 + T10);
                        T11 = (17.0 + 20.0 * T10) * T12;
                        dT11_dVb = -pParam->B3SOIPDketa * T12 * T12 * dVbsh_dVb;
                     }

                     T10 = pParam->B3SOIPDphi + pParam->B3SOIPDketas;
                     T13 = (Vbsh * T11) / T10;
                     dT13_dVb = (Vbsh * dT11_dVb + T11 * dVbsh_dVb) / T10;

                     /* limit 1/sqrt(1-T13) to 6, starting at T13=0.96 */
                     if (T13 < 0.96) {
                        T14 = 1 / sqrt(1-T13);
                        T10 = 0.5 * T14 / (1-T13);
                        dT14_dVb = T10 * dT13_dVb;
                     }
                     else {
                        T11 = 1.0 / (1.0 - 1.043406*T13);
                        T14 = (6.00167 - 6.26044 * T13) * T11;
                        T10 = 0.001742 * T11 * T11;
                        dT14_dVb = T10 * dT13_dVb;
                     }

                     T10 = 0.5 * pParam->B3SOIPDk1eff 
                         / sqrt(pParam->B3SOIPDphi + pParam->B3SOIPDketas);
                     T1 = T10 * T14;
                     dT1_dVb = T10 * dT14_dVb;

                     T9 = sqrt(model->B3SOIPDxj * Xdep);
                     tmp1 = Leff + 2.0 * T9;
                     T5 = Leff / tmp1;
                     tmp2 = pParam->B3SOIPDa0 * T5;
                     tmp3 = pParam->B3SOIPDweff + pParam->B3SOIPDb1;
                     tmp4 = pParam->B3SOIPDb0 / tmp3;
                     T2 = tmp2 + tmp4;
                     dT2_dVb = -T9 * tmp2 / tmp1 / Xdep * dXdep_dVb;
                     T6 = T5 * T5;
                     T7 = T5 * T6;
 
                     Abulk0 = 1 + T1 * T2;
                     dAbulk0_dVb = T1 * dT2_dVb + T2 * dT1_dVb;
 
                     T8 = pParam->B3SOIPDags * pParam->B3SOIPDa0 * T7;
                     dAbulk_dVg = -T1 * T8;
                     Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
 
                     dAbulk_dVb = dAbulk0_dVb 
                                - T8 * Vgsteff * (dT1_dVb + 3.0 * T1 * dT2_dVb / tmp2);
                  }

                  if (Abulk0 < 0.01)
                  {
                     T9 = 1.0 / (3.0 - 200.0 * Abulk0);
                     Abulk0 = (0.02 - Abulk0) * T9;
                     dAbulk0_dVb *= T9 * T9;
                  }

                  if (Abulk < 0.01)
                  {
                     T9 = 1.0 / (3.0 - 200.0 * Abulk);
                     Abulk = (0.02 - Abulk) * T9;
                     dAbulk_dVb *= T9 * T9;
                  }

/* Mobility calculation */
		  if (model->B3SOIPDmobMod == 1)
		  {   T0 = Vgsteff + Vth + Vth;
		      T2 = ua + uc * Vbseff;
		      T3 = T0 / model->B3SOIPDtox;
		      T5 = T3 * (T2 + ub * T3);
		      dDenomi_dVg = (T2 + 2.0 * ub * T3) / model->B3SOIPDtox;
		      dDenomi_dVd = dDenomi_dVg * 2 * dVth_dVd;
		      dDenomi_dVb = dDenomi_dVg * 2 * dVth_dVb + uc * T3 ;
                      if (selfheat)
                         dDenomi_dT = dDenomi_dVg * 2 * dVth_dT 
                                    + (dua_dT + Vbseff * duc_dT
                                    + dub_dT * T3 ) * T3;
                      else
                         dDenomi_dT = 0.0;
		  }
		  else if (model->B3SOIPDmobMod == 2)
		  {   T5 = Vgsteff / model->B3SOIPDtox * (ua
			 + uc * Vbseff + ub * Vgsteff
			 / model->B3SOIPDtox);
		      dDenomi_dVg = (ua + uc * Vbseff
				  + 2.0 * ub * Vgsteff / model->B3SOIPDtox)
				  / model->B3SOIPDtox;
		      dDenomi_dVd = 0.0;
		      dDenomi_dVb = Vgsteff * uc / model->B3SOIPDtox ;
                      if (selfheat)
                         dDenomi_dT = Vgsteff / model->B3SOIPDtox
                                    * (dua_dT + Vbseff * duc_dT + dub_dT
                                    * Vgsteff / model->B3SOIPDtox);
                      else
                         dDenomi_dT = 0.0;
		  }
		  else  /*  mobMod == 3  */
		  {   T0 = Vgsteff + Vth + Vth;
		      T2 = 1.0 + uc * Vbseff;
		      T3 = T0 / model->B3SOIPDtox;
		      T4 = T3 * (ua + ub * T3);
		      T5 = T4 * T2;
		      dDenomi_dVg = (ua + 2.0 * ub * T3) * T2
				  / model->B3SOIPDtox;
		      dDenomi_dVd = dDenomi_dVg * 2.0 * dVth_dVd;
		      dDenomi_dVb = dDenomi_dVg * 2.0 * dVth_dVb 
				  + uc * T4 ;
                      if (selfheat)
                         dDenomi_dT = dDenomi_dVg * 2.0 * dVth_dT
                                    + (dua_dT + dub_dT * T3) * T3 * T2
                                    + T4 * Vbseff * duc_dT;
                      else
                         dDenomi_dT = 0.0;
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
                      if (selfheat)  dDenomi_dT *= T9;
                      else   dDenomi_dT = 0.0;
		  }

		  here->B3SOIPDueff = ueff = u0temp / Denomi;
		  T9 = -ueff / Denomi;
		  dueff_dVg = T9 * dDenomi_dVg;
		  dueff_dVd = T9 * dDenomi_dVd;
		  dueff_dVb = T9 * dDenomi_dVb;
                  if (selfheat)  dueff_dT = T9 * dDenomi_dT + du0temp_dT / Denomi;
                  else  dueff_dT = 0.0;

/* Saturation Drain Voltage  Vdsat */
		  WVCox = Weff * vsattemp * model->B3SOIPDcox;
		  WVCoxRds = WVCox * Rds; 

/*                  dWVCoxRds_dT = WVCox * dRds_dT
                                 + Weff * model->B3SOIPDcox * Rds * dvsattemp_dT; */

		  Esat = 2.0 * vsattemp / ueff;
		  EsatL = Esat * Leff;
		  T0 = -EsatL /ueff;
		  dEsatL_dVg = T0 * dueff_dVg;
		  dEsatL_dVd = T0 * dueff_dVd;
		  dEsatL_dVb = T0 * dueff_dVb;
                  if (selfheat)
                     dEsatL_dT = T0 * dueff_dT + EsatL / vsattemp * dvsattemp_dT;
                  else
                     dEsatL_dT = 0.0;
	  
		  /* Sqrt() */
		  a1 = pParam->B3SOIPDa1;
		  if (a1 == 0.0)
		  {   Lambda = pParam->B3SOIPDa2;
		      dLambda_dVg = 0.0;
		  }
		  else if (a1 > 0.0)
/* Added to avoid the discontinuity problem caused by a1 and a2 (Lambda) */
		  {   T0 = 1.0 - pParam->B3SOIPDa2;
		      T1 = T0 - pParam->B3SOIPDa1 * Vgsteff - 0.0001;
		      T2 = sqrt(T1 * T1 + 0.0004 * T0);
		      Lambda = pParam->B3SOIPDa2 + T0 - 0.5 * (T1 + T2);
		      dLambda_dVg = 0.5 * pParam->B3SOIPDa1 * (1.0 + T1 / T2);
		  }
		  else
		  {   T1 = pParam->B3SOIPDa2 + pParam->B3SOIPDa1 * Vgsteff - 0.0001;
		      T2 = sqrt(T1 * T1 + 0.0004 * pParam->B3SOIPDa2);
		      Lambda = 0.5 * (T1 + T2);
		      dLambda_dVg = 0.5 * pParam->B3SOIPDa1 * (1.0 + T1 / T2);
		  }

		  here->B3SOIPDAbovVgst2Vtm = Abulk /Vgst2Vtm; /* v2.2.3 bug fix */

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
		      dT0_dVb = -(Abulk * dEsatL_dVb + EsatL * dAbulk_dVb) * T1;
                      if (selfheat)
		         dT0_dT  = -(Abulk * dEsatL_dT + dVgst2Vtm_dT) * T1;
                      else dT0_dT  = 0.0;

		      dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
		      dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
		      dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
                      if (selfheat)
		         dVdsat_dT  = T3 * dT0_dT  + T2 * dEsatL_dT
				    + EsatL * T0 * dVgst2Vtm_dT;
                      else dVdsat_dT  = 0.0;
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
/*		      dT0_dVb = 2.0 * (T8 * tmp3  this is equivalent to one below, but simpler
			      + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbulk_dVg);  */
		      dT0_dVb = 2.0 * (T8 * (2.0 / Abulk * dAbulk_dVb + tmp3)
			      + (1.0 / Lambda - 1.0) * dAbulk_dVb);
		      dT0_dVd = 0.0; 

                      if (selfheat)
                      {

                         if (Rds!=0.0) 
		            tmp4 = dRds_dT / Rds + dvsattemp_dT / vsattemp;
                         else 
                            tmp4 = dvsattemp_dT / vsattemp;

                         dT0_dT = 2.0 * T8 * tmp4;
                      } else tmp4 = dT0_dT = 0.0;

		      T1 = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abulk * EsatL + 3.0 * T7;
		     
		      dT1_dVg = (2.0 / Lambda - 1.0) - 2.0 * Vgst2Vtm * tmp1
			      + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 3.0 * (T9
			      + T7 * tmp2 + T6 * dAbulk_dVg);
		      dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
			      + 3.0 * (T6 * dAbulk_dVb + T7 * tmp3);
		      dT1_dVd = Abulk * dEsatL_dVd;


                      if (selfheat)
                      {
		         tmp4 += dVgst2Vtm_dT / Vgst2Vtm;
		         dT1_dT  = (2.0 / Lambda - 1.0) * dVgst2Vtm_dT
				 + Abulk * dEsatL_dT + 3.0 * T7 * tmp4;
                      } else dT1_dT = 0.0;

		      T2 = Vgst2Vtm * (EsatL + 2.0 * T6);
		      dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg
			      + T6 * (4.0 + 2.0 * Vgst2Vtm * tmp2);
		      dT2_dVb = Vgst2Vtm * (dEsatL_dVb + 2.0 * T6 * tmp3);
		      dT2_dVd = Vgst2Vtm * dEsatL_dVd;
                      if (selfheat)
		         dT2_dT  = Vgst2Vtm * dEsatL_dT + EsatL * dVgst2Vtm_dT
				 + 2.0 * T6 * (dVgst2Vtm_dT + Vgst2Vtm * tmp4);
                      else
		         dT2_dT  = 0.0;

		      T3 = sqrt(T1 * T1 - 2.0 * T0 * T2);
		      Vdsat = (T1 - T3) / T0;

		      dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2
				 - T0 * dT2_dVg) / T3 - Vdsat * dT0_dVg) / T0;
		      dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2
				 - T0 * dT2_dVb) / T3 - Vdsat * dT0_dVb) / T0;
		      dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) / T3) / T0;
                      if (selfheat)
		         dVdsat_dT  = (dT1_dT - (T1 * dT1_dT - dT0_dT * T2
				    - T0 * dT2_dT) / T3 - Vdsat * dT0_dT) / T0;
                      else dVdsat_dT  = 0.0;
		  }
		  here->B3SOIPDvdsat = Vdsat;


/* Effective Vds (Vdseff) Calculation */
		  T1 = Vdsat - Vds - pParam->B3SOIPDdelta;
		  dT1_dVg = dVdsat_dVg;
		  dT1_dVd = dVdsat_dVd - 1.0;
		  dT1_dVb = dVdsat_dVb;
		  dT1_dT  = dVdsat_dT;

		  T2 = sqrt(T1 * T1 + 4.0 * pParam->B3SOIPDdelta * Vdsat);
		  T0 = T1 / T2;
		  T3 = 2.0 * pParam->B3SOIPDdelta / T2;
		  dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
		  dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
		  dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;
                  if (selfheat)
		     dT2_dT  = T0 * dT1_dT  + T3 * dVdsat_dT;
                  else dT2_dT  = 0.0;

		  Vdseff = Vdsat - 0.5 * (T1 + T2);
		  dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg); 
		  dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd); 
		  dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb); 
                  if (selfheat)
		     dVdseff_dT  = dVdsat_dT  - 0.5 * (dT1_dT  + dT2_dT);
                  else dVdseff_dT  = 0.0;

		  if (Vdseff > Vds)
		      Vdseff = Vds; /* This code is added to fixed the problem
				       caused by computer precision when
				       Vds is very close to Vdseff. */
		  diffVds = Vds - Vdseff;
		  here->B3SOIPDVdseff = Vdseff; /* v2.2.3 bug fix */

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

                  if (selfheat)
                  {

                     if (Rds!=0.0)
		        tmp4 = dRds_dT / Rds + dvsattemp_dT / vsattemp;
                     else tmp4 = dvsattemp_dT / vsattemp;      
 
		     dT0_dT  = dEsatL_dT + dVdsat_dT + T7 * tmp4 * Vgsteff
			     - T8 * (Abulk * dVdsat_dT - Abulk * Vdsat * dVgst2Vtm_dT
			     / Vgst2Vtm);
                  } else
                     dT0_dT = 0.0;

		  T9 = WVCoxRds * Abulk; 
		  T1 = 2.0 / Lambda - 1.0 + T9; 
		  dT1_dVg = -2.0 * tmp1 +  WVCoxRds * (Abulk * tmp2 + dAbulk_dVg);
		  dT1_dVb = dAbulk_dVb * WVCoxRds + T9 * tmp3;
                  if (selfheat)
		     dT1_dT  = T9 * tmp4;
                  else
		     dT1_dT  = 0.0;

		  Vasat = T0 / T1;
		  dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) / T1;
		  dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) / T1;
		  dVasat_dVd = dT0_dVd / T1;
                  if (selfheat) dVasat_dT  = (dT0_dT  - Vasat * dT1_dT)  / T1;
                  else dVasat_dT  = 0.0;

/* Calculate VACLM */
		  if ((pParam->B3SOIPDpclm > 0.0) && (diffVds > 1.0e-10))
		  {   T0 = 1.0 / (pParam->B3SOIPDpclm * Abulk * pParam->B3SOIPDlitl);
		      dT0_dVb = -T0 / Abulk * dAbulk_dVb;
		      dT0_dVg = -T0 / Abulk * dAbulk_dVg; 
		      
		      T2 = Vgsteff / EsatL;
		      T1 = Leff * (Abulk + T2); 
		      dT1_dVg = Leff * ((1.0 - T2 * dEsatL_dVg) / EsatL + dAbulk_dVg);
		      dT1_dVb = Leff * (dAbulk_dVb - T2 * dEsatL_dVb / EsatL);
		      dT1_dVd = -T2 * dEsatL_dVd / Esat;
                      if (selfheat) dT1_dT  = -T2 * dEsatL_dT / Esat;
                      else dT1_dT  = 0.0;

		      T9 = T0 * T1;
		      VACLM = T9 * diffVds;
		      dVACLM_dVg = T0 * dT1_dVg * diffVds - T9 * dVdseff_dVg
				 + T1 * diffVds * dT0_dVg;
		      dVACLM_dVb = (dT0_dVb * T1 + T0 * dT1_dVb) * diffVds
				 - T9 * dVdseff_dVb;
		      dVACLM_dVd = T0 * dT1_dVd * diffVds + T9 * (1.0 - dVdseff_dVd);
                      if (selfheat)
		         dVACLM_dT  = T0 * dT1_dT * diffVds - T9 * dVdseff_dT;
                      else dVACLM_dT  = 0.0;

		  }
		  else
		  {   VACLM = MAX_EXPL;
		      dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = dVACLM_dT = 0.0;
		  }


/* Calculate VADIBL */
		  if (pParam->B3SOIPDthetaRout > 0.0)
		  {   T8 = Abulk * Vdsat;
		      T0 = Vgst2Vtm * T8;
		      T1 = Vgst2Vtm + T8;
		      dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8
			      + Vgst2Vtm * Vdsat * dAbulk_dVg;
		      dT1_dVg = 1.0 + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
		      dT1_dVb = dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb;
		      dT0_dVb = Vgst2Vtm * dT1_dVb;
		      dT1_dVd = Abulk * dVdsat_dVd;
		      dT0_dVd = Vgst2Vtm * dT1_dVd;
                      if (selfheat)
                      {
		         dT0_dT  = dVgst2Vtm_dT * T8 + Abulk * Vgst2Vtm * dVdsat_dT;
		         dT1_dT  = dVgst2Vtm_dT + Abulk * dVdsat_dT;
                      } else
                         dT0_dT = dT1_dT = 0.0;

		      T9 = T1 * T1;
		      T2 = pParam->B3SOIPDthetaRout;
		      VADIBL = (Vgst2Vtm - T0 / T1) / T2;
		      dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2;
		      dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2;
		      dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2;
                      if (selfheat)
		         dVADIBL_dT = (dVgst2Vtm_dT - dT0_dT/T1 + T0*dT1_dT/T9) / T2;
                      else dVADIBL_dT = 0.0;

		      T7 = pParam->B3SOIPDpdiblb * Vbseff;
		      if (T7 >= -0.9)
		      {   T3 = 1.0 / (1.0 + T7);
			  VADIBL *= T3;
			  dVADIBL_dVg *= T3;
			  dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam->B3SOIPDpdiblb)
				      * T3;
			  dVADIBL_dVd *= T3;
			  if (selfheat)  dVADIBL_dT  *= T3;
			  else  dVADIBL_dT  = 0.0;
		      }
		      else
/* Added to avoid the discontinuity problem caused by pdiblcb */
		      {   T4 = 1.0 / (0.8 + T7);
			  T3 = (17.0 + 20.0 * T7) * T4;
			  dVADIBL_dVg *= T3;
			  dVADIBL_dVb = dVADIBL_dVb * T3
				      - VADIBL * pParam->B3SOIPDpdiblb * T4 * T4;
			  dVADIBL_dVd *= T3;
			  if (selfheat)  dVADIBL_dT  *= T3;
			  else  dVADIBL_dT  = 0.0;
			  VADIBL *= T3;
		      }
		  }
		  else
		  {   VADIBL = MAX_EXPL;
		      dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = dVADIBL_dT = 0.0;
		  }

/* Calculate VA */
		  
		  T8 = pParam->B3SOIPDpvag / EsatL;
		  T9 = T8 * Vgsteff;
		  if (T9 > -0.9)
		  {   T0 = 1.0 + T9;
		      dT0_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL);
		      dT0_dVb = -T9 * dEsatL_dVb / EsatL;
		      dT0_dVd = -T9 * dEsatL_dVd / EsatL;
                      if (selfheat)
		         dT0_dT  = -T9 * dEsatL_dT / EsatL;
                      else
		         dT0_dT  = 0.0;
		  }
		  else /* Added to avoid the discontinuity problems caused by pvag */
		  {   T1 = 1.0 / (17.0 + 20.0 * T9);
		      T0 = (0.8 + T9) * T1;
		      T1 *= T1;
		      dT0_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T1;

		      T9 *= T1 / EsatL;
		      dT0_dVb = -T9 * dEsatL_dVb;
		      dT0_dVd = -T9 * dEsatL_dVd;
                      if (selfheat)
		         dT0_dT  = -T9 * dEsatL_dT;
                      else
		         dT0_dT  = 0.0;
		  }
		
		  tmp1 = VACLM * VACLM;
		  tmp2 = VADIBL * VADIBL;
		  tmp3 = VACLM + VADIBL;

		  T1 = VACLM * VADIBL / tmp3;
		  tmp3 *= tmp3;
		  dT1_dVg = (tmp1 * dVADIBL_dVg + tmp2 * dVACLM_dVg) / tmp3;
		  dT1_dVd = (tmp1 * dVADIBL_dVd + tmp2 * dVACLM_dVd) / tmp3;
		  dT1_dVb = (tmp1 * dVADIBL_dVb + tmp2 * dVACLM_dVb) / tmp3;
                  if (selfheat)
		     dT1_dT  = (tmp1 * dVADIBL_dT  + tmp2 * dVACLM_dT ) / tmp3;
                  else dT1_dT  = 0.0;

		  Va = Vasat + T0 * T1;
		  dVa_dVg = dVasat_dVg + T1 * dT0_dVg + T0 * dT1_dVg;
		  dVa_dVd = dVasat_dVd + T1 * dT0_dVd + T0 * dT1_dVd;
		  dVa_dVb = dVasat_dVb + T1 * dT0_dVb + T0 * dT1_dVb;
                  if (selfheat)
		     dVa_dT  = dVasat_dT  + T1 * dT0_dT  + T0 * dT1_dT;
                  else dVa_dT  = 0.0;

/* Calculate Ids */
		  CoxWovL = model->B3SOIPDcox * Weff / Leff;
		  beta = ueff * CoxWovL;
		  dbeta_dVg = CoxWovL * dueff_dVg + beta * dWeff_dVg / Weff ;
		  dbeta_dVd = CoxWovL * dueff_dVd;
		  dbeta_dVb = CoxWovL * dueff_dVb + beta * dWeff_dVb / Weff ;
		  if (selfheat)  dbeta_dT  = CoxWovL * dueff_dT;
		  else  dbeta_dT  = 0.0;

		  T0 = 1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm;
		  dT0_dVg = -0.5 * (Abulk * dVdseff_dVg 
			  - Abulk * Vdseff / Vgst2Vtm + Vdseff * dAbulk_dVg) / Vgst2Vtm;
		  dT0_dVd = -0.5 * Abulk * dVdseff_dVd / Vgst2Vtm;
		  dT0_dVb = -0.5 * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff)
			  / Vgst2Vtm;
		  if (selfheat)  
                     dT0_dT  = -0.5 * (Abulk * dVdseff_dT
                             - Abulk * Vdseff / Vgst2Vtm * dVgst2Vtm_dT)
			     / Vgst2Vtm;
		  else dT0_dT = 0.0;

		  fgche1 = Vgsteff * T0;
		  dfgche1_dVg = Vgsteff * dT0_dVg + T0; 
		  dfgche1_dVd = Vgsteff * dT0_dVd;
		  dfgche1_dVb = Vgsteff * dT0_dVb;
		  if (selfheat)  dfgche1_dT  = Vgsteff * dT0_dT;
		  else  dfgche1_dT  = 0.0;

		  T9 = Vdseff / EsatL;
		  fgche2 = 1.0 + T9;
		  dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) / EsatL;
		  dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) / EsatL;
		  dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) / EsatL;
		  if (selfheat)  dfgche2_dT  = (dVdseff_dT  - T9 * dEsatL_dT)  / EsatL;
		  else  dfgche2_dT  = 0.0;

		  gche = beta * fgche1 / fgche2;
		  dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg
			    - gche * dfgche2_dVg) / fgche2;
		  dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd
			    - gche * dfgche2_dVd) / fgche2;
		  dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb
			    - gche * dfgche2_dVb) / fgche2;
		  if (selfheat)
		     dgche_dT  = (beta * dfgche1_dT  + fgche1 * dbeta_dT
			       - gche * dfgche2_dT)  / fgche2;
		  else dgche_dT  = 0.0;

		  T0 = 1.0 + gche * Rds;
		  T9 = Vdseff / T0;
		  Idl = gche * T9;

/*  Whoa, these formulas for the derivatives of Idl are convoluted, but I
    verified them to be correct  */

		  dIdl_dVg = (gche * dVdseff_dVg + T9 * dgche_dVg) / T0
			   - Idl * gche / T0 * dRds_dVg ; 
		  dIdl_dVd = (gche * dVdseff_dVd + T9 * dgche_dVd) / T0; 
		  dIdl_dVb = (gche * dVdseff_dVb + T9 * dgche_dVb 
			   - Idl * dRds_dVb * gche) / T0; 
		  if (selfheat)
		     dIdl_dT  = (gche * dVdseff_dT + T9 * dgche_dT
			      - Idl * dRds_dT * gche) / T0;
		  else dIdl_dT  = 0.0;

		  T9 =  diffVds / Va;
		  T0 =  1.0 + T9;
		  here->B3SOIPDids = Ids = Idl * T0 / here->B3SOIPDnseg;

		  Gm0 = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVa_dVg) / Va;
		  Gds0 = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd
			    - T9 * dVa_dVd) / Va;
		  Gmb0 = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVa_dVb) / Va;
                  Gmc = 0.0;
                  if (selfheat) 
		     GmT0 = T0 * dIdl_dT - Idl * (dVdseff_dT + T9 * dVa_dT) / Va;
                  else GmT0 = 0.0;

/* This includes all dependencies from Vgsteff, Vbseff */

		  Gm = Gm0 * dVgsteff_dVg / here->B3SOIPDnseg;
		  Gmb = (Gm0 * dVgsteff_dVb + Gmb0 * dVbseff_dVb) / here->B3SOIPDnseg;
		  Gds = (Gm0 * dVgsteff_dVd + Gds0) / here->B3SOIPDnseg;
		  if (selfheat)
		     GmT = (Gm0 * dVgsteff_dT + GmT0) / here->B3SOIPDnseg;
		  else GmT = 0.0;

/*  calculate GIDL current  */
		  T0 = 3 * model->B3SOIPDtox;
		  /* For drain side */
		  T1 = (Vds - Vgs_eff - pParam->B3SOIPDngidl) / T0;
		  if ((pParam->B3SOIPDagidl <= 0.0) || (pParam->B3SOIPDbgidl <= 0.0) ||
		      (T1 <= 0.0))
		  {   Idgidl = Gdgidld = Gdgidlg = 0.0;
		  }
		  else {
		     dT1_dVd = 1 / T0;
		     dT1_dVg = - dT1_dVd * dVgs_eff_dVg;
		     T2 = pParam->B3SOIPDbgidl / T1;
		     if (T2 < EXPL_THRESHOLD)
		     {
			Idgidl = wdiod * pParam->B3SOIPDagidl * T1 * exp(-T2);
			T3 = Idgidl / T1 * (T2 + 1);
			Gdgidld = T3 * dT1_dVd;
			Gdgidlg = T3 * dT1_dVg;
		     } else
		     {
			T3 = wdiod * pParam->B3SOIPDagidl * MIN_EXPL;
			Idgidl = T3 * T1 ;
			Gdgidld  = T3 * dT1_dVd;
			Gdgidlg  = T3 * dT1_dVg;
		     } 
		  }
                  here->B3SOIPDigidl = Idgidl;

		  /* For source side */
		  T1 = (- Vgs_eff - pParam->B3SOIPDngidl) / T0;
		  if ((pParam->B3SOIPDagidl <= 0.0) || (pParam->B3SOIPDbgidl <= 0.0) 
                        || (T1 <= 0.0))
		  {   Isgidl = Gsgidlg = 0;
		  }
		  else
		  {
		     dT1_dVg = - dVgs_eff_dVg / T0;
		     T2 = pParam->B3SOIPDbgidl / T1;
		     if (T2 < EXPL_THRESHOLD)
		     {
			Isgidl = wdios * pParam->B3SOIPDagidl * T1 * exp(-T2);
			T3 = Isgidl / T1 * (T2 + 1);
			Gsgidlg = T3 * dT1_dVg;
		     } else
		     {
			T3 = wdios * pParam->B3SOIPDagidl * MIN_EXPL;
			Isgidl = T3 * T1 ;
			Gsgidlg = T3 * dT1_dVg;
		     } 
		  }

/* calculate diode and BJT current */
                  WsTsi = wdios * model->B3SOIPDtsi;
                  WdTsi = wdiod * model->B3SOIPDtsi;

                  NVtm1 = Vtm * pParam->B3SOIPDndiode;
                  if (selfheat)
                     dNVtm1_dT = pParam->B3SOIPDndiode * dVtm_dT;
                  else 
                     dNVtm1_dT = 0;

                  T0 = Vbs / NVtm1;
                  dT0_dVb = 1.0 / NVtm1;
                  if (selfheat)
                     dT0_dT = -Vbs / NVtm1 / NVtm1 * dNVtm1_dT;
                  else
                     dT0_dT = 0;
                  DEXP(T0, ExpVbsNVtm, T1);
                  dExpVbsNVtm_dVb = T1 * dT0_dVb;
                  if (selfheat)
                     dExpVbsNVtm_dT = T1 * dT0_dT;
                  else
                     dExpVbsNVtm_dT = 0;

                  T0 = Vbd / NVtm1;
                  dT0_dVb = 1.0 / NVtm1;
                  dT0_dVd = -dT0_dVb;
                  if (selfheat)
                     dT0_dT = -Vbd / NVtm1 / NVtm1 * dNVtm1_dT;
                  else
                     dT0_dT = 0;
                  DEXP(T0, ExpVbdNVtm, T1);
                  dExpVbdNVtm_dVb = T1 * dT0_dVb;
                  dExpVbdNVtm_dVd = -dExpVbdNVtm_dVb;
                  if (selfheat)
                     dExpVbdNVtm_dT = T1 * dT0_dT;
                  else
                     dExpVbdNVtm_dT = 0;
           
		  /* Ibs1 / Ibd1 : diffusion current */
                  if (jdif == 0) {
                     Ibs1 = dIbs1_dVb = dIbs1_dT = Ibd1 = dIbd1_dVb = dIbd1_dVd = dIbd1_dT = 0;
                  } 
                  else {
                     T0 = WsTsi * jdif;
                     if (selfheat)
                        dT0_dT = WsTsi * djdif_dT;
                     else
                        dT0_dT = 0;
                     Ibs1 = T0 * (ExpVbsNVtm - 1);
                     dIbs1_dVb = T0 * dExpVbsNVtm_dVb;
                     if (selfheat)
                        dIbs1_dT = T0 * dExpVbsNVtm_dT + (ExpVbsNVtm - 1) * dT0_dT;
                     else  
                        dIbs1_dT = 0;

                     T0 = WdTsi * jdif;
                     if (selfheat)
                        dT0_dT = WdTsi * djdif_dT;
                     else
                        dT0_dT = 0;
                     Ibd1 = T0 * (ExpVbdNVtm - 1);
                     dIbd1_dVb = T0 * dExpVbdNVtm_dVb;
                     dIbd1_dVd = -dIbd1_dVb;
                     if (selfheat)
                        dIbd1_dT = T0 * dExpVbdNVtm_dT + (ExpVbdNVtm -1) * dT0_dT;
                     else  
                        dIbd1_dT = 0;
                  }

		  /* Ibs2:recombination/trap-assisted tunneling current */
                  NVtmf = 0.026 * pParam->B3SOIPDnrecf0 
                        * (1 + model->B3SOIPDntrecf * (TempRatio - 1));
                  NVtmr = 0.026 * pParam->B3SOIPDnrecr0 /* v2.2.2 bug fix */
                        * (1 + model->B3SOIPDntrecr * (TempRatio - 1));
                  if (selfheat) {
                     dNVtmf_dT = pParam->B3SOIPDnrecf0 * 0.026 
                               * model->B3SOIPDntrecf * dTempRatio_dT;
                     dNVtmr_dT = pParam->B3SOIPDnrecr0 * 0.026 /* v2.2.2 bug fix */
                               * model->B3SOIPDntrecr * dTempRatio_dT;
                  }
                  else  
                     dNVtmf_dT = dNVtmr_dT = 0;

                  if (jrec == 0) {
                     Ibs2 = dIbs2_dVb = dIbs2_dT = 0;
                     Ibd2 = dIbd2_dVb = dIbd2_dVd = dIbd2_dT = 0;
                  } 
                  else {
                     /* forward bias */
                     T0 = Vbs / NVtmf;
                     DEXP(T0,T10,T2);
                     T4 = 1 / NVtmf;
                     dT10_dVb = T4 * T2;
                     if (selfheat)
                        dT10_dT  = - T4 * T2 * Vbs / NVtmf * dNVtmf_dT ;
                     else   dT10_dT  = 0.0;

                     /* reverse bias */
                     if ((pParam->B3SOIPDvrec0 - Vbs) < 1e-3) {

                     /* v2.2.3 bug fix */
                        T1 = 1e3;
                        T0 = -Vbs / NVtmr * pParam->B3SOIPDvrec0 * T1; 
                        T11 = -exp(T0);

                        dT11_dVb = dT11_dT = 0;
                     }
                     else {
                        T1 = 1 / (pParam->B3SOIPDvrec0 - Vbs);
                        T0 = -Vbs / NVtmr * pParam->B3SOIPDvrec0 * T1;
                        dT0_dVb = -pParam->B3SOIPDvrec0 / NVtmr * (T1 + Vbs * T1 * T1) ;
                        if (selfheat)
                           dT0_dT = -T0 / NVtmr * dNVtmr_dT;
                        else   dT0_dT = 0;

                        DEXP(T0, T11, T2);
                        T11 = -T11;
                        dT11_dVb = -T2 * dT0_dVb;
                        if (selfheat)
                           dT11_dT = -T2 * dT0_dT;
                        else   dT11_dT = 0;
                     }
                     T3 = WsTsi * jrec;
                     Ibs2 = T3 * (T10 + T11);
                     dIbs2_dVb = T3 * (dT10_dVb + dT11_dVb);
                     if (selfheat)
                        dIbs2_dT = T3 * (dT10_dT + dT11_dT) + WsTsi * (T10 + T11) * djrec_dT;
                     else   dIbs2_dT = 0;

                     /* Ibd2 */
                     T0 = Vbd / NVtmf;
                     DEXP(T0,T10,T2);
                     T4 = 1 / NVtmf;
                     dT10_dVb = T4 * T2;
                     if (selfheat)
                        dT10_dT  = - T4 * T2 * Vbd / NVtmf * dNVtmf_dT ;
                     else   dT10_dT  = 0.0;

                     if ((pParam->B3SOIPDvrec0 - Vbd) < 1e-3) {

                     /* v2.2.3 bug fix */
                        T1 = 1e3;
                        T0 = -Vbd / NVtmr * pParam->B3SOIPDvrec0 * T1;
                        T11 = -exp(T0);

                        dT11_dVb = dT11_dT = 0;
                     }
                     else {
                        T1 = 1 / (pParam->B3SOIPDvrec0 - Vbd);
                        T0 = -Vbd / NVtmr * pParam->B3SOIPDvrec0 * T1;
                        dT0_dVb = -pParam->B3SOIPDvrec0 / NVtmr * (T1 + Vbd * T1 * T1) ;
                        if (selfheat)
                           dT0_dT = -T0 / NVtmr * dNVtmr_dT;
                        else
                           dT0_dT = 0;
                        DEXP(T0, T11, T2);
                        T11 = - T11;
                        dT11_dVb = -T2 * dT0_dVb;
                        if (selfheat)
                           dT11_dT = -T2 * dT0_dT;
                        else
                           dT11_dT = 0;
                     }
                     T3 = WdTsi * jrec;
                     Ibd2 = T3 * (T10 + T11);
                     dIbd2_dVb = T3 * (dT10_dVb + dT11_dVb);
                     dIbd2_dVd = -dIbd2_dVb;
                     if (selfheat)
                        dIbd2_dT = T3 * (dT10_dT + dT11_dT) + WdTsi * (T10 + T11) * djrec_dT;
                     else
                        dIbd2_dT = 0;
                  }

                  /* Ibs3/Ibd3:  recombination current in neutral body */
                  WTsi = pParam->B3SOIPDweff / here->B3SOIPDnseg * model->B3SOIPDtsi;
                  if (jbjt == 0.0)
		  {  
		     Ibs3 = dIbs3_dVb = dIbs3_dVd = dIbs3_dT = 0.0;
		     Ibd3 = dIbd3_dVb = dIbd3_dVd = dIbd3_dT = 0.0;
                     Ibsdif = dIbsdif_dVb = dIbsdif_dT = 0;
                     Ibddif = dIbddif_dVb = dIbddif_dVd = dIbddif_dT = 0;
		     here->B3SOIPDic = Ic = Gcd = Gcb = GcT = 0.0;
		  } 
		  else {
                     Ien = WTsi * jbjt * pParam->B3SOIPDlratio;
                     if (selfheat)
                        dIen_dT = WTsi * djbjt_dT * pParam->B3SOIPDlratio;
                     else
                        dIen_dT = 0;

                     /* high level injection of source side */
                     if ((Ehlis = Ahli * (ExpVbsNVtm - 1)) < 1e-5) {
                        Ehlis = dEhlis_dVb = dEhlis_dT = 0;
                        EhlisFactor = 1;
                        dEhlisFactor_dVb = dEhlisFactor_dT = 0;
                     }
                     else {
                        dEhlis_dVb = Ahli * dExpVbsNVtm_dVb;
                        if (selfheat)
                           dEhlis_dT = Ahli * dExpVbsNVtm_dT + (ExpVbsNVtm - 1) * dAhli_dT;
                        else
                           dEhlis_dT = 0;
                        EhlisFactor = 1.0 / sqrt(1 + Ehlis);
                        T0 = -0.5 * EhlisFactor / (1 + Ehlis);
                        dEhlisFactor_dVb = T0 * dEhlis_dVb;
                        if (selfheat)
                           dEhlisFactor_dT = T0 * dEhlis_dT;
                        else
                           dEhlisFactor_dT = 0;
                     }
                       
                     /* high level injection of drain side */
                     if ((Ehlid = Ahli * (ExpVbdNVtm - 1)) < 1e-5) {
                        Ehlid = dEhlid_dVb = dEhlid_dVd = dEhlid_dT = 0;
                        EhlidFactor = 1;
                        dEhlidFactor_dVb = dEhlidFactor_dVd = dEhlidFactor_dT = 0;
                     }
                     else {
                        dEhlid_dVb = Ahli * dExpVbdNVtm_dVb;
                        dEhlid_dVd = -dEhlid_dVb;
                        if (selfheat)
                           dEhlid_dT = Ahli * dExpVbdNVtm_dT + (ExpVbdNVtm - 1) * dAhli_dT;
                        else
                           dEhlid_dT = 0;
                        EhlidFactor = 1.0 / sqrt(1 + Ehlid);
                        T0 = -0.5 * EhlidFactor / (1 + Ehlid);
                        dEhlidFactor_dVb = T0 * dEhlid_dVb;
                        dEhlidFactor_dVd = -dEhlidFactor_dVb;
                        if (selfheat)
                           dEhlidFactor_dT = T0 * dEhlid_dT;
                        else
                           dEhlidFactor_dT = 0;
                     }
                   
                     if ((T0 = (1 - pParam->B3SOIPDarfabjt)) < 1e-2) {
                        Ibs3 = dIbs3_dVb = dIbs3_dT = 0;

                        dIbs3_dVd = 0;

                        Ibd3 = dIbd3_dVb = dIbd3_dVd = dIbd3_dT = 0;
                     }
                     else {
                        T1 = T0 * Ien;
                        if (selfheat)
                           dT1_dT = T0 * dIen_dT;
                        else
                           dT1_dT = 0;

                        Ibs3 = T1 * (ExpVbsNVtm - 1) * EhlisFactor;
                        dIbs3_dVb = T1 * (dExpVbsNVtm_dVb * EhlisFactor
                                  + (ExpVbsNVtm - 1) * dEhlisFactor_dVb);
                        dIbs3_dVd = 0;
                        if (selfheat)
                           dIbs3_dT = dT1_dT * (ExpVbsNVtm - 1) * EhlisFactor
                                    + T1 * (dExpVbsNVtm_dT * EhlisFactor
                                    + (ExpVbsNVtm - 1) * dEhlisFactor_dT);
                        else
                           dIbs3_dT = 0.0;

                        Ibd3 = T1 * (ExpVbdNVtm - 1) * EhlidFactor;
                        dIbd3_dVb = T1 * (dExpVbdNVtm_dVb * EhlidFactor
                                  + (ExpVbdNVtm - 1) * dEhlidFactor_dVb);
                        dIbd3_dVd = -dIbd3_dVb;
                        if (selfheat)
                           dIbd3_dT = dT1_dT * (ExpVbdNVtm - 1) * EhlidFactor
                                    + T1 * (dExpVbdNVtm_dT * EhlidFactor
                                    + (ExpVbdNVtm - 1) * dEhlidFactor_dT);
                        else
                           dIbd3_dT = 0.0;
                     }

                     /* effective diffusion current for capacitance calcu. */
                     Iendif = WTsi * jbjt * pParam->B3SOIPDlratiodif;
                     if (selfheat)
                        dIendif_dT = WTsi * djbjt_dT * pParam->B3SOIPDlratiodif;
                     else
                        dIendif_dT = 0;

                     Ibsdif = Iendif * (ExpVbsNVtm - 1) * EhlisFactor;
                     dIbsdif_dVb = Iendif * (dExpVbsNVtm_dVb * EhlisFactor
                                 + (ExpVbsNVtm - 1) * dEhlisFactor_dVb);
                     if (selfheat)
                        dIbsdif_dT = dIendif_dT * (ExpVbsNVtm - 1) * EhlisFactor
                                   + Iendif * (dExpVbsNVtm_dT * EhlisFactor
                                   + (ExpVbsNVtm - 1) * dEhlisFactor_dT);
                     else
                        dIbsdif_dT = 0;

                     Ibddif = Iendif * (ExpVbdNVtm - 1) * EhlidFactor;
                     dIbddif_dVb = Iendif * (dExpVbdNVtm_dVb * EhlidFactor
                                 + (ExpVbdNVtm - 1) * dEhlidFactor_dVb);
                     dIbddif_dVd = -dIbddif_dVb;
                     if (selfheat)
                        dIbddif_dT = dIendif_dT * (ExpVbdNVtm - 1) * EhlidFactor
                                   + Iendif * (dExpVbdNVtm_dT * EhlidFactor
                                   + (ExpVbdNVtm - 1) * dEhlidFactor_dT);
                     else
                        dIbddif_dT = 0;

                     /* Ic: Bjt collector current */
                     if ((here->B3SOIPDbjtoff == 1) || (Vds == 0.0)) {
                        here->B3SOIPDic = Ic = Gcd = Gcb = GcT = 0.0;
                     }
                     else {
                        /* second order effects */
                        T0 = 1 + (Vbs + Vbd) / pParam->B3SOIPDvearly;
                        dT0_dVb = 2.0 / pParam->B3SOIPDvearly;
                        dT0_dVd = -1.0 / pParam->B3SOIPDvearly;

                        T1 = Ehlis + Ehlid; 
                        dT1_dVb = dEhlis_dVb + dEhlid_dVb;
                        dT1_dVd = dEhlid_dVd;
                        if (selfheat)
                           dT1_dT = dEhlis_dT + dEhlid_dT;
                        else 
                           dT1_dT = 0;

                        T3 = sqrt(T0 * T0 + 4 * T1);
                        dT3_dVb = 0.5 / T3 * (2 * T0 * dT0_dVb + 4 * dT1_dVb);
                        dT3_dVd = 0.5 / T3 * (2 * T0 * dT0_dVd + 4 * dT1_dVd);
                        if (selfheat)
                           dT3_dT = 2 * dT1_dT / T3;
                        else
                           dT3_dT = 0;

                        T2 = (T0 + T3) / 2.0;
                        dT2_dVb = (dT0_dVb + dT3_dVb) / 2.0;
                        dT2_dVd = (dT0_dVd + dT3_dVd) / 2.0;
                        if (selfheat) 
                           dT2_dT = dT3_dT /2.0;
                        else
                           dT2_dT = 0;

                        if (T2 < .1) 
                        {
                           E2ndFactor = 10.0;
                           dE2ndFactor_dVb = dE2ndFactor_dVd = dE2ndFactor_dT = 0;
                        } 

                        else {
                           E2ndFactor = 1.0 / T2;
                           dE2ndFactor_dVb = -E2ndFactor / T2 * dT2_dVb;
                           dE2ndFactor_dVd = -E2ndFactor / T2 * dT2_dVd;
                           if (selfheat)
                              dE2ndFactor_dT = -E2ndFactor / T2 * dT2_dT;
                           else 
                              dE2ndFactor_dT = 0;
                        }

                        T0 = pParam->B3SOIPDarfabjt * Ien;
                        if (selfheat)
                           dT0_dT = pParam->B3SOIPDarfabjt * dIen_dT;
                        else
                           dT0_dT = 0;
                        here->B3SOIPDic = Ic 
                                      = T0 * (ExpVbsNVtm - ExpVbdNVtm) * E2ndFactor;
                        Gcb = dIc_dVb 
                            = T0 * ((dExpVbsNVtm_dVb - dExpVbdNVtm_dVb) * E2ndFactor 
                            + (ExpVbsNVtm - ExpVbdNVtm) * dE2ndFactor_dVb);
                        Gcd = dIc_dVd 
                            = T0 * (-dExpVbdNVtm_dVd * E2ndFactor 
                            + (ExpVbsNVtm - ExpVbdNVtm) * dE2ndFactor_dVd);
                        if (selfheat) 
                           GcT = T0 * (dExpVbsNVtm_dT - dExpVbdNVtm_dT) * E2ndFactor 
                               + dT0_dT * (ExpVbsNVtm - ExpVbdNVtm) * E2ndFactor
                               + T0 * (ExpVbsNVtm - ExpVbdNVtm) * dE2ndFactor_dT;
                        else
                           GcT = 0;
                     }
                  }

                  /* Ibs4/Ibd4 : tunneling */
                  NVtm2 = 0.026 * pParam->B3SOIPDntun;
                  if (jtun == 0)
                  {  Ibs4 = Ibd4 = dIbs4_dVb = dIbs4_dT = dIbd4_dVb = dIbd4_dVd = dIbd4_dT = 0;
                  } else
                  {
                     if ((pParam->B3SOIPDvtun0 - Vbs) < 1e-3)
                     {
                     /* v2.2.3 bug fix */
                        T1=1e3;
                        T0 = -Vbs / NVtm2 * pParam->B3SOIPDvtun0 * T1;
                        T1 = exp(T0);
                        T3 = WsTsi * jtun;
                        Ibs4 = T3 * (1- T1);

                        dIbs4_dVb = dIbs4_dT = 0;
                     }
                     else {
                        T1 = 1 / (pParam->B3SOIPDvtun0 - Vbs);
                        T0 = -Vbs / NVtm2 * pParam->B3SOIPDvtun0 * T1;
                        dT0_dVb = -pParam->B3SOIPDvtun0 / NVtm2 * (T1 + Vbs * T1 * T1) ;

                        DEXP(T0, T1, T2);
                        T3 = WsTsi * jtun;
                        Ibs4 =  T3 * (1- T1);
                        dIbs4_dVb = -T3 * T2 * dT0_dVb;
                        if (selfheat)
                           dIbs4_dT = (1 - T1) * WsTsi * djtun_dT;
                        else   dIbs4_dT = 0;
                     }

                     if ((pParam->B3SOIPDvtun0 - Vbd) < 1e-3) {

                     /* v2.2.3 bug fix */
                        T1=1e3;
                        T0 = -Vbd / NVtm2 * pParam->B3SOIPDvtun0 * T1;
                        T1 = exp(T0);
                        T3 = WdTsi * jtun;
                        Ibd4 = T3 * (1- T1);

                        dIbd4_dVb = dIbd4_dT = 0;
                        dIbd4_dVd = 0;
              
                     }
                     else {
                        T1 = 1 / (pParam->B3SOIPDvtun0 - Vbd);
                        T0 = -Vbd / NVtm2 * pParam->B3SOIPDvtun0 * T1;
                        dT0_dVb = -pParam->B3SOIPDvtun0 / NVtm2 * (T1 + Vbd * T1 * T1) ;

                        DEXP(T0, T1, T2);
                        T3 = WdTsi * jtun;
                        Ibd4 =  T3 * (1- T1);
                        dIbd4_dVb = -T3 * T2 * dT0_dVb;

                        dIbd4_dVd = -dIbd4_dVb;

                        if (selfheat)
                           dIbd4_dT = (1 - T1) * WdTsi * djtun_dT;
                        else   dIbd4_dT = 0;
                     }
                  }

                  here->B3SOIPDitun = - Ibd3 - Ibd4;
		  here->B3SOIPDibs = Ibs = Ibs1 + Ibs2 + Ibs3 + Ibs4;
		  here->B3SOIPDibd = Ibd = Ibd1 + Ibd2 + Ibd3 + Ibd4;

		  Gjsb = dIbs1_dVb + dIbs2_dVb + dIbs3_dVb + dIbs4_dVb;
		  Gjsd = dIbs3_dVd;
		  if (selfheat)  GjsT = dIbs1_dT + dIbs2_dT + dIbs3_dT + dIbs4_dT;
                  else   GjsT = 0.0;

		  Gjdb = dIbd1_dVb + dIbd2_dVb + dIbd3_dVb + dIbd4_dVb;
		  Gjdd = dIbd1_dVd + dIbd2_dVd + dIbd3_dVd + dIbd4_dVd;
                  if (selfheat)  GjdT = dIbd1_dT  + dIbd2_dT + dIbd3_dT + dIbd4_dT;
                  else   GjdT = 0.0;


/* v2.2: calculate gate-tunneling-to-body current */

                  if (model->B3SOIPDigMod >= 1) {
                     Vgb = Vgs_eff - Vbs;
                     dVgb_dVg = dVgs_eff_dVg;
                     dVgb_dVb = -1;

                     /* Calculate Vox first */
                     Vfb = model->B3SOIPDtype * pParam->B3SOIPDvth0 - phi - pParam->B3SOIPDk1eff * sqrtPhi;

                     T3 = Vfb - Vgs_eff + Vbs - DELTA_3;
                     dT3_dVg = -dVgs_eff_dVg;
                     dT3_dVd = 0;
                     dT3_dVb = 1;

                     if (Vfb <= 0.0) {
                        T0 = sqrt(T3 * T3 - 4.0 * DELTA_3 * Vfb);
                        dT0_dVg = 1.0/(2.0 * T0) * 2.0*T3 * dT3_dVg;
                        dT0_dVb = 0.5*(1.0/T0) * 2.0*T3 * dT3_dVb;
                     } 
                     else {
                        T0 = sqrt(T3 * T3 + 4.0 * DELTA_3 * Vfb);
                        dT0_dVg = 1.0/(2.0 * T0) * 2.0*T3 * dT3_dVg;
                        dT0_dVb = 0.5*(1.0/T0) * 2.0*T3 * dT3_dVb;
                     }

                     Vfbeff = Vfb - 0.5 * (T3 + T0);
                     dVfbeff_dVg = -0.5 * (dT3_dVg + dT0_dVg);
                     dVfbeff_dVb = -0.5 * (dT3_dVb + dT0_dVb);

                     Voxacc = Vfb - Vfbeff;
                     dVoxacc_dVg = -dVfbeff_dVg;
                     dVoxacc_dVd = 0.0;
                     dVoxacc_dVb = -dVfbeff_dVb;
                     if (Voxacc < 0.0)
                        Voxacc = dVoxacc_dVg = dVoxacc_dVb = 0.0;

                     T0 = Vgs_eff - Vgsteff - Vfbeff - Vbseff;
                     dT0_dVg = dVgs_eff_dVg - dVgsteff_dVg - dVfbeff_dVg;
                     dT0_dVd = -dVgsteff_dVd;
                     dT0_dVb = -dVgsteff_dVb - dVfbeff_dVb - dVbseff_dVb;
                     if (selfheat)
                        dT0_dT = -dVgsteff_dT;

                     if (pParam->B3SOIPDk1eff == 0.0) {
                        Voxdepinv = dVoxdepinv_dVg = dVoxdepinv_dVd = dVoxdepinv_dVb 
                                  = dVoxdepinv_dT = 0.0;
                     } else {
                        if (T0 < 0.0) { 
                           T1 = T0/pParam->B3SOIPDk1eff;
                           dT1_dVg = dT0_dVg/pParam->B3SOIPDk1eff;
                           dT1_dVd = dT0_dVd/pParam->B3SOIPDk1eff;
                           dT1_dVb = dT0_dVb/pParam->B3SOIPDk1eff;
                           if (selfheat) dT1_dT = dT0_dT/pParam->B3SOIPDk1eff;
                        }
                        else {
                           T1 = pParam->B3SOIPDk1eff/2*(-1 + sqrt(1 +
                                4*T0/pParam->B3SOIPDk1eff/pParam->B3SOIPDk1eff));
                           T2 = pParam->B3SOIPDk1eff/2 *
                                0.5/sqrt(1 + 4*T0/pParam->B3SOIPDk1eff/pParam->B3SOIPDk1eff) *
                                4/pParam->B3SOIPDk1eff/pParam->B3SOIPDk1eff;
                           dT1_dVg = T2 * dT0_dVg;
                           dT1_dVd = T2 * dT0_dVd;
                           dT1_dVb = T2 * dT0_dVb;
                           if (selfheat) 
                              dT1_dT = T2 * dT0_dT;
                         }

                        Voxdepinv = Vgs_eff - (T1*T1 + Vbs) - Vfb;
                        dVoxdepinv_dVg = dVgs_eff_dVg - (2.0*T1*dT1_dVg);
                        dVoxdepinv_dVd = -(2.0*T1*dT1_dVd);
                        dVoxdepinv_dVb = -(2.0*T1*dT1_dVb + 1);
                        if (selfheat)
                           dVoxdepinv_dT = -(2.0*T1*dT1_dT);
                     }


                     OxideRatio = pParam->B3SOIPDoxideRatio;

                     Vox = Voxdepinv;
                     /* Voxeff is Vox limited below Voxh */
                     T0 = model->B3SOIPDvoxh;
                     T1 = T0 - Vox - model->B3SOIPDdeltavox;
                     T3 = sqrt(T1 * T1 + 4*model->B3SOIPDdeltavox * T0);
                     Voxeff = T0 - 0.5 * (T1 + T3);
                     dVoxeff_dVox = 0.5 * (1.0 + T1 / T3);

                     Vox = Voxeff;
                     dVox_dVg = dVoxdepinv_dVg * dVoxeff_dVox;
                     dVox_dVd = dVoxdepinv_dVd * dVoxeff_dVox;
                     dVox_dVb = dVoxdepinv_dVb * dVoxeff_dVox;
                     dVox_dT = dVoxdepinv_dT * dVoxeff_dVox;


                     T0 = (Vox - model->B3SOIPDebg)/model->B3SOIPDvevb;
                     if (selfheat)
                        dT0_dT = dVox_dT /model->B3SOIPDvevb;

                     DEXP(T0, T1, T2); /* T1=exp(T0), T2=dT1_dT0 */
                     if (selfheat)
                        dT1_dT = T2 * dT0_dT;

                     Vaux = model->B3SOIPDvevb * log(1 + T1);
                     dVaux_dVg = T2 / (1 + T1) * dVox_dVg;
                     dVaux_dVd = T2 / (1 + T1) * dVox_dVd;
                     dVaux_dVb = T2 / (1 + T1) * dVox_dVb;
                     if (selfheat)
                        dVaux_dT = T2 / (1 + T1) * dVox_dT;

                     if (model->B3SOIPDvgb1 != 0) {
                        T0 = 1 - Vox / model->B3SOIPDvgb1;
                        dT0_dVox = -1.0/model->B3SOIPDvgb1;
                        if (selfheat)
                           dT0_dT = -dVox_dT / model->B3SOIPDvgb1;
                     } else {
                          T0 = 1;
                          dT0_dVox = dT0_dT = 0.0;
                       }

                     if (T0 < 0.01) {
                        T0 = 0.01;
                        dT0_dVox = dT0_dT = 0.0;
                     }

/* v2.2.3 bug fix */
                     T1 = Leff * Weff * 3.7622e-7 * OxideRatio / here->B3SOIPDnseg;

                     T2 = -3.1051e10 * model->B3SOIPDtoxqm;
                     T3 = model->B3SOIPDalphaGB1;
                     T4 = model->B3SOIPDbetaGB1;

                     T6 = T2*(T3 - T4 * Vox) / T0;
                     if (selfheat) dT6_dT = -T2 * T4 * dVox_dT / T0 - T6/T0 * dT0_dT;

                     DEXP(T6, T5, T7); /* T5=exp(T6), T7=dT5_dT6 */
                     dT5_dVg = -T7 * dVox_dVg * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     dT5_dVd = -T7 * dVox_dVd * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     dT5_dVb = -T7 * dVox_dVb * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     if (selfheat)
                        dT5_dT = T7 * dT6_dT;
          
                     Igb1 = T1 * Vgb * Vaux * T5;
                     dIgb1_dVg = T1 * (Vgb*Vaux*dT5_dVg + dVgb_dVg*Vaux*T5 +
                                 Vgb*T5*dVaux_dVg);
                     dIgb1_dVd = T1 * (Vgb*Vaux*dT5_dVd + Vgb*T5*dVaux_dVd);
                     dIgb1_dVb = T1 * (Vgb*Vaux*dT5_dVb + dVgb_dVb*Vaux*T5 +
                                 Vgb*T5*dVaux_dVb);
                     if (selfheat)
                        dIgb1_dT = T1 * Vgb * (Vaux*dT5_dT + T5*dVaux_dT);
                     else dIgb1_dT = 0.0;


                     Vox = Voxacc;
                     /* Voxeff is Vox limited below Voxh */
                     T0 = model->B3SOIPDvoxh;
                     T1 = T0 - Vox - model->B3SOIPDdeltavox;
                     T3 = sqrt(T1 * T1 + 4*model->B3SOIPDdeltavox * T0);
                     Voxeff = T0 - 0.5 * (T1 + T3);
                     dVoxeff_dVox = 0.5 * (1.0 + T1 / T3);

                     Vox = Voxeff;
                     dVox_dVg = dVoxacc_dVg * dVoxeff_dVox;
                     dVox_dVd = dVoxacc_dVd * dVoxeff_dVox;
                     dVox_dVb = dVoxacc_dVb * dVoxeff_dVox;
                     dVox_dT = 0;

                     T0 = (-Vgb+(Vfb))/model->B3SOIPDvecb;
                     if (selfheat)
                        dT0_dT = 0;

                     DEXP(T0, T1, T2); /* T1=exp(T0), T2=dT1_dT0 */
                     if (selfheat)
                        dT1_dT = 0;

                     Vaux = model->B3SOIPDvecb* log(1 + T1);
                     dVaux_dVg = -T2 / (1 + T1);
                     dVaux_dVd = 0;
                     dVaux_dVb = -dVaux_dVg;
                     if (selfheat)
                        dVaux_dT = 0;

                     if (model->B3SOIPDvgb2 != 0) {
                        T0 = 1 - Vox / model->B3SOIPDvgb2;
                        dT0_dVox = -1.0/model->B3SOIPDvgb2;
                        if (selfheat) dT0_dT = -dVox_dT / model->B3SOIPDvgb2;
                     } else {
                          T0 = 1;
                          dT0_dVox = dT0_dT =0.0;
                       }

                     if (T0 < 0.01) {
                        T0 = 0.01;
                        dT0_dVox = dT0_dT =0.0;
                     }

/* v2.2.3 bug fix */
                     T1 = Leff * Weff * 4.9758e-7  * OxideRatio / here->B3SOIPDnseg;

                     T2 = -2.357e10 * model->B3SOIPDtoxqm;
                     T3 = model->B3SOIPDalphaGB2;
                     T4 = model->B3SOIPDbetaGB2;

                     T6 = T2*(T3 - T4 * Vox) / T0;
                     if (selfheat) dT6_dT = -T2 * T4 * dVox_dT / T0 - T6/T0 * dT0_dT;

                     DEXP(T6, T5, T7); /* T5=exp(T6), T7=dT5_dT6 */
                     dT5_dVg = -T7 * dVox_dVg * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     dT5_dVd = -T7 * dVox_dVd * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     dT5_dVb = -T7 * dVox_dVb * T2 / T0 * (T4 + (T3 - T4 * Vox) / T0 * dT0_dVox);
                     if (selfheat)
                        dT5_dT = T7 * dT6_dT;

                     Igb2 = T1 * Vgb * Vaux * T5;
                     dIgb2_dVg = T1 * (Vgb*Vaux*dT5_dVg + dVgb_dVg*Vaux*T5 +
                                 Vgb*T5*dVaux_dVg);
                     dIgb2_dVd = T1 * (Vgb*Vaux*dT5_dVd + Vgb*T5*dVaux_dVd);
                     dIgb2_dVb = T1 * (Vgb*Vaux*dT5_dVb + dVgb_dVb*Vaux*T5 +
                                 Vgb*T5*dVaux_dVb);
                     if (selfheat)
                        dIgb2_dT = T1 * Vgb * (Vaux*dT5_dT + T5*dVaux_dT);
                     else dIgb2_dT = 0.0;
            

/* Igb1 dominates in inversion region, while Igb2 doninates in accumulation */
/* v2.2.3 bug fix for residue at low Vgb */
                     if (Vgb >= 0)
                     {
                        Igb = Igb1;
                        dIgb_dVg = dIgb1_dVg;
                        dIgb_dVd = dIgb1_dVd;
                        dIgb_dVb = dIgb1_dVb;
                        dIgb_dT = dIgb1_dT;
                     }
                     else
                     {
                        Igb = Igb2;
                        dIgb_dVg = dIgb2_dVg;
                        dIgb_dVd = dIgb2_dVd;
                        dIgb_dVb = dIgb2_dVb;
                        dIgb_dT = dIgb2_dT;
                     }


                 }
                 else {
                   Igb = 0.0;
                   dIgb_dVg = 0.0;
                   dIgb_dVd = 0.0;
                   dIgb_dVb = 0.0;
                   dIgb_dT = 0.0;
                 }

                 here->B3SOIPDig = Igb;
                 here->B3SOIPDgigg = dIgb_dVg;
                 here->B3SOIPDgigd = dIgb_dVd;
                 here->B3SOIPDgigb = dIgb_dVb;
                 here->B3SOIPDgigs = -(dIgb_dVg + dIgb_dVd + dIgb_dVb);
                 here->B3SOIPDgigT = dIgb_dT;
/* end of v2.2 gate current */



/* calculate substrate current Iii */

                  if (pParam->B3SOIPDalpha0 <= 0.0) {
                     Giig = Giib = Giid = GiiT = 0.0;
                     here->B3SOIPDiii = Iii = 0.0;
                  }
                  else {
                     Vdsatii0 = pParam->B3SOIPDvdsatii0 * (1 + model->B3SOIPDtii * (TempRatio-1.0))
                        - pParam->B3SOIPDlii / Leff;
                     if (selfheat) 
                        dVdsatii0_dT = pParam->B3SOIPDvdsatii0 * model->B3SOIPDtii * dTempRatio_dT;
                     else 
			dVdsatii0_dT = 0;

                     /* Calculate VgsStep */
                     T0 = pParam->B3SOIPDesatii * Leff;
                     T1 = pParam->B3SOIPDsii0 * T0 / (1.0 + T0);

                     T0 = 1 / (1 + pParam->B3SOIPDsii1 * Vgsteff);
                     if (selfheat)
                        dT0_dT = - pParam->B3SOIPDsii1 * T0 * T0 *dVgsteff_dT;
                     else
                        dT0_dT = 0;
                     T3 = T0 + pParam->B3SOIPDsii2;
                     T4 = Vgst * pParam->B3SOIPDsii1 * T0 * T0;
                     T2 = Vgst * T3;
                     dT2_dVg = T3 * dVgst_dVg - T4 * dVgsteff_dVg;
                     dT2_dVb = T3 * dVgst_dVb * dVbseff_dVb - T4 * dVgsteff_dVb;
                     dT2_dVd = T3 * dVgst_dVd - T4 * dVgsteff_dVd;
                     if (selfheat)
                        dT2_dT = -dVth_dT * T3 + Vgst * dT0_dT;
                     else dT2_dT = 0;

                     T3 = 1 / (1 + pParam->B3SOIPDsiid * Vds);
                     dT3_dVd = - pParam->B3SOIPDsiid * T3 * T3;

                     VgsStep = T1 * T2 * T3;
                     if (selfheat) 
                        dVgsStep_dT = T1 * T3 * dT2_dT;
                     else dVgsStep_dT = 0;
                     Vdsatii = Vdsatii0 + VgsStep;
                     Vdiff = Vds - Vdsatii;
                     dVdiff_dVg = - T1 * T3 * dT2_dVg;
                     dVdiff_dVb = - T1 * T3 * dT2_dVb;
                     dVdiff_dVd = 1.0 - T1 * (T3 * dT2_dVd + T2 * dT3_dVd);
                     if (selfheat)
                        dVdiff_dT  = -(dVdsatii0_dT + dVgsStep_dT);
                     else dVdiff_dT = 0;

                     T0 = pParam->B3SOIPDbeta2 + pParam->B3SOIPDbeta1 * Vdiff
                        + pParam->B3SOIPDbeta0 * Vdiff * Vdiff;
                     if (T0 < 1e-5)
                     {
                        T0 = 1e-5;
                        dT0_dVg = dT0_dVd = dT0_dVb = dT0_dT = 0.0;
                     } 
                     else
                     {
                        T1 = pParam->B3SOIPDbeta1 + 2 * pParam->B3SOIPDbeta0 * Vdiff;
                        dT0_dVg = T1 * dVdiff_dVg;
                        dT0_dVb = T1 * dVdiff_dVb;
                        dT0_dVd = T1 * dVdiff_dVd;
                        if (selfheat)  
                           dT0_dT = T1 * dVdiff_dT;
                        else  
                           dT0_dT = 0;
                     } 

                     if ((T0 < Vdiff / EXPL_THRESHOLD) && (Vdiff > 0.0)) {
                        Ratio = pParam->B3SOIPDalpha0 * MAX_EXPL;
                        dRatio_dVg = dRatio_dVb = dRatio_dVd = dRatio_dT = 0.0;
                     }
                     else if ((T0 < -Vdiff / EXPL_THRESHOLD) && (Vdiff < 0.0)) {
                        Ratio = pParam->B3SOIPDalpha0 * MIN_EXPL;
                        dRatio_dVg = dRatio_dVb = dRatio_dVd = dRatio_dT = 0.0;
                     }
                     else {
                        Ratio = pParam->B3SOIPDalpha0 * exp(Vdiff / T0);
                        T1 = Ratio / T0 / T0;
                        dRatio_dVg = T1 * (T0 * dVdiff_dVg - Vdiff * dT0_dVg);
                        dRatio_dVb = T1 * (T0 * dVdiff_dVb - Vdiff * dT0_dVb);
                        dRatio_dVd = T1 * (T0 * dVdiff_dVd - Vdiff * dT0_dVd);
                        if (selfheat)
                           dRatio_dT = T1 * (T0 * dVdiff_dT - Vdiff * dT0_dT);
                        else  
                           dRatio_dT = 0;
                     }

                     /* Avoid too high ratio */
                     if (Ratio > 10.0) {
                        Ratio = 10.0;
                        dRatio_dVg = dRatio_dVb = dRatio_dVd = dRatio_dT = 0.0;
                     }

                     T0 = Ids + pParam->B3SOIPDfbjtii * Ic;
                     here->B3SOIPDiii = Iii = Ratio * T0;
                     Giig = Ratio * Gm + T0 * dRatio_dVg;
                     Giib = Ratio * (Gmb + pParam->B3SOIPDfbjtii * Gcb) 
                          + T0 * dRatio_dVb;
                     Giid = Ratio * (Gds + pParam->B3SOIPDfbjtii * Gcd) 
                          + T0 * dRatio_dVd;

                     if (selfheat)
                        GiiT = Ratio * (GmT + pParam->B3SOIPDfbjtii * GcT) 
                             + T0 * dRatio_dT;
                     else
                        GiiT = 0.0;

                  }

	/* Current through body resistor */
		  /* Current going out is +ve */
		  if ((here->B3SOIPDbodyMod == 0) || (here->B3SOIPDbodyMod == 2)) 
                  {
		     Ibp = Gbpbs = Gbpps = 0.0;
		  }
		  else { /* here->B3SOIPDbodyMod == 1 */
                     if (pParam->B3SOIPDrbody < 1e-30)
                     {
                        if (here->B3SOIPDrbodyext <= 1e-30)
                           T0 = 1.0 / 1e-30;
                        else
                           T0 = 1.0 / here->B3SOIPDrbodyext;
                        Ibp = Vbp * T0;
                        Gbpbs = T0 * dVbp_dVb;
                        Gbpps = -T0 * dVbp_dVb;
                     } else
                     {
		   	Gbpbs = 1.0 / (pParam->B3SOIPDrbody + here->B3SOIPDrbodyext);
			Ibp = Vbp * Gbpbs;
			Gbpps = - Gbpbs;
                     }
		  }

		  here->B3SOIPDibp = Ibp;
		  here->B3SOIPDgbpbs = Gbpbs;
		  here->B3SOIPDgbpps = Gbpps;
		  here->B3SOIPDgbpT = 0.0;
		  here->B3SOIPDcbodcon = Ibp - (Gbpbs * Vbs + Gbpps * Vps);



		  /*  Current going out of drainprime node into the drain of device  */
		  /*  "node" means the SPICE circuit node  */

                  here->B3SOIPDcdrain = Ids + Ic;
                  here->B3SOIPDcd = Ids + Ic - Ibd + Iii + Idgidl;
                  here->B3SOIPDcb = Ibs + Ibd + Ibp - Iii - Idgidl - Isgidl - Igb;

   		  here->B3SOIPDgds = Gds + Gcd;
   		  here->B3SOIPDgm = Gm;
   		  here->B3SOIPDgmbs = Gmb + Gcb;
		  if (selfheat)
		      here->B3SOIPDgmT = GmT + GcT;
		  else
		      here->B3SOIPDgmT = 0.0;

                  /*  note that sign is switched because power flows out 
                      of device into the temperature node.  
                      Currently ommit self-heating due to bipolar current
                      because it can cause convergence problem*/

                  here->B3SOIPDgtempg = -Gm * Vds;
                  here->B3SOIPDgtempb = -Gmb * Vds;
                  here->B3SOIPDgtempT = -GmT * Vds;
                  here->B3SOIPDgtempd = -Gds * Vds - Ids;
		  here->B3SOIPDcth = - Ids * Vds - model->B3SOIPDtype * 
                                  (here->B3SOIPDgtempg * Vgs + here->B3SOIPDgtempb * Vbs 
                                 + here->B3SOIPDgtempd * Vds)
                                 - here->B3SOIPDgtempT * delTemp;

                  /*  Body current which flows into drainprime node from the drain of device  */

		  here->B3SOIPDgjdb = Gjdb - Giib;
		  here->B3SOIPDgjdd = Gjdd - (Giid + Gdgidld);
		  here->B3SOIPDgjdg = - (Giig + Gdgidlg);
		  if (selfheat) here->B3SOIPDgjdT = GjdT - GiiT;
		  else here->B3SOIPDgjdT = 0.0;
		  here->B3SOIPDcjd = Ibd - Iii - Idgidl 
				 - (here->B3SOIPDgjdb * Vbs + here->B3SOIPDgjdd * Vds
				 +  here->B3SOIPDgjdg * Vgs + here->B3SOIPDgjdT * delTemp);

		  /*  Body current which flows into sourceprime node from the source of device  */

   		  here->B3SOIPDgjsb = Gjsb;
   		  here->B3SOIPDgjsd = Gjsd;
		  here->B3SOIPDgjsg = - Gsgidlg;
		  if (selfheat) here->B3SOIPDgjsT = GjsT;
		  else here->B3SOIPDgjsT = 0.0;
		  here->B3SOIPDcjs = Ibs - Isgidl
				  - (here->B3SOIPDgjsb * Vbs + here->B3SOIPDgjsd * Vds
				  +  here->B3SOIPDgjsg * Vgs + here->B3SOIPDgjsT * delTemp);

		  /*  Current flowing into body node  */

		  here->B3SOIPDgbbs = Giib - Gjsb - Gjdb - Gbpbs;
		  here->B3SOIPDgbgs = Giig + Gdgidlg + Gsgidlg;
   		  here->B3SOIPDgbds = Giid + Gdgidld - Gjsd - Gjdd;
		  here->B3SOIPDgbps = - Gbpps;
		  if (selfheat) here->B3SOIPDgbT = GiiT - GjsT - GjdT;
		  else here->B3SOIPDgbT = 0.0;


		  here->B3SOIPDcbody = Iii + Idgidl + Isgidl - Ibs - Ibd - Ibp + Igb
				   - ( (here->B3SOIPDgbbs + dIgb_dVb) * Vbs 
                                     + (here->B3SOIPDgbgs + dIgb_dVg) * Vgs
				     + (here->B3SOIPDgbds + dIgb_dVd) * Vds 
                                     + here->B3SOIPDgbps * Vps
				     + (here->B3SOIPDgbT + dIgb_dT) * delTemp);


                  here->B3SOIPDcgate = Igb 
                          - (dIgb_dVb * Vbs + dIgb_dVg * Vgs + dIgb_dVd * Vds + dIgb_dT * delTemp);


	/* Calculate Qinv for Noise analysis */

		  T1 = Vgsteff * (1.0 - 0.5 * Abulk * Vdseff / Vgst2Vtm);
		  here->B3SOIPDqinv = -model->B3SOIPDcox * pParam->B3SOIPDweff * Leff * T1;


	/*  Begin CV (charge) model  */

		  if ((model->B3SOIPDxpart < 0) || (!ChargeComputationNeeded))
		  {   qgate  = qdrn = qsrc = qbody = qsub = 0.0; /* v2.2.3 bug fix */
		      here->B3SOIPDcggb = here->B3SOIPDcgsb = here->B3SOIPDcgdb = 0.0;
		      here->B3SOIPDcdgb = here->B3SOIPDcdsb = here->B3SOIPDcddb = 0.0;
		      here->B3SOIPDcbgb = here->B3SOIPDcbsb = here->B3SOIPDcbdb = 0.0;
		      goto finished;
		  }
		  else
		  {
		       CoxWL  = model->B3SOIPDcox * (pParam->B3SOIPDweffCV / here->B3SOIPDnseg
			      * pParam->B3SOIPDleffCV + here->B3SOIPDagbcp);
                       CoxWLb = model->B3SOIPDfbody * model->B3SOIPDcox 
                              * (pParam->B3SOIPDweffCV / here->B3SOIPDnseg
                              * pParam->B3SOIPDleffCVb + here->B3SOIPDagbcp);
 
                       /* By using this Vgsteff,cv, discontinuity in moderate
                          inversion charges can be avoid. */

		       if ((VgstNVt > -EXPL_THRESHOLD) && (VgstNVt < EXPL_THRESHOLD))
		       {   ExpVgst *= ExpVgst;
                           ExpVgst *= exp( -(pParam->B3SOIPDdelvt / (n * Vtm)));
			   Vgsteff = n * Vtm * log(1.0 + ExpVgst);
			   T0 = ExpVgst / (1.0 + ExpVgst);
			   T1 = -T0 * (dVth_dVb + Vgst / n * dn_dVb) + Vgsteff / n * dn_dVb;
			   dVgsteff_dVd = -T0 * (dVth_dVd + Vgst / n * dn_dVd)
					  + Vgsteff / n * dn_dVd;
			   dVgsteff_dVg = T0 * dVgs_eff_dVg;
                           dVgsteff_dVb = T1 * dVbseff_dVb;
                           if (selfheat)
			      dVgsteff_dT = -T0 * (dVth_dT + (Vgst - pParam->B3SOIPDdelvt) / Temp) 
                                          + Vgsteff / Temp;
                           else dVgsteff_dT  = 0.0;
		       } 

		       if (model->B3SOIPDcapMod == 2)
		       {   
                          Vfb = Vth - phi - pParam->B3SOIPDk1eff * sqrtPhis + pParam->B3SOIPDdelvt;
                          dVfb_dVb = dVth_dVb - pParam->B3SOIPDk1eff * dsqrtPhis_dVb;
                          dVfb_dVd = dVth_dVd;
                          dVfb_dT  = dVth_dT;

			  V3 = Vfb - Vgs_eff + Vbseff - DELTA_3_SOI;
			  if (Vfb <= 0.0)
			  {   T0 = sqrt(V3 * V3 - 4.0 * DELTA_3_SOI * Vfb);
			      T2 = -DELTA_3_SOI / T0;
			  }
			  else
			  {   T0 = sqrt(V3 * V3 + 4.0 * DELTA_3_SOI * Vfb);
			      T2 = DELTA_3_SOI / T0;
			  }

			  T1 = 0.5 * (1.0 + V3 / T0);
			  Vfbeff = Vfb - 0.5 * (V3 + T0);
			  dVfbeff_dVd = (1.0 - T1 - T2) * dVfb_dVd;
			  dVfbeff_dVb = (1.0 - T1 - T2) * dVfb_dVb - T1;
			  dVfbeff_dVrg = T1 * dVgs_eff_dVg;
			  if (selfheat) dVfbeff_dT = (1.0 - T1 - T2) * dVfb_dT;
                          else  dVfbeff_dT = 0.0;

			  Qac0 = CoxWLb * (Vfbeff - Vfb);
			  dQac0_dVrg = CoxWLb * dVfbeff_dVrg;
			  dQac0_dVd = CoxWLb * (dVfbeff_dVd - dVfb_dVd);
			  dQac0_dVb = CoxWLb * (dVfbeff_dVb - dVfb_dVb);
			  if (selfheat) dQac0_dT = CoxWLb * (dVfbeff_dT - dVfb_dT);
                          else  dQac0_dT = 0.0;

			  T0 = 0.5 * K1;
			  T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff;
			  if (pParam->B3SOIPDk1eff == 0.0)
			  {   T1 = 0.0;
			      T2 = 0.0;
			  }
			  else if (T3 < 0.0)
			  {   T1 = T0 + T3 / pParam->B3SOIPDk1eff;
			      T2 = CoxWLb;
			  }
			  else
			  {   T1 = sqrt(T0 * T0 + T3);
			      T2 = CoxWLb * T0 / T1;
			  }

			  Qsub0 = CoxWLb * K1 * (T1 - T0);
			  dQsub0_dVrg = T2 * (dVgs_eff_dVg - dVfbeff_dVrg);
			  dQsub0_dVg = -T2;
			  dQsub0_dVd = -T2 * dVfbeff_dVd;
			  dQsub0_dVb = -T2 * (dVfbeff_dVb + 1);
			  if (selfheat) dQsub0_dT  = -T2 * dVfbeff_dT;
                          else  dQsub0_dT = 0.0;

			  AbulkCV = Abulk0 * pParam->B3SOIPDabulkCVfactor;
			  dAbulkCV_dVb = pParam->B3SOIPDabulkCVfactor * dAbulk0_dVb;

			  VdsatCV = Vgsteff / AbulkCV;
			  dVdsatCV_dVg = 1.0 / AbulkCV;
			  dVdsatCV_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;

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
			  T5 = (6.0 * T0 * (4.0 * Vgsteff- T0) / (T1 * T1) - 0.5);
			  T6 = 12.0 * T2 * T2 * Vgsteff;

			  T7 = 1.0 - AbulkCV;
			  qbulk = CoxWLb * T7 * (0.5 * VdseffCV - T3); 
			  T4 = -T7 * (T4 - 1.0);
			  T5 = -T7 * T5;
			  T6 = -(T7 * T6 + (0.5 * VdseffCV - T3));

			  Cbg1 = CoxWLb * (T4 + T5 * dVdseffCV_dVg);
			  Cbd1 = CoxWLb * T5 * dVdseffCV_dVd ;
			  Cbb1 = CoxWLb * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb);

			  /* Total inversion charge */
			  T0 = AbulkCV * VdseffCV;
			  T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1e-20);
			  T2 = VdseffCV / T1;
			  T3 = T0 * T2;

			  T4 = (1.0 - 12.0 * T2 * T2 * AbulkCV);
			  T5 = (6.0 * T0 * (4.0 * Vgsteff - T0) / (T1 * T1) - 0.5);
			  T6 = 12.0 * T2 * T2 * Vgsteff;

			  qinv = CoxWL * (Vgsteff - 0.5 * VdseffCV + T3);
			  Cgg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
			  Cgd1 = CoxWL * T5 * dVdseffCV_dVd;
			  Cgb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb);
	     
			  /* Inversion charge partitioning into S / D */
			  if (model->B3SOIPDxpart > 0.5)
			  {   /* 0/100 Charge partition model */
			     T1 = T1 + T1;
			     qsrc = -CoxWL * (0.5 * Vgsteff + 0.25 * T0
				  - T0 * T0 / T1);
			     T7 = (4.0 * Vgsteff - T0) / (T1 * T1);
			     T4 = -(0.5 + 24.0 * T0 * T0 / (T1 * T1));
			     T5 = -(0.25 * AbulkCV - 12.0 * AbulkCV * T0 * T7);
			     T6 = -(0.25 * VdseffCV - 12.0 * T0 * VdseffCV * T7);
			     Csg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
			     Csd1 = CoxWL * T5 * dVdseffCV_dVd;
			     Csb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb);
				 
			  }
			  else if (model->B3SOIPDxpart < 0.5)
			  {   /* 40/60 Charge partition model */
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
			     Csg1 = T4 + T5 * dVdseffCV_dVg;
			     Csd1 = T5 * dVdseffCV_dVd;
			     Csb1 = T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb;
			  }
			  else
			  {   /* 50/50 Charge partition model */
			     qsrc = - 0.5 * (qinv + qbulk);
			     Csg1 = - 0.5 * (Cgg1 + Cbg1);
			     Csb1 = - 0.5 * (Cgb1 + Cbb1);
			     Csd1 = - 0.5 * (Cgd1 + Cbd1);
			  }

		          /* Backgate charge */
		          CboxWL = pParam->B3SOIPDkb1 * model->B3SOIPDfbody * Cbox  
                                 * (pParam->B3SOIPDweffCV / here->B3SOIPDnseg
                                 * pParam->B3SOIPDleffCVbg + here->B3SOIPDaebcp);
                          Qe1 = CboxWL * (Vesfb - Vbs);
		          dQe1_dVb = -CboxWL;
		          dQe1_dVe = CboxWL;
                          if (selfheat) dQe1_dT = -CboxWL * dvfbb_dT;
                          else dQe1_dT = 0;

			  qgate = qinv + Qac0 + Qsub0;
   			  qbody = (qbulk - Qac0 - Qsub0 - Qe1);
			  qsub = Qe1;
			  qdrn = -(qgate + qsrc + qbody + qsub);

			  /* This transform all the dependency on Vgsteff, Vbseff
			     into real ones */
			  Ce1b = dQe1_dVb;
			  Ce1e = dQe1_dVe;

			  Csg = Csg1 * dVgsteff_dVg;
			  Csd = Csd1 + Csg1 * dVgsteff_dVd;
   			  Csb = Csg1 * dVgsteff_dVb + Csb1 * dVbseff_dVb;
                          if (selfheat) CsT = Csg1 * dVgsteff_dT;
                          else  CsT = 0.0;
 
			  Cgg = (Cgg1 + dQsub0_dVg) * dVgsteff_dVg 
                                + dQac0_dVrg + dQsub0_dVrg;
			  Cgd = (Cgg1 + dQsub0_dVg) * dVgsteff_dVd + Cgd1
                                + dQac0_dVd + dQsub0_dVd;
                          Cgb = (Cgg1 + dQsub0_dVg) * dVgsteff_dVb
                                + (Cgb1 + dQsub0_dVb + dQac0_dVb) * dVbseff_dVb;
                          if (selfheat)
                             CgT = (Cgg1 + dQsub0_dVg) * dVgsteff_dT
                                   + dQac0_dT + dQsub0_dT;
                          else  CgT = 0.0;

                          Cbg = (Cbg1 - dQsub0_dVg) * dVgsteff_dVg
                                - dQac0_dVrg - dQsub0_dVrg;
                          Cbd = (Cbg1 - dQsub0_dVg) * dVgsteff_dVd + Cbd1
                                - dQac0_dVd - dQsub0_dVd;
                          Cbb = (Cbg1 - dQsub0_dVg) * dVgsteff_dVb - dQe1_dVb
                                + (Cbb1 - dQsub0_dVb - dQac0_dVb) * dVbseff_dVb;
                          if (selfheat)
                             CbT = (Cbg1 - dQsub0_dVg) * dVgsteff_dT
                                    - dQac0_dT - dQsub0_dT - dQe1_dT;
                          else CbT = 0.0;

			  here->B3SOIPDcggb = Cgg ;
			  here->B3SOIPDcgsb = - (Cgg  + Cgd  + Cgb);
			  here->B3SOIPDcgdb = Cgd; 
                          here->B3SOIPDcgT = CgT;

			  here->B3SOIPDcbgb = Cbg;
			  here->B3SOIPDcbsb = -(Cbg  + Cbd  + Cbb)
					  + Ce1e;
			  here->B3SOIPDcbdb = Cbd;
			  here->B3SOIPDcbeb = - Ce1e ;
                          here->B3SOIPDcbT = CbT;

			  here->B3SOIPDceeb = Ce1e ;
                          here->B3SOIPDceT = dQe1_dT;
 
			  here->B3SOIPDcdgb = -(Cgg + Cbg + Csg);
			  here->B3SOIPDcddb = -(Cgd + Cbd + Csd);
			  here->B3SOIPDcdeb = 0;
                          here->B3SOIPDcdT = -(CgT + CbT + CsT) - dQe1_dT;
			  here->B3SOIPDcdsb = (Cgg + Cgd + Cgb 
					  + Cbg + Cbd + Cbb 
					  + Csg + Csd + Csb) + Ce1b; 
		      } /* End of if capMod == 2 */

                      else if (model->B3SOIPDcapMod == 3)
                      {
                         dVgsteff_dVb /= dVbseff_dVb;
 
                         if (selfheat) {
                            Vfbzb = Vthzb - phi - pParam->B3SOIPDk1eff * sqrtPhi 
                                  + pParam->B3SOIPDdelvt;
                            dVfbzb_dT = dVthzb_dT;      
                         }
                         else {
                            Vfbzb = pParam->B3SOIPDvfbzb + pParam->B3SOIPDdelvt;
                            dVfbzb_dT = 0;
                         }

                         V3 = Vfbzb - Vgs_eff + Vbseff - DELTA_3;
                         if (Vfbzb <= 0.0)
                         {   T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfbzb);
                             T2 = -DELTA_3 / T0;
                         }
                         else
                         {   T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfbzb);
                             T2 = DELTA_3 / T0;
                         }

                         T1 = 0.5 * (1.0 + V3 / T0);
                         Vfbeff = Vfbzb - 0.5 * (V3 + T0);
                         dVfbeff_dVg = T1 * dVgs_eff_dVg;
                         dVfbeff_dVb = -T1;
                         if (selfheat) dVfbeff_dT = (1.0 - T1 - T2) * dVfbzb_dT;
                         else  dVfbeff_dT = 0.0;

/* v2.2.3 */
                         Cox = 3.453133e-11 / (model->B3SOIPDtox - model->B3SOIPDdtoxcv);
                         CoxWL *= model->B3SOIPDtox/ (model->B3SOIPDtox - model->B3SOIPDdtoxcv);
                         CoxWLb *= model->B3SOIPDtox/ (model->B3SOIPDtox - model->B3SOIPDdtoxcv);
                         Tox = 1.0e8 * (model->B3SOIPDtox - model->B3SOIPDdtoxcv); 

                         T0 = (Vgs_eff - Vbseff - Vfbzb) / Tox;
                         dT0_dVg = dVgs_eff_dVg / Tox;
                         dT0_dVb = -1.0 / Tox;

                         tmp = T0 * pParam->B3SOIPDacde;
                         if ((-EXPL_THRESHOLD < tmp) && (tmp < EXPL_THRESHOLD))
                         {   Tcen = pParam->B3SOIPDldeb * exp(tmp);
                             dTcen_dVg = pParam->B3SOIPDacde * Tcen;
                             dTcen_dVb = dTcen_dVg * dT0_dVb;
                             dTcen_dVg *= dT0_dVg;
                             if (selfheat)
                                dTcen_dT = -Tcen * pParam->B3SOIPDacde * dVfbzb_dT / Tox;
                             else dTcen_dT = 0;
                         }
                         else if (tmp <= -EXPL_THRESHOLD)
                         {   Tcen = pParam->B3SOIPDldeb * MIN_EXPL;
                             dTcen_dVg = dTcen_dVb = dTcen_dT = 0.0;
                         }
                         else
                         {   Tcen = pParam->B3SOIPDldeb * MAX_EXPL;
                             dTcen_dVg = dTcen_dVb = dTcen_dT = 0.0;
                         }

                         LINK = 1.0e-3 * (model->B3SOIPDtox - model->B3SOIPDdtoxcv); /* v2.2.3 */
                         V3 = pParam->B3SOIPDldeb - Tcen - LINK;
                         V4 = sqrt(V3 * V3 + 4.0 * LINK * pParam->B3SOIPDldeb);
                         Tcen = pParam->B3SOIPDldeb - 0.5 * (V3 + V4);
                         T1 = 0.5 * (1.0 + V3 / V4);
                         dTcen_dVg *= T1;
                         dTcen_dVb *= T1;
                         if (selfheat) 
                            dTcen_dT *= T1;
                         else dTcen_dT = 0;

                         Ccen = EPSSI / Tcen;
                         T2 = Cox / (Cox + Ccen);
                         Coxeff = T2 * Ccen;
                         T3 = -Ccen / Tcen;
                         dCoxeff_dVg = T2 * T2 * T3;
                         dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
                         dCoxeff_dVg *= dTcen_dVg;
                         if (selfheat)
                            dCoxeff_dT = T3 * dTcen_dT * (T2 - Coxeff / (Cox + Ccen));
                         else dCoxeff_dT = 0;
                         CoxWLcenb = CoxWLb * Coxeff / Cox;
                         if (selfheat)
                            dCoxWLcenb_dT = CoxWLb * dCoxeff_dT / Cox;
                         else dCoxWLcenb_dT = 0;

                         Qac0 = CoxWLcenb * (Vfbeff - Vfbzb);
                         QovCox = Qac0 / Coxeff;
                         dQac0_dVg = CoxWLcenb * dVfbeff_dVg
                                   + QovCox * dCoxeff_dVg;
                         dQac0_dVb = CoxWLcenb * dVfbeff_dVb
                                   + QovCox * dCoxeff_dVb;
                         if (selfheat) dQac0_dT = CoxWLcenb * (dVfbeff_dT - dVfbzb_dT)
                                                + dCoxWLcenb_dT * (Vfbeff - Vfbzb);
                         else  dQac0_dT = 0.0;

                         T0 = 0.5 * pParam->B3SOIPDk1eff;
                         T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff;
                         if (pParam->B3SOIPDk1eff == 0.0)
                         {   T1 = 0.0;
                             T2 = 0.0;
                         }
                         else if (T3 < 0.0)
                         {   T1 = T0 + T3 / pParam->B3SOIPDk1eff;
                             T2 = CoxWLcenb;
                         }
                         else
                         {   T1 = sqrt(T0 * T0 + T3);
                             T2 = CoxWLcenb * T0 / T1;
                         }

                         Qsub0 = CoxWLcenb * pParam->B3SOIPDk1eff * (T1 - T0);
                         QovCox = Qsub0 / Coxeff;
                         dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg)
                                    + QovCox * dCoxeff_dVg;
                         dQsub0_dVd = -T2 * dVgsteff_dVd;
                         dQsub0_dVb = -T2 * (dVfbeff_dVb + 1 + dVgsteff_dVb)
                                    + QovCox * dCoxeff_dVb;
                         if (selfheat) 
                            dQsub0_dT = -T2 * (dVfbeff_dT + dVgsteff_dT)
                                      + dCoxWLcenb_dT * pParam->B3SOIPDk1eff * (T1 - T0);
                         else  dQsub0_dT = 0.0;

                         /* Gate-bias dependent delta Phis begins */
                         if (pParam->B3SOIPDk1eff <= 0.0)
                         {   Denomi = 0.25 * pParam->B3SOIPDmoin * Vtm;
                             T0 = 0.5 * pParam->B3SOIPDsqrtPhi;
                         }
                         else
                         {   Denomi = pParam->B3SOIPDmoin * Vtm
                                    * pParam->B3SOIPDk1eff * pParam->B3SOIPDk1eff;
                             T0 = pParam->B3SOIPDk1eff * pParam->B3SOIPDsqrtPhi;
                         }
                         T1 = 2.0 * T0 + Vgsteff;

                         DeltaPhi = Vtm * log(1.0 + T1 * Vgsteff / Denomi);
                         dDeltaPhi_dVg = 2.0 * Vtm * (T1 -T0) / (Denomi + T1 * Vgsteff);
                         dDeltaPhi_dVd = dDeltaPhi_dVg * dVgsteff_dVd;
                         dDeltaPhi_dVb = dDeltaPhi_dVg * dVgsteff_dVb;
                         /* End of delta Phis */

                         T3 = 4.0 * (Vth - Vfbzb - phi);
                         Tox += Tox;
                         if ((T0 = (Vgsteff + T3) / Tox) > 1e-20) {
                             tmp = exp(0.7 * log(T0));
                             T1 = 1.0 + tmp;
                             T2 = 0.7 * tmp / (T0 * Tox);
                             Tcen = 1.9e-9 / T1;
                             dTcen_dVg = -1.9e-9 * T2 / T1 /T1;
                             dTcen_dVd = dTcen_dVg * (4.0 * dVth_dVd + dVgsteff_dVd);
                             dTcen_dVb = dTcen_dVg * (4.0 * dVth_dVb + dVgsteff_dVb);
                             dTcen_dVg *= dVgsteff_dVg;
                             if (selfheat)
                                dTcen_dT = -Tcen * T2 / T1 
                                         * (4.0 * (dVth_dT - dVfbzb_dT) + dVgsteff_dT);
                             else dTcen_dT = 0;
                         }
                         else {
                             T0 = 1e-20;
                             tmp = exp(0.7 * log(T0));
                             T1 = 1.0 + tmp;
                             T2 = 0.7 * tmp / (T0 * Tox);
                             Tcen = 1.9e-9 / T1;
                             dTcen_dVg = 0;
                             dTcen_dVd = 0;
                             dTcen_dVb = 0;
                             dTcen_dT = 0;
                         }

                         Ccen = EPSSI / Tcen;
                         T0 = Cox / (Cox + Ccen);
                         Coxeff = T0 * Ccen;
                         T1 = -Ccen / Tcen;
                         dCoxeff_dVg = T0 * T0 * T1;
                         dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd;
                         dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
                         dCoxeff_dVg *= dTcen_dVg;
                         if (selfheat)
                            dCoxeff_dT = T1 * dTcen_dT * (T0 - Coxeff / (Cox + Ccen));
                         else dCoxeff_dT = 0;
                         CoxWLcen = CoxWL * Coxeff / Cox;
                         CoxWLcenb = CoxWLb * Coxeff / Cox;

                         AbulkCV = Abulk0 * pParam->B3SOIPDabulkCVfactor;
                         dAbulkCV_dVb = pParam->B3SOIPDabulkCVfactor * dAbulk0_dVb;
                         VdsatCV = (Vgsteff - DeltaPhi) / AbulkCV;
                         V4 = VdsatCV - Vds - DELTA_4;
                         T0 = sqrt(V4 * V4 + 4.0 * DELTA_4 * VdsatCV);
                         VdseffCV = VdsatCV - 0.5 * (V4 + T0);
                         T1 = 0.5 * (1.0 + V4 / T0);
                         T2 = DELTA_4 / T0;
                         T3 = (1.0 - T1 - T2) / AbulkCV;
                         T4 = T3 * ( 1.0 - dDeltaPhi_dVg);
                         dVdseffCV_dVg = T4;
                         dVdseffCV_dVd = T1;
                         dVdseffCV_dVb = -T3 * VdsatCV * dAbulkCV_dVb;

                         T0 = AbulkCV * VdseffCV;
                         T1 = Vgsteff - DeltaPhi;
                         T2 = 12.0 * (T1 - 0.5 * T0 + 1.0e-20);
                         T3 = T0 / T2;
                         T4 = 1.0 - 12.0 * T3 * T3;
                         T5 = AbulkCV * (6.0 * T0 * (4.0 * T1 - T0) / (T2 * T2) - 0.5);
                         T6 = T5 * VdseffCV / AbulkCV;

                         qinv = qgate = qinoi = CoxWLcen * (T1 - T0 * (0.5 - T3));
                         QovCox = qgate / Coxeff;
                         Cgg1 = CoxWLcen * (T4 * (1.0 - dDeltaPhi_dVg)
                              + T5 * dVdseffCV_dVg);
                         Cgd1 = CoxWLcen * T5 * dVdseffCV_dVd + Cgg1
                              * dVgsteff_dVd + QovCox * dCoxeff_dVd;
                         Cgb1 = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
                              + Cgg1 * dVgsteff_dVb + QovCox * dCoxeff_dVb;
                         Cgg1 = Cgg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;

                         T7 = 1.0 - AbulkCV;
                         T8 = T2 * T2;
                         T9 = 12.0 * T7 * T0 * T0 / (T8 * AbulkCV);
                         T10 = T9 * (1.0 - dDeltaPhi_dVg);
                         T11 = -T7 * T5 / AbulkCV;
                         T12 = -(T9 * T1 / AbulkCV + VdseffCV * (0.5 - T0 / T2));

                         qbulk = CoxWLcenb * T7 * (0.5 * VdseffCV - T0 * VdseffCV / T2);
                         QovCox = qbulk / Coxeff;
                         Cbg1 = CoxWLcenb * (T10 + T11 * dVdseffCV_dVg);
                         Cbd1 = CoxWLcenb * T11 * dVdseffCV_dVd + Cbg1
                              * dVgsteff_dVd + QovCox * dCoxeff_dVd;
                         Cbb1 = CoxWLcenb * (T11 * dVdseffCV_dVb + T12 * dAbulkCV_dVb)
                              + Cbg1 * dVgsteff_dVb + QovCox * dCoxeff_dVb;
                         Cbg1 = Cbg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;

                         if (model->B3SOIPDxpart > 0.5)
                         {   /* 0/100 partition */
                             qsrc = -CoxWLcen * (T1 / 2.0 + T0 / 4.0
                                  - 0.5 * T0 * T0 / T2);
                             QovCox = qsrc / Coxeff;
                             T2 += T2;
                             T3 = T2 * T2;
                             T7 = -(0.25 - 12.0 * T0 * (4.0 * T1 - T0) / T3);
                             T4 = -(0.5 + 24.0 * T0 * T0 / T3) * (1.0 - dDeltaPhi_dVg);
                             T5 = T7 * AbulkCV;
                             T6 = T7 * VdseffCV;

                             Csg = CoxWLcen * (T4 + T5 * dVdseffCV_dVg);
                             Csd = CoxWLcen * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd
                                 + QovCox * dCoxeff_dVd;
                             Csb = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
                                 + Csg * dVgsteff_dVb + QovCox * dCoxeff_dVb;
                             Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg;
                         }
                         else if (model->B3SOIPDxpart < 0.5)
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

                             Csg = T5 * (1.0 - dDeltaPhi_dVg) + T6 * dVdseffCV_dVg;
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

                         /* Backgate charge */
                         CboxWL = pParam->B3SOIPDkb1 * model->B3SOIPDfbody * Cbox 
                                * (pParam->B3SOIPDweffCV / here->B3SOIPDnseg
                                * pParam->B3SOIPDleffCVbg + here->B3SOIPDaebcp);
                         Qe1 = CboxWL * (Vesfb - Vbs);
                         Ce1b = dQe1_dVb = -CboxWL;
                         Ce1e = dQe1_dVe = CboxWL;
                         if (selfheat) Ce1T = dQe1_dT = -CboxWL * dvfbb_dT;
                         else Ce1T = dQe1_dT = 0.0;

                         qgate += Qac0 + Qsub0 - qbulk;
                         qbody = qbulk - Qac0 - Qsub0 - Qe1;
                         qsub = Qe1;
                         qdrn = -(qgate + qbody + qsub + qsrc);

                         Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
                         Cbd = Cbd1 - dQsub0_dVd;
                         Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb - Ce1b / dVbseff_dVb;
                         if (selfheat)
                            CbT = Cbg1 * dVgsteff_dT - dQac0_dT
                                - dQsub0_dT - dQe1_dT;
                         else CbT = 0.0;

                         Cgg = Cgg1 - Cbg;
                         Cgd = Cgd1 - Cbd;
                         Cgb = Cgb1 - Cbb - Ce1b / dVbseff_dVb;
                         if (selfheat)
                            CgT = Cgg1 * dVgsteff_dT + dQac0_dT
                                + dQsub0_dT;
                         else  CgT = 0.0;

                         Cgb *= dVbseff_dVb;
                         Cbb *= dVbseff_dVb;
                         Csb *= dVbseff_dVb;
                         if (selfheat) CsT = Csg * dVgsteff_dT;
                         else  CsT = 0.0;
 
                         here->B3SOIPDcggb = Cgg;
                         here->B3SOIPDcgsb = -(Cgg + Cgd + Cgb);
                         here->B3SOIPDcgdb = Cgd;
                         here->B3SOIPDcgT  = CgT;

                         here->B3SOIPDcbgb = Cbg;
                         here->B3SOIPDcbsb = -(Cbg + Cbd + Cbb)
                                         + Ce1e;
                         here->B3SOIPDcbdb = Cbd;
                         here->B3SOIPDcbeb = -Ce1e;
                         here->B3SOIPDcbT  = CbT;

                         here->B3SOIPDceT = Ce1T;
                         here->B3SOIPDceeb = Ce1e ;

                         here->B3SOIPDcdgb = -(Cgg + Cbg + Csg);
                         here->B3SOIPDcddb = -(Cgd + Cbd + Csd);
                         here->B3SOIPDcdeb = 0;
                         here->B3SOIPDcdT   = -(CgT+CbT+CsT) - Ce1T;
                         here->B3SOIPDcdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb
                                         + Csg + Csd + Csb) + Ce1b;
                         here->B3SOIPDqinv = -qinoi;

                      } /* End of if capMod ==3 */
                  }


	finished: /* returning Values to Calling Routine */
		  /*
		   *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
		   */
		  if (ChargeComputationNeeded)
		  {   
                      /* Intrinsic S/D junction charge */
		      PhiBSWG = model->B3SOIPDGatesidewallJctPotential;
                      dPhiBSWG_dT = -model->B3SOIPDtpbswg;
                      PhiBSWG += dPhiBSWG_dT * (Temp - model->B3SOIPDtnom);
		      MJSWG = model->B3SOIPDbodyJctGateSideGradingCoeff;

		      cjsbs = model->B3SOIPDunitLengthGateSidewallJctCap
			 * wdiosCV * model->B3SOIPDtsi / 1e-7;
                      dcjsbs_dT = cjsbs * model->B3SOIPDtcjswg;
                      cjsbs += dcjsbs_dT * (Temp - model->B3SOIPDtnom);

                      cjdbs = model->B3SOIPDunitLengthGateSidewallJctCap
                         * wdiodCV * model->B3SOIPDtsi / 1e-7;
                      dcjdbs_dT = cjdbs * model->B3SOIPDtcjswg;
                      cjdbs += dcjdbs_dT * (Temp - model->B3SOIPDtnom);

                      DioMax = 0.9 * (PhiBSWG);

                      arg = 1.0 - (Vbs > DioMax ? DioMax : Vbs) / PhiBSWG;

                      if (selfheat)
                         darg_dT = (1 - arg) / PhiBSWG * dPhiBSWG_dT;

		      if (MJSWG == 0.5) {
		         dT3_dVb = 1.0 / sqrt(arg);

                         if (selfheat) ddT3_dVb_dT = -0.5 * dT3_dVb / arg * darg_dT;
                      }
		      else {
			 dT3_dVb = exp(-MJSWG * log(arg));

                         if (selfheat) ddT3_dVb_dT = -MJSWG * dT3_dVb / arg * darg_dT; 
                      }
		      T3 = (1.0 - arg * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);

                      if (selfheat)
                         dT3_dT = (1.0 - arg * dT3_dVb) * dPhiBSWG_dT / (1.0 - MJSWG)
                            - (arg * ddT3_dVb_dT + darg_dT * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);

                      if (Vbs > DioMax)
                         T3 += dT3_dVb * (Vbs - DioMax);

                      qjs = cjsbs * T3 + model->B3SOIPDtt * Ibsdif;
                      gcjsbs = cjsbs * dT3_dVb + model->B3SOIPDtt * dIbsdif_dVb;

                      if (selfheat) 
                         gcjsT = model->B3SOIPDtt * dIbsdif_dT + dcjsbs_dT * T3 + dT3_dT * cjsbs;
                      else  gcjsT = 0.0;


		      arg = 1.0 - (Vbd > DioMax ? DioMax : Vbd) / PhiBSWG;

                      if (selfheat)
                         darg_dT = (1 - arg) / PhiBSWG * dPhiBSWG_dT;

	              if (MJSWG == 0.5) {
			 dT3_dVb = 1.0 / sqrt(arg);

                         if (selfheat) ddT3_dVb_dT = -0.5 * dT3_dVb / arg * darg_dT;
                      }
		      else {
			 dT3_dVb = exp(-MJSWG * log(arg));

                         if (selfheat) ddT3_dVb_dT = -MJSWG * dT3_dVb / arg * darg_dT;
                      }
		      T3 = (1.0 - arg * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);

                      if (selfheat)
                         dT3_dT = (1.0 - arg * dT3_dVb) * dPhiBSWG_dT / (1.0 - MJSWG)
                         - (arg * ddT3_dVb_dT + darg_dT * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);

                      if (Vbd > DioMax)
                         T3 += dT3_dVb * (Vbd - DioMax);

                      dT3_dVd = -dT3_dVb;

                      qjd = cjdbs * T3 + model->B3SOIPDtt * Ibddif;
                      gcjdbs = cjdbs * dT3_dVb + model->B3SOIPDtt * dIbddif_dVb;
                      gcjdds = cjdbs * dT3_dVd + model->B3SOIPDtt * dIbddif_dVd;

                      if (selfheat)
                         gcjdT = model->B3SOIPDtt * dIbddif_dT + dcjdbs_dT * T3 + dT3_dT * cjdbs;
                      else  gcjdT = 0.0;


		      qdrn -= qjd;
		      qbody += (qjs + qjd);
		      qsrc = -(qgate + qbody + qdrn + qsub);

		      /* Update the conductance */
		      here->B3SOIPDcddb -= gcjdds;
                      here->B3SOIPDcdT -= gcjdT;
		      here->B3SOIPDcdsb += gcjdds + gcjdbs;

		      here->B3SOIPDcbdb += (gcjdds);
                      here->B3SOIPDcbT += (gcjdT + gcjsT);
		      here->B3SOIPDcbsb -= (gcjdds + gcjdbs + gcjsbs);


		      /* Extrinsic Bottom S/D to substrate charge */
		      T10 = -model->B3SOIPDtype * ves;
		      /* T10 is vse without type conversion */
                      T11 = model->B3SOIPDtype * (vds - ves);
                      /* T11 is vde without type conversion */

                      if (model->B3SOIPDcsdmin != 0.0) 
                      {
		         if ( ((pParam->B3SOIPDnsub > 0) && (model->B3SOIPDtype > 0)) ||
		              ((pParam->B3SOIPDnsub < 0) && (model->B3SOIPDtype < 0)) )
		         {
		            if (T10 < pParam->B3SOIPDvsdfb)
		            {  here->B3SOIPDqse = here->B3SOIPDcsbox * (T10 - pParam->B3SOIPDvsdfb);
			       here->B3SOIPDgcse = here->B3SOIPDcsbox;
		            }
		            else if (T10 < pParam->B3SOIPDsdt1)
		            {  T0 = T10 - pParam->B3SOIPDvsdfb;
			       T1 = T0 * T0;
			       here->B3SOIPDqse = T0 * (here->B3SOIPDcsbox - 
                                                pParam->B3SOIPDst2 / 3 * T1) ;
			       here->B3SOIPDgcse = here->B3SOIPDcsbox - pParam->B3SOIPDst2 * T1;
		            }
		            else if (T10 < pParam->B3SOIPDvsdth)
		            {  T0 = T10 - pParam->B3SOIPDvsdth;
			       T1 = T0 * T0;
			       here->B3SOIPDqse = here->B3SOIPDcsmin * T10 + here->B3SOIPDst4 + 
                                                pParam->B3SOIPDst3 / 3 * T0 * T1;
   			    here->B3SOIPDgcse = here->B3SOIPDcsmin + pParam->B3SOIPDst3 * T1;
		            }
		            else 
		            {  here->B3SOIPDqse = here->B3SOIPDcsmin * T10 + here->B3SOIPDst4;
			       here->B3SOIPDgcse = here->B3SOIPDcsmin;
		            }
		         } else
		         {
		            if (T10 < pParam->B3SOIPDvsdth)
		            {  here->B3SOIPDqse = here->B3SOIPDcsmin * (T10 - pParam->B3SOIPDvsdth);
			       here->B3SOIPDgcse = here->B3SOIPDcsmin;
		            }
		            else if (T10 < pParam->B3SOIPDsdt1)
		            {  T0 = T10 - pParam->B3SOIPDvsdth;
   			       T1 = T0 * T0;
   			       here->B3SOIPDqse = T0 * (here->B3SOIPDcsmin - pParam->B3SOIPDst2 / 3 * T1) ;
			       here->B3SOIPDgcse = here->B3SOIPDcsmin - pParam->B3SOIPDst2 * T1;
		            }
		            else if (T10 < pParam->B3SOIPDvsdfb)
		            {  T0 = T10 - pParam->B3SOIPDvsdfb;
			       T1 = T0 * T0;
			       here->B3SOIPDqse = here->B3SOIPDcsbox * T10 + here->B3SOIPDst4 + 
                                                pParam->B3SOIPDst3 / 3 * T0 * T1;
			       here->B3SOIPDgcse = here->B3SOIPDcsbox + pParam->B3SOIPDst3 * T1;
		            }
		            else 
		            {  here->B3SOIPDqse = here->B3SOIPDcsbox * T10 + here->B3SOIPDst4;
			       here->B3SOIPDgcse = here->B3SOIPDcsbox;
		            }
		         }

		         if ( ((pParam->B3SOIPDnsub > 0) && (model->B3SOIPDtype > 0)) ||
		              ((pParam->B3SOIPDnsub < 0) && (model->B3SOIPDtype < 0)) )
		         {
		            if (T11 < pParam->B3SOIPDvsdfb)
		            {  here->B3SOIPDqde = here->B3SOIPDcdbox * (T11 - pParam->B3SOIPDvsdfb);
			       here->B3SOIPDgcde = here->B3SOIPDcdbox;
		            }
		            else if (T11 < pParam->B3SOIPDsdt1)
		            {  T0 = T11 - pParam->B3SOIPDvsdfb;
   			       T1 = T0 * T0;
   			       here->B3SOIPDqde = T0 * (here->B3SOIPDcdbox - pParam->B3SOIPDdt2 / 3 * T1) ;
			       here->B3SOIPDgcde = here->B3SOIPDcdbox - pParam->B3SOIPDdt2 * T1;
		            }
		            else if (T11 < pParam->B3SOIPDvsdth)
		            {  T0 = T11 - pParam->B3SOIPDvsdth;
			       T1 = T0 * T0;
			       here->B3SOIPDqde = here->B3SOIPDcdmin * T11 + here->B3SOIPDdt4 + 
                                                pParam->B3SOIPDdt3 / 3 * T0 * T1;
			       here->B3SOIPDgcde = here->B3SOIPDcdmin + pParam->B3SOIPDdt3 * T1;
		            }
		            else 
		            {  here->B3SOIPDqde = here->B3SOIPDcdmin * T11 + here->B3SOIPDdt4;
			       here->B3SOIPDgcde = here->B3SOIPDcdmin;
		            }
		         } else
		         {
		            if (T11 < pParam->B3SOIPDvsdth)
		            {  here->B3SOIPDqde = here->B3SOIPDcdmin * (T11 - pParam->B3SOIPDvsdth);
			       here->B3SOIPDgcde = here->B3SOIPDcdmin;
		            }
		            else if (T11 < pParam->B3SOIPDsdt1)
		            {  T0 = T11 - pParam->B3SOIPDvsdth;
   			       T1 = T0 * T0;
   			       here->B3SOIPDqde = T0 * (here->B3SOIPDcdmin - pParam->B3SOIPDdt2 / 3 * T1) ;
			       here->B3SOIPDgcde = here->B3SOIPDcdmin - pParam->B3SOIPDdt2 * T1;
		            }
		            else if (T11 < pParam->B3SOIPDvsdfb)
		            {  T0 = T11 - pParam->B3SOIPDvsdfb;
			       T1 = T0 * T0;
			       here->B3SOIPDqde = here->B3SOIPDcdbox * T11 + here->B3SOIPDdt4 + 
                                                pParam->B3SOIPDdt3 / 3 * T0 * T1;
			       here->B3SOIPDgcde = here->B3SOIPDcdbox + pParam->B3SOIPDdt3 * T1;
		            }
		            else 
		            {  here->B3SOIPDqde = here->B3SOIPDcdbox * T11 + here->B3SOIPDdt4;
			       here->B3SOIPDgcde = here->B3SOIPDcdbox;
		            }
		         } 
                      }
                      else {
                         here->B3SOIPDqse = here->B3SOIPDcsbox * T10;
                         here->B3SOIPDgcse = here->B3SOIPDcsbox;
                         here->B3SOIPDqde = here->B3SOIPDcdbox * T11;
                         here->B3SOIPDgcde = here->B3SOIPDcdbox;
                      }

		      /* Extrinsic : Sidewall fringing S/D charge */
                      here->B3SOIPDqse += here->B3SOIPDcsesw * T10;
                      here->B3SOIPDgcse += here->B3SOIPDcsesw;
                      here->B3SOIPDqde += here->B3SOIPDcdesw * T11;
                      here->B3SOIPDgcde += here->B3SOIPDcdesw;

		      /* All charge are mutliplied with type at the end, but qse and qde
			 have true polarity => so pre-mutliplied with type */
		      here->B3SOIPDqse *= model->B3SOIPDtype;
		      here->B3SOIPDqde *= model->B3SOIPDtype;
		  }


		  here->B3SOIPDcbb = Cbb;
		  here->B3SOIPDcbd = Cbd;
		  here->B3SOIPDcbg = Cbg;
		  here->B3SOIPDqbf = -Qsub0 - Qac0;
		  here->B3SOIPDqjs = qjs;
		  here->B3SOIPDqjd = qjd;

		  /*
		   *  check convergence
		   */
		  if ((here->B3SOIPDoff == 0) || (!(ckt->CKTmode & MODEINITFIX)))
		  {   if (Check == 1)
		      {   ckt->CKTnoncon++;
		      }
		  }

                  *(ckt->CKTstate0 + here->B3SOIPDvg) = vg;
                  *(ckt->CKTstate0 + here->B3SOIPDvd) = vd;
                  *(ckt->CKTstate0 + here->B3SOIPDvs) = vs;
                  *(ckt->CKTstate0 + here->B3SOIPDvp) = vp;
                  *(ckt->CKTstate0 + here->B3SOIPDve) = ve;

		  *(ckt->CKTstate0 + here->B3SOIPDvbs) = vbs;
		  *(ckt->CKTstate0 + here->B3SOIPDvbd) = vbd;
		  *(ckt->CKTstate0 + here->B3SOIPDvgs) = vgs;
		  *(ckt->CKTstate0 + here->B3SOIPDvds) = vds;
		  *(ckt->CKTstate0 + here->B3SOIPDves) = ves;
		  *(ckt->CKTstate0 + here->B3SOIPDvps) = vps;
		  *(ckt->CKTstate0 + here->B3SOIPDdeltemp) = delTemp;

		  /* bulk and channel charge plus overlaps */

		  if (!ChargeComputationNeeded)
		      goto line850; 
#ifndef NOBYPASS
	line755:
#endif
		  ag0 = ckt->CKTag[0];

		  T0 = vgd + DELTA_1;
		  T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
		  T2 = 0.5 * (T0 - T1);

/* v2.2.3 bug fix */
                  T3 = pParam->B3SOIPDwdiodCV * pParam->B3SOIPDcgdl;

		  T4 = sqrt(1.0 - 4.0 * T2 / pParam->B3SOIPDckappa);
		  cgdo = pParam->B3SOIPDcgdo + T3 - T3 * (1.0 - 1.0 / T4)
			 * (0.5 - 0.5 * T0 / T1);
		  qgdo = (pParam->B3SOIPDcgdo + T3) * vgd - T3 * (T2
			 + 0.5 * pParam->B3SOIPDckappa * (T4 - 1.0));

		  T0 = vgs + DELTA_1;
		  T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
		  T2 = 0.5 * (T0 - T1);

/* v2.2.3 bug fix */
                  T3 = pParam->B3SOIPDwdiosCV * pParam->B3SOIPDcgsl;

		  T4 = sqrt(1.0 - 4.0 * T2 / pParam->B3SOIPDckappa);
		  cgso = pParam->B3SOIPDcgso + T3 - T3 * (1.0 - 1.0 / T4)
			 * (0.5 - 0.5 * T0 / T1);
		  qgso = (pParam->B3SOIPDcgso + T3) * vgs - T3 * (T2
			 + 0.5 * pParam->B3SOIPDckappa * (T4 - 1.0));



                  if (here->B3SOIPDdebugMod < 0)
                     goto line850;


		  if (here->B3SOIPDmode > 0)
		  {   gcdgb = (here->B3SOIPDcdgb - cgdo) * ag0;
		      gcddb = (here->B3SOIPDcddb + cgdo + here->B3SOIPDgcde) * ag0;
		      gcdsb = here->B3SOIPDcdsb * ag0;
		      gcdeb = (here->B3SOIPDcdeb - here->B3SOIPDgcde) * ag0;
                      gcdT = model->B3SOIPDtype * here->B3SOIPDcdT * ag0;

		      gcsgb = -(here->B3SOIPDcggb + here->B3SOIPDcbgb + here->B3SOIPDcdgb
			    + cgso) * ag0;
		      gcsdb = -(here->B3SOIPDcgdb + here->B3SOIPDcbdb + here->B3SOIPDcddb) * ag0;
		      gcssb = (cgso + here->B3SOIPDgcse - (here->B3SOIPDcgsb + here->B3SOIPDcbsb
			    + here->B3SOIPDcdsb)) * ag0;
		      gcseb = -(here->B3SOIPDgcse + here->B3SOIPDcbeb + here->B3SOIPDcdeb
			    + here->B3SOIPDceeb) * ag0;
                      gcsT = - model->B3SOIPDtype * (here->B3SOIPDcgT + here->B3SOIPDcbT + here->B3SOIPDcdT
                            + here->B3SOIPDceT) * ag0;

		      gcggb = (here->B3SOIPDcggb + cgdo + cgso + pParam->B3SOIPDcgeo) * ag0;
		      gcgdb = (here->B3SOIPDcgdb - cgdo) * ag0;
		      gcgsb = (here->B3SOIPDcgsb - cgso) * ag0;
		      gcgeb = (- pParam->B3SOIPDcgeo) * ag0;
                      gcgT = model->B3SOIPDtype * here->B3SOIPDcgT * ag0;

		      gcbgb = here->B3SOIPDcbgb * ag0;
		      gcbdb = here->B3SOIPDcbdb * ag0;
		      gcbsb = here->B3SOIPDcbsb * ag0;
   		      gcbeb = here->B3SOIPDcbeb * ag0;
                      gcbT = model->B3SOIPDtype * here->B3SOIPDcbT * ag0;

		      gcegb = (- pParam->B3SOIPDcgeo) * ag0;
		      gcedb = (- here->B3SOIPDgcde) * ag0;
		      gcesb = (- here->B3SOIPDgcse) * ag0;
		      gceeb = (here->B3SOIPDgcse + here->B3SOIPDgcde +
                               here->B3SOIPDceeb + pParam->B3SOIPDcgeo) * ag0;
 
                      gceT = model->B3SOIPDtype * here->B3SOIPDceT * ag0;

                      gcTt = pParam->B3SOIPDcth * ag0;

		      sxpart = 0.6;
		      dxpart = 0.4;

		      /* Lump the overlap capacitance and S/D parasitics */
		      qgd = qgdo;
		      qgs = qgso;
		      qge = pParam->B3SOIPDcgeo * vge;
		      qgate += qgd + qgs + qge;
		      qdrn += here->B3SOIPDqde - qgd;
		      qsub -= qge + here->B3SOIPDqse + here->B3SOIPDqde; 
		      qsrc = -(qgate + qbody + qdrn + qsub);
		  }
		  else
		  {   gcsgb = (here->B3SOIPDcdgb - cgso) * ag0;
		      gcssb = (here->B3SOIPDcddb + cgso + here->B3SOIPDgcse) * ag0;
		      gcsdb = here->B3SOIPDcdsb * ag0;
		      gcseb = (here->B3SOIPDcdeb - here->B3SOIPDgcse) * ag0;
                      gcsT = model->B3SOIPDtype * here->B3SOIPDcdT * ag0;

		      gcdgb = -(here->B3SOIPDcggb + here->B3SOIPDcbgb + here->B3SOIPDcdgb
			    + cgdo) * ag0;
		      gcdsb = -(here->B3SOIPDcgdb + here->B3SOIPDcbdb + here->B3SOIPDcddb) * ag0;
		      gcddb = (cgdo + here->B3SOIPDgcde - (here->B3SOIPDcgsb + here->B3SOIPDcbsb 
			    + here->B3SOIPDcdsb)) * ag0;
		      gcdeb = -(here->B3SOIPDgcde + here->B3SOIPDcbeb + here->B3SOIPDcdeb
			    + here->B3SOIPDceeb) * ag0;
                      gcdT = - model->B3SOIPDtype * (here->B3SOIPDcgT + here->B3SOIPDcbT 
                             + here->B3SOIPDcdT + here->B3SOIPDceT) * ag0;

		      gcggb = (here->B3SOIPDcggb + cgdo + cgso + pParam->B3SOIPDcgeo) * ag0;
		      gcgsb = (here->B3SOIPDcgdb - cgso) * ag0;
		      gcgdb = (here->B3SOIPDcgsb - cgdo) * ag0;
		      gcgeb = (- pParam->B3SOIPDcgeo) * ag0;
                      gcgT = model->B3SOIPDtype * here->B3SOIPDcgT * ag0;

		      gcbgb = here->B3SOIPDcbgb * ag0;
		      gcbsb = here->B3SOIPDcbdb * ag0;
		      gcbdb = here->B3SOIPDcbsb * ag0;
		      gcbeb = here->B3SOIPDcbeb * ag0;
                      gcbT = model->B3SOIPDtype * here->B3SOIPDcbT * ag0;

		      gcegb = (-pParam->B3SOIPDcgeo) * ag0;
		      gcesb = (- here->B3SOIPDgcse) * ag0;
		      gcedb = (- here->B3SOIPDgcde) * ag0;
		      gceeb = (here->B3SOIPDceeb + pParam->B3SOIPDcgeo +
			       here->B3SOIPDgcse + here->B3SOIPDgcde) * ag0;
                      gceT = model->B3SOIPDtype * here->B3SOIPDceT * ag0;
		     
                      gcTt = pParam->B3SOIPDcth * ag0;

		      dxpart = 0.6;
		      sxpart = 0.4;

		      /* Lump the overlap capacitance */
		      qgd = qgdo;
		      qgs = qgso;
		      qge = pParam->B3SOIPDcgeo * vge;
		      qgate += qgd + qgs + qge;
		      qsrc = qdrn - qgs + here->B3SOIPDqse;
		      qsub -= qge + here->B3SOIPDqse + here->B3SOIPDqde; 
		      qdrn = -(qgate + qbody + qsrc + qsub);
		  }

		  here->B3SOIPDcgdo = cgdo;
		  here->B3SOIPDcgso = cgso;

                  if (ByPass) goto line860;

		  *(ckt->CKTstate0 + here->B3SOIPDqe) = qsub;
		  *(ckt->CKTstate0 + here->B3SOIPDqg) = qgate;
		  *(ckt->CKTstate0 + here->B3SOIPDqd) = qdrn;
                  *(ckt->CKTstate0 + here->B3SOIPDqb) = qbody;
		  if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0.0)) 
                     *(ckt->CKTstate0 + here->B3SOIPDqth) = pParam->B3SOIPDcth * delTemp;


		  /* store small signal parameters */
		  if (ckt->CKTmode & MODEINITSMSIG)
		  {   goto line1000;
		  }
		  if (!ChargeComputationNeeded)
		      goto line850;
	       

		  if (ckt->CKTmode & MODEINITTRAN)
		  {   *(ckt->CKTstate1 + here->B3SOIPDqb) =
			    *(ckt->CKTstate0 + here->B3SOIPDqb);
		      *(ckt->CKTstate1 + here->B3SOIPDqg) =
			    *(ckt->CKTstate0 + here->B3SOIPDqg);
		      *(ckt->CKTstate1 + here->B3SOIPDqd) =
			    *(ckt->CKTstate0 + here->B3SOIPDqd);
		      *(ckt->CKTstate1 + here->B3SOIPDqe) =
			    *(ckt->CKTstate0 + here->B3SOIPDqe);
                      *(ckt->CKTstate1 + here->B3SOIPDqth) =
                            *(ckt->CKTstate0 + here->B3SOIPDqth);
		  }
	       
                  error = NIintegrate(ckt, &geq, &ceq,0.0,here->B3SOIPDqb);
                  if (error) return(error);
                  error = NIintegrate(ckt, &geq, &ceq, 0.0, here->B3SOIPDqg);
                  if (error) return(error);
                  error = NIintegrate(ckt,&geq, &ceq, 0.0, here->B3SOIPDqd);
                  if (error) return(error);
                  error = NIintegrate(ckt,&geq, &ceq, 0.0, here->B3SOIPDqe);
                  if (error) return(error);
		  if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0.0)) 
                  {
                      error = NIintegrate(ckt, &geq, &ceq, 0.0, here->B3SOIPDqth);
                      if (error) return (error);
                  }
		 
		  goto line860;

	line850:
		  /* initialize to zero charge conductance and current */
		  ceqqe = ceqqg = ceqqb = ceqqd = ceqqth= 0.0;

		  gcdgb = gcddb = gcdsb = gcdeb = gcdT = 0.0;
		  gcsgb = gcsdb = gcssb = gcseb = gcsT = 0.0;
		  gcggb = gcgdb = gcgsb = gcgeb = gcgT = 0.0;
		  gcbgb = gcbdb = gcbsb = gcbeb = gcbT = 0.0;
		  gcegb = gcedb = gceeb = gcesb = gceT = 0.0;
                  gcTt = 0.0;

		  sxpart = (1.0 - (dxpart = (here->B3SOIPDmode > 0) ? 0.4 : 0.6));

		  goto line900;
		    
	line860:
		  /* evaluate equivalent charge current */

                  cqgate = *(ckt->CKTstate0 + here->B3SOIPDcqg);
                  cqbody = *(ckt->CKTstate0 + here->B3SOIPDcqb);
                  cqdrn = *(ckt->CKTstate0 + here->B3SOIPDcqd);
                  cqsub = *(ckt->CKTstate0 + here->B3SOIPDcqe);
                  cqtemp = *(ckt->CKTstate0 + here->B3SOIPDcqth);

                  here->B3SOIPDcb += cqbody;
                  here->B3SOIPDcd += cqdrn;

		  ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs 
                          - gcgeb * veb - gcgT * delTemp;
		  ceqqb = cqbody - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs
			  - gcbeb * veb - gcbT * delTemp;
		  ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs 
                          - gcdeb * veb - gcdT * delTemp;
		  ceqqe = cqsub - gcegb * vgb + gcedb * vbd + gcesb * vbs
			  - gceeb * veb - gceT * delTemp;;
                  ceqqth = cqtemp - gcTt * delTemp;
	 
		  if (ckt->CKTmode & MODEINITTRAN)
		  {   *(ckt->CKTstate1 + here->B3SOIPDcqe) =  
			    *(ckt->CKTstate0 + here->B3SOIPDcqe);
		      *(ckt->CKTstate1 + here->B3SOIPDcqb) =  
			    *(ckt->CKTstate0 + here->B3SOIPDcqb);
		      *(ckt->CKTstate1 + here->B3SOIPDcqg) =  
			    *(ckt->CKTstate0 + here->B3SOIPDcqg);
		      *(ckt->CKTstate1 + here->B3SOIPDcqd) =  
			    *(ckt->CKTstate0 + here->B3SOIPDcqd);
		      *(ckt->CKTstate1 + here->B3SOIPDcqth) =  
			    *(ckt->CKTstate0 + here->B3SOIPDcqth);
		  }

		  /*
		   *  load current vector
		   */
	line900:

                  m = here->B3SOIPDm;
		  
		  if (here->B3SOIPDmode >= 0)
		  {   Gm = here->B3SOIPDgm;
		      Gmbs = here->B3SOIPDgmbs;
		      GmT = model->B3SOIPDtype * here->B3SOIPDgmT;
		      FwdSum = Gm + Gmbs;
		      RevSum = 0.0;

                      /* v2.2.2 bug fix */
		      cdreq = model->B3SOIPDtype * (here->B3SOIPDcdrain - here->B3SOIPDgds * vds
			    - Gm * vgs - Gmbs * vbs) - GmT * delTemp;

		      /* ceqbs now is compatible with cdreq, ie. going in is +ve */
		      /* Equivalent current source from the diode */
		      ceqbs = here->B3SOIPDcjs;
		      ceqbd = here->B3SOIPDcjd;
		      /* Current going in is +ve */
		      ceqbody = -here->B3SOIPDcbody;


                      ceqgate = here->B3SOIPDcgate;
                      gigg = here->B3SOIPDgigg;
                      gigb = here->B3SOIPDgigb;
                      gigs = here->B3SOIPDgigs;
                      gigd = here->B3SOIPDgigd;
                      gigT = model->B3SOIPDtype * here->B3SOIPDgigT;

		      ceqth = here->B3SOIPDcth;
		      ceqbodcon = here->B3SOIPDcbodcon;

		      gbbg  = -here->B3SOIPDgbgs;
		      gbbdp = -here->B3SOIPDgbds;
		      gbbb  = -here->B3SOIPDgbbs;
		      gbbp  = -here->B3SOIPDgbps;
		      gbbT  = -model->B3SOIPDtype * here->B3SOIPDgbT;
		      gbbsp = - ( gbbg + gbbdp + gbbb + gbbp);

		      gddpg  = -here->B3SOIPDgjdg;
		      gddpdp = -here->B3SOIPDgjdd;
		      gddpb  = -here->B3SOIPDgjdb;
		      gddpT  = -model->B3SOIPDtype * here->B3SOIPDgjdT;
		      gddpsp = - ( gddpg + gddpdp + gddpb);

		      gsspg  = -here->B3SOIPDgjsg;
		      gsspdp = -here->B3SOIPDgjsd;
		      gsspb  = -here->B3SOIPDgjsb;
		      gsspT  = -model->B3SOIPDtype * here->B3SOIPDgjsT;
		      gsspsp = - (gsspg + gsspdp + gsspb );

		      gppb = -here->B3SOIPDgbpbs;
		      gppp = -here->B3SOIPDgbpps;
		      gppT = -model->B3SOIPDtype * here->B3SOIPDgbpT;

                      gTtg  = here->B3SOIPDgtempg;
                      gTtb  = here->B3SOIPDgtempb;
                      gTtdp = here->B3SOIPDgtempd;
                      gTtt  = here->B3SOIPDgtempT;
                      gTtsp = - (gTtg + gTtb + gTtdp);
		  }
		  else
		  {   Gm = -here->B3SOIPDgm;
		      Gmbs = -here->B3SOIPDgmbs;
		      GmT = -model->B3SOIPDtype * here->B3SOIPDgmT;
		      FwdSum = 0.0;
		      RevSum = -(Gm + Gmbs);

                      /* v2.2.2 bug fix */
		      cdreq = -model->B3SOIPDtype * (here->B3SOIPDcdrain + here->B3SOIPDgds*vds
			    + Gm * vgd + Gmbs * vbd) + GmT * delTemp;

		      ceqbs = here->B3SOIPDcjd;
		      ceqbd = here->B3SOIPDcjs;
		      /* Current going in is +ve */
		      ceqbody = -here->B3SOIPDcbody;


                      ceqgate = here->B3SOIPDcgate;
                      gigg = here->B3SOIPDgigg;
                      gigb = here->B3SOIPDgigb;
                      gigs = here->B3SOIPDgigd;
                      gigd = here->B3SOIPDgigs;
                      gigT = model->B3SOIPDtype * here->B3SOIPDgigT;

		      ceqth = here->B3SOIPDcth;
		      ceqbodcon = here->B3SOIPDcbodcon;

		      gbbg  = -here->B3SOIPDgbgs;
		      gbbb  = -here->B3SOIPDgbbs;
		      gbbp  = -here->B3SOIPDgbps;
		      gbbsp = -here->B3SOIPDgbds;
		      gbbT  = -model->B3SOIPDtype * here->B3SOIPDgbT;
		      gbbdp = - ( gbbg + gbbsp + gbbb + gbbp);

		      gddpg  = -here->B3SOIPDgjsg;
		      gddpsp = -here->B3SOIPDgjsd;
		      gddpb  = -here->B3SOIPDgjsb;
		      gddpT  = -model->B3SOIPDtype * here->B3SOIPDgjsT;
		      gddpdp = - (gddpg + gddpsp + gddpb );

		      gsspg  = -here->B3SOIPDgjdg;
		      gsspsp = -here->B3SOIPDgjdd;
		      gsspb  = -here->B3SOIPDgjdb;
		      gsspT  = -model->B3SOIPDtype * here->B3SOIPDgjdT;
		      gsspdp = - ( gsspg + gsspsp + gsspb );

		      gppb = -here->B3SOIPDgbpbs;
		      gppp = -here->B3SOIPDgbpps;
		      gppT = -model->B3SOIPDtype * here->B3SOIPDgbpT;

                      gTtg  = here->B3SOIPDgtempg;
                      gTtb  = here->B3SOIPDgtempb;
                      gTtsp = here->B3SOIPDgtempd;
                      gTtt  = here->B3SOIPDgtempT;
                      gTtdp = - (gTtg + gTtb + gTtsp);
		  }

		   if (model->B3SOIPDtype < 0)
		   {   
		       ceqbodcon = -ceqbodcon;
		       ceqbody = -ceqbody;
                       ceqgate = -ceqgate;
		       ceqbs = -ceqbs;
		       ceqbd = -ceqbd;
		       ceqqg = -ceqqg;
		       ceqqb = -ceqqb;
		       ceqqd = -ceqqd;
		       ceqqe = -ceqqe;
		   }

                   (*(ckt->CKTrhs + here->B3SOIPDbNode) -= m * (ceqbody + ceqqb));

		   (*(ckt->CKTrhs + here->B3SOIPDgNode) -= m * (ceqgate + ceqqg));
		   (*(ckt->CKTrhs + here->B3SOIPDdNodePrime) += m * (ceqbd - cdreq - ceqqd));
		   (*(ckt->CKTrhs + here->B3SOIPDsNodePrime) += m * (cdreq + ceqbs + ceqqg
							  + ceqqb + ceqqd + ceqqe));
		   (*(ckt->CKTrhs + here->B3SOIPDeNode) -= m * ceqqe);

                   if (here->B3SOIPDbodyMod == 1) {
		       (*(ckt->CKTrhs + here->B3SOIPDpNode) += m * ceqbodcon);
                   }

		   if (selfheat) {
		       (*(ckt->CKTrhs + here->B3SOIPDtempNode) -= m * (ceqth + ceqqth));
                   }



                   if (here->B3SOIPDdebugMod != 0)
		   {
	              *(ckt->CKTrhs + here->B3SOIPDvbsNode) = here->B3SOIPDvbseff;
		      *(ckt->CKTrhs + here->B3SOIPDidsNode) = FLOG(here->B3SOIPDids);
		      *(ckt->CKTrhs + here->B3SOIPDicNode) = FLOG(here->B3SOIPDic);
		      *(ckt->CKTrhs + here->B3SOIPDibsNode) = FLOG(here->B3SOIPDibs);
		      *(ckt->CKTrhs + here->B3SOIPDibdNode) = FLOG(here->B3SOIPDibd);
		      *(ckt->CKTrhs + here->B3SOIPDiiiNode) = FLOG(here->B3SOIPDiii); 
                      *(ckt->CKTrhs + here->B3SOIPDigNode) = here->B3SOIPDig;
                      *(ckt->CKTrhs + here->B3SOIPDgiggNode) = here->B3SOIPDgigg;
                      *(ckt->CKTrhs + here->B3SOIPDgigdNode) = here->B3SOIPDgigd;
                      *(ckt->CKTrhs + here->B3SOIPDgigbNode) = here->B3SOIPDgigb;
		      *(ckt->CKTrhs + here->B3SOIPDigidlNode) = here->B3SOIPDigidl;
		      *(ckt->CKTrhs + here->B3SOIPDitunNode) = here->B3SOIPDitun;
		      *(ckt->CKTrhs + here->B3SOIPDibpNode) = here->B3SOIPDibp;
		      *(ckt->CKTrhs + here->B3SOIPDcbbNode) = here->B3SOIPDcbb;
		      *(ckt->CKTrhs + here->B3SOIPDcbdNode) = here->B3SOIPDcbd;
		      *(ckt->CKTrhs + here->B3SOIPDcbgNode) = here->B3SOIPDcbg;
		      *(ckt->CKTrhs + here->B3SOIPDqbfNode) = here->B3SOIPDqbf;
		      *(ckt->CKTrhs + here->B3SOIPDqjsNode) = here->B3SOIPDqjs;
		      *(ckt->CKTrhs + here->B3SOIPDqjdNode) = here->B3SOIPDqjd;
		   }


		   /*
		    *  load y matrix
		    */
		   Gmin = ckt->CKTgmin * 1e-6;

                   *(here->B3SOIPDEdpPtr) += m * gcedb;
                   *(here->B3SOIPDEspPtr) += m * gcesb;
                   *(here->B3SOIPDDPePtr) += m * gcdeb;
                   *(here->B3SOIPDSPePtr) += m * gcseb;
                   *(here->B3SOIPDEgPtr) += m * gcegb;
                   *(here->B3SOIPDGePtr) += m * gcgeb;

                   (*(here->B3SOIPDEbPtr) -= m * (gcegb + gcedb + gcesb + gceeb));
                   (*(here->B3SOIPDGbPtr) -= m * (-gigb + gcggb + gcgdb + gcgsb + gcgeb));
                   (*(here->B3SOIPDDPbPtr) -= m * (-gddpb - Gmbs + gcdgb + gcddb + gcdeb + gcdsb));

                   (*(here->B3SOIPDSPbPtr) -= m * (-gsspb + Gmbs + gcsgb + gcsdb + gcseb + gcssb + Gmin));         


                   (*(here->B3SOIPDBePtr) += m * gcbeb);
                   (*(here->B3SOIPDBgPtr) += m * (-gigg + gcbgb + gbbg));
                   (*(here->B3SOIPDBdpPtr) += (-gigd + gcbdb + gbbdp));
                   (*(here->B3SOIPDBspPtr) += m * (gcbsb + gbbsp - Gmin 
                                         - gigs));
                   (*(here->B3SOIPDBbPtr) += m * (-gigb + gbbb - gcbgb - gcbdb - gcbsb - gcbeb + Gmin));
		   (*(here->B3SOIPDEePtr) += m * gceeb);

		   (*(here->B3SOIPDGgPtr) += m * (gigg + gcggb + ckt->CKTgmin));
		   (*(here->B3SOIPDGdpPtr) += m * (gigd + gcgdb - ckt->CKTgmin));
		   (*(here->B3SOIPDGspPtr) += m * (gcgsb + gigs));

		   (*(here->B3SOIPDDPgPtr) += m * ((Gm + gcdgb) + gddpg - ckt->CKTgmin));
		   (*(here->B3SOIPDDPdpPtr) += m * ((here->B3SOIPDdrainConductance
					 + here->B3SOIPDgds + gddpdp
					 + RevSum + gcddb) + ckt->CKTgmin));
		   (*(here->B3SOIPDDPspPtr) -= m * (-gddpsp + here->B3SOIPDgds + FwdSum - gcdsb));
					 
		   (*(here->B3SOIPDDPdPtr) -= m * here->B3SOIPDdrainConductance);

		   (*(here->B3SOIPDSPgPtr) += m * (gcsgb - Gm + gsspg));
		   (*(here->B3SOIPDSPdpPtr) -= m * (here->B3SOIPDgds - gsspdp + RevSum - gcsdb));

                   (*(here->B3SOIPDSPspPtr) += m * (here->B3SOIPDsourceConductance
                                         + here->B3SOIPDgds + gsspsp
                                         + FwdSum + gcssb
                                         + Gmin));


		   (*(here->B3SOIPDSPsPtr) -= m * here->B3SOIPDsourceConductance);


		   (*(here->B3SOIPDDdPtr) += m * here->B3SOIPDdrainConductance);
		   (*(here->B3SOIPDDdpPtr) -= m * here->B3SOIPDdrainConductance);


		   (*(here->B3SOIPDSsPtr) += m * here->B3SOIPDsourceConductance);
		   (*(here->B3SOIPDSspPtr) -= m * here->B3SOIPDsourceConductance);

		   if (here->B3SOIPDbodyMod == 1) {
		      (*(here->B3SOIPDBpPtr) -= m * gppp);
		      (*(here->B3SOIPDPbPtr) += m * gppb);
		      (*(here->B3SOIPDPpPtr) += m * gppp);
		   }

		   if (selfheat) 
                   {
		      (*(here->B3SOIPDDPtempPtr) += m * (GmT + gddpT + gcdT));
		      (*(here->B3SOIPDSPtempPtr) += m * (-GmT + gsspT + gcsT));
		      (*(here->B3SOIPDBtempPtr) += m * (gbbT + gcbT - gigT));
                      (*(here->B3SOIPDEtempPtr) += m * gceT);
                      (*(here->B3SOIPDGtempPtr) += m * (gcgT + gigT));
		      (*(here->B3SOIPDTemptempPtr) += m * (gTtt  + 1/pParam->B3SOIPDrth + gcTt));
                      (*(here->B3SOIPDTempgPtr) += m * gTtg);
                      (*(here->B3SOIPDTempbPtr) += m * gTtb);
                      (*(here->B3SOIPDTempdpPtr) += m * gTtdp);
                      (*(here->B3SOIPDTempspPtr) += m * gTtsp);
		   }

		   if (here->B3SOIPDdebugMod != 0)
		   {
		      *(here->B3SOIPDVbsPtr) += m * 1;  
		      *(here->B3SOIPDIdsPtr) += m * 1;
		      *(here->B3SOIPDIcPtr) += m * 1;
		      *(here->B3SOIPDIbsPtr) += m * 1;
		      *(here->B3SOIPDIbdPtr) += m * 1;
		      *(here->B3SOIPDIiiPtr) += m * 1;
                      *(here->B3SOIPDIgPtr) += m * 1;
                      *(here->B3SOIPDGiggPtr) += m * 1;
                      *(here->B3SOIPDGigdPtr) += m * 1;
                      *(here->B3SOIPDGigbPtr) += m * 1;
		      *(here->B3SOIPDIgidlPtr) += m * 1;
		      *(here->B3SOIPDItunPtr) += m * 1;
		      *(here->B3SOIPDIbpPtr) += m * 1;
		      *(here->B3SOIPDCbgPtr) += m * 1;
		      *(here->B3SOIPDCbbPtr) += m * 1;
		      *(here->B3SOIPDCbdPtr) += m * 1;
		      *(here->B3SOIPDQbfPtr) += m * 1;
		      *(here->B3SOIPDQjsPtr) += m * 1;
		      *(here->B3SOIPDQjdPtr) += m * 1;

		   }

	line1000:  ;


     }  /* End of Mosfet Instance */
}   /* End of Model Instance */


return(OK);
}

