/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Pin Su, Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
File: b3soiddld.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
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
#define KboQ 8.617087e-5  /*  Kb / q   */
#define Eg300 1.115   /*  energy gap at 300K  */
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
#define DELTA_4 0.02
#define DELT_Vbs0eff 0.02
#define DELT_Vbsmos  0.005
#define DELT_Vbseff  0.005
#define DELT_Xcsat   0.2
#define DELT_Vbs0dio 1e-7
#define DELTA_VFB  0.02
#define DELTA_Vcscv  0.0004
#define DELT_Vbsdio 0.01
#define CONST_2OV3 0.6666666666
#define OFF_Vbsdio  2e-2
#define OFF_Vbs0_dio 2.02e-2
#define QEX_FACT  20


    /* B3SOIDDSmartVbs(Vbs, Old, here, check)
     *  Smart Vbs guess.
     */

static double
B3SOIDDSmartVbs(double New, double Old, B3SOIDDinstance *here, 
                CKTcircuit *ckt, int *check)
{
    NG_IGNORE(Old);
    NG_IGNORE(check);

   /* only do it for floating body and DC */
   if (here->B3SOIDDfloat && (ckt->CKTmode & (MODEDC | MODEDCOP)))
   {
      /* Vbs cannot be negative in DC */
      if (New < 0.0)  New = 0.0;
   }
   return(New);
}


    /* B3SOIDDlimit(vnew,vold)
     *  limits the per-iteration change of any absolute voltage value
     */

static double
B3SOIDDlimit(double vnew, double vold, double limit, int *check)
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
B3SOIDDload(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;
int selfheat;

double Gmin;
double ag0, qgd, qgs, von, cbhat, VgstNVt, ExpVgst = 0.0;
double cdhat, cdreq, ceqbd, ceqbs, ceqqb, ceqqd, ceqqg, ceq, geq;
double arg;
double delvbd, delvbs, delvds, delvgd, delvgs;
double Vfbeff, dVfbeff_dVd, dVfbeff_dVb, V3, V4;
double PhiBSWG, MJSWG;
double gcgdb, gcggb, gcgsb, gcgeb, gcgT;
double gcsdb, gcsgb, gcssb, gcseb, gcsT;
double gcddb, gcdgb, gcdsb, gcdeb, gcdT;
double gcbdb, gcbgb, gcbsb, gcbeb, gcbT;
double gcedb, gcegb, gcesb, gceeb, gceT;
double gcTt, gTtg, gTtb, gTte, gTtdp, gTtt, gTtsp;
double vbd, vbs, vds, vgb, vgd, vgs, vgdo;
#ifndef PREDICTOR
double xfact;
#endif
double vg, vd, vs, vp, ve, vb;
double Vds, Vgs, Vbs, Gmbs, FwdSum, RevSum;
double Vgs_eff, Vfb, dVfb_dVb, dVfb_dVd, dVfb_dT;
double Phis, dPhis_dVb, sqrtPhis, dsqrtPhis_dVb, Vth = 0.0;
double dVth_dVb, dVth_dVd, dVth_dT;
double Vgst, dVgs_eff_dVg;
double n, dn_dVb, Vtm;
double ExpArg, V0;
double ueff = 0.0, dueff_dVg, dueff_dVd, dueff_dVb, dueff_dT;
double Esat, Vdsat = 0.0;
double EsatL, dEsatL_dVg, dEsatL_dVd, dEsatL_dVb, dEsatL_dT;
double dVdsat_dVg, dVdsat_dVb, dVdsat_dVd, dVdsat_dT, Vasat;
double dVasat_dVg, dVasat_dVb, dVasat_dVd, dVasat_dT;
double Va, dVa_dVd, dVa_dVg, dVa_dVb, dVa_dT;
double Vbseff, dVbseff_dVb;
double One_Third_CoxWL, Two_Third_CoxWL, CoxWL;
double T0, dT0_dVg, dT0_dVd, dT0_dVb, dT0_dVc, dT0_dVe, dT0_dT;
double T1, dT1_dVg, dT1_dVd, dT1_dVb, dT1_dVc, dT1_dVe, dT1_dT;
double T2, dT2_dVg, dT2_dVd, dT2_dVb, dT2_dVc, dT2_dVe, dT2_dT;
double T3, dT3_dVg, dT3_dVd, dT3_dVb, dT3_dVc, dT3_dVe, dT3_dT;
double T4, dT4_dVg, dT4_dVd, dT4_dVb, dT4_dVc, dT4_dVe, dT4_dT;
double T5, dT5_dVg, dT5_dVd, dT5_dVb, dT5_dVc, dT5_dVe, dT5_dT;
double T6, dT6_dVg, dT6_dVd, dT6_dVb, dT6_dVc, dT6_dVe, dT6_dT;
double T7;
double T8;
double T9;
double T10;
double T11, T12;
double Abulk, dAbulk_dVb, Abulk0, dAbulk0_dVb;
double VACLM, dVACLM_dVg, dVACLM_dVd, dVACLM_dVb, dVACLM_dT;
double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb, dVADIBL_dT;
double Xdep, dXdep_dVb, lt1, dlt1_dVb, ltw, dltw_dVb;
double Delt_vth, dDelt_vth_dVb, dDelt_vth_dT;
double Theta0, dTheta0_dVb;
double TempRatio, tmp1, tmp2, tmp3, tmp4;
double DIBL_Sft, dDIBL_Sft_dVd, Lambda, dLambda_dVg;
double a1;
 
double Vgsteff = 0.0, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb;
double dVgsteff_dVe, dVgsteff_dT;
double Vdseff = 0.0, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb, dVdseff_dT;
double VdseffCV, dVdseffCV_dVg, dVdseffCV_dVd, dVdseffCV_dVb;
double diffVds;
double dAbulk_dVg, dn_dVd ;
double beta, dbeta_dVg, dbeta_dVd, dbeta_dVb, dbeta_dT;
double gche, dgche_dVg, dgche_dVd, dgche_dVb, dgche_dT;
double fgche1, dfgche1_dVg, dfgche1_dVd, dfgche1_dVb, dfgche1_dT;
double fgche2, dfgche2_dVg, dfgche2_dVd, dfgche2_dVb, dfgche2_dT;
double Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb, dIdl_dT;
double Ids = 0.0, Gm, Gds = 0.0, Gmb;
double CoxWovL;
double Rds, dRds_dVg, dRds_dVb, dRds_dT, WVCox, WVCoxRds;
double Vgst2Vtm, dVgst2Vtm_dT, VdsatCV, dVdsatCV_dVg, dVdsatCV_dVb;
double Leff, Weff, dWeff_dVg, dWeff_dVb;
double AbulkCV, dAbulkCV_dVb;
double qgdo, qgso, cgdo, cgso;
 
double dxpart, sxpart;
 
struct b3soiddSizeDependParam *pParam;
int ByPass, Check, ChargeComputationNeeded = 0, error;
 
double gbbsp, gbbdp, gbbg, gbbb, gbbe, gbbp, gbbT;
double gddpsp, gddpdp, gddpg, gddpb, gddpe, gddpT;
double gsspsp, gsspdp, gsspg, gsspb, gsspe, gsspT;
double Gbpbs, Gbpgs, Gbpds, Gbpes, Gbpps, GbpT;
double ves, ved, veb, vge = 0.0, delves, vedo, delved;
double vps, vpd, Vps, delvps;
double Vbd, Ves, Vesfb, sqrtXdep, DeltVthtemp, dDeltVthtemp_dT;
double Vbp, dVbp_dVp, dVbp_dVb, dVbp_dVg, dVbp_dVd, dVbp_dVe, dVbp_dT;
double Vpsdio, dVpsdio_dVg, dVpsdio_dVd, dVpsdio_dVe, dVpsdio_dVp, dVpsdio_dT;
double DeltVthw, dDeltVthw_dVb, dDeltVthw_dT;
double dVbseff_dVd, dVbseff_dVe, dVbseff_dT;
double dVdsat_dVc, dVasat_dVc, dVACLM_dVc, dVADIBL_dVc, dVa_dVc;
double dfgche1_dVc, dfgche2_dVc, dgche_dVc, dVdseff_dVc, dIdl_dVc;
double Gm0, Gds0, Gmb0, GmT0, Gmc, Gme, GmT, dVbseff_dVg;
double dDIBL_Sft_dVb, BjtA, dBjtA_dVd;
double diffVdsii  ;
double Idgidl = 0.0, Gdgidld, Gdgidlg, Isgidl = 0.0, Gsgidlg;
double Gjsd, Gjsb, GjsT, Gjdd, Gjdb, GjdT;
double Ibp = 0.0, Iii = 0.0, Giid, Giig, Giib, Giie, GiiT, Gcd, Gcb, GcT;
double ceqbody, ceqbodcon = 0.0;
double gppg = 0.0, gppdp = 0.0, gppb = 0.0, gppe = 0.0, gppp = 0.0;
double gppsp = 0.0, gppT;
double delTemp, deldelTemp, Temp;
double ceqth, ceqqth;
double K1;
double qjs = 0.0, gcjsbs, gcjsT;
double qjd = 0.0, gcjdbs, gcjdds, gcjdT;
double qge;
double ceqqe;
double ni, Eg, Cbox, Nfb, CboxWL;
double cjsbs;
double Qbf0, Qsicv, dVfbeff_dVrg, Cbe = 0.0;
double qinv = 0.0, qgate = 0.0, qbody = 0.0, qdrn = 0.0, qsrc, qsub = 0.0;
double cqgate, cqbody = 0.0, cqdrn = 0.0, cqsub, cqtemp;
double Cgg, Cgd, Cgb, Cge;
double Csg, Csd, Csb, Cse, Cbg = 0.0, Cbd = 0.0, Cbb = 0.0;
double Cgg1, Cgb1, Cgd1, Csg1, Csd1, Csb1;
double Vbs0t = 0.0, dVbs0t_dT ;
double Vbs0 = 0.0 ,dVbs0_dVe, dVbs0_dT;
double Vbs0eff = 0.0 ,dVbs0eff_dVg ,dVbs0eff_dVd ,dVbs0eff_dVe, dVbs0eff_dT;
double Vbs0teff = 0.0,dVbs0teff_dVg ,dVbs0teff_dVd, dVbs0teff_dVe;
double dVbs0teff_dT;
double Vbsdio = 0.0, dVbsdio_dVg, dVbsdio_dVd, dVbsdio_dVe, dVbsdio_dVb;
double dVbsdio_dT;
double Vthfd = 0.0 ,dVthfd_dVd ,dVthfd_dVe, dVthfd_dT;
double Vbs0mos = 0.0 ,dVbs0mos_dVe, dVbs0mos_dT;
double Vbsmos ,dVbsmos_dVg ,dVbsmos_dVb ,dVbsmos_dVd, dVbsmos_dVe, dVbsmos_dT;
double Abeff ,dAbeff_dVg ,dAbeff_dVb, dAbeff_dVc;
double Vcs ,dVcs_dVg ,dVcs_dVb ,dVcs_dVd ,dVcs_dVe, dVcs_dT;
double Xcsat = 0.0, dXcsat_dVg, dXcsat_dVc;
double Vdsatii ,dVdsatii_dVg ,dVdsatii_dVd, dVdsatii_dVb, dVdsatii_dT;
double Vdseffii ,dVdseffii_dVg ,dVdseffii_dVd, dVdseffii_dVb, dVdseffii_dT;
double VcsCV = 0.0 ,dVcsCV_dVg = 0.0 ,dVcsCV_dVb = 0.0;
double dVcsCV_dVd = 0.0  ,dVcsCV_dVc = 0.0;
double VdsCV = 0.0 ,dVdsCV_dVg = 0.0 ,dVdsCV_dVb = 0.0;
double dVdsCV_dVd = 0.0  ,dVdsCV_dVc = 0.0;
double Phisc ,dPhisc_dVg ,dPhisc_dVb ,dPhisc_dVd,  dPhisc_dVc;
double Phisd ,dPhisd_dVg ,dPhisd_dVb ,dPhisd_dVd,  dPhisd_dVc;
double sqrtPhisc;
double sqrtPhisd;
double Xc = 0.0 ,dXc_dVg = 0.0 ,dXc_dVb = 0.0 ,dXc_dVd = 0.0 ,dXc_dVc = 0.0;
double Ibjt = 0.0 ,dIbjt_dVb ,dIbjt_dVd ,dIbjt_dT = 0.0;
double Ibs1 ,dIbs1_dVb ,dIbs1_dT = 0.0;
double Ibs2 ,dIbs2_dVb ,dIbs2_dT = 0.0;
double Ibs3 ,dIbs3_dVb ,dIbs3_dVd, dIbs3_dT = 0.0;
double Ibs4 ,dIbs4_dVb ,dIbs4_dT = 0.0;
double Ibd1 ,dIbd1_dVb ,dIbd1_dVd ,dIbd1_dT = 0.0;
double Ibd2 ,dIbd2_dVb ,dIbd2_dVd ,dIbd2_dT = 0.0;
double Ibd3 ,dIbd3_dVb ,dIbd3_dVd ,dIbd3_dT = 0.0;
double Ibd4 ,dIbd4_dVb ,dIbd4_dVd ,dIbd4_dT = 0.0;
double ExpVbs1, dExpVbs1_dVb, dExpVbs1_dT = 0.0;
double ExpVbs2, dExpVbs2_dVb, dExpVbs2_dT = 0.0;
double ExpVbs4 = 0.0, dExpVbs4_dVb = 0.0, dExpVbs4_dT = 0.0;
double ExpVbd1, dExpVbd1_dVb, dExpVbd1_dT = 0.0;
double ExpVbd2, dExpVbd2_dVb, dExpVbd2_dT = 0.0;
double ExpVbd4 = 0.0, dExpVbd4_dVb = 0.0, dExpVbd4_dT = 0.0;
double WTsi, NVtm1, NVtm2;
double Ic = 0.0;
double Ibs = 0.0;
double Ibd = 0.0;
double Nomi ,dNomi_dVg ,dNomi_dVb ,dNomi_dVd ,dNomi_dVc;
double Denomi ,dDenomi_dVg ,dDenomi_dVd ,dDenomi_dVb ,dDenomi_dVc, dDenomi_dT;
double Qbf = 0.0 ,dQbf_dVg = 0.0 ,dQbf_dVb = 0.0 ,dQbf_dVd = 0.0;
double dQbf_dVc = 0.0 ,dQbf_dVe = 0.0;
double Qsubs1 = 0.0 ,dQsubs1_dVg  ,dQsubs1_dVb ,dQsubs1_dVd ,dQsubs1_dVc;
double Qsubs2 = 0.0 ,dQsubs2_dVg  ,dQsubs2_dVb ,dQsubs2_dVd ,dQsubs2_dVc ,dQsubs2_dVe;
double Qsub0 = 0.0  ,dQsub0_dVg   ,dQsub0_dVb  ,dQsub0_dVd ;
double Qac0 = 0.0 ,dQac0_dVb   ,dQac0_dVd;
double Qdep0 ,dQdep0_dVb;
double Qe1 = 0.0 , dQe1_dVg ,dQe1_dVb, dQe1_dVd, dQe1_dVe, dQe1_dT;
double Ce1g ,Ce1b ,Ce1d ,Ce1e, Ce1T;
double Ce2g ,Ce2b ,Ce2d ,Ce2e, Ce2T;
double Qe2 = 0.0 , dQe2_dVg ,dQe2_dVb, dQe2_dVd, dQe2_dVe, dQe2_dT;
double dQbf_dVrg = 0.0, dQac0_dVrg, dQsub0_dVrg; 
double dQsubs2_dVrg, dQbf0_dVe, dQbf0_dT;

/*  for self-heating  */
double vbi, vfbb, phi, sqrtPhi, Xdep0, jbjt, jdif, jrec, jtun, u0temp, vsattemp;
double rds0, ua, ub, uc;
double dvbi_dT, dvfbb_dT, djbjt_dT, djdif_dT, djrec_dT, djtun_dT, du0temp_dT;
double dvsattemp_dT, drds0_dT, dua_dT, dub_dT, duc_dT, dni_dT, dVtm_dT;
double dVfbeff_dT, dQac0_dT, dQsub0_dT;
double dQbf_dT = 0.0, dVdsCV_dT = 0.0, dPhisd_dT;
double dNomi_dT, dXc_dT = 0.0, dQsubs1_dT, dQsubs2_dT;
double dVcsCV_dT = 0.0, dPhisc_dT, dQsicv_dT;
double CbT, CsT, CgT;

double Qex, dQex_dVg, dQex_dVb, dQex_dVd, dQex_dVe, dQex_dT;

/* clean up last */
FILE *fpdebug = NULL;
/* end clean up */
int nandetect;
static int nanfound = 0;
char nanmessage [12];

double m;


 for (; model != NULL; model = B3SOIDDnextModel(model))
{    for (here = B3SOIDDinstances(model); here != NULL; 
          here = B3SOIDDnextInstance(here))
     {     
          Check = 0;
          ByPass = 0;
          selfheat = (model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0);
	  pParam = here->pParam;

          if (here->B3SOIDDdebugMod > 3)
          {
             if (model->B3SOIDDtype > 0)
                fpdebug = fopen("b3soiddn.log", "a");
             else
                fpdebug = fopen("b3soiddp.log", "a");

             fprintf(fpdebug, "******* Time : %.5e ******* Device:  %s  Iteration:  %d\n",
                     ckt->CKTtime, here->B3SOIDDname, here->B3SOIDDiterations);
          }

          if ((ckt->CKTmode & MODEINITSMSIG))
	  {   vbs = *(ckt->CKTstate0 + here->B3SOIDDvbs);
              vgs = *(ckt->CKTstate0 + here->B3SOIDDvgs);
              ves = *(ckt->CKTstate0 + here->B3SOIDDves);
              vps = *(ckt->CKTstate0 + here->B3SOIDDvps);
              vds = *(ckt->CKTstate0 + here->B3SOIDDvds);
              delTemp = *(ckt->CKTstate0 + here->B3SOIDDdeltemp);

              vg = *(ckt->CKTrhsOld + here->B3SOIDDgNode);
              vd = *(ckt->CKTrhsOld + here->B3SOIDDdNodePrime);
              vs = *(ckt->CKTrhsOld + here->B3SOIDDsNodePrime);
              vp = *(ckt->CKTrhsOld + here->B3SOIDDpNode);
              ve = *(ckt->CKTrhsOld + here->B3SOIDDeNode);
              vb = *(ckt->CKTrhsOld + here->B3SOIDDbNode);

              if (here->B3SOIDDdebugMod > 2)
              {
                  fprintf(fpdebug, "... INIT SMSIG ...\n");
              }
              if (here->B3SOIDDdebugMod > 0)
              {
                 fprintf(stderr,"DC op. point converge with %d iterations\n", 
		         here->B3SOIDDiterations);
              }
          }
	  else if ((ckt->CKTmode & MODEINITTRAN))
	  {   vbs = *(ckt->CKTstate1 + here->B3SOIDDvbs);
              vgs = *(ckt->CKTstate1 + here->B3SOIDDvgs);
              ves = *(ckt->CKTstate1 + here->B3SOIDDves);
              vps = *(ckt->CKTstate1 + here->B3SOIDDvps);
              vds = *(ckt->CKTstate1 + here->B3SOIDDvds);
              delTemp = *(ckt->CKTstate1 + here->B3SOIDDdeltemp);

              vg = *(ckt->CKTrhsOld + here->B3SOIDDgNode);
              vd = *(ckt->CKTrhsOld + here->B3SOIDDdNodePrime);
              vs = *(ckt->CKTrhsOld + here->B3SOIDDsNodePrime);
              vp = *(ckt->CKTrhsOld + here->B3SOIDDpNode);
              ve = *(ckt->CKTrhsOld + here->B3SOIDDeNode);
              vb = *(ckt->CKTrhsOld + here->B3SOIDDbNode);

              if (here->B3SOIDDdebugMod > 2)
              {
                 fprintf(fpdebug, "... Init Transient ....\n");
              }
              if (here->B3SOIDDdebugMod > 0)
              {
                 fprintf(stderr, "Transient operation point converge with %d iterations\n",
here->B3SOIDDiterations);  
              }
              here->B3SOIDDiterations = 0;
          }
	  else if ((ckt->CKTmode & MODEINITJCT) && !here->B3SOIDDoff)
	  {   vds = model->B3SOIDDtype * here->B3SOIDDicVDS;
              vgs = model->B3SOIDDtype * here->B3SOIDDicVGS;
              ves = model->B3SOIDDtype * here->B3SOIDDicVES;
              vbs = model->B3SOIDDtype * here->B3SOIDDicVBS;
              vps = model->B3SOIDDtype * here->B3SOIDDicVPS;

	      vg = vd = vs = vp = ve = 0.0;

              here->B3SOIDDiterations = 0;  /*  initialize iteration number  */

              delTemp = 0.0;
	      here->B3SOIDDphi = pParam->B3SOIDDphi;


              if (here->B3SOIDDdebugMod > 2)
		fprintf(fpdebug, "... INIT JCT ...\n");

	      if ((vds == 0.0) && (vgs == 0.0) && (vbs == 0.0) && 
	         ((ckt->CKTmode & (MODETRAN | MODEAC|MODEDCOP |
		   MODEDCTRANCURVE)) || (!(ckt->CKTmode & MODEUIC))))
	      {   vbs = 0.0;
		  vgs = model->B3SOIDDtype*0.1 + pParam->B3SOIDDvth0;
		  vds = 0.0;
		  ves = 0.0;
		  vps = 0.0;
	      }
	  }
	  else if ((ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) && 
		  (here->B3SOIDDoff)) 
	  {    delTemp = vps = vbs = vgs = vds = ves = 0.0;
               vg = vd = vs = vp = ve = 0.0;
               here->B3SOIDDiterations = 0;  /*  initialize iteration number  */
	  }
	  else
	  {
#ifndef PREDICTOR
	       if ((ckt->CKTmode & MODEINITPRED))
	       {   xfact = ckt->CKTdelta / ckt->CKTdeltaOld[1];
		   *(ckt->CKTstate0 + here->B3SOIDDvbs) = 
			 *(ckt->CKTstate1 + here->B3SOIDDvbs);
		   vbs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDvbs))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDvbs)));
		   *(ckt->CKTstate0 + here->B3SOIDDvgs) = 
			 *(ckt->CKTstate1 + here->B3SOIDDvgs);
		   vgs = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDvgs))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDvgs)));
		   *(ckt->CKTstate0 + here->B3SOIDDves) = 
			 *(ckt->CKTstate1 + here->B3SOIDDves);
		   ves = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDves))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDves)));
		   *(ckt->CKTstate0 + here->B3SOIDDvps) = 
			 *(ckt->CKTstate1 + here->B3SOIDDvps);
		   vps = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDvps))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDvps)));
		   *(ckt->CKTstate0 + here->B3SOIDDvds) = 
			 *(ckt->CKTstate1 + here->B3SOIDDvds);
		   vds = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDvds))
			 - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDvds)));
		   *(ckt->CKTstate0 + here->B3SOIDDvbd) = 
			 *(ckt->CKTstate0 + here->B3SOIDDvbs)
			 - *(ckt->CKTstate0 + here->B3SOIDDvds);

                   *(ckt->CKTstate0 + here->B3SOIDDvg) = *(ckt->CKTstate1 + here->B3SOIDDvg);
                   *(ckt->CKTstate0 + here->B3SOIDDvd) = *(ckt->CKTstate1 + here->B3SOIDDvd);
                   *(ckt->CKTstate0 + here->B3SOIDDvs) = *(ckt->CKTstate1 + here->B3SOIDDvs);
                   *(ckt->CKTstate0 + here->B3SOIDDvp) = *(ckt->CKTstate1 + here->B3SOIDDvp);
                   *(ckt->CKTstate0 + here->B3SOIDDve) = *(ckt->CKTstate1 + here->B3SOIDDve);

                   /* Only predict ve */
                   ve = (1.0 + xfact)* (*(ckt->CKTstate1 + here->B3SOIDDve))

                        - (xfact * (*(ckt->CKTstate2 + here->B3SOIDDve)));
                   /* Then update vg, vs, vb, vd, vp base on ve */
                   vs = ve - model->B3SOIDDtype * ves;
                   vg = model->B3SOIDDtype * vgs + vs;
                   vd = model->B3SOIDDtype * vds + vs;
                   vb = model->B3SOIDDtype * vbs + vs;
                   vp = model->B3SOIDDtype * vps + vs;

		   delTemp = (1.0 + xfact)* (*(ckt->CKTstate1 +
			 here->B3SOIDDdeltemp))-(xfact * (*(ckt->CKTstate2 +
			 here->B3SOIDDdeltemp)));

		   if (selfheat)
		   {
		       here->B3SOIDDphi = 2.0 * here->B3SOIDDvtm
					* log (pParam->B3SOIDDnpeak /
					       here->B3SOIDDni); 
		   }

		   if (here->B3SOIDDdebugMod > 0)
		   {
                      fprintf(stderr, "Time = %.6e converge with %d iterations\n", ckt->CKTtime, here->B3SOIDDiterations);  
                   }
		   if (here->B3SOIDDdebugMod > 2)
		   {
		      fprintf(fpdebug, "... PREDICTOR calculation ....\n");
		   }
                   here->B3SOIDDiterations = 0;
	       }
	       else
	       {
#endif /* PREDICTOR */

                   vg = B3SOIDDlimit(*(ckt->CKTrhsOld + here->B3SOIDDgNode),
                                 *(ckt->CKTstate0 + here->B3SOIDDvg), 3.0, &Check);
                   vd = B3SOIDDlimit(*(ckt->CKTrhsOld + here->B3SOIDDdNodePrime),
                                 *(ckt->CKTstate0 + here->B3SOIDDvd), 3.0, &Check);
                   vs = B3SOIDDlimit(*(ckt->CKTrhsOld + here->B3SOIDDsNodePrime),
                                 *(ckt->CKTstate0 + here->B3SOIDDvs), 3.0, &Check);
                   vp = B3SOIDDlimit(*(ckt->CKTrhsOld + here->B3SOIDDpNode),
                                 *(ckt->CKTstate0 + here->B3SOIDDvp), 3.0, &Check);
                   ve = B3SOIDDlimit(*(ckt->CKTrhsOld + here->B3SOIDDeNode),
                                 *(ckt->CKTstate0 + here->B3SOIDDve), 3.0, &Check);
                   delTemp = *(ckt->CKTrhsOld + here->B3SOIDDtempNode);

		   vbs = model->B3SOIDDtype * (*(ckt->CKTrhsOld+here->B3SOIDDbNode)
                                - *(ckt->CKTrhsOld+here->B3SOIDDsNodePrime));

		   vps = model->B3SOIDDtype * (vp - vs);
		   vgs = model->B3SOIDDtype * (vg - vs);
		   ves = model->B3SOIDDtype * (ve - vs);
		   vds = model->B3SOIDDtype * (vd - vs);

		   if (here->B3SOIDDdebugMod > 2)
		   {
		      fprintf(fpdebug, "... DC calculation ....\n");
fprintf(fpdebug, "Vg = %.10f; Vb = %.10f; Vs = %.10f\n",
			 *(ckt->CKTrhsOld + here->B3SOIDDgNode),
			 *(ckt->CKTrhsOld + here->B3SOIDDbNode),
			 *(ckt->CKTrhsOld + here->B3SOIDDsNode));
fprintf(fpdebug, "Vd = %.10f; Vsp = %.10f; Vdp = %.10f\n",
			 *(ckt->CKTrhsOld + here->B3SOIDDdNode),
			 *(ckt->CKTrhsOld + here->B3SOIDDsNodePrime),
			 *(ckt->CKTrhsOld + here->B3SOIDDdNodePrime));
fprintf(fpdebug, "Ve = %.10f; Vp = %.10f; delTemp = %.10f\n",
			 *(ckt->CKTrhsOld + here->B3SOIDDeNode),
			 *(ckt->CKTrhsOld + here->B3SOIDDpNode),
			 *(ckt->CKTrhsOld + here->B3SOIDDtempNode));
                     
		   }

#ifndef PREDICTOR
	       }
#endif /* PREDICTOR */

	       vbd = vbs - vds;
	       vgd = vgs - vds;
               ved = ves - vds;
	       vgdo = *(ckt->CKTstate0 + here->B3SOIDDvgs)
		    - *(ckt->CKTstate0 + here->B3SOIDDvds);
	       vedo = *(ckt->CKTstate0 + here->B3SOIDDves)
		    - *(ckt->CKTstate0 + here->B3SOIDDvds);
	       delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIDDvbs);
	       delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIDDvbd);
	       delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIDDvgs);
	       delves = ves - *(ckt->CKTstate0 + here->B3SOIDDves);
	       delvps = vps - *(ckt->CKTstate0 + here->B3SOIDDvps);
	       deldelTemp = delTemp - *(ckt->CKTstate0 + here->B3SOIDDdeltemp);
	       delvds = vds - *(ckt->CKTstate0 + here->B3SOIDDvds);
	       delvgd = vgd - vgdo;
               delved = ved - vedo;

	       if (here->B3SOIDDmode >= 0) 
	       {   
                   cdhat = here->B3SOIDDcd + (here->B3SOIDDgm-here->B3SOIDDgjdg) * delvgs
                         + (here->B3SOIDDgds - here->B3SOIDDgjdd) * delvds
                         + (here->B3SOIDDgmbs - here->B3SOIDDgjdb) * delvbs
                         + (here->B3SOIDDgme - here->B3SOIDDgjde) * delves
			 + (here->B3SOIDDgmT - here->B3SOIDDgjdT) * deldelTemp;
	       }
	       else
	       {   
                   cdhat = here->B3SOIDDcd + (here->B3SOIDDgm-here->B3SOIDDgjdg) * delvgd
                         - (here->B3SOIDDgds - here->B3SOIDDgjdd) * delvds
                         + (here->B3SOIDDgmbs - here->B3SOIDDgjdb) * delvbd
                         + (here->B3SOIDDgme - here->B3SOIDDgjde) * delved
                         + (here->B3SOIDDgmT - here->B3SOIDDgjdT) * deldelTemp;

	       }
	       cbhat = here->B3SOIDDcb + here->B3SOIDDgbgs * delvgs
		     + here->B3SOIDDgbbs * delvbs + here->B3SOIDDgbds * delvds
                     + here->B3SOIDDgbes * delves + here->B3SOIDDgbps * delvps
		     + here->B3SOIDDgbT * deldelTemp;

#ifndef NOBYPASS
	   /* following should be one big if connected by && all over
	    * the place, but some C compilers can't handle that, so
	    * we split it up here to let them digest it in stages
	    */

	       if (here->B3SOIDDdebugMod > 3)
               {
fprintf(fpdebug, "Convergent Criteria : vbs %d  vds %d  vgs %d  ves %d  vps %d  temp %d\n",
	((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
	fabs(*(ckt->CKTstate0+here->B3SOIDDvbs))) + ckt->CKTvoltTol))) ? 1 : 0,
	((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
	fabs(*(ckt->CKTstate0+here->B3SOIDDvds))) + ckt->CKTvoltTol))) ? 1 : 0,
	((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
	fabs(*(ckt->CKTstate0+here->B3SOIDDvgs))) + ckt->CKTvoltTol))) ? 1 : 0,
	((fabs(delves) < (ckt->CKTreltol * MAX(fabs(ves),
	fabs(*(ckt->CKTstate0+here->B3SOIDDves))) + ckt->CKTvoltTol))) ? 1 : 0,
	((fabs(delvps) < (ckt->CKTreltol * MAX(fabs(vps),
	fabs(*(ckt->CKTstate0+here->B3SOIDDvps))) + ckt->CKTvoltTol))) ? 1 : 0,
	((fabs(deldelTemp) < (ckt->CKTreltol * MAX(fabs(delTemp),
	fabs(*(ckt->CKTstate0+here->B3SOIDDdeltemp))) + ckt->CKTvoltTol*1e4))) ? 1 : 0);
fprintf(fpdebug, "delCd %.4e, delCb %.4e\n",  fabs(cdhat - here->B3SOIDDcd) ,
        fabs(cbhat - here->B3SOIDDcb));

               }
	       if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass) && Check == 0)
	       if ((fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDvbs))) + ckt->CKTvoltTol))  )
	       if ((fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDvbd))) + ckt->CKTvoltTol))  )
	       if ((fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDvgs))) + ckt->CKTvoltTol)))
	       if ((fabs(delves) < (ckt->CKTreltol * MAX(fabs(ves),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDves))) + ckt->CKTvoltTol)))
	       if ( (here->B3SOIDDbodyMod == 0) || (here->B3SOIDDbodyMod == 2) ||
                  (fabs(delvps) < (ckt->CKTreltol * MAX(fabs(vps),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDvps))) + ckt->CKTvoltTol)) )
	       if ( (here->B3SOIDDtempNode == 0)  ||
                  (fabs(deldelTemp) < (ckt->CKTreltol * MAX(fabs(delTemp),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDdeltemp)))
		   + ckt->CKTvoltTol*1e4)))
	       if ((fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
		   fabs(*(ckt->CKTstate0+here->B3SOIDDvds))) + ckt->CKTvoltTol)))
	       if ((fabs(cdhat - here->B3SOIDDcd) < ckt->CKTreltol 
		   * MAX(fabs(cdhat),fabs(here->B3SOIDDcd)) + ckt->CKTabstol)) 
	       if ((fabs(cbhat - here->B3SOIDDcb) < ckt->CKTreltol 
		   * MAX(fabs(cbhat),fabs(here->B3SOIDDcb)) + ckt->CKTabstol) )
	       {   /* bypass code */
	           vbs = *(ckt->CKTstate0 + here->B3SOIDDvbs);
	           vbd = *(ckt->CKTstate0 + here->B3SOIDDvbd);
	           vgs = *(ckt->CKTstate0 + here->B3SOIDDvgs);
	           ves = *(ckt->CKTstate0 + here->B3SOIDDves);
	           vps = *(ckt->CKTstate0 + here->B3SOIDDvps);
	           vds = *(ckt->CKTstate0 + here->B3SOIDDvds);
	           delTemp = *(ckt->CKTstate0 + here->B3SOIDDdeltemp);

		   /*  calculate Vds for temperature conductance calculation
		       in bypass (used later when filling Temp node matrix)  */
		   Vds = here->B3SOIDDmode > 0 ? vds : -vds;

	           vgd = vgs - vds;
	           vgb = vgs - vbs;
                   veb = ves - vbs;

	           if (here->B3SOIDDdebugMod > 2)
	           {
fprintf(stderr, "Bypass for %s...\n", here->B3SOIDDname);
	    	      fprintf(fpdebug, "... By pass  ....\n");
		      fprintf(fpdebug, "vgs=%.4f, vds=%.4f, vbs=%.4f, ",
			   vgs, vds, vbs);
	    	      fprintf(fpdebug, "ves=%.4f, vps=%.4f\n", ves, vps);
		   }
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
		       von = here->B3SOIDDvon;

			if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
		       {
			   here->B3SOIDDdum1 = here->B3SOIDDdum2 = here->B3SOIDDdum3 = 0.0;
                           here->B3SOIDDdum4 = here->B3SOIDDdum5 = 0.0;
			   Qac0 = Qsub0 = Qsubs1 = Qsubs2 = Qbf = Qe1 = Qe2 = 0.0;
			   qjs = qjd = Cbg = Cbb = Cbd = Cbe = Xc = qdrn = qgate = 0.0;
			   qbody = qsub = 0.0;
		       }

		       if (here->B3SOIDDdebugMod > 2) {
			      fprintf(fpdebug, "Limited : vgs = %.8f\n", vgs);
			      fprintf(fpdebug, "Limited : vds = %.8f\n", vds);
		       }

                       if (*(ckt->CKTstate0 + here->B3SOIDDvds) >= 0.0)
                          T0 = *(ckt->CKTstate0 + here->B3SOIDDvbs);
                       else
                          T0 = *(ckt->CKTstate0 + here->B3SOIDDvbd);

		       if (here->B3SOIDDdebugMod > 2)
			  fprintf(fpdebug, "Before lim : vbs = %.8f, after = ", T0);

		       if (vds >= 0.0) 
		       {   
		           vbs = B3SOIDDlimit(vbs, T0, 0.2, &Check);
                           vbs = B3SOIDDSmartVbs(vbs, T0, here, ckt, &Check);
			   vbd = vbs - vds;
                           vb = model->B3SOIDDtype * vbs + vs;
		           if (here->B3SOIDDdebugMod > 2)
			      fprintf(fpdebug, "%.8f\n", vbs);
		       } else 
		       {   
		           vbd = B3SOIDDlimit(vbd, T0, 0.2, &Check);
                           vbd = B3SOIDDSmartVbs(vbd, T0, here, ckt, &Check);
			   vbs = vbd + vds;
                           vb = model->B3SOIDDtype * vbs + vd;
		           if (here->B3SOIDDdebugMod > 2)
			      fprintf(fpdebug, "%.8f\n", vbd);
		       }

		       delTemp =B3SOIDDlimit(delTemp, *(ckt->CKTstate0 + here->B3SOIDDdeltemp),5.0,&Check);

                  }

/*  Calculate temperature dependent values for self-heating effect  */
		  Temp = delTemp + ckt->CKTtemp;
/* for debugging  
  Temp = ckt->CKTtemp;
  selfheat = 1;
  if (here->B3SOIDDname[1] == '2')
  {
     Temp += 0.01; 
  } */
		  TempRatio = Temp / model->B3SOIDDtnom;

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

                      T0 = log(1.0e20 * pParam->B3SOIDDnpeak / (ni * ni));
		      vbi = Vtm * T0;
                      dvbi_dT = KboQ * T0 + Vtm * (-2.0 * dni_dT / ni);

		      if (pParam->B3SOIDDnsub > 0) {
                         T0 = log(pParam->B3SOIDDnpeak / pParam->B3SOIDDnsub);
		         vfbb = -model->B3SOIDDtype * Vtm*T0;
                         dvfbb_dT = -model->B3SOIDDtype * KboQ*T0;
                      } 
		      else {
                         T0 = log(-pParam->B3SOIDDnpeak*pParam->B3SOIDDnsub/ni/ni);
		         vfbb = -model->B3SOIDDtype * Vtm*T0;
                         dvfbb_dT = -model->B3SOIDDtype *
                                   (KboQ * T0 + Vtm * 2.0 * dni_dT / ni);
                      }

/*		      phi = 2.0 * Vtm * log(pParam->B3SOIDDnpeak / ni);  */
		      phi = here->B3SOIDDphi;
		      sqrtPhi = sqrt(phi);
		      Xdep0 = sqrt(2.0 * EPSSI / (Charge_q
				         * pParam->B3SOIDDnpeak * 1.0e6))
				         * sqrtPhi;
		      /*  Save the values below for phi calculation in B3SOIDDaccept()  */
		      here->B3SOIDDvtm = Vtm;
		      here->B3SOIDDni = ni;

                      /*  Use dTx_dVe variables to act as dTx_dT variables  */

                      T8 = 1 / model->B3SOIDDtnom;
                      T7 = model->B3SOIDDxbjt / pParam->B3SOIDDndiode;
		      T0 = pow(TempRatio, T7);
                      dT0_dVe = T7 * pow(TempRatio, T7 - 1.0) * T8;

                      T7 = model->B3SOIDDxdif / pParam->B3SOIDDndiode;
		      T1 = pow(TempRatio, T7);
                      dT1_dVe = T7 * pow(TempRatio, T7 - 1.0) * T8;

                      T7 = model->B3SOIDDxrec / pParam->B3SOIDDndiode / 2.0;
		      T2 = pow(TempRatio, T7);
                      dT2_dVe = T7 * pow(TempRatio, T7 - 1.0) * T8;

		      T3 = TempRatio - 1.0;
		      T4 = Eg300 / pParam->B3SOIDDndiode / Vtm * T3;
                      dT4_dVe = Eg300 / pParam->B3SOIDDndiode / Vtm / Vtm *
                                (Vtm * T8 - T3 * KboQ);
		      T5 = exp(T4);
                      dT5_dVe = dT4_dVe * T5;
		      T6 = sqrt(T5);
                      dT6_dVe = 0.5 / T6 * dT5_dVe;

		      jbjt = pParam->B3SOIDDisbjt * T0 * T5;
		      jdif = pParam->B3SOIDDisdif * T1 * T5;
		      jrec = pParam->B3SOIDDisrec * T2 * T6;
                      djbjt_dT = pParam->B3SOIDDisbjt * (T0 * dT5_dVe + T5 * dT0_dVe);
                      djdif_dT = pParam->B3SOIDDisdif * (T1 * dT5_dVe + T5 * dT1_dVe);
                      djrec_dT = pParam->B3SOIDDisrec * (T2 * dT6_dVe + T6 * dT2_dVe);

                      T7 = model->B3SOIDDxtun / pParam->B3SOIDDntun;
		      T0 = pow(TempRatio, T7);
		      jtun = model->B3SOIDDistun * T0;
                      djtun_dT = model->B3SOIDDistun * T7 * pow(TempRatio, T7 - 1.0) * T8;

		      u0temp = pParam->B3SOIDDu0 * pow(TempRatio, pParam->B3SOIDDute);
                      du0temp_dT = pParam->B3SOIDDu0 * pParam->B3SOIDDute *
                                   pow(TempRatio, pParam->B3SOIDDute - 1.0) * T8;

		      vsattemp = pParam->B3SOIDDvsat - pParam->B3SOIDDat * T3;
                      dvsattemp_dT = -pParam->B3SOIDDat * T8;

		      rds0 = (pParam->B3SOIDDrdsw + pParam->B3SOIDDprt
		          * T3) / pParam->B3SOIDDrds0denom;
                      drds0_dT = pParam->B3SOIDDprt / pParam->B3SOIDDrds0denom * T8;

		      ua = pParam->B3SOIDDuatemp + pParam->B3SOIDDua1 * T3;
		      ub = pParam->B3SOIDDubtemp + pParam->B3SOIDDub1 * T3;
		      uc = pParam->B3SOIDDuctemp + pParam->B3SOIDDuc1 * T3;
                      dua_dT = pParam->B3SOIDDua1 * T8;
                      dub_dT = pParam->B3SOIDDub1 * T8;
                      duc_dT = pParam->B3SOIDDuc1 * T8;
		  }
		  else {
                      vbi = pParam->B3SOIDDvbi;
                      vfbb = pParam->B3SOIDDvfbb;
                      phi = pParam->B3SOIDDphi;
                      sqrtPhi = pParam->B3SOIDDsqrtPhi;
                      Xdep0 = pParam->B3SOIDDXdep0;
                      jbjt = pParam->B3SOIDDjbjt;
                      jdif = pParam->B3SOIDDjdif;
                      jrec = pParam->B3SOIDDjrec;
                      jtun = pParam->B3SOIDDjtun;
                      u0temp = pParam->B3SOIDDu0temp;
                      vsattemp = pParam->B3SOIDDvsattemp;
                      rds0 = pParam->B3SOIDDrds0;
                      ua = pParam->B3SOIDDua;
                      ub = pParam->B3SOIDDub;
                      uc = pParam->B3SOIDDuc;
                      dni_dT = dvbi_dT = dvfbb_dT = djbjt_dT = djdif_dT = 0.0;
                      djrec_dT = djtun_dT = du0temp_dT = dvsattemp_dT = 0.0;
                      drds0_dT = dua_dT = dub_dT = duc_dT = 0.0;
		  }
		  
		  /* TempRatio used for Vth and mobility */
		  if (selfheat) {
		      TempRatio = Temp / model->B3SOIDDtnom - 1.0;
		  }
		  else {
		      TempRatio =  ckt->CKTtemp / model->B3SOIDDtnom - 1.0;
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
		      here->B3SOIDDmode = 1;
		      Vds = vds;
		      Vgs = vgs;
		      Vbs = vbs;
		      Vbd = vbd;
		      Ves = ves;
		      Vps = vps;
		  }
		  else
		  {   /* inverse mode */
		      here->B3SOIDDmode = -1;
		      Vds = -vds;
		      Vgs = vgd;
		      Vbs = vbd;
		      Vbd = vbs;
		      Ves = ved;
		      Vps = vpd;
		  }


                  if (here->B3SOIDDdebugMod > 2)
		  {
		     fprintf(fpdebug, "Vgs=%.4f, Vds=%.4f, Vbs=%.4f, ",
			   Vgs, Vds, Vbs);
		     fprintf(fpdebug, "Ves=%.4f, Vps=%.4f, Temp=%.1f\n", 
			   Ves, Vps, Temp);
		  }

		  Vesfb = Ves - vfbb;
		  Cbox = model->B3SOIDDcbox;
		  K1 = pParam->B3SOIDDk1;

		  ChargeComputationNeeded =  
			 ((ckt->CKTmode & (MODEAC | MODETRAN | MODEINITSMSIG)) ||
			 ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)))
			 ? 1 : 0;

                  if (here->B3SOIDDdebugMod == -1)
                     ChargeComputationNeeded = 1;
                  



/* Poly Gate Si Depletion Effect */
		  T0 = pParam->B3SOIDDvfb + phi;
		  if ((pParam->B3SOIDDngate > 1.e18) && (pParam->B3SOIDDngate < 1.e25) 
		       && (Vgs > T0))
		  /* added to avoid the problem caused by ngate */
		  {   T1 = 1.0e6 * Charge_q * EPSSI * pParam->B3SOIDDngate
			 / (model->B3SOIDDcox * model->B3SOIDDcox);
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


		  Leff = pParam->B3SOIDDleff;

		  if (selfheat) {
		      Vtm = KboQ * Temp;
                      dVtm_dT = KboQ;
		  }
		  else {
		      Vtm = model->B3SOIDDvtm;
                      dVtm_dT = 0.0;
		  }

		  V0 = vbi - phi;

/* Prepare Vbs0t */
		      T0 = -pParam->B3SOIDDdvbd1 * pParam->B3SOIDDleff / pParam->B3SOIDDlitl;
		      T1 = pParam->B3SOIDDdvbd0 * (exp(0.5*T0) + 2*exp(T0));
		      T2 = T1 * (vbi - phi);
		      T3 = 0.5 * model->B3SOIDDqsi / model->B3SOIDDcsi;
		      Vbs0t = phi - T3 + pParam->B3SOIDDvbsa + T2;
                      if (selfheat)
                         dVbs0t_dT = T1 * dvbi_dT;
                      else
                         dVbs0t_dT = 0.0;

/* Prepare Vbs0 */
			  T0 = 1 + model->B3SOIDDcsieff / Cbox;
                          T1 = pParam->B3SOIDDkb1 / T0;
			  T2 = T1 * (Vbs0t - Vesfb);

                          /* T6 is Vbs0 before limiting */
                          T6 = Vbs0t - T2;
                          dT6_dVe = T1;
                          if (selfheat)
                             dT6_dT = dVbs0t_dT - T1 * (dVbs0t_dT + dvfbb_dT);
                          else
                             dT6_dT = 0.0;
 
                          /* limit Vbs0 to below phi */
                          T1 = phi - pParam->B3SOIDDdelp;
                          T2 = T1 - T6 - DELT_Vbseff;
                          T3 = sqrt(T2 * T2 + 4.0 * DELT_Vbseff);
                          Vbs0 = T1 - 0.5 * (T2 + T3);
                          T4 = 0.5 * (1 + T2/T3);
                          dVbs0_dVe = T4 * dT6_dVe;
                          if (selfheat)  dVbs0_dT = T4 * dT6_dT;
                          else  dVbs0_dT = 0.0;

			  T1 = Vbs0t - Vbs0 - DELT_Vbsmos;
			  T2 = sqrt(T1 * T1 + DELT_Vbsmos * DELT_Vbsmos);
			  T3 = 0.5 * (T1 + T2);
			  T4 = T3 * model->B3SOIDDcsieff / model->B3SOIDDqsieff;
			  Vbs0mos = Vbs0 - 0.5 * T3 * T4;
                          T5 = 0.5 * T4 * (1 + T1 / T2);
			  dVbs0mos_dVe = dVbs0_dVe * (1 + T5);
                          if (selfheat)
			     dVbs0mos_dT = dVbs0_dT - (dVbs0t_dT - dVbs0_dT) * T5;
                          else
			     dVbs0mos_dT = 0.0;

/* Prepare Vthfd - treat Vbs0mos as if it were independent variable Vb */
		     Phis = phi - Vbs0mos;
		     dPhis_dVb = -1;
		     sqrtPhis = sqrt(Phis);
		     dsqrtPhis_dVb = -0.5 / sqrtPhis;
		     Xdep = Xdep0 * sqrtPhis / sqrtPhi;
		     dXdep_dVb = (Xdep0 / sqrtPhi)
				 * dsqrtPhis_dVb;
		     sqrtXdep = sqrt(Xdep);

		     T0 = pParam->B3SOIDDdvt2 * Vbs0mos;
		     if (T0 >= - 0.5)
		     {   T1 = 1.0 + T0;
			 T2 = pParam->B3SOIDDdvt2;
		     }
		     else /* Added to avoid any discontinuity problems caused by dvt2*/ 
		     {   T4 = 1.0 / (3.0 + 8.0 * T0);
			 T1 = (1.0 + 3.0 * T0) * T4; 
			 T2 = pParam->B3SOIDDdvt2 * T4 * T4;
		     }
		     lt1 = model->B3SOIDDfactor1 * sqrtXdep * T1;
		     dlt1_dVb = model->B3SOIDDfactor1 * (0.5 / sqrtXdep * T1 * dXdep_dVb
			      + sqrtXdep * T2);

		     T0 = pParam->B3SOIDDdvt2w * Vbs0mos;
		     if (T0 >= - 0.5)
		     {   T1 = 1.0 + T0;
			 T2 = pParam->B3SOIDDdvt2w;
		     }
		     else /* Added to avoid any discontinuity problems caused by
			     dvt2w */
		     {   T4 = 1.0 / (3.0 + 8.0 * T0);
			 T1 = (1.0 + 3.0 * T0) * T4;
			 T2 = pParam->B3SOIDDdvt2w * T4 * T4;
		     }
		     ltw= model->B3SOIDDfactor1 * sqrtXdep * T1;
		     dltw_dVb = model->B3SOIDDfactor1 * (0.5 / sqrtXdep * T1 * dXdep_dVb 
			      + sqrtXdep * T2);

		     T0 = -0.5 * pParam->B3SOIDDdvt1 * Leff / lt1;
		     if (T0 > -EXP_THRESHOLD)
		     {   T1 = exp(T0);
			 dT1_dVb = -T0 / lt1 * T1 * dlt1_dVb;
			 Theta0 = T1 * (1.0 + 2.0 * T1);
			 dTheta0_dVb = (1.0 + 4.0 * T1) * dT1_dVb;
		     }
		     else
		     {   T1 = MIN_EXP;
			 Theta0 = T1 * (1.0 + 2.0 * T1);
			 dTheta0_dVb = 0.0;
		     }
		     here->B3SOIDDthetavth = pParam->B3SOIDDdvt0 * Theta0;
		     Delt_vth = here->B3SOIDDthetavth * V0;
		     dDelt_vth_dVb = pParam->B3SOIDDdvt0 * dTheta0_dVb * V0;
                     if (selfheat) dDelt_vth_dT = here->B3SOIDDthetavth * dvbi_dT;
                     else dDelt_vth_dT = 0.0;

		     T0 = -0.5*pParam->B3SOIDDdvt1w * pParam->B3SOIDDweff*Leff/ltw;
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
		     T0 = pParam->B3SOIDDdvt0w * T2;
		     DeltVthw = T0 * V0;
		     dDeltVthw_dVb = pParam->B3SOIDDdvt0w * dT2_dVb * V0;
                     if (selfheat)  dDeltVthw_dT = T0 * dvbi_dT;
                     else  dDeltVthw_dT = 0.0;

		     T0 = sqrt(1.0 + pParam->B3SOIDDnlx / Leff);
                     T1 = (pParam->B3SOIDDkt1 + pParam->B3SOIDDkt1l / Leff
                           + pParam->B3SOIDDkt2 * Vbs0mos);
		     DeltVthtemp = pParam->B3SOIDDk1 * (T0 - 1.0) * sqrtPhi + T1 * TempRatio;
                     if (selfheat)   dDeltVthtemp_dT = T1 / model->B3SOIDDtnom;
                     else   dDeltVthtemp_dT = 0.0;

		     tmp2 = model->B3SOIDDtox * phi
			  / (pParam->B3SOIDDweff + pParam->B3SOIDDw0);

		     T3 = pParam->B3SOIDDeta0 + pParam->B3SOIDDetab * Vbs0mos;
		     if (T3 < 1.0e-4) /* avoid discontinuity problems caused by etab */
		     {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
			 T3 = (2.0e-4 - T3) * T9;
			 T4 = T9 * T9 * pParam->B3SOIDDetab;
			 dT3_dVb = T4;
		     }
		     else
		     {
			 dT3_dVb = pParam->B3SOIDDetab;
		     }
		     DIBL_Sft = T3 * pParam->B3SOIDDtheta0vb0 * Vds;
		     dDIBL_Sft_dVd = T3 * pParam->B3SOIDDtheta0vb0;
		     dDIBL_Sft_dVb = pParam->B3SOIDDtheta0vb0 * Vds * dT3_dVb;

		     Vthfd = model->B3SOIDDtype * pParam->B3SOIDDvth0 + pParam->B3SOIDDk1
			 * (sqrtPhis - sqrtPhi) - pParam->B3SOIDDk2
			 * Vbs0mos-Delt_vth-DeltVthw +(pParam->B3SOIDDk3 +pParam->B3SOIDDk3b
			 * Vbs0mos) * tmp2 + DeltVthtemp - DIBL_Sft;

		     T6 = pParam->B3SOIDDk3b * tmp2 - pParam->B3SOIDDk2 +
			  pParam->B3SOIDDkt2 * TempRatio;
		     dVthfd_dVd = -dDIBL_Sft_dVd;
		     T7 = pParam->B3SOIDDk1 * dsqrtPhis_dVb
			  - dDelt_vth_dVb - dDeltVthw_dVb
			  + T6 - dDIBL_Sft_dVb;
		     dVthfd_dVe = T7 * dVbs0mos_dVe;
                     if (selfheat)
                        dVthfd_dT = dDeltVthtemp_dT - dDelt_vth_dT - dDeltVthw_dT
                                  + T7 * dVbs0mos_dT;
                     else
                        dVthfd_dT = 0.0;

/* Effective Vbs0 and Vbs0t for all Vgs */
		     T1 = Vthfd - Vgs_eff - DELT_Vbs0eff;
		     T2 = sqrt(T1 * T1 + DELT_Vbs0eff * DELT_Vbs0eff );
	   
		     Vbs0teff = Vbs0t - 0.5 * (T1 + T2);
		     dVbs0teff_dVg = 0.5  * (1 + T1/T2) * dVgs_eff_dVg;
		     dVbs0teff_dVd = - 0.5 * (1 + T1 / T2) * dVthfd_dVd;
		     dVbs0teff_dVe = - 0.5 * (1 + T1 / T2) * dVthfd_dVe;
                     if (selfheat)
                        dVbs0teff_dT = dVbs0t_dT - 0.5 * (1 + T1 / T2) * dVthfd_dT;
                     else
                        dVbs0teff_dT = 0.0;

                     /* Calculate nfb */
                     T3 = 1 / (K1 * K1);
			T4 = pParam->B3SOIDDkb3 * Cbox / model->B3SOIDDcox;
			T8 = sqrt(phi - Vbs0mos);
			T5 = sqrt(1 + 4 * T3 * (phi + K1 * T8 - Vbs0mos));
			T6 = 1 + T4 * T5;
			Nfb = model->B3SOIDDnfb = 1 / T6;
			T7 = 2 * T3 * T4 * Nfb * Nfb / T5 * (0.5 * K1 / T8 + 1);
			Vbs0eff = Vbs0 - Nfb * 0.5 * (T1 + T2);
			dVbs0eff_dVg = Nfb * 0.5  * (1 + T1/T2) * dVgs_eff_dVg;
			dVbs0eff_dVd = - Nfb * 0.5 * (1 + T1 / T2) * dVthfd_dVd;
			dVbs0eff_dVe = dVbs0_dVe - Nfb * 0.5 * (1 + T1 / T2) 
				     * dVthfd_dVe - T7 * 0.5 * (T1 + T2) * dVbs0mos_dVe;
                        if (selfheat)
                           dVbs0eff_dT = dVbs0_dT - Nfb * 0.5 * (1 + T1 / T2)
                                       * dVthfd_dT - T7 * 0.5 * (T1 + T2) * dVbs0mos_dT;
                        else
                           dVbs0eff_dT = 0.0;

/* Simple check of Vbs */
/* Prepare Vbsdio */
                        T1 =  Vbs - (Vbs0eff + OFF_Vbsdio) - DELT_Vbsdio;
                        T2 = sqrt(T1*T1 + DELT_Vbsdio * DELT_Vbsdio);
                        T3 = 0.5 * (1 + T1/T2);
 
                        Vbsdio = Vbs0eff + OFF_Vbsdio + 0.5 * (T1 + T2);
                        dVbsdio_dVg = (1 - T3) * dVbs0eff_dVg;
                        dVbsdio_dVd = (1 - T3) * dVbs0eff_dVd;
                        dVbsdio_dVe = (1 - T3) * dVbs0eff_dVe;
                        if (selfheat)  dVbsdio_dT = (1 - T3) * dVbs0eff_dT;
                        else  dVbsdio_dT = 0.0;
                        dVbsdio_dVb = T3;

/* Prepare Vbseff */
	         	T1 = Vbs0teff - Vbsdio - DELT_Vbsmos;
			T2 = sqrt(T1 * T1 + DELT_Vbsmos * DELT_Vbsmos);
			T3 = 0.5 * (T1 + T2);
			T5 = 0.5 * (1 + T1/T2);
			dT3_dVg = T5 * (dVbs0teff_dVg - dVbsdio_dVg);
			dT3_dVd = T5 * (dVbs0teff_dVd - dVbsdio_dVd);
			dT3_dVb = - T5 * dVbsdio_dVb;
			dT3_dVe = T5 * (dVbs0teff_dVe - dVbsdio_dVe);
                        if (selfheat)  dT3_dT = T5 * (dVbs0teff_dT - dVbsdio_dT);
                        else  dT3_dT = 0.0;
			T4 = T3 * model->B3SOIDDcsieff / model->B3SOIDDqsieff;
	   
			Vbsmos = Vbsdio - 0.5 * T3 * T4;
			dVbsmos_dVg = dVbsdio_dVg - T4 * dT3_dVg;
			dVbsmos_dVd = dVbsdio_dVd - T4 * dT3_dVd;
			dVbsmos_dVb = dVbsdio_dVb - T4 * dT3_dVb;
			dVbsmos_dVe = dVbsdio_dVe - T4 * dT3_dVe;
                        if (selfheat)  dVbsmos_dT = dVbsdio_dT - T4 * dT3_dT;
                        else  dVbsmos_dT = 0.0;

/* Prepare Vcs */
		     Vcs = Vbsdio - Vbs0eff;
		     dVcs_dVb = dVbsdio_dVb;
		     dVcs_dVg = dVbsdio_dVg - dVbs0eff_dVg;
		     dVcs_dVd = dVbsdio_dVd - dVbs0eff_dVd;
		     dVcs_dVe = dVbsdio_dVe - dVbs0eff_dVe;
                     dVcs_dT = dVbsdio_dT - dVbs0eff_dT;

/* Check Vps */
                     /* Note : if Vps is less Vbs0eff => non-physical */
                     T1 = Vps - Vbs0eff + DELT_Vbs0dio;
                     T2 = sqrt(T1 * T1 + DELT_Vbs0dio * DELT_Vbs0dio);
                     T3 = 0.5 * (1 + T1/T2);
                     Vpsdio = Vbs0eff + 0.5 * (T1 + T2);
                     dVpsdio_dVg = (1 - T3) * dVbs0eff_dVg;
                     dVpsdio_dVd = (1 - T3) * dVbs0eff_dVd;
                     dVpsdio_dVe = (1 - T3) * dVbs0eff_dVe;
                     if (selfheat)  dVpsdio_dT = (1 - T3) * dVbs0eff_dT;
                     else  dVpsdio_dT = 0.0;
                     dVpsdio_dVp = T3;
                     Vbp = Vbsdio - Vpsdio;
                     dVbp_dVb = dVbsdio_dVb;
                     dVbp_dVg = dVbsdio_dVg - dVpsdio_dVg;
                     dVbp_dVd = dVbsdio_dVd - dVpsdio_dVd;
                     dVbp_dVe = dVbsdio_dVe - dVpsdio_dVe;
                     dVbp_dT = dVbsdio_dT - dVpsdio_dT;
                     dVbp_dVp = - dVpsdio_dVp;

                  here->B3SOIDDvbsdio = Vbsdio;
                  here->B3SOIDDvbs0eff = Vbs0eff;

		  T1 = phi - pParam->B3SOIDDdelp;
		  T2 = T1 - Vbsmos - DELT_Vbseff;
		  T3 = sqrt(T2 * T2 + 4.0 * DELT_Vbseff * T1);
		  Vbseff = T1 - 0.5 * (T2 + T3);
		  T4 = 0.5 * (1 + T2/T3);
		  dVbseff_dVg = T4 * dVbsmos_dVg;
		  dVbseff_dVd = T4 * dVbsmos_dVd;
		  dVbseff_dVb = T4 * dVbsmos_dVb;
		  dVbseff_dVe = T4 * dVbsmos_dVe;
                  if (selfheat)  dVbseff_dT = T4 * dVbsmos_dT;
                  else  dVbseff_dT = 0.0;

                  here->B3SOIDDvbseff = Vbseff;

		  Phis = phi - Vbseff;
		  dPhis_dVb = -1;
		  sqrtPhis = sqrt(Phis);
		  dsqrtPhis_dVb = -0.5 / sqrtPhis ;

		  Xdep = Xdep0 * sqrtPhis / sqrtPhi;
		  dXdep_dVb = (Xdep0 / sqrtPhi)
			    * dsqrtPhis_dVb;

/* Vth Calculation */
		  T3 = sqrt(Xdep);

		  T0 = pParam->B3SOIDDdvt2 * Vbseff;
		  if (T0 >= - 0.5)
		  {   T1 = 1.0 + T0;
		      T2 = pParam->B3SOIDDdvt2 ;
		  }
		  else /* Added to avoid any discontinuity problems caused by dvt2 */ 
		  {   T4 = 1.0 / (3.0 + 8.0 * T0);
		      T1 = (1.0 + 3.0 * T0) * T4; 
		      T2 = pParam->B3SOIDDdvt2 * T4 * T4 ;
		  }
		  lt1 = model->B3SOIDDfactor1 * T3 * T1;
		  dlt1_dVb =model->B3SOIDDfactor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

		  T0 = pParam->B3SOIDDdvt2w * Vbseff;
		  if (T0 >= - 0.5)
		  {   T1 = 1.0 + T0;
		      T2 = pParam->B3SOIDDdvt2w ;
		  }
		  else /* Added to avoid any discontinuity problems caused by dvt2w */ 
		  {   T4 = 1.0 / (3.0 + 8.0 * T0);
		      T1 = (1.0 + 3.0 * T0) * T4; 
		      T2 = pParam->B3SOIDDdvt2w * T4 * T4 ;
		  }
		  ltw= model->B3SOIDDfactor1 * T3 * T1;
		  dltw_dVb=model->B3SOIDDfactor1*(0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

		  T0 = -0.5 * pParam->B3SOIDDdvt1 * Leff / lt1;
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

		  here->B3SOIDDthetavth = pParam->B3SOIDDdvt0 * Theta0;
		  Delt_vth = here->B3SOIDDthetavth * V0;
		  dDelt_vth_dVb = pParam->B3SOIDDdvt0 * dTheta0_dVb * V0;
                  if (selfheat)  dDelt_vth_dT = here->B3SOIDDthetavth * dvbi_dT;
                  else  dDelt_vth_dT = 0.0;

		  T0 = -0.5 * pParam->B3SOIDDdvt1w * pParam->B3SOIDDweff * Leff / ltw;
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

		  T0 = pParam->B3SOIDDdvt0w * T2;
		  DeltVthw = T0 * V0;
		  dDeltVthw_dVb = pParam->B3SOIDDdvt0w * dT2_dVb * V0;
                  if (selfheat)   dDeltVthw_dT = T0 * dvbi_dT;
                  else   dDeltVthw_dT = 0.0;

		  T0 = sqrt(1.0 + pParam->B3SOIDDnlx / Leff);
                  T1 = (pParam->B3SOIDDkt1 + pParam->B3SOIDDkt1l / Leff
                        + pParam->B3SOIDDkt2 * Vbseff);
		  DeltVthtemp = pParam->B3SOIDDk1 * (T0 - 1.0) * sqrtPhi + T1 * TempRatio;
                  if (selfheat)
                     dDeltVthtemp_dT = T1 / model->B3SOIDDtnom;
                  else
                     dDeltVthtemp_dT = 0.0;

		  tmp2 = model->B3SOIDDtox * phi
		       / (pParam->B3SOIDDweff + pParam->B3SOIDDw0);

		  T3 = pParam->B3SOIDDeta0 + pParam->B3SOIDDetab * Vbseff;
		  if (T3 < 1.0e-4) /* avoid  discontinuity problems caused by etab */ 
		  {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
		      T3 = (2.0e-4 - T3) * T9;
		      T4 = T9 * T9 * pParam->B3SOIDDetab;
		      dT3_dVb = T4 ;
		  }
		  else
		  {   
		      dT3_dVb = pParam->B3SOIDDetab ;
		  }
		  DIBL_Sft = T3 * pParam->B3SOIDDtheta0vb0 * Vds;
		  dDIBL_Sft_dVd = pParam->B3SOIDDtheta0vb0 * T3;
		  dDIBL_Sft_dVb = pParam->B3SOIDDtheta0vb0 * Vds * dT3_dVb;

		  Vth = model->B3SOIDDtype * pParam->B3SOIDDvth0 + pParam->B3SOIDDk1 
		      * (sqrtPhis - sqrtPhi) - pParam->B3SOIDDk2 
		      * Vbseff- Delt_vth - DeltVthw +(pParam->B3SOIDDk3 + pParam->B3SOIDDk3b
		      * Vbseff) * tmp2 + DeltVthtemp - DIBL_Sft;

		  here->B3SOIDDvon = Vth; 

		  T6 = pParam->B3SOIDDk3b * tmp2 - pParam->B3SOIDDk2 
		       + pParam->B3SOIDDkt2 * TempRatio;          
		  dVth_dVb = pParam->B3SOIDDk1 * dsqrtPhis_dVb 
			   - dDelt_vth_dVb - dDeltVthw_dVb
			   + T6 - dDIBL_Sft_dVb;  /*  this is actually dVth_dVbseff  */
		  dVth_dVd = -dDIBL_Sft_dVd;
                  if (selfheat)  dVth_dT = dDeltVthtemp_dT - dDelt_vth_dT - dDeltVthw_dT;
                  else  dVth_dT = 0.0;

/* Calculate n */
		  T2 = pParam->B3SOIDDnfactor * EPSSI / Xdep;
		  dT2_dVb = - T2 / Xdep * dXdep_dVb;

		  T3 = pParam->B3SOIDDcdsc + pParam->B3SOIDDcdscb * Vbseff
		       + pParam->B3SOIDDcdscd * Vds;
		  dT3_dVb = pParam->B3SOIDDcdscb;
		  dT3_dVd = pParam->B3SOIDDcdscd;

		  T4 = (T2 + T3 * Theta0 + pParam->B3SOIDDcit) / model->B3SOIDDcox;
		  dT4_dVb = (dT2_dVb + Theta0 * dT3_dVb + dTheta0_dVb * T3)
                            / model->B3SOIDDcox;
		  dT4_dVd = Theta0 * dT3_dVd / model->B3SOIDDcox;

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

		  T10 = 2.0 * n * Vtm;
		  VgstNVt = Vgst / T10;
		  ExpArg = (2.0 * pParam->B3SOIDDvoff - Vgst) / T10;

		  /* MCJ: Very small Vgst */
		  if (VgstNVt > EXP_THRESHOLD)
		  {   Vgsteff = Vgst;
                      /* T0 is dVgsteff_dVbseff */
                      T0 = -dVth_dVb;
		      dVgsteff_dVg = dVgs_eff_dVg + T0 * dVbseff_dVg;
		      dVgsteff_dVd = -dVth_dVd + T0 * dVbseff_dVd;
		      dVgsteff_dVb = T0 * dVbseff_dVb;
                      dVgsteff_dVe = T0 * dVbseff_dVe;
                      if (selfheat)
                         dVgsteff_dT  = -dVth_dT + T0 * dVbseff_dT;
                      else
                         dVgsteff_dT = 0.0;
		  }
		  else if (ExpArg > EXP_THRESHOLD)
		  {   T0 = (Vgst - pParam->B3SOIDDvoff) / (n * Vtm);
		      ExpVgst = exp(T0);
		      Vgsteff = Vtm * pParam->B3SOIDDcdep0 / model->B3SOIDDcox * ExpVgst;
		      T3 = Vgsteff / (n * Vtm) ;
                      /* T1 is dVgsteff_dVbseff */
		      T1  = -T3 * (dVth_dVb + T0 * Vtm * dn_dVb);
		      dVgsteff_dVg = T3 * dVgs_eff_dVg + T1 * dVbseff_dVg;
		      dVgsteff_dVd = -T3 * (dVth_dVd + T0 * Vtm * dn_dVd) + T1 * dVbseff_dVd;
                      dVgsteff_dVe = T1 * dVbseff_dVe;
                      dVgsteff_dVb = T1 * dVbseff_dVb;
                      if (selfheat)
                         dVgsteff_dT = -T3 * (dVth_dT + T0 * dVtm_dT * n)
                                     + Vgsteff / Temp + T1 * dVbseff_dT;
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

		      dT2_dVg = -model->B3SOIDDcox / (Vtm * pParam->B3SOIDDcdep0)
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
                      dVgsteff_dVe = T4 * dVbseff_dVe;
		      dVgsteff_dVg = (T2 * dT1_dVg - T1 * dT2_dVg) / T3 * dVgs_eff_dVg
                                     + T4 * dVbseff_dVg;
		      dVgsteff_dVd = (T2 * dT1_dVd - T1 * dT2_dVd) / T3 + T4 * dVbseff_dVd;
                      if (selfheat)
                         dVgsteff_dT = (T2 * dT1_dT - T1 * dT2_dT) / T3 + T4 * dVbseff_dT;
                      else
                         dVgsteff_dT = 0.0;
		  }
		  Vgst2Vtm = Vgsteff + 2.0 * Vtm;
                  if (selfheat)  dVgst2Vtm_dT = 2.0 * dVtm_dT;  
                  else  dVgst2Vtm_dT = 0.0;

/* Calculate Effective Channel Geometry */
		  T9 = sqrtPhis - sqrtPhi;
		  Weff = pParam->B3SOIDDweff - 2.0 * (pParam->B3SOIDDdwg * Vgsteff 
		       + pParam->B3SOIDDdwb * T9); 
		  dWeff_dVg = -2.0 * pParam->B3SOIDDdwg;
		  dWeff_dVb = -2.0 * pParam->B3SOIDDdwb * dsqrtPhis_dVb;

		  if (Weff < 2.0e-8) /* to avoid the discontinuity problem due to Weff*/
		  {   T0 = 1.0 / (6.0e-8 - 2.0 * Weff);
		      Weff = 2.0e-8 * (4.0e-8 - Weff) * T0;
		      T0 *= T0 * 4.0e-16;
		      dWeff_dVg *= T0;
		      dWeff_dVb *= T0;
		  }

		  T0 = pParam->B3SOIDDprwg * Vgsteff + pParam->B3SOIDDprwb * T9;
		  if (T0 >= -0.9)
		  {   Rds = rds0 * (1.0 + T0);
		      dRds_dVg = rds0 * pParam->B3SOIDDprwg;
		      dRds_dVb = rds0 * pParam->B3SOIDDprwb * dsqrtPhis_dVb;
                      if (selfheat)  dRds_dT = (1.0 + T0) * drds0_dT;
                      else  dRds_dT = 0.0;
		  }
		  else
		   /* to avoid the discontinuity problem due to prwg and prwb*/
		  {   T1 = 1.0 / (17.0 + 20.0 * T0);
		      Rds = rds0 * (0.8 + T0) * T1;
		      T1 *= T1;
		      dRds_dVg = rds0 * pParam->B3SOIDDprwg * T1;
		      dRds_dVb = rds0 * pParam->B3SOIDDprwb * dsqrtPhis_dVb
			       * T1;
                      if (selfheat)  dRds_dT = (0.8 + T0) * T1 * drds0_dT;
                      else  dRds_dT = 0.0;
		  }

/* Calculate Abulk */
                  if (pParam->B3SOIDDa0 == 0.0)
                  {
                     Abulk0 = Abulk = dAbulk0_dVb = dAbulk_dVg = dAbulk_dVb = 0.0;
                  }
                  else
                  {
		     T1 = 0.5 * pParam->B3SOIDDk1 / sqrtPhi;
		     T9 = sqrt(model->B3SOIDDxj * Xdep);
		     tmp1 = Leff + 2.0 * T9;
		     T5 = Leff / tmp1; 
		     tmp2 = pParam->B3SOIDDa0 * T5;
		     tmp3 = pParam->B3SOIDDweff + pParam->B3SOIDDb1; 
		     tmp4 = pParam->B3SOIDDb0 / tmp3;
		     T2 = tmp2 + tmp4;
		     dT2_dVb = -T9 * tmp2 / tmp1 / Xdep * dXdep_dVb;
		     T6 = T5 * T5;
		     T7 = T5 * T6;

		     Abulk0 = T1 * T2; 
		     dAbulk0_dVb = T1 * dT2_dVb;

		     T8 = pParam->B3SOIDDags * pParam->B3SOIDDa0 * T7;
		     dAbulk_dVg = -T1 * T8;
		     Abulk = Abulk0 + dAbulk_dVg * Vgsteff; 

		     dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * 3.0 * T1 * dT2_dVb
                                             / tmp2;
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

		  T2 = pParam->B3SOIDDketa * Vbseff;
		  if (T2 >= -0.9)
		  {   T0 = 1.0 / (1.0 + T2);
		      dT0_dVb = -pParam->B3SOIDDketa * T0 * T0 ;
		  }
		  else
		  /* added to avoid the problems caused by Keta */
		  {   T1 = 1.0 / (0.8 + T2);
		      T0 = (17.0 + 20.0 * T2) * T1;
		      dT0_dVb = -pParam->B3SOIDDketa * T1 * T1 ;
		  }
		  dAbulk_dVg *= T0;
		  dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
		  dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
		  Abulk *= T0;
		  Abulk0 *= T0;

		  Abulk += 1;
		  Abulk0 += 1;

/* Prepare Abeff */
		      T0 = pParam->B3SOIDDabp * Vgst2Vtm;
		      T1 = 1 - Vcs / T0 - DELT_Xcsat;
		      T2 = sqrt(T1 * T1 + DELT_Xcsat * DELT_Xcsat);
		      T3 = 1 - 0.5 * (T1 + T2);
		      T5 = -0.5 * (1 + T1 / T2);
		      dT1_dVg = Vcs / Vgst2Vtm / T0;
		      dT3_dVg = T5 * dT1_dVg;
		      dT1_dVc = - 1 / T0;
		      dT3_dVc = T5 * dT1_dVc;

		      Xcsat = pParam->B3SOIDDmxc * T3 * T3 + (1 - pParam->B3SOIDDmxc)*T3;
		      T4 = 2 * pParam->B3SOIDDmxc * T3 + (1 - pParam->B3SOIDDmxc);
		      dXcsat_dVg = T4 * dT3_dVg;
		      dXcsat_dVc = T4 * dT3_dVc;

		      Abeff = Xcsat * Abulk + (1 - Xcsat) * model->B3SOIDDadice;
		      T0 = Xcsat * dAbulk_dVg + Abulk * dXcsat_dVg;
		      dAbeff_dVg = T0 - model->B3SOIDDadice * dXcsat_dVg;
		      dAbeff_dVb = Xcsat * dAbulk_dVb;
		      dAbeff_dVc = (Abulk - model->B3SOIDDadice) * dXcsat_dVc;
                 here->B3SOIDDabeff = Abeff;

/* Mobility calculation */
		  if (model->B3SOIDDmobMod == 1)
		  {   T0 = Vgsteff + Vth + Vth;
		      T2 = ua + uc * Vbseff;
		      T3 = T0 / model->B3SOIDDtox;
		      T5 = T3 * (T2 + ub * T3);
		      dDenomi_dVg = (T2 + 2.0 * ub * T3) / model->B3SOIDDtox;
		      dDenomi_dVd = dDenomi_dVg * 2 * dVth_dVd;
		      dDenomi_dVb = dDenomi_dVg * 2 * dVth_dVb + uc * T3 ;
                      if (selfheat)
                         dDenomi_dT = dDenomi_dVg * 2 * dVth_dT 
                                    + (dua_dT + Vbseff * duc_dT
                                    + dub_dT * T3 ) * T3;
                      else
                         dDenomi_dT = 0.0;
		  }
		  else if (model->B3SOIDDmobMod == 2)
		  {   T5 = Vgsteff / model->B3SOIDDtox * (ua
			 + uc * Vbseff + ub * Vgsteff
			 / model->B3SOIDDtox);
		      dDenomi_dVg = (ua + uc * Vbseff
				  + 2.0 * ub * Vgsteff / model->B3SOIDDtox)
				  / model->B3SOIDDtox;
		      dDenomi_dVd = 0.0;
		      dDenomi_dVb = Vgsteff * uc / model->B3SOIDDtox ;
                      if (selfheat)
                         dDenomi_dT = Vgsteff / model->B3SOIDDtox
                                    * (dua_dT + Vbseff * duc_dT + dub_dT
                                    * Vgsteff / model->B3SOIDDtox);
                      else
                         dDenomi_dT = 0.0;
		  }
		  else  /*  mobMod == 3  */
		  {   T0 = Vgsteff + Vth + Vth;
		      T2 = 1.0 + uc * Vbseff;
		      T3 = T0 / model->B3SOIDDtox;
		      T4 = T3 * (ua + ub * T3);
		      T5 = T4 * T2;
		      dDenomi_dVg = (ua + 2.0 * ub * T3) * T2
				  / model->B3SOIDDtox;
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

		  here->B3SOIDDueff = ueff = u0temp / Denomi;
		  T9 = -ueff / Denomi;
		  dueff_dVg = T9 * dDenomi_dVg;
		  dueff_dVd = T9 * dDenomi_dVd;
		  dueff_dVb = T9 * dDenomi_dVb;
                  if (selfheat)  dueff_dT = T9 * dDenomi_dT + du0temp_dT / Denomi;
                  else  dueff_dT = 0.0;

/* Saturation Drain Voltage  Vdsat */
		  WVCox = Weff * vsattemp * model->B3SOIDDcox;
		  WVCoxRds = WVCox * Rds; 

/*                  dWVCoxRds_dT = WVCox * dRds_dT
                                 + Weff * model->B3SOIDDcox * Rds * dvsattemp_dT; */

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
		  a1 = pParam->B3SOIDDa1;
		  if (a1 == 0.0)
		  {   Lambda = pParam->B3SOIDDa2;
		      dLambda_dVg = 0.0;
		  }
		  else if (a1 > 0.0)
/* Added to avoid the discontinuity problem caused by a1 and a2 (Lambda) */
		  {   T0 = 1.0 - pParam->B3SOIDDa2;
		      T1 = T0 - pParam->B3SOIDDa1 * Vgsteff - 0.0001;
		      T2 = sqrt(T1 * T1 + 0.0004 * T0);
		      Lambda = pParam->B3SOIDDa2 + T0 - 0.5 * (T1 + T2);
		      dLambda_dVg = 0.5 * pParam->B3SOIDDa1 * (1.0 + T1 / T2);
		  }
		  else
		  {   T1 = pParam->B3SOIDDa2 + pParam->B3SOIDDa1 * Vgsteff - 0.0001;
		      T2 = sqrt(T1 * T1 + 0.0004 * pParam->B3SOIDDa2);
		      Lambda = 0.5 * (T1 + T2);
		      dLambda_dVg = 0.5 * pParam->B3SOIDDa1 * (1.0 + T1 / T2);
		  }

		  if (Rds > 0)
		  {   tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff;
		      tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff;
		  }
		  else
		  {   tmp2 = dWeff_dVg / Weff;
		      tmp3 = dWeff_dVb / Weff;
		  }
		  if ((Rds == 0.0) && (Lambda == 1.0))
		  {   T0 = 1.0 / (Abeff * EsatL + Vgst2Vtm);
		      tmp1 = 0.0;
		      T1 = T0 * T0;
		      T2 = Vgst2Vtm * T0;
		      T3 = EsatL * Vgst2Vtm;
		      Vdsat = T3 * T0;
				   
		      dT0_dVg = -(Abeff * dEsatL_dVg + EsatL * dAbeff_dVg + 1.0) * T1;
		      dT0_dVd = -(Abeff * dEsatL_dVd) * T1; 
		      dT0_dVb = -(Abeff * dEsatL_dVb + EsatL * dAbeff_dVb) * T1;
                      dT0_dVc = -(EsatL * dAbeff_dVc) * T1;
                      if (selfheat)
		         dT0_dT  = -(Abeff * dEsatL_dT + dVgst2Vtm_dT) * T1;
                      else dT0_dT  = 0.0;

		      dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
		      dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
		      dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
		      dVdsat_dVc = T3 * dT0_dVc;
                      if (selfheat)
		         dVdsat_dT  = T3 * dT0_dT  + T2 * dEsatL_dT
				    + EsatL * T0 * dVgst2Vtm_dT;
                      else dVdsat_dT  = 0.0;
		  }
		  else
		  {   tmp1 = dLambda_dVg / (Lambda * Lambda);
		      T9 = Abeff * WVCoxRds;
		      T8 = Abeff * T9;
		      T7 = Vgst2Vtm * T9;
		      T6 = Vgst2Vtm * WVCoxRds;
		      T0 = 2.0 * Abeff * (T9 - 1.0 + 1.0 / Lambda); 
		      dT0_dVg = 2.0 * (T8 * tmp2 - Abeff * tmp1
			      + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbeff_dVg);
/*		      dT0_dVb = 2.0 * (T8 * tmp3  this is equivalent to one below, but simpler
			      + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbeff_dVg);  */
		      dT0_dVb = 2.0 * (T8 * (2.0 / Abeff * dAbeff_dVb + tmp3)
			      + (1.0 / Lambda - 1.0) * dAbeff_dVb);
		      dT0_dVd = 0.0; 
		      dT0_dVc = 4.0 * T9 * dAbeff_dVc;

                      if (selfheat)
                      {
		         tmp4 = dRds_dT / Rds + dvsattemp_dT / vsattemp;
		         dT0_dT  = 2.0 * T8 * tmp4;
                      } else tmp4 = dT0_dT = 0.0;

		      T1 = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abeff * EsatL + 3.0 * T7;
		     
		      dT1_dVg = (2.0 / Lambda - 1.0) - 2.0 * Vgst2Vtm * tmp1
			      + Abeff * dEsatL_dVg + EsatL * dAbeff_dVg + 3.0 * (T9
			      + T7 * tmp2 + T6 * dAbeff_dVg);
		      dT1_dVb = Abeff * dEsatL_dVb + EsatL * dAbeff_dVb
			      + 3.0 * (T6 * dAbeff_dVb + T7 * tmp3);
		      dT1_dVd = Abeff * dEsatL_dVd;
		      dT1_dVc = EsatL * dAbeff_dVc + 3.0 * T6 * dAbeff_dVc;

                      if (selfheat)
                      {
		         tmp4 += dVgst2Vtm_dT / Vgst2Vtm;
		         dT1_dT  = (2.0 / Lambda - 1.0) * dVgst2Vtm_dT
				 + Abeff * dEsatL_dT + 3.0 * T7 * tmp4;
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
		      dVdsat_dVc = (dT1_dVc - (T1 * dT1_dVc - dT0_dVc * T2) / T3
			         - Vdsat * dT0_dVc) / T0;
                      if (selfheat)
		         dVdsat_dT  = (dT1_dT - (T1 * dT1_dT - dT0_dT * T2
				    - T0 * dT2_dT) / T3 - Vdsat * dT0_dT) / T0;
                      else dVdsat_dT  = 0.0;
		  }
		  here->B3SOIDDvdsat = Vdsat;

/* Vdsatii for impact ionization */
                  if (pParam->B3SOIDDaii > 0.0) 
                  {
                     if (pParam->B3SOIDDcii != 0.0)
                     {
                        T0 = pParam->B3SOIDDcii / sqrt(3.0) + pParam->B3SOIDDdii;
                        /* Hard limit Vds to T0 => T4  i.e. limit T0 to 3.0 */
                        T1 = Vds - T0 - 0.1;
                        T2 = sqrt(T1 * T1 + 0.4);
                        T3 = T0 + 0.5 * (T1 + T2);
                        dT3_dVd = 0.5 * (1.0 + T1/T2);

                        T4 = T3 - pParam->B3SOIDDdii;
                        T5 = pParam->B3SOIDDcii / T4;
                        T0 = T5 * T5;
                        dT0_dVd = - 2 * T0 / T4 * dT3_dVd;
                     } else
                     {
                        T0 = dT0_dVd = 0.0;
                     }
                     T0 += 1.0;

                     T3 = pParam->B3SOIDDaii + pParam->B3SOIDDbii / Leff;
                     T4 = 1.0 / (T0 * Vgsteff + T3 * EsatL);
                     T5 = -T4 * T4;
                     T6 = Vgsteff * T4;
                     T7 = EsatL * Vgsteff;
                     Vdsatii = T7 * T4;
   
                     dT4_dVg = T5 * (T0 + T3 * dEsatL_dVg);
                     dT4_dVb = T5 * T3 * dEsatL_dVb;
                     dT4_dVd = T5 * (Vgsteff * dT0_dVd + T3 * dEsatL_dVd);

                     if (selfheat) dT4_dT = T5 * (T3 * dEsatL_dT);
                     else          dT4_dT = 0.0;
   
                     T8 = T4 * Vgsteff;
                     dVdsatii_dVg = T7 * dT4_dVg + T4 * (EsatL + Vgsteff * dEsatL_dVg);
                     dVdsatii_dVb = T7 * dT4_dVb + T8 * dEsatL_dVb;
                     dVdsatii_dVd = T7 * dT4_dVd + T8 * dEsatL_dVd;
                     if (selfheat) dVdsatii_dT  = T7 * dT4_dT  + T8 * dEsatL_dT;
                     else          dVdsatii_dT  = 0.0;
                  } else
                  {
                    Vdsatii = Vdsat;
                    dVdsatii_dVg = dVdsat_dVg;
                    dVdsatii_dVb = dVdsat_dVb;
                    dVdsatii_dVd = dVdsat_dVd;
                    dVdsatii_dT  = dVdsat_dT;
                  }

/* Effective Vds (Vdseff) Calculation */
		  T1 = Vdsat - Vds - pParam->B3SOIDDdelta;
		  dT1_dVg = dVdsat_dVg;
		  dT1_dVd = dVdsat_dVd - 1.0;
		  dT1_dVb = dVdsat_dVb;
		  dT1_dVc = dVdsat_dVc;
		  dT1_dT  = dVdsat_dT;

		  T2 = sqrt(T1 * T1 + 4.0 * pParam->B3SOIDDdelta * Vdsat);
		  T0 = T1 / T2;
		  T3 = 2.0 * pParam->B3SOIDDdelta / T2;
		  dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
		  dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
		  dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;
		  dT2_dVc = T0 * dT1_dVc + T3 * dVdsat_dVc;
                  if (selfheat)
		     dT2_dT  = T0 * dT1_dT  + T3 * dVdsat_dT;
                  else dT2_dT  = 0.0;

		  Vdseff = Vdsat - 0.5 * (T1 + T2);
		  dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg); 
		  dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd); 
		  dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb); 
		  dVdseff_dVc = dVdsat_dVc - 0.5 * (dT1_dVc + dT2_dVc);
                  if (selfheat)
		     dVdseff_dT  = dVdsat_dT  - 0.5 * (dT1_dT  + dT2_dT);
                  else dVdseff_dT  = 0.0;

		  if (Vdseff > Vds)
		      Vdseff = Vds; /* This code is added to fixed the problem
				       caused by computer precision when
				       Vds is very close to Vdseff. */
		  diffVds = Vds - Vdseff;

/* Effective Vdsii for Iii calculation */
		  T1 = Vdsatii - Vds - pParam->B3SOIDDdelta;

		  T2 = sqrt(T1 * T1 + 4.0 * pParam->B3SOIDDdelta * Vdsatii);
		  T0 = T1 / T2;
		  T3 = 2.0 * pParam->B3SOIDDdelta / T2;
                  T4 = T0 + T3;
		  dT2_dVg = T4 * dVdsatii_dVg;
		  dT2_dVd = T4 * dVdsatii_dVd - T0;
                  dT2_dVb = T4 * dVdsatii_dVb;
                  if (selfheat) dT2_dT = T4*dVdsatii_dT;
                  else dT2_dT  = 0.0;

		  Vdseffii = Vdsatii - 0.5 * (T1 + T2);
		  dVdseffii_dVg = 0.5 * (dVdsatii_dVg - dT2_dVg);
		  dVdseffii_dVd = 0.5 * (dVdsatii_dVd - dT2_dVd + 1.0);
		  dVdseffii_dVb = 0.5 * (dVdsatii_dVb - dT2_dVb);
                  if (selfheat)
		     dVdseffii_dT  = 0.5 * (dVdsatii_dT - dT2_dT);
                  else dVdseffii_dT  = 0.0;
		  diffVdsii = Vds - Vdseffii;

/* Calculate VAsat */
		  tmp4 = 1.0 - 0.5 * Abeff * Vdsat / Vgst2Vtm;
		  T9 = WVCoxRds * Vgsteff;
		  T8 = T9 / Vgst2Vtm;
		  T0 = EsatL + Vdsat + 2.0 * T9 * tmp4;
		 
		  T7 = 2.0 * WVCoxRds * tmp4;
		  dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (1.0 + tmp2 * Vgsteff) 
                          - T8 * (Abeff * dVdsat_dVg - Abeff * Vdsat / Vgst2Vtm
			  + Vdsat * dAbeff_dVg);   
			  
		  dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3 * Vgsteff
			  - T8 * (dAbeff_dVb * Vdsat + Abeff * dVdsat_dVb);
		  dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abeff * dVdsat_dVd;
		  dT0_dVc = dVdsat_dVc - T8 * (Abeff * dVdsat_dVc + Vdsat * dAbeff_dVc);

                  if (selfheat)
                  {
		     tmp4 = dRds_dT / Rds + dvsattemp_dT / vsattemp;
		     dT0_dT  = dEsatL_dT + dVdsat_dT + T7 * tmp4 * Vgsteff
			     - T8 * (Abeff * dVdsat_dT - Abeff * Vdsat * dVgst2Vtm_dT
			     / Vgst2Vtm);
                  } else
                     dT0_dT = 0.0;

		  T9 = WVCoxRds * Abeff; 
		  T1 = 2.0 / Lambda - 1.0 + T9; 
		  dT1_dVg = -2.0 * tmp1 +  WVCoxRds * (Abeff * tmp2 + dAbeff_dVg);
		  dT1_dVb = dAbeff_dVb * WVCoxRds + T9 * tmp3;
                  dT1_dVc = dAbeff_dVc * WVCoxRds;
                  if (selfheat)
		     dT1_dT  = T9 * tmp4;
                  else
		     dT1_dT  = 0.0;

		  Vasat = T0 / T1;
		  dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) / T1;
		  dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) / T1;
		  dVasat_dVd = dT0_dVd / T1;
                  dVasat_dVc = (dT0_dVc - Vasat * dT1_dVc) / T1;
                  if (selfheat) dVasat_dT  = (dT0_dT  - Vasat * dT1_dT)  / T1;
                  else dVasat_dT  = 0.0;

/* Calculate VACLM */
		  if ((pParam->B3SOIDDpclm > 0.0) && (diffVds > 1.0e-10))
		  {   T0 = 1.0 / (pParam->B3SOIDDpclm * Abeff * pParam->B3SOIDDlitl);
		      dT0_dVb = -T0 / Abeff * dAbeff_dVb;
		      dT0_dVg = -T0 / Abeff * dAbeff_dVg; 
                      dT0_dVc = -T0 / Abeff * dAbeff_dVc;
		      
		      T2 = Vgsteff / EsatL;
		      T1 = Leff * (Abeff + T2); 
		      dT1_dVg = Leff * ((1.0 - T2 * dEsatL_dVg) / EsatL + dAbeff_dVg);
		      dT1_dVb = Leff * (dAbeff_dVb - T2 * dEsatL_dVb / EsatL);
		      dT1_dVd = -T2 * dEsatL_dVd / Esat;
		      dT1_dVc = Leff * dAbeff_dVc;
                      if (selfheat) dT1_dT  = -T2 * dEsatL_dT / Esat;
                      else dT1_dT  = 0.0;

		      T9 = T0 * T1;
		      VACLM = T9 * diffVds;
		      dVACLM_dVg = T0 * dT1_dVg * diffVds - T9 * dVdseff_dVg
				 + T1 * diffVds * dT0_dVg;
		      dVACLM_dVb = (dT0_dVb * T1 + T0 * dT1_dVb) * diffVds
				 - T9 * dVdseff_dVb;
		      dVACLM_dVd = T0 * dT1_dVd * diffVds + T9 * (1.0 - dVdseff_dVd);
		      dVACLM_dVc = (T1 * dT0_dVc + T0 * dT1_dVc) * diffVds
		         	    - T9 * dVdseff_dVc;
                      if (selfheat)
		         dVACLM_dT  = T0 * dT1_dT * diffVds - T9 * dVdseff_dT;
                      else dVACLM_dT  = 0.0;

		  }
		  else
		  {   VACLM = MAX_EXP;
		      dVACLM_dVd = dVACLM_dVg = dVACLM_dVb = dVACLM_dVc = dVACLM_dT = 0.0;
		  }


/* Calculate VADIBL */
		  if (pParam->B3SOIDDthetaRout > 0.0)
		  {   T8 = Abeff * Vdsat;
		      T0 = Vgst2Vtm * T8;
		      T1 = Vgst2Vtm + T8;
		      dT0_dVg = Vgst2Vtm * Abeff * dVdsat_dVg + T8
			      + Vgst2Vtm * Vdsat * dAbeff_dVg;
		      dT1_dVg = 1.0 + Abeff * dVdsat_dVg + Vdsat * dAbeff_dVg;
		      dT1_dVb = dAbeff_dVb * Vdsat + Abeff * dVdsat_dVb;
		      dT0_dVb = Vgst2Vtm * dT1_dVb;
		      dT1_dVd = Abeff * dVdsat_dVd;
		      dT0_dVd = Vgst2Vtm * dT1_dVd;
		      dT1_dVc = (Abeff * dVdsat_dVc + Vdsat * dAbeff_dVc);
		      dT0_dVc = Vgst2Vtm * dT1_dVc;
                      if (selfheat)
                      {
		         dT0_dT  = dVgst2Vtm_dT * T8 + Abeff * Vgst2Vtm * dVdsat_dT;
		         dT1_dT  = dVgst2Vtm_dT + Abeff * dVdsat_dT;
                      } else
                         dT0_dT = dT1_dT = 0.0;

		      T9 = T1 * T1;
		      T2 = pParam->B3SOIDDthetaRout;
		      VADIBL = (Vgst2Vtm - T0 / T1) / T2;
		      dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2;
		      dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2;
		      dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2;
		      dVADIBL_dVc = (-dT0_dVc / T1 + T0 * dT1_dVc / T9) / T2;
                      if (selfheat)
		         dVADIBL_dT = (dVgst2Vtm_dT - dT0_dT/T1 + T0*dT1_dT/T9) / T2;
                      else dVADIBL_dT = 0.0;

		      T7 = pParam->B3SOIDDpdiblb * Vbseff;
		      if (T7 >= -0.9)
		      {   T3 = 1.0 / (1.0 + T7);
			  VADIBL *= T3;
			  dVADIBL_dVg *= T3;
			  dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam->B3SOIDDpdiblb)
				      * T3;
			  dVADIBL_dVd *= T3;
			  dVADIBL_dVc *= T3;
			  if (selfheat)  dVADIBL_dT  *= T3;
			  else  dVADIBL_dT  = 0.0;
		      }
		      else
/* Added to avoid the discontinuity problem caused by pdiblcb */
		      {   T4 = 1.0 / (0.8 + T7);
			  T3 = (17.0 + 20.0 * T7) * T4;
			  dVADIBL_dVg *= T3;
			  dVADIBL_dVb = dVADIBL_dVb * T3
				      - VADIBL * pParam->B3SOIDDpdiblb * T4 * T4;
			  dVADIBL_dVd *= T3;
			  dVADIBL_dVc *= T3;
			  if (selfheat)  dVADIBL_dT  *= T3;
			  else  dVADIBL_dT  = 0.0;
			  VADIBL *= T3;
		      }
		  }
		  else
		  {   VADIBL = MAX_EXP;
		      dVADIBL_dVd = dVADIBL_dVg = dVADIBL_dVb = dVADIBL_dVc 
			= dVADIBL_dT = 0.0;
		  }

/* Calculate VA */
		  
		  T8 = pParam->B3SOIDDpvag / EsatL;
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
		  dT1_dVc = (tmp1 * dVADIBL_dVc + tmp2 * dVACLM_dVc) / tmp3;
                  if (selfheat)
		     dT1_dT  = (tmp1 * dVADIBL_dT  + tmp2 * dVACLM_dT ) / tmp3;
                  else dT1_dT  = 0.0;

		  Va = Vasat + T0 * T1;
		  dVa_dVg = dVasat_dVg + T1 * dT0_dVg + T0 * dT1_dVg;
		  dVa_dVd = dVasat_dVd + T1 * dT0_dVd + T0 * dT1_dVd;
		  dVa_dVb = dVasat_dVb + T1 * dT0_dVb + T0 * dT1_dVb;
		  dVa_dVc = dVasat_dVc + T0 * dT1_dVc;
                  if (selfheat)
		     dVa_dT  = dVasat_dT  + T1 * dT0_dT  + T0 * dT1_dT;
                  else dVa_dT  = 0.0;

/* Calculate Ids */
		  CoxWovL = model->B3SOIDDcox * Weff / Leff;
		  beta = ueff * CoxWovL;
		  dbeta_dVg = CoxWovL * dueff_dVg + beta * dWeff_dVg / Weff;
		  dbeta_dVd = CoxWovL * dueff_dVd;
		  dbeta_dVb = CoxWovL * dueff_dVb + beta * dWeff_dVb / Weff;
		  if (selfheat)  dbeta_dT  = CoxWovL * dueff_dT;
		  else  dbeta_dT  = 0.0;

		  T0 = 1.0 - 0.5 * Abeff * Vdseff / Vgst2Vtm;
		  dT0_dVg = -0.5 * (Abeff * dVdseff_dVg 
			  - Abeff * Vdseff / Vgst2Vtm + Vdseff * dAbeff_dVg) / Vgst2Vtm;
		  dT0_dVd = -0.5 * Abeff * dVdseff_dVd / Vgst2Vtm;
		  dT0_dVb = -0.5 * (Abeff * dVdseff_dVb + dAbeff_dVb * Vdseff)
			  / Vgst2Vtm;
		  dT0_dVc = -0.5 * (Abeff * dVdseff_dVc + dAbeff_dVc * Vdseff)
			     / Vgst2Vtm;
		  if (selfheat)  
                     dT0_dT  = -0.5 * (Abeff * dVdseff_dT
                             - Abeff * Vdseff / Vgst2Vtm * dVgst2Vtm_dT)
			     / Vgst2Vtm;
		  else dT0_dT = 0.0;

		  fgche1 = Vgsteff * T0;
		  dfgche1_dVg = Vgsteff * dT0_dVg + T0; 
		  dfgche1_dVd = Vgsteff * dT0_dVd;
		  dfgche1_dVb = Vgsteff * dT0_dVb;
		  dfgche1_dVc = Vgsteff * dT0_dVc;
		  if (selfheat)  dfgche1_dT  = Vgsteff * dT0_dT;
		  else  dfgche1_dT  = 0.0;

		  T9 = Vdseff / EsatL;
		  fgche2 = 1.0 + T9;
		  dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) / EsatL;
		  dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) / EsatL;
		  dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) / EsatL;
		  dfgche2_dVc = (dVdseff_dVc) / EsatL;
		  if (selfheat)  dfgche2_dT  = (dVdseff_dT  - T9 * dEsatL_dT)  / EsatL;
		  else  dfgche2_dT  = 0.0;

		  gche = beta * fgche1 / fgche2;
		  dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg
			    - gche * dfgche2_dVg) / fgche2;
		  dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd
			    - gche * dfgche2_dVd) / fgche2;
		  dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb
			    - gche * dfgche2_dVb) / fgche2;
		  dgche_dVc = (beta * dfgche1_dVc - gche * dfgche2_dVc) / fgche2;
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
		  dIdl_dVc = (gche * dVdseff_dVc + T9 * dgche_dVc) / T0;
		  if (selfheat)
		     dIdl_dT  = (gche * dVdseff_dT + T9 * dgche_dT
			      - Idl * dRds_dT * gche) / T0;
		  else dIdl_dT  = 0.0;

		  T9 =  diffVds / Va;
		  T0 =  1.0 + T9;
		  here->B3SOIDDids = Ids = Idl * T0;

		  Gm0 = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVa_dVg) / Va;
		  Gds0 = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd
			    - T9 * dVa_dVd) / Va;
		  Gmb0 = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVa_dVb) / Va;
		  Gmc = T0 * dIdl_dVc - Idl * (dVdseff_dVc + T9 * dVa_dVc) / Va;
                  if (selfheat) 
		     GmT0 = T0 * dIdl_dT - Idl * (dVdseff_dT + T9 * dVa_dT) / Va;
                  else GmT0 = 0.0;

/* This includes all dependencies from Vgsteff, Vbseff, Vcs */
		  Gm = Gm0 * dVgsteff_dVg + Gmb0 * dVbseff_dVg + Gmc * dVcs_dVg;
		  Gmb = Gm0 * dVgsteff_dVb + Gmb0 * dVbseff_dVb + Gmc * dVcs_dVb;
		  Gds = Gm0 * dVgsteff_dVd + Gmb0 * dVbseff_dVd + Gmc * dVcs_dVd + Gds0;
		  Gme = Gm0 * dVgsteff_dVe + Gmb0 * dVbseff_dVe + Gmc * dVcs_dVe;
		  if (selfheat)
		     GmT = Gm0 * dVgsteff_dT + Gmb0 * dVbseff_dT  + Gmc * dVcs_dT + GmT0;
		  else GmT = 0.0;

/* calculate substrate current Iii */
	          T2 = pParam->B3SOIDDalpha1 + pParam->B3SOIDDalpha0 / Leff;
		  if ((T2 <= 0.0) || (pParam->B3SOIDDbeta0 <= 0.0))
		  {   Giig = Giib = Giid = Giie = GiiT = 0.0;
		      here->B3SOIDDiii = Iii = 0.0;
		  }
		  else 
		  {   
		      T5 = pParam->B3SOIDDbeta0;
		      if (diffVdsii > T5 / EXP_THRESHOLD)
		      {   
			  T0 = -T5 / diffVdsii;
                          T10 = T0 / diffVdsii;
                          dT0_dVg = T10 * dVdseffii_dVg;
			  T1 = T2 * diffVdsii * exp(T0);

			  T3 = T1 / diffVdsii * (T0 - 1.0);
			  dT1_dVg = T1 * (dT0_dVg - dVdseffii_dVg / diffVdsii);
			  dT1_dVd = -T3 * (1.0 - dVdseffii_dVd);
			  dT1_dVb =  T3 * dVdseffii_dVb;
			  if (selfheat)  dT1_dT  = T3 * dVdseffii_dT;
			  else  dT1_dT  = 0.0;
		      }
		      else
		      {   T3 = T2 * MIN_EXP;
			  T1 = T3 * diffVdsii;
			  dT1_dVg = -T3 * dVdseffii_dVg;
			  dT1_dVd = T3 * (1.0 - dVdseffii_dVd);
			  dT1_dVb = -T3 * dVdseffii_dVb;
			  if (selfheat)  dT1_dT  = -T3 * dVdseffii_dT;
			  else  dT1_dT  = 0.0;
		      }

		      here->B3SOIDDiii = Iii = T1 * Ids;

		      T2 = T1 * Gm0 + Ids * dT1_dVg;
		      T3 = T1 * Gds0 + Ids * dT1_dVd;
		      T4 = T1 * Gmb0 + Ids * dT1_dVb;
		      T5 = T1 * Gmc;
		      if (selfheat)  T6 = T1 * GmT0 + Ids * dT1_dT;
		      else  T6 = 0.0;

		      Giig = T2 * dVgsteff_dVg + T4 * dVbseff_dVg + T5 * dVcs_dVg;
		      Giib = T2 * dVgsteff_dVb + T4 * dVbseff_dVb + T5 * dVcs_dVb;
		      Giid = T2 * dVgsteff_dVd + T4 * dVbseff_dVd + T5 * dVcs_dVd + T3;
		      Giie = T2 * dVgsteff_dVe + T4 * dVbseff_dVe + T5 * dVcs_dVe;
                      if (selfheat)
		         GiiT = T2 * dVgsteff_dT  + T4 * dVbseff_dT  + T5 * dVcs_dT + T6;
                      else
		         GiiT = 0.0;
		  }

/*  calculate GIDL current  */
		  T0 = 3 * model->B3SOIDDtox;
		  /* For drain side */
		  T1 = (Vds - Vgs_eff - pParam->B3SOIDDngidl) / T0;
		  if ((pParam->B3SOIDDagidl <= 0.0) || (pParam->B3SOIDDbgidl <= 0.0) ||
		      (T1 <= 0.0))
		  {   Idgidl = Gdgidld = Gdgidlg = 0.0;
		  }
		  else {
		     dT1_dVd = 1 / T0;
		     dT1_dVg = - dT1_dVd * dVgs_eff_dVg;
		     T2 = pParam->B3SOIDDbgidl / T1;
		     if (T2 < EXP_THRESHOLD)
		     {
			Idgidl = pParam->B3SOIDDweff * pParam->B3SOIDDagidl * T1 * exp(-T2);
			T3 = Idgidl / T1 * (T2 + 1);
			Gdgidld = T3 * dT1_dVd;
			Gdgidlg = T3 * dT1_dVg;
		     } else
		     {
			T3 = pParam->B3SOIDDweff * pParam->B3SOIDDagidl * MIN_EXP;
			Idgidl = T3 * T1 ;
			Gdgidld  = T3 * dT1_dVd;
			Gdgidlg  = T3 * dT1_dVg;
		     } 
		  }
                  here->B3SOIDDigidl = Idgidl;

		  /* For source side */
		  T1 = (- Vgs_eff - pParam->B3SOIDDngidl) / T0;
		  if ((pParam->B3SOIDDagidl <= 0.0) || (pParam->B3SOIDDbgidl <= 0.0) 
                        || (T1 <= 0.0))
		  {   Isgidl = Gsgidlg = 0;
		  }
		  else
		  {
		     dT1_dVg = - dVgs_eff_dVg / T0;
		     T2 = pParam->B3SOIDDbgidl / T1;
		     if (T2 < EXP_THRESHOLD)
		     {
			Isgidl = pParam->B3SOIDDweff * pParam->B3SOIDDagidl * T1 * exp(-T2);
			T3 = Isgidl / T1 * (T2 + 1);
			Gsgidlg = T3 * dT1_dVg;
		     } else
		     {
			T3 = pParam->B3SOIDDweff * pParam->B3SOIDDagidl * MIN_EXP;
			Isgidl = T3 * T1 ;
			Gsgidlg = T3 * dT1_dVg;
		     } 
		  }

/* calculate diode and BJT current */

		  WTsi = pParam->B3SOIDDweff * model->B3SOIDDtsi;
		  NVtm1 = Vtm * pParam->B3SOIDDndiode;
		  NVtm2 = Vtm * pParam->B3SOIDDntun;

		  /* Create exponents first */
		  T0 = Vbs / NVtm1;
		  if (T0 < 30)
		  {
		     ExpVbs1 = exp(T0);
		     dExpVbs1_dVb = ExpVbs1 / NVtm1;
                     if (selfheat)  dExpVbs1_dT = - T0 * ExpVbs1 / Temp;
                  } else
		  {
                     T1 = 1.0686e13; /* exp(30) */
                     dExpVbs1_dVb = T1 / NVtm1;
                     ExpVbs1 = dExpVbs1_dVb * Vbs - 29.0 * T1;
                     if (selfheat)  dExpVbs1_dT = - dExpVbs1_dVb * Vbs / Temp;
                  } 

		  T0 = Vbd / NVtm1;
		  if (T0 < 30)
		  {
		     ExpVbd1 = exp(T0);
		     dExpVbd1_dVb = ExpVbd1 / NVtm1;
                     if (selfheat)  dExpVbd1_dT = - T0 * ExpVbd1 / Temp;
                  } else
		  {
                     T1 = 1.0686e13; /* exp(30) */
                     dExpVbd1_dVb = T1 / NVtm1;
                     ExpVbd1 = dExpVbd1_dVb * Vbd - 29.0 * T1;
                     if (selfheat)  dExpVbd1_dT = - dExpVbd1_dVb * Vbd / Temp;
                  } 

                  if (jtun > 0.0)
		  {
                     T0 = -Vbs / NVtm2;
                     if (T0 < 30)
                     {
                        ExpVbs4 = exp(T0);
		        dExpVbs4_dVb = - ExpVbs4 / NVtm2;
                        if (selfheat)  dExpVbs4_dT = - T0 * ExpVbs4 / Temp;
                     } else
                     {
                        T1 = 1.0686e13; /* exp(30) */
		        dExpVbs4_dVb = - T1 / NVtm2;
                        ExpVbs4 = dExpVbs4_dVb * Vbs - 29.0 * T1;
                        if (selfheat)  dExpVbs4_dT = - dExpVbs4_dVb * Vbs / Temp;
                     }

                     T0 = -Vbd / NVtm2;
                     if (T0 < 30)
                     {
                        ExpVbd4 = exp(T0);
		        dExpVbd4_dVb = - ExpVbd4 / NVtm2;
                        if (selfheat)  dExpVbd4_dT = - T0 * ExpVbd4 / Temp;
                     } else
                     {
                        T1 = 1.0686e13; /* exp(30) */
		        dExpVbd4_dVb = - T1 / NVtm2;
                        ExpVbd4 = dExpVbd4_dVb * Vbd - 29.0 * T1;
                        if (selfheat)  dExpVbd4_dT = - dExpVbd4_dVb * Vbd / Temp;
                     }
		  }

		  /* Ibs1 / Ibd1 */
                  if (jdif == 0.0)
                  {
                     Ibs1 = dIbs1_dVb = dIbs1_dT = 0.0;
		     Ibd1 = dIbd1_dVd = dIbd1_dVb = dIbd1_dT = 0.0;
                  }
                  else
                  {
		     T5 = WTsi * jdif;
		     Ibs1 = T5 * (ExpVbs1 - 1.0);
                     dIbs1_dVb = T5 * dExpVbs1_dVb;
                     if (selfheat)
                        dIbs1_dT = Ibs1 / jdif * djdif_dT + T5 * dExpVbs1_dT;

                     Ibd1 = T5 * (ExpVbd1 - 1.0);
                     dIbd1_dVb = T5 * dExpVbd1_dVb;
                     dIbd1_dVd = -dIbd1_dVb;
                     if (selfheat)
                        dIbd1_dT = Ibd1 / jdif * djdif_dT + T5 * dExpVbd1_dT;
                  }

		  /* Ibs2 */
                  if (jrec == 0.0)
                  {
                     Ibs2 = dIbs2_dVb = dIbs2_dT = 0.0;
                     Ibd2 = dIbd2_dVb = dIbd2_dVd = dIbd2_dT = 0.0;
                  } 
                  else 
                  { 
                     ExpVbs2 = sqrt(ExpVbs1);
		     if (ExpVbs2 > 1e-20)
		     {
		        dExpVbs2_dVb = 0.5 / ExpVbs2 * dExpVbs1_dVb;
                        if (selfheat)  dExpVbs2_dT = 0.5 / ExpVbs2 * dExpVbs1_dT;
		     }
		     else
		     {
		        dExpVbs2_dVb = dExpVbs2_dT = 0.0;
		     }

                     ExpVbd2 = sqrt(ExpVbd1);
		     if (ExpVbd2 > 1e-20)
		     {
		        dExpVbd2_dVb = 0.5 / ExpVbd2 * dExpVbd1_dVb;
                        if (selfheat)  dExpVbd2_dT = 0.5 / ExpVbd2 * dExpVbd1_dT;
		     }
		     else
		     {
		        dExpVbd2_dVb = dExpVbd2_dT = 0.0;
		     }

		     T8 = WTsi * jrec;
		     T9 = 0.5 * T8 / NVtm1;
		     Ibs2 = T8 * (ExpVbs2 - 1.0);
                     dIbs2_dVb = T8 * dExpVbs2_dVb;
                     if (selfheat)
		        dIbs2_dT  = Ibs2 / jrec * djrec_dT + T8 * dExpVbs2_dT;

		     T8 = WTsi * jrec;
		     T9 = 0.5 * T8 / NVtm1;
		     Ibd2 = T8 * (ExpVbd2 - 1.0);
                     dIbd2_dVb = T8 * dExpVbd2_dVb;
                     dIbd2_dVd = -dIbd2_dVb;
                     if (selfheat)
		        dIbd2_dT  = Ibd2 / jrec * djrec_dT + T8 * dExpVbd2_dT;
                  }

		  /* Ibjt */
		  if ((here->B3SOIDDbjtoff == 1) || (Vds == 0.0) || 
                     (jbjt == 0.0))
		  {  
		     Ibs3 = dIbs3_dVb = dIbs3_dVd = dIbs3_dT = 0.0;
		     Ibd3 = dIbd3_dVb = dIbd3_dVd = dIbd3_dT = 0.0;
		     here->B3SOIDDic = Ic = Gcd = Gcb = GcT = 0.0;
		  } 
		  else {
		     T0 = Leff - pParam->B3SOIDDkbjt1 * Vds;
   		     T1 = T0 / pParam->B3SOIDDedl;
                     dT1_dVd = - pParam->B3SOIDDkbjt1 / pParam->B3SOIDDedl;
                     if (T1 < 1e-3) /* Limit to 1/2e4 */
                     {  T2 = 1.0 / (3.0 - 2.0e3 * T1);
                        T1 = (2.0e-3 - T1) * T2;
                        dT1_dVd *= T2 * T2;
                     } else if (T1 > 1.0)
                     {  T1 = 1.0;
                        dT1_dVd = 0.0;
                     }
		     BjtA = 1 - 0.5 * T1 * T1;
   		     dBjtA_dVd = - T1 * dT1_dVd;

		     T5 = WTsi * jbjt;
		     Ibjt = T5 * (ExpVbs1 - ExpVbd1);
                     dIbjt_dVb = T5 * (dExpVbs1_dVb - dExpVbd1_dVb);
                     dIbjt_dVd = T5 * dExpVbd1_dVb;
                     if (selfheat)
		        dIbjt_dT  = T5 * (dExpVbs1_dT - dExpVbd1_dT)
				  + Ibjt / jbjt * djbjt_dT;

		     T3 = (1.0 - BjtA) * T5;
		     T4 = - T5 * dBjtA_dVd;
		     Ibs3 = T3 * ExpVbs1;
		     dIbs3_dVb = T3 * dExpVbs1_dVb;
		     dIbs3_dVd = T4 * ExpVbs1;
		     if (selfheat)  dIbs3_dT  = Ibs3 / jbjt * djbjt_dT + T3 * dExpVbs1_dT;

		     Ibd3 = T3 * ExpVbd1;
		     dIbd3_dVb = T3 * dExpVbd1_dVb;
		     dIbd3_dVd = T4 * ExpVbd1 - dIbd3_dVb;
		     if (selfheat)  dIbd3_dT  = Ibd3 / jbjt * djbjt_dT + T3 * dExpVbd1_dT;

		     here->B3SOIDDic = Ic = Ibjt - Ibs3 + Ibd3;
		     Gcd = dIbjt_dVd - dIbs3_dVd + dIbd3_dVd;
		     Gcb = dIbjt_dVb - dIbs3_dVb + dIbd3_dVb;
		     if (selfheat)  GcT = dIbjt_dT  - dIbs3_dT  + dIbd3_dT;
		     else	    GcT = 0.0;
		  }

		  if (jtun == 0.0) 
                  {
		     Ibs4 = dIbs4_dVb = dIbs4_dT = 0.0;
                     Ibd4 = dIbd4_dVb = dIbd4_dVd = dIbd4_dT = 0.0;
		  }
		  else 
		  {
                     T5 = WTsi * jtun;
                     Ibs4 = T5 * (1.0 - ExpVbs4);
                     dIbs4_dVb = - T5 * dExpVbs4_dVb;
                     if (selfheat)
                        dIbs4_dT = Ibs4 / jtun * djtun_dT - T5 * dExpVbs4_dT;

                     Ibd4 = T5 * (1.0 - ExpVbd4);
                     dIbd4_dVb = - T5 * dExpVbd4_dVb;
                     dIbd4_dVd = -dIbd4_dVb;
                     if (selfheat)
                        dIbd4_dT = Ibd4 / jtun * djtun_dT - T5 * dExpVbd4_dT;
		  }

here->B3SOIDDdum1 = Ibs3 + Ibd4;
here->B3SOIDDdum2 = Ibs1;
here->B3SOIDDdum3 = Ibjt;
here->B3SOIDDdum4 = Ic;

                  here->B3SOIDDitun = - Ibd3 - Ibd4;
		  here->B3SOIDDibs = Ibs = Ibs1 + Ibs2 + Ibs3 + Ibs4;
		  here->B3SOIDDibd = Ibd = Ibd1 + Ibd2 + Ibd3 + Ibd4;

		  Gjsb = dIbs1_dVb + dIbs2_dVb + dIbs3_dVb + dIbs4_dVb;
		  Gjsd = dIbs3_dVd;
		  if (selfheat)  GjsT = dIbs1_dT  + dIbs2_dT  + dIbs3_dT  + dIbs4_dT;
                  else   GjsT = 0.0;

		  Gjdb = dIbd1_dVb + dIbd2_dVb + dIbd3_dVb + dIbd4_dVb;
		  Gjdd = dIbd1_dVd + dIbd2_dVd + dIbd3_dVd + dIbd4_dVd;
                  if (selfheat)  GjdT = dIbd1_dT  + dIbd2_dT + dIbd3_dT + dIbd4_dT;
                  else   GjdT = 0.0;


	/* Current through body resistor */
		  /* Current going out is +ve */
		  if ((here->B3SOIDDbodyMod == 0) || (here->B3SOIDDbodyMod == 2)) 
                  {
		     Ibp = Gbpbs = Gbpgs = Gbpds = Gbpes = Gbpps = GbpT = 0.0;
		  }
		  else { /* here->B3SOIDDbodyMod == 1 */
                     if (pParam->B3SOIDDrbody < 1e-30)
                     {
                        if (here->B3SOIDDrbodyext <= 1e-30)
                           T0 = 1.0 / 1e-30;
                        else
                           T0 = 1.0 / here->B3SOIDDrbodyext;
                        Ibp = Vbp * T0;
                        Gbpbs = T0 * dVbp_dVb;
			Gbpps = T0 * dVbp_dVp;
			Gbpgs = T0 * dVbp_dVg;
                        Gbpds = T0 * dVbp_dVd;
                        Gbpes = 0.0;
			if (selfheat)  GbpT = T0 * dVbp_dT;
			else  GbpT = 0.0;
                     } else
                     {  T0 = 1.0 / pParam->B3SOIDDrbody;
                           if (Vbp >= 0.0)
                           {
                              T1 = sqrt(Vcs);
                              T3 = T1 * T0;
                              T5 = 1.0 + here->B3SOIDDrbodyext * T3;
                              T6 = T3 / T5;
                              T2 = 0.5 * T0 / T1;
                              T7 = T2 / (T5 * T5);
                              Ibp = Vbp * T6;

		/*  Whoa, again these derivatives are convoluted, but correct  */

                              Gbpbs = T6 * dVbp_dVb + Vbp * T7 * dVcs_dVb;
                              Gbpps = T6 * dVbp_dVp;
                              Gbpgs = T6 * dVbp_dVg + Vbp * T7 * dVcs_dVg;
                              Gbpds = T6 * dVbp_dVg + Vbp * T7 * dVcs_dVd;
                              Gbpes = T6 * dVbp_dVg + Vbp * T7 * dVcs_dVe;
                              if (selfheat)
			         GbpT = T6 * dVbp_dT + Vbp * T7 * dVcs_dT;
                              else  GbpT = 0.0;

                           } else
                           {
                              T1 = sqrt(Vpsdio - Vbs0eff);
                              T3 = T1 * T0;
                              T5 = 1.0 + here->B3SOIDDrbodyext * T3;
                              T6 = T3 / T5;
                              T2 = 0.5 * T0 / T1;
                              Ibp = Vbp * T6;
                              T7 = T2 / (T5 * T5);
                              Gbpbs = T6 * dVbp_dVb;
                              Gbpps = T6 * dVbp_dVp + Vbp * T7 * dVpsdio_dVp;
                              Gbpgs = Vbp * T7 * (dVpsdio_dVg - dVbs0eff_dVg);
                              Gbpds = Vbp * T7 * (dVpsdio_dVd - dVbs0eff_dVd);
                              Gbpes = Vbp * T7 * (dVpsdio_dVe - dVbs0eff_dVe);
			      if (selfheat)
                                 GbpT = Vbp * T7 * (dVpsdio_dT  - dVbs0eff_dT);
			      else  GbpT = 0.0;
                           }
                     }
		  }

		  here->B3SOIDDibp = Ibp;
		  here->B3SOIDDgbpbs = Gbpbs;
		  here->B3SOIDDgbpgs = Gbpgs;
		  here->B3SOIDDgbpds = Gbpds;
		  here->B3SOIDDgbpes = Gbpes;
		  here->B3SOIDDgbpps = Gbpps;
		  if (selfheat)
		      here->B3SOIDDgbpT = GbpT;
		  else {
		      GbpT = 0.0;
		      here->B3SOIDDgbpT = 0.0;
		  }
		  here->B3SOIDDcbodcon = Ibp - (Gbpbs * Vbs + Gbpgs * Vgs 
				     + Gbpds * Vds + Gbpes * Ves + Gbpps * Vps 
                                     + GbpT * delTemp);

		  /*  Current going out of drainprime node into the drain of device  */
		  /*  "node" means the SPICE circuit node  */

                  here->B3SOIDDcdrain = Ids + Ic;
                  here->B3SOIDDcd = Ids + Ic - Ibd + Iii + Idgidl;
                  here->B3SOIDDcb = Ibs + Ibd + Ibp - Iii - Idgidl - Isgidl;

   		  here->B3SOIDDgds = Gds + Gcd;
   		  here->B3SOIDDgm = Gm;
   		  here->B3SOIDDgmbs = Gmb + Gcb;
		  here->B3SOIDDgme = Gme;
		  if (selfheat)
		      here->B3SOIDDgmT = GmT + GcT;
		  else
		      here->B3SOIDDgmT = 0.0;

                  /*  note that sign is switched because power flows out 
                      of device into the temperature node.  
                      Currently ommit self-heating due to bipolar current
                      because it can cause convergence problem*/

                  here->B3SOIDDgtempg = -Gm * Vds;
                  here->B3SOIDDgtempb = -Gmb * Vds;
                  here->B3SOIDDgtempe = -Gme * Vds;
                  here->B3SOIDDgtempT = -GmT * Vds;
                  here->B3SOIDDgtempd = -Gds * Vds - Ids;
		  here->B3SOIDDcth = - Ids * Vds - model->B3SOIDDtype * 
                                  (here->B3SOIDDgtempg * Vgs + here->B3SOIDDgtempb * Vbs 
                                 + here->B3SOIDDgtempe * Ves + here->B3SOIDDgtempd * Vds)
                                 - here->B3SOIDDgtempT * delTemp;

                  /*  Body current which flows into drainprime node from the drain of device  */

		  here->B3SOIDDgjdb = Gjdb - Giib;
		  here->B3SOIDDgjdd = Gjdd - (Giid + Gdgidld);
		  here->B3SOIDDgjdg = - (Giig + Gdgidlg);
		  here->B3SOIDDgjde = - Giie;
		  if (selfheat) here->B3SOIDDgjdT = GjdT - GiiT;
		  else here->B3SOIDDgjdT = 0.0;
		  here->B3SOIDDcjd = Ibd - Iii - Idgidl - here->B3SOIDDminIsub/2
				 - (here->B3SOIDDgjdb * Vbs + here->B3SOIDDgjdd * Vds
				 +  here->B3SOIDDgjdg * Vgs + here->B3SOIDDgjde * Ves
				 +  here->B3SOIDDgjdT * delTemp);

		  /*  Body current which flows into sourceprime node from the source of device  */

   		  here->B3SOIDDgjsb = Gjsb;
   		  here->B3SOIDDgjsd = Gjsd;
		  here->B3SOIDDgjsg = - Gsgidlg;
		  if (selfheat) here->B3SOIDDgjsT = GjsT;
		  else here->B3SOIDDgjsT = 0.0;
		  here->B3SOIDDcjs = Ibs - Isgidl - here->B3SOIDDminIsub/2
				  - (here->B3SOIDDgjsb * Vbs + here->B3SOIDDgjsd * Vds
				  +  here->B3SOIDDgjsg * Vgs + here->B3SOIDDgjsT * delTemp);

		  /*  Current flowing into body node  */

		  here->B3SOIDDgbbs = Giib - Gjsb - Gjdb - Gbpbs;
		  here->B3SOIDDgbgs = Giig + Gdgidlg + Gsgidlg - Gbpgs;
   		  here->B3SOIDDgbds = Giid + Gdgidld - Gjsd - Gjdd - Gbpds;
		  here->B3SOIDDgbes = Giie - Gbpes;
		  here->B3SOIDDgbps = - Gbpps;
		  if (selfheat) here->B3SOIDDgbT = GiiT - GjsT - GjdT - GbpT;
		  else here->B3SOIDDgbT = 0.0;
		  here->B3SOIDDcbody = Iii + Idgidl + Isgidl - Ibs - Ibd - Ibp + here->B3SOIDDminIsub
				   - (here->B3SOIDDgbbs * Vbs + here->B3SOIDDgbgs * Vgs
				   + here->B3SOIDDgbds * Vds + here->B3SOIDDgbps * Vps
				   + here->B3SOIDDgbes * Ves + here->B3SOIDDgbT * delTemp);

	/* Calculate Qinv for Noise analysis */

		  T1 = Vgsteff * (1.0 - 0.5 * Abeff * Vdseff / Vgst2Vtm);
		  here->B3SOIDDqinv = -model->B3SOIDDcox * pParam->B3SOIDDweff * Leff * T1;

	/*  Begin CV (charge) model  */

		  if ((model->B3SOIDDxpart < 0) || (!ChargeComputationNeeded))
		  {   qgate  = qdrn = qsrc = qbody = 0.0;
		      here->B3SOIDDcggb = here->B3SOIDDcgsb = here->B3SOIDDcgdb = 0.0;
		      here->B3SOIDDcdgb = here->B3SOIDDcdsb = here->B3SOIDDcddb = 0.0;
		      here->B3SOIDDcbgb = here->B3SOIDDcbsb = here->B3SOIDDcbdb = 0.0;
		      goto finished;
		  }
		  else 
		  {
		       CoxWL  = model->B3SOIDDcox * pParam->B3SOIDDweffCV
			      * pParam->B3SOIDDleffCV;

                       /* By using this Vgsteff,cv, discontinuity in moderate
                          inversion charges can be avoid.  However, in capMod=3,
                          Vdsat from IV is used.  The dVdsat_dVg is referred to
                          the IV Vgsteff and therefore induces error in the charges
                          derivatives.  Fortunately, Vgsteff,iv and Vgsteff,cv are
                          different only in subthreshold where Qsubs is neglectible.
                          So the errors in derivatives is not a serious problem */

		       if ((VgstNVt > -EXP_THRESHOLD) && (VgstNVt < EXP_THRESHOLD))
		       {   ExpVgst *= ExpVgst;
			   Vgsteff = n * Vtm * log(1.0 + ExpVgst);
			   T0 = ExpVgst / (1.0 + ExpVgst);
			   T1 = -T0 * (dVth_dVb + Vgst / n * dn_dVb) + Vgsteff / n * dn_dVb;
			   dVgsteff_dVd = -T0 * (dVth_dVd + Vgst / n * dn_dVd)
					  + Vgsteff / n * dn_dVd + T1 * dVbseff_dVd;
			   dVgsteff_dVg = T0 * dVgs_eff_dVg + T1 * dVbseff_dVg;
                           dVgsteff_dVb = T1 * dVbseff_dVb;
                           dVgsteff_dVe = T1 * dVbseff_dVe;
                           if (selfheat)
			      dVgsteff_dT  = -T0 * (dVth_dT + Vgst / Temp) + Vgsteff / Temp
					  + T1 * dVbseff_dT;
                           else dVgsteff_dT  = 0.0;
		       } 

		       Vfb = Vth - phi - pParam->B3SOIDDk1 * sqrtPhis;

		       dVfb_dVb = dVth_dVb - pParam->B3SOIDDk1 * dsqrtPhis_dVb;
		       dVfb_dVd = dVth_dVd;
		       dVfb_dT  = dVth_dT;

		      if ((model->B3SOIDDcapMod == 2) || (model->B3SOIDDcapMod == 3))
		      {   
			  /* Necessary because charge behaviour very strange at 
			     Vgsteff = 0 */
			  Vgsteff += 1e-4;

			  /* Something common in capMod 2 and 3 */
			  V3 = Vfb - Vgs_eff + Vbseff - DELTA_3;
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
			  dVfbeff_dVb = (1.0 - T1 - T2) * dVfb_dVb - T1;
			  dVfbeff_dVrg = T1 * dVgs_eff_dVg;
			  if (selfheat) dVfbeff_dT  = (1.0 - T1 - T2) * dVfb_dT;
                          else  dVfbeff_dT = 0.0;

			  Qac0 = -CoxWL * (Vfbeff - Vfb);
			  dQac0_dVrg = -CoxWL * dVfbeff_dVrg;
			  dQac0_dVd = -CoxWL * (dVfbeff_dVd - dVfb_dVd);
			  dQac0_dVb = -CoxWL * (dVfbeff_dVb - dVfb_dVb);
			  if (selfheat) dQac0_dT  = -CoxWL * (dVfbeff_dT - dVfb_dT);
                          else  dQac0_dT = 0.0;

			  T0 = 0.5 * K1;
			  T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff;
			  if (pParam->B3SOIDDk1 == 0.0)
			  {   T1 = 0.0;
			      T2 = 0.0;
			  }
			  else if (T3 < 0.0)
			  {   T1 = T0 + T3 / pParam->B3SOIDDk1;
			      T2 = CoxWL;
			  }
			  else
			  {   T1 = sqrt(T0 * T0 + T3);
			      T2 = CoxWL * T0 / T1;
			  }

			  Qsub0 = CoxWL * K1 * (T0 - T1);

			  dQsub0_dVrg = T2 * (dVfbeff_dVrg - dVgs_eff_dVg);
			  dQsub0_dVg = T2;
			  dQsub0_dVd = T2 * dVfbeff_dVd;
			  dQsub0_dVb = T2 * (dVfbeff_dVb + 1);
			  if (selfheat) dQsub0_dT  = T2 * dVfbeff_dT;
                          else  dQsub0_dT = 0.0;

			  One_Third_CoxWL = CoxWL / 3.0;
			  Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
			  AbulkCV = Abulk0 * pParam->B3SOIDDabulkCVfactor;
			  dAbulkCV_dVb = pParam->B3SOIDDabulkCVfactor * dAbulk0_dVb;

			  /*  This is actually capMod=2 calculation  */
			  VdsatCV = Vgsteff / AbulkCV;
			  dVdsatCV_dVg = 1.0 / AbulkCV;
			  dVdsatCV_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;
                          VdsatCV += 1e-5;

			  V4 = VdsatCV - Vds - DELTA_4;
			  T0 = sqrt(V4 * V4 + 4.0 * DELTA_4 * VdsatCV);
			  VdseffCV = VdsatCV - 0.5 * (V4 + T0);
			  T1 = 0.5 * (1.0 + V4 / T0);
			  T2 = DELTA_4 / T0;
			  T3 = (1.0 - T1 - T2) / AbulkCV;
			  dVdseffCV_dVg = T3;
			  dVdseffCV_dVd = T1;
			  dVdseffCV_dVb = -T3 * VdsatCV * dAbulkCV_dVb;

			  if (model->B3SOIDDcapMod == 2)
			  {
			      /* VdsCV Make it compatible with capMod 3 */
			      VdsCV = VdseffCV;
			      dVdsCV_dVg = dVdseffCV_dVg;
			      dVdsCV_dVd = dVdseffCV_dVd;
			      dVdsCV_dVb = dVdseffCV_dVb;
                              dVdsCV_dVc = 0.0;

			       /* This is good for Xc calculation */
			       VdsCV += 1e-5;
			       if (VdsCV > (VdsatCV-1e-7)) VdsCV=VdsatCV-1e-7;

			       /* VcsCV calculation */ 
                               T1 = VdsCV - Vcs - VdsCV * VdsCV * DELTA_Vcscv;
                               T5 = 2 * DELTA_Vcscv;
                               T2 = sqrt(T1 * T1 + T5 * VdsCV * VdsCV);

                               dT1_dVb = dVdsCV_dVb * (1.0 - 2.0 * VdsCV * DELTA_Vcscv);
                               dT2_dVb = (T1 * dT1_dVb + T5 * VdsCV * dVdsCV_dVb)/T2;

                               dT1_dVd = dVdsCV_dVd * (1.0 - 2.0 * VdsCV * DELTA_Vcscv);
                               dT2_dVd = (T1 * dT1_dVd + T5 * VdsCV * dVdsCV_dVd)/ T2;

                               dT1_dVg = dVdsCV_dVg * (1.0 - 2.0 * VdsCV * DELTA_Vcscv) ;
                               dT2_dVg = (T1 * dT1_dVg + T5 * VdsCV * dVdsCV_dVg)/T2;

                               dT1_dVc = -1;
                               dT2_dVc = T1 * dT1_dVc / T2;

			       VcsCV = Vcs + 0.5 * (T1 - T2);

			       dVcsCV_dVb = 0.5 * (dT1_dVb - dT2_dVb);
			       dVcsCV_dVg = 0.5 * (dT1_dVg - dT2_dVg);
			       dVcsCV_dVd = 0.5 * (dT1_dVd - dT2_dVd);
			       dVcsCV_dVc = 1.0 + 0.5 * (dT1_dVc - dT2_dVc);

			       if (VcsCV < 0.0) VcsCV = 0.0;
			       else if (VcsCV > VdsCV) VcsCV = VdsCV;

			       /* Xc calculation */
			       T3 = 2 * VdsatCV  - VcsCV;
			       T4 = 2 * VdsatCV - VdsCV;
			       dT4_dVg = 2 * dVdsatCV_dVg - dVdsCV_dVg;
			       dT4_dVd = - dVdsCV_dVd;
			       dT4_dVb = 2 * dVdsatCV_dVb - dVdsCV_dVb;
			       T0 = T3 * VcsCV;
			       T1 = T4 * VdsCV; 
			       Xc = T0 / T1;

			       dT0_dVb = VcsCV * (2 * dVdsatCV_dVb - dVcsCV_dVb) +
					 T3 * dVcsCV_dVb;
			       dT0_dVg = VcsCV * (2 * dVdsatCV_dVg - dVcsCV_dVg) +
					 T3 * dVcsCV_dVg;
			       dT0_dVd = 2 * dVcsCV_dVd * (VdsatCV - VcsCV); 
			       dT0_dVc = 2 * dVcsCV_dVc * (VdsatCV - VcsCV);

			       dT1_dVb = VdsCV * dT4_dVb + T4 * dVdsCV_dVb;
			       dT1_dVg = VdsCV * dT4_dVg + T4 * dVdsCV_dVg;
			       dT1_dVd = dVdsCV_dVd * T4 + VdsCV * dT4_dVd; 
			       T3 = T1 * T1;

			       dXc_dVb = (dT0_dVb - dT1_dVb * Xc) / T1;
			       dXc_dVg = (dT0_dVg - dT1_dVg * Xc) / T1;
			       dXc_dVd = (dT0_dVd - dT1_dVd * Xc) / T1;
			       dXc_dVc = dT0_dVc / T1;

			       T0 = AbulkCV * VcsCV;
			       dT0_dVb = dAbulkCV_dVb * VcsCV + dVcsCV_dVb * AbulkCV;
			       dT0_dVg = dVcsCV_dVg * AbulkCV;
			       dT0_dVd = AbulkCV * dVcsCV_dVd;
			       dT0_dVc = AbulkCV * dVcsCV_dVc;
	  
			       T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1e-20);
			       dT1_dVb = -6.0 * dT0_dVb;
			       dT1_dVg = 12.0 * (1.0 - 0.5 * dT0_dVg);
			       dT1_dVd = -6.0 * dT0_dVd;
			       dT1_dVc = -6.0 * dT0_dVc;

			       T2 = VcsCV / T1;
			       T4 = T1 * T1;
			       dT2_dVb = ( dVcsCV_dVb * T1 - dT1_dVb * VcsCV ) / T4;
			       dT2_dVg = ( dVcsCV_dVg * T1 - dT1_dVg * VcsCV ) / T4;
			       dT2_dVd = ( dVcsCV_dVd * T1 - dT1_dVd * VcsCV ) / T4;
			       dT2_dVc = ( dVcsCV_dVc * T1 - dT1_dVc * VcsCV ) / T4;

			       T3 = T0 * T2;
			       dT3_dVb = dT0_dVb * T2 + dT2_dVb * T0;
			       dT3_dVg = dT0_dVg * T2 + dT2_dVg * T0;
			       dT3_dVd = dT0_dVd * T2 + dT2_dVd * T0;
			       dT3_dVc = dT0_dVc * T2 + dT2_dVc * T0;

			       T4 = 1.0 - AbulkCV;
			       dT4_dVb = - dAbulkCV_dVb;

			       T5 = 0.5 * VcsCV - T3;
			       dT5_dVb = 0.5 * dVcsCV_dVb - dT3_dVb;
			       dT5_dVg = 0.5 * dVcsCV_dVg - dT3_dVg;
			       dT5_dVd = 0.5 * dVcsCV_dVd - dT3_dVd;
			       dT5_dVc = 0.5 * dVcsCV_dVc - dT3_dVc;

			       T6 = T4 * T5 * CoxWL;
			       T7 = CoxWL * Xc;

			       Qsubs1 = CoxWL * Xc * T4 * T5;
			       dQsubs1_dVb = T6 * dXc_dVb + T7 * ( T4*dT5_dVb + dT4_dVb*T5 );
			       dQsubs1_dVg = T6 * dXc_dVg + T7 * T4 * dT5_dVg;
			       dQsubs1_dVd = T6 * dXc_dVd + T7 * T4 * dT5_dVd;
			       dQsubs1_dVc = T6 * dXc_dVc + T7 * T4 * dT5_dVc;

                               Qsubs2 = -CoxWL * (1-Xc) * (AbulkCV - 1.0) * Vcs;

                               T2 = CoxWL * (AbulkCV - 1.0) * Vcs;
                               dQsubs2_dVb = T2 * dXc_dVb - CoxWL * (1-Xc) * Vcs * dAbulkCV_dVb;
                               dQsubs2_dVg = T2 * dXc_dVg;
                               dQsubs2_dVd = T2 * dXc_dVd;
                               dQsubs2_dVc = T2 * dXc_dVc - CoxWL * (1-Xc) * (AbulkCV - 1.0);

			       Qbf = Qac0 + Qsub0 + Qsubs1 + Qsubs2;
			       dQbf_dVrg = dQac0_dVrg + dQsub0_dVrg;
			       dQbf_dVg = dQsub0_dVg + dQsubs1_dVg + dQsubs2_dVg;
			       dQbf_dVd = dQac0_dVd + dQsub0_dVd + dQsubs1_dVd
					+ dQsubs2_dVd;
			       dQbf_dVb = dQac0_dVb + dQsub0_dVb + dQsubs1_dVb
					+ dQsubs2_dVb;
			       dQbf_dVc = dQsubs1_dVc + dQsubs2_dVc;
			       dQbf_dVe = 0.0;
			       if (selfheat)  dQbf_dT  = dQac0_dT + dQsub0_dT;
                               else  dQbf_dT = 0.0;
			 }  /* End of if (capMod == 2) */
			 else if (model->B3SOIDDcapMod == 3)
			 {   
			    /* Front gate strong inversion depletion charge */
			    /* VdssatCV calculation */
	   
			     T1 = Vgsteff + K1*sqrtPhis + 0.5*K1*K1;
			     T2 = Vgsteff + K1*sqrtPhis + Phis + 0.25*K1*K1;
	   
			     dT1_dVb = K1*dsqrtPhis_dVb;
			     dT2_dVb = dT1_dVb + dPhis_dVb;
			     dT1_dVg = dT2_dVg = 1;
	   
                             /* Note VdsatCV is redefined in capMod = 3 */
			     VdsatCV = T1 - K1*sqrt(T2);
	   
			     dVdsatCV_dVb = dT1_dVb - K1/2/sqrt(T2)*dT2_dVb;
			     dVdsatCV_dVg = dT1_dVg - K1/2/sqrt(T2)*dT2_dVg;
	   
			     T1 = VdsatCV - Vdsat;
			     dT1_dVg = dVdsatCV_dVg - dVdsat_dVg;
			     dT1_dVb = dVdsatCV_dVb - dVdsat_dVb;
			     dT1_dVd = - dVdsat_dVd;
			     dT1_dVc = - dVdsat_dVc;
			     dT1_dT  = - dVdsat_dT;
	   
			     if (!(T1 == 0.0)) 
			     {  T3 = -0.5 * Vdsat / T1;  /* Vdsmax */
				T2 = T3 * Vdsat; 
				T4 = T2 + T1 * T3 * T3;     /* fmax */
				if ((Vdseff > T2) && (T1 < 0)) 
				{  
				   VdsCV = T4;
				   T5 = -0.5 / (T1 * T1);
				   dT3_dVg = T5 * (T1 * dVdsat_dVg - Vdsat * dT1_dVg);
				   dT3_dVb = T5 * (T1 * dVdsat_dVb - Vdsat * dT1_dVb);
				   dT3_dVd = T5 * (T1 * dVdsat_dVd - Vdsat * dT1_dVd);
				   dT3_dVc = T5 * (T1 * dVdsat_dVc - Vdsat * dT1_dVc);
                                   if (selfheat)
				      dT3_dT=T5 * (T1 * dVdsat_dT  - Vdsat * dT1_dT);
                                   else  dT3_dT=0.0;
	   
				   dVdsCV_dVd = T3 * dVdsat_dVd + Vdsat * dT3_dVd
					      + T3 * (2 * T1 * dT3_dVd + T3 * dT1_dVd);
				   dVdsCV_dVg = T3 * dVdsat_dVg + Vdsat * dT3_dVg
					      + T3 * (2 * T1 * dT3_dVg + T3 * dT1_dVg);
				   dVdsCV_dVb = T3 * dVdsat_dVb + Vdsat * dT3_dVb
					      + T3 * (2 * T1 * dT3_dVb + T3 * dT1_dVb);
				   dVdsCV_dVc = T3 * dVdsat_dVc + Vdsat * dT3_dVc
					         + T3 * (2 * T1 * dT3_dVc + T3 * dT1_dVc);
                                   if (selfheat)
				      dVdsCV_dT  = T3 * dVdsat_dT  + Vdsat * dT3_dT 
				   	         + T3 * (2 * T1 * dT3_dT  + T3 * dT1_dT );
                                   else  dVdsCV_dT = 0.0;
				} else 
				{  T5 = Vdseff / Vdsat;
				   T6 = T5 * T5;
				   T7 = 2 * T1 * T5 / Vdsat;
				   T8 = T7 / Vdsat;
				   VdsCV = Vdseff + T1 * T6;
				   dVdsCV_dVd = dVdseff_dVd + T8 * 
					      ( Vdsat * dVdseff_dVd 
					      - Vdseff * dVdsat_dVd)
					      + T6 * dT1_dVd;
				   dVdsCV_dVb = dVdseff_dVb + T8 * ( Vdsat * 
						dVdseff_dVb - Vdseff * dVdsat_dVb)
						+ T6 * dT1_dVb;
				   dVdsCV_dVg = dVdseff_dVg + T8 * 
					      ( Vdsat * dVdseff_dVg 
					      - Vdseff * dVdsat_dVg)
					      + T6 * dT1_dVg;
				   dVdsCV_dVc = dVdseff_dVc + T8 *
					         ( Vdsat * dVdseff_dVc 
					         - Vdseff * dVdsat_dVc)
					         + T6 * dT1_dVc;
				   if (selfheat)
				      dVdsCV_dT  = dVdseff_dT  + T8 *
					         ( Vdsat * dVdseff_dT  
					         - Vdseff * dVdsat_dT )
					         + T6 * dT1_dT ;
				   else dVdsCV_dT  = 0.0;
				}
			     } else 
			     {  VdsCV = Vdseff;
				dVdsCV_dVb = dVdseff_dVb;
				dVdsCV_dVd = dVdseff_dVd;
				dVdsCV_dVg = dVdseff_dVg;
				dVdsCV_dVc = dVdseff_dVc;
				dVdsCV_dT  = dVdseff_dT;
			     }
			     if (VdsCV < 0.0) VdsCV = 0.0;
			     VdsCV += 1e-4;
	   
			     if (VdsCV > (VdsatCV - 1e-7))
			     {
				VdsCV = VdsatCV - 1e-7;
			     }
			     Phisd = Phis + VdsCV;
			     dPhisd_dVb = dPhis_dVb + dVdsCV_dVb;
			     dPhisd_dVd = dVdsCV_dVd;
			     dPhisd_dVg = dVdsCV_dVg;
			     dPhisd_dVc = dVdsCV_dVc;
			     dPhisd_dT  = dVdsCV_dT;
			     sqrtPhisd = sqrt(Phisd);
	    
			     /* Qdep0 - Depletion charge at Vgs=Vth */
			     T10 = CoxWL * K1;
			     Qdep0 = T10 * sqrtPhis;
			     dQdep0_dVb = T10 * dsqrtPhis_dVb; 

			       /* VcsCV calculation */ 
                               T1 = VdsCV - Vcs - VdsCV * VdsCV * DELTA_Vcscv;
                               T5 = 2 * DELTA_Vcscv;
                               T2 = sqrt(T1 * T1 + T5 * VdsCV * VdsCV);

                               dT1_dVb = dVdsCV_dVb * (1.0 - 2.0 * VdsCV * DELTA_Vcscv);
                               dT2_dVb = (T1 * dT1_dVb + T5 * VdsCV * dVdsCV_dVb)/T2;

                               dT1_dVd = dVdsCV_dVd * (1.0 - 2.0 * VdsCV * DELTA_Vcscv);
                               dT2_dVd = (T1 * dT1_dVd + T5 * VdsCV * dVdsCV_dVd)/ T2;

                               dT1_dVg = dVdsCV_dVg * (1.0 - 2.0 * VdsCV * DELTA_Vcscv) ;
                               dT2_dVg = (T1 * dT1_dVg + T5 * VdsCV * dVdsCV_dVg)/T2;

                               dT1_dVc = dVdsCV_dVc * (1.0 - 2.0 * VdsCV * DELTA_Vcscv) - 1;
                               dT2_dVc = (T1 * dT1_dVc + T5 * VdsCV * dVdsCV_dVc)/T2;

                               if (selfheat)
                               {
                                  dT1_dT  = dVdsCV_dT  * (1.0 - 2.0 * VdsCV * DELTA_Vcscv);
                                  dT2_dT  = (T1 * dT1_dT  + T5 * VdsCV * dVdsCV_dT )/ T2;
                               } else dT1_dT = dT2_dT = 0.0;

			       VcsCV = Vcs + 0.5*(T1 - T2);
	     
			       dVcsCV_dVb = 0.5 * (dT1_dVb - dT2_dVb);
			       dVcsCV_dVg = 0.5 * (dT1_dVg - dT2_dVg);
			       dVcsCV_dVd = 0.5 * (dT1_dVd - dT2_dVd);
			       dVcsCV_dVc = 1 + 0.5 * (dT1_dVc - dT2_dVc);
                               if (selfheat)
			          dVcsCV_dT  = 0.5 * (dT1_dT  - dT2_dT);
                               else  dVcsCV_dT = 0.0;

			       Phisc = Phis + VcsCV;
			       dPhisc_dVb = dPhis_dVb + dVcsCV_dVb;
			       dPhisc_dVd = dVcsCV_dVd;
			       dPhisc_dVg = dVcsCV_dVg;
			       dPhisc_dVc = dVcsCV_dVc;
			       dPhisc_dT  = dVcsCV_dT;
			       sqrtPhisc = sqrt(Phisc);
	     
			       /* Xc calculation */
			       T1   = Vgsteff + K1*sqrtPhis - 0.5*VdsCV;
			       T2   = CONST_2OV3*K1*(Phisd*sqrtPhisd - Phis*sqrtPhis);
			       T3   = Vgsteff + K1*sqrtPhis - 0.5*VcsCV;
			       T4   = CONST_2OV3*K1*(Phisc*sqrtPhisc - Phis*sqrtPhis);
			       T5   = T1*VdsCV - T2;
			       T6   = T3*VcsCV - T4;
			       Xc   = T6/T5;
	     
			       dT1_dVb = K1*dsqrtPhis_dVb - 0.5*dVdsCV_dVb;
			       dT2_dVb = K1*(sqrtPhisd*dPhisd_dVb - sqrtPhis*dPhis_dVb);
			       dT3_dVb = K1*dsqrtPhis_dVb - 0.5*dVcsCV_dVb;
			       dT4_dVb = K1*(sqrtPhisc*dPhisc_dVb - sqrtPhis*dPhis_dVb);
	   
			       dT1_dVd = - 0.5*dVdsCV_dVd;
			       dT2_dVd = K1 * (sqrtPhisd*dPhisd_dVd);
			       dT3_dVd = - 0.5*dVcsCV_dVd;
			       dT4_dVd = K1 * (sqrtPhisc*dPhisc_dVd);
	     
			       dT1_dVg = 1 - 0.5*dVdsCV_dVg;
			       dT2_dVg = K1 * (sqrtPhisd*dPhisd_dVg);
			       dT3_dVg = 1 - 0.5*dVcsCV_dVg;
			       dT4_dVg = K1 * (sqrtPhisc*dPhisc_dVg);
	     
			       dT1_dVc = - 0.5*dVdsCV_dVc;
			       dT2_dVc = K1 * (sqrtPhisd*dPhisd_dVc);
			       dT3_dVc = - 0.5*dVcsCV_dVc;
			       dT4_dVc = K1 * (sqrtPhisc*dPhisc_dVc);
	     
                               if (selfheat)
                               {
			          dT1_dT  = - 0.5*dVdsCV_dT;
			          dT2_dT  = K1 * (sqrtPhisd*dPhisd_dT);
			          dT3_dT  = - 0.5*dVcsCV_dT;
			          dT4_dT  = K1 * (sqrtPhisc*dPhisc_dT);
                               } else  dT1_dT = dT2_dT = dT3_dT = dT4_dT = 0.0;
	     
			       dT5_dVb = T1 * dVdsCV_dVb + VdsCV * dT1_dVb - dT2_dVb;
			       dT6_dVb = T3 * dVcsCV_dVb + VcsCV * dT3_dVb - dT4_dVb;
	     
			       dT5_dVd = T1 * dVdsCV_dVd + VdsCV * dT1_dVd - dT2_dVd;
			       dT6_dVd = T3 * dVcsCV_dVd + VcsCV * dT3_dVd - dT4_dVd;
	     
			       dT5_dVg = T1 * dVdsCV_dVg + VdsCV * dT1_dVg - dT2_dVg;
			       dT6_dVg = T3 * dVcsCV_dVg + VcsCV * dT3_dVg - dT4_dVg;
	     
			       dT5_dVc = T1 * dVdsCV_dVc + VdsCV * dT1_dVc - dT2_dVc;
			       dT6_dVc = T3 * dVcsCV_dVc + VcsCV * dT3_dVc - dT4_dVc;
	     
                               if (selfheat)
                               {
			          dT5_dT  = T1 * dVdsCV_dT  + VdsCV * dT1_dT  - dT2_dT;
			          dT6_dT  = T3 * dVcsCV_dT  + VcsCV * dT3_dT  - dT4_dT;
                               } else  dT5_dT = dT6_dT = 0.0;
	     
			       dXc_dVb  = (dT6_dVb - T6/T5 * dT5_dVb) / T5;
			       dXc_dVd  = (dT6_dVd - T6/T5 * dT5_dVd) / T5;
			       dXc_dVg  = (dT6_dVg - T6/T5 * dT5_dVg) / T5;
			       dXc_dVc  = (dT6_dVc - T6/T5 * dT5_dVc) / T5;
                               if (selfheat)
			          dXc_dT   = (dT6_dT  - T6/T5 * dT5_dT ) / T5;
                               else  dXc_dT = 0.0;
	     
			       T10 = Phis  * sqrtPhis ;
			       T5  = Phisc * sqrtPhisc;
			       T0    = T5 - T10;
			       T1    = Vgsteff + K1*sqrtPhis + Phis;
			       T2    = Phisc*T5 - Phis*T10;
			       T3    = K1*VcsCV*(Phis + 0.5*VcsCV);
	     
			       dT0_dVb = 1.5 *(sqrtPhisc*dPhisc_dVb-sqrtPhis*dPhis_dVb);
			       dT1_dVb = (0.5*K1/sqrtPhis + 1) * dPhis_dVb;
			       dT2_dVb = 2.5 * (T5 * dPhisc_dVb - T10 * dPhis_dVb);
			       dT3_dVb = K1 * ( VcsCV * (dPhis_dVb + 0.5 * dVcsCV_dVb)
				       + dVcsCV_dVb * (Phis + 0.5*VcsCV));
	     
			       dT0_dVd = 1.5 * sqrtPhisc * dPhisc_dVd;
			       dT1_dVd = 0;
			       dT2_dVd = 2.5 * T5 * dPhisc_dVd;
			       dT3_dVd = K1 * (Phis + VcsCV) * dVcsCV_dVd;

			       dT0_dVg = 1.5 * sqrtPhisc * dPhisc_dVg;
			       dT1_dVg = 1;
			       dT2_dVg = 2.5 * T5 * dPhisc_dVg;
			       dT3_dVg = K1 * (VcsCV * 0.5 * dVcsCV_dVg
				       + dVcsCV_dVg * (Phis + 0.5*VcsCV));
	  
			       dT0_dVc = 1.5 * sqrtPhisc * dPhisc_dVc;
			       dT1_dVc = 0.0;
			       dT2_dVc = 2.5 * T5 * dPhisc_dVc;
			       dT3_dVc = K1 * (VcsCV * 0.5 * dVcsCV_dVc
				       + dVcsCV_dVc * (Phis + 0.5*VcsCV));
	  
                               if (selfheat)
                               {
			          dT0_dT  = 1.5 * sqrtPhisc * dPhisc_dT;
			          dT1_dT  = 0;
			          dT2_dT  = 2.5 * T5 * dPhisc_dT;
			          dT3_dT  = K1 * (Phis + VcsCV) * dVcsCV_dT;
                               } else  dT0_dT = dT1_dT = dT2_dT = dT3_dT = 0.0;

			       Nomi  = K1*(CONST_2OV3*T1*T0 - 0.4*T2 - T3);
	   
			       dNomi_dVb = K1*(CONST_2OV3 * (T1 * dT0_dVb + T0*dT1_dVb)
					 - 0.4 * dT2_dVb - dT3_dVb);
			       dNomi_dVd = K1*(CONST_2OV3 * (T1 * dT0_dVd + T0*dT1_dVd)
					 - 0.4 * dT2_dVd - dT3_dVd);
			       dNomi_dVg = K1*(CONST_2OV3 * (T1 * dT0_dVg + T0*dT1_dVg)
					 - 0.4 * dT2_dVg - dT3_dVg);
			       dNomi_dVc = K1*(CONST_2OV3 * (T1 * dT0_dVc + T0*dT1_dVc)
					 - 0.4 * dT2_dVc - dT3_dVc);
                               if (selfheat)
			          dNomi_dT  = K1*(CONST_2OV3 * (T1 * dT0_dT  + T0*dT1_dT )
					    - 0.4 * dT2_dT  - dT3_dT );
                               else
			          dNomi_dT  = 0.0;
	    
			       T4    = Vgsteff + K1*sqrtPhis - 0.5*VdsCV;
			       T5    = CONST_2OV3*K1*(Phisd*sqrtPhisd - T10);
	     
			       dT4_dVb = K1 * dsqrtPhis_dVb - 0.5*dVdsCV_dVb;
			       dT5_dVb = K1*(sqrtPhisd*dPhisd_dVb - sqrtPhis*dPhis_dVb);
	     
			       dT4_dVd = - 0.5*dVdsCV_dVd;
			       dT5_dVd = K1*( sqrtPhisd * dPhisd_dVd);
	     
			       dT4_dVg = 1 - 0.5 * dVdsCV_dVg;
			       dT5_dVg = K1* sqrtPhisd * dPhisd_dVg;
	     
			       dT4_dVc = - 0.5 * dVdsCV_dVc;
			       dT5_dVc = K1* sqrtPhisd * dPhisd_dVc;
	   
                               if (selfheat)
                               {
			          dT4_dT  = - 0.5 * dVdsCV_dT;
			          dT5_dT  = K1* sqrtPhisd * dPhisd_dT;
                               } else  dT4_dT = dT5_dT = 0.0;
	   
			       Denomi = T4*VdsCV - T5;
	     
			       dDenomi_dVb = VdsCV*dT4_dVb + T4*dVdsCV_dVb - dT5_dVb;
			       dDenomi_dVd = VdsCV*dT4_dVd + T4*dVdsCV_dVd - dT5_dVd;
			       dDenomi_dVg = VdsCV*dT4_dVg + T4*dVdsCV_dVg - dT5_dVg;
			       dDenomi_dVc = VdsCV*dT4_dVc + T4*dVdsCV_dVc - dT5_dVc;
                               if (selfheat)
			          dDenomi_dT  = VdsCV*dT4_dT  + T4*dVdsCV_dT  - dT5_dT;
                               else  dDenomi_dT = 0.0;
	     
			       T6 = -CoxWL / Denomi;
			       Qsubs1 = T6 * Nomi;
			       dQsubs1_dVb = T6*(dNomi_dVb - Nomi / Denomi*dDenomi_dVb);
			       dQsubs1_dVg = T6*(dNomi_dVg - Nomi / Denomi*dDenomi_dVg);
			       dQsubs1_dVd = T6*(dNomi_dVd - Nomi / Denomi*dDenomi_dVd);
			       dQsubs1_dVc = T6*(dNomi_dVc - Nomi / Denomi*dDenomi_dVc);
                               if (selfheat)
			          dQsubs1_dT  = T6*(dNomi_dT  - Nomi / Denomi*dDenomi_dT );
                               else  dQsubs1_dT = 0.0;

			       T6 = sqrt(1e-4 + phi - Vbs0eff);
			       T7 = K1 * CoxWL;
			       T8 = 1 - Xc;
			       T10 = T7 * T6;
                               T11 = T7 * T8 * 0.5 / T6;
			       Qsubs2 = -T10 * T8 ;
			       dQsubs2_dVg = T10 * dXc_dVg;
			       dQsubs2_dVb = T10 * dXc_dVb;
			       dQsubs2_dVd = T10 * dXc_dVd + T11 * dVbs0eff_dVd;
			       dQsubs2_dVc = T10 * dXc_dVc;
			       dQsubs2_dVe = T11 * dVbs0eff_dVe;
                               dQsubs2_dVrg = T11 * dVbs0eff_dVg;
                               if (selfheat)
			          dQsubs2_dT  = T10 * dXc_dT + T11 * dVbs0eff_dT;
                               else dQsubs2_dT  = 0.0;

			       Qbf = Qac0 + Qsub0 + Qsubs1 + Qsubs2 + Qdep0;
			       dQbf_dVrg = dQac0_dVrg + dQsub0_dVrg + dQsubs2_dVrg;
			       dQbf_dVg = dQsub0_dVg + dQsubs1_dVg + dQsubs2_dVg ;
			       dQbf_dVd = dQac0_dVd + dQsub0_dVd + dQsubs1_dVd 
					+ dQsubs2_dVd;
			       dQbf_dVb = dQac0_dVb + dQsub0_dVb + dQsubs1_dVb 
					+ dQsubs2_dVb + dQdep0_dVb;
			       dQbf_dVc = dQsubs1_dVc + dQsubs2_dVc;
			       dQbf_dVe = dQsubs2_dVe;
                               if (selfheat)
			          dQbf_dT  = dQac0_dT + dQsub0_dT + dQsubs1_dT + dQsubs2_dT;
                               else  dQbf_dT = 0.0;
			  }    /* End of if capMod == 3 */

		            /* Something common in both capMod 2 or 3 */

		            /* Backgate charge */
		            CboxWL = pParam->B3SOIDDkb3 * Cbox * pParam->B3SOIDDweffCV 
                                   * pParam->B3SOIDDleffCV;
			    T0 = 0.5 * K1;
			    T2 = sqrt(phi - Vbs0t);
			    T3 = phi + K1 * T2 - Vbs0t;
			    T4 = sqrt(T0 * T0 + T3);
			    Qsicv = K1 * CoxWL * ( T0 - T4);
                            T6 = CoxWL * T0 / T4 * (1 + T0 / T2);
                            if (selfheat)  dQsicv_dT = T6 * dVbs0t_dT;
                            else  dQsicv_dT = 0.0;

			    T2 = sqrt(phi - Vbs0mos);
			    T3 = phi + K1 * T2 - Vbs0mos;
			    T4 = sqrt(T0 * T0 + T3);
			    Qbf0 = K1 * CoxWL * ( T0 - T4);
                            T6 = CoxWL * T0 / T4 * (1 + T0 / T2);
			    dQbf0_dVe = T6 * dVbs0mos_dVe;
                            if (selfheat)  dQbf0_dT  = T6 * dVbs0mos_dT;
                            else  dQbf0_dT = 0.0;

			    T5 = -CboxWL * (Vbsdio - Vbs0);
                            T6 = CboxWL * Xc;
			    Qe1 = -Qsicv + Qbf0 + T5 * Xc;
			    dQe1_dVg = T5 * (dXc_dVg * dVgsteff_dVg + dXc_dVb * dVbseff_dVg
                                             + dXc_dVc * dVcs_dVg) - T6 * dVbsdio_dVg;
			    dQe1_dVb = T5 * (dXc_dVg * dVgsteff_dVb + dXc_dVb * dVbseff_dVb
                                             + dXc_dVc * dVcs_dVb) - T6 * dVbsdio_dVb;
			    dQe1_dVd = T5 * (dXc_dVg * dVgsteff_dVd + dXc_dVb * dVbseff_dVd
                                             + dXc_dVc * dVcs_dVd + dXc_dVd) - T6 * dVbsdio_dVd;
			    dQe1_dVe = dQbf0_dVe + T6 * (dVbs0_dVe - dVbsdio_dVe);
                            if (selfheat)
                               dQe1_dT  = -dQsicv_dT + dQbf0_dT
                                        + T5 * (dXc_dVg * dVgsteff_dT  + dXc_dVb * dVbseff_dT
                                        + dXc_dVc  * dVcs_dT  + dXc_dT )
                                        + T6 * (dVbs0_dT - dVbsdio_dT);
                            else  dQe1_dT = 0.0;

			    T2 = -model->B3SOIDDcboxt * pParam->B3SOIDDweffCV
			       * pParam->B3SOIDDleffCV;
			    T3 = T2 * 0.5 * (1 - Xc);
			    T4 = T2 * 0.5 * (VdsCV - VcsCV);
			    Qe2 = T2 * 0.5 * (1 - Xc) * (VdsCV - VcsCV); 

                            /* T10 - dVgsteff, T11 - dVbseff, T12 - dVcs */
			    T10 = T3 * (dVdsCV_dVg - dVcsCV_dVg) - T4 * dXc_dVg;
			    T11 = T3 * (dVdsCV_dVb - dVcsCV_dVb) - T4 * dXc_dVb;
			    T12 = T3 * (dVdsCV_dVc - dVcsCV_dVc) - T4 * dXc_dVc;
                            dQe2_dVg = T10 * dVgsteff_dVg + T11 * dVbseff_dVg + T12 * dVcs_dVg;
                            dQe2_dVb = T10 * dVgsteff_dVb + T11 * dVbseff_dVb + T12 * dVcs_dVb;
                            dQe2_dVd = T10 * dVgsteff_dVd + T11 * dVbseff_dVd + T12 * dVcs_dVd
			               + T3 * (dVdsCV_dVd - dVcsCV_dVd) - T4 * dXc_dVd;
                            dQe2_dVe = T10 * dVgsteff_dVe + T11 * dVbseff_dVe + T12 * dVcs_dVe;
                            if (selfheat)
                               dQe2_dT  = T10 * dVgsteff_dT  + T11 * dVbseff_dT  + T12 * dVcs_dT
                                          + T3 * (dVdsCV_dT  - dVcsCV_dT ) - T4 * dXc_dT;
                            else dQe2_dT  = 0.0;

			  /* This transform all the dependency on Vgsteff, Vbseff, 
			     Vcs into real ones */
                          Cbg = dQbf_dVrg + dQbf_dVg * dVgsteff_dVg
                              + dQbf_dVb * dVbseff_dVg + dQbf_dVc * dVcs_dVg;
                          Cbb = dQbf_dVg * dVgsteff_dVb + dQbf_dVb * dVbseff_dVb
			      + dQbf_dVc * dVcs_dVb;
			  Cbd = dQbf_dVg * dVgsteff_dVd + dQbf_dVb * dVbseff_dVd
			      + dQbf_dVc * dVcs_dVd + dQbf_dVd;
			  Cbe = dQbf_dVg * dVgsteff_dVe + dQbf_dVb * dVbseff_dVe 
                              + dQbf_dVc * dVcs_dVe + dQbf_dVe;
                          if (selfheat)
                             CbT = dQbf_dVg * dVgsteff_dT  + dQbf_dVb * dVbseff_dT
                                 + dQbf_dVc * dVcs_dT  + dQbf_dT;
                          else CbT = 0.0;
 
			  Ce1g = dQe1_dVg;
			  Ce1b = dQe1_dVb;
			  Ce1d = dQe1_dVd;
			  Ce1e = dQe1_dVe;
                          Ce1T = dQe1_dT;

			  Ce2g = dQe2_dVg;
			  Ce2b = dQe2_dVb;
			  Ce2d = dQe2_dVd;
			  Ce2e = dQe2_dVe;
                          Ce2T = dQe2_dT;

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
			  if (model->B3SOIDDxpart > 0.5)
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
			  else if (model->B3SOIDDxpart < 0.5)
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
			     qsrc = - 0.5 * qinv;
			     Csg1 = - 0.5 * Cgg1;
			     Csb1 = - 0.5 * Cgb1;
			     Csd1 = - 0.5 * Cgd1;
			  }

			  Csg = Csg1 * dVgsteff_dVg + Csb1 * dVbseff_dVg;
			  Csd = Csd1 + Csg1 * dVgsteff_dVd + Csb1 * dVbseff_dVd;
   			  Csb = Csg1 * dVgsteff_dVb + Csb1 * dVbseff_dVb;
			  Cse = Csg1 * dVgsteff_dVe + Csb1 * dVbseff_dVe;
                          if (selfheat)
                             CsT = Csg1 * dVgsteff_dT  + Csb1 * dVbseff_dT;
                          else  CsT = 0.0;
 
                             T0 = QEX_FACT * K1 * CoxWL;
                             Qex = T0 * (Vbs - Vbsdio);
                             dQex_dVg = - T0 * dVbsdio_dVg;
                             dQex_dVb = T0 * (1 - dVbsdio_dVb);
                             dQex_dVd = - T0 * dVbsdio_dVd;
                             dQex_dVe = - T0 * dVbsdio_dVe;
                             if (selfheat)  dQex_dT  = - T0 * dVbsdio_dT;
                             else  dQex_dT = 0.0;

			  qgate = qinv - (Qbf + Qe2);
   			  qbody = (Qbf - Qe1 + Qex);
			  qsub = Qe1 + Qe2 - Qex;
			  qdrn = -(qinv + qsrc);

			  Cgg = (Cgg1 * dVgsteff_dVg + Cgb1 * dVbseff_dVg) - Cbg ;
			  Cgd = (Cgd1  + Cgg1 * dVgsteff_dVd + Cgb1 * dVbseff_dVd)-Cbd;
			  Cgb = (Cgb1 * dVbseff_dVb + Cgg1 * dVgsteff_dVb) - Cbb;
			  Cge = (Cgg1 * dVgsteff_dVe + Cgb1 * dVbseff_dVe) - Cbe;
                          if (selfheat)
                             CgT = (Cgg1 * dVgsteff_dT  + Cgb1 * dVbseff_dT ) - CbT;
                          else  CgT = 0.0;

			  here->B3SOIDDcggb = Cgg - Ce2g;
			  here->B3SOIDDcgsb = - (Cgg  + Cgd  + Cgb  + Cge)
					  + (Ce2g + Ce2d + Ce2b + Ce2e);
			  here->B3SOIDDcgdb = Cgd - Ce2d;
			  here->B3SOIDDcgeb = Cge - Ce2e;
                          here->B3SOIDDcgT = CgT - Ce2T;

			  here->B3SOIDDcbgb = Cbg - Ce1g + dQex_dVg;
			  here->B3SOIDDcbsb = -(Cbg  + Cbd  + Cbb  + Cbe)
					  + (Ce1g + Ce1d + Ce1b + Ce1e)
                                          - (dQex_dVg + dQex_dVd + dQex_dVb + dQex_dVe);
			  here->B3SOIDDcbdb = Cbd - Ce1d + dQex_dVd;
			  here->B3SOIDDcbeb = Cbe - Ce1e + dQex_dVe;
                          here->B3SOIDDcbT = CbT - Ce1T + dQex_dT;

			  here->B3SOIDDcegb = Ce1g + Ce2g - dQex_dVg;
			  here->B3SOIDDcesb = -(Ce1g + Ce1d + Ce1b + Ce1e)
					  -(Ce2g + Ce2d + Ce2b + Ce2e)
                                          +(dQex_dVg + dQex_dVd + dQex_dVb + dQex_dVe);
			  here->B3SOIDDcedb = Ce1d + Ce2d - dQex_dVd;
			  here->B3SOIDDceeb = Ce1e + Ce2e - dQex_dVe;
                          here->B3SOIDDceT = Ce1T + Ce2T - dQex_dT;
 
			  here->B3SOIDDcdgb = -(Cgg + Cbg + Csg);
			  here->B3SOIDDcddb = -(Cgd + Cbd + Csd);
			  here->B3SOIDDcdeb = -(Cge + Cbe + Cse);
                          here->B3SOIDDcdT = -(CgT + CbT + CsT);
			  here->B3SOIDDcdsb = (Cgg + Cgd + Cgb + Cge 
					  + Cbg + Cbd + Cbb + Cbe 
					  + Csg + Csd + Csb + Cse); 

		      } /* End of if capMod == 2 or capMod ==3 */
		  }

	finished: /* returning Values to Calling Routine */
		  /*
		   *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
		   */
		  if (ChargeComputationNeeded)
                  {
                      /* Intrinsic S/D junction charge */
		      PhiBSWG = model->B3SOIDDGatesidewallJctPotential;
		      MJSWG = model->B3SOIDDbodyJctGateSideGradingCoeff;
		      cjsbs = model->B3SOIDDunitLengthGateSidewallJctCap
				 *  pParam->B3SOIDDweff * model->B3SOIDDtsi / 1e-7;
		      if (Vbs < 0.0) 
		      {  arg = 1.0 - Vbs / PhiBSWG;
			 if (MJSWG == 0.5)
			    dT3_dVb = 1.0 / sqrt(arg);
			 else
			    dT3_dVb = exp(-MJSWG * log(arg));
			 T3 = (1.0 - arg * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);
		      }
		      else 
		      {  T3 = Vbs * ( 1 + 0.5 * MJSWG * Vbs / PhiBSWG);
			 dT3_dVb = 1 + MJSWG * Vbs / PhiBSWG;
		      }

                      qjs = cjsbs * T3 + model->B3SOIDDtt * Ibs1;
                      gcjsbs = cjsbs * dT3_dVb + model->B3SOIDDtt * dIbs1_dVb;
                      if (selfheat)  gcjsT = model->B3SOIDDtt * dIbs1_dT;
                      else  gcjsT = 0.0;

		      if (Vbd < 0.0)
		      {  arg = 1.0 - Vbd / PhiBSWG;
			 if (MJSWG == 0.5)
			    dT3_dVb = 1.0 / sqrt(arg);
			 else
			    dT3_dVb = exp(-MJSWG * log(arg));
			 T3 = (1.0 - arg * dT3_dVb) * PhiBSWG / (1.0 - MJSWG);
		      }
		      else 
		      {  T3 = Vbd * ( 1 + 0.5 * MJSWG * Vbd / PhiBSWG);
			 dT3_dVb = 1 + MJSWG * Vbd / PhiBSWG;
		      }
		      dT3_dVd = - dT3_dVb;
	  
                      qjd = cjsbs * T3 + model->B3SOIDDtt * Ibd1;
                      gcjdbs = cjsbs * dT3_dVb + model->B3SOIDDtt * dIbd1_dVb;
                      gcjdds = cjsbs * dT3_dVd + model->B3SOIDDtt * dIbd1_dVd;
                      if (selfheat) gcjdT = model->B3SOIDDtt * dIbd1_dT;
                      else  gcjdT = 0.0;

		      qdrn -= qjd;
		      qbody += (qjs + qjd);
		      qsrc = -(qgate + qbody + qdrn + qsub);

		      /* Update the conductance */
		      here->B3SOIDDcddb -= gcjdds;
                      here->B3SOIDDcdT -= gcjdT;
		      here->B3SOIDDcdsb += gcjdds + gcjdbs;

		      here->B3SOIDDcbdb += (gcjdds);
                      here->B3SOIDDcbT += (gcjdT + gcjsT);
		      here->B3SOIDDcbsb -= (gcjdds + gcjdbs + gcjsbs);

		      /* Extrinsic Bottom S/D to substrate charge */
		      T10 = -model->B3SOIDDtype * ves;
		      /* T10 is vse without type conversion */
		      if ( ((pParam->B3SOIDDnsub > 0) && (model->B3SOIDDtype > 0)) ||
		           ((pParam->B3SOIDDnsub < 0) && (model->B3SOIDDtype < 0)) )
		      {
		         if (T10 < pParam->B3SOIDDvsdfb)
		         {  here->B3SOIDDqse = here->B3SOIDDcsbox * (T10 - pParam->B3SOIDDvsdfb);
			    here->B3SOIDDgcse = here->B3SOIDDcsbox;
		         }
		         else if (T10 < pParam->B3SOIDDsdt1)
		         {  T0 = T10 - pParam->B3SOIDDvsdfb;
			    T1 = T0 * T0;
			    here->B3SOIDDqse = T0 * (here->B3SOIDDcsbox - 
                                             pParam->B3SOIDDst2 / 3 * T1) ;
			    here->B3SOIDDgcse = here->B3SOIDDcsbox - pParam->B3SOIDDst2 * T1;
		         }
		         else if (T10 < pParam->B3SOIDDvsdth)
		         {  T0 = T10 - pParam->B3SOIDDvsdth;
			    T1 = T0 * T0;
			    here->B3SOIDDqse = here->B3SOIDDcsmin * T10 + here->B3SOIDDst4 + 
                                             pParam->B3SOIDDst3 / 3 * T0 * T1;
   			 here->B3SOIDDgcse = here->B3SOIDDcsmin + pParam->B3SOIDDst3 * T1;
		         }
		         else 
		         {  here->B3SOIDDqse = here->B3SOIDDcsmin * T10 + here->B3SOIDDst4;
			    here->B3SOIDDgcse = here->B3SOIDDcsmin;
		         }
		      } else
		      {
		         if (T10 < pParam->B3SOIDDvsdth)
		         {  here->B3SOIDDqse = here->B3SOIDDcsmin * (T10 - pParam->B3SOIDDvsdth);
			    here->B3SOIDDgcse = here->B3SOIDDcsmin;
		         }
		         else if (T10 < pParam->B3SOIDDsdt1)
		         {  T0 = T10 - pParam->B3SOIDDvsdth;
   			    T1 = T0 * T0;
   			    here->B3SOIDDqse = T0 * (here->B3SOIDDcsmin - pParam->B3SOIDDst2 / 3 * T1) ;
			    here->B3SOIDDgcse = here->B3SOIDDcsmin - pParam->B3SOIDDst2 * T1;
		         }
		         else if (T10 < pParam->B3SOIDDvsdfb)
		         {  T0 = T10 - pParam->B3SOIDDvsdfb;
			    T1 = T0 * T0;
			    here->B3SOIDDqse = here->B3SOIDDcsbox * T10 + here->B3SOIDDst4 + 
                                             pParam->B3SOIDDst3 / 3 * T0 * T1;
			    here->B3SOIDDgcse = here->B3SOIDDcsbox + pParam->B3SOIDDst3 * T1;
		         }
		         else 
		         {  here->B3SOIDDqse = here->B3SOIDDcsbox * T10 + here->B3SOIDDst4;
			    here->B3SOIDDgcse = here->B3SOIDDcsbox;
		         }
		      }

		      /* T11 is vde without type conversion */
		      T11 = model->B3SOIDDtype * (vds - ves);
		      if ( ((pParam->B3SOIDDnsub > 0) && (model->B3SOIDDtype > 0)) ||
		           ((pParam->B3SOIDDnsub < 0) && (model->B3SOIDDtype < 0)) )
		      {
		         if (T11 < pParam->B3SOIDDvsdfb)
		         {  here->B3SOIDDqde = here->B3SOIDDcdbox * (T11 - pParam->B3SOIDDvsdfb);
			    here->B3SOIDDgcde = here->B3SOIDDcdbox;
		         }
		         else if (T11 < pParam->B3SOIDDsdt1)
		         {  T0 = T11 - pParam->B3SOIDDvsdfb;
   			    T1 = T0 * T0;
   			    here->B3SOIDDqde = T0 * (here->B3SOIDDcdbox - pParam->B3SOIDDdt2 / 3 * T1) ;
			    here->B3SOIDDgcde = here->B3SOIDDcdbox - pParam->B3SOIDDdt2 * T1;
		         }
		         else if (T11 < pParam->B3SOIDDvsdth)
		         {  T0 = T11 - pParam->B3SOIDDvsdth;
			    T1 = T0 * T0;
			    here->B3SOIDDqde = here->B3SOIDDcdmin * T11 + here->B3SOIDDdt4 + 
                                             pParam->B3SOIDDdt3 / 3 * T0 * T1;
			    here->B3SOIDDgcde = here->B3SOIDDcdmin + pParam->B3SOIDDdt3 * T1;
		         }
		         else 
		         {  here->B3SOIDDqde = here->B3SOIDDcdmin * T11 + here->B3SOIDDdt4;
			    here->B3SOIDDgcde = here->B3SOIDDcdmin;
		         }
		      } else
		      {
		         if (T11 < pParam->B3SOIDDvsdth)
		         {  here->B3SOIDDqde = here->B3SOIDDcdmin * (T11 - pParam->B3SOIDDvsdth);
			    here->B3SOIDDgcde = here->B3SOIDDcdmin;
		         }
		         else if (T11 < pParam->B3SOIDDsdt1)
		         {  T0 = T11 - pParam->B3SOIDDvsdth;
   			    T1 = T0 * T0;
   			    here->B3SOIDDqde = T0 * (here->B3SOIDDcdmin - pParam->B3SOIDDdt2 / 3 * T1) ;
			    here->B3SOIDDgcde = here->B3SOIDDcdmin - pParam->B3SOIDDdt2 * T1;
		         }
		         else if (T11 < pParam->B3SOIDDvsdfb)
		         {  T0 = T11 - pParam->B3SOIDDvsdfb;
			    T1 = T0 * T0;
			    here->B3SOIDDqde = here->B3SOIDDcdbox * T11 + here->B3SOIDDdt4 + 
                                             pParam->B3SOIDDdt3 / 3 * T0 * T1;
			    here->B3SOIDDgcde = here->B3SOIDDcdbox + pParam->B3SOIDDdt3 * T1;
		         }
		         else 
		         {  here->B3SOIDDqde = here->B3SOIDDcdbox * T11 + here->B3SOIDDdt4;
			    here->B3SOIDDgcde = here->B3SOIDDcdbox;
		         }
		      } 

		      /* Extrinsic : Sidewall fringing S/D charge */
		      here->B3SOIDDqse += pParam->B3SOIDDcsesw * T10;
		      here->B3SOIDDgcse += pParam->B3SOIDDcsesw;
		      here->B3SOIDDqde += pParam->B3SOIDDcdesw * T11;
		      here->B3SOIDDgcde += pParam->B3SOIDDcdesw;

		      /* All charge are mutliplied with type at the end, but qse and qde
			 have true polarity => so pre-mutliplied with type */
		      here->B3SOIDDqse *= model->B3SOIDDtype;
		      here->B3SOIDDqde *= model->B3SOIDDtype;
		  }

		  here->B3SOIDDxc = Xc;
		  here->B3SOIDDcbb = Cbb;
		  here->B3SOIDDcbd = Cbd;
		  here->B3SOIDDcbg = Cbg;
		  here->B3SOIDDqbf = Qbf;
		  here->B3SOIDDqjs = qjs;
		  here->B3SOIDDqjd = qjd;

                  if (here->B3SOIDDdebugMod == -1)
                     ChargeComputationNeeded = 0;

		  /*
		   *  check convergence
		   */
		  if ((here->B3SOIDDoff == 0) || (!(ckt->CKTmode & MODEINITFIX)))
		  {   if (Check == 1)
		      {   ckt->CKTnoncon++;
if (here->B3SOIDDdebugMod > 2)
   fprintf(fpdebug, "Check is on, noncon=%d\n", ckt->CKTnoncon++);
		      }
		  }

                  *(ckt->CKTstate0 + here->B3SOIDDvg) = vg;
                  *(ckt->CKTstate0 + here->B3SOIDDvd) = vd;
                  *(ckt->CKTstate0 + here->B3SOIDDvs) = vs;
                  *(ckt->CKTstate0 + here->B3SOIDDvp) = vp;
                  *(ckt->CKTstate0 + here->B3SOIDDve) = ve;

		  *(ckt->CKTstate0 + here->B3SOIDDvbs) = vbs;
		  *(ckt->CKTstate0 + here->B3SOIDDvbd) = vbd;
		  *(ckt->CKTstate0 + here->B3SOIDDvgs) = vgs;
		  *(ckt->CKTstate0 + here->B3SOIDDvds) = vds;
		  *(ckt->CKTstate0 + here->B3SOIDDves) = ves;
		  *(ckt->CKTstate0 + here->B3SOIDDvps) = vps;
		  *(ckt->CKTstate0 + here->B3SOIDDdeltemp) = delTemp;

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

		  T3 = pParam->B3SOIDDweffCV * pParam->B3SOIDDcgdl;
		  T4 = sqrt(1.0 - 4.0 * T2 / pParam->B3SOIDDckappa);
		  cgdo = pParam->B3SOIDDcgdo + T3 - T3 * (1.0 - 1.0 / T4)
			 * (0.5 - 0.5 * T0 / T1);
		  qgdo = (pParam->B3SOIDDcgdo + T3) * vgd - T3 * (T2
			 + 0.5 * pParam->B3SOIDDckappa * (T4 - 1.0));

		  T0 = vgs + DELTA_1;
		  T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
		  T2 = 0.5 * (T0 - T1);
		  T3 = pParam->B3SOIDDweffCV * pParam->B3SOIDDcgsl;
		  T4 = sqrt(1.0 - 4.0 * T2 / pParam->B3SOIDDckappa);
		  cgso = pParam->B3SOIDDcgso + T3 - T3 * (1.0 - 1.0 / T4)
			 * (0.5 - 0.5 * T0 / T1);
		  qgso = (pParam->B3SOIDDcgso + T3) * vgs - T3 * (T2
			 + 0.5 * pParam->B3SOIDDckappa * (T4 - 1.0));


		  if (here->B3SOIDDmode > 0)
		  {   gcdgb = (here->B3SOIDDcdgb - cgdo) * ag0;
		      gcddb = (here->B3SOIDDcddb + cgdo + here->B3SOIDDgcde) * ag0;
		      gcdsb = here->B3SOIDDcdsb * ag0;
		      gcdeb = (here->B3SOIDDcdeb - here->B3SOIDDgcde) * ag0;
                      gcdT = model->B3SOIDDtype * here->B3SOIDDcdT * ag0;

		      gcsgb = -(here->B3SOIDDcggb + here->B3SOIDDcbgb + here->B3SOIDDcdgb
			    + here->B3SOIDDcegb + cgso) * ag0;
		      gcsdb = -(here->B3SOIDDcgdb + here->B3SOIDDcbdb + here->B3SOIDDcddb
			    + here->B3SOIDDcedb) * ag0;
		      gcssb = (cgso + here->B3SOIDDgcse - (here->B3SOIDDcgsb + here->B3SOIDDcbsb
			    + here->B3SOIDDcdsb + here->B3SOIDDcesb)) * ag0;
		      gcseb = -(here->B3SOIDDgcse + here->B3SOIDDcgeb + here->B3SOIDDcbeb + here->B3SOIDDcdeb
			    + here->B3SOIDDceeb) * ag0;
                      gcsT = - model->B3SOIDDtype * (here->B3SOIDDcgT + here->B3SOIDDcbT + here->B3SOIDDcdT
                            + here->B3SOIDDceT) * ag0;

		      gcggb = (here->B3SOIDDcggb + cgdo + cgso + pParam->B3SOIDDcgeo) * ag0;
		      gcgdb = (here->B3SOIDDcgdb - cgdo) * ag0;
		      gcgsb = (here->B3SOIDDcgsb - cgso) * ag0;
		      gcgeb = (here->B3SOIDDcgeb - pParam->B3SOIDDcgeo) * ag0;
                      gcgT = model->B3SOIDDtype * here->B3SOIDDcgT * ag0;

		      gcbgb = here->B3SOIDDcbgb * ag0;
		      gcbdb = here->B3SOIDDcbdb * ag0;
		      gcbsb = here->B3SOIDDcbsb * ag0;
   		      gcbeb = here->B3SOIDDcbeb * ag0;
                      gcbT = model->B3SOIDDtype * here->B3SOIDDcbT * ag0;

		      gcegb = (here->B3SOIDDcegb - pParam->B3SOIDDcgeo) * ag0;
		      gcedb = (here->B3SOIDDcedb - here->B3SOIDDgcde) * ag0;
		      gcesb = (here->B3SOIDDcesb - here->B3SOIDDgcse) * ag0;
		      gceeb = (here->B3SOIDDgcse + here->B3SOIDDgcde +
                               here->B3SOIDDceeb + pParam->B3SOIDDcgeo) * ag0;
 
                      gceT = model->B3SOIDDtype * here->B3SOIDDceT * ag0;

                      gcTt = pParam->B3SOIDDcth * ag0;

		      sxpart = 0.6;
		      dxpart = 0.4;

		      /* Lump the overlap capacitance and S/D parasitics */
		      qgd = qgdo;
		      qgs = qgso;
		      qge = pParam->B3SOIDDcgeo * vge;
		      qgate += qgd + qgs + qge;
		      qdrn += here->B3SOIDDqde - qgd;
		      qsub -= qge + here->B3SOIDDqse + here->B3SOIDDqde; 
		      qsrc = -(qgate + qbody + qdrn + qsub);
		  }
		  else
		  {   gcsgb = (here->B3SOIDDcdgb - cgso) * ag0;
		      gcssb = (here->B3SOIDDcddb + cgso + here->B3SOIDDgcse) * ag0;
		      gcsdb = here->B3SOIDDcdsb * ag0;
		      gcseb = (here->B3SOIDDcdeb - here->B3SOIDDgcse) * ag0;
                      gcsT = model->B3SOIDDtype * here->B3SOIDDcdT * ag0;

		      gcdgb = -(here->B3SOIDDcggb + here->B3SOIDDcbgb + here->B3SOIDDcdgb
			    + here->B3SOIDDcegb + cgdo) * ag0;
		      gcdsb = -(here->B3SOIDDcgdb + here->B3SOIDDcbdb + here->B3SOIDDcddb
			    + here->B3SOIDDcedb) * ag0;
		      gcddb = (cgdo + here->B3SOIDDgcde - (here->B3SOIDDcgsb + here->B3SOIDDcbsb 
			    + here->B3SOIDDcdsb + here->B3SOIDDcesb)) * ag0;
		      gcdeb = -(here->B3SOIDDgcde + here->B3SOIDDcgeb + here->B3SOIDDcbeb + here->B3SOIDDcdeb
			    + here->B3SOIDDceeb) * ag0;
                      gcdT = - model->B3SOIDDtype * (here->B3SOIDDcgT + here->B3SOIDDcbT 
                             + here->B3SOIDDcdT + here->B3SOIDDceT) * ag0;

		      gcggb = (here->B3SOIDDcggb + cgdo + cgso + pParam->B3SOIDDcgeo) * ag0;
		      gcgsb = (here->B3SOIDDcgdb - cgso) * ag0;
		      gcgdb = (here->B3SOIDDcgsb - cgdo) * ag0;
		      gcgeb = (here->B3SOIDDcgeb - pParam->B3SOIDDcgeo) * ag0;
                      gcgT = model->B3SOIDDtype * here->B3SOIDDcgT * ag0;

		      gcbgb = here->B3SOIDDcbgb * ag0;
		      gcbsb = here->B3SOIDDcbdb * ag0;
		      gcbdb = here->B3SOIDDcbsb * ag0;
		      gcbeb = here->B3SOIDDcbeb * ag0;
                      gcbT = model->B3SOIDDtype * here->B3SOIDDcbT * ag0;

		      gcegb = (here->B3SOIDDcegb - pParam->B3SOIDDcgeo) * ag0;
		      gcesb = (here->B3SOIDDcedb - here->B3SOIDDgcse) * ag0;
		      gcedb = (here->B3SOIDDcesb - here->B3SOIDDgcde) * ag0;
		      gceeb = (here->B3SOIDDceeb + pParam->B3SOIDDcgeo +
			       here->B3SOIDDgcse + here->B3SOIDDgcde) * ag0;
                      gceT = model->B3SOIDDtype * here->B3SOIDDceT * ag0;
		     
                      gcTt = pParam->B3SOIDDcth * ag0;

		      dxpart = 0.6;
		      sxpart = 0.4;

		      /* Lump the overlap capacitance */
		      qgd = qgdo;
		      qgs = qgso;
		      qge = pParam->B3SOIDDcgeo * vge;
		      qgate += qgd + qgs + qge;
		      qsrc = qdrn - qgs + here->B3SOIDDqse;
		      qsub -= qge + here->B3SOIDDqse + here->B3SOIDDqde; 
		      qdrn = -(qgate + qbody + qsrc + qsub);
		  }

		  here->B3SOIDDcgdo = cgdo;
		  here->B3SOIDDcgso = cgso;

                  if (ByPass) goto line860;

		  *(ckt->CKTstate0 + here->B3SOIDDqe) = qsub;
		  *(ckt->CKTstate0 + here->B3SOIDDqg) = qgate;
		  *(ckt->CKTstate0 + here->B3SOIDDqd) = qdrn;
                  *(ckt->CKTstate0 + here->B3SOIDDqb) = qbody;
		  if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0.0)) 
                     *(ckt->CKTstate0 + here->B3SOIDDqth) = pParam->B3SOIDDcth * delTemp;


		  /* store small signal parameters */
		  if (ckt->CKTmode & MODEINITSMSIG)
		  {   goto line1000;
		  }
		  if (!ChargeComputationNeeded)
		      goto line850;
	       

		  if (ckt->CKTmode & MODEINITTRAN)
		  {   *(ckt->CKTstate1 + here->B3SOIDDqb) =
			    *(ckt->CKTstate0 + here->B3SOIDDqb);
		      *(ckt->CKTstate1 + here->B3SOIDDqg) =
			    *(ckt->CKTstate0 + here->B3SOIDDqg);
		      *(ckt->CKTstate1 + here->B3SOIDDqd) =
			    *(ckt->CKTstate0 + here->B3SOIDDqd);
		      *(ckt->CKTstate1 + here->B3SOIDDqe) =
			    *(ckt->CKTstate0 + here->B3SOIDDqe);
                      *(ckt->CKTstate1 + here->B3SOIDDqth) =
                            *(ckt->CKTstate0 + here->B3SOIDDqth);
		  }
	       
                  error = NIintegrate(ckt, &geq, &ceq,0.0,here->B3SOIDDqb);
                  if (error) return(error);
                  error = NIintegrate(ckt, &geq, &ceq, 0.0, here->B3SOIDDqg);
                  if (error) return(error);
                  error = NIintegrate(ckt,&geq, &ceq, 0.0, here->B3SOIDDqd);
                  if (error) return(error);
                  error = NIintegrate(ckt,&geq, &ceq, 0.0, here->B3SOIDDqe);
                  if (error) return(error);
		  if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0.0)) 
                  {
                      error = NIintegrate(ckt, &geq, &ceq, 0.0, here->B3SOIDDqth);
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

		  sxpart = (1.0 - (dxpart = (here->B3SOIDDmode > 0) ? 0.4 : 0.6));

		  goto line900;
		    
	line860:
		  /* evaluate equivalent charge current */

                  cqgate = *(ckt->CKTstate0 + here->B3SOIDDcqg);
                  cqbody = *(ckt->CKTstate0 + here->B3SOIDDcqb);
                  cqdrn = *(ckt->CKTstate0 + here->B3SOIDDcqd);
                  cqsub = *(ckt->CKTstate0 + here->B3SOIDDcqe);
                  cqtemp = *(ckt->CKTstate0 + here->B3SOIDDcqth);

                  here->B3SOIDDcb += cqbody;
                  here->B3SOIDDcd += cqdrn;

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
		  {   *(ckt->CKTstate1 + here->B3SOIDDcqe) =  
			    *(ckt->CKTstate0 + here->B3SOIDDcqe);
		      *(ckt->CKTstate1 + here->B3SOIDDcqb) =  
			    *(ckt->CKTstate0 + here->B3SOIDDcqb);
		      *(ckt->CKTstate1 + here->B3SOIDDcqg) =  
			    *(ckt->CKTstate0 + here->B3SOIDDcqg);
		      *(ckt->CKTstate1 + here->B3SOIDDcqd) =  
			    *(ckt->CKTstate0 + here->B3SOIDDcqd);
		      *(ckt->CKTstate1 + here->B3SOIDDcqth) =  
			    *(ckt->CKTstate0 + here->B3SOIDDcqth);
		  }

		  /*
		   *  load current vector
		   */
	line900:

                  m = here->B3SOIDDm;

		  if (here->B3SOIDDmode >= 0)
		  {   Gm = here->B3SOIDDgm;
		      Gmbs = here->B3SOIDDgmbs;
		      Gme = here->B3SOIDDgme;
		      GmT = model->B3SOIDDtype * here->B3SOIDDgmT;
		      FwdSum = Gm + Gmbs + Gme;
		      RevSum = 0.0;
		      cdreq = model->B3SOIDDtype * (here->B3SOIDDcdrain - here->B3SOIDDgds * vds
			    - Gm * vgs - Gmbs * vbs - Gme * ves - GmT * delTemp);
		      /* ceqbs now is compatible with cdreq, ie. going in is +ve */
		      /* Equivalent current source from the diode */
		      ceqbs = here->B3SOIDDcjs;
		      ceqbd = here->B3SOIDDcjd;
		      /* Current going in is +ve */
		      ceqbody = -here->B3SOIDDcbody;
		      ceqth = here->B3SOIDDcth;
		      ceqbodcon = here->B3SOIDDcbodcon;

		      gbbg  = -here->B3SOIDDgbgs;
		      gbbdp = -here->B3SOIDDgbds;
		      gbbb  = -here->B3SOIDDgbbs;
		      gbbe  = -here->B3SOIDDgbes;
		      gbbp  = -here->B3SOIDDgbps;
		      gbbT  = -model->B3SOIDDtype * here->B3SOIDDgbT;
		      gbbsp = - ( gbbg + gbbdp + gbbb + gbbe + gbbp);

		      gddpg  = -here->B3SOIDDgjdg;
		      gddpdp = -here->B3SOIDDgjdd;
		      gddpb  = -here->B3SOIDDgjdb;
		      gddpe  = -here->B3SOIDDgjde;
		      gddpT  = -model->B3SOIDDtype * here->B3SOIDDgjdT;
		      gddpsp = - ( gddpg + gddpdp + gddpb + gddpe);

		      gsspg  = -here->B3SOIDDgjsg;
		      gsspdp = -here->B3SOIDDgjsd;
		      gsspb  = -here->B3SOIDDgjsb;
		      gsspe  = 0.0;
		      gsspT  = -model->B3SOIDDtype * here->B3SOIDDgjsT;
		      gsspsp = - (gsspg + gsspdp + gsspb + gsspe);

		      gppg = -here->B3SOIDDgbpgs;
		      gppdp = -here->B3SOIDDgbpds;
		      gppb = -here->B3SOIDDgbpbs;
		      gppe = -here->B3SOIDDgbpes;
		      gppp = -here->B3SOIDDgbpps;
		      gppT = -model->B3SOIDDtype * here->B3SOIDDgbpT;
		      gppsp = - (gppg + gppdp + gppb + gppe + gppp);

                      gTtg  = here->B3SOIDDgtempg;
                      gTtb  = here->B3SOIDDgtempb;
                      gTte  = here->B3SOIDDgtempe;
                      gTtdp = here->B3SOIDDgtempd;
                      gTtt  = here->B3SOIDDgtempT;
                      gTtsp = - (gTtg + gTtb + gTte + gTtdp);
		  }
		  else
		  {   Gm = -here->B3SOIDDgm;
		      Gmbs = -here->B3SOIDDgmbs;
		      Gme = -here->B3SOIDDgme;
		      GmT = -model->B3SOIDDtype * here->B3SOIDDgmT;
		      FwdSum = 0.0;
		      RevSum = -(Gm + Gmbs + Gme);
		      cdreq = -model->B3SOIDDtype * (here->B3SOIDDcdrain + here->B3SOIDDgds*vds
			    + Gm * vgd + Gmbs * vbd + Gme * (ves - vds) + GmT * delTemp);
		      ceqbs = here->B3SOIDDcjd;
		      ceqbd = here->B3SOIDDcjs;
		      /* Current going in is +ve */
		      ceqbody = -here->B3SOIDDcbody;
		      ceqth = here->B3SOIDDcth;
		      ceqbodcon = here->B3SOIDDcbodcon;

		      gbbg  = -here->B3SOIDDgbgs;
		      gbbb  = -here->B3SOIDDgbbs;
		      gbbe  = -here->B3SOIDDgbes;
		      gbbp  = -here->B3SOIDDgbps;
		      gbbsp = -here->B3SOIDDgbds;
		      gbbT  = -model->B3SOIDDtype * here->B3SOIDDgbT;
		      gbbdp = - ( gbbg + gbbsp + gbbb + gbbe + gbbp);

		      gddpg  = -here->B3SOIDDgjsg;
		      gddpsp = -here->B3SOIDDgjsd;
		      gddpb  = -here->B3SOIDDgjsb;
		      gddpe  = 0.0;
		      gddpT  = -model->B3SOIDDtype * here->B3SOIDDgjsT;
		      gddpdp = - (gddpg + gddpsp + gddpb + gddpe);

		      gsspg  = -here->B3SOIDDgjdg;
		      gsspsp = -here->B3SOIDDgjdd;
		      gsspb  = -here->B3SOIDDgjdb;
		      gsspe  = -here->B3SOIDDgjde;
		      gsspT  = -model->B3SOIDDtype * here->B3SOIDDgjdT;
		      gsspdp = - ( gsspg + gsspsp + gsspb + gsspe);

		      gppg = -here->B3SOIDDgbpgs;
		      gppsp = -here->B3SOIDDgbpds;
		      gppb = -here->B3SOIDDgbpbs;
		      gppe = -here->B3SOIDDgbpes;
		      gppp = -here->B3SOIDDgbpps;
		      gppT = -model->B3SOIDDtype * here->B3SOIDDgbpT;
		      gppdp = - (gppg + gppsp + gppb + gppe + gppp);

                      gTtg  = here->B3SOIDDgtempg;
                      gTtb  = here->B3SOIDDgtempb;
                      gTte  = here->B3SOIDDgtempe;
                      gTtsp = here->B3SOIDDgtempd;
                      gTtt  = here->B3SOIDDgtempT;
                      gTtdp = - (gTtg + gTtb + gTte + gTtsp);
		  }

		   if (model->B3SOIDDtype < 0)
		   {   
		       ceqbodcon = -ceqbodcon;
		       ceqbody = -ceqbody;
		       ceqbs = -ceqbs;
		       ceqbd = -ceqbd;
		       ceqqg = -ceqqg;
		       ceqqb = -ceqqb;
		       ceqqd = -ceqqd;
		       ceqqe = -ceqqe;
		   }

                   (*(ckt->CKTrhs + here->B3SOIDDbNode) -= m * (ceqbody+ceqqb));

		   (*(ckt->CKTrhs + here->B3SOIDDgNode) -= m * ceqqg);
		   (*(ckt->CKTrhs + here->B3SOIDDdNodePrime) += m * (ceqbd - cdreq - ceqqd));
		   (*(ckt->CKTrhs + here->B3SOIDDsNodePrime) += m * ((cdreq + ceqbs + ceqqg
							  + ceqqb + ceqqd + ceqqe)));
		   (*(ckt->CKTrhs + here->B3SOIDDeNode) -= m * ceqqe);

                   if (here->B3SOIDDbodyMod == 1) {
		       (*(ckt->CKTrhs + here->B3SOIDDpNode) += m * ceqbodcon);
                   }

		   if (selfheat) {
		       (*(ckt->CKTrhs + here->B3SOIDDtempNode) -= m * (ceqth + ceqqth));
                   }



 if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
		   {
	              *(ckt->CKTrhs + here->B3SOIDDvbsNode) = here->B3SOIDDvbsdio;
		      *(ckt->CKTrhs + here->B3SOIDDidsNode) = here->B3SOIDDids;
		      *(ckt->CKTrhs + here->B3SOIDDicNode) = here->B3SOIDDic;
		      *(ckt->CKTrhs + here->B3SOIDDibsNode) = here->B3SOIDDibs;
		      *(ckt->CKTrhs + here->B3SOIDDibdNode) = here->B3SOIDDibd;
		      *(ckt->CKTrhs + here->B3SOIDDiiiNode) = here->B3SOIDDiii; 
		      *(ckt->CKTrhs + here->B3SOIDDigidlNode) = here->B3SOIDDigidl;
		      *(ckt->CKTrhs + here->B3SOIDDitunNode) = here->B3SOIDDitun;
		      *(ckt->CKTrhs + here->B3SOIDDibpNode) = here->B3SOIDDibp;
		      *(ckt->CKTrhs + here->B3SOIDDabeffNode) = here->B3SOIDDabeff;
		      *(ckt->CKTrhs + here->B3SOIDDvbs0effNode) = here->B3SOIDDvbs0eff;
		      *(ckt->CKTrhs + here->B3SOIDDvbseffNode) = here->B3SOIDDvbseff;
		      *(ckt->CKTrhs + here->B3SOIDDxcNode) = here->B3SOIDDxc;
		      *(ckt->CKTrhs + here->B3SOIDDcbbNode) = here->B3SOIDDcbb;
		      *(ckt->CKTrhs + here->B3SOIDDcbdNode) = here->B3SOIDDcbd;
		      *(ckt->CKTrhs + here->B3SOIDDcbgNode) = here->B3SOIDDcbg;
		      *(ckt->CKTrhs + here->B3SOIDDqbfNode) = here->B3SOIDDqbf;
		      *(ckt->CKTrhs + here->B3SOIDDqjsNode) = here->B3SOIDDqjs;
		      *(ckt->CKTrhs + here->B3SOIDDqjdNode) = here->B3SOIDDqjd;

                      /* clean up last */
		      *(ckt->CKTrhs + here->B3SOIDDgmNode) = Gm;
		      *(ckt->CKTrhs + here->B3SOIDDgmbsNode) = Gmbs;
		      *(ckt->CKTrhs + here->B3SOIDDgdsNode) = Gds;
		      *(ckt->CKTrhs + here->B3SOIDDgmeNode) = Gme;
		      *(ckt->CKTrhs + here->B3SOIDDqdNode) = qdrn;
		      *(ckt->CKTrhs + here->B3SOIDDcbeNode) = Cbe;
		      *(ckt->CKTrhs + here->B3SOIDDvbs0teffNode) = Vbs0teff;
		      *(ckt->CKTrhs + here->B3SOIDDvthNode) = here->B3SOIDDvon;
		      *(ckt->CKTrhs + here->B3SOIDDvgsteffNode) = Vgsteff;
		      *(ckt->CKTrhs + here->B3SOIDDxcsatNode) = Xcsat;
		      *(ckt->CKTrhs + here->B3SOIDDqaccNode) = -Qac0;
		      *(ckt->CKTrhs + here->B3SOIDDqsub0Node) = Qsub0;
		      *(ckt->CKTrhs + here->B3SOIDDqsubs1Node) = Qsubs1;
		      *(ckt->CKTrhs + here->B3SOIDDqsubs2Node) = Qsubs2;
		      *(ckt->CKTrhs + here->B3SOIDDvdscvNode) = VdsCV;
		      *(ckt->CKTrhs + here->B3SOIDDvcscvNode) = VcsCV;
		      *(ckt->CKTrhs + here->B3SOIDDqgNode) = qgate;
		      *(ckt->CKTrhs + here->B3SOIDDqbNode) = qbody;
		      *(ckt->CKTrhs + here->B3SOIDDqeNode) = qsub;
		      *(ckt->CKTrhs + here->B3SOIDDdum1Node) = here->B3SOIDDdum1;
		      *(ckt->CKTrhs + here->B3SOIDDdum2Node) = here->B3SOIDDdum2;
		      *(ckt->CKTrhs + here->B3SOIDDdum3Node) = here->B3SOIDDdum3; 
		      *(ckt->CKTrhs + here->B3SOIDDdum4Node) = here->B3SOIDDdum4;
		      *(ckt->CKTrhs + here->B3SOIDDdum5Node) = here->B3SOIDDdum5; 
                      /* end clean up last */
		   }


		   /*
		    *  load y matrix
		    */
		      (*(here->B3SOIDDEgPtr) += m * gcegb);
   		      (*(here->B3SOIDDEdpPtr) += m * gcedb);
   		      (*(here->B3SOIDDEspPtr) += m * gcesb);
		      (*(here->B3SOIDDGePtr) += m * gcgeb);
		      (*(here->B3SOIDDDPePtr) += m * (Gme + gddpe + gcdeb));
		      (*(here->B3SOIDDSPePtr) += m * (gsspe - Gme + gcseb));

		      Gmin = ckt->CKTgmin * 1e-6;
                      (*(here->B3SOIDDEbPtr) -= m * (gcegb + gcedb + gcesb + gceeb));
                      (*(here->B3SOIDDGbPtr) -= m * (gcggb + gcgdb + gcgsb + gcgeb));
                      (*(here->B3SOIDDDPbPtr) -= m * (-gddpb - Gmbs + gcdgb + gcddb + gcdeb + gcdsb));
                      (*(here->B3SOIDDSPbPtr) -= m * (-gsspb + Gmbs + gcsgb + gcsdb + gcseb + gcssb));
                      (*(here->B3SOIDDBePtr) += m * (gbbe + gcbeb));
                      (*(here->B3SOIDDBgPtr) += m * (gcbgb + gbbg));
                      (*(here->B3SOIDDBdpPtr) += m * (gcbdb + gbbdp));
                      (*(here->B3SOIDDBspPtr) += m * (gcbsb + gbbsp - Gmin));
                      (*(here->B3SOIDDBbPtr) += m * (gbbb - gcbgb - gcbdb - gcbsb - gcbeb + Gmin)) ;
                   
		   (*(here->B3SOIDDEePtr) += m * gceeb);

		   (*(here->B3SOIDDGgPtr) += m * (gcggb + ckt->CKTgmin));
		   (*(here->B3SOIDDGdpPtr) += m * (gcgdb - ckt->CKTgmin));
		   (*(here->B3SOIDDGspPtr) += m * gcgsb );

		   (*(here->B3SOIDDDPgPtr) += m * ((Gm + gcdgb) + gddpg - ckt->CKTgmin));
		   (*(here->B3SOIDDDPdpPtr) += m * ((here->B3SOIDDdrainConductance
					 + here->B3SOIDDgds + gddpdp
					 + RevSum + gcddb) + ckt->CKTgmin));
		   (*(here->B3SOIDDDPspPtr) -= m * (-gddpsp + here->B3SOIDDgds + FwdSum - gcdsb));
					 
		   (*(here->B3SOIDDDPdPtr) -= m * here->B3SOIDDdrainConductance);

		   (*(here->B3SOIDDSPgPtr) += m * (gcsgb - Gm + gsspg));
		   (*(here->B3SOIDDSPdpPtr) -= m * (here->B3SOIDDgds - gsspdp + RevSum - gcsdb));
		   (*(here->B3SOIDDSPspPtr) += m * (here->B3SOIDDsourceConductance
					 + here->B3SOIDDgds + gsspsp
					 + FwdSum + gcssb));
		   (*(here->B3SOIDDSPsPtr) -= m * here->B3SOIDDsourceConductance);


		   (*(here->B3SOIDDDdPtr) += m * here->B3SOIDDdrainConductance);
		   (*(here->B3SOIDDDdpPtr) -= m * here->B3SOIDDdrainConductance);


		   (*(here->B3SOIDDSsPtr) += m * here->B3SOIDDsourceConductance);
		   (*(here->B3SOIDDSspPtr) -= m * here->B3SOIDDsourceConductance);

		   if (here->B3SOIDDbodyMod == 1) {
		      (*(here->B3SOIDDBpPtr) -= m * gppp);
		      (*(here->B3SOIDDPbPtr) += m * gppb);
		      (*(here->B3SOIDDPpPtr) += m * gppp);
	               (*(here->B3SOIDDPgPtr) += m * gppg);
			(*(here->B3SOIDDPdpPtr) += m * gppdp);
			(*(here->B3SOIDDPspPtr) += m * gppsp);
			(*(here->B3SOIDDPePtr) += m * gppe);
		   }

		   if (selfheat) 
                   {
		      (*(here->B3SOIDDDPtempPtr) += m * (GmT + gddpT + gcdT));
		      (*(here->B3SOIDDSPtempPtr) += m * (-GmT + gsspT + gcsT));
		      (*(here->B3SOIDDBtempPtr) += m * (gbbT + gcbT));
                      (*(here->B3SOIDDEtempPtr) += m * (gceT));
                      (*(here->B3SOIDDGtempPtr) += m * (gcgT));
		      if (here->B3SOIDDbodyMod == 1) {
			  (*(here->B3SOIDDPtempPtr) += m * gppT);
		      }
		      (*(here->B3SOIDDTemptempPtr) += m * (gTtt  + 1/pParam->B3SOIDDrth + gcTt));
                      (*(here->B3SOIDDTempgPtr) += m * gTtg);
                      (*(here->B3SOIDDTempbPtr) += m * gTtb);
                      (*(here->B3SOIDDTempePtr) += m * gTte);
                      (*(here->B3SOIDDTempdpPtr) += m * gTtdp);
                      (*(here->B3SOIDDTempspPtr) += m * gTtsp);
		   }

		   if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1))
		   {
		      *(here->B3SOIDDVbsPtr) += m * 1;  
		      *(here->B3SOIDDIdsPtr) += m * 1;
		      *(here->B3SOIDDIcPtr) += m * 1;
		      *(here->B3SOIDDIbsPtr) += m * 1;
		      *(here->B3SOIDDIbdPtr) += m * 1;
		      *(here->B3SOIDDIiiPtr) += m * 1;
		      *(here->B3SOIDDIgidlPtr) += m * 1;
		      *(here->B3SOIDDItunPtr) += m * 1;
		      *(here->B3SOIDDIbpPtr) += m * 1;
		      *(here->B3SOIDDAbeffPtr) += m * 1;
		      *(here->B3SOIDDVbs0effPtr) += m * 1;
		      *(here->B3SOIDDVbseffPtr) += m * 1;
		      *(here->B3SOIDDXcPtr) += m * 1;
		      *(here->B3SOIDDCbgPtr) += m * 1;
		      *(here->B3SOIDDCbbPtr) += m * 1;
		      *(here->B3SOIDDCbdPtr) += m * 1;
		      *(here->B3SOIDDqbPtr) += m * 1;
		      *(here->B3SOIDDQbfPtr) += m * 1;
		      *(here->B3SOIDDQjsPtr) += m * 1;
		      *(here->B3SOIDDQjdPtr) += m * 1;

                      /* clean up last */
		      *(here->B3SOIDDGmPtr) += m * 1;
		      *(here->B3SOIDDGmbsPtr) += m * 1;
		      *(here->B3SOIDDGdsPtr) += m * 1;
		      *(here->B3SOIDDGmePtr) += m * 1;
		      *(here->B3SOIDDVbs0teffPtr) += m * 1;
		      *(here->B3SOIDDVgsteffPtr) += m * 1;
		      *(here->B3SOIDDCbePtr) += m * 1;
		      *(here->B3SOIDDVthPtr) += m * 1;
		      *(here->B3SOIDDXcsatPtr) += m * 1;
		      *(here->B3SOIDDVdscvPtr) += m * 1;
		      *(here->B3SOIDDVcscvPtr) += m * 1;
		      *(here->B3SOIDDQaccPtr) += m * 1;
		      *(here->B3SOIDDQsub0Ptr) += m * 1;
		      *(here->B3SOIDDQsubs1Ptr) += m * 1;
		      *(here->B3SOIDDQsubs2Ptr) += m * 1;
		      *(here->B3SOIDDqgPtr) += m * 1;
		      *(here->B3SOIDDqdPtr) += m * 1;
		      *(here->B3SOIDDqePtr) += m * 1;
		      *(here->B3SOIDDDum1Ptr) += m * 1;
		      *(here->B3SOIDDDum2Ptr) += m * 1;
		      *(here->B3SOIDDDum3Ptr) += m * 1;
		      *(here->B3SOIDDDum4Ptr) += m * 1;
		      *(here->B3SOIDDDum5Ptr) += m * 1; 
                      /* end clean up last */
		   }

	line1000:  ;

/*  Here NaN will be detected in any conductance or equivalent current.  Note
    that nandetect is initialized within the "if" statements  */

                   if ((nandetect = isnan (*(here->B3SOIDDGbPtr))) != 0)
                      { strcpy (nanmessage, "GbPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDEbPtr))) != 0)
                      { strcpy (nanmessage, "EbPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDDPbPtr))) != 0)
                      { strcpy (nanmessage, "DPbPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDSPbPtr))) != 0)
                      { strcpy (nanmessage, "SPbPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDBbPtr))) != 0)
                      { strcpy (nanmessage, "BbPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDBgPtr))) != 0)
                      { strcpy (nanmessage, "BgPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDBePtr))) != 0)
                      { strcpy (nanmessage, "BePtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDBdpPtr))) != 0)
                      { strcpy (nanmessage, "BdpPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDBspPtr))) != 0)
                      { strcpy (nanmessage, "BspPtr"); }
                   
                   else if ((nandetect = isnan (*(here->B3SOIDDGgPtr))) != 0)
                   { strcpy (nanmessage, "GgPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDGdpPtr))) != 0)
                   { strcpy (nanmessage, "GdpPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDGspPtr))) != 0)
                   { strcpy (nanmessage, "GspPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDDPgPtr))) != 0)
                   { strcpy (nanmessage, "DPgPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDDPdpPtr))) != 0)
                   { strcpy (nanmessage, "DPdpPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDDPspPtr))) != 0)
                   { strcpy (nanmessage, "DPspPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDSPgPtr))) != 0)
                   { strcpy (nanmessage, "SPgPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDSPdpPtr))) != 0)
                   { strcpy (nanmessage, "SPdpPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDSPspPtr))) != 0)
                   { strcpy (nanmessage, "SPspPtr"); }
                   else if ((nandetect = isnan (*(here->B3SOIDDEePtr))) != 0)
                   { strcpy (nanmessage, "EePtr"); }
 
                   /*  At this point, nandetect = 0 if none of the
                       conductances checked so far are NaN  */
 
                   if (nandetect == 0)
                   {
                     if ((nandetect = isnan (*(here->B3SOIDDEgPtr))) != 0)
                      { strcpy (nanmessage, "EgPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDEdpPtr))) != 0)
                      { strcpy (nanmessage, "EdpPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDEspPtr))) != 0)
                      { strcpy (nanmessage, "EspPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDGePtr))) != 0)
                      { strcpy (nanmessage, "GePtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDDPePtr))) != 0)
                      { strcpy (nanmessage, "DPePtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDSPePtr))) != 0)
                      { strcpy (nanmessage, "SPePtr"); } }
 
                   /*  Now check if self-heating caused NaN if nothing else
                       has so far (check tempnode current also)  */
 
                   if (selfheat && nandetect == 0)
                   {
                     if ((nandetect = isnan (*(here->B3SOIDDTemptempPtr))) != 0)
                      { strcpy (nanmessage, "TemptempPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDTempgPtr))) != 0)
                      { strcpy (nanmessage, "TempgPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDTempbPtr))) != 0)
                      { strcpy (nanmessage, "TempbPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDTempePtr))) != 0)
                      { strcpy (nanmessage, "TempePtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDTempdpPtr))) != 0)
                      { strcpy (nanmessage, "TempdpPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDTempspPtr))) != 0)
                      { strcpy (nanmessage, "TempspPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDGtempPtr))) != 0)
                      { strcpy (nanmessage, "GtempPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDDPtempPtr))) != 0)
                      { strcpy (nanmessage, "DPtempPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDSPtempPtr))) != 0)
                      { strcpy (nanmessage, "SPtempPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDEtempPtr))) != 0)
                      { strcpy (nanmessage, "EtempPtr"); }
                     else if ((nandetect = isnan (*(here->B3SOIDDBtempPtr))) != 0)
                      { strcpy (nanmessage, "BtempPtr"); }
                     else if ((nandetect = isnan (*(ckt->CKTrhs + here->B3SOIDDtempNode))) != 0)
                      { strcpy (nanmessage, "tempNode"); }
                   }
 
                   /*  Lastly, check all equivalent currents (tempnode is
                       checked above  */
 
                   if (nandetect == 0)
                   {
                      if ((nandetect = isnan (*(ckt->CKTrhs
                                                + here->B3SOIDDgNode))) != 0)
                      { strcpy (nanmessage, "gNode"); }
                      else if ((nandetect = isnan (*(ckt->CKTrhs
                                                     + here->B3SOIDDbNode))) != 0)
                      { strcpy (nanmessage, "bNode"); }
                      else if ((nandetect = isnan (*(ckt->CKTrhs
                                                     + here->B3SOIDDdNodePrime))) != 0)
                      { strcpy (nanmessage, "dpNode"); }
                      else if ((nandetect = isnan (*(ckt->CKTrhs
                                                     + here->B3SOIDDsNodePrime))) != 0)
                      { strcpy (nanmessage, "spNode"); }
                      else if ((nandetect = isnan (*(ckt->CKTrhs
                                                     + here->B3SOIDDeNode))) != 0)
                      { strcpy (nanmessage, "eNode"); } 
                   }

                   /*  Now print error message if NaN detected.  Note that
                       error will only be printed once (the first time it is
                       encountered) each time SPICE is run since nanfound is
                       static variable  */

                   if (nanfound == 0 && nandetect)
                   {
                      fprintf(stderr, "Alberto says:  YOU TURKEY!  %s is NaN for instance %s at time %g!\n", nanmessage, here->B3SOIDDname, ckt->CKTtime);
                      nanfound = nandetect;
		      fprintf(stderr, " The program exit!\n");
		      controlled_exit(EXIT_FAILURE);
                   }

		   if (here->B3SOIDDdebugMod > 2)
		   {
      fprintf(fpdebug, "Ids = %.4e, Ic = %.4e, cqdrn = %.4e, gmin=%.3e\n", 
                        Ids, Ic, cqdrn, ckt->CKTgmin);
		      fprintf(fpdebug, "Iii = %.4e, Idgidl = %.4e, Ibs = %.14e\n",
			      Iii, Idgidl, Ibs);
		      fprintf(fpdebug, "Ibd = %.4e, Ibp = %.4e\n", Ibd, Ibp);
		      fprintf(fpdebug, "qbody = %.5e, qbf = %.5e, qbe = %.5e\n",
			      qbody, Qbf, -(Qe1+Qe2));
		      fprintf(fpdebug, "qbs = %.5e, qbd = %.5e\n", qjs, qjd);
		      fprintf(fpdebug, "qdrn = %.5e, qinv = %.5e\n", qdrn, qinv);




/*  I am trying to debug the convergence problems here by printing out
    the entire Jacobian and equivalent current matrix  */


  if (here->B3SOIDDdebugMod > 4) {
  fprintf(fpdebug, "Ibtot = %.6e;\t Cbtot = %.6e;\n", Ibs+Ibp+Ibd-Iii-Idgidl-Isgidl, cqbody);
  fprintf(fpdebug, "ceqg = %.6e;\t ceqb = %.6e;\t ceqdp = %.6e;\t ceqsp = %.6e;\n",
                    *(ckt->CKTrhs + here->B3SOIDDgNode),
                    *(ckt->CKTrhs + here->B3SOIDDbNode),
                    *(ckt->CKTrhs + here->B3SOIDDdNodePrime),
                    *(ckt->CKTrhs + here->B3SOIDDsNodePrime));
  fprintf(fpdebug, "ceqe = %.6e;\t ceqp = %.6e;\t ceqth = %.6e;\n",
                    *(ckt->CKTrhs + here->B3SOIDDeNode),
                    *(ckt->CKTrhs + here->B3SOIDDpNode), 
                    *(ckt->CKTrhs + here->B3SOIDDtempNode));

  fprintf(fpdebug, "Eg = %.5e;\t Edp = %.5e;\t Esp = %.5e;\t Eb = %.5e;\n",
		   *(here->B3SOIDDEgPtr), *(here->B3SOIDDEdpPtr), *(here->B3SOIDDEspPtr),
		   *(here->B3SOIDDEbPtr));
  fprintf(fpdebug, "Ee = %.5e;\t Gg = %.5e;\t Gdp = %.5e;\t Gsp = %.5e;\n",
		   *(here->B3SOIDDEePtr),
		   *(here->B3SOIDDGgPtr),
		   *(here->B3SOIDDGdpPtr),
		   *(here->B3SOIDDGspPtr));
  fprintf(fpdebug, "Gb = %.5e;\t Ge = %.5e;\t DPg = %.5e;\t DPdp = %.5e;\n",
		   *(here->B3SOIDDGbPtr),
		   *(here->B3SOIDDGePtr),
		   *(here->B3SOIDDDPgPtr),
		   *(here->B3SOIDDDPdpPtr));
  fprintf(fpdebug, "DPsp = %.5e;\t DPb = %.5e;\t DPe = %.5e;\t\n",
		   *(here->B3SOIDDDPspPtr),
		   *(here->B3SOIDDDPbPtr),
		   *(here->B3SOIDDDPePtr));
  fprintf(fpdebug, "DPd = %.5e;\t SPg = %.5e;\t SPdp = %.5e;\t SPsp = %.5e;\n",
		   *(here->B3SOIDDDPdPtr),
		   *(here->B3SOIDDSPgPtr),
		   *(here->B3SOIDDSPdpPtr),
		   *(here->B3SOIDDSPspPtr));
  fprintf(fpdebug, "SPb = %.5e;\t SPe = %.5e;\t SPs = %.5e;\n",
		   *(here->B3SOIDDSPbPtr),
		   *(here->B3SOIDDSPePtr),
		   *(here->B3SOIDDSPsPtr));
  fprintf(fpdebug, "Dd = %.5e;\t Ddp = %.5e;\t Ss = %.5e;\t Ssp = %.5e;\n",
		   *(here->B3SOIDDDdPtr),
		   *(here->B3SOIDDDdpPtr),
		   *(here->B3SOIDDSsPtr),
		   *(here->B3SOIDDSspPtr));
  fprintf(fpdebug, "Bg = %.5e;\t Bdp = %.5e;\t Bsp = %.5e;\t Bb = %.5e;\n",
		   *(here->B3SOIDDBgPtr),
		   *(here->B3SOIDDBdpPtr),
		   *(here->B3SOIDDBspPtr),
		   *(here->B3SOIDDBbPtr));
  fprintf(fpdebug, "Be = %.5e;\t Btot = %.5e;\t DPtot = %.5e;\n",
                   *(here->B3SOIDDBePtr), 
                   *(here->B3SOIDDBgPtr) + *(here->B3SOIDDBdpPtr)
                   + *(here->B3SOIDDBspPtr) + *(here->B3SOIDDBbPtr)
                   + *(here->B3SOIDDBePtr),
		   *(here->B3SOIDDDPePtr)
		   + *(here->B3SOIDDDPgPtr) + *(here->B3SOIDDDPdpPtr)
		   + *(here->B3SOIDDDPspPtr) + *(here->B3SOIDDDPbPtr));
  if (selfheat) {
    fprintf (fpdebug, "DPtemp = %.5e;\t SPtemp = %.5e;\t Btemp = %.5e;\n",
                      *(here->B3SOIDDDPtempPtr), *(here->B3SOIDDSPtempPtr),
                      *(here->B3SOIDDBtempPtr));
    fprintf (fpdebug, "Gtemp = %.5e;\t Etemp = %.5e;\n",
                      *(here->B3SOIDDGtempPtr), *(here->B3SOIDDEtempPtr));
    fprintf (fpdebug, "Tempg = %.5e;\t Tempdp = %.5e;\t Tempsp = %.5e;\t Tempb = %.5e;\n",
		      *(here->B3SOIDDTempgPtr), *(here->B3SOIDDTempdpPtr),
		      *(here->B3SOIDDTempspPtr), *(here->B3SOIDDTempbPtr));
    fprintf (fpdebug, "Tempe = %.5e;\t TempT = %.5e;\t Temptot = %.5e;\n",
		      *(here->B3SOIDDTempePtr), *(here->B3SOIDDTemptempPtr),
		      *(here->B3SOIDDTempgPtr) + *(here->B3SOIDDTempdpPtr)
		      + *(here->B3SOIDDTempspPtr)+ *(here->B3SOIDDTempbPtr)
		      + *(here->B3SOIDDTempePtr));
  }

   if (here->B3SOIDDbodyMod == 1)
   {
      fprintf(fpdebug, "ceqbodcon=%.5e;\t", ceqbodcon);
      fprintf(fpdebug, "Bp = %.5e;\t Pb = %.5e;\t Pp = %.5e;\n", -gppp, gppb, gppp);
      fprintf(fpdebug, "Pg=%.5e;\t Pdp=%.5e;\t Psp=%.5e;\t Pe=%.5e;\n",
                    gppg, gppdp, gppsp, gppe);
   }
}

	if (here->B3SOIDDdebugMod > 3)
        {
           fprintf(fpdebug, "Vth = %.4f, Vbs0eff = %.8f, Vdsat = %.4f\n",
                   Vth, Vbs0eff, Vdsat);
           fprintf(fpdebug, "ueff = %g, Vgsteff = %.4f, Vdseff = %.4f\n",
                   ueff, Vgsteff, Vdseff);
           fprintf(fpdebug, "Vthfd = %.4f, Vbs0mos = %.4f, Vbs0 = %.4f\n",
                   Vthfd, Vbs0mos, Vbs0);
           fprintf(fpdebug, "Vbs0t = %.4f, Vbsdio = %.8f\n",
                   Vbs0t, Vbsdio);
        }

              fclose(fpdebug);
           }

     here->B3SOIDDiterations++;  /*  increment the iteration counter  */

     }  /* End of Mosfet Instance */
}   /* End of Model Instance */


return(OK);
}

