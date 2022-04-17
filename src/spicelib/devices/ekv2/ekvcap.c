/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */
#include "ngspice/cktdefs.h"
#include "ekvdefs.h"
#include "ngspice/const.h"

extern void F_v_simple(double,double*,double*,double*);
extern void F_v_n(double,double*,double*,double*);
void EKVcap(
EKVmodel *,
EKVinstance *,
double ,
double ,
double *,
double *,
double *);

void EKVcap(
EKVmodel *model,
EKVinstance *here,
double vds,
double vbs,
double *cgs,
double *cgd,
double *cgb)
{
	double sqrt_if, dif_dv;
	double sqrt_ir, dir_dv;
	double cgsi, cgdi, cgbi;
	double ratioV_ir, ratioV_if;
	double QD, QG, QS, QB, QI;
	double dQD_dVD, dQD_dVS, dQD_dVG;
	double dQS_dVD, dQS_dVS, dQS_dVG;
	double dQB_dVD, dQB_dVS, dQB_dVG;
	double dQG_dVD, dQG_dVS, dQG_dVG;
	double dVP_dVD, dVP_dVS, dVP_dVG;
	double dif_dVD, dif_dVS, dif_dVG;
	double dir_dVD, dir_dVS, dir_dVG;
	double sir2, sif, sif2, sir, sif3, sir3;
	double sif_sir_2, sif_sir_3;
	double n_1, n_1_n, n_Vt_COX;
	double VP_PHI_eps, sqrt_PHI_VP_2, WLCox;
	double dQ_i_factor, pgamma;
	double vp, Vt, phi, i_f, i_r, ir, n;
	double weff, leff;
	/*
 * wg 17-SEP-2K  additional declarations for EKV v2.6 rev.XII   
 */
	double VS, VD, PHI_VS, PHI_VD, sqrt_PHI_VS_Vt, sqrt_PHI_VD_Vt;
	double VGprime, eps_COX, WETA_W, LETA_L, big_sqrt_VP, big_sqrt_VP0, VP0, sqrt_PHI_VP0;
	double GAMMAstar, sqrt_PHI_VS, sqrt_PHI_VD, sqrt_GAMMAstar, GAMMAprime;
	double tmp1,  tmp2, tmp3, tmp4, sqrt_PHI_VP2_2, VGstar, Vt_01, Vt_Vt_16;
	double dGAMMAprime_dVD, dGAMMAprime_dVS, dGAMMAprime_dVG;
	double dQI_dVD, dQI_dVS, dQI_dVG, sqrt_VGstar;

	VS = -vbs;
	VD = vds-vbs;

	Vt       = here->EKVtemp*CONSTKoverQ;
	Vt_01    = 0.1*Vt;                               /* wg 17-SEP-2K*/
	Vt_Vt_16 = 16.0*Vt*Vt;

	vp  = here->EKVvp;
	phi = here->EKVtPhi;
	VGprime = here->EKVvgprime;                       /* wg 17-SEP-2K*/
	VGstar  = here->EKVvgstar;                        /* wg 17-SEP-2K*/
	sqrt_VGstar = sqrt(VGstar*VGstar + 2.0*Vt_Vt_16); /* !! mb  99/09/09  */

	n   = here->EKVslope;

	pgamma = model->EKVgamma;

	leff  = here->EKVl+model->EKVdl;
	weff  = here->EKVw+model->EKVdw;

	WLCox   = model->EKVcox*weff*leff;
	eps_COX = epssil/model->EKVcox;
	WETA_W  = eps_COX*3.0*model->EKVweta/weff;
	LETA_L  = eps_COX*model->EKVleta/leff;

	/*
 *    Reverse normalized current for intrinsic capacitances interpolation: 
 *    !! mb 95/05/10  ir placed here 
 *    !! mb 95/05/23  replaced F(v) by sqrt_F(v), changed ratioV_ir 
 */

	i_f = here->EKVif;
	i_r = here->EKVir;


	/* Reverse normalized current: */
	ratioV_ir = (vp-VD)/Vt;

	if (model->EKVekvint == 0.0)
		F_v_n(ratioV_ir, &ir, &sqrt_ir, &dir_dv);
	else
		F_v_simple(ratioV_ir, &ir, &sqrt_ir, &dir_dv);

	/* Forward normalized current: */
	ratioV_if = (vp-VS) / Vt;

	if (model->EKVekvint == 0.0)
		F_v_n(ratioV_if, &i_f, &sqrt_if, &dif_dv);
	else
		F_v_simple(ratioV_if, &i_f, &sqrt_if, &dif_dv);


	/*----------------  NEW DYNAMIC MODEL  -------------------*/

	sif2 = 0.25+i_f;    /* !! mb 96/10/16  simplified computation (jcp)  */
	sir2 = 0.25+i_r;    /* !! mb 96/10/16  simplified computation (jcp)  */
	sif  = sqrt(sif2);
	sir  = sqrt(sir2);
	sif3 = sif*sif2;
	sir3 = sir*sir2;
	sif_sir_2 = (sif+sir)*(sif+sir); /* wg 17-SEP-2K */
	sif_sir_3 = (sif+sir)*sif_sir_2;

	VP_PHI_eps = vp+phi+1.0e-6;
	sqrt_PHI_VP_2 = 2.0*sqrt(VP_PHI_eps);
	/*
 * !! mb 97/08/21  replaced GAMMA by GAMMAa in n_1, n_1_n, QB (sa) 
 */
	n_1 = pgamma/sqrt_PHI_VP_2;
	n_1_n = pgamma/(sqrt_PHI_VP_2+pgamma);

	/*  Pinch-off voltage VP, limited to VP >= -PHI  wg 17-SEP-2K */

	PHI_VD = phi+VD;
	sqrt_PHI_VD_Vt = sqrt(PHI_VD*PHI_VD+Vt_Vt_16);
	sqrt_PHI_VD = sqrt(0.5*(PHI_VD+sqrt_PHI_VD_Vt));

	PHI_VS = phi+VS;
	sqrt_PHI_VS_Vt = sqrt(PHI_VS*PHI_VS+Vt_Vt_16);
	sqrt_PHI_VS = sqrt(0.5*(PHI_VS+sqrt_PHI_VS_Vt));
	/*  
 * !! mb 97/09/14  symmetric version of GAMMAprime necessary with charges model                   
 */
	big_sqrt_VP0 = sqrt(VGprime + 0.25*pgamma*pgamma);
	VP0 = VGprime - phi - pgamma*(big_sqrt_VP0 - 0.5*pgamma);
	sqrt_PHI_VP0 = sqrt(VP0+phi+Vt_01);
	GAMMAstar = pgamma - LETA_L *(sqrt_PHI_VS+sqrt_PHI_VD) + WETA_W * sqrt_PHI_VP0;

	/* keep GAMMAprime from becoming negative  */
	sqrt_GAMMAstar = sqrt(GAMMAstar*GAMMAstar+Vt_01);
	GAMMAprime = 0.5*(GAMMAstar+sqrt_GAMMAstar);
	big_sqrt_VP = sqrt(VGprime+0.25*GAMMAprime*GAMMAprime);

	/*  Pinch-off voltage derivatives:  */

	/*  
 *  !! mb 97/09/14  symmetric version of GAMMAprime necessary with charges model
 *  !! mb 99/05/10  (r12) New VGprime formulation (REVISION III) allows
 *		    VP derivatives to be expressed with a single equation
 */
	tmp1 = GAMMAprime / (sqrt_GAMMAstar+sqrt_GAMMAstar);
	tmp2 = VGprime/sqrt_VGstar;				 /*  dVGprime_dVG  */

	dGAMMAprime_dVD = -LETA_L * tmp1 * sqrt_PHI_VD / sqrt_PHI_VD_Vt;
	dGAMMAprime_dVS = -LETA_L * tmp1 * sqrt_PHI_VS / sqrt_PHI_VS_Vt;
	dGAMMAprime_dVG =  WETA_W * tmp1 * (big_sqrt_VP0-0.5*pgamma)
	    / (big_sqrt_VP0*sqrt_PHI_VP0) * tmp2;

	tmp3 = (vp+phi) / big_sqrt_VP;

	dVP_dVD = -tmp3 * dGAMMAprime_dVD;
	dVP_dVS = -tmp3 * dGAMMAprime_dVS;
	dVP_dVG = -tmp3 * dGAMMAprime_dVG 
	    + (1.0 - GAMMAprime/(big_sqrt_VP+big_sqrt_VP)) * tmp2;

	/*  Forward normalized current derivatives:  */

	tmp1 = dif_dv / Vt;   /*  !!  mb  95/08/28, 97/04/21 */
	dif_dVD = tmp1 * dVP_dVD;
	dif_dVS = tmp1 * (dVP_dVS-1.0);
	dif_dVG = tmp1 * dVP_dVG;

	/*  Transconductance factor derivatives:  */

	tmp1 = dir_dv / Vt;
	dir_dVD = tmp1 * (dVP_dVD-1.0);
	dir_dVS = tmp1 * dVP_dVS;
	dir_dVG = tmp1 * dVP_dVG;

	/* new charges expressions (v2.6 Revision III)  */
	tmp1 = sqrt(phi+0.5*vp);
	sqrt_PHI_VP2_2 = tmp1+tmp1;
	n_Vt_COX = (1.0 + GAMMAprime/sqrt_PHI_VP2_2) * Vt*WLCox;
	dQ_i_factor = -n_Vt_COX*four15th/sif_sir_3;

	QD = -n_Vt_COX*(four15th*(3.0*sir3+6.0*sir2*sif+4.0*sir*sif2+2.0*sif3)/sif_sir_2 - 0.5);
	QS = -n_Vt_COX*(four15th*(3.0*sif3+6.0*sif2*sir+4.0*sif*sir2+2.0*sir3)/sif_sir_2 - 0.5);
	QI = QS+QD;
	QB = WLCox * (-0.5*GAMMAprime*sqrt_PHI_VP_2 + VGprime - VGstar)
	    - QI*GAMMAprime/(GAMMAprime+sqrt_PHI_VP2_2);
	/* (end) new charges expressions (v2.6 Revision III)  */

	/* QG = -QI -Qox -QB;  */
	QG = -QI -QB;
	/* !! mb 96/11/28 Analytic derivatives of the node charges (transcapacitances) */

	/* new charges derivatives (v2.6 Revision III)  */
	tmp1 = sir2+3.0*sir*sif+sif2;
	tmp2 = 1.5*tmp1+2.5*sif2;
	tmp3 = QD/(GAMMAprime+sqrt_PHI_VP2_2);
	tmp4 = 0.5*GAMMAprime/(phi+phi+vp);
	dQD_dVD = tmp3*(dGAMMAprime_dVD-tmp4*dVP_dVD) + dQ_i_factor*(tmp1*dif_dVD + tmp2*dir_dVD);
	dQD_dVS = tmp3*(dGAMMAprime_dVS-tmp4*dVP_dVS) + dQ_i_factor*(tmp1*dif_dVS + tmp2*dir_dVS);
	dQD_dVG = tmp3*(dGAMMAprime_dVG-tmp4*dVP_dVG) + dQ_i_factor*(tmp1*dif_dVG + tmp2*dir_dVG);

	tmp2 = 1.5*tmp1+2.5*sir2;
	tmp3 = QS/(GAMMAprime+sqrt_PHI_VP2_2);
	dQS_dVD = tmp3*(dGAMMAprime_dVD-tmp4*dVP_dVD) + dQ_i_factor*(tmp2*dif_dVD + tmp1*dir_dVD);
	dQS_dVS = tmp3*(dGAMMAprime_dVS-tmp4*dVP_dVS) + dQ_i_factor*(tmp2*dif_dVS + tmp1*dir_dVS);
	dQS_dVG = tmp3*(dGAMMAprime_dVG-tmp4*dVP_dVG) + dQ_i_factor*(tmp2*dif_dVG + tmp1*dir_dVG);

	dQI_dVD = dQD_dVD+dQS_dVD;
	dQI_dVS = dQD_dVS+dQS_dVS;
	dQI_dVG = dQD_dVG+dQS_dVG;

	tmp1 = GAMMAprime + sqrt_PHI_VP2_2;
	tmp2 = sqrt_PHI_VP2_2/(tmp1*tmp1);
	tmp3 = GAMMAprime/tmp1;
	tmp4 = -WLCox*0.5*sqrt_PHI_VP_2 - QI*tmp2;
	tmp2 = GAMMAprime * (QI/(tmp1*tmp1*sqrt_PHI_VP2_2) - WLCox/sqrt_PHI_VP_2);

	dQB_dVD = tmp4 * dGAMMAprime_dVD + tmp2 * dVP_dVD - tmp3 * dQI_dVD;
	dQB_dVS = tmp4 * dGAMMAprime_dVS + tmp2 * dVP_dVS - tmp3 * dQI_dVS;
	dQB_dVG = tmp4 * dGAMMAprime_dVG + tmp2 * dVP_dVG - tmp3 * dQI_dVG
	    + WLCox*(VGprime/sqrt_VGstar - 1.0);

	dQG_dVD = -dQD_dVD - dQS_dVD - dQB_dVD;
	dQG_dVS = -dQD_dVS - dQS_dVS - dQB_dVS;
	dQG_dVG = -dQD_dVG - dQS_dVG - dQB_dVG;
	/*
 * wg 17-SEP-2K NOTE: the meaning of the variables and conversion 
 * depending on the reference of the voltages:
 * 
 * With the following denominations....
 *      derivatives of charge Qx vs. vgb, vdb, vsb:        cxgb, cxdb, cxsb
 *      derivatives of charge Qx vs. vgs, vds, vbs:        cxgs, cxds, cxbs
 *      derivatives of charge Qx vs. vgr, vdr, vsr, vbr:   cxg, cxd, cxs, cxb
 *   
 * (where X=G,B,D,S, and r is an arbitrary reference point)
 * the correspondences can be established:
 * 
 *      cxg = cxgb              = cxgs
 *      cxd = cxdb              = cxds
 *      cxs = cgsb              = -(cxgs+cxbs+cxds)
 *      cxb = -(cxgb+cxdb+cxsb) = cxbs
 * 
 * which allow to form the total of 16 transcapacitances.
 *    cggb = dQG_dVG;   cbgb = dQB_dVG;  cdgb = dQD_dVG; csgb = dQS_dVG;
 *    cgdb = dQG_dVD;   cbdb = dQB_dVD;  cddb = dQD_dVD; csdb = dQS_dVD;
 *    cgsb = dQG_dVS;   cbsb = dQB_dVS;  cdsb = dQD_dVS; cssb = dQS_dVS;
 */

	*cgs = dQG_dVS;
	*cgd = dQG_dVD;
	*cgb = -(dQG_dVG + dQG_dVD + dQG_dVS);

/*
 * XQC = 1.0 
 * Only the 5 intrinsic capacitances are computed. Capacitances
 * are derived from the new charge expressions   !! mb 96/08/23
 * These expressions do not account for the n dependence on VG, VS, VD).
 * These intrinsic capacitances should only be computed in case the
 * simulator demands capacitances rather than charges.
 */
	cgsi = two3rds*(1.0-(sir2+sir+0.5*sif)/sif_sir_2);
	cgdi = two3rds*(1.0-(sif2+sif+0.5*sir)/sif_sir_2);
	/* !! mb 98/08/08  erroneous n_1 term replaced by n_1_n  (jpm)  */
	/*cgbi = n_1*(1.0-cgsi-cgdi);  */
	cgbi = n_1_n*(1.0-cgsi-cgdi);
/*
 *   *cgs = WLCox * cgsi;
 *   *cgd = WLCox * cgdi;
 *   *cgb = WLCox * cgbi;
 */
/*
 *   csbi = cgsi * n_1;
 *   cdbi = cgdi * n_1;
 */

}/* end of EKVcap */
