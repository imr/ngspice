/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ekvdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/suffix.h"

void F_v_simple(
double,
double *, double *, double *);
void F_v_n(
double,
double *, double *, double *);
void
EKVevaluate(
CKTcircuit  *,
EKVmodel   *,
EKVinstance *,
double ,
double ,
double ,
double *,
double *);

/*----------------------------------------------------------------------------
*        __________________________________________
*
*        PLEASE READ THESE INSTRUCTIONS CAREFULLY!!
*        __________________________________________
*
*
*        Use of the EPFL-EKV MOST Model Code for Simulation
*        --------------------------------------------------
*
*        The use of this code is subject to written agreement of the modelling
*        group at the Swiss Federal Institute of Technology, Lausanne (EPFL).
*        
*        This implementation is based on the equation set described in
*        'The EPFL-EKV MOSFET Model Equations for Simulation, Version 2.6',
*        M. Bucher, C. Lallement, F. Theodoloz, C. Enz, F. Krummenacher,
*        EPFL-DE-LEG, June, 1997
*        
*        Theoretical discussion of the model can  be found in
*        C. Enz, F. Krummenacher, E. Vittoz, 'An analytical MOS transistor
*        model valid in all regions of Operation and dedicated to low-voltage
*        and low-current applications', Journal on Analog Integrated Circuits
*        and Signal Processsing, Kluwer Academic Publishers, pp. 83-114,
*        July 1995
*
*        This model is part of the public domain. Written consent of the
*        Ecole Polytechnique Federale de Lausanne (EPFL) is necessary for
*        use of this code. Third parties wishing to use this code or parts of
*        it for implementation in simulators, do have to agree to use the model
*        under its name EKV or EPFL-EKV MOST model. Any release of such
*        implementation should be authorized by EPFL, especially if it does not
*        contain all model features. EPFL can in no way be considered as being
*        responsible for damage to material or persons due to improper use of this
*        code or parts of it, or errors in this code or in the documentation.
*        Please contact the modelling group at EPFL for code error reports,
*        model & code upgrades or changes. This helps to keep this code as
*        'clean' as possible.
*
*        This code is not a stand-alone version for simulation, and adaptations
*        to particular environments are necessary. This version of the code
*        should be considered with care, although the C code has been compiled.
*        No guarantees can be given regarding its functionality. This code is
*        intended to give an idea of what a part of the implementation of the
*        EKV model could look like as a help for further implementations. The
*        EPFL-EKV model itself is constantly being improved and changes to the
*        model or its coding and thus its documentation may be frequent.
*        Version numbers are used to differentiate various releases. Backward
*        compatibility with previous releases is an aim but cannot be guaranteed.
*
*        Since the model is intended to be used in circuit simulators which
*        can handle a large number of transistors for various simulation models,
*        a particular data structure is proposed which helps to differentiate
*        what belongs to the simulation model (in this case, the EKV model)
*        and what is relied to the devices and their particular geometries.
*        To optimize the overall code and take advantage of this data structure,
*        the code parts have been 'distributed' into several functions, which
*        need to be called respectively for necessary initializations, or any
*        bias or temperature change, according to the needs of the circuit
*        simulator.
*
*        Included is the code for the intrinsic MOS model:
*          - static and quasi-static model equations,
*          - analytical computation of derivatives
*            (transconductances and transcapacitances),
*          - temperature effects,
*          - noise model equations.
*
*        The code regarding the extrinsic part of the MOS transistor is not
*        included here. Implementation in simulators should provide similar
*        functionality as commonly used for other MOS models.
*      
*        Please visit:
*        ---------------
*        hppt://legwww.epfl.ch/ekv
*
----------------------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include <ctype.h>

/*************************************************************************/
/*          F(x) interpolation function and derivative:                  */
/*************************************************************************/

/*  !!  mb  95/08/28  EKV code version 2.2

    This new 'true' interpolation function replaces the simplified,
    approximate or tabulated versions used before.
 
    To speed up computation, the current, its square root, and the
    derivative are put together in one function.
 
    !!  mb  95/12/19  modified names for int. function constants
 
    !!  mb  97/02/10  ekv v2.4 interpolation function
 
*/


/*************************************************************************/
void F_v_simple(
double v,
double *i_v, double *sqrt_i_v, double *d_i_v)
/*  !!  mb  96/08/23  introduced simple interpolation function as an option  */
{
	double  u = 0.5*v;

	if (u > 20.0)
	{
		*sqrt_i_v = u;
		*i_v = u*u;
		*d_i_v = u;
	}
	else if (u > -18.0)
	{
		double exp_u = exp(u);
		double s_i_v = log(1.0+exp_u);

		*sqrt_i_v = s_i_v;
		*i_v = s_i_v*s_i_v;
		*d_i_v = exp_u*s_i_v/(1.0+exp_u);
	}
	else
	{
		double exp_u = exp(u) + 1.0e-32;
		double exp_u2 = exp_u*exp_u;

		*sqrt_i_v = exp_u;
		*i_v = exp_u2;
		*d_i_v = exp_u2;
	}
}


/*************************************************************************/
void F_v_n(
double v,
double *i_v, double *sqrt_i_v, double *d_i_v)
/*  !!  mb  97/02/10  ekv v2.4 interpolation function  */
{
	double  yk;

	if (v > -0.35)
	{
		double  z0 = 2.0/(1.3+v-log(v+1.6));
		double  zk = (2.0+z0)/(1.0+v+log(z0));
		yk = (1.0+v+log(zk))/(2.0+zk);
	}
	else if (v > -15.0)
	{
		double  z0 = 1.55+exp(-v);
		double  zk = (2.0+z0)/(1.0+v+log(z0));
		yk = (1.0+v+log(zk))/(2.0+zk);
	}
	else if (v > -23.0)
		yk = 1.0/(2.0+exp(-v));
	else
		yk = exp(v) + 1.0e-64;

	*i_v = yk*(1.0+yk);
	*sqrt_i_v = sqrt(*i_v);
	*d_i_v = yk;

}


void
EKVevaluate(
CKTcircuit  *ckt,
EKVmodel   *model,
EKVinstance *here,
double vds,
double vbs,
double vgs,
double *vonPointer,
double *vdsatPointer)
{
	NG_IGNORE(ckt);

	/*************************************/
	/*        EKV FUNCTIONS              */
	/*************************************/

	double VD, VS, VG;              /*  intrinsic voltages ref. to bulk  */
	double VGprime, GAMMAprime=0.0; /*  short and narrow channel effect  */
	double VGstar;
	double VP, VPprime;             /*  pinch-off voltage                */
	double if_;                     /*  forward normalized current       */
	double VDSS;                    /*  saturation voltage               */
	double irprime;                 /*  reverse normalized current       */
	double deltaL;          /*  channel length reduction         */
	double beta;                    /*  transconductance factor          */
	double n;                       /*  slope factor                     */
	double Ispec, IDS;              /*  specific current, drain current  */
	double ir;

	double big_sqrt_VP=0.0, sqrt_VP_Vt;
	double sqrt_VGstar, big_sqrt_VP0, VP0, sqrt_PHI_VP0; /* wg 17-SEP-2K */

	double sqrt_if, dif_dv, ratioV_if, ratioV_irprime;
	double sqrt_irprime, ddeltaV_factor;
	double THETA_VP_1, sqrt_PHI_VP, if_ir, Leq;
	double dGAMMAprime_dVD, dGAMMAprime_dVS, dGAMMAprime_dVG;
	double dVP_dVD, dVP_dVG, dVP_dVS;
	double dVPprime_dVD, dVPprime_dVG, dVPprime_dVS;
	double dif_factor, dif_dVD, dif_dVG, dif_dVS;
	double dVDSS_factor, dVDSS_dVD, dVDSS_dVG, dVDSS_dVS;
	double dirprime_factor, dirprime_dVD, dirprime_dVG, dirprime_dVS;
	double ddeltaV_dVD, ddeltaV_dVG, ddeltaV_dVS;
	double ddeltaL_factor, ddeltaL_dVD, ddeltaL_dVG, ddeltaL_dVS;
	double dbeta_dVD, dbeta_dVG, dbeta_dVS;
	double dn_dVD, dn_dVG, dn_dVS;
	double GAMMA_sqrt_PHI, eps_COX, Lc, Lc_LAMBDA, WETA_W, LETA_L;
	double Vt_Vt, Vt_Vt_2, Vt_Vt_16, Vt_2, Vt_4, Lc_UCRIT;
	double KP_Weff, COX_KP_Vt, deltaV_2, Vip;
	double VDSS_sqrt, sqrt_VDSS_deltaV, sqrt_Vds_VDSS_deltaV;
	double VDSSprime_sqrt, sqrt_VDSSprime_deltaV, sqrt_Vds_VDSSprime_deltaV;
	double Lprime, Lmin, sqrt_Lprime_Lmin;
	double dVip_dVD, dVip_dVG, dVip_dVS;
	double VDSSprime,dVDSSprime_factor;
	double dVDSSprime_dVD, dVDSSprime_dVG, dVDSSprime_dVS;
	double dLeq_factor, dLeq_dVD, dLeq_dVG, dLeq_dVS;
	double Vds, Vc, log_Vc_Vt, pgamma, plambda, GAMMA_2;
	double GAMMAstar, sqrt_GAMMAstar=0.0;
	double inv_Vib, Vib, isubprime, gm_isub, gds_isub, gms_isub;
	double PHI_VD=0.0, sqrt_PHI_VD=0.0, sqrt_PHI_VD_Vt=0.0;
	double PHI_VS=0.0, sqrt_PHI_VS=0.0, sqrt_PHI_VS_Vt=0.0;
	double dirprime_dv, cox, V0, Vt_01, inv_Vt;
	double sqv, vL, deltaVFB, IBN_2;
	double inv_sqrt_Vds_VDSS_deltaV;
	double inv_sqrt_VDSSprime_deltaV;
	double inv_sqrt_Vds_VDSSprime_deltaV;
	double inv_sqrt_VDSS_deltaV, sqrt_ir, ratioV_ir, ibn, lk;
	double leff, weff, iba, xj, theta, e0, weta, ucrit, phi, Vt;
	double kp, tibb, vt0, gds, gms, gm, dids_dvg1, dids_dvb1;
	double dids_dv21, isub, leta;
	/*
 * wg 17-SEP-2K  additional declarations for EKV v2.6 rev.XII   
 */
	double sif2, sir2, sif, sir, sif_sir_2, VP_PHI_eps, sqrt_PHI_VP_2;
	double n_1, n_1_n, qi, qb, E0_Q_1, T0, eta_qi, T0_GAMMA_1;
	double tmp1, tmp2, tmp3;
	double dir_dv, dir_dVD, dir_dVS, dir_dVG, dQI_dVD, dQI_dVS, dQI_dVG;
	double dQB_dVD, dQB_dVS, dQB_dVG;
	{
		VS = -vbs;
		VG = vgs-vbs;
		VD = vds-vbs;

		leff = here->EKVl+model->EKVdl;
		weff = here->EKVw+model->EKVdw;

		pgamma  = model->EKVgamma;
		plambda = model->EKVlambda;
		iba     = model->EKViba;
		ibn     = model->EKVibn;
		cox     = model->EKVcox;
		xj      = model->EKVxj;
		theta   = model->EKVtheta;
		e0      = model->EKVe0;
		leta    = model->EKVleta;
		weta    = model->EKVweta;
		cox     = model->EKVcox;
		ucrit   = here->EKVtucrit;
		phi     = here->EKVtPhi;
		Vt      = here->EKVtemp*CONSTKoverQ;
		kp      = here->EKVtkp;
		tibb    = here->EKVtibb;
		vt0     = model->EKVtype*here->EKVtVto;
		lk      = model->EKVlk;

		Vt_2     = 2.0*Vt;
		Vt_4     = 4.0*Vt;
		Vt_Vt    = Vt*Vt;
		Vt_Vt_2  = 2.0*Vt_Vt;
		Vt_Vt_16 = 16.0*Vt_Vt;

		Vt_01    = 0.1*Vt;

		inv_Vt = 1.0/Vt;

		GAMMA_sqrt_PHI = pgamma*sqrt(phi);

		eps_COX = epssil/cox;
		T0 = 1.0 / (eps_COX * e0);
		eta_qi = model->EKVtype > 0 ? 0.5 : 0.3333333333333;

		Lc = sqrt(eps_COX*xj);
		Lc_LAMBDA = Lc*plambda;
		Lc_UCRIT  = Lc*ucrit;

		COX_KP_Vt = cox/(kp*Vt_2);

		GAMMA_2 = 0.01*pgamma*pgamma;   /*  !!  mb  95/12/19  */

		KP_Weff = kp*weff;
		Vc = ucrit*leff;
		log_Vc_Vt = Vt*(log(Vc/Vt_2)-1.0);

		WETA_W = eps_COX*3.0*weta/weff;
		LETA_L = eps_COX*leta/leff;

		V0 = 2.0*model->EKVq0/cox;        /*  !!   mb  97/04/14  */

		IBN_2 = 2.0*ibn;

		/****************************************/
		/*     STATIC MODEL EQUATIONS           */
		/****************************************/

		/*  Pinch-off voltage, including short and narrow channel effects:  */

		/*  VGprime: !! mb 97/04/14 introduced RSCE equations (v2.5)  */
		if (V0 == 0.0)
			deltaVFB = 0.0;
		else
		{
			vL = c_a * (leff/lk - 0.1);
			sqv = 1.0 / (1.0 + 0.5*(vL + sqrt(vL*vL + c_ee)));
			deltaVFB = V0 * sqv * sqv;
		}
		/*  NOTE: New VGprime formulation (REVISION III) allows VP to be
       *        expressed with a single equation
       */

		/*  !!  mb  99/05/10  (r12) */
		VGstar = VG - vt0 - deltaVFB + phi + GAMMA_sqrt_PHI;
		sqrt_VGstar = sqrt(VGstar*VGstar + 2.0*Vt_Vt_16);
		/*  !!  mb  99/09/09  */
		VGprime = 0.5*(VGstar + sqrt_VGstar);

		/*  Pinch-off voltage VP, limited to VP >= -PHI  */

		PHI_VD = phi+VD;
		sqrt_PHI_VD_Vt = sqrt(PHI_VD*PHI_VD+Vt_Vt_16);
		sqrt_PHI_VD    = sqrt(0.5*(PHI_VD+sqrt_PHI_VD_Vt));

		PHI_VS = phi+VS;
		sqrt_PHI_VS_Vt = sqrt(PHI_VS*PHI_VS+Vt_Vt_16);
		sqrt_PHI_VS    = sqrt(0.5*(PHI_VS+sqrt_PHI_VS_Vt));

		/*  !!  mb  97/09/14  symmetric version of GAMMAprime 
         *                    necessary with charges model 
         */
		big_sqrt_VP0 = sqrt(VGprime + 0.25*pgamma*pgamma);
		VP0 = VGprime - phi - pgamma*(big_sqrt_VP0 - 0.5*pgamma);
		sqrt_PHI_VP0 = sqrt(VP0+phi+Vt_01);
		GAMMAstar = pgamma - LETA_L * (sqrt_PHI_VS+sqrt_PHI_VD)
		    + WETA_W * sqrt_PHI_VP0;

		/*  keep GAMMAprime from becoming negative  */
		sqrt_GAMMAstar = sqrt(GAMMAstar*GAMMAstar+Vt_01);

		GAMMAprime = 0.5*(GAMMAstar+sqrt_GAMMAstar);

		big_sqrt_VP = sqrt(VGprime+0.25*GAMMAprime*GAMMAprime);

		VP = VGprime-phi-GAMMAprime*(big_sqrt_VP-0.5*GAMMAprime);


		/*  Forward normalized current:  */

		ratioV_if = (VP-VS)*inv_Vt;                    /*  !!  mb  95/08/14  */
		ratioV_ir = (VP-VD)*inv_Vt;

		if (model->EKVekvint==0.0) {
			F_v_n(ratioV_ir, &ir,  &sqrt_ir, &dir_dv);   /*  !!  wg  95/08/14  */
			F_v_n(ratioV_if, &if_, &sqrt_if, &dif_dv);   /*  !!  mb  95/08/14  */
		}
		else {
			F_v_simple(ratioV_ir, &ir,  &sqrt_ir, &dir_dv);
			F_v_simple(ratioV_if, &if_, &sqrt_if, &dif_dv);
		}

		/*  Saturation voltage:  */

		VDSS_sqrt = sqrt(0.25+sqrt_if*Vt/Vc);

		VDSS = Vc*(VDSS_sqrt-0.5);

		Vds = 0.5*(VD-VS);
		deltaV_2 = Vt_Vt_16*(plambda*(sqrt_if-VDSS/Vt)+15.625e-3);

		sqrt_VDSS_deltaV     = sqrt(VDSS*VDSS+deltaV_2);
		sqrt_Vds_VDSS_deltaV = sqrt((Vds-VDSS)*(Vds-VDSS)+deltaV_2);

		Vip = sqrt_VDSS_deltaV-sqrt_Vds_VDSS_deltaV;

		VDSSprime_sqrt = sqrt(0.25+(sqrt_if-0.75*log(if_))*Vt/Vc);
		VDSSprime      = Vc*(VDSSprime_sqrt-0.5)+log_Vc_Vt;

		/*  Reverse normalized current:  */

		sqrt_VDSSprime_deltaV     = sqrt(VDSSprime*VDSSprime+deltaV_2);
		sqrt_Vds_VDSSprime_deltaV = sqrt((Vds-VDSSprime)*(Vds-VDSSprime)+deltaV_2);

		ratioV_irprime = (VP-Vds-VS-sqrt_VDSSprime_deltaV+sqrt_Vds_VDSSprime_deltaV)
		    *inv_Vt;

		if (model->EKVekvint == 0.0)
			F_v_n(ratioV_irprime, &irprime, &sqrt_irprime, &dirprime_dv);
		else
			F_v_simple(ratioV_irprime, &irprime, &sqrt_irprime, &dirprime_dv);

		/*  Channel length modulation & mobility reduction due to longitudinal field:  */

		deltaL = Lc_LAMBDA*log(1.0+(Vds-Vip)/Lc_UCRIT);

		Lprime = leff-deltaL+(Vds+Vip)/ucrit;
		Lmin = 0.1*leff;

		sqrt_Lprime_Lmin = sqrt(Lprime*Lprime+Lmin*Lmin);
		Leq = 0.5*(Lprime+sqrt_Lprime_Lmin);

		/*  Transconductance factor:  */

		sif2 = 0.25+if_;
		sir2 = 0.25+ir;
		sif = sqrt(sif2);
		sir = sqrt(sir2);
		sif_sir_2 = (sif+sir)*(sif+sir);

		VP_PHI_eps = VP+phi+1.0e-6;
		sqrt_PHI_VP_2 = 2.0*sqrt(VP_PHI_eps);

		n_1 = pgamma/sqrt_PHI_VP_2;
		n_1_n = pgamma/(sqrt_PHI_VP_2+pgamma);

		/*  Normalized inversion charge  (qi=QI/WLCox)  */
		qi = -(1.0+n_1)*Vt*((two3rds+two3rds)*(sir2+sir*sif+sif2)/(sif+sir) - 1.0);

		/*  Normalized depletion charge  (qb=QB/WLCox), for depletion to inversion  */
		qb = -0.5*pgamma*sqrt_PHI_VP_2 - n_1_n*qi;


		if (e0 == 0.0)
		{

			/*  NOTE: this version of the simple mobility model from prior versions
         *   of the EKV Model Is reinstated. In case E0 is *not* specified, this
         *   simple mobility Model Is used according to THETA, if specified.
         */

			/*  VPprime:  */

			/*  !!  mb  eliminated discontinuity of derivative of 1+THETA*VP  */
			sqrt_VP_Vt = sqrt(VP*VP + Vt_Vt_2);
			VPprime = 0.5 * (VP + sqrt_VP_Vt);

			THETA_VP_1 = 1.0+theta*VPprime;

			beta = kp * weff / (Leq * THETA_VP_1);    /*  !!  mb  97/07/18  */

		}
		else
		{
			/*  new model for mobility reduction, linked to the charges model  */
			/*  !!  mb  98/10/11  (r10)  introduced fabs(Eeff) (jpm)  */
			E0_Q_1 = 1.0 + T0*fabs(qb+eta_qi*qi);
			/*  !!  mb  97/06/02  ekv v2.6  */
			T0_GAMMA_1 = 1.0 + T0*GAMMA_sqrt_PHI;
			/*  !!  mb  97/07/18  */
			beta = kp * weff * T0_GAMMA_1 / (Leq * E0_Q_1);
		}

		/*  Slope factor:  */

		/*  !!  mb introduced new formula to avoid divergence of n for VP->-PHI  */
		/*  !!  mb suppressed q_NFS_COX term (computation not complete)          */

		sqrt_PHI_VP = sqrt(phi+VP+Vt_4);   /*  !!  mb  95/12/19  introduced Vt_4  */

		n = 1.0 + pgamma/(2.0*sqrt_PHI_VP);

		/*  Drain current:  */

		if_ir = if_-irprime;
		Ispec = Vt_Vt_2 * n * beta;

		IDS = Ispec * if_ir;

		/*******     ASSIGN COMPUTED CURRENT    *******/

		if (vds < 0.0) here->EKVcd *= -1.0;

		/*  Store forward and reverse currents for noise analysis  */

		here->EKVif = if_;
		here->EKVir = ir;
		here->EKVirprime = irprime;


		/*******        START OF ANALYTIC DERIVATIVES COMPUTATION                *******/

		/*  NOTE:  Some simulators calculate derivatives only numerically. In this case
           the following part on analytical derivatives should be suppressed. In
           some cases, simulators let the user choose if analytical or numerical
           derivatives should be used. The following flag needs to be set
           internally by the simulator. 'analytic_deriv' should be set only when
           really needed.
*/

		/*  Short and narrow channel effects derivatives:  */

		/*  Pinch-off voltage derivatives:  */
		/*
 * !! mb 97/09/14 symmetric version of GAMMAprime necessary with charges model
 * !! mb 99/05/10 (r12) New VGprime formulation (REVISION III) allows
 *                VP derivatives to be expressed with a single equation
 */


		tmp1 = GAMMAprime / (sqrt_GAMMAstar+sqrt_GAMMAstar);
		tmp2 = VGprime/sqrt_VGstar;             /*  dVGprime_dVG  */

		dGAMMAprime_dVD = -LETA_L * tmp1 * sqrt_PHI_VD / sqrt_PHI_VD_Vt;
		dGAMMAprime_dVS = -LETA_L * tmp1 * sqrt_PHI_VS / sqrt_PHI_VS_Vt;
		dGAMMAprime_dVG =  WETA_W * tmp1 * (big_sqrt_VP0-0.5*pgamma)
		    / (big_sqrt_VP0*sqrt_PHI_VP0) * tmp2;

		tmp3 = (VP+phi) / big_sqrt_VP;

		dVP_dVD = -tmp3 * dGAMMAprime_dVD;
		dVP_dVS = -tmp3 * dGAMMAprime_dVS;
		dVP_dVG = -tmp3 * dGAMMAprime_dVG + (1.0 - GAMMAprime/(big_sqrt_VP+big_sqrt_VP)) * tmp2;


		/*  Forward normalized current derivatives:  */

		dif_factor = dif_dv*inv_Vt;   /*  !!  mb  95/08/28  */

		dif_dVD = dif_factor*dVP_dVD;
		dif_dVS = dif_factor*(dVP_dVS-1.0);
		dif_dVG = dif_factor*dVP_dVG;

		/*  Saturation voltage derivatives:  */

		dVDSS_factor = Vt/(4.0*VDSS_sqrt*sqrt_if);

		dVDSS_dVD = dVDSS_factor*dif_dVD;
		dVDSS_dVS = dVDSS_factor*dif_dVS;
		dVDSS_dVG = dVDSS_factor*dif_dVG;

		/*  deltaV derivatives:  */

		ddeltaV_factor = (Vt_4+Vt_4) * plambda;

		ddeltaV_dVD = ddeltaV_factor*(0.5*dif_dVD/sqrt_if-dVDSS_dVD*inv_Vt);
		ddeltaV_dVS = ddeltaV_factor*(0.5*dif_dVS/sqrt_if-dVDSS_dVS*inv_Vt);
		ddeltaV_dVG = ddeltaV_factor*(0.5*dif_dVG/sqrt_if-dVDSS_dVG*inv_Vt);

		/*  Vip derivatives:  */

		inv_sqrt_VDSS_deltaV = 1.0/sqrt_VDSS_deltaV;            /*  !!  mb  97/04/21  */
		inv_sqrt_Vds_VDSS_deltaV = 1.0/sqrt_Vds_VDSS_deltaV;


		dVip_dVD = (VDSS*dVDSS_dVD+ddeltaV_dVD)*inv_sqrt_VDSS_deltaV
		    -((Vds-VDSS)*(0.5-dVDSS_dVD)+ddeltaV_dVD)*inv_sqrt_Vds_VDSS_deltaV;

		dVip_dVS = (VDSS*dVDSS_dVS+ddeltaV_dVS)*inv_sqrt_VDSS_deltaV
		    -((Vds-VDSS)*(-0.5-dVDSS_dVS)+ddeltaV_dVS)*inv_sqrt_Vds_VDSS_deltaV;

		dVip_dVG = (VDSS*dVDSS_dVG+ddeltaV_dVG)*inv_sqrt_VDSS_deltaV
		    -((Vds-VDSS)*(-dVDSS_dVG)+ddeltaV_dVG)*inv_sqrt_Vds_VDSS_deltaV;

		/*  VDSSprime derivatives:  */

		dVDSSprime_factor = Vt*(sqrt_if-1.5)/(4.0*VDSSprime_sqrt*if_);

		dVDSSprime_dVD = dVDSSprime_factor*dif_dVD;
		dVDSSprime_dVS = dVDSSprime_factor*dif_dVS;
		dVDSSprime_dVG = dVDSSprime_factor*dif_dVG;

		/*  Reverse normalized current derivatives:  */

		dirprime_factor = dirprime_dv*inv_Vt;

		inv_sqrt_VDSSprime_deltaV = 1.0/sqrt_VDSSprime_deltaV;            /*  !!  mb  97/04/21  */
		inv_sqrt_Vds_VDSSprime_deltaV = 1.0/sqrt_Vds_VDSSprime_deltaV;

		dirprime_dVD = dirprime_factor*(dVP_dVD-0.5
		    -(VDSSprime*dVDSSprime_dVD+ddeltaV_dVD)*inv_sqrt_VDSSprime_deltaV
		    +((Vds-VDSSprime)*(0.5-dVDSSprime_dVD)+ddeltaV_dVD)*inv_sqrt_Vds_VDSSprime_deltaV);

		dirprime_dVS = dirprime_factor*(dVP_dVS-0.5
		    -(VDSSprime*dVDSSprime_dVS+ddeltaV_dVS)*inv_sqrt_VDSSprime_deltaV
		    +((Vds-VDSSprime)*(-0.5-dVDSSprime_dVS)+ddeltaV_dVS)*inv_sqrt_Vds_VDSSprime_deltaV);

		dirprime_dVG = dirprime_factor*(dVP_dVG
		    -(VDSSprime*dVDSSprime_dVG+ddeltaV_dVG)*inv_sqrt_VDSSprime_deltaV
		    +((Vds-VDSSprime)*(-dVDSSprime_dVG)+ddeltaV_dVG)*inv_sqrt_Vds_VDSSprime_deltaV);

		/*  Channel length modulation & mobility reduction derivatives:  */

		/*  deltaL derivatives:   */

		ddeltaL_factor = Lc_LAMBDA/(Lc_UCRIT+Vds-Vip);

		ddeltaL_dVD = ddeltaL_factor*(0.5-dVip_dVD);
		ddeltaL_dVS = ddeltaL_factor*(-0.5-dVip_dVS);
		ddeltaL_dVG = -ddeltaL_factor*dVip_dVG;

		/*  Leq derivatives:  */

		dLeq_factor = 1.0/sqrt_Lprime_Lmin;     /*  !!  mb  95/12/19  new code  */

		dLeq_dVD = dLeq_factor*(-ddeltaL_dVD+(0.5+dVip_dVD)/ucrit);
		dLeq_dVS = dLeq_factor*(-ddeltaL_dVS+(-0.5+dVip_dVS)/ucrit);
		dLeq_dVG = dLeq_factor*(-ddeltaL_dVG+dVip_dVG/ucrit);

		/*  Transconductance factor derivatives:  */
		tmp1 = dir_dv*inv_Vt;

		dir_dVD = tmp1 * (dVP_dVD-1.0);
		dir_dVS = tmp1 * dVP_dVS;
		dir_dVG = tmp1 * dVP_dVG;

		tmp1 = -(1.0+n_1)*Vt*two3rds/sif_sir_2;
		tmp2 = tmp1*(sif+2.0*sir);
		tmp3 = tmp1*(sir+2.0*sif);

		/*  !!  mb  97/09/04  replaced QI by qi  (dQI_dVx are normalized to WLCOX)  */
		tmp1 = -n_1*qi/((2.0+n_1+n_1)*VP_PHI_eps);

		dQI_dVD = tmp1 * dVP_dVD + tmp2 * dif_dVD + tmp3 * dir_dVD;
		dQI_dVS = tmp1 * dVP_dVS + tmp2 * dif_dVS + tmp3 * dir_dVS;
		dQI_dVG = tmp1 * dVP_dVG + tmp2 * dif_dVG + tmp3 * dir_dVG;

		/*  !!  mb  97/09/04  replaced QI by qi  (dQB_dVx are normalized to WLCOX)  */
		tmp1 = (1.0+n_1)-qi/(2.0*(1.0+n_1)*VP_PHI_eps);

		dQB_dVD = -n_1_n * (tmp1 * dVP_dVD + dQI_dVD);
		dQB_dVS = -n_1_n * (tmp1 * dVP_dVS + dQI_dVS);
		dQB_dVG = -n_1_n * (tmp1 * dVP_dVG + dQI_dVG);

		if (e0 == 0.0)
		{
			tmp1 = theta * VPprime / (THETA_VP_1 * sqrt_VP_Vt);

			/*  VPprime derivatives:  */
			dVPprime_dVD = tmp1 * dVP_dVD;
			dVPprime_dVS = tmp1 * dVP_dVS;
			dVPprime_dVG = tmp1 * dVP_dVG;

			/*  in fact dbeta_dVX / beta  */
			dbeta_dVD = -dLeq_dVD - dVPprime_dVD;
			dbeta_dVS = -dLeq_dVS - dVPprime_dVS;
			dbeta_dVG = -dLeq_dVG - dVPprime_dVG;
		}
		else
		{
			tmp1 = T0 / E0_Q_1;

			/*  in fact dbeta_dVX / beta  */
			/*  !!  mb   97/08/21 corrected derivatives (sa)  */
			dbeta_dVD = -dLeq_dVD + tmp1 * (dQB_dVD+eta_qi*dQI_dVD);
			dbeta_dVS = -dLeq_dVS + tmp1 * (dQB_dVS+eta_qi*dQI_dVS);
			dbeta_dVG = -dLeq_dVG + tmp1 * (dQB_dVG+eta_qi*dQI_dVG);
		}

		/*  Slope factor derivatives:  */

		/*  !!  mb  97/08/21  replaced GAMMA by GAMMAa (sa)  */
		/*  !!  mb  95/12/19  */
		tmp1 = -pgamma/(4.0*n*sqrt_PHI_VP*(phi+VP+Vt_4));
		dn_dVD = tmp1 * dVP_dVD;
		dn_dVS = tmp1 * dVP_dVS;
		dn_dVG = tmp1 * dVP_dVG;

		/*  Transconductances:  */

		gds = Ispec*((dn_dVD + dbeta_dVD)*if_ir + dif_dVD - dirprime_dVD);
		gms = -Ispec*((dn_dVS + dbeta_dVS)*if_ir + dif_dVS - dirprime_dVS);
		gm  = Ispec*((dn_dVG + dbeta_dVG)*if_ir + dif_dVG - dirprime_dVG);

		/*******      ASSIGN DERIVATIVES    *******/

		dids_dvg1 = gm;
		dids_dvb1 = gms - gm - gds;    /* gmb = gms - gm - gds; */
		dids_dv21 = gds;

		if (vds<0)
		{
			dids_dvg1 = -dids_dvg1;
			dids_dvb1 = -dids_dvb1;
			dids_dv21 = gms;        /* gms = gds+gm+gmb */
		}

		/*  NOTE:  The fourth derivative of the current is the negative of the sum of the
           three other derivatives, which would result in -(gm+gmbs+gds) = -gms.
 
           Until here the 'electrical' variables have been computed and need to
           be converted to 'topological' ones.
 
           In case source and drain voltages have been permuted, the conductances
           dids_dvg1 and dids_dvb1 change sign, dids_dv21 and the fourth derivative
           are permuted and change sign
*/
	}

	*vdsatPointer = VDSS;

	here->EKVvgeff = n*(VP-VS);
	here->EKVslope = n;
	here->EKVvp    = VP;

	*vonPointer = vgs-here->EKVvgeff;

	/*  Impact ionization current:  */
	/*  !!  mb  95/12/19  introduced impact ionization  */

	Vib = VD-VS-IBN_2*VDSS; /*  !!  mb  96/07/15  corrected VDSS factor  */

	if ( (Vib>0.0) && ((iba/tibb) > 0.0) ) /*  !!  mb  98/05/14  */
	{
		inv_Vib   = 1.0/Vib;
		isubprime = iba/tibb*Vib*exp(-Lc*tibb*inv_Vib);
		isub      = IDS*isubprime;
		gds_isub  = isubprime*(gds+IDS*inv_Vib*(1.0+Lc*tibb*inv_Vib)
		    *(1.0-model->EKVibn*dVDSS_dVD));
		gm_isub   = isubprime*(gm+IDS*inv_Vib*(1.0+Lc*tibb*inv_Vib)
		    *(-model->EKVibn*dVDSS_dVG));
		gms_isub  = isubprime*(gms+IDS*inv_Vib*(1.0+Lc*tibb*inv_Vib)
		    *(-model->EKVibn*dVDSS_dVS));
	}
	else {
		isub=0.0;
		gds_isub =0.0;
		gm_isub  =0.0;
		gms_isub =0.0;
	}

	here->EKVisub = isub;
	here->EKVcd   = IDS+isub;
	here->EKVgds  = gds+gds_isub;
	here->EKVgm   = gm+gm_isub;
	here->EKVgms  = gms+gms_isub;
	here->EKVgmbs = gms - gm - gds;

	here->EKVvgprime = VGprime;
	here->EKVvgstar  = VGstar;  /* wg 17-SEP-2K needed for new Q model */


}
