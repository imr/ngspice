/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Min-Chie Jeng, Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/suffix.h"

/* This routine evaluates the drain current, its derivatives and the
 * charges associated with the gate,bulk and drain terminal
 * using the B2 (Berkeley Short-Channel IGFET Model) Equations.
 */
void
B2evaluate(double Vds, double Vbs, double Vgs, B2instance *here, B2model *model,
           double *gm, double *gds, double *gmb, double *qg, double *qb, double *qd,
           double *cgg,double *cgd,double *cgs, double *cbg, double *cbd, double *cbs,
           double *cdg, double *cdd, double *cds, double *Ids, double *von, 
           double *vdsat, CKTcircuit *ckt)
 {
    double Vth, Vdsat = 0.0;
    double Phisb, T1s, Eta, Gg, Aa, Inv_Aa, U1, U1s, Vc, Kk, SqrtKk;
    double dPhisb_dVb, dT1s_dVb, dVth_dVb, dVth_dVd, dAa_dVb, dVc_dVd;
    double dVc_dVg, dVc_dVb, dKk_dVc;
    double dVdsat_dVd = 0.0, dVdsat_dVg = 0.0, dVdsat_dVb = 0.0;
    double dUvert_dVg, dUvert_dVd, dUvert_dVb, Inv_Kk;
    double dUtot_dVd, dUtot_dVb, dUtot_dVg, Ai, Bi, Vghigh, Vglow, Vgeff, Vof;
    double Vbseff, Vgst, Vgdt, Qbulk, Utot;
    double T0, T1, T2, T3, T4, T5, Arg1, Arg2, Exp0 = 0.0, Exp1 = 0.0;
    double tmp, tmp1, tmp2, tmp3, Uvert, Beta1, Beta2, Beta0, dGg_dVb;
    double T6, T7, T8, T9, n = 0.0, ExpArg, ExpArg1;
    double Beta, dQbulk_dVb, dVgdt_dVg, dVgdt_dVd;
    double dVbseff_dVb, Ua, Ub, dVgdt_dVb, dQbulk_dVd;
    double Con1, Con3, Con4, SqrVghigh, SqrVglow, CubVghigh, CubVglow;
    double delta, Coeffa, Coeffb, Coeffc, Coeffd, Inv_Uvert, Inv_Utot;
    double Inv_Vdsat, tanh, Sqrsech, dBeta1_dVb, dU1_dVd, dU1_dVg, dU1_dVb;
    double Betaeff, FR, dFR_dVd, dFR_dVg, dFR_dVb, Betas, Beta3, Beta4;
    double dBeta_dVd, dBeta_dVg, dBeta_dVb, dVgeff_dVg, dVgeff_dVd, dVgeff_dVb;
    double dCon3_dVd, dCon3_dVb, dCon4_dVd, dCon4_dVb, dCoeffa_dVd, dCoeffa_dVb;
    double dCoeffb_dVd, dCoeffb_dVb, dCoeffc_dVd, dCoeffc_dVb;
    double dCoeffd_dVd, dCoeffd_dVb;
    int ChargeComputationNeeded;
    int	valuetypeflag;			/* added  3/19/90 JSD   */

    if ((ckt->CKTmode & (MODEAC | MODETRAN)) ||
        ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ||
        (ckt->CKTmode & MODEINITSMSIG) )
    {   ChargeComputationNeeded  =  1;
    }
    else
    {   ChargeComputationNeeded  =  0;
    }

    if (Vbs < model->B2vbb2) Vbs = model->B2vbb2;
    if (Vgs > model->B2vgg2) Vgs = model->B2vgg2;
    if (Vds > model->B2vdd2) Vds = model->B2vdd2;

/* Threshold Voltage. */
    if (Vbs <= 0.0)
    {   Phisb = here->pParam->B2phi - Vbs;
        dPhisb_dVb = -1.0;
        T1s = sqrt(Phisb);
        dT1s_dVb = -0.5 / T1s; 
    }
    else
    {   tmp = here->pParam->B2phi / (here->pParam->B2phi + Vbs);
        Phisb = here->pParam->B2phi * tmp;
        dPhisb_dVb = -tmp * tmp;
        T1s = here->pParam->Phis3 / (here->pParam->B2phi + 0.5 * Vbs);
        dT1s_dVb = -0.5 * T1s * T1s / here->pParam->Phis3;
    }

    Eta = here->pParam->B2eta0 + here->pParam->B2etaB * Vbs;
    Ua = here->pParam->B2ua0 + here->pParam->B2uaB * Vbs;
    Ub = here->pParam->B2ub0 + here->pParam->B2ubB * Vbs;
    U1s = here->pParam->B2u10 + here->pParam->B2u1B * Vbs;

    Vth = here->pParam->B2vfb + here->pParam->B2phi + here->pParam->B2k1
        * T1s - here->pParam->B2k2 * Phisb - Eta * Vds;
    dVth_dVd = -Eta;
    dVth_dVb = here->pParam->B2k1 * dT1s_dVb + here->pParam->B2k2 
	     - here->pParam->B2etaB * Vds;

    Vgst = Vgs - Vth;

    tmp = 1.0 / (1.744 + 0.8364 * Phisb);
    Gg = 1.0 - tmp;
    dGg_dVb = 0.8364 * tmp * tmp * dPhisb_dVb;
    T0 = Gg / T1s;
    tmp1 = 0.5 * T0 * here->pParam->B2k1;
    Aa = 1.0 + tmp1;
    dAa_dVb = (Aa - 1.0) * (dGg_dVb / Gg - dT1s_dVb / T1s);
    Inv_Aa = 1.0 / Aa;

    Vghigh = here->pParam->B2vghigh;
    Vglow = here->pParam->B2vglow;

    if ((Vgst >= Vghigh) || (here->pParam->B2n0 == 0.0))
    {   Vgeff = Vgst;
	dVgeff_dVg = 1.0;
	dVgeff_dVd = -dVth_dVd;
	dVgeff_dVb = -dVth_dVb;
    }
    else
    {   Vof = here->pParam->B2vof0 + here->pParam->B2vofB * Vbs
	    + here->pParam->B2vofD * Vds;
	n = here->pParam->B2n0 + here->pParam->B2nB / T1s 
	  + here->pParam->B2nD * Vds;
	tmp = 0.5 / (n * model->B2Vtm);

	ExpArg1 = -Vds / model->B2Vtm;
	ExpArg1 = MAX(ExpArg1, -30.0);
	Exp1 = exp(ExpArg1);
	tmp1 = 1.0 - Exp1;
	tmp1 = MAX(tmp1, 1.0e-18);
	tmp2 = 2.0 * Aa * tmp1;

	if (Vgst <= Vglow)
	{
            ExpArg = Vgst * tmp;
	    ExpArg = MAX(ExpArg, -30.0);
	    Exp0 = exp(0.5 * Vof + ExpArg);
	    Vgeff = sqrt(tmp2) * model->B2Vtm * Exp0;
	    T0 = n * model->B2Vtm;
	    dVgeff_dVg = Vgeff * tmp;
	    dVgeff_dVd = dVgeff_dVg * (n / tmp1 * Exp1 - dVth_dVd - Vgst
		       * here->pParam->B2nD / n + T0 * here->pParam->B2vofD);
	    dVgeff_dVb = dVgeff_dVg * (here->pParam->B2vofB * T0
		       - dVth_dVb + here->pParam->B2nB * Vgst
		       / (n * T1s * T1s) * dT1s_dVb + T0 * Inv_Aa * dAa_dVb);
        }
	else
	{
            ExpArg = Vglow * tmp;
	    ExpArg = MAX(ExpArg, -30.0);
	    Exp0 = exp(0.5 * Vof + ExpArg);
	    Vgeff = sqrt(2.0 * Aa * (1.0 - Exp1)) * model->B2Vtm * Exp0;
	    Con1 = Vghigh;
	    Con3 = Vgeff;
	    Con4 = Con3 * tmp;
	    SqrVghigh = Vghigh * Vghigh;
	    SqrVglow = Vglow * Vglow;
	    CubVghigh = Vghigh * SqrVghigh;
	    CubVglow = Vglow * SqrVglow;
	    T0 = 2.0 * Vghigh;
	    T1 = 2.0 * Vglow;
	    T2 = 3.0 * SqrVghigh;
	    T3 = 3.0 * SqrVglow;
	    T4 = Vghigh - Vglow;
            T5 = SqrVghigh - SqrVglow;
            T6 = CubVghigh - CubVglow;
            T7 = Con1 - Con3;
            delta = (T1 - T0) * T6 + (T2 - T3) * T5 + (T0 * T3 - T1 * T2) * T4;
	    delta = 1.0 / delta;
            Coeffb = (T1 - Con4 * T0) * T6 + (Con4 * T2 - T3) * T5 
		   + (T0 * T3 - T1 * T2) * T7;
            Coeffc = (Con4 - 1.0) * T6 + (T2 - T3) * T7 + (T3 - Con4 * T2) * T4;
            Coeffd = (T1 - T0) * T7 + (1.0 - Con4) * T5 + (Con4 * T0 - T1) * T4;
            Coeffa = SqrVghigh * (Coeffc + Coeffd * T0);
            Vgeff = (Coeffa + Vgst * (Coeffb + Vgst * (Coeffc + Vgst * Coeffd)))
		  * delta;
	    dVgeff_dVg = (Coeffb + Vgst * (2.0 * Coeffc + 3.0 * Vgst * Coeffd))
		       * delta;
	    T7 = Con3 * tmp;
	    T8 = dT1s_dVb * here->pParam->B2nB / (T1s * T1s * n);
	    T9 = n * model->B2Vtm;
	    dCon3_dVd = T7 * (n * Exp1 / tmp1 - Vglow * here->pParam->B2nD 
		      / n + T9 * here->pParam->B2vofD);
	    dCon3_dVb = T7 * (T9 * Inv_Aa * dAa_dVb + Vglow * T8
		      + T9 * here->pParam->B2vofB);
	    dCon4_dVd = tmp * dCon3_dVd - T7 * here->pParam->B2nD / n;
	    dCon4_dVb = tmp * dCon3_dVb + T7 * T8;

            dCoeffb_dVd = dCon4_dVd * (T2 * T5 - T0 * T6) + dCon3_dVd 
			* (T1 * T2 - T0 * T3);
            dCoeffc_dVd = dCon4_dVd * (T6 - T2 * T4) + dCon3_dVd * (T3 - T2);
            dCoeffd_dVd = dCon4_dVd * (T0 * T4 - T5) + dCon3_dVd * (T0 - T1);
            dCoeffa_dVd = SqrVghigh * (dCoeffc_dVd + dCoeffd_dVd * T0);

	    dVgeff_dVd = -dVgeff_dVg * dVth_dVd + (dCoeffa_dVd + Vgst 
		       * (dCoeffb_dVd + Vgst * (dCoeffc_dVd + Vgst 
		       * dCoeffd_dVd))) * delta;

            dCoeffb_dVb = dCon4_dVb * (T2 * T5 - T0 * T6) + dCon3_dVb 
			* (T1 * T2 - T0 * T3);
            dCoeffc_dVb = dCon4_dVb * (T6 - T2 * T4) + dCon3_dVb * (T3 - T2);
            dCoeffd_dVb = dCon4_dVb * (T0 * T4 - T5) + dCon3_dVb * (T0 - T1);
            dCoeffa_dVb = SqrVghigh * (dCoeffc_dVb + dCoeffd_dVb * T0);

	    dVgeff_dVb = -dVgeff_dVg * dVth_dVb + (dCoeffa_dVb + Vgst 
		       * (dCoeffb_dVb + Vgst * (dCoeffc_dVb + Vgst 
		       * dCoeffd_dVb))) * delta;
        }
    }

    if (Vgeff > 0.0)
    {
        Uvert = 1.0 + Vgeff * (Ua + Vgeff * Ub);
        Uvert = MAX(Uvert, 0.2);
        Inv_Uvert = 1.0 / Uvert;
        T8 = Ua + 2.0 * Ub * Vgeff;
        dUvert_dVg = T8 * dVgeff_dVg;
        dUvert_dVd = T8 * dVgeff_dVd;
        dUvert_dVb = T8 * dVgeff_dVb + Vgeff * (here->pParam->B2uaB
	           + Vgeff * here->pParam->B2ubB);

        T8 = U1s * Inv_Aa * Inv_Uvert;
        Vc = T8 * Vgeff;
        T9 = Vc * Inv_Uvert;
        dVc_dVg = T8 * dVgeff_dVg - T9 * dUvert_dVg;
        dVc_dVd = T8 * dVgeff_dVd - T9 * dUvert_dVd;
        dVc_dVb = T8 * dVgeff_dVb + here->pParam->B2u1B * Vgeff * Inv_Aa 
	        * Inv_Uvert - Vc * Inv_Aa * dAa_dVb - T9 * dUvert_dVb;


        tmp2 = sqrt(1.0 + 2.0 * Vc);
        Kk = 0.5 * (1.0 + Vc + tmp2);
        Inv_Kk = 1.0 / Kk;
        dKk_dVc = 0.5  + 0.5 / tmp2;
        SqrtKk = sqrt(Kk);

        T8 = Inv_Aa / SqrtKk;
        Vdsat = Vgeff * T8;
        Vdsat = MAX(Vdsat, 1.0e-18);
        Inv_Vdsat = 1.0 / Vdsat;
        T9 = 0.5 * Vdsat * Inv_Kk * dKk_dVc;
        dVdsat_dVd = T8 * dVgeff_dVd - T9 * dVc_dVd;
        dVdsat_dVg = T8 * dVgeff_dVg - T9 * dVc_dVg;
        dVdsat_dVb = T8 * dVgeff_dVb - T9 * dVc_dVb - Vdsat* Inv_Aa * dAa_dVb;

        Beta0 = here->pParam->B2beta0 + here->pParam->B2beta0B * Vbs;
        Betas = here->pParam->B2betas0 + here->pParam->B2betasB * Vbs;
        Beta2 = here->pParam->B2beta20 + here->pParam->B2beta2B * Vbs
	      + here->pParam->B2beta2G * Vgs;
        Beta3 = here->pParam->B2beta30 + here->pParam->B2beta3B * Vbs
	      + here->pParam->B2beta3G * Vgs;
        Beta4 = here->pParam->B2beta40 + here->pParam->B2beta4B * Vbs
	      + here->pParam->B2beta4G * Vgs;
        Beta1 = Betas - (Beta0 + model->B2vdd * (Beta3 - model->B2vdd 
	      * Beta4)); 

        T0 = Vds * Beta2 * Inv_Vdsat;
        T0 = MIN(T0, 30.0);
        T1 = exp(T0);
        T2 = T1 * T1;
        T3 = T2 + 1.0;
        tanh = (T2 - 1.0) / T3;
        Sqrsech = 4.0 * T2 / (T3 * T3);

        Beta = Beta0 + Beta1 * tanh + Vds * (Beta3 - Beta4 * Vds);
        T4 = Beta1 * Sqrsech * Inv_Vdsat;
        T5 = model->B2vdd * tanh;
        dBeta_dVd = Beta3 - 2.0 * Beta4 * Vds + T4 * (Beta2 - T0 * dVdsat_dVd);
        dBeta_dVg = T4 * (here->pParam->B2beta2G * Vds - T0 * dVdsat_dVg) 
	          + here->pParam->B2beta3G * (Vds - T5) 
		  - here->pParam->B2beta4G * (Vds * Vds - model->B2vdd * T5);
        dBeta1_dVb = here->pParam->Arg;
        dBeta_dVb = here->pParam->B2beta0B + dBeta1_dVb * tanh + Vds
	          * (here->pParam->B2beta3B - Vds * here->pParam->B2beta4B)
	          + T4 * (here->pParam->B2beta2B * Vds - T0 * dVdsat_dVb);
	    

        if (Vgst > Vglow)
        {   
	    if (Vds <= Vdsat) /* triode region */
	    {
	        T3 = Vds * Inv_Vdsat;
	        T4 = T3 - 1.0;
	        T2 =  1.0 - here->pParam->B2u1D * T4 * T4;
	        U1 =  U1s * T2;
	        Utot = Uvert + U1 * Vds;
	        Utot = MAX(Utot, 0.5);
	        Inv_Utot = 1.0 / Utot;
	        T5 = 2.0 * U1s * here->pParam->B2u1D * Inv_Vdsat * T4;
	        dU1_dVd = T5 * (T3 * dVdsat_dVd - 1.0);
	        dU1_dVg = T5 * T3 * dVdsat_dVg;
	        dU1_dVb = T5 * T3 * dVdsat_dVb + here->pParam->B2u1B * T2;
	        dUtot_dVd = dUvert_dVd + U1 + Vds * dU1_dVd;
	        dUtot_dVg = dUvert_dVg + Vds * dU1_dVg;
	        dUtot_dVb = dUvert_dVb + Vds * dU1_dVb;

	        tmp1 = (Vgeff - 0.5 * Aa * Vds);
                tmp3 = tmp1 * Vds;
	        Betaeff = Beta * Inv_Utot;
                *Ids = Betaeff * tmp3;
	        T6 = *Ids / Betaeff * Inv_Utot;
	        *gds = T6 * (dBeta_dVd - Betaeff * dUtot_dVd) + Betaeff * (tmp1
	            + (dVgeff_dVd - 0.5 * Aa) * Vds); 
                *gm = T6 * (dBeta_dVg - Betaeff * dUtot_dVg) + Betaeff * Vds 
	           * dVgeff_dVg;
	        *gmb = T6 * (dBeta_dVb - Betaeff * dUtot_dVb) + Betaeff * Vds
	            * (dVgeff_dVb - 0.5 * Vds * dAa_dVb);
           } 
           else  /* Saturation */
           {  tmp1 = Vgeff * Inv_Aa * Inv_Kk;
              tmp3 = 0.5 * Vgeff * tmp1;
	      Betaeff = Beta * Inv_Uvert;
              *Ids = Betaeff * tmp3;
	      T0 = *Ids / Betaeff * Inv_Uvert;
	      T1 = Betaeff * Vgeff * Inv_Aa * Inv_Kk;
	      T2 = *Ids * Inv_Kk * dKk_dVc;

	      if (here->pParam->B2ai0 != 0.0)
	      {
	          Ai = here->pParam->B2ai0 + here->pParam->B2aiB * Vbs;
	          Bi = here->pParam->B2bi0 + here->pParam->B2biB * Vbs;
	          T5 = Bi / (Vds - Vdsat);
	          T5 = MIN(T5, 30.0);
	          T6 = exp(-T5);
	          FR = 1.0 + Ai * T6;
	          T7 = T5 / (Vds - Vdsat);
	          T8 = (1.0 - FR) * T7;
	          dFR_dVd = T8 * (dVdsat_dVd - 1.0);
	          dFR_dVg = T8 * dVdsat_dVg;
	          dFR_dVb = T8 * dVdsat_dVb + T6 * (here->pParam->B2aiB - Ai
		          * here->pParam->B2biB / (Vds - Vdsat));

	          *gds = (T0 * (dBeta_dVd - Betaeff * dUvert_dVd) + T1 
		       * dVgeff_dVd - T2 * dVc_dVd) * FR + *Ids * dFR_dVd;
	          *gm = (T0 * (dBeta_dVg - Betaeff * dUvert_dVg) 
		      + T1 * dVgeff_dVg - T2 * dVc_dVg) * FR + *Ids * dFR_dVg;
	          *gmb = (T0 * (dBeta_dVb - Betaeff * dUvert_dVb) + T1 
		       * dVgeff_dVb - T2 * dVc_dVb - *Ids * Inv_Aa * dAa_dVb) 
		       * FR + *Ids * dFR_dVb;
	          *Ids *= FR;
	       }
	       else
	       {  *gds = T0 * (dBeta_dVd - Betaeff * dUvert_dVd) + T1 
		       * dVgeff_dVd - T2 * dVc_dVd;
	          *gm = T0 * (dBeta_dVg - Betaeff * dUvert_dVg) + T1 * dVgeff_dVg
		      - T2 * dVc_dVg;
	          *gmb = T0 * (dBeta_dVb - Betaeff * dUvert_dVb) + T1 
		       * dVgeff_dVb - T2 * dVc_dVb - *Ids * Inv_Aa * dAa_dVb;
	       }
           } /* end of Saturation */
       }
       else
       {   T0 = Exp0 * Exp0;
           T1 = Exp1;
           *Ids = Beta * model->B2Vtm * model->B2Vtm * T0 * (1.0 - T1);
           T2 = *Ids / Beta;
           T4 = n * model->B2Vtm;
           T3 = *Ids / T4;
           if ((Vds > Vdsat) && here->pParam->B2ai0 != 0.0)
           {   Ai = here->pParam->B2ai0 + here->pParam->B2aiB * Vbs;
               Bi = here->pParam->B2bi0 + here->pParam->B2biB * Vbs;
               T5 = Bi / (Vds - Vdsat);
               T5 = MIN(T5, 30.0);
               T6 = exp(-T5);
               FR = 1.0 + Ai * T6;
               T7 = T5 / (Vds - Vdsat);
               T8 = (1.0 - FR) * T7;
               dFR_dVd = T8 * (dVdsat_dVd - 1.0);
               dFR_dVg = T8 * dVdsat_dVg;
               dFR_dVb = T8 * dVdsat_dVb + T6 * (here->pParam->B2aiB - Ai
	               * here->pParam->B2biB / (Vds - Vdsat));
           }
           else
           {   FR = 1.0;
	       dFR_dVd = 0.0;
	       dFR_dVg = 0.0;
	       dFR_dVb = 0.0;
           }

           *gds = (T2 * dBeta_dVd + T3 * (here->pParam->B2vofD * T4 - dVth_dVd
	        - here->pParam->B2nD * Vgst / n) + Beta * model->B2Vtm 
		* T0 * T1) * FR + *Ids * dFR_dVd;
           *gm = (T2 * dBeta_dVg + T3) * FR + *Ids * dFR_dVg;
           *gmb = (T2 * dBeta_dVb + T3 * (here->pParam->B2vofB * T4 - dVth_dVb 
	        + here->pParam->B2nB * Vgst / (n * T1s * T1s) * dT1s_dVb)) * FR
	        + *Ids * dFR_dVb;
           *Ids *= FR;
       }
   }
   else
   {   *Ids = 0.0;
       *gm = 0.0;
       *gds = 0.0;
       *gmb = 0.0;
   }

    /* Some Limiting of DC Parameters */
    *gds = MAX(*gds,1.0e-20);


    if ((model->B2channelChargePartitionFlag > 1)
	 || ((!ChargeComputationNeeded) &&
	 (model->B2channelChargePartitionFlag > -5)))
    {  
        *qg  = 0.0;
        *qd = 0.0;
        *qb = 0.0;
        *cgg = 0.0;
        *cgs = 0.0;
        *cgd = 0.0;
        *cdg = 0.0;
        *cds = 0.0;
        *cdd = 0.0;
        *cbg = 0.0;
        *cbs = 0.0;
        *cbd = 0.0;
        goto finished;
    }
    else
    {
       if (Vbs < 0.0)
       {   Vbseff = Vbs;
	   dVbseff_dVb = 1.0;
       }
       else
       {   Vbseff = here->pParam->B2phi - Phisb;
	   dVbseff_dVb = -dPhisb_dVb;
       }
       Arg1 = Vgs - Vbseff - here->pParam->B2vfb;
       Arg2 = Arg1 - Vgst;
       Qbulk = here->pParam->One_Third_CoxWL * Arg2;
       dQbulk_dVb = here->pParam->One_Third_CoxWL * (dVth_dVb - dVbseff_dVb);
       dQbulk_dVd = here->pParam->One_Third_CoxWL * dVth_dVd;
       if (Arg1 <= 0.0)
       {
          *qg = here->pParam->CoxWL * Arg1;
          *qb = -(*qg);
          *qd = 0.0;

	  *cgg = here->pParam->CoxWL;
	  *cgd = 0.0;
	  *cgs = -*cgg * (1.0 - dVbseff_dVb);

	  *cdg = 0.0;
	  *cdd = 0.0;
	  *cds = 0.0;

	  *cbg = -here->pParam->CoxWL;
	  *cbd = 0.0;
	  *cbs = -*cgs;
       }
       else if (Vgst <= 0.0)
       {  T2 = Arg1 / Arg2;
          T3 = T2 * T2 * (here->pParam->CoxWL - here->pParam->Two_Third_CoxWL
	     * T2);

	  *qg = here->pParam->CoxWL * Arg1 * (1.0 - T2 * (1.0 - T2 / 3.0));
          *qb = -(*qg);
          *qd = 0.0;

	  *cgg = here->pParam->CoxWL * (1.0 - T2 * (2.0 - T2));
	  tmp = T3 * dVth_dVb - (*cgg + T3) * dVbseff_dVb;
	  *cgd = T3 * dVth_dVd;
	  *cgs = -(*cgg + *cgd + tmp);

	  *cdg = 0.0;
	  *cdd = 0.0;
	  *cds = 0.0;

	  *cbg = -*cgg;
	  *cbd = -*cgd;
	  *cbs = -*cgs;
       }
       else
       {  if (Vgst < here->pParam->B2vghigh)
	  {    Uvert = 1.0 + Vgst * (Ua + Vgst * Ub);
               Uvert = MAX(Uvert, 0.2);
               Inv_Uvert = 1.0 / Uvert;
               dUvert_dVg = Ua + 2.0 * Ub * Vgst;
               dUvert_dVd = -dUvert_dVg * dVth_dVd;
               dUvert_dVb = -dUvert_dVg * dVth_dVb + Vgst 
		    * (here->pParam->B2uaB + Vgst * here->pParam->B2ubB);

               T8 = U1s * Inv_Aa * Inv_Uvert;
               Vc = T8 * Vgst;
               T9 = Vc * Inv_Uvert;
               dVc_dVg = T8 - T9 * dUvert_dVg;
               dVc_dVd = -T8 * dVth_dVd - T9 * dUvert_dVd;
               dVc_dVb = -T8 * dVth_dVb + here->pParam->B2u1B * Vgst * Inv_Aa 
	               * Inv_Uvert - Vc * Inv_Aa * dAa_dVb - T9 * dUvert_dVb;

               tmp2 = sqrt(1.0 + 2.0 * Vc);
               Kk = 0.5 * (1.0 + Vc + tmp2);
               Inv_Kk = 1.0 / Kk;
               dKk_dVc = 0.5  + 0.5 / tmp2;
               SqrtKk = sqrt(Kk);

               T8 = Inv_Aa / SqrtKk;
               Vdsat = Vgst * T8;
               T9 = 0.5 * Vdsat * Inv_Kk * dKk_dVc;
               dVdsat_dVd = -T8 * dVth_dVd - T9 * dVc_dVd;
               dVdsat_dVg = T8 - T9 * dVc_dVg;
               dVdsat_dVb = -T8 * dVth_dVb - T9 * dVc_dVb 
			  - Vdsat* Inv_Aa * dAa_dVb;
          }
          if (Vds >= Vdsat)
          {       /* saturation region */
	      *cgg = here->pParam->Two_Third_CoxWL;
	      *cgd = -*cgg * dVth_dVd + dQbulk_dVd;
	      tmp = -*cgg * dVth_dVb + dQbulk_dVb;
	      *cgs = -(*cgg + *cgd + tmp);

	      *cbg = 0.0;
	      *cbd = -dQbulk_dVd;
	      *cbs = dQbulk_dVd + dQbulk_dVb;

	      *cdg = -0.4 * *cgg;
	      tmp = -*cdg * dVth_dVb;
	      *cdd = -*cdg * dVth_dVd;
	      *cds = -(*cdg + *cdd + tmp);

	      *qb = -Qbulk;
	      *qg = here->pParam->Two_Third_CoxWL * Vgst + Qbulk;
	      *qd = *cdg * Vgst;
           }
           else
           {       /* linear region  */
	      T7 = Vds / Vdsat;
	      T8 = Vgst / Vdsat;
	      T6 = T7 * T8;
	      T9 = 1.0 - T7;
              Vgdt = Vgst * T9;
	      T0 = Vgst / (Vgst + Vgdt);
	      T1 = Vgdt / (Vgst + Vgdt);
	      T5 = T0 * T1;
	      T2 = 1.0 -  T1 + T5;
	      T3 = 1.0 -  T0 + T5;

	      dVgdt_dVg = T9 + T6 * dVdsat_dVg;
	      dVgdt_dVd = T6 * dVdsat_dVd - T8 -T9 * dVth_dVd;
	      dVgdt_dVb = T6 * dVdsat_dVb -T9 * dVth_dVb;

	      *qg = here->pParam->Two_Third_CoxWL * (Vgst + Vgdt 
	         - Vgdt * T0) + Qbulk;
	      *qb = -Qbulk;
	      *qd = -here->pParam->One_Third_CoxWL * (0.2 * Vgdt 
		 + 0.8 * Vgst + Vgdt * T1 
		 + 0.2 * T5 * (Vgdt - Vgst));

	      *cgg = here->pParam->Two_Third_CoxWL * (T2 + T3 * dVgdt_dVg);
	      tmp = dQbulk_dVb + here->pParam->Two_Third_CoxWL * (T3 * dVgdt_dVb 
	          - T2 * dVth_dVb);
	      *cgd = here->pParam->Two_Third_CoxWL * (T3 * dVgdt_dVd 
		   - T2 * dVth_dVd) + dQbulk_dVd;
	      *cgs = -(*cgg + *cgd + tmp);

	      T2 = 0.8 - 0.4 * T1 * (2.0 * T1 + T0 + T0 * (T1 - T0));
	      T3 = 0.2 + T1 + T0 * (1.0 - 0.4 * T0 * (T1 + 3.0 * T0));
	      *cdg = -here->pParam->One_Third_CoxWL * (T2 + T3 * dVgdt_dVg);
	      tmp = here->pParam->One_Third_CoxWL * (T2 * dVth_dVb 
		  - T3 * dVgdt_dVb);
	      *cdd = here->pParam->One_Third_CoxWL * (T2 * dVth_dVd 
		   - T3 * dVgdt_dVd);
	      *cds = -(*cdg + tmp + *cdd);

	      *cbg = 0.0;
	      *cbd = -dQbulk_dVd;
	      *cbs = dQbulk_dVd + dQbulk_dVb;
           }
       }
    }

finished:       /* returning Values to Calling Routine */
    valuetypeflag = (int) model->B2channelChargePartitionFlag;
    switch (valuetypeflag)
     {
      case 0: *Ids = MAX(*Ids,1e-50);
              break;
      case -1: *Ids = MAX(*Ids,1e-50);
              break;
      case -2: *Ids = *gm;
              break;
      case -3: *Ids = *gds;
              break;
      case -4: *Ids = 1.0 / *gds;
              break;
      case -5: *Ids = *gmb;
              break;
      case -6: *Ids = *qg / 1.0e-12;
              break;
      case -7: *Ids = *qb / 1.0e-12;
              break;
      case -8: *Ids = *qd / 1.0e-12;
              break;
      case -9: *Ids = -(*qb + *qg + *qd) / 1.0e-12;
              break;
      case -10: *Ids = *cgg / 1.0e-12;
              break;
      case -11: *Ids = *cgd / 1.0e-12;
              break;
      case -12: *Ids = *cgs / 1.0e-12;
              break;
      case -13: *Ids = -(*cgg + *cgd + *cgs) / 1.0e-12;
              break;
      case -14: *Ids = *cbg / 1.0e-12;
              break;
      case -15: *Ids = *cbd / 1.0e-12;
              break;
      case -16: *Ids = *cbs / 1.0e-12;
              break;
      case -17: *Ids = -(*cbg + *cbd + *cbs) / 1.0e-12;
              break;
      case -18: *Ids = *cdg / 1.0e-12;
              break;
      case -19: *Ids = *cdd / 1.0e-12;
              break;
      case -20: *Ids = *cds / 1.0e-12;
              break;
      case -21: *Ids = -(*cdg + *cdd + *cds) / 1.0e-12;
              break;
      case -22: *Ids = -(*cgg + *cdg + *cbg) / 1.0e-12;
              break;
      case -23: *Ids = -(*cgd + *cdd + *cbd) / 1.0e-12;
              break;
      case -24: *Ids = -(*cgs + *cds + *cbs) / 1.0e-12;
              break;
      default: *Ids = MAX(*Ids, 1.0e-50);
	      break;
     }
    *von = Vth;
    *vdsat = Vdsat;
}   

