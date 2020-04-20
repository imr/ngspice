/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

/*
 * This is the function called each iteration to evaluate the
 * HICUMs in the circuit and load them into the matrix as appropriate
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "hicumL2.hpp"

#define VPT_thresh      1.0e2
#define Dexp_lim        80.0
#define Cexp_lim        80.0
#define DFa_fj          1.921812
#define RTOLC           1.0e-5
#define l_itmax         100
#define MIN_R           0.001

void QJMODF(double, double, double, double, double, double, double *, double *, double *);
void QJMOD(double,double, double, double, double, double, double, double *, double *, double *);
void HICJQ(double, double, double, double, double, double, double *, double *, double *);
void HICFCI(double, double, double, double *, double *);
void HICFCT(double, double, double *, double *);
void HICQFC(HICUMinstance *here, HICUMmodel *model, double, double, double, double *, double *, double *, double *);
void HICQFF(HICUMinstance *here, HICUMmodel *model, double, double, double *, double *, double *, double *, double *);
void HICDIO(double, double, double, double, double, double *, double *);

double FFdVc, FFdVc_ditf;




//////////////Explicit Capacitance and Charge Expression///////////////

// DEPLETION CHARGE CALCULATION
// Hyperbolic smoothing used; no punch-through
// INPUT:
//  c_0     : zero-bias capacitance
//  u_d     : built-in voltage
//  z       : exponent coefficient
//  a_j     : control parameter for C peak value at high forward bias
//  U_cap   : voltage across junction
// OUTPUT:
//  Qz      : depletion Charge
//  C       : depletion capacitance
void QJMODF(double vt, double c_0, double u_d, double z, double a_j, double U_cap, double *C, double *dC_dV, double *Qz)
{
double DFV_f,DFv_e,DFs_q,DFs_q2,DFv_j,DFdvj_dv,DFb,DFC_j1,DFQ_j;
double C1,DFv_e_u,DFs_q_u,DFs_q2_u,DFv_j_u,DFdvj_dv_u,DFb_u,d1,d1_u,DFC_j1_u;
    if(c_0 > 0.0) {
        C1       = 1.0-exp(-log(a_j)/z);
        DFV_f    = u_d*C1;
        DFv_e    = (DFV_f-U_cap)/vt;
        DFv_e_u  = -1.0/vt;
        DFs_q    = sqrt(DFv_e*DFv_e+DFa_fj);
        DFs_q_u  = DFv_e*DFv_e_u/DFs_q;
        DFs_q2   = (DFv_e+DFs_q)*0.5;
        DFs_q2_u = (DFv_e_u+DFs_q_u)*0.5;
        DFv_j    = DFV_f-vt*DFs_q2;
        DFv_j_u  = -vt*DFs_q2_u;
        DFdvj_dv = DFs_q2/DFs_q;
        DFdvj_dv_u=(DFs_q2_u*DFs_q-DFs_q_u*DFs_q2)/(DFs_q*DFs_q);
        DFb      = log(1.0-DFv_j/u_d);
        DFb_u    = -DFv_j_u/(1-DFv_j/u_d)/u_d;
        d1       = c_0*exp(-z*DFb);
        d1_u     = -d1*DFb_u*z;
        DFC_j1   = d1*DFdvj_dv;
        DFC_j1_u = d1*DFdvj_dv_u + d1_u*DFdvj_dv_u;
        *C       = DFC_j1+a_j*c_0*(1.0-DFdvj_dv);
        *dC_dV   = DFC_j1_u-a_j*c_0*DFdvj_dv_u;
        DFQ_j    = c_0*u_d*(1.0-exp(DFb*(1.0-z)))/(1.0-z);
        *Qz      = DFQ_j+a_j*c_0*(U_cap-DFv_j);
    } else {
        *C       = 0.0;
        *dC_dV   = 0.0;
        *Qz      = 0.0;
    }
}

//////////////Explicit Capacitance and Charge Expression///////////////


// DEPLETION CHARGE CALCULATION CONSIDERING PUNCH THROUGH
// smoothing of reverse bias region (punch-through)
// and limiting to a_j=Cj,max/Cj0 for forward bias.
// Important for base-collector and collector-substrate junction
// INPUT:
//  c_0     : zero-bias capacitance
//  u_d     : built-in voltage
//  z       : exponent coefficient
//  a_j     : control parameter for C peak value at high forward bias
//  v_pt    : punch-through voltage (defined as qNw^2/2e)
//  U_cap   : voltage across junction
// OUTPUT:
//  Qz      : depletion charge
//  C       : depletion capacitance
void QJMOD(double vt,double c_0, double u_d, double z, double a_j, double v_pt, double U_cap, double *C, double *C_u, double *Qz)
{
double Dz_r,Dv_p,DV_f,DC_max,DC_c,Dv_e,De,De_1,Dv_j1,Da,Dv_r,De_2,Dv_j2,Dv_j4,DCln1,DCln2,Dz1,Dzr1,DC_j1,DC_j2,DC_j3,DQ_j1,DQ_j2,DQ_j3;
double d1,d1_u,d2,Dv_e_u,De_u,De_1_u,Dv_j1_u,Dv_r_u,De_2_u,Dv_j2_u,Dv_j4_u,DCln1_u,DCln2_u,DC_j1_u,DC_j2_u,DC_j3_u;
    if(c_0 > 0.0) {
        Dz_r   = z/4.0;
        Dv_p   = v_pt-u_d;
        DV_f   = u_d*(1.0-exp(-log(a_j)/z));
        DC_max = a_j*c_0;
        DC_c   = c_0*exp((Dz_r-z)*log(v_pt/u_d));
        Dv_e   = (DV_f-U_cap)/vt;
        Dv_e_u = -1.0/vt;
        if(Dv_e < Cexp_lim) {
            De      = exp(Dv_e);
            De_u    = De*Dv_e_u;
            De_1    = De/(1.0+De);
            De_1_u  = De_u/(1.0+De)-De*De_u/((1.0+De)*(1.0 + De));
            Dv_j1   = DV_f-vt*log(1.0+De);
            Dv_j1_u = -De_u*vt/(1.0+De);
        } else {
            De_1    = 1.0;
            De_1_u  = 0.0;
            Dv_j1   = U_cap;
            Dv_j1_u = 1.0;
        }
        Da      = 0.1*Dv_p+4.0*vt;
        Dv_r    = (Dv_p+Dv_j1)/Da;
        Dv_r_u  = Dv_j1_u/Da;
        if(Dv_r < Cexp_lim) {
            De      = exp(Dv_r);
            De_u    = De*Dv_r_u;
            De_2    = De/(1.0+De);
            De_2_u  = De_u/(1.0+De)-De*De_u/((1.0+De)*(1.0 + De));
            Dv_j2   = -Dv_p+Da*log(1.0+De)-exp(-(Dv_p+DV_f/Da));
            Dv_j2_u = Da*De_u/(1.0+De);
        } else {
            De_2    = 1.0;
            De_2_u  = 0.0;
            Dv_j2   = Dv_j1;
            Dv_j2_u = Dv_j1_u;
        }
        Dv_j4   = U_cap-Dv_j1;
        Dv_j4_u = 1.0-Dv_j1_u;
        DCln1   = log(1.0-Dv_j1/u_d);
        DCln1_u = -Dv_j1_u/((1.0-Dv_j1/u_d)*u_d);
        DCln2   = log(1.0-Dv_j2/u_d);
        DCln2_u = -Dv_j2_u/((1.0-Dv_j2/u_d)*u_d);
        Dz1     = 1.0-z;
        Dzr1    = 1.0-Dz_r;
        d1      = c_0*exp(DCln2*(-z));
        d1_u    =-d1*z*DCln2_u;
        DC_j1   = d1*De_1*De_2;
        DC_j1_u = De_1*De_2*d1_u+De_1*d1_u*De_2_u+De_1_u*d1*De_2;
        d2      = DC_c*exp(DCln1*(-Dz_r));
        DC_j2   = d2*(1.0-De_2);
        DC_j2_u =-d2*De_2_u-Dz_r*d2*(1-De_2)*DCln1_u;
        DC_j3   = DC_max*(1.0-De_1);
        DC_j3_u =-DC_max*De_1_u;
        *C       = DC_j1+DC_j2+DC_j3;
        *C_u     = DC_j1_u+DC_j2_u+DC_j3_u;
        DQ_j1   = c_0*(1.0-exp(DCln2*Dz1))/Dz1;
        DQ_j2   = DC_c*(1.0-exp(DCln1*Dzr1))/Dzr1;
        DQ_j3   = DC_c*(1.0-exp(DCln2*Dzr1))/Dzr1;
        *Qz     = (DQ_j1+DQ_j2-DQ_j3)*u_d+DC_max*Dv_j4;
    } else {
        *C      = 0.0;
        *C_u    = 0.0;
        *Qz     = 0.0;
    }
}

// DEPLETION CHARGE & CAPACITANCE CALCULATION SELECTOR
// Dependent on junction punch-through voltage
// Important for collector related junctions
void HICJQ(double vt, double c_0, double u_d, double z, double v_pt, double U_cap, double *C, double *dC_dV, double *Qz)
{
    if(v_pt < VPT_thresh) {
        QJMOD(vt,c_0,u_d,z,2.4,v_pt,U_cap,C,dC_dV,Qz);
    } else {
        QJMODF(vt,c_0,u_d,z,2.4,U_cap,C,dC_dV,Qz);
    }
}

// A CALCULATION NEEDED FOR COLLECTOR MINORITY CHARGE FORMULATION
// INPUT:
//  zb,zl       : zeta_b and zeta_l (model parameters, TED 10/96)
//  w           : normalized injection width
// OUTPUT:
// hicfcio      : function of equation (2.1.17-10)
void HICFCI(double zb, double zl, double w, double *hicfcio, double *dhicfcio_dw)
{
double z,lnzb,x,a,a2,a3,r;
    z       = zb*w;
    lnzb    = log(1+zb*w);
    if(z > 1.0e-6) {
        x               = 1.0+z;
        a               = x*x;
        a2              = 0.250*(a*(2.0*lnzb-1.0)+1.0);
        a3              = (a*x*(3.0*lnzb-1.0)+1.0)/9.0;
        r               = zl/zb;
        *hicfcio        = ((1.0-r)*a2+r*a3)/zb;
        *dhicfcio_dw    = ((1.0-r)*x+r*a)*lnzb;
    } else {
        a               = z*z;
        a2              = 3.0+z-0.25*a+0.10*z*a;
        a3              = 2.0*z+0.75*a-0.20*a*z;
        *hicfcio        = (zb*a2+zl*a3)*w*w/6.0;
        *dhicfcio_dw    = (1+zl*w)*(1+z)*lnzb;
    }
}

// NEEDED TO CALCULATE WEIGHTED ICCR COLLECTOR MINORITY CHARGE
// INPUT:
//  z : zeta_b or zeta_l
//  w : normalized injection width
// OUTPUT:
//  hicfcto     : output
//  dhicfcto_dw : derivative of output wrt w
void HICFCT(double z, double w, double *hicfcto, double *dhicfcto_dw)
{
double a,lnz;
    a = z*w;
    lnz = log(1+z*w);
    if (a > 1.0e-6) {
        *hicfcto     = (a - lnz)/z;
        *dhicfcto_dw = a / (1.0 + a);
    } else {
        *hicfcto     = 0.5 * a * w;
        *dhicfcto_dw = a;
    }
}

// COLLECTOR CURRENT SPREADING CALCULATION
// collector minority charge incl. 2D/3D current spreading (TED 10/96)
// INPUT:
//  Ix                          : forward transport current component (itf)
//  I_CK                        : critical current
//  FFT_pcS                     : dependent on fthc and thcs (parameters)
// IMPLICIT INPUT:
//  ahc, latl, latb             : model parameters
//  vt                          : thermal voltage
// OUTPUT:
//  Q_fC, Q_CT: actual and ICCR (weighted) hole charge
//  T_fC, T_cT: actual and ICCR (weighted) transit time
//  Derivative dfCT_ditf not properly implemented yet
void HICQFC(HICUMinstance *here, HICUMmodel *model, double Ix, double I_CK, double FFT_pcS, double *Q_fC, double *Q_CT, double *T_fC, double *T_cT)
{
double FCa,FCrt,FCa_ck,FCdaick_ditf,FCz,FCxl,FCxb,FCln,FCa1,FCd_a,FCw,FCdw_daick,FCda1_dw;
double FCf3,FCdf2_dw,FCdf3_dw,FCd_f,FCz_1,FCf2,FCdfCT_dw,FCdf1_dw,FCw2,FCf1,FCf_ci,FCdfc_dw,FCdw_ditf,FCdfc_ditf,FCf_CT,FCdfCT_ditf;

    *Q_fC           = FFT_pcS*Ix;
    FCa             = 1.0-I_CK/Ix;
    FCrt            = sqrt(FCa*FCa+model->HICUMahc);
    FCa_ck          = 1.0-(FCa+FCrt)/(1.0+sqrt(1.0+model->HICUMahc));
    FCdaick_ditf    = (FCa_ck-1.0)*(1-FCa)/(FCrt*Ix);
    if(model->HICUMlatb > model->HICUMlatl) {
        FCz             = model->HICUMlatb-model->HICUMlatl;
        FCxl            = 1.0+model->HICUMlatl;
        FCxb            = 1.0+model->HICUMlatb;
        if(model->HICUMlatb > 0.01) {
            FCln            = log(FCxb/FCxl);
            FCa1            = exp((FCa_ck-1.0)*FCln);
            FCd_a           = 1.0/(model->HICUMlatl-FCa1*model->HICUMlatb);
            FCw             = (FCa1-1.0)*FCd_a;
            FCdw_daick      = -FCz*FCa1*FCln*FCd_a*FCd_a;
            FCa1            = log((1.0+model->HICUMlatb*FCw)/(1.0+model->HICUMlatl*FCw));
            FCda1_dw        = model->HICUMlatb/(1.0+model->HICUMlatb*FCw) - model->HICUMlatl/(1.0+model->HICUMlatl*FCw);
        } else {
            FCf1            = 1.0-FCa_ck;
            FCd_a           = 1.0/(1.0+FCa_ck*model->HICUMlatb);
            FCw             = FCf1*FCd_a;
            FCdw_daick      = -1.0*FCd_a*FCd_a*FCxb*FCd_a;
            FCa1            = FCz*FCw;
            FCda1_dw        = FCz;
        }
        FCf_CT          = 2.0/FCz;
        FCw2            = FCw*FCw;
        FCf1            = model->HICUMlatb*model->HICUMlatl*FCw*FCw2/3.0+(model->HICUMlatb+model->HICUMlatl)*FCw2/2.0+FCw;
        FCdf1_dw        = model->HICUMlatb*model->HICUMlatl*FCw2 + (model->HICUMlatb+model->HICUMlatl)*FCw + 1.0;
        HICFCI(model->HICUMlatb,model->HICUMlatl,FCw,&FCf2,&FCdf2_dw);
        HICFCI(model->HICUMlatl,model->HICUMlatb,FCw,&FCf3,&FCdf3_dw);
        FCf_ci          = FCf_CT*(FCa1*FCf1-FCf2+FCf3);
        FCdfc_dw        = FCf_CT*(FCa1*FCdf1_dw+FCda1_dw*FCf1-FCdf2_dw+FCdf3_dw);
        FCdw_ditf       = FCdw_daick*FCdaick_ditf;
        FCdfc_ditf      = FCdfc_dw*FCdw_ditf;
        if(model->HICUMflcomp == 0.0 || model->HICUMflcomp == 2.1) {
            HICFCT(model->HICUMlatb,FCw,&FCf2,&FCdf2_dw);
            HICFCT(model->HICUMlatl,FCw,&FCf3,&FCdf3_dw);
            FCf_CT          = FCf_CT*(FCf2-FCf3);
            FCdfCT_dw       = FCf_CT*(FCdf2_dw-FCdf3_dw);
            FCdfCT_ditf     = FCdfCT_dw*FCdw_ditf;
        } else {
            FCf_CT          = FCf_ci;
            FCdfCT_ditf     = FCdfc_ditf;
        }
    } else {
        if(model->HICUMlatb > 0.01) {
            FCd_a           = 1.0/(1.0+FCa_ck*model->HICUMlatb);
            FCw             = (1.0-FCa_ck)*FCd_a;
            FCdw_daick      = -(1.0+model->HICUMlatb)*FCd_a*FCd_a;
        } else {
            FCw             = 1.0-FCa_ck-FCa_ck*model->HICUMlatb;
            FCdw_daick      = -(1.0+model->HICUMlatb);
        }
        FCw2            = FCw*FCw;
        FCz             = model->HICUMlatb*FCw;
        FCz_1           = 1.0+FCz;
        FCd_f           = 1.0/(FCz_1);
        FCf_ci          = FCw2*(1.0+FCz/3.0)*FCd_f;
        FCdfc_dw        = 2.0*FCw*(FCz_1+FCz*FCz/3.0)*FCd_f*FCd_f;
        FCdw_ditf       = FCdw_daick*FCdaick_ditf;
        FCdfc_ditf      = FCdfc_dw*FCdw_ditf;
        if(model->HICUMflcomp == 0.0 || model->HICUMflcomp == 2.1) {
            if (FCz > 0.001) {
                FCf_CT          = 2.0*(FCz_1*log(FCz_1)-FCz)/(model->HICUMlatb*model->HICUMlatb*FCz_1);
                FCdfCT_dw       = 2.0*FCw*FCd_f*FCd_f;
            } else {
                FCf_CT          = FCw2*(1.0-FCz/3.0)*FCd_f;
                FCdfCT_dw       = 2.0*FCw*(1.0-FCz*FCz/3.0)*FCd_f*FCd_f;
            }
            FCdfCT_ditf     = FCdfCT_dw*FCdw_ditf;
        } else {
            FCf_CT          = FCf_ci;
            FCdfCT_ditf     = FCdfc_ditf;
        }
    }
    *Q_CT    = *Q_fC*FCf_CT*exp((FFdVc-model->HICUMvcbar)/here->HICUMvt);
    *Q_fC    = *Q_fC*FCf_ci*exp((FFdVc-model->HICUMvcbar)/here->HICUMvt);
    *T_fC    = FFT_pcS*exp((FFdVc-model->HICUMvcbar)/here->HICUMvt)*(FCf_ci+Ix*FCdfc_ditf)+ *Q_fC/here->HICUMvt*FFdVc_ditf;
    *T_cT    = FFT_pcS*exp((FFdVc-model->HICUMvcbar)/here->HICUMvt)*(FCf_CT+Ix*FCdfCT_ditf)+ *Q_CT/here->HICUMvt*FFdVc_ditf;
}

// TRANSIT-TIME AND STORED MINORITY CHARGE
// INPUT:
//  itf         : forward transport current
//  I_CK        : critical current
//  T_f         : transit time
//  Q_f         : minority charge / for low current
// IMPLICIT INPUT:
//  tef0, gtfe, fthc, thcs, ahc, latl, latb     : model parameters
// OUTPUT:
//  T_f         : transit time
//  Q_f         : minority charge / transient analysis
//  T_fT        : transit time
//  Q_fT        : minority charge / ICCR (transfer current)
//  Q_bf        : excess base charge
void HICQFF(HICUMinstance *here, HICUMmodel *model, double itf, double I_CK, double *T_f, double *Q_f, double *T_fT, double *Q_fT, double *Q_bf)
{
double FFitf_ick,FFdTef,FFdQef,FFib,FFfcbar,FFdib_ditf,FFdQbfb,FFdTbfb,FFic,FFw,FFdQfhc,FFdTfhc,FFdQcfc,FFdTcfc,FFdQcfcT,FFdTcfcT,FFdQbfc,FFdTbfc;

    if(itf < 1.0e-6*I_CK) {
        *Q_fT            = *Q_f;
        *T_fT            = *T_f;
        *Q_bf            = 0;
    } else {
        FFitf_ick = itf/I_CK;
        FFdTef  = here->HICUMtef0_t*exp(model->HICUMgtfe*log(FFitf_ick));
        FFdQef  = FFdTef*itf/(1+model->HICUMgtfe);
        if (model->HICUMicbar<0.05*(model->HICUMvlim/model->HICUMrci0)) {
            FFdVc = 0;
            FFdVc_ditf = 0;
        } else {
            FFib    = (itf-I_CK)/model->HICUMicbar;
            if (FFib < -1.0e10) {
                FFib = -1.0e10;
            }
            FFfcbar = (FFib+sqrt(FFib*FFib+model->HICUMacbar))/2.0;
            FFdib_ditf = FFfcbar/sqrt(FFib*FFib+model->HICUMacbar)/model->HICUMicbar;
            FFdVc = model->HICUMvcbar*exp(-1.0/FFfcbar);
            FFdVc_ditf = FFdVc/(FFfcbar*FFfcbar)*FFdib_ditf;
        }
        FFdQbfb = (1-model->HICUMfthc)*here->HICUMthcs_t*itf*(exp(FFdVc/here->HICUMvt)-1);
        FFdTbfb = FFdQbfb/itf+(1-model->HICUMfthc)*here->HICUMthcs_t*itf*exp(FFdVc/here->HICUMvt)/here->HICUMvt*FFdVc_ditf;
        FFic    = 1-1.0/FFitf_ick;
        FFw     = (FFic+sqrt(FFic*FFic+model->HICUMahc))/(1+sqrt(1+model->HICUMahc));
        FFdQfhc = here->HICUMthcs_t*itf*FFw*FFw*exp((FFdVc-model->HICUMvcbar)/here->HICUMvt);
        FFdTfhc = FFdQfhc*(1.0/itf*(1.0+2.0/(FFitf_ick*sqrt(FFic*FFic+model->HICUMahc)))+1.0/here->HICUMvt*FFdVc_ditf);
        if(model->HICUMlatb <= 0.0 && model->HICUMlatl <= 0.0) {
             FFdQcfc = model->HICUMfthc*FFdQfhc;
             FFdTcfc = model->HICUMfthc*FFdTfhc;
             FFdQcfcT = FFdQcfc;
             FFdTcfcT = FFdTcfc;
        } else {
             HICQFC(here,model,itf,I_CK,model->HICUMfthc*here->HICUMthcs_t,&FFdQcfc,&FFdQcfcT,&FFdTcfc,&FFdTcfcT);
        }
        FFdQbfc = (1-model->HICUMfthc)*FFdQfhc;
        FFdTbfc = (1-model->HICUMfthc)*FFdTfhc;
        *Q_fT    = here->HICUMhf0_t* *Q_f+FFdQbfb+FFdQbfc+here->HICUMhfe_t*FFdQef+here->HICUMhfc_t*FFdQcfcT;
        *T_fT    = here->HICUMhf0_t*(*T_f)+FFdTbfb+FFdTbfc+here->HICUMhfe_t*FFdTef+here->HICUMhfc_t*FFdTcfcT;
        *Q_f     = *Q_f+(FFdQbfb+FFdQbfc)+FFdQef+FFdQcfc;
        *T_f     = *T_f+(FFdTbfb+FFdTbfc)+FFdTef+FFdTcfc;
        *Q_bf    = FFdQbfb+FFdQbfc;
      }
}


// IDEAL DIODE (WITHOUT CAPACITANCE):
// conductance calculation not required
// INPUT:
//  IS, IST     : saturation currents (model parameter related)
//  UM1         : ideality factor
//  U           : branch voltage
// IMPLICIT INPUT:
//  vt          : thermal voltage
// OUTPUT:
//  Iz          : diode current
void HICDIO(double vt, double IS, double IST, double UM1, double U, double *Iz, double *Gz)
{
double DIOY, le, vtn;
    vtn = UM1*vt;
    DIOY = U/vtn;
    if (IS > 0.0) {
        if (DIOY > Dexp_lim) {
            le      = (1 + (DIOY - Dexp_lim));
            DIOY    = Dexp_lim;
            le      = le*exp(DIOY);
            *Iz     = IST*(le-1.0);
            *Gz     = IST*exp(Dexp_lim)/vtn;
        } else {
            le      = exp(DIOY);
            *Iz     = IST*(le-1.0);
            *Gz     = IST*le/vtn;
        }
        if(DIOY <= -14.0) {
            *Iz      = -IST;
            *Gz      = 0.0;
        }
    } else {
        *Iz      = 0.0;
        *Gz      = 0.0;
    }
}






