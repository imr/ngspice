/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

/*
 * This file defines the HICUM L2.4.0 model load function
 * Comments on the Code:
 * - We use dual numbers to calculate derivatives, this is readable and error proof.
 * - The code is targeted to be readable and maintainable, speed is sacrificed for this purpose.
 * - The verilog a code is available at the website of TU Dresden, Michael Schroeter's chair.
 * - lambda functions are used to calculate derivatives of larger Verilog Macros
 */

#include <cmath>
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#include "duals/dual"
#include "hicumL2.hpp"
#include "hicumL2temp.hpp"
#include <functional>

//ngspice header files written in C
#ifdef __cplusplus
extern "C"
{
#endif
#include "ngspice/typedefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "hicum2defs.h"
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#ifdef __cplusplus
}
#endif

using namespace duals;
#define e1 (dual<double>(0.0, 1.0))

//HICUM DEFINITIONS
#define VPT_thresh      1.0e2
#define Dexp_lim        80.0
#define Cexp_lim        80.0
#define DFa_fj          1.921812
#define RTOLC           1.0e-5
#define l_itmax         100
#define MIN_R           0.001


// IDEAL DIODE (WITHOUT CAPACITANCE):
// conductance calculation not required
// INPUT:
//  IS, IST     : saturation currents (model parameter related)
//  UM1         : ideality factor
//  U           : branch voltage
// IMPLICIT INPUT:
//  T           : Temperature
// OUTPUT:
//  Iz          : diode current
duals::duald HICDIO(duals::duald T, duals::duald IST, double UM1, duals::duald U)
{
duals::duald DIOY, le, vt;

    vt = CONSTboltz * T / CHARGE;
    DIOY = U/(UM1*vt);
    // le = exp(DIOY); // would be the best way... But stay close to HICUML2.va
    // return IST*(le-1.0);
    if (IST > 0.0) {
        if (DIOY > Dexp_lim) {
            le      = (1 + (DIOY - Dexp_lim));
            DIOY    = Dexp_lim;
            return IST*(le*exp(DIOY)-1.0);
        } else if (DIOY <= -14.0) {
            return -IST;
        } else {
            le      = exp(DIOY);
            return IST*(le-1.0);
        }
    } else {
        return 0.0;
    }
}

// DEPLETION CHARGE CALCULATION
// Hyperbolic smoothing used; no punch-through
// INPUT:
//  c_0		: zero-bias capacitance
//  u_d		: built-in voltage
//  z		: exponent coefficient
//  a_j		: control parameter for C peak value at high forward bias
//  U_cap	: voltage across junction
// IMPLICIT INPUT:
//  T		: Temperature
// OUTPUT:
//  Qz		: depletion Charge
//  C		: depletion capacitance
void QJMODF(duals::duald T, duals::duald c_0, duals::duald u_d, double z, duals::duald a_j, duals::duald U_cap, duals::duald * C, duals::duald * Qz)
{
    duals::duald DFV_f, DFv_e, DFs_q, DFs_q2, DFv_j, DFdvj_dv, DFQ_j, DFC_j1, DFb, vt;
    vt = CONSTboltz * T / CHARGE; 
    if (c_0 > 0.0) {
        DFV_f	 = u_d*(1.0-exp(-log(a_j)/z));
        DFv_e	 = (DFV_f-U_cap)/vt;
        DFs_q	 = sqrt(DFv_e*DFv_e+DFa_fj);
        DFs_q2	 = (DFv_e+DFs_q)*0.5;
        DFv_j	 = DFV_f-vt*DFs_q2;
        DFdvj_dv = DFs_q2/DFs_q;
        DFb	     = log(1.0-DFv_j/u_d);
        DFC_j1   = c_0*exp(-z*DFb)*DFdvj_dv;
        *C		 = DFC_j1+a_j*c_0*(1.0-DFdvj_dv);
        DFQ_j	 = c_0*u_d*(1.0-exp(DFb*(1.0-z)))/(1.0-z);
        *Qz	     = DFQ_j+a_j*c_0*(U_cap-DFv_j);
     } else {
        *C       = 0.0;
        *Qz	     = 0.0;
     }
}

// DEPLETION CHARGE CALCULATION CONSIDERING PUNCH THROUGH
// smoothing of reverse bias region (punch-through)
// and limiting to a_j=Cj,max/Cj0 for forward bias.
// Important for base-collector and collector-substrate junction
// INPUT:
//  c_0		: zero-bias capacitance
//  u_d		: built-in voltage
//  z 		: exponent coefficient
//  a_j		: control parameter for C peak value at high forward bias
//  v_pt	: punch-through voltage (defined as qNw^2/2e)
//  U_cap	: voltage across junction
// IMPLICIT INPUT:
//  VT		: thermal voltage
// OUTPUT:
//  Qz		: depletion charge
//  C		: depletion capacitance
void QJMOD(duals::duald T, duals::duald c_0, duals::duald u_d, double z, double a_j, duals::duald v_pt, duals::duald U_cap, duals::duald * C, duals::duald * Qz)
{
    duals::duald DQ_j1, DQ_j2, DQ_j3, DC_j1, DC_j2, DC_j3, De_1, De_2, Dzr1, DCln1, DCln2, Dz1, Dv_j1, Dv_j2, De, Da, Dv_r, Dv_j4, Dv_e, DC_c, DC_max, DV_f, Dv_p, Dz_r, vt;
    vt = CONSTboltz * T / CHARGE;
    if (c_0 > 0.0){
        Dz_r	= z/4.0;
        Dv_p	= v_pt-u_d;
        DV_f	= u_d*(1.0-exp(-log(a_j)/z));
        DC_max	= a_j*c_0;
        DC_c	= c_0*exp((Dz_r-z)*log(v_pt/u_d));
        Dv_e	= (DV_f-U_cap)/vt;
        if(Dv_e < Cexp_lim) {
            De	    = exp(Dv_e);
            De_1	= De/(1.0+De);
            Dv_j1	= DV_f-vt*log(1.0+De);
        } else {
            De_1	= 1.0;
            Dv_j1	= U_cap;
        }
        Da	    = 0.1*Dv_p+4.0*vt;
        Dv_r	= (Dv_p+Dv_j1)/Da;
        if(Dv_r < Cexp_lim){
            De	    = exp(Dv_r);
            De_2	= De/(1.0+De);
            Dv_j2	= -Dv_p+Da*(log(1.0+De)-exp(-(Dv_p+DV_f)/Da));
        } else {
            De_2	= 1.0;
            Dv_j2	= Dv_j1;
        }
        Dv_j4	= U_cap-Dv_j1;
        DCln1	= log(1.0-Dv_j1/u_d);
        DCln2	= log(1.0-Dv_j2/u_d);
        Dz1	    = 1.0-z;
        Dzr1	= 1.0-Dz_r;
        DC_j1	= c_0*exp(DCln2*(-z))*De_1*De_2;
        DC_j2	= DC_c*exp(DCln1*(-Dz_r))*(1.0-De_2);
        DC_j3	= DC_max*(1.0-De_1);
        *C		= DC_j1+DC_j2+DC_j3;
        DQ_j1	= c_0*(1.0-exp(DCln2*Dz1))/Dz1;
        DQ_j2	= DC_c*(1.0-exp(DCln1*Dzr1))/Dzr1;
        DQ_j3	= DC_c*(1.0-exp(DCln2*Dzr1))/Dzr1;
        *Qz	    = (DQ_j1+DQ_j2-DQ_j3)*u_d+DC_max*Dv_j4;
     } else {
        *C	    = 0.0;
        *Qz	    = 0.0;
     }
}

// A CALCULATION NEEDED FOR COLLECTOR MINORITY CHARGE FORMULATION
// INPUT:
//  zb,zl       : zeta_b and zeta_l (model parameters, TED 10/96)
//  w           : normalized injection width
// OUTPUT:
// hicfcio      : function of equation (2.1.17-10)
void HICFCI(double zb, double zl, duals::duald w, duals::duald * hicfcio, duals::duald * dhicfcio_dw)
{
    duals::duald a, a2, a3, r, lnzb, x, z;
    z       = zb*w;
    lnzb    = log(1+zb*w);
    if(z > 1.0e-6){
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
        *hicfcio         = (zb*a2+zl*a3)*w*w/6.0;
        *dhicfcio_dw     = (1+zl*w)*(1+z)*lnzb;
     }
}

// NEEDED TO CALCULATE WEIGHTED ICCR COLLECTOR MINORITY CHARGE
// INPUT:
//  z : zeta_b or zeta_l
//  w : normalized injection width
// OUTPUT:
//  hicfcto     : output
//  dhicfcto_dw : derivative of output wrt w
void HICFCT(double z, duals::duald w, duals::duald * hicfcto, duals::duald *dhicfcto_dw)
{
    duals::duald a, lnz;
    a = z*w;
    lnz = log(1+z*w);
    if (a > 1.0e-6){
        *hicfcto     = (a - lnz)/z;
        *dhicfcto_dw = a / (1.0 + a);
    } else {
        *hicfcto     = 0.5 * a * w;
        *dhicfcto_dw = a;
    }
}

// DEPLETION CHARGE & CAPACITANCE CALCULATION SELECTOR
// Dependent on junction punch-through voltage
// Important for collector related junctions
void HICJQ(duals::duald T, duals::duald c_0, duals::duald u_d, double z, duals::duald v_pt, duals::duald U_cap, duals::duald * C,duals::duald * Qz)
{
    if(v_pt.rpart() < VPT_thresh){
        QJMOD(T,c_0,u_d,z,2.4,v_pt,U_cap,C,Qz);
    } else {
        QJMODF(T,c_0,u_d,z,2.4,U_cap,C,Qz);
    }
}

duals::duald calc_hjei_vbe(duals::duald Vbiei, duals::duald T, HICUMinstance * here, HICUMmodel * model){
    //calculates hje_vbe
    //wrapping in a routine allows easy calculation of derivatives with dual numbers
    duals::duald vj, vj_z, vt, vdei_t, hjei0_t, ahjei_t;
    if (model->HICUMahjei == 0.0){
        return model->HICUMhjei;
    }else{
        double T_dpart = T.dpart();
        vt  = CONSTboltz * T   / CHARGE;
        vdei_t = here->HICUMvdei_t.rpart;
        hjei0_t = here->HICUMhjei0_t.rpart;
        ahjei_t = here->HICUMahjei_t.rpart;
        if (T_dpart!=0.0){
            vdei_t.dpart(here->HICUMvdei_t.dpart);
            hjei0_t.dpart(here->HICUMhjei0_t.dpart);
            ahjei_t.dpart(here->HICUMahjei_t.dpart);
        }
        //vendhjei = vdei_t*(1.0-exp(-ln(ajei_t)/z_h));
        vj = (vdei_t-Vbiei)/(model->HICUMrhjei*vt);
        vj = vdei_t-model->HICUMrhjei*vt*(vj+sqrt(vj*vj+DFa_fj))*0.5;
        vj = (vj-vt)/vt;
        vj = vt*(1.0+(vj+sqrt(vj*vj+DFa_fj))*0.5);
        vj_z = (1.0-exp(model->HICUMzei*log(1.0-vj/vdei_t)))*ahjei_t;
        return hjei0_t*(exp(vj_z)-1.0)/vj_z;
    }
}


void hicum_diode(duals::duald T, dual_double IS, double UM1, double U, double *Iz, double *Gz, double *Tz)
{
    // T is T_dev + e1*T_dev_Vrth
    //wrapper for hicum diode equation that also generates derivatives
    duals::duald result = 0;

    duals::duald is_t = IS.rpart;
    result = HICDIO(T.rpart(), is_t, UM1, U+e1);
    *Iz    = result.rpart();
    *Gz    = result.dpart(); //derivative for U
    is_t   = IS.rpart + e1*IS.dpart;
    result = HICDIO(T, is_t, UM1, U);
    *Tz    = result.dpart(); //derivative for T
}

void hicum_qjmodf(duals::duald T, dual_double c_0, dual_double u_d, double z, dual_double a_j, double U_cap, double *C, double *C_dU, double *C_dT, double *Qz, double *Qz_dU, double *Qz_dT)
{
    //wrapper for QJMODF that also generates derivatives
    duals::duald Cresult = 0;
    duals::duald Qresult = 0;
    duals::duald c_0_t = c_0.rpart;
    duals::duald u_d_t = u_d.rpart;
    duals::duald a_j_t = a_j.rpart;
    QJMODF(T.rpart(), c_0_t, u_d_t, z, a_j_t, U_cap+e1, &Cresult, &Qresult);
    *C     = Cresult.rpart();
    *C_dU  = Cresult.dpart();
    *Qz    = Qresult.rpart();
    *Qz_dU = Qresult.dpart();

    c_0_t.dpart(c_0.dpart);
    u_d_t.dpart(u_d.dpart);
    a_j_t.dpart(a_j.dpart);
    QJMODF(T, c_0_t, u_d_t, z, a_j_t, U_cap, &Cresult, &Qresult);
    *Qz_dT = Qresult.dpart();
    *C_dT  = Cresult.dpart();
}

void hicum_HICJQ(duals::duald T, dual_double c_0, dual_double u_d, double z, dual_double v_pt, double U_cap, double * C, double * C_dU, double * C_dT, double * Qz, double * Qz_dU, double * Qz_dT)
{
    //wrapper for HICJQ that also generates derivatives
    duals::duald Cresult = 0;
    duals::duald Qresult = 0;
    duals::duald c_0_t = c_0.rpart;
    duals::duald u_d_t = u_d.rpart;
    duals::duald v_pt_t = v_pt.rpart;
    HICJQ(T.rpart(), c_0_t, u_d_t, z, v_pt_t, U_cap+e1, &Cresult, &Qresult);
    *C     = Cresult.rpart();
    *C_dU  = Cresult.dpart();
    *Qz    = Qresult.rpart();
    *Qz_dU = Qresult.dpart();

    c_0_t.dpart(c_0.dpart);
    u_d_t.dpart(u_d.dpart);
    v_pt_t.dpart(v_pt.dpart);
    HICJQ(T, c_0_t, u_d_t, z, v_pt_t, U_cap, &Cresult, &Qresult);
    *Qz_dT = Qresult.dpart();
    *C_dT  = Cresult.dpart();
}

int
HICUMload(GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMmodel *model = (HICUMmodel*)inModel;
    HICUMinstance *here;

    //Declaration of variables

    double cbcpar1,cbcpar2,cbepar2,cbepar1,Oich,Otbhrec;

    //Charges, capacitances and currents
    double Qjci,Qjei,Qjep;
    double Qdei,Qrbi;
    double Qdei_Vbiei, Qdei_Vbici, Qdei_dT;
    double it,ibei,irei,ibci,ibep,irep,ibh_rec;
    double ibet,iavl,iavl_dT,iavl_Vbiei,iavl_Vbici;
    double ijbcx,ijbcx_dT,ijbcx_Vbpci,ijsc,ijsc_Vsici,ijsc_Vrth,Qjs,Qscp,HSI_Tsu,Qdsu;
    double HSI_Tsu_Vbpci, HSI_Tsu_Vsici, HSI_Tsu_dT;
    double Qdsu_Vbpci, Qdsu_Vsici, Qdsu_dT;
    duals::duald result_Qdsu, result_HSI_TSU;
    double Qscp_Vsc, Qscp_dT;
    double Cscp_Vsc, Cscp_dT;

    //Base resistance and self-heating power
    double rbi=0.0,pterm,pterm_dT;

    //Model initialization
    double C_1;

    //Model evaluation
    double Crbi,Cjci,Cjei,Cjep,Cscp;
    double Cjs, Cjs_dT;
    double Cjcx_i, Cjcx_i_Vbci, Cjcx_i_dT;
    double Cjcx_ii, Cjcx_ii_Vbpci, Cjcx_ii_dT;
    double Qjcx_i , Qjcx_i_Vbci  , Qjcx_i_dT ;
    double Qjcx_ii, Qjcx_ii_Vbpci, Qjcx_ii_dT;
    double Qjs_Vsici, Qjs_dT;

    double itf,itr,Tf,Tr,a_bpt,Q_0;
    double itf_Vbiei, itf_Vbici, itf_dT;
    double itr_Vbiei, itr_Vbici, itr_dT;
    double Tf_Vbiei, Tf_Vbici, Tf_dT;
    double it_Vbiei, it_Vbici, it_dT;
    double Qf_Vbiei, Qf_Vbici, Qf_dT;
    double Qr_Vbiei, Qr_Vbici, Qr_dT;
    duals::duald result_itf, result_itr, result_Qf, result_Qr, result_Q_bf, result_a_h, result_Q_p, result_Tf; //intermediate variables when calling void dual functions
    double T_f0, Q_p, a_h;
    double Q_bf, Q_bf_Vbiei, Q_bf_Vbici, Q_bf_dT;
    double Q_pT, Q_pT_dVbiei, Q_pT_dVbici, Q_pT_dT;
    double Qf, Cdei, Qr, Cdci;
    double Cdei_Vbiei, Cdei_Vbici, Cdei_Vrth;
    double Cdci_Vbiei, Cdci_Vbici, Cdci_Vrth;
    double Crbi_Vbiei, Crbi_Vbici, Crbi_Vrth;
    double ick, ick_Vciei, ick_dT;

    //NQS
    double Ixf1,Ixf2,Qxf1,Qxf2;
    double Ixf1_Vxf1, Ixf1_Vxf2, Ixf1_ditf, Ixf1_dTf, Ixf1_Vbiei, Ixf1_Vbici, Ixf1_dT;
    double Ixf2_Vxf1, Ixf2_Vbiei, Ixf2_Vbici, Ixf2_dT;
    double Ixf_Vxf, Ixf_Tf,Ixf_Qdei,Ixf_Vbiei, Ixf_Vbici, Ixf_dT;
    double Ixf2_Vxf2, Ixf2_dTf;
    double Itxf, Qdeix;
    double Itxf_Vxf2, Itxf_Vbici, Itxf_Vbiei, Itxf_dT;
    double Qdeix_Vxf, Qdeix_dT, Qdeix_Vbici, Qdeix_Vbiei;
    double Qxf, Ixf, Vxf;
    double Vxf1, Vxf2;

    double hjei_vbe;

    double Vbiei, Vbici, Vciei, Vbpei, Vbpbi, Vbpci, Vsici, Vbci, Vsc;
    double Vbici_temp, Vaval;

    //Model flags
    int use_aval;

    //helpers for ngspice implementation
    duals::duald result;

    //end of variables

#ifndef PREDICTOR
    double xfact;
#endif
    double delvbiei, delvbici, delvbpei, delvbpbi, delvbpci, delvsici, delvbbp, delveie, delvciei, delvcic, delvxf2;
    double ibieihat;
    double ibpeihat;
    double icieihat;
    double ibicihat;
    double ibpcihat;
    double ibpbihat;
    double isicihat;
    double ibpsihat;
    double ithhat;
    double ceq, geq;
    double rhs_current;
    int icheck=1;
    int ichk1, ichk2, ichk3, ichk4, ichk5, ichk6;
    int error;
    double Vbe, Vcic, Vbbp, Veie, Vsis, Vbpe;

    double Ibiei, Ibiei_Vbiei, Ibiei_Vxf=0.0, Ibiei_Vbici;
    double Ibici, Ibici_Vbici, Ibici_Vbiei;
    double Ibpei, Ibpei_Vbpei;
    double Ibpci, Ibpci_Vbpci;
    double Isici, Isici_Vsici;//
    double Isc=0.0, Isc_Vsc=0.0;
    double Iciei, Iciei_Vbiei, Iciei_Vbici, Iciei_Vrth, Iciei_Vxf2;
    double Ibbp_Vbbp=0.0;
    double Isis_Vsis;
    double Ieie, Ieie_Veie=0.0;
    double Ibpbi, Ibpbi_Vbpbi, Ibpbi_Vbici, Ibpbi_Vbiei;
    double Ibpsi, Ibpsi_Vbpci, Ibpsi_Vsici, Ibpsi_Vrth;
    double Icic_Vcic=0.0;
    double Ibci=0.0, Ibci_Vbci=0.0; 
    double hjei_vbe_Vbiei, hjei_vbe_dT, ibet_Vbpei, ibet_dT, ibet_Vbiei, ibh_rec_Vbiei, ibh_rec_dT, ibh_rec_Vbici;
    double irei_Vbiei, irei_dT;
    double ibep_Vbpei, ibep_dT;
    double irep_Vbpei, irep_dT, rbi_dT, rbi_Vbiei, rbi_Vbici;
    double ibei_Vbiei, ibei_dT;
    double ibci_Vbici, ibci_dT;
    double Q_0_Vbiei, Q_0_Vbici, Q_0_dT;

    double Temp;
    double Tdev_Vrth; //derivative device temperature to Vrth

    //below variable has a real part equal to the device temperature and a dual part equal to dTdev/dVrth
    //this is necessary, since for some Vrth, HICUM sets Tdev constant (eg very high self heating beyond 300K)
    //then, dTdev/dVrth. Else it is equal to 1.
    duals::duald Temp_dual; 

    double Cjei_Vbiei,Cjci_Vbici,Cjep_Vbpei,Cjep_dT,Cjs_Vsici;
    double Cjei_dT, Cjci_dT;
    double Qjei_Vbiei, Qjei_dT, Qjci_Vbici, Qjci_dT;
    double T_f0_Vbici,T_f0_dT;
    double Qbepar1;
    double Qbepar2;
    double Qbcpar1;
    double Qbcpar2;
    double Qsu;
    double Qcth;
    double Qcth_Vrth;

    double Qrbi_Vbpbi;
    double Qrbi_Vrth;
    double Qrbi_Vbiei;
    double Qrbi_Vbici;
    double Qjep_Vbpei,Qjep_dT;
    double Qbepar1_Vbe;
    double Qbepar2_Vbpe;
    double Qbcpar1_Vbci;
    double Qbcpar2_Vbpci;
    double Qsu_Vsis;

    double cqbepar1, gqbepar1;
    double cqbepar2, gqbepar2;
    double cqbcpar1, gqbcpar1;
    double cqbcpar2, gqbcpar2;
    double cqsu, gqsu;

    double Qxf_Vxf, Qxf1_Vxf1, Qxf2_Vxf2;

    double Ith=0.0, Vrth=0.0, Icth, Icth_Vrth, delvrth;

    double Ibiei_Vrth;
    double Ibici_Vrth;
    double Ibpei_Vrth;
    double Ibpci_Vrth;
    double Isici_Vrth;
    double Ibpbi_Vrth;
    double Ieie_Vrth;
    double Icic_Vrth=0.0;
    double Irth_Vrth=0.0;
    double Ibbp_Vrth=0.0;

    double Ith_Vrth=0.0;
    double Ith_Vciei=0.0;
    double Ith_Vbiei=0.0;
    double Ith_Vbici=0.0;
    double Ith_Vbpei=0.0;
    double Ith_Vbpci=0.0;

    double Ith_Vsici=0.0;
    double Ith_Vbpbi=0.0;
    double Ith_Veie=0.0;
    double Ith_Vcic=0.0;
    double Ith_Vbbp=0.0;

    // COLLECTOR CURRENT SPREADING CALCULATION
    // collector minority charge incl. 2D/3D current spreading (TED 10/96)
    // INPUT:
    //  Ix                          : forward transport current component (itf)
    //  I_CK                        : critical current
    //  FFT_pcS                     : dependent on fthc and thcs (parameters)
    // IMPLICIT INPUT:
    //  ahc, latl, latb             : model parameters
    //  VT                          : thermal voltage
    // OUTPUT:
    //  Q_fC, Q_CT: actual and ICCR (weighted) hole charge
    //  T_fC, T_cT: actual and ICCR (weighted) transit time
    //  Derivative dfCT_ditf not properly implemented yet
    std::function<void (duals::duald, duals::duald, duals::duald, duals::duald, duals::duald*, duals::duald*, duals::duald*, duals::duald*)> HICQFC = [&](duals::duald T, duals::duald Ix, duals::duald I_CK, duals::duald FFT_pcS, duals::duald * Q_fC, duals::duald * Q_CT, duals::duald * T_fC, duals::duald * T_cT)
    {
        duals::duald FCln, FCa, FCa1, FCd_a, FCw, FCdw_daick, FCda1_dw, FCf_ci, FCdfCT_ditf, FCw2, FCz, FCdfc_dw, FFdVc_ditf, FCf_CT, FCf1, FCf2, FCrt;
        duals::duald FCa_ck, FCdaick_ditf, FCxl, FCxb, FCdf1_dw, FCz_1, FCf3, FCdf2_dw, FCdf3_dw, FCdw_ditf, FCdfc_ditf;
        duals::duald FCdfCT_dw, FCd_f, FFdVc;

        duals::duald vt;

        vt = CONSTboltz * T / CHARGE;

        *Q_fC           = FFT_pcS*Ix;
        FCa             = 1.0-I_CK/Ix;
        FCrt            = sqrt(FCa*FCa+model->HICUMahc);
        FCa_ck          = 1.0-(FCa+FCrt)/(1.0+sqrt(1.0+model->HICUMahc));
        FCdaick_ditf    = (FCa_ck-1.0)*(1-FCa)/(FCrt*Ix);
        if(model->HICUMlatb > model->HICUMlatl){
            FCz             = model->HICUMlatb-model->HICUMlatl;
            FCxl            = 1.0+model->HICUMlatl;
            FCxb            = 1.0+model->HICUMlatb;
            if(model->HICUMlatb > 0.01){
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
            if(model->HICUMflcomp < 2.3) {
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
            if(model->HICUMflcomp < 2.3){
                if (FCz > 0.001){
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
        *Q_CT    = *Q_fC*FCf_CT*exp((FFdVc-model->HICUMvcbar)/vt);
        *Q_fC    = *Q_fC*FCf_ci*exp((FFdVc-model->HICUMvcbar)/vt);
        *T_fC    = FFT_pcS*exp((FFdVc-model->HICUMvcbar)/vt)*(FCf_ci+Ix*FCdfc_ditf) +*Q_fC/vt*FFdVc_ditf;
        *T_cT    = FFT_pcS*exp((FFdVc-model->HICUMvcbar)/vt)*(FCf_CT+Ix*FCdfCT_ditf)+*Q_CT/vt*FFdVc_ditf;
    };

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
    //  Q_f         : minority charge  transient analysis
    //  T_fT        : transit time
    //  Q_fT        : minority charge  ICCR (transfer current)
    //  Q_bf        : excess base charge
    std::function<void (duals::duald, duals::duald, duals::duald, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*)> HICQFF = [&](duals::duald T, duals::duald itf, duals::duald I_CK, duals::duald * T_f, duals::duald * Q_f, duals::duald * T_fT, duals::duald * Q_fT, duals::duald * Q_bf)
    {
        duals::duald FFitf_ick, FFdTef, FFdQef, FFdVc, FFdVc_ditf, FFib, FFfcbar, FFdib_ditf;
        duals::duald vt,tef0_t,thcs_t,hf0_t,hfe_t,hfc_t;
        duals::duald FFdQbfb, FFdTbfb, FFdQfhc, FFdTfhc, FFdQcfc,FFdTcfc, FFdQbfc,FFdTbfc;
        duals::duald FFdQcfcT, FFic, FFw, FFdTcfcT;
        double T_dpart = T.dpart();
        vt = CONSTboltz * T / CHARGE;
        tef0_t = here->HICUMtef0_t.rpart;
        thcs_t = here->HICUMthcs_t.rpart;
        hf0_t = here->HICUMhf0_t.rpart;
        hfe_t = here->HICUMhfe_t.rpart;
        hfc_t = here->HICUMhfc_t.rpart;
        if (T_dpart!=0.0){
            tef0_t.dpart(here->HICUMtef0_t.dpart);
            thcs_t.dpart(here->HICUMthcs_t.dpart);
            hf0_t.dpart(here->HICUMhf0_t.dpart);
            hfe_t.dpart(here->HICUMhfe_t.dpart);
            hfc_t.dpart(here->HICUMhfc_t.dpart);
        }

        if(itf < 1.0e-6*I_CK){
            *Q_fT            = *Q_f;
            *T_fT            = *T_f;
            *Q_bf            = 0;
        } else {
            FFitf_ick = itf/I_CK;
            FFdTef  = tef0_t*exp(model->HICUMgtfe*log(FFitf_ick));
            FFdQef  = FFdTef*itf/(1+model->HICUMgtfe);
            if (here->HICUMicbar_scaled<0.05*(model->HICUMvlim/here->HICUMrci0_scaled)) {
                FFdVc = 0;
                FFdVc_ditf = 0;
            } else {
                FFib    = (itf-I_CK)/here->HICUMicbar_scaled;
                if (FFib < -1.0e10) {
                    FFib = -1.0e10;
                }
                FFfcbar    = (FFib+sqrt(FFib*FFib+model->HICUMacbar))/2.0;
                FFdib_ditf = FFfcbar/sqrt(FFib*FFib+model->HICUMacbar)/here->HICUMicbar_scaled;
                FFdVc      = model->HICUMvcbar*exp(-1.0/FFfcbar);
                FFdVc_ditf = FFdVc/(FFfcbar*FFfcbar)*FFdib_ditf;
            }
            FFdQbfb = (1-model->HICUMfthc)*thcs_t*itf*(exp(FFdVc/vt)-1);
            FFdTbfb = FFdQbfb/itf+(1-model->HICUMfthc)*thcs_t*itf*exp(FFdVc/vt)/vt*FFdVc_ditf;
            FFic    = 1-1.0/FFitf_ick;
            FFw     = (FFic+sqrt(FFic*FFic+model->HICUMahc))/(1+sqrt(1+model->HICUMahc));
            FFdQfhc = thcs_t*itf*FFw*FFw*exp((FFdVc-model->HICUMvcbar)/vt);
            FFdTfhc = FFdQfhc*(1.0/itf*(1.0+2.0/(FFitf_ick*sqrt(FFic*FFic+model->HICUMahc)))+1.0/vt*FFdVc_ditf);
            if(model->HICUMlatb <= 0.0 && model->HICUMlatl <= 0.0){
                FFdQcfc = model->HICUMfthc*FFdQfhc;
                FFdTcfc = model->HICUMfthc*FFdTfhc;
                FFdQcfcT = FFdQcfc;
                FFdTcfcT = FFdTcfc;
            } else {
                HICQFC(T, itf,I_CK,model->HICUMfthc*thcs_t,&FFdQcfc,&FFdQcfcT,&FFdTcfc,&FFdTcfcT);
            }
            FFdQbfc = (1-model->HICUMfthc)*FFdQfhc;
            FFdTbfc = (1-model->HICUMfthc)*FFdTfhc;
            *Q_fT	= hf0_t*(*Q_f)+FFdQbfb+FFdQbfc+hfe_t*FFdQef+hfc_t*FFdQcfcT;
            *T_fT	= hf0_t*(*T_f)+FFdTbfb+FFdTbfc+hfe_t*FFdTef+hfc_t*FFdTcfcT;
            *Q_f	= *Q_f+(FFdQbfb+FFdQbfc)+FFdQef+FFdQcfc;
            *T_f 	= *T_f+(FFdTbfb+FFdTbfc)+FFdTef+FFdTcfc;
            *Q_bf   = FFdQbfb+FFdQbfc;
        }
    };
    //Hole charge at low bias
    std::function<duals::duald (duals::duald, duals::duald, duals::duald, duals::duald)> calc_Q_0 = [&](duals::duald T, duals::duald Qjei, duals::duald Qjci, duals::duald hjei_vbe){
        duals::duald Q_0, b_q, Q_bpt, qp0_t;
        qp0_t = here->HICUMqp0_t.rpart;
        double T_dpart = T.dpart();
        if (T_dpart!=0.0){
            qp0_t.dpart(here->HICUMqp0_t.dpart);
        }
        a_bpt   = 0.05;
        Q_0     = qp0_t + hjei_vbe*Qjei + model->HICUMhjci*Qjci;
        Q_bpt   = a_bpt*qp0_t;
        b_q     = Q_0/Q_bpt-1;
        Q_0     = Q_bpt*(1+(b_q +sqrt(b_q*b_q+1.921812))/2);
        return Q_0;
    };

    std::function<duals::duald (duals::duald, duals::duald)> calc_T_f0 = [&](duals::duald T, duals::duald Vbici){
        //Transit time calculation at low current density
        duals::duald vt, vdci_t, cjci0_t, t0_t;
        duals::duald cV_f,cv_e,cs_q,cs_q2,cv_j,cdvj_dv,Cjcit,cc;
        double T_dpart = T.dpart();

        vt = CONSTboltz * T / CHARGE;
        vdci_t = here->HICUMvdci_t.rpart;
        cjci0_t = here->HICUMcjci0_t.rpart;
        t0_t = here->HICUMt0_t.rpart;
        if (T_dpart!=0.0){
            vdci_t.dpart(here->HICUMvdci_t.dpart);
            cjci0_t.dpart(here->HICUMcjci0_t.dpart);
            t0_t.dpart(here->HICUMt0_t.dpart);
        }
        if(here->HICUMcjci0_t.rpart > 0.0){ // CJMODF
            cV_f    = vdci_t*(1.0-exp(-log(2.4)/model->HICUMzci));
            cv_e    = (cV_f-Vbici)/vt;
            cs_q    = sqrt(cv_e*cv_e+1.921812);
            cs_q2   = (cv_e+cs_q)*0.5;
            cv_j    = cV_f-vt*cs_q2;
            cdvj_dv = cs_q2/cs_q;
            Cjcit   = cjci0_t*exp(-model->HICUMzci*log(1.0-cv_j/vdci_t))*cdvj_dv+2.4*cjci0_t*(1.0-cdvj_dv);
        } else {
            Cjcit   = 0.0;
        }
        if(Cjcit > 0.0) {
            cc      = cjci0_t/Cjcit;
        } else {
            cc      = 1.0;
        }
        return t0_t+model->HICUMdt0h*(cc-1.0)+model->HICUMtbvl*(1/cc-1.0);
    };
    std::function<duals::duald (duals::duald, duals::duald)> calc_ick = [&](duals::duald T, duals::duald Vciei){
        duals::duald ick, vces_t, rci0_t, vlim_t, Orci0_t;
        duals::duald Ovpt,a,d1,vceff,a1,a11,Odelck,ick1,ick2,ICKa, vc, vt;
        double T_dpart = T.dpart();

        vces_t = here->HICUMvces_t.rpart;
        rci0_t = here->HICUMrci0_t.rpart;
        vlim_t = here->HICUMvlim_t.rpart;
        if (T_dpart!=0.0){
            vces_t.dpart(here->HICUMvces_t.dpart);
            rci0_t.dpart(here->HICUMrci0_t.dpart);
            vlim_t.dpart(here->HICUMvlim_t.dpart);
        }
        //Effective collector voltage
        vc      = Vciei-vces_t;
        vt      = CONSTboltz * T / CHARGE;

        //Inverse of low-field internal collector resistance: needed in HICICK
        Orci0_t = 1.0/rci0_t;

        //Critical current for onset of high-current effects
        //begin : HICICK
            Ovpt    = 1.0/model->HICUMvpt;
            a       = vc/vt;
            d1      = a-1;
            vceff   = (1.0+((d1+sqrt(d1*d1+1.921812))/2))*vt;
            // a       = vceff/vlim_t;
            // ick     = vceff*Orci0_t/sqrt(1.0+a*a);
            // ICKa    = (vceff-vlim_t)*Ovpt;
            // ick     = ick*(1.0+0.5*(ICKa+sqrt(ICKa*ICKa+1.0e-3)));

            a1       = vceff/vlim_t;
            a11      = vceff*Orci0_t;
            Odelck   = 1/model->HICUMdelck;
            ick1     = exp(Odelck*log(1+exp(model->HICUMdelck*log(a1))));
            ick2     = a11/ick1;
            ICKa     = (vceff-vlim_t)*Ovpt;
            ick      = ick2*(1.0+0.5*(ICKa+sqrt(ICKa*ICKa+model->HICUMaick)));
            return ick;

        //end
    };


    std::function<duals::duald (duals::duald, duals::duald, duals::duald)> calc_ibet = [&](duals::duald Vbiei, duals::duald Vbpei, duals::duald T){
        //Tunneling current
        duals::duald ibet;
        if (here->HICUMibets_scaled > 0 && (Vbpei <0.0 || Vbiei < 0.0)){ //begin : HICTUN
            duals::duald pocce,czz, cje0_t, vde_t, ibets_t, abet_t;
            double T_dpart = T.dpart();
            ibets_t = here->HICUMibets_t.rpart;
            abet_t = here->HICUMabet_t.rpart;
            if (T_dpart!=0.0){
                abet_t.dpart(here->HICUMabet_t.dpart);
                ibets_t.dpart(here->HICUMibets_t.dpart);
            }
            if(model->HICUMtunode==1 && here->HICUMcjep0_t.rpart > 0.0 && here->HICUMvdep_t.rpart >0.0){
                cje0_t = here->HICUMcjep0_t.rpart;
                vde_t = here->HICUMvdep_t.rpart;
                if (T_dpart!=0.0){
                    cje0_t.dpart(here->HICUMcjep0_t.dpart);
                    vde_t.dpart(here->HICUMvdep_t.dpart);
                }
                pocce   = exp((1-1/model->HICUMzep)*log(Cjep/cje0_t));
                czz     = -(Vbpei/vde_t)*ibets_t*pocce;
                ibet    = czz*exp(-abet_t/pocce);
            } else if (model->HICUMtunode==0 && here->HICUMcjei0_t.rpart > 0.0 && here->HICUMvdei_t.rpart >0.0){
                cje0_t = here->HICUMcjei0_t.rpart;
                vde_t = here->HICUMvdei_t.rpart;
                if (T_dpart!=0.0){
                    cje0_t.dpart(here->HICUMcjei0_t.dpart);
                    vde_t.dpart(here->HICUMvdei_t.dpart);
                }
                pocce   = exp((1-1/model->HICUMzei)*log(Cjei/cje0_t));
                czz     = -(Vbiei/vde_t)*ibets_t*pocce;
                ibet    = czz*exp(-abet_t/pocce);
            } else {
                ibet    = 0.0;
            }
        } else {
            ibet    = 0.0;
        }
        return ibet;
    };

    std::function<duals::duald (duals::duald, duals::duald, duals::duald, duals::duald)> calc_iavl = [&](duals::duald Vbici, duals::duald Cjci, duals::duald itf, duals::duald T){
        //Avalanche current
        duals::duald iavl;
        duals::duald v_bord,v_q,U0,av,avl,cjci0_t, vdci_t, qavl_t,favl_t, kavl_t;
        if (use_aval == 1) {//begin : HICAVL
            double T_dpart = T.dpart();
            cjci0_t = here->HICUMcjci0_t.rpart;
            vdci_t = here->HICUMvdci_t.rpart;
            qavl_t = here->HICUMqavl_t.rpart;
            favl_t = here->HICUMfavl_t.rpart;
            kavl_t = here->HICUMkavl_t.rpart;
            if (T_dpart!=0.0){
                cjci0_t.dpart(here->HICUMcjci0_t.dpart);
                vdci_t.dpart(here->HICUMvdci_t.dpart);
                qavl_t.dpart(here->HICUMqavl_t.dpart);
                favl_t.dpart(here->HICUMfavl_t.dpart);
                kavl_t.dpart(here->HICUMkavl_t.dpart);
            }
            v_bord   = vdci_t-Vbici;
            if (v_bord > 0) {
                v_q     = qavl_t/Cjci;
                U0      = qavl_t/cjci0_t;
                if(v_bord > U0){
                    av      = favl_t*exp(-v_q/U0);
                    avl     = av*(U0+(1.0+v_q/U0)*(v_bord-U0));
                } else {
                    avl     = favl_t * v_bord * exp(-v_q / v_bord);
                }
                /* This model turns strong avalanche on. The parameter kavl can turn this
                * model extension off (kavl = 0). Although this is numerically stable, a
                * conditional statement is applied in order to reduce the numerical over-
                * head for simulations without the new model.
                */
                if (model->HICUMkavl > 0) { //: HICAVLHIGH
                    duals::duald denom,sq_smooth,hl;
                    denom = 1-kavl_t*avl;
                    // Avoid denom < 0 using a smoothing function
                    sq_smooth = sqrt(denom*denom+0.01);
                    hl        = 0.5*(denom+sq_smooth);
                    iavl      = itf*avl/hl;
                } else {
                    iavl    = itf*avl;
                }
            } else {
                iavl = 0.0;
            }
        } else{
            iavl = 0;
        }
        return iavl;
    };

    std::function<duals::duald (duals::duald, duals::duald, duals::duald)> calc_rbi = [&](duals::duald T, duals::duald Qjei, duals::duald Qf){
        //Internal base resistance
        duals::duald vt,rbi;
        vt      = CONSTboltz * T / CHARGE;
        if(here->HICUMrbi0_t.rpart > 0.0){ //: HICRBI
            duals::duald Qz_nom,f_QR,ETA,Qz0,fQz, qp0_t;
            double T_dpart = T.dpart();
            rbi = here->HICUMrbi0_t.rpart;
            qp0_t = here->HICUMqp0_t.rpart;
            if (T_dpart!=0.0) {
                rbi.dpart(here->HICUMrbi0_t.dpart);
                qp0_t.dpart(here->HICUMqp0_t.dpart);
            }
            // Consideration of conductivity modulation
            // To avoid convergence problem hyperbolic smoothing used
            f_QR    = (1+model->HICUMfdqr0)*qp0_t;
            Qz0     = Qjei+Qjci+Qf;
            Qz_nom  = 1+Qz0/f_QR;
            fQz     = 0.5*(Qz_nom+sqrt(Qz_nom*Qz_nom+0.01));
            rbi     = rbi/fQz;
            // Consideration of emitter current crowding
            if( ibei > 0.0) {
                ETA     = rbi*ibei*model->HICUMfgeo/vt;
                if(ETA < 1.0e-6) {
                    rbi     = rbi*(1.0-0.5*ETA);
                } else {
                    rbi     = rbi*log(1.0+ETA)/ETA;
                }
            }
            // Consideration of peripheral charge
            if(Qf > 0.0) {
                rbi     = rbi*(Qjei+Qf*model->HICUMfqi)/(Qjei+Qf);
            }
        } else {
            rbi     = 0.0;
        }
        return rbi;
    };

    std::function<void (duals::duald, duals::duald, duals::duald, duals::duald, duals::duald, duals::duald, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*)> calc_it_final = [&](duals::duald T, duals::duald Vbiei, duals::duald Vbici, duals::duald Q_pT, duals::duald T_f0, duals::duald ick, duals::duald *itf, duals::duald *itr, duals::duald *Qf, duals::duald *Qr, duals::duald *Q_bf, duals::duald * Tf){
        // given T,Q_pT, ick, T_f0, Tr, Vbiei, Vbici -> calculate itf, itr, Qf, Qr
        duals::duald VT, VT_f, i_0f, i_0r, I_Tf1, a_h, Q_fT,T_fT;
        duals::duald c10_t;

        double T_dpart = T.dpart();

        VT      = CONSTboltz * T / CHARGE;
        c10_t   = here->HICUMc10_t.rpart;
        if (T_dpart!=0.0) {
            c10_t.dpart(here->HICUMc10_t.dpart);
        }
        VT_f    = model->HICUMmcf*VT;
        i_0f    = c10_t * exp(Vbiei/VT_f);
        i_0r    = c10_t * exp(Vbici/VT);


        I_Tf1   = i_0f/Q_pT;
        a_h     = Oich*I_Tf1;
        *itf     = I_Tf1*(1.0+a_h);
        *itr     = i_0r/Q_pT;

        //Final transit times, charges and transport current components
        *Tf      = T_f0;
        *Qf      = T_f0*(*itf);
        HICQFF(T,*itf,ick,Tf,Qf,&T_fT,&Q_fT,Q_bf);
        *Qr      = Tr*(*itr);
    };

    std::function<void (duals::duald, duals::duald, duals::duald, duals::duald, duals::duald, duals::duald, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*, duals::duald*)> calc_it_initial = [&](duals::duald T, duals::duald Vbiei, duals::duald Vbici, duals::duald Q_0, duals::duald T_f0, duals::duald ick, duals::duald *itf, duals::duald *itr, duals::duald *Qf, duals::duald *Qr, duals::duald *Q_bf, duals::duald *a_h, duals::duald *Q_p, duals::duald *Tf){
        // given T,Q_pT, ick, T_f0, Tr, Vbiei, Vbici -> calculate itf, itr, Qf, Qr
        duals::duald VT, VT_f, i_0f, i_0r, I_Tf1, Q_fT, T_fT, A;
        duals::duald c10_t;
        double T_dpart = T.dpart();

        VT      = CONSTboltz * T / CHARGE;
        c10_t   = here->HICUMc10_t.rpart;
        if (T_dpart!=0.0) {
            c10_t.dpart(here->HICUMc10_t.dpart);
        }
        VT_f    = model->HICUMmcf*VT;
        i_0f    = c10_t * exp(Vbiei/VT_f);
        i_0r    = c10_t * exp(Vbici/VT);

        *Q_p     = Q_0;
        if (T_f0 > 0.0 || Tr > 0.0) {
            A       = 0.5*Q_0;
            *Q_p    = A+sqrt(A*A+T_f0*i_0f+Tr*i_0r);
        }
        I_Tf1   =i_0f/(*Q_p);
        *a_h     = Oich*I_Tf1;
        *itf    = I_Tf1*(1.0+*a_h);
        *itr    = i_0r/(*Q_p);

        //Initial formulation of forward transit time, diffusion, GICCR and excess b-c charge
        *Q_bf    = 0.0;
        *Tf      = T_f0;
        *Qf      = T_f0*(*itf);
        HICQFF(T,*itf,ick,Tf,Qf,&T_fT,&Q_fT,Q_bf);

        //Initial formulation of reverse diffusion charge
        *Qr      = Tr*(*itr);
    };

    std::function<duals::duald (duals::duald, duals::duald, duals::duald, duals::duald, duals::duald, duals::duald)> calc_it = [&](duals::duald T, duals::duald Vbiei, duals::duald Vbici, duals::duald Q_0, duals::duald T_f0, duals::duald ick){
        // This function calculates Q_pT in a dual way
        // Tr also as argument here?
        duals::duald VT, VT_f,i_0f,i_0r, Q_p, A, I_Tf1,itf, itr, a_h, Qf, Qr, d_Q0, Q_pT, a, d_Q, Tf, T_fT, Q_bf, Q_fT;
        duals::duald c10_t;
        int extra_round=0;
        double T_dpart = T.dpart();
        int l_it;

        VT      = CONSTboltz * T / CHARGE;
        c10_t   = here->HICUMc10_t.rpart;
        if (T_dpart!=0.0) {
            c10_t.dpart(here->HICUMc10_t.dpart);
        }
        VT_f    = model->HICUMmcf*VT;
        i_0f    = c10_t * exp(Vbiei/VT_f);
        i_0r    = c10_t * exp(Vbici/VT);

        //Initial formulation of forward and reverse component of transfer current
        Q_p     = Q_0;
        if (T_f0 > 0.0 || Tr > 0.0) {
            A       = 0.5*Q_0;
            Q_p     = A+sqrt(A*A+T_f0*i_0f+Tr*i_0r);
        }
        I_Tf1   = i_0f/Q_p;
        a_h     = Oich*I_Tf1;
        itf     = I_Tf1*(1.0+a_h);
        itr     = i_0r/Q_p;

        //Initial formulation of forward transit time, diffusion, GICCR and excess b-c charge
        Q_bf    = 0.0;
        Tf      = T_f0;
        Qf      = T_f0*itf;
        HICQFF(T,itf,ick,&Tf,&Qf,&T_fT,&Q_fT,&Q_bf);

        //Initial formulation of reverse diffusion charge
        Qr      = Tr*itr;

        //Preparation for iteration to get total hole charge and related variables
        l_it    = 0;
        extra_round = 1;
        if(Qf > RTOLC*Q_p || a_h > RTOLC) {
            //Iteration for Q_pT is required for improved initial solution
            Qf      = sqrt(T_f0*itf*Q_fT);
            Q_pT    = Q_0+Qf+Qr;
            d_Q     = Q_pT;
            while ( ((abs(d_Q) >= RTOLC*abs(Q_pT)) && (l_it <= l_itmax)) || (extra_round < 5)) {
                d_Q0    = d_Q;
                I_Tf1   = i_0f/Q_pT;
                a_h     = Oich*I_Tf1;
                itf     = I_Tf1*(1.0+a_h);
                itr     = i_0r/Q_pT;
                Tf      = T_f0;
                Qf      = T_f0*itf;
                HICQFF(T,itf,ick,&Tf,&Qf,&T_fT,&Q_fT,&Q_bf);
                Qr      = Tr*itr;
                if(Oich == 0.0) {
                    a       = 1.0+(T_fT*itf+Qr)/Q_pT;
                } else {
                    a       = 1.0+(T_fT*I_Tf1*(1.0+2.0*a_h)+Qr)/Q_pT;
                }
                d_Q     = -(Q_pT-(Q_0+Q_fT+Qr))/a;
                //Limit maximum change of Q_pT
                a       = abs(0.3*Q_pT);
                if(abs(d_Q) > a) {
                    if (d_Q>=0) {
                        d_Q     = a;
                    } else {
                        d_Q     = -a;
                    }
                }
                Q_pT    = Q_pT+d_Q;
                l_it    = l_it+1;
                if (!(abs(d_Q) >= RTOLC*abs(Q_pT))) { //extra_rounds to get rid of derivative noise
                    extra_round = extra_round + 1;
                }
            }

            //final calculations afterwards, see later in load where this is called

        }
        return Q_pT;

    };

    std::function<void (duals::duald, duals::duald, duals::duald, duals::duald*, duals::duald*)> calc_itss = [&](duals::duald T, duals::duald Vbpci, duals::duald Vsici, duals::duald * HSI_Tsu, duals::duald * Qdsu){
        duals::duald HSUM, vt, HSa, HSb, itss_t, tsf_t;
        double T_dpart = T.dpart();
        vt      = CONSTboltz * T / CHARGE;
        itss_t = here->HICUMitss_t.rpart;
        tsf_t = here->HICUMtsf_t.rpart;
        if (T_dpart!=0.0){
            itss_t.dpart(here->HICUMitss_t.dpart);
            tsf_t.dpart(here->HICUMtsf_t.dpart);
        }
        if(model->HICUMitss > 0.0) { // : Sub_Transfer
            HSUM    = model->HICUMmsf*vt;
            HSa     = exp(Vbpci/HSUM);
            HSb     = exp(Vsici/HSUM);
            *HSI_Tsu = itss_t*(HSa-HSb);
            if(model->HICUMtsf > 0.0) {
                *Qdsu    = tsf_t*itss_t*HSa;
            } else {
                *Qdsu    = 0.0;
            }
        } else {
            *HSI_Tsu = 0.0;
            *Qdsu    = 0.0;
        };
    };


    /*  loop through all the models */
    for (; model != NULL; model = HICUMnextModel(model)) {

        // Avoid divide-by-zero and define infinity other way
        // High current correction for 2D and 3D effects
        if (model->HICUMich != 0.0) {
            Oich = 1.0 / model->HICUMich;
        }
        else {
            Oich = 0.0;
        }

        // Base current recombination time constant at b-c barrier
        if (model->HICUMtbhrec != 0.0) {
            Otbhrec = 1.0 / model->HICUMtbhrec;
        }
        else {
            Otbhrec = 0.0;
        }

        // Turn avalanche calculation on depending of parameters
        if ((model->HICUMfavl > 0.0) && (model->HICUMcjci0 > 0.0)) {
            use_aval = 1;
        } else {
            use_aval = 0;
        }

        // end of Model_initialization

        /* loop through all the instances of the model */
        for (here = HICUMinstances(model); here != NULL ;
                here=HICUMnextInstance(here)) {

            // Depletion capacitance splitting at b-c junction
            // Capacitances at peripheral and external base node
            C_1 = (1.0 - model->HICUMfbcpar) *
                    (here->HICUMcjcx0_scaled + here->HICUMcbcpar_scaled);
            if (C_1 >= here->HICUMcbcpar_scaled) {
                cbcpar1 = here->HICUMcbcpar_scaled;
                cbcpar2 = 0.0;
            }
            else {
                cbcpar1 = C_1;
                cbcpar2 = here->HICUMcbcpar_scaled - cbcpar1;
            }

            // Parasitic b-e capacitance partitioning: No temperature dependence
            cbepar2 = model->HICUMfbepar * here->HICUMcbepar_scaled;
            cbepar1 = here->HICUMcbepar_scaled - cbepar2;

            gqbepar1 = 0.0;
            gqbepar2 = 0.0;
            gqbcpar1 = 0.0;
            gqbcpar2 = 0.0;
            gqsu = 0.0;
            Icth = 0.0, Icth_Vrth = 0.0;

            /*
             *   initialization
             */
            icheck=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                Vbiei = *(ckt->CKTstate0 + here->HICUMvbiei);
                Vbici = *(ckt->CKTstate0 + here->HICUMvbici);
                // dead assign:
                //Vciei = Vbiei - Vbici;
                Vbpei = *(ckt->CKTstate0 + here->HICUMvbpei);
                Vbpci = *(ckt->CKTstate0 + here->HICUMvbpci);
                Vbci = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollCINode));
                Vsici = *(ckt->CKTstate0 + here->HICUMvsici);
                Vsc = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));

                Vbpbi = *(ckt->CKTstate0 + here->HICUMvbpbi);
                Vbe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vcic = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMcollCINode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));
                Vbbp = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode));
                Vbpe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Veie = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMemitEINode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vsis = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsSINode)-
                    *(ckt->CKTrhsOld+here->HICUMsubsNode));
                Vxf  = *(ckt->CKTrhsOld + here->HICUMxfNode);
                Vxf1 = *(ckt->CKTrhsOld + here->HICUMxf1Node);
                Vxf2 = *(ckt->CKTrhsOld + here->HICUMxf2Node);
                if (model->HICUMselfheat) {
                    Vrth = *(ckt->CKTstate0 + here->HICUMvrth);
                }
            } else if(ckt->CKTmode & MODEINITTRAN) {
                Vbiei = *(ckt->CKTstate1 + here->HICUMvbiei);
                Vbici = *(ckt->CKTstate1 + here->HICUMvbici);
                // dead assign:
                //Vciei = Vbiei - Vbici;
                Vbpei = *(ckt->CKTstate1 + here->HICUMvbpei);
                Vbpci = *(ckt->CKTstate1 + here->HICUMvbpci);
                Vbci = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollCINode));
                Vsici = *(ckt->CKTstate1 + here->HICUMvsici);
                Vsc = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));

                Vbpbi = *(ckt->CKTstate1 + here->HICUMvbpbi);
                Vbe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vcic = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMcollCINode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));
                Vbbp = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode));
                Vbpe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Veie = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMemitEINode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vsis = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsSINode)-
                    *(ckt->CKTrhsOld+here->HICUMsubsNode));
                Vxf  = *(ckt->CKTrhsOld + here->HICUMxfNode);
                Vxf1 = *(ckt->CKTrhsOld + here->HICUMxf1Node);
                Vxf2 = *(ckt->CKTrhsOld + here->HICUMxf2Node);
                if (model->HICUMselfheat) {
                    Vrth = *(ckt->CKTstate1 + here->HICUMvrth);
                }
            } else if((ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)){
                Vbiei = model->HICUMtype * here->HICUMicVBE; // multiplication by type has historical reasons
                Vciei = model->HICUMtype * here->HICUMicVCE; // the user sets anytime positive numbers independent from type
                Vbici = Vbiei - Vciei;
                Vsc = Vsici = model->HICUMtype * here->HICUMicVCS;
                Vbe = Vbpei = Vbiei;
                Vbci = Vbpci = Vbici;
                Vbpbi = 0.0;
                Vbbp  = 0.0;
                Vbpe  = 0.0;
                Vcic  = 0.0;
                Veie  = 0.0;
                Vrth  = 0.0;
                Vsis  = 0.0;
                Icth  = 0.0, Icth_Vrth=0.0;
                Vxf=Vxf1=Vxf2=0.0;
            } else if((ckt->CKTmode & MODEINITJCT) && (here->HICUMoff==0)) {
                Vbiei = here->HICUMtVcrit;
                Vbici = 0.0;
                // dead assign:
                //Vciei = Vbiei - Vbici;
                Vsc = Vsici = 0.0;
                Vbe = Vbpei = Vbiei;
                Vbci = Vbpci = Vbici;
                Vbpbi = 0.0;
                Vbbp  = 0.0;
                Vbpe  = 0.0;
                Vcic  = 0.0;
                Veie  = 0.0;
                Vrth  = 0.0;
                Vsis  = 0.0;
                Icth  = 0.0, Icth_Vrth=0.0;
                Vxf=Vxf1=Vxf2=0.0;
            } else if((ckt->CKTmode & MODEINITJCT) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (here->HICUMoff!=0))) {
                Vbe=0.0;
                Vbiei=Vbe;
                // dead assign:
                //Vciei=0.0;
                Vbci=Vbici=Vbpci=0.0;
                Vbpei=0.0;
                Vsc=Vsici=0.0;
                Vbpbi=Vbbp=Vbpe=0.0;
                Vcic=Veie=Vsis=0.0;
                Vrth=0.0,Icth=0.0,Icth_Vrth=0.0;
                Vxf=Vxf1=Vxf2=0.0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact = ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    Vbiei = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbiei)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbiei);
                    Vbici = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbici)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbici);
                    Vciei = Vbiei - Vbici;
                    Vbpei = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbpei)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbpei);
                    Vbpci = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbpci)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbpci);
                    Vsici = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvsici)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvsici);
                    Vbpbi = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbpbi)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbpbi);
                    // dead assign:
                    // Vxf   = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvxf)-
                    //         xfact * *(ckt->CKTstate2 + here->HICUMvxf);
                    // Vxf1  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvxf1)-
                    //         xfact * *(ckt->CKTstate2 + here->HICUMvxf1);
                    Vxf2  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvxf2)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvxf2);
                    Vrth  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvrth);
                    Veie  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMveie)-
                            xfact * *(ckt->CKTstate2 + here->HICUMveie);
                    Vcic  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvcic)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvcic);
                    Vbbp  = (1+xfact) * *(ckt->CKTstate1 + here->HICUMvbbp)-
                            xfact * *(ckt->CKTstate2 + here->HICUMvbbp);
                    /////////////////////////
                    // begin copy state vector
                    /////////////////////////
                    *(ckt->CKTstate0+here->HICUMvbiei)=*(ckt->CKTstate1+here->HICUMvbiei);
                    *(ckt->CKTstate0+here->HICUMvbici)=*(ckt->CKTstate1+here->HICUMvbici);
                    *(ckt->CKTstate0+here->HICUMvbpei)=*(ckt->CKTstate1+here->HICUMvbpei);
                    *(ckt->CKTstate0+here->HICUMvbpbi)=*(ckt->CKTstate1+here->HICUMvbpbi);
                    *(ckt->CKTstate0+here->HICUMvbpci)=*(ckt->CKTstate1+here->HICUMvbpci);
                    *(ckt->CKTstate0+here->HICUMvsici)=*(ckt->CKTstate1+here->HICUMvsici);
                    *(ckt->CKTstate0+here->HICUMvcic)=*(ckt->CKTstate1+here->HICUMvcic);
                    *(ckt->CKTstate0+here->HICUMvbbp)=*(ckt->CKTstate1+here->HICUMvbbp);
                    *(ckt->CKTstate0+here->HICUMveie)=*(ckt->CKTstate1+here->HICUMveie);
                    *(ckt->CKTstate0+here->HICUMibiei)=*(ckt->CKTstate1+here->HICUMibiei);
                    *(ckt->CKTstate0+here->HICUMibiei_Vbiei)=*(ckt->CKTstate1+here->HICUMibiei_Vbiei);
                    *(ckt->CKTstate0+here->HICUMibiei_Vxf)=*(ckt->CKTstate1+here->HICUMibiei_Vxf);
                    *(ckt->CKTstate0+here->HICUMibiei_Vbici)=*(ckt->CKTstate1+here->HICUMibiei_Vbici);
                    *(ckt->CKTstate0+here->HICUMibiei_Vrth)=*(ckt->CKTstate1+here->HICUMibiei_Vrth);
                    *(ckt->CKTstate0+here->HICUMibpei)=*(ckt->CKTstate1+here->HICUMibpei);
                    *(ckt->CKTstate0+here->HICUMibpei_Vbpei)=*(ckt->CKTstate1+here->HICUMibpei_Vbpei);
                    *(ckt->CKTstate0+here->HICUMibpei_Vrth)=*(ckt->CKTstate1+here->HICUMibpei_Vrth);
                    *(ckt->CKTstate0+here->HICUMiciei)=*(ckt->CKTstate1+here->HICUMiciei);
                    *(ckt->CKTstate0+here->HICUMiciei_Vbiei)=*(ckt->CKTstate1+here->HICUMiciei_Vbiei);
                    *(ckt->CKTstate0+here->HICUMiciei_Vbici)=*(ckt->CKTstate1+here->HICUMiciei_Vbici);
                    *(ckt->CKTstate0+here->HICUMiciei_Vxf2)=*(ckt->CKTstate1+here->HICUMiciei_Vxf2);
                    *(ckt->CKTstate0+here->HICUMiciei_Vrth)=*(ckt->CKTstate1+here->HICUMiciei_Vrth);
                    *(ckt->CKTstate0+here->HICUMibici)=*(ckt->CKTstate1+here->HICUMibici);
                    *(ckt->CKTstate0+here->HICUMibici_Vbici)=*(ckt->CKTstate1+here->HICUMibici_Vbici);
                    *(ckt->CKTstate0+here->HICUMibici_Vbiei)=*(ckt->CKTstate1+here->HICUMibici_Vbiei);
                    *(ckt->CKTstate0+here->HICUMibici_Vrth)=*(ckt->CKTstate1+here->HICUMibici_Vrth);
                    *(ckt->CKTstate0+here->HICUMibpbi)=*(ckt->CKTstate1+here->HICUMibpbi);
                    *(ckt->CKTstate0+here->HICUMibpbi_Vbpbi)=*(ckt->CKTstate1+here->HICUMibpbi_Vbpbi);
                    *(ckt->CKTstate0+here->HICUMibpbi_Vbiei)=*(ckt->CKTstate1+here->HICUMibpbi_Vbiei);
                    *(ckt->CKTstate0+here->HICUMibpbi_Vbici)=*(ckt->CKTstate1+here->HICUMibpbi_Vbici);
                    *(ckt->CKTstate0+here->HICUMibpbi_Vrth)=*(ckt->CKTstate1+here->HICUMibpbi_Vrth);
                    *(ckt->CKTstate0+here->HICUMibpci)=*(ckt->CKTstate1+here->HICUMibpci);
                    *(ckt->CKTstate0+here->HICUMibpci_Vbpci)=*(ckt->CKTstate1+here->HICUMibpci_Vbpci);
                    *(ckt->CKTstate0+here->HICUMibpci_Vrth)=*(ckt->CKTstate1+here->HICUMibpci_Vrth);
                    *(ckt->CKTstate0+here->HICUMisici)=*(ckt->CKTstate1+here->HICUMisici);
                    *(ckt->CKTstate0+here->HICUMisici_Vsici)=*(ckt->CKTstate1+here->HICUMisici_Vsici);
                    *(ckt->CKTstate0+here->HICUMisici_Vrth)=*(ckt->CKTstate1+here->HICUMisici_Vrth);
                    *(ckt->CKTstate0+here->HICUMibpsi)=*(ckt->CKTstate1+here->HICUMibpsi);
                    *(ckt->CKTstate0+here->HICUMibpsi_Vbpci)=*(ckt->CKTstate1+here->HICUMibpsi_Vbpci);
                    *(ckt->CKTstate0+here->HICUMibpsi_Vsici)=*(ckt->CKTstate1+here->HICUMibpsi_Vsici);
                    *(ckt->CKTstate0+here->HICUMibpsi_Vrth)=*(ckt->CKTstate1+here->HICUMibpsi_Vrth);
                    *(ckt->CKTstate0+here->HICUMisis_Vsis)=*(ckt->CKTstate1+here->HICUMisis_Vsis);
                    *(ckt->CKTstate0+here->HICUMieie)=*(ckt->CKTstate1+here->HICUMieie);
                    *(ckt->CKTstate0+here->HICUMieie_Vrth)=*(ckt->CKTstate1+here->HICUMieie_Vrth);
                    *(ckt->CKTstate0+here->HICUMqrbi)=*(ckt->CKTstate1+here->HICUMqrbi);
                    *(ckt->CKTstate0+here->HICUMcqrbi)=*(ckt->CKTstate1+here->HICUMcqrbi);
                    *(ckt->CKTstate0+here->HICUMqjei)=*(ckt->CKTstate1+here->HICUMqjei);
                    *(ckt->CKTstate0+here->HICUMcqjei)=*(ckt->CKTstate1+here->HICUMcqjei);
                    *(ckt->CKTstate0+here->HICUMqf)=*(ckt->CKTstate1+here->HICUMqf);
                    *(ckt->CKTstate0+here->HICUMcqf)=*(ckt->CKTstate1+here->HICUMcqf);
                    *(ckt->CKTstate0+here->HICUMqr)=*(ckt->CKTstate1+here->HICUMqr);
                    *(ckt->CKTstate0+here->HICUMcqr)=*(ckt->CKTstate1+here->HICUMcqr);
                    *(ckt->CKTstate0+here->HICUMqjci)=*(ckt->CKTstate1+here->HICUMqjci);
                    *(ckt->CKTstate0+here->HICUMcqjci)=*(ckt->CKTstate1+here->HICUMcqjci);
                    *(ckt->CKTstate0+here->HICUMqjep)=*(ckt->CKTstate1+here->HICUMqjep);
                    *(ckt->CKTstate0+here->HICUMcqjep)=*(ckt->CKTstate1+here->HICUMcqjep);
                    *(ckt->CKTstate0+here->HICUMqjcx0_i)=*(ckt->CKTstate1+here->HICUMqjcx0_i);
                    *(ckt->CKTstate0+here->HICUMcqcx0_t_i)=*(ckt->CKTstate1+here->HICUMcqcx0_t_i);
                    *(ckt->CKTstate0+here->HICUMqjcx0_ii)=*(ckt->CKTstate1+here->HICUMqjcx0_ii);
                    *(ckt->CKTstate0+here->HICUMcqcx0_t_ii)=*(ckt->CKTstate1+here->HICUMcqcx0_t_ii);
                    *(ckt->CKTstate0+here->HICUMqdsu)=*(ckt->CKTstate1+here->HICUMqdsu);
                    *(ckt->CKTstate0+here->HICUMcqdsu)=*(ckt->CKTstate1+here->HICUMcqdsu);
                    *(ckt->CKTstate0+here->HICUMqjs)=*(ckt->CKTstate1+here->HICUMqjs);
                    *(ckt->CKTstate0+here->HICUMcqjs)=*(ckt->CKTstate1+here->HICUMcqjs);
                    *(ckt->CKTstate0+here->HICUMqscp)=*(ckt->CKTstate1+here->HICUMqscp);
                    *(ckt->CKTstate0+here->HICUMcqscp)=*(ckt->CKTstate1+here->HICUMcqscp);
                    *(ckt->CKTstate0+here->HICUMqbepar1)=*(ckt->CKTstate1+here->HICUMqbepar1);
                    *(ckt->CKTstate0+here->HICUMcqbepar1)=*(ckt->CKTstate1+here->HICUMcqbepar1);
                    *(ckt->CKTstate0+here->HICUMgqbepar1)=*(ckt->CKTstate1+here->HICUMgqbepar1);
                    *(ckt->CKTstate0+here->HICUMqbepar2)=*(ckt->CKTstate1+here->HICUMqbepar2);
                    *(ckt->CKTstate0+here->HICUMcqbepar2)=*(ckt->CKTstate1+here->HICUMcqbepar2);
                    *(ckt->CKTstate0+here->HICUMgqbepar2)=*(ckt->CKTstate1+here->HICUMgqbepar2);
                    *(ckt->CKTstate0+here->HICUMqbcpar1)=*(ckt->CKTstate1+here->HICUMqbcpar1);
                    *(ckt->CKTstate0+here->HICUMcqbcpar1)=*(ckt->CKTstate1+here->HICUMcqbcpar1);
                    *(ckt->CKTstate0+here->HICUMgqbcpar1)=*(ckt->CKTstate1+here->HICUMgqbcpar1);
                    *(ckt->CKTstate0+here->HICUMqbcpar2)=*(ckt->CKTstate1+here->HICUMqbcpar2);
                    *(ckt->CKTstate0+here->HICUMcqbcpar2)=*(ckt->CKTstate1+here->HICUMcqbcpar2);
                    *(ckt->CKTstate0+here->HICUMgqbcpar2)=*(ckt->CKTstate1+here->HICUMgqbcpar2);
                    *(ckt->CKTstate0+here->HICUMqsu)=*(ckt->CKTstate1+here->HICUMqsu);
                    *(ckt->CKTstate0+here->HICUMcqsu)=*(ckt->CKTstate1+here->HICUMcqsu);
                    *(ckt->CKTstate0+here->HICUMgqsu)=*(ckt->CKTstate1+here->HICUMgqsu);
                    *(ckt->CKTstate0+here->HICUMqcth)=*(ckt->CKTstate1+here->HICUMqcth);
                    *(ckt->CKTstate0+here->HICUMcqcth)=*(ckt->CKTstate1+here->HICUMcqcth);
                    *(ckt->CKTstate0+here->HICUMvrth)=*(ckt->CKTstate1+here->HICUMvrth);
                    *(ckt->CKTstate0+here->HICUMvxf)=*(ckt->CKTstate1+here->HICUMvxf);
                    *(ckt->CKTstate0+here->HICUMqxf)=*(ckt->CKTstate1+here->HICUMqxf);
                    *(ckt->CKTstate0+here->HICUMcqxf)=*(ckt->CKTstate1+here->HICUMcqxf);
                    *(ckt->CKTstate0+here->HICUMgqxf)=*(ckt->CKTstate1+here->HICUMgqxf);
                    *(ckt->CKTstate0+here->HICUMixf_Vbiei)=*(ckt->CKTstate1+here->HICUMixf_Vbiei);
                    *(ckt->CKTstate0+here->HICUMixf_Vbici)=*(ckt->CKTstate1+here->HICUMixf_Vbici);
                    *(ckt->CKTstate0+here->HICUMixf_Vxf)=*(ckt->CKTstate1+here->HICUMixf_Vxf);
                    *(ckt->CKTstate0+here->HICUMixf_Vrth)=*(ckt->CKTstate1+here->HICUMixf_Vrth);
                    *(ckt->CKTstate0+here->HICUMvxf1)=*(ckt->CKTstate1+here->HICUMvxf1);
                    *(ckt->CKTstate0+here->HICUMqxf1)=*(ckt->CKTstate1+here->HICUMqxf1);
                    *(ckt->CKTstate0+here->HICUMcqxf1)=*(ckt->CKTstate1+here->HICUMcqxf1);
                    *(ckt->CKTstate0+here->HICUMgqxf1)=*(ckt->CKTstate1+here->HICUMgqxf1);
                    *(ckt->CKTstate0+here->HICUMixf1_Vbiei)=*(ckt->CKTstate1+here->HICUMixf1_Vbiei);
                    *(ckt->CKTstate0+here->HICUMixf1_Vbici)=*(ckt->CKTstate1+here->HICUMixf1_Vbici);
                    *(ckt->CKTstate0+here->HICUMixf1_Vxf2)=*(ckt->CKTstate1+here->HICUMixf1_Vxf2);
                    *(ckt->CKTstate0+here->HICUMixf1_Vxf1)=*(ckt->CKTstate1+here->HICUMixf1_Vxf1);
                    *(ckt->CKTstate0+here->HICUMixf1_Vrth)=*(ckt->CKTstate1+here->HICUMixf1_Vrth);
                    *(ckt->CKTstate0+here->HICUMvxf2)=*(ckt->CKTstate1+here->HICUMvxf2);
                    *(ckt->CKTstate0+here->HICUMqxf2)=*(ckt->CKTstate1+here->HICUMqxf2);
                    *(ckt->CKTstate0+here->HICUMcqxf2)=*(ckt->CKTstate1+here->HICUMcqxf2);
                    *(ckt->CKTstate0+here->HICUMgqxf2)=*(ckt->CKTstate1+here->HICUMgqxf2);
                    *(ckt->CKTstate0+here->HICUMixf2_Vbiei)=*(ckt->CKTstate1+here->HICUMixf2_Vbiei);
                    *(ckt->CKTstate0+here->HICUMixf2_Vbici)=*(ckt->CKTstate1+here->HICUMixf2_Vbici);
                    *(ckt->CKTstate0+here->HICUMixf2_Vxf1)=*(ckt->CKTstate1+here->HICUMixf2_Vxf1);
                    *(ckt->CKTstate0+here->HICUMixf2_Vxf2)=*(ckt->CKTstate1+here->HICUMixf2_Vxf2);
                    *(ckt->CKTstate0+here->HICUMixf2_Vrth)=*(ckt->CKTstate1+here->HICUMixf2_Vrth);
                    *(ckt->CKTstate0+here->HICUMith)=*(ckt->CKTstate1+here->HICUMith);
                    *(ckt->CKTstate0+here->HICUMith_Vrth)=*(ckt->CKTstate1+here->HICUMith_Vrth);
                    *(ckt->CKTstate0+here->HICUMith_Vbiei)=*(ckt->CKTstate1+here->HICUMith_Vbiei);
                    *(ckt->CKTstate0+here->HICUMith_Vbici)=*(ckt->CKTstate1+here->HICUMith_Vbici);
                    *(ckt->CKTstate0+here->HICUMith_Vbpbi)=*(ckt->CKTstate1+here->HICUMith_Vbpbi);
                    *(ckt->CKTstate0+here->HICUMith_Vbpci)=*(ckt->CKTstate1+here->HICUMith_Vbpci);
                    *(ckt->CKTstate0+here->HICUMith_Vbpei)=*(ckt->CKTstate1+here->HICUMith_Vbpei);
                    *(ckt->CKTstate0+here->HICUMith_Vciei)=*(ckt->CKTstate1+here->HICUMith_Vciei);
                    *(ckt->CKTstate0+here->HICUMith_Vsici)=*(ckt->CKTstate1+here->HICUMith_Vsici);
                    *(ckt->CKTstate0+here->HICUMith_Vcic)=*(ckt->CKTstate1+here->HICUMith_Vcic);
                    *(ckt->CKTstate0+here->HICUMith_Vbbp)=*(ckt->CKTstate1+here->HICUMith_Vbbp);
                    *(ckt->CKTstate0+here->HICUMith_Veie)=*(ckt->CKTstate1+here->HICUMith_Veie);
                    *(ckt->CKTstate0+here->HICUMit)=*(ckt->CKTstate1+here->HICUMit);
                    /////////////////////////
                    // end copy state vector
                    /////////////////////////
                } else {
#endif /* PREDICTOR */
                    /*
                     *   compute new nonlinear branch voltages
                     */
                    Vbiei = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseBINode)-
                        *(ckt->CKTrhsOld+here->HICUMemitEINode));
                    // no direct Vbe branch exists, dont need this voltage:
                    // Vbe   = model->HICUMtype*(
                    //     *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    //     *(ckt->CKTrhsOld+here->HICUMemitNode));
                    Vbici = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseBINode)-
                        *(ckt->CKTrhsOld+here->HICUMcollCINode));
                    Vbpei = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                        *(ckt->CKTrhsOld+here->HICUMemitEINode));
                    Vbpbi = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                        *(ckt->CKTrhsOld+here->HICUMbaseBINode));
                    Veie = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMemitEINode)-
                        *(ckt->CKTrhsOld+here->HICUMemitNode));
                    Vcic = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMcollCINode)-
                        *(ckt->CKTrhsOld+here->HICUMcollNode));
                    Vbbp = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                        *(ckt->CKTrhsOld+here->HICUMbaseBPNode));
                    Vbpci = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                        *(ckt->CKTrhsOld+here->HICUMcollCINode));
                    Vsici = model->HICUMtype*(
                        *(ckt->CKTrhsOld+here->HICUMsubsSINode)-
                        *(ckt->CKTrhsOld+here->HICUMcollCINode));
                    // not needed because convergence in NQS network is not checked here
                    //Vxf   = *(ckt->CKTrhsOld + here->HICUMxfNode);
                    //Vxf1  = *(ckt->CKTrhsOld + here->HICUMxf1Node);
                    Vxf2  = *(ckt->CKTrhsOld + here->HICUMxf2Node);
                    Vciei = Vbiei - Vbici;
                    if (model->HICUMselfheat) {
                        Vrth = *(ckt->CKTrhsOld + here->HICUMtempNode);
                    }

#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvbiei = Vbiei - *(ckt->CKTstate0 + here->HICUMvbiei);
                delvbici = Vbici - *(ckt->CKTstate0 + here->HICUMvbici);
                delvbpei = Vbpei - *(ckt->CKTstate0 + here->HICUMvbpei);
                delvbpbi = Vbpbi - *(ckt->CKTstate0 + here->HICUMvbpbi);
                delvbpci = Vbpci - *(ckt->CKTstate0 + here->HICUMvbpci);
                delvsici = Vsici - *(ckt->CKTstate0 + here->HICUMvsici);
                delvcic  = Vcic  - *(ckt->CKTstate0 + here->HICUMvcic);
                delvbbp  = Vbbp  - *(ckt->CKTstate0 + here->HICUMvbbp);
                delveie  = Veie  - *(ckt->CKTstate0 + here->HICUMveie);
                delvxf2  = Vxf2  - *(ckt->CKTstate0 + here->HICUMvxf2);
                delvciei = delvbiei-delvbici;
                if (model->HICUMselfheat) {
                    delvrth  = Vrth  - *(ckt->CKTstate0 + here->HICUMvrth);
                } else {
                    delvrth  = 0;
                }
                Vbe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vsc = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));
                Vbci = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMcollCINode));
                Vbpe = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Veie = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMemitEINode)-
                    *(ckt->CKTrhsOld+here->HICUMemitNode));
                Vcic = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMcollCINode)-
                    *(ckt->CKTrhsOld+here->HICUMcollNode));
                Vbbp = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                    *(ckt->CKTrhsOld+here->HICUMbaseBPNode));
                Vsis = model->HICUMtype*(
                    *(ckt->CKTrhsOld+here->HICUMsubsSINode)-
                    *(ckt->CKTrhsOld+here->HICUMsubsNode));
                Vxf = *(ckt->CKTrhsOld + here->HICUMxfNode);
                Vxf1 = *(ckt->CKTrhsOld + here->HICUMxf1Node);
                Vxf2 = *(ckt->CKTrhsOld + here->HICUMxf2Node);
                if (model->HICUMselfheat) {
                    Vrth = *(ckt->CKTrhsOld + here->HICUMtempNode);
                }
                //todo: maybe add ibiei_Vxf
                ibieihat = *(ckt->CKTstate0 + here->HICUMibiei) +
                         *(ckt->CKTstate0 + here->HICUMibiei_Vbiei)*delvbiei+
                         *(ckt->CKTstate0 + here->HICUMibiei_Vrth)*delvrth+
                         *(ckt->CKTstate0 + here->HICUMibiei_Vbici)*delvbici;
                ibicihat = *(ckt->CKTstate0 + here->HICUMibici) +
                         *(ckt->CKTstate0 + here->HICUMibici_Vbici)*delvbici+
                         *(ckt->CKTstate0 + here->HICUMibici_Vrth)*delvrth+
                         *(ckt->CKTstate0 + here->HICUMibici_Vbiei)*delvbiei;
                ibpeihat = *(ckt->CKTstate0 + here->HICUMibpei) +
                         *(ckt->CKTstate0 + here->HICUMibpei_Vbpei)*delvbpei+
                         *(ckt->CKTstate0 + here->HICUMibpei_Vrth)*delvrth;
                ibpcihat = *(ckt->CKTstate0 + here->HICUMibpci) +
                         *(ckt->CKTstate0 + here->HICUMibpci_Vbpci)*delvbpci+
                         *(ckt->CKTstate0 + here->HICUMibpci_Vrth)*delvrth;
                icieihat = *(ckt->CKTstate0 + here->HICUMiciei) +
                         *(ckt->CKTstate0 + here->HICUMiciei_Vbiei)*delvbiei +
                         *(ckt->CKTstate0 + here->HICUMiciei_Vrth)*delvrth +
                         *(ckt->CKTstate0 + here->HICUMiciei_Vxf2)*delvxf2 +
                         *(ckt->CKTstate0 + here->HICUMiciei_Vbici)*delvbici;
                ibpbihat = *(ckt->CKTstate0 + here->HICUMibpbi) +
                         *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi)*delvbpbi +
                         *(ckt->CKTstate0 + here->HICUMibpbi_Vrth)*delvrth +
                         *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei)*delvbiei +
                         *(ckt->CKTstate0 + here->HICUMibpbi_Vbici)*delvbici;
                isicihat = *(ckt->CKTstate0 + here->HICUMisici) +
                         *(ckt->CKTstate0 + here->HICUMisici_Vsici)*delvsici+
                         *(ckt->CKTstate0 + here->HICUMisici_Vrth)*delvrth;
                ibpsihat = *(ckt->CKTstate0 + here->HICUMibpsi) +
                         *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci)*delvbpci +
                         *(ckt->CKTstate0 + here->HICUMibpsi_Vrth)*delvrth +
                         *(ckt->CKTstate0 + here->HICUMibpsi_Vsici)*delvsici;
                ithhat   = *(ckt->CKTstate0 + here->HICUMith) +
                         *(ckt->CKTstate0 + here->HICUMith_Vbiei)*delvbiei+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vbici)*delvbici+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vbpbi)*delvbpbi+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vbpci)*delvbpci+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vbpei)*delvbpei+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vciei)*delvciei+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vsici)*delvsici+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vcic)*delvcic+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vbbp)*delvbbp+ 
                         *(ckt->CKTstate0 + here->HICUMith_Veie)*delveie+ 
                         *(ckt->CKTstate0 + here->HICUMith_Vrth)*delvrth; 
                /*
                 *    bypass if solution has not changed
                 */
                /* the following collections of if's would be just one
                 * if the average compiler could handle it, but many
                 * find the expression too complicated, thus the split.
                 * ... no bypass in case of selfheating
                 */
                if( (ckt->CKTbypass) && (!(ckt->CKTmode & MODEINITPRED)) && !model->HICUMselfheat &&
                        (fabs(delvbiei) < (ckt->CKTreltol*MAX(fabs(Vbiei),
                            fabs(*(ckt->CKTstate0 + here->HICUMvbiei)))+
                            ckt->CKTvoltTol)) )
                    if( (fabs(delvbici) < ckt->CKTreltol*MAX(fabs(Vbici),
                            fabs(*(ckt->CKTstate0 + here->HICUMvbici)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbpei) < ckt->CKTreltol*MAX(fabs(Vbpei),
                            fabs(*(ckt->CKTstate0 + here->HICUMvbpei)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbpbi) < ckt->CKTreltol*MAX(fabs(Vbpbi),
                            fabs(*(ckt->CKTstate0 + here->HICUMvbpbi)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvsici) < ckt->CKTreltol*MAX(fabs(Vsici),
                            fabs(*(ckt->CKTstate0 + here->HICUMvsici)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvcic) < ckt->CKTreltol*MAX(fabs(Vcic),
                            fabs(*(ckt->CKTstate0 + here->HICUMvcic)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbbp) < ckt->CKTreltol*MAX(fabs(Vbbp),
                            fabs(*(ckt->CKTstate0 + here->HICUMvbbp)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delveie) < ckt->CKTreltol*MAX(fabs(Veie),
                            fabs(*(ckt->CKTstate0 + here->HICUMveie)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(ibieihat-*(ckt->CKTstate0 + here->HICUMibiei)) <
                            ckt->CKTreltol* MAX(fabs(ibieihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibiei)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibpeihat-*(ckt->CKTstate0 + here->HICUMibpei)) <
                            ckt->CKTreltol* MAX(fabs(ibpeihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibpei)))+
                            ckt->CKTabstol) )
                    if( (fabs(icieihat-*(ckt->CKTstate0 + here->HICUMiciei)) <
                            ckt->CKTreltol* MAX(fabs(icieihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMiciei)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibicihat-*(ckt->CKTstate0 + here->HICUMibici)) <
                            ckt->CKTreltol* MAX(fabs(ibicihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibici)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibpcihat-*(ckt->CKTstate0 + here->HICUMibpci)) <
                            ckt->CKTreltol* MAX(fabs(ibpcihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibpci)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibpbihat-*(ckt->CKTstate0 + here->HICUMibpbi)) <
                            ckt->CKTreltol* MAX(fabs(ibpbihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibpbi)))+
                            ckt->CKTabstol) )
                    if( (fabs(isicihat-*(ckt->CKTstate0 + here->HICUMisici)) <
                            ckt->CKTreltol* MAX(fabs(isicihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMisici)))+
                            ckt->CKTabstol) )
                    if( (fabs(ithhat-*(ckt->CKTstate0 + here->HICUMith)) <
                            ckt->CKTreltol* MAX(fabs(ithhat),
                            fabs(*(ckt->CKTstate0 + here->HICUMith)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibpsihat-*(ckt->CKTstate0 + here->HICUMibpsi)) <
                            ckt->CKTreltol* MAX(fabs(ibpsihat),
                            fabs(*(ckt->CKTstate0 + here->HICUMibpsi)))+
                            ckt->CKTabstol) ) {
                    /*
                     * bypassing....
                     */
                    Vbiei = *(ckt->CKTstate0 + here->HICUMvbiei);
                    Vbici = *(ckt->CKTstate0 + here->HICUMvbici);
                    Vbpei = *(ckt->CKTstate0 + here->HICUMvbpei);
                    Vbpbi = *(ckt->CKTstate0 + here->HICUMvbpbi);
                    Vbpci = *(ckt->CKTstate0 + here->HICUMvbpci);
                    Vsici = *(ckt->CKTstate0 + here->HICUMvsici);
                    Vcic  = *(ckt->CKTstate0 + here->HICUMvcic);
                    Vbbp  = *(ckt->CKTstate0 + here->HICUMvbbp);
                    Veie  = *(ckt->CKTstate0 + here->HICUMveie);
                    Vrth  = *(ckt->CKTstate0 + here->HICUMvrth);
                    Vxf   = *(ckt->CKTstate0 + here->HICUMvxf);
                    Vxf1  = *(ckt->CKTstate0 + here->HICUMvxf1);
                    Vxf2  = *(ckt->CKTstate0 + here->HICUMvxf2);

                    Ibiei       = *(ckt->CKTstate0 + here->HICUMibiei);
                    Ibiei_Vbiei = *(ckt->CKTstate0 + here->HICUMibiei_Vbiei);
                    Ibiei_Vxf   = *(ckt->CKTstate0 + here->HICUMibiei_Vxf);
                    Ibiei_Vbici = *(ckt->CKTstate0 + here->HICUMibiei_Vbici);
                    Ibiei_Vrth  = *(ckt->CKTstate0 + here->HICUMibiei_Vrth);

                    Ibpei       = *(ckt->CKTstate0 + here->HICUMibpei);
                    Ibpei_Vbpei = *(ckt->CKTstate0 + here->HICUMibpei_Vbpei);
                    Ibpei_Vrth  = *(ckt->CKTstate0 + here->HICUMibpei_Vrth);

                    Iciei       = *(ckt->CKTstate0 + here->HICUMiciei);
                    Iciei_Vbiei = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
                    Iciei_Vbici = *(ckt->CKTstate0 + here->HICUMiciei_Vbici);
                    Iciei_Vrth  = *(ckt->CKTstate0 + here->HICUMiciei_Vrth);
                    Iciei_Vxf2  = *(ckt->CKTstate0 + here->HICUMiciei_Vxf2);

                    Ibici       = *(ckt->CKTstate0 + here->HICUMibici);
                    Ibici_Vbici = *(ckt->CKTstate0 + here->HICUMibici_Vbici);
                    Ibici_Vbiei = *(ckt->CKTstate0 + here->HICUMibici_Vbiei);
                    Ibici_Vrth  = *(ckt->CKTstate0 + here->HICUMibici_Vrth);

                    Ibpbi       = *(ckt->CKTstate0 + here->HICUMibpbi);
                    Ibpbi_Vbpbi = *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi);
                    Ibpbi_Vbiei = *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei);
                    Ibpbi_Vbici = *(ckt->CKTstate0 + here->HICUMibpbi_Vbici);
                    Ibpbi_Vrth    = *(ckt->CKTstate0 + here->HICUMibpbi_Vrth);

                    Isici       = *(ckt->CKTstate0 + here->HICUMisici);
                    Isici_Vsici = *(ckt->CKTstate0 + here->HICUMisici_Vsici);
                    Isici_Vrth    = *(ckt->CKTstate0 + here->HICUMisici_Vrth);

                    Ibpsi       = *(ckt->CKTstate0 + here->HICUMibpsi);
                    Ibpsi_Vbpci = *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci);
                    Ibpsi_Vsici = *(ckt->CKTstate0 + here->HICUMibpsi_Vsici);
                    Ibpsi_Vrth  = *(ckt->CKTstate0 + here->HICUMibpsi_Vrth);

                    Ibpci       = *(ckt->CKTstate0 + here->HICUMibpci);
                    Ibpci_Vbpci = *(ckt->CKTstate0 + here->HICUMibpci_Vbpci);
                    Ibpci_Vrth  = *(ckt->CKTstate0 + here->HICUMibpci_Vrth);

                    //dead assign:
                    //Ieie        = *(ckt->CKTstate0 + here->HICUMieie);
                    Ieie_Vrth   = *(ckt->CKTstate0 + here->HICUMieie_Vrth);

                    Isis_Vsis   = *(ckt->CKTstate0 + here->HICUMisis_Vsis);

                    gqbepar1    = *(ckt->CKTstate0 + here->HICUMgqbepar1);
                    gqbepar2    = *(ckt->CKTstate0 + here->HICUMgqbepar2);
                    gqbcpar1    = *(ckt->CKTstate0 + here->HICUMgqbcpar1);
                    gqbcpar2    = *(ckt->CKTstate0 + here->HICUMgqbcpar2);

                    Ixf        = *(ckt->CKTstate0 + here->HICUMixf)         ;
                    Ixf_Vbiei  = *(ckt->CKTstate0 + here->HICUMixf_Vbiei)   ;
                    Ixf_Vbici  = *(ckt->CKTstate0 + here->HICUMixf_Vbici)   ;
                    Ixf_Vxf    = *(ckt->CKTstate0 + here->HICUMixf_Vxf)     ;
                    Ixf_dT     = *(ckt->CKTstate0 + here->HICUMixf_Vrth)    ;
                    Ixf1       = *(ckt->CKTstate0 + here->HICUMixf1)        ;
                    Ixf1_Vbiei = *(ckt->CKTstate0 + here->HICUMixf1_Vbiei)  ;
                    Ixf1_Vbici = *(ckt->CKTstate0 + here->HICUMixf1_Vbici)  ;
                    Ixf1_Vxf2  = *(ckt->CKTstate0 + here->HICUMixf1_Vxf2)   ;
                    Ixf1_Vxf1  = *(ckt->CKTstate0 + here->HICUMixf1_Vxf1)   ;
                    Ixf1_dT    = *(ckt->CKTstate0 + here->HICUMixf1_Vrth)   ;
                    Ixf2       = *(ckt->CKTstate0 + here->HICUMixf2)        ;
                    Ixf2_Vbiei = *(ckt->CKTstate0 + here->HICUMixf2_Vbiei)  ;
                    Ixf2_Vbici = *(ckt->CKTstate0 + here->HICUMixf2_Vbici)  ;
                    Ixf2_Vxf1  = *(ckt->CKTstate0 + here->HICUMixf2_Vxf1)   ;
                    Ixf2_Vxf2  = *(ckt->CKTstate0 + here->HICUMixf2_Vxf2)   ;
                    Ixf2_dT    = *(ckt->CKTstate0 + here->HICUMixf2_Vrth)   ;

                    goto load;
                }
                /*
                 *   limit nonlinear branch voltages
                 */
                ichk1 = 1, ichk2 = 1, ichk3 = 1, ichk4 = 1, ichk5=1, ichk6 = 0;
                Vbiei = DEVpnjlim(Vbiei,*(ckt->CKTstate0 + here->HICUMvbiei),here->HICUMvt.rpart,
                        here->HICUMtVcrit,&icheck);
                Vaval = 3 * here->HICUMvdci_t.rpart;//limit step around 3*vdci_t -> somehow this brings convergence
                if ((model->HICUMfavlGiven) && (Vbici < MIN(0, -Vaval))) {
                    Vbici_temp = -(Vbici + Vaval);
                    Vbici_temp = DEVpnjlim(
                            Vbici_temp,
                            -(*(ckt->CKTstate0 + here->HICUMvbici) + Vaval),
                            here->HICUMvt.rpart,
                            here->HICUMtVcrit,
                            &ichk1
                    );
                    Vbici      = -(Vbici_temp + Vaval);
                } else {
                    Vbici = DEVpnjlim(Vbici,*(ckt->CKTstate0 + here->HICUMvbici),here->HICUMvt.rpart,
                            here->HICUMtVcrit,&ichk1);
                }
                Vbpei = DEVpnjlim(Vbpei,*(ckt->CKTstate0 + here->HICUMvbpei),here->HICUMvt.rpart,
                        here->HICUMtVcrit,&ichk2);
                Vbpci = DEVpnjlim(Vbpci,*(ckt->CKTstate0 + here->HICUMvbpci),here->HICUMvt.rpart,
                        here->HICUMtVcrit,&ichk3);
                Vsici = DEVpnjlim(Vsici,*(ckt->CKTstate0 + here->HICUMvsici),here->HICUMvt.rpart,
                        here->HICUMtVcrit,&ichk4);
                Vbpbi = DEVpnjlim(Vbpbi,*(ckt->CKTstate0 + here->HICUMvbpbi),here->HICUMvt.rpart,
                        here->HICUMtVcrit,&ichk5);
                if (model->HICUMselfheat) {
                    Vrth = DEVlimitlog(Vrth,
                        *(ckt->CKTstate0 + here->HICUMvrth),1,&ichk6);
                }
                if ((ichk1 == 1) || (ichk2 == 1) || (ichk3 == 1) || (ichk4 == 1) || (ichk5 == 1) || (ichk6 == 1)) icheck=1;
            }
            /*
             *   determine dc current and derivatives
             */
            Vbiei = model->HICUMtype*Vbiei;
            Vbici = model->HICUMtype*Vbici;
            Vciei = (Vbiei-Vbici);
            Vbpei = model->HICUMtype*Vbpei;
            Vbpci = model->HICUMtype*Vbpci;
            Vbci  = model->HICUMtype*Vbci;
            Vsici = model->HICUMtype*Vsici;
            Vsc   = model->HICUMtype*Vsc;

            // Thermal update
            if (model->HICUMselfheat) {
                Temp =  here->HICUMtemp+Vrth;
                hicum_thermal_update(model, here, &Temp, &Tdev_Vrth);

                here->HICUMdtemp_sh  = Temp - here->HICUMtemp;
                here->HICUMtemp_Vrth = Tdev_Vrth;
            } else {
                Temp =  here->HICUMtemp;
                Tdev_Vrth = 0;
                here->HICUMdtemp_sh  = 0;
                here->HICUMtemp_Vrth = 0;
            }

            Temp_dual = Temp + e1*Tdev_Vrth;

            // Model_evaluation

            //Intrinsic transistor
            //Internal base currents across b-e junction
            hicum_diode(Temp_dual,here->HICUMibeis_t,model->HICUMmbei, Vbiei, &ibei, &ibei_Vbiei, &ibei_dT);
            hicum_diode(Temp_dual,here->HICUMireis_t,model->HICUMmrei, Vbiei, &irei, &irei_Vbiei, &irei_dT);

            //Internal b-e and b-c junction capacitances and charges
            //Cjei    = ddx(Qjei,V(bi));
            hicum_qjmodf(Temp_dual,here->HICUMcjei0_t,here->HICUMvdei_t,model->HICUMzei,here->HICUMajei_t,Vbiei,&Cjei,&Cjei_Vbiei, &Cjei_dT,&Qjei, &Qjei_Vbiei, &Qjei_dT);

            result         = calc_hjei_vbe(Vbiei+e1, Temp, here, model);
            hjei_vbe       = result.rpart();
            hjei_vbe_Vbiei = result.dpart();
            result         = calc_hjei_vbe(Vbiei, Temp_dual, here, model);
            hjei_vbe_dT    = result.dpart();


            //Cjci    = ddx(Qjci,V(bi));
            hicum_HICJQ(Temp_dual, here->HICUMcjci0_t,here->HICUMvdci_t,model->HICUMzci,here->HICUMvptci_t, Vbici, &Cjci, &Cjci_Vbici, &Cjci_dT, &Qjci, &Qjci_Vbici, &Qjci_dT);

            //Hole charge at low bias
            result    = calc_Q_0(Temp     , Qjei+e1*Qjei_Vbiei, Qjci, hjei_vbe+e1*hjei_vbe_Vbiei);
            Q_0       = result.rpart();
            Q_0_Vbiei = result.dpart();

            result    = calc_Q_0(Temp    , Qjei, Qjci+e1*Qjci_Vbici, hjei_vbe);
            Q_0_Vbici = result.dpart();

            result       = calc_Q_0(Temp_dual   , Qjei+e1*Qjei_dT, Qjci+e1*Qjci_dT,  hjei_vbe+e1*hjei_vbe_dT);
            Q_0_dT       = result.dpart();

            //Transit time calculation at low current density
            result      = calc_T_f0(Temp, Vbici+e1);
            T_f0        = result.rpart();
            T_f0_Vbici  = result.dpart();

            result      = calc_T_f0(Temp_dual, Vbici);
            T_f0_dT     = result.dpart() ;

            //Critical current
            result      = calc_ick(Temp, Vciei+e1);
            ick         = result.rpart();
            ick_Vciei   = result.dpart();

            result      = calc_ick(Temp_dual, Vciei);
            ick_dT      = result.dpart();

            here->HICUMick = ick;

            //begin Q_pT calculation

            //Initial formulation of forward and reverse component of transfer current

            Tr = model->HICUMtr;

            //begin initial transfer current calculations -> itf, itr, Qf, Qr------------
            calc_it_initial(Temp_dual, Vbiei    , Vbici    , Q_0+e1*Q_0_dT    , T_f0+e1*T_f0_dT    , ick+e1*ick_dT    , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_a_h, &result_Q_p, &result_Tf);
            itf    = result_itf.rpart();
            itr    = result_itr.rpart();
            Qf     = result_Qf.rpart();
            Qr     = result_Qr.rpart();
            Q_bf   = result_Q_bf.rpart();
            a_h    = result_a_h.rpart(); //needed to check if newton iteration needed
            Q_p    = result_Q_p.rpart(); //needed to check if newton iteration needed
            Tf     = result_Tf.rpart();  //needed to check if newton iteration needed
            // dead assign:
            //itf_dT = result_itf.dpart();
            //itr_dT = result_itr.dpart();
            //Qf_dT  = result_Qf.dpart();
            //Qr_dT  = result_Qr.dpart();
            //Q_bf_dT= result_Q_bf.dpart();
            //Tf_dT  = result_Tf.dpart();

            if (!(Qf > RTOLC*Q_p || a_h > RTOLC)) { // in this case the newton is not run and the derivatives of the initial solution are needed
                calc_it_initial(Temp_dual, Vbiei    , Vbici    , Q_0+e1*Q_0_dT      , T_f0+e1*T_f0_dT     , ick+e1*ick_dT      , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_a_h, &result_Q_p, &result_Tf);
                itf_dT  = result_itf.dpart();
                itr_dT  = result_itr.dpart();
                Qf_dT   = result_Qf.dpart();
                Qr_dT   = result_Qr.dpart();
                Q_bf_dT = result_Q_bf.dpart();
                Tf_dT   = result_Tf.dpart();

                calc_it_initial(Temp    , Vbiei+e1, Vbici    , Q_0+e1*Q_0*Vbiei    , T_f0                 ,  ick+e1*ick_Vciei , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_a_h, &result_Q_p, &result_Tf);
                itf_Vbiei  = result_itf.dpart();
                itr_Vbiei  = result_itr.dpart();
                Qf_Vbiei   = result_Qf.dpart();
                Qr_Vbiei   = result_Qr.dpart();
                Q_bf_Vbiei = result_Q_bf.dpart();
                Tf_Vbiei   = result_Tf.dpart();

                calc_it_initial(Temp    , Vbiei    , Vbici+e1, Q_0+e1*Q_0_Vbici    , T_f0+e1*T_f0_Vbici  , ick-e1*ick_Vciei  , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_a_h, &result_Q_p, &result_Tf);
                itf_Vbici = result_itf.dpart();
                itr_Vbici = result_itr.dpart();
                Qf_Vbici  = result_Qf.dpart();
                Qr_Vbici  = result_Qr.dpart();
                Q_bf_Vbici= result_Q_bf.dpart();
                Tf_Vbici  = result_Tf.dpart();

            } else { //Newton needed
                result      = calc_it(Temp_dual, Vbiei    , Vbici    , Q_0+e1*Q_0_dT    , T_f0+e1*T_f0_dT    , ick+e1*ick_dT    );
                Q_pT        = result.rpart();
                Q_pT_dT     = result.dpart();
                result      = calc_it(Temp    , Vbiei+e1, Vbici    , Q_0+e1*Q_0_Vbiei , T_f0                , ick+e1*ick_Vciei );
                Q_pT_dVbiei = result.dpart();
                result      = calc_it(Temp    , Vbiei    , Vbici+e1, Q_0+e1*Q_0_Vbici , T_f0+e1*T_f0_Vbici , ick-e1*ick_Vciei );
                Q_pT_dVbici = result.dpart();

                //end Q_pT -------------------------------------------------------------------------------

                //begin final transfer current calculations -> itf, itr, Qf, Qr------------
                calc_it_final(Temp_dual, Vbiei    , Vbici     , Q_pT+e1*Q_pT_dT    , T_f0+e1*T_f0_dT    , ick+e1*ick_dT    , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_Tf);
                itf    = result_itf.rpart();
                itr    = result_itr.rpart();
                Qf     = result_Qf.rpart();
                Qr     = result_Qr.rpart();
                Q_bf   = result_Q_bf.rpart();
                Tf     = result_Tf.rpart();
                itf_dT = result_itf.dpart();
                itr_dT = result_itr.dpart();
                Qf_dT  = result_Qf.dpart();
                Qr_dT  = result_Qr.dpart();
                Q_bf_dT= result_Q_bf.dpart();
                Tf_dT  = result_Tf.dpart();

                calc_it_final(Temp    , Vbiei+e1 , Vbici     , Q_pT+e1*Q_pT_dVbiei , T_f0               , ick+e1*ick_Vciei    , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_Tf);
                itf_Vbiei  = result_itf.dpart();
                itr_Vbiei  = result_itr.dpart();
                Qf_Vbiei   = result_Qf.dpart();
                Qr_Vbiei   = result_Qr.dpart();
                Q_bf_Vbiei = result_Q_bf.dpart();
                Tf_Vbiei   = result_Tf.dpart();

                calc_it_final(Temp    , Vbiei     , Vbici+e1, Q_pT+e1*Q_pT_dVbici  , T_f0+e1*T_f0_Vbici , ick-e1*ick_Vciei    , &result_itf, &result_itr, &result_Qf, &result_Qr, &result_Q_bf, &result_Tf);
                itf_Vbici = result_itf.dpart();
                itr_Vbici = result_itr.dpart();
                Qf_Vbici  = result_Qf.dpart();
                Qr_Vbici  = result_Qr.dpart();
                Q_bf_Vbici= result_Q_bf.dpart();
                Tf_Vbici  = result_Tf.dpart();

            }

            // finally the transfer current
            it       = itf       - itr;
            it_Vbiei = itf_Vbiei - itr_Vbiei;
            it_Vbici = itf_Vbici - itr_Vbici;
            it_dT    = itf_dT    - itr_dT;

            //end final calculations --------------------------------------------------

            here->HICUMtf = Tf;

            //NQS effect implemented with gyrator network
            //Once the delay in ITF is considered, IT_NQS is calculated afterwards

            //Diffusion charges for further use (remember derivatives if this will be used someday)
            Qdei       = Qf;
            Qdei_Vbici = Qf_Vbici;
            Qdei_Vbiei = Qf_Vbiei;
            Qdei_dT    = Qf_dT;
            //Qdci    = Qr; //we just use Qr herein

            //High-frequency emitter current crowding (lateral NQS)
            Cdei       = T_f0*itf/here->HICUMvt.rpart;
            Cdei_Vbiei = T_f0*itf_Vbiei/here->HICUMvt.rpart;
            Cdei_Vbici = T_f0_Vbici*itf/here->HICUMvt.rpart;
            Cdei_Vbici+= T_f0*itf_Vbici/here->HICUMvt.rpart;
            Cdei_Vrth  = T_f0_dT*itf/here->HICUMvt.rpart + T_f0*itf_dT/here->HICUMvt.rpart - T_f0*itf/here->HICUMvt.rpart/here->HICUMvt.rpart*here->HICUMvt.dpart;

            Cdci       = model->HICUMtr*itr/here->HICUMvt.rpart;
            Cdci_Vbiei = model->HICUMtr*itr_Vbiei/here->HICUMvt.rpart;
            Cdci_Vbici = model->HICUMtr*itr_Vbici/here->HICUMvt.rpart;
            Cdci_Vrth  = model->HICUMtr*itr_dT/here->HICUMvt.rpart - model->HICUMtr*itr/here->HICUMvt.rpart/here->HICUMvt.rpart*here->HICUMvt.dpart;

            Crbi       = model->HICUMfcrbi*(Cjei+Cjci+Cdei+Cdci);
            Crbi_Vrth  = model->HICUMfcrbi*(Cjei_dT+Cjci_dT+Cdei_Vrth+Cdci_Vrth);
            Crbi_Vbiei = model->HICUMfcrbi*(Cjei_Vbiei+Cdei_Vbiei+Cdci_Vbiei);
            Crbi_Vbici = model->HICUMfcrbi*(Cjci_Vbici+Cdei_Vbici+Cdci_Vbici);

            Qrbi       = Crbi*Vbpbi; //Vbpbi=(Vbpei-Vbiei)=(Vbpci-Vbici)
            Qrbi_Vbpbi = Crbi;
            Qrbi_Vbiei = Vbpbi*Crbi_Vbiei;
            Qrbi_Vbici = Vbpbi*Crbi_Vbici;
            Qrbi_Vrth  = Vbpbi*Crbi_Vrth;

            //HICCR: }

            //Internal base current across b-c junction
            hicum_diode(Temp_dual,here->HICUMibcis_t,model->HICUMmbci, Vbici, &ibci, &ibci_Vbici, &ibci_dT);

            //Avalanche current
            result      = calc_iavl(Vbici+e1, Cjci+e1*Cjci_Vbici, itf+e1*itf_Vbici    , Temp);
            iavl        = result.rpart();
            iavl_Vbici  = result.dpart();

            result      = calc_iavl(Vbici, Cjci, itf+e1*itf_Vbiei    , Temp);
            iavl_Vbiei  = result.dpart();

            result      = calc_iavl(Vbici    , Cjci+e1*Cjci_dT    , itf+e1*itf_dT    , Temp_dual);
            iavl_dT     = result.dpart(); 

            here->HICUMiavl = iavl;

            //Excess base current from recombination at the b-c barrier
            ibh_rec       = Q_bf*Otbhrec;
            ibh_rec_Vbiei = Otbhrec*Q_bf_Vbiei ;
            ibh_rec_Vbici = Otbhrec*Q_bf_Vbici ;
            ibh_rec_dT    = Otbhrec*Q_bf_dT ;

            //internal base resistance
            result    = calc_rbi(Temp_dual, Qjei+e1*Qjei_dT    , Qf+e1*Qf_dT    );
            rbi       = result.rpart();
            rbi_dT    = result.dpart();

            result    = calc_rbi(Temp, Qjei+e1*Qjei_Vbiei    , Qf+e1*Qf_Vbiei    );
            rbi_Vbiei = result.dpart();

            result    = calc_rbi(Temp, Qjei                   , Qf+e1*Qf_Vbici    );
            rbi_Vbici = result.dpart();

            here->HICUMrbi = rbi;

            //Base currents across peripheral b-e junction
            hicum_diode(Temp_dual,here->HICUMibeps_t,model->HICUMmbep, Vbpei, &ibep, &ibep_Vbpei, &ibep_dT);
            hicum_diode(Temp_dual,here->HICUMireps_t,model->HICUMmrep, Vbpei, &irep, &irep_Vbpei, &irep_dT);

            //Peripheral b-e junction capacitance and charge
            hicum_qjmodf(Temp_dual,here->HICUMcjep0_t,here->HICUMvdep_t,model->HICUMzep,here->HICUMajep_t,Vbpei,&Cjep,&Cjep_Vbpei, &Cjep_dT,&Qjep, &Qjep_Vbpei, &Qjep_dT);

            //Tunneling current
            result      = calc_ibet(Vbiei    , Vbpei+e1, Temp);
            ibet        = result.rpart();
            ibet_Vbpei  = result.dpart();

            result      = calc_ibet(Vbiei+e1, Vbpei, Temp);
            ibet_Vbiei  = result.dpart();

            result      = calc_ibet(Vbiei    , Vbpei, Temp_dual);
            ibet_dT     = result.dpart();

            //Base currents across peripheral b-c junction (bp,ci)
            hicum_diode(Temp_dual,here->HICUMibcxs_t,model->HICUMmbcx, Vbpci, &ijbcx, &ijbcx_Vbpci, &ijbcx_dT);

            //Depletion capacitance and charge at external b-c junction (b,ci)
            hicum_HICJQ(Temp_dual, here->HICUMcjcx01_t,here->HICUMvdcx_t,model->HICUMzcx,here->HICUMvptcx_t, Vbci, &Cjcx_i, &Cjcx_i_Vbci, &Cjcx_i_dT, &Qjcx_i, &Qjcx_i_Vbci, &Qjcx_i_dT);

            //Depletion capacitance and charge at peripheral b-c junction (bp,ci)
            hicum_HICJQ(Temp_dual, here->HICUMcjcx02_t,here->HICUMvdcx_t,model->HICUMzcx,here->HICUMvptcx_t, Vbpci, &Cjcx_ii, &Cjcx_ii_Vbpci, &Cjcx_ii_dT, &Qjcx_ii, &Qjcx_ii_Vbpci, &Qjcx_ii_dT);

            //Depletion substrate capacitance and charge at inner s-c junction (si,ci)
            hicum_HICJQ(Temp_dual, here->HICUMcjs0_t,here->HICUMvds_t,model->HICUMzs,here->HICUMvpts_t, Vsici, &Cjs, &Cjs_Vsici, &Cjs_dT, &Qjs, &Qjs_Vsici, &Qjs_dT);
            /* 
             * Peripheral substrate capacitance and charge at s-c junction (s,c)
             * Bias dependent only if model->HICUMvdsp > 0
             */
            if (model->HICUMvdsp > 0) {
                hicum_HICJQ(Temp_dual, here->HICUMcscp0_t,here->HICUMvdsp_t,model->HICUMzsp,here->HICUMvptsp_t, Vsc, &Cscp, &Cscp_Vsc, &Cscp_dT, &Qscp, &Qscp_Vsc, &Qscp_dT);
            } else {
                // Constant, temperature independent capacitance
                Cscp     = model->HICUMcscp0;
                Cscp_Vsc = 0;
                Cscp_dT  = 0;
                Qscp     = model->HICUMcscp0*Vsc;
                Qscp_Vsc = model->HICUMcscp0;
                Qscp_dT  = 0;
            }

            //Parasitic substrate transistor transfer current and diffusion charge
            calc_itss(Temp_dual, Vbpci    , Vsici    , &result_HSI_TSU, &result_Qdsu);
            HSI_Tsu          = result_HSI_TSU.rpart();
            Qdsu             = result_Qdsu.rpart();
            HSI_Tsu_dT       = result_HSI_TSU.dpart();
            Qdsu_dT          = result_Qdsu.dpart();

            calc_itss(Temp    , Vbpci+e1, Vsici    , &result_HSI_TSU, &result_Qdsu);
            HSI_Tsu_Vbpci    = result_HSI_TSU.dpart();
            Qdsu_Vbpci       = result_Qdsu.dpart();
            calc_itss(Temp    , Vbpci    , Vsici+e1, &result_HSI_TSU, &result_Qdsu);
            HSI_Tsu_Vsici    = result_HSI_TSU.dpart();
            Qdsu_Vsici       = result_Qdsu.dpart();

            // Current gain computation for correlated noise implementation
            if (ibei > 0.0) {
                here->HICUMbetadc=it/ibei;
            } else {
                here->HICUMbetadc=0.0;
            }
            if (here->HICUMre_t.rpart >= MIN_R) {
                Ieie = Veie/here->HICUMre_t.rpart; // only needed for re flicker noise
            } else {
                Ieie = 0.0;
            }

            //Diode current for s-c junction (si,ci)
            hicum_diode(Temp_dual,here->HICUMiscs_t,model->HICUMmsc, Vsici, &ijsc, &ijsc_Vsici, &ijsc_Vrth);

            // Self-heating calculation
            if (model->HICUMflsh == 1 && here->HICUMrth_scaled >= MIN_R) {
                pterm   =  (Vbiei-Vbici)*it    + (here->HICUMvdci_t.rpart-Vbici)*iavl;
                pterm_dT = (Vbiei-Vbici)*it_dT + (here->HICUMvdci_t.rpart-Vbici)*iavl_dT + here->HICUMvdci_t.dpart*iavl;
            } else if (model->HICUMflsh == 2 && here->HICUMrth_scaled >= MIN_R) {
                pterm   =  (Vbiei-Vbici)*it    + (here->HICUMvdci_t.rpart-Vbici)*iavl + ibei*Vbiei + ibci*Vbici + ibep*Vbpei + ijbcx*Vbpci + ijsc*Vsici;
                pterm_dT = (Vbiei-Vbici)*it_dT + (here->HICUMvdci_t.rpart-Vbici)*iavl_dT + here->HICUMvdci_t.dpart*iavl +
                    ibei_dT*Vbiei + ibci_dT*Vbici + ibep_dT*Vbpei + ijbcx_dT*Vbpci + ijsc_Vrth*Vsici;

                if (rbi > 0.0) {
                    pterm    += Vbpbi*Vbpbi/rbi;
                    pterm_dT += -Vbpbi*Vbpbi*rbi_dT/rbi/rbi;
                }
                if (here->HICUMre_t.rpart >= MIN_R) {
                    pterm    += Veie*Veie/here->HICUMre_t.rpart;
                    pterm_dT += -Veie*Veie*here->HICUMre_t.dpart/here->HICUMre_t.rpart/here->HICUMre_t.rpart;
                }
                if (here->HICUMrcx_t.rpart >= MIN_R) {
                    pterm    += Vcic*Vcic/here->HICUMrcx_t.rpart;
                    pterm_dT += -Vcic*Vcic*here->HICUMrcx_t.dpart/here->HICUMrcx_t.rpart/here->HICUMrcx_t.rpart;
                }
                if (here->HICUMrbx_t.rpart >= MIN_R) {
                    pterm    += Vbbp*Vbbp/here->HICUMrbx_t.rpart;
                    pterm_dT += -Vbbp*Vbbp*here->HICUMrbx_t.dpart/here->HICUMrbx_t.rpart/here->HICUMrbx_t.rpart;
                }
            } else {
                pterm = 0; // default value...
                pterm_dT = 0;
            }
            here->HICUMpterm = pterm;

            Itxf        = itf;
            Itxf_Vbici  = itf_Vbici;
            Itxf_Vbiei  = itf_Vbiei;
            Itxf_dT     = itf_dT;
            Itxf_Vxf2   = 0.0;
            Qdeix       = Qdei;
            Qdeix_Vbiei = Qdei_Vbiei;
            Qdeix_Vbici = Qdei_Vbici;
            Qdeix_dT    = Qdei_dT;
            Qdeix_Vxf   = 0.0;

            // Excess Phase calculation -> hand implementation instead of dual numbers
            if (model->HICUMnqs) { // && (ckt->CKTmode & (MODETRAN | MODEAC) ) ) { //evaluate nqs network only in TRANSIENT and AC modes.
                Ixf1       = (Vxf2-itf)/Tf*model->HICUMt0;
                Ixf1_Vxf1  =  0.0;
                Ixf1_Vxf2  =  1.0/Tf*model->HICUMt0;
                Ixf1_ditf  = -Ixf1_Vxf2;
                Ixf1_dTf   = -Ixf1/Tf; //Tf(Vbiei,Vciei=Vbiei-Vbici)
                Ixf1_Vbiei = Ixf1_ditf*itf_Vbiei + Ixf1_dTf*Tf_Vbiei;
                Ixf1_Vbici = Ixf1_ditf*itf_Vbici + Ixf1_dTf*Tf_Vbici;
                Ixf1_dT    = Ixf1_ditf*itf_dT    + Ixf1_dTf*Tf_dT;

                Ixf2       = (Vxf2-Vxf1)/Tf*model->HICUMt0;
                Ixf2_Vxf2  = 1.0/Tf*model->HICUMt0;
                Ixf2_Vxf1  = -Ixf2_Vxf2;
                Ixf2_dTf   = -Ixf2/Tf;
                Ixf2_Vbiei = Ixf2_dTf*Tf_Vbiei;
                Ixf2_Vbici = Ixf2_dTf*Tf_Vbici;
                Ixf2_dT    = Ixf2_dTf*Tf_dT;

                Qxf1       = model->HICUMalit*model->HICUMt0*Vxf1;
                Qxf1_Vxf1  = model->HICUMalit*model->HICUMt0;
                Qxf2       = model->HICUMalit*model->HICUMt0*Vxf2/3;
                Qxf2_Vxf2  = model->HICUMalit*model->HICUMt0/3;

                Itxf       = Vxf2;
                Itxf_Vbiei = 0;
                Itxf_Vbici = 0;
                Itxf_dT    = 0;
                Itxf_Vxf2  = 1.0;

                Ixf       = (Vxf - Qdei)*model->HICUMt0/Tf;      //for RC nw
                Ixf_Vxf   = 1.0*model->HICUMt0/Tf;
                Ixf_Qdei  = -Ixf_Vxf;
                Ixf_Tf    = -Ixf/Tf;
                Ixf_Vbiei = Ixf_Tf*Tf_Vbiei + Ixf_Qdei * Qdei_Vbiei;
                Ixf_Vbici = Ixf_Tf*Tf_Vbici + Ixf_Qdei * Qdei_Vbici;
                Ixf_dT    = Ixf_Tf*Tf_dT               + Ixf_Qdei * Qdei_dT;

                Qxf       = model->HICUMalqf*model->HICUMt0*Vxf; //for RC nw
                Qxf_Vxf   = model->HICUMalqf*model->HICUMt0;     //for RC nw

                Qdeix       = Vxf;                               //for RC nw
                Qdeix_Vxf   = 1.0;
                Qdeix_Vbiei = 0;
                Qdeix_Vbici = 0;
                Qdeix_dT    = 0;
            } else {
                Ixf1       = Vxf1;
                Ixf1_Vxf1  = 1.0;
                Ixf1_Vxf2  = 0;
                // dead assign:
                //Ixf1_ditf  = 0;
                //Ixf1_dTf   = 0;
                Ixf1_Vbiei = 0;
                Ixf1_Vbici = 0;
                Ixf1_dT    = 0;

                Ixf2       = Vxf2;
                Ixf2_Vxf2  = 1.0;
                Ixf2_Vxf1  = 0;
                // dead assign:
                //Ixf2_dTf   = 0;
                Ixf2_Vbiei = 0;
                Ixf2_Vbici = 0;
                Ixf2_dT    = 0;

                Qxf1       = 0;
                Qxf1_Vxf1  = 0;

                Qxf2       = 0;
                Qxf2_Vxf2  = 0;

                Ixf        = Vxf;
                Ixf_Vxf    = 1.0;
                // dead assign:
                //Ixf_Tf     = 0;
                //Ixf_Qdei   = 0;
                Ixf_Vbiei  = 0;
                Ixf_Vbici  = 0;
                Ixf_dT     = 0;

                Qxf        = 0;
                Qxf_Vxf    = 0;

            }

            // end of Model_evaluation

            //resistors
            if(model->HICUMrcxGiven && model->HICUMrcx != 0) {
                Icic_Vcic    = 1/here->HICUMrcx_t.rpart;
                Icic_Vrth    = -Vcic*here->HICUMrcx_t.dpart/here->HICUMrcx_t.rpart/here->HICUMrcx_t.rpart;
            } else {
                Icic_Vcic    = 0.0;
                Icic_Vrth    = 0.0;
            }
            if(model->HICUMrbxGiven && model->HICUMrbx != 0) {
                Ibbp_Vbbp    = 1/here->HICUMrbx_t.rpart;
                Ibbp_Vrth    = -Vbbp*here->HICUMrbx_t.dpart/here->HICUMrbx_t.rpart/here->HICUMrbx_t.rpart;
            } else {
                Ibbp_Vbbp    = 0.0;
                Ibbp_Vrth    = 0.0;
            }
            if(model->HICUMreGiven && model->HICUMre != 0) {
                Ieie_Veie    = 1/here->HICUMre_t.rpart;
                Ieie_Vrth    = -Veie*here->HICUMre_t.dpart /here->HICUMre_t.rpart/here->HICUMre_t.rpart;
            } else {
                Ieie_Veie    = 0.0;
                Ieie_Vrth    = 0.0;
            }
            if(model->HICUMrsuGiven && model->HICUMrsu != 0) {
                Isis_Vsis    = 1/model->HICUMrsu*here->HICUMm;
            } else {
                Isis_Vsis    = 0.0;
            }
            if(model->HICUMselfheat){
                Irth_Vrth   = (1/here->HICUMrth_t.rpart - Vrth/(here->HICUMrth_t.rpart*here->HICUMrth_t.rpart) * here->HICUMrth_t.dpart);
            } else {
                Irth_Vrth   = 0.0;
            }
 
            Ibpei        = model->HICUMtype*ibep;
            Ibpei       += model->HICUMtype*irep;
            Ibpei_Vbpei  = model->HICUMtype*ibep_Vbpei;
            Ibpei_Vbpei += model->HICUMtype*irep_Vbpei;
            Ibpei_Vrth   = model->HICUMtype*ibep_dT;
            Ibpei_Vrth  += model->HICUMtype*irep_dT;

            Ibiei        = model->HICUMtype*ibei;
            Ibiei_Vbiei  = model->HICUMtype*ibei_Vbiei;
            Ibiei_Vrth   = model->HICUMtype*ibei_dT;
            Ibiei       += model->HICUMtype*irei;
            Ibiei_Vbiei += model->HICUMtype*irei_Vbiei;
            Ibiei_Vrth  += model->HICUMtype*irei_dT;
            Ibiei       += model->HICUMtype*ibh_rec;
            Ibiei_Vbiei += model->HICUMtype*ibh_rec_Vbiei;
            Ibiei_Vrth  += model->HICUMtype*ibh_rec_dT;
            Ibiei_Vbici  = model->HICUMtype*ibh_rec_Vbici;

            if (model->HICUMtunode==1.0) {
                Ibpei       += -model->HICUMtype*ibet;
                Ibpei_Vbpei += -model->HICUMtype*ibet_Vbpei;
                Ibpei_Vrth  += -model->HICUMtype*ibet_dT;
            } else {
                Ibiei       += -model->HICUMtype*ibet;
                Ibiei_Vbiei += -model->HICUMtype*ibet_Vbiei;
                Ibiei_Vrth  += -model->HICUMtype*ibet_dT;
            }

            Ibpsi       = model->HICUMtype*HSI_Tsu;
            Ibpsi_Vbpci = model->HICUMtype*HSI_Tsu_Vbpci;
            Ibpsi_Vsici = model->HICUMtype*HSI_Tsu_Vsici;
            Ibpsi_Vrth  = model->HICUMtype*HSI_Tsu_dT;

            Ibpci       = model->HICUMtype*ijbcx;
            Ibpci_Vbpci = model->HICUMtype*ijbcx_Vbpci;
            Ibpci_Vrth  = model->HICUMtype*ijbcx_dT;

            Ibici       = model->HICUMtype*(ibci       - iavl);
            Ibici_Vbici = model->HICUMtype*(ibci_Vbici - iavl_Vbici); 
            Ibici_Vbiei = model->HICUMtype*(           - iavl_Vbiei); 
            Ibici_Vrth  = model->HICUMtype*(ibci_dT    - iavl_dT);

            Isici       = model->HICUMtype*ijsc;
            Isici_Vsici = model->HICUMtype*ijsc_Vsici;
            Isici_Vrth  = model->HICUMtype*ijsc_Vrth;

            Iciei       =  model->HICUMtype*(Itxf       - itr);
            Iciei_Vbiei =  model->HICUMtype*(Itxf_Vbiei - itr_Vbiei);
            Iciei_Vbici =  model->HICUMtype*(Itxf_Vbici - itr_Vbici);
            Iciei_Vrth  =  model->HICUMtype*(Itxf_dT    - itr_dT);
            Iciei_Vxf2  =  model->HICUMtype*Itxf_Vxf2;

            if (rbi > 0.0) {
                Ibpbi         = Vbpbi / rbi;
                Ibpbi_Vbpbi   = 1 / rbi;
                Ibpbi_Vbiei   = -Vbpbi * rbi_Vbiei / (rbi*rbi); 
                Ibpbi_Vbici   = -Vbpbi * rbi_Vbici / (rbi*rbi);
                Ibpbi_Vrth    = -Vbpbi * rbi_dT    / (rbi*rbi);
            } else {
                // fallback in case bp-bi nodes are collapsed
                Ibpbi       = 0.0;
                Ibpbi_Vbpbi = 0.0;
                Ibpbi_Vbiei = 0.0;
                Ibpbi_Vbici = 0.0;
                Ibpbi_Vrth  = 0.0;
            };

            Ith_Vbiei  = 0.0;
            Ith_Vbici  = 0.0;
            Ith_Vbpbi  = 0.0;
            Ith_Vbpci  = 0.0;
            Ith_Vbpei  = 0.0;
            Ith_Vciei  = 0.0;
            Ith_Vsici  = 0.0;
            Ith_Vcic   = 0.0;
            Ith_Vbbp   = 0.0;
            Ith_Veie   = 0.0;
            Ith_Vrth   = 0.0;
            if(!model->HICUMselfheat) {
                Ith      = 0;
            } else {
                Ith      = pterm; //Current from gnd to T
                Ith_Vrth = pterm_dT;
                if (model->HICUMflsh == 1) {
                    //it(Vbiei,Vbici)*(Vbiei-Vbici)
                    Ith_Vbiei  += it_Vbiei*(Vbiei-Vbici) + it;
                    Ith_Vbici  += it_Vbici*(Vbiei-Vbici) - it;
                    //avalanche current 
                    Ith_Vbici  += (here->HICUMvdci_t.rpart-Vbici)*iavl_Vbici - iavl;
                    Ith_Vbiei  += (here->HICUMvdci_t.rpart-Vbici)*iavl_Vbiei;
                } else if (model->HICUMflsh == 2) {
                    //remember:
                    //pterm = (Vbiei-Vbici)*it + (here->HICUMvdci_t.rpart-Vbici)*iavl + ibei*Vbiei + ibci*Vbici + ibep*Vbpei + ijbcx*Vbpci + ijsc*Vsici;
                    //it(Vbiei,Vbici)*(Vbiei-Vbici) 
                    Ith_Vbiei  += it_Vbiei*(Vbiei-Vbici) + it;
                    Ith_Vbici  += it_Vbici*(Vbiei-Vbici) - it;
                    //Vbiei*Ibiei(Vbiei)
                    Ith_Vbiei  += ibei + ibei_Vbiei*Vbiei;
                    //Vbici*Ibici(Vbici,Vbiei)
                    Ith_Vbici  += ibci + ibci_Vbici*Vbici;
                    Ith_Vbici  += (here->HICUMvdci_t.rpart-Vbici)*iavl_Vbici - iavl;
                    Ith_Vbiei  += (here->HICUMvdci_t.rpart-Vbici)*iavl_Vbiei;
                    //Vbpei*Ibpei(Vbpei)
                    Ith_Vbpei  += ibep + ibep_Vbpei*Vbpei;
                    //Vpbci*Ibpci(Vbpci)
                    Ith_Vbpci  += Ibpci + Ibpci_Vbpci*Vbpci;
                    //Vsici*Isici(Vsici)
                    Ith_Vsici  += Isici + Isici_Vsici*Vsici;
                    if (rbi > 0.0) {
                        //Vbpbi*Ibpbi(Vbpbi,Vbiei,Vbici)
                        Ith_Vbpbi  += Ibpbi + Ibpbi_Vbpbi*Vbpbi;
                        Ith_Vbiei  +=         Ibpbi_Vbiei*Vbpbi;
                        Ith_Vbici  +=         Ibpbi_Vbici*Vbpbi;
                    }
                    if (here->HICUMre_t.rpart >= MIN_R) {
                        Ith_Veie   = 2*Veie/here->HICUMre_t.rpart;
                    }
                    if (here->HICUMrcx_t.rpart >= MIN_R) {
                        Ith_Vcic   = 2*Vcic/here->HICUMrcx_t.rpart;
                    }
                    if (here->HICUMrbx_t.rpart >= MIN_R) {
                        Ith_Vbbp   = 2*Vbbp/here->HICUMrbx_t.rpart;
                    }
                }

            }
            // ********************************************

            // add gmin over parallel to all non-linear branches
            Ibiei += ckt->CKTgmin*Vbiei;
            Ibiei_Vbiei += ckt->CKTgmin;
            Ibici += ckt->CKTgmin*Vbici;
            Ibici_Vbici += ckt->CKTgmin;
            Ibpei += ckt->CKTgmin*Vbpei;
            Ibpei_Vbpei += ckt->CKTgmin;
            Ibpci += ckt->CKTgmin*Vbpci;
            Ibpci_Vbpci += ckt->CKTgmin;
            Isici += ckt->CKTgmin*Vsici;
            Isici_Vsici += ckt->CKTgmin;

            // calculate charge derivatives for electrostatic caps
            Qbepar1_Vbe      = cbepar1;
            Qbepar2_Vbpe     = cbepar2;
            Qbcpar1_Vbci     = cbcpar1;
            Qbcpar2_Vbpci    = cbcpar2;
            Qsu_Vsis         = model->HICUMcsu;
            Qcth_Vrth        = here->HICUMcth_scaled;
            if( (ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN | MODEAC)) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ||
                    (ckt->CKTmode & MODEINITSMSIG)) {

                // calculate charges over electrostatic caps
                Qbepar1 = cbepar1*Vbe;
                Qbepar2 = cbepar2*Vbpe;
                Qbcpar1 = cbcpar1*Vbci;
                Qbcpar2 = cbcpar2*Vbpci;
                Qsu     = model->HICUMcsu*Vsis;
                if (model->HICUMselfheat) {
                    Qcth    = here->HICUMcth_scaled*Vrth;
                } else {
                    Qcth    = 0;
                }

                // store charges and their derivatives in state vector
                //Qrbi
                *(ckt->CKTstate0 + here->HICUMqrbi)         = Qrbi;
                *(ckt->CKTstate0 + here->HICUMcqrbi)        = Qrbi_Vbpbi;
                //Qjei
                *(ckt->CKTstate0 + here->HICUMqjei)         = Qjei;
                *(ckt->CKTstate0 + here->HICUMcqjei)        = Qjei_Vbiei;
                //Qf
                *(ckt->CKTstate0 + here->HICUMqf)           = Qdeix;
                *(ckt->CKTstate0 + here->HICUMcqf)          = Qdeix_Vbiei;
                //Qr
                *(ckt->CKTstate0 + here->HICUMqr)           = Qr;
                *(ckt->CKTstate0 + here->HICUMcqr)          = Qr_Vbici;
                //Qjci
                *(ckt->CKTstate0 + here->HICUMqjci)         = Qjci;
                *(ckt->CKTstate0 + here->HICUMcqjci)        = Qjci_Vbici;
                //Qjep
                *(ckt->CKTstate0 + here->HICUMqjep)         = Qjep;
                *(ckt->CKTstate0 + here->HICUMcqjep)        = Qjep_Vbpei;
                //Qjcx_i
                *(ckt->CKTstate0 + here->HICUMqjcx0_i)      = Qjcx_i;
                *(ckt->CKTstate0 + here->HICUMcqcx0_t_i)    = Qjcx_i_Vbci;
                //Qjcx_ii
                *(ckt->CKTstate0 + here->HICUMqjcx0_ii)     = Qjcx_ii;
                *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii)   = Qjcx_ii_Vbpci;
                //Qdsu
                *(ckt->CKTstate0 + here->HICUMqdsu)         = Qdsu;
                *(ckt->CKTstate0 + here->HICUMcqdsu)        = Qdsu_Vbpci;
                //Qs
                *(ckt->CKTstate0 + here->HICUMqjs)          = Qjs;
                *(ckt->CKTstate0 + here->HICUMcqjs)         = Qjs_Vsici;
                //Qscp
                *(ckt->CKTstate0 + here->HICUMqscp)         = Qscp;
                *(ckt->CKTstate0 + here->HICUMcqscp)        = Qscp_Vsc;
                //Qbepar1
                *(ckt->CKTstate0 + here->HICUMqbepar1)      = Qbepar1;
                *(ckt->CKTstate0 + here->HICUMcqbepar1)     = Qbepar1_Vbe;
                //Qbepar2
                *(ckt->CKTstate0 + here->HICUMqbepar2)      = Qbepar2;
                *(ckt->CKTstate0 + here->HICUMcqbepar2)     = Qbepar2_Vbpe;
                //Qbcpar1
                *(ckt->CKTstate0 + here->HICUMqbcpar1)      = Qbcpar1;
                *(ckt->CKTstate0 + here->HICUMcqbcpar1)     = Qbcpar1_Vbci;
                //Qbcpar2
                *(ckt->CKTstate0 + here->HICUMqbcpar2)      = Qbcpar2;
                *(ckt->CKTstate0 + here->HICUMcqbcpar2)     = Qbcpar2_Vbpci;
                //Qsu
                *(ckt->CKTstate0 + here->HICUMqsu)          = Qsu;
                *(ckt->CKTstate0 + here->HICUMcqsu)         = Qsu_Vsis;

                *(ckt->CKTstate0 + here->HICUMqxf1)         = Qxf1;
                *(ckt->CKTstate0 + here->HICUMcqxf1)        = Qxf1_Vxf1;
                *(ckt->CKTstate0 + here->HICUMqxf2)         = Qxf2;
                *(ckt->CKTstate0 + here->HICUMcqxf2)        = Qxf2_Vxf2;
                *(ckt->CKTstate0 + here->HICUMqxf)          = Qxf;
                *(ckt->CKTstate0 + here->HICUMcqxf)         = Qxf_Vxf;
                if (model->HICUMselfheat) {
                    //Qth
                    *(ckt->CKTstate0 + here->HICUMqcth)         = Qcth;
                    *(ckt->CKTstate0 + here->HICUMcqcth)        = Qcth_Vrth;
                }

                //below is not important for load
                here->HICUMcaprbi      = Qrbi_Vbpbi;
                here->HICUMcapdeix     = Cdei;
                here->HICUMcapjei      = Cjei;
                here->HICUMcapdci      = Cdci;
                here->HICUMcapjci      = Cjci;
                here->HICUMcapjep      = Cjep;
                here->HICUMcapjcx_t_i  = Cjcx_i;
                here->HICUMcapjcx_t_ii = Cjcx_ii;
                here->HICUMcapdsu      = Qdsu_Vbpci;
                here->HICUMcapjs       = Cjs;
                here->HICUMcapscp      = Cscp;
                here->HICUMcapsu       = model->HICUMcsu;
                here->HICUMcapcth      = here->HICUMcth_scaled;

                //derivatives of charges due to cross coupling
                here->HICUMqrbi_Vbiei    = Qrbi_Vbiei;
                here->HICUMqrbi_Vbici    = Qrbi_Vbici;
                here->HICUMqrbi_Vrth     = Qrbi_Vrth;
                here->HICUMqjei_Vrth     = Qjei_dT;
                here->HICUMqjep_Vrth     = Qjep_dT;
                here->HICUMqf_Vbici      = Qdeix_Vbici;
                here->HICUMqf_Vxf        = Qdeix_Vxf;
                here->HICUMqf_Vrth       = Qdeix_dT;
                here->HICUMqr_Vbiei      = Qr_Vbiei;
                here->HICUMqr_Vrth       = Qr_dT;
                here->HICUMqjci_Vrth     = Qjci_dT;
                here->HICUMqjcx0_i_Vrth  = Qjcx_i_dT;
                here->HICUMqjcx0_ii_Vrth = Qjcx_ii_dT;
                here->HICUMqdsu_Vrth     = Qdsu_dT;
                here->HICUMqdsu_Vsici    = Qdsu_Vsici;
                here->HICUMqjs_Vrth      = Qjs_dT;
                here->HICUMqscp_Vrth     = Qscp_dT;
                here->HICUMicth_dT       = Icth_Vrth;

                /*
                 *   store small-signal parameters
                 */
                if ( (!(ckt->CKTmode & MODETRANOP))||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {

                        *(ckt->CKTstate0 + here->HICUMcqrbi)      = Qrbi_Vbpbi;
                        *(ckt->CKTstate0 + here->HICUMcqf)        = Qdeix_Vbiei;
                        *(ckt->CKTstate0 + here->HICUMcqr)        = Qr_Vbici;

                        *(ckt->CKTstate0 + here->HICUMcqjei)      = Cjei;
                        *(ckt->CKTstate0 + here->HICUMcqjci)      = Cjci;
                        *(ckt->CKTstate0 + here->HICUMcqjep)      = Qjep_Vbpei;
                        *(ckt->CKTstate0 + here->HICUMcqcx0_t_i)  = Qjcx_i_Vbci;
                        *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii) = Qjcx_ii_Vbpci;
                        *(ckt->CKTstate0 + here->HICUMcqdsu)      = Qdsu_Vbpci;
                        *(ckt->CKTstate0 + here->HICUMcqjs)       = Qjs_Vsici;
                        *(ckt->CKTstate0 + here->HICUMcqscp)      = Qscp_Vsc;
                        *(ckt->CKTstate0 + here->HICUMcqbepar1)   = Qbepar1_Vbe;
                        *(ckt->CKTstate0 + here->HICUMcqbepar2)   = Qbepar2_Vbpe;
                        *(ckt->CKTstate0 + here->HICUMcqbcpar1)   = Qbcpar1_Vbci;
                        *(ckt->CKTstate0 + here->HICUMcqbcpar2)   = Qbcpar2_Vbpci;
                        *(ckt->CKTstate0 + here->HICUMcqsu)       = Qsu_Vsis;

                        *(ckt->CKTstate0 + here->HICUMcqxf1)      = Qxf1_Vxf1;
                        *(ckt->CKTstate0 + here->HICUMcqxf2)      = Qxf2_Vxf2;
                        *(ckt->CKTstate0 + here->HICUMcqxf)       = Qxf_Vxf;
                        if (model->HICUMselfheat)
                            *(ckt->CKTstate0 + here->HICUMcqcth)  = here->HICUMcth_scaled;
                        continue; /* go to 1000 */
                    }
                    //transient analysis
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->HICUMqrbi)         = 
                            *(ckt->CKTstate0 + here->HICUMqrbi)         ; 
                        *(ckt->CKTstate1 + here->HICUMqjei)         = 
                            *(ckt->CKTstate0 + here->HICUMqjei)         ; 
                        *(ckt->CKTstate1 + here->HICUMqf)           =
                            *(ckt->CKTstate0 + here->HICUMqf)           ;
                        *(ckt->CKTstate1 + here->HICUMqr)           =
                            *(ckt->CKTstate0 + here->HICUMqr)           ;
                        *(ckt->CKTstate1 + here->HICUMqjci)         =
                            *(ckt->CKTstate0 + here->HICUMqjci)         ;
                        *(ckt->CKTstate1 + here->HICUMqjep)         =
                            *(ckt->CKTstate0 + here->HICUMqjep)         ;
                        *(ckt->CKTstate1 + here->HICUMqjcx0_i)      =
                            *(ckt->CKTstate0 + here->HICUMqjcx0_i)      ;
                        *(ckt->CKTstate1 + here->HICUMqjcx0_ii)     =
                            *(ckt->CKTstate0 + here->HICUMqjcx0_ii)     ;
                        *(ckt->CKTstate1 + here->HICUMqdsu)         =
                            *(ckt->CKTstate0 + here->HICUMqdsu)         ;
                        *(ckt->CKTstate1 + here->HICUMqjs)          =
                            *(ckt->CKTstate0 + here->HICUMqjs)          ;
                        *(ckt->CKTstate1 + here->HICUMqscp)         =
                            *(ckt->CKTstate0 + here->HICUMqscp)         ;
                        *(ckt->CKTstate1 + here->HICUMqbepar1)      =
                            *(ckt->CKTstate0 + here->HICUMqbepar1)      ;
                        *(ckt->CKTstate1 + here->HICUMqbepar2)      =
                            *(ckt->CKTstate0 + here->HICUMqbepar2)      ;
                        *(ckt->CKTstate1 + here->HICUMqbcpar1)      =
                            *(ckt->CKTstate0 + here->HICUMqbcpar1)      ;
                        *(ckt->CKTstate1 + here->HICUMqbcpar2)      =
                            *(ckt->CKTstate0 + here->HICUMqbcpar2)      ;
                        *(ckt->CKTstate1 + here->HICUMqsu)          =
                            *(ckt->CKTstate0 + here->HICUMqsu)          ;
                        *(ckt->CKTstate1 + here->HICUMqxf1)     =
                            *(ckt->CKTstate0 + here->HICUMqxf1)     ;
                        *(ckt->CKTstate1 + here->HICUMqxf2)     =
                            *(ckt->CKTstate0 + here->HICUMqxf2)     ;
                        *(ckt->CKTstate1 + here->HICUMqxf)      =
                            *(ckt->CKTstate0 + here->HICUMqxf)      ;

                        if (model->HICUMselfheat)
                            *(ckt->CKTstate1 + here->HICUMqcth) =
                                *(ckt->CKTstate0 + here->HICUMqcth) ;
                    }

                    //Integrate all charges and add the displacement current to the branch currents
                    //Qrbi
                    error = NIintegrate(ckt,&geq,&ceq,Qrbi_Vbpbi,here->HICUMqrbi);
                    if(error) return(error);
                    Ibpbi_Vbpbi += geq;
                    Ibpbi += *(ckt->CKTstate0 + here->HICUMcqrbi);
                    //Qjei
                    error = NIintegrate(ckt,&geq,&ceq,Cjei,here->HICUMqjei);
                    if(error) return(error);
                    Ibiei_Vbiei += geq;
                    Ibiei += *(ckt->CKTstate0 + here->HICUMcqjei);
                    //Qdeix
                    if (!model->HICUMnqs) {
                        error = NIintegrate(ckt,&geq,&ceq,Qdeix_Vbiei,here->HICUMqf);
                        if(error) return(error);
                        Ibiei_Vbiei += geq;
                        Ibiei_Vxf    = 0;
                        Ibiei += *(ckt->CKTstate0 + here->HICUMcqf);
                    } else { //Qdeix is now a function of Vxf(t)
                        error = NIintegrate(ckt,&geq,&ceq,Qdeix_Vxf,here->HICUMqf);
                        if(error) return(error);
                        Ibiei_Vxf = geq;
                        Ibiei     += *(ckt->CKTstate0 + here->HICUMcqf);
                    }
                    //Qr
                    error = NIintegrate(ckt,&geq,&ceq,Qr_Vbici,here->HICUMqr);
                    if(error) return(error);
                    Ibici_Vbici += geq;
                    Ibici += *(ckt->CKTstate0 + here->HICUMcqr);
                    //Qjci
                    error = NIintegrate(ckt,&geq,&ceq,Cjci,here->HICUMqjci);
                    if(error) return(error);
                    Ibici_Vbici += geq;
                    Ibici += *(ckt->CKTstate0 + here->HICUMcqjci);
                    //Qjep
                    error = NIintegrate(ckt,&geq,&ceq,Cjep,here->HICUMqjep);
                    if(error) return(error);
                    Ibpei_Vbpei += geq;
                    Ibpei += *(ckt->CKTstate0 + here->HICUMcqjep);
                    //Qjcx_i
                    error = NIintegrate(ckt,&geq,&ceq,Cjcx_i,here->HICUMqjcx0_i);
                    if(error) return(error);
                    Ibci_Vbci = geq;
                    Ibci = *(ckt->CKTstate0 + here->HICUMcqcx0_t_i);
                    //Qjcx_ii
                    error = NIintegrate(ckt,&geq,&ceq,Cjcx_ii,here->HICUMqjcx0_ii);
                    if(error) return(error);
                    Ibpci_Vbpci += geq;
                    Ibpci += *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii);
                    //Qdsu
                    error = NIintegrate(ckt,&geq,&ceq,Qdsu_Vbpci,here->HICUMqdsu);
                    if(error) return(error);
                    Ibpci_Vbpci += geq;
                    Ibpci += *(ckt->CKTstate0 + here->HICUMcqdsu);
                    //Qjs
                    error = NIintegrate(ckt,&geq,&ceq,Cjs,here->HICUMqjs);
                    if(error) return(error);
                    Isici_Vsici += geq;
                    Isici       += *(ckt->CKTstate0 + here->HICUMcqjs);
                    //Qscp
                    error = NIintegrate(ckt,&geq,&ceq,Cscp,here->HICUMqscp);
                    if(error) return(error);
                    Isc_Vsc = geq;
                    Isc     = *(ckt->CKTstate0 + here->HICUMcqscp);
                    if (model->HICUMnqs) {
                        //Qxf1
                        error = NIintegrate(ckt,&geq,&ceq,Qxf1_Vxf1,here->HICUMqxf1);
                        if(error) return(error);
                        Ixf1_Vxf1 += geq;
                        Ixf1 += *(ckt->CKTstate0 + here->HICUMcqxf1);
                        //Qxf2
                        error = NIintegrate(ckt,&geq,&ceq,Qxf2_Vxf2,here->HICUMqxf2);
                        if(error) return(error);
                        Ixf2_Vxf2 += geq;
                        Ixf2 += *(ckt->CKTstate0 + here->HICUMcqxf2);
                        //Qxf
                        error = NIintegrate(ckt,&geq,&ceq,Qxf_Vxf,here->HICUMqxf);
                        if(error) return(error);
                        Ixf_Vxf += geq;
                        Ixf += *(ckt->CKTstate0 + here->HICUMcqxf);
                    }

                    if (model->HICUMselfheat)
                    {
                        //Qth
                        error = NIintegrate(ckt,&geq,&ceq,here->HICUMcth_scaled,here->HICUMqcth);
                        if(error) return(error);
                        Icth_Vrth = geq;
                        Icth = *(ckt->CKTstate0 + here->HICUMcqcth);
                    }

                    if(ckt->CKTmode & MODEINITTRAN) {
                        //copy from state1 to state0
                        *(ckt->CKTstate1 + here->HICUMcqrbi)         = 
                            *(ckt->CKTstate0 + here->HICUMcqrbi)         ; 
                        *(ckt->CKTstate1 + here->HICUMcqjei)         = 
                            *(ckt->CKTstate0 + here->HICUMcqjei)         ; 
                        *(ckt->CKTstate1 + here->HICUMcqf)           =
                            *(ckt->CKTstate0 + here->HICUMcqf)           ;
                        *(ckt->CKTstate1 + here->HICUMcqr)           =
                            *(ckt->CKTstate0 + here->HICUMcqr)           ;
                        *(ckt->CKTstate1 + here->HICUMcqjci)         =
                            *(ckt->CKTstate0 + here->HICUMcqjci)         ;
                        *(ckt->CKTstate1 + here->HICUMcqjep)         =
                            *(ckt->CKTstate0 + here->HICUMcqjep)         ;
                        *(ckt->CKTstate1 + here->HICUMcqcx0_t_i)      =
                            *(ckt->CKTstate0 + here->HICUMcqcx0_t_i)      ;
                        *(ckt->CKTstate1 + here->HICUMcqcx0_t_ii)     =
                            *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii)     ;
                        *(ckt->CKTstate1 + here->HICUMcqdsu)         =
                            *(ckt->CKTstate0 + here->HICUMcqdsu)         ;
                        *(ckt->CKTstate1 + here->HICUMcqjs)          =
                            *(ckt->CKTstate0 + here->HICUMcqjs)          ;
                        *(ckt->CKTstate1 + here->HICUMcqscp)        =
                            *(ckt->CKTstate0 + here->HICUMcqscp)        ;
                        *(ckt->CKTstate1 + here->HICUMcqxf1)     =
                            *(ckt->CKTstate0 + here->HICUMcqxf1)     ;
                        *(ckt->CKTstate1 + here->HICUMcqxf2)     =
                            *(ckt->CKTstate0 + here->HICUMcqxf2)     ;
                        *(ckt->CKTstate1 + here->HICUMcqxf)      =
                            *(ckt->CKTstate0 + here->HICUMcqxf)      ;
                        if (model->HICUMselfheat)
                            *(ckt->CKTstate1 + here->HICUMcqcth) =
                                *(ckt->CKTstate0 + here->HICUMcqcth) ;
                    }
                }
            }

            /*
             *   check convergence
             */
            if ( (!(ckt->CKTmode & MODEINITFIX))||(!(here->HICUMoff))) {
                if (icheck == 1) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }

            /*
             *      charge storage for electrostatic caps 
             */
            if(ckt->CKTmode & (MODETRAN | MODEAC)) {
//            Ibe        += ddt(cbepar1*Vbe);
                error = NIintegrate(ckt,&gqbepar1,&cqbepar1,cbepar1,here->HICUMqbepar1);
                if(error) return(error);
//            Ibpe       += ddt(cbepar2*Vbpe);
                error = NIintegrate(ckt,&gqbepar2,&cqbepar2,cbepar2,here->HICUMqbepar2);
                if(error) return(error);
//            Ibci       += ddt(cbcpar1*Vbci);
                error = NIintegrate(ckt,&gqbcpar1,&cqbcpar1,cbcpar1,here->HICUMqbcpar1);
                if(error) return(error);
//            Ibpci      += ddt(cbcpar2*Vbpci);
                error = NIintegrate(ckt,&gqbcpar2,&cqbcpar2,cbcpar2,here->HICUMqbcpar2);
                if(error) return(error);
//            Isis      += ddt(model->HICUMcsu*Vsis);
                error = NIintegrate(ckt,&gqsu,&cqsu,model->HICUMcsu,here->HICUMqsu);
                if(error) return(error);
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->HICUMcqbepar1) =
                            *(ckt->CKTstate0 + here->HICUMcqbepar1);
                    *(ckt->CKTstate1 + here->HICUMcqbepar2) =
                            *(ckt->CKTstate0 + here->HICUMcqbepar2);
                    *(ckt->CKTstate1 + here->HICUMcqbcpar1) =
                            *(ckt->CKTstate0 + here->HICUMcqbcpar1);
                    *(ckt->CKTstate1 + here->HICUMcqbcpar2) =
                            *(ckt->CKTstate0 + here->HICUMcqbcpar2);
                    *(ckt->CKTstate1 + here->HICUMcqsu) =
                            *(ckt->CKTstate0 + here->HICUMcqsu);
                }
            }

            // Write branch currents and voltages to state vector
            *(ckt->CKTstate0 + here->HICUMvbiei)       = Vbiei;
            *(ckt->CKTstate0 + here->HICUMvbici)       = Vbici;
            *(ckt->CKTstate0 + here->HICUMvbpei)       = Vbpei;
            *(ckt->CKTstate0 + here->HICUMvbpbi)       = Vbpbi;
            *(ckt->CKTstate0 + here->HICUMvbpci)       = Vbpci;
            *(ckt->CKTstate0 + here->HICUMvsici)       = Vsici;
            *(ckt->CKTstate0 + here->HICUMvcic)        = Vcic;
            *(ckt->CKTstate0 + here->HICUMvbbp)        = Vbbp;
            *(ckt->CKTstate0 + here->HICUMveie)        = Veie;
            *(ckt->CKTstate0 + here->HICUMvxf1)        = Vxf1;
            *(ckt->CKTstate0 + here->HICUMvxf2)        = Vxf2;
            *(ckt->CKTstate0 + here->HICUMvxf)         = Vxf;

            *(ckt->CKTstate0 + here->HICUMibiei)       = Ibiei;
            *(ckt->CKTstate0 + here->HICUMibiei_Vbiei) = Ibiei_Vbiei;
            *(ckt->CKTstate0 + here->HICUMibiei_Vxf)   = Ibiei_Vxf;
            *(ckt->CKTstate0 + here->HICUMibiei_Vbici) = Ibiei_Vbici;
            *(ckt->CKTstate0 + here->HICUMibiei_Vrth)  = Ibiei_Vrth;

            *(ckt->CKTstate0 + here->HICUMibpei)       = Ibpei;
            *(ckt->CKTstate0 + here->HICUMibpei_Vbpei) = Ibpei_Vbpei;
            *(ckt->CKTstate0 + here->HICUMibpei_Vrth)  = Ibpei_Vrth;

            *(ckt->CKTstate0 + here->HICUMiciei)       = Iciei;
            *(ckt->CKTstate0 + here->HICUMiciei_Vbiei) = Iciei_Vbiei;
            *(ckt->CKTstate0 + here->HICUMiciei_Vbici) = Iciei_Vbici;
            *(ckt->CKTstate0 + here->HICUMiciei_Vrth)  = Iciei_Vrth;
            *(ckt->CKTstate0 + here->HICUMiciei_Vxf2)  = Iciei_Vxf2;

            *(ckt->CKTstate0 + here->HICUMibici)       = Ibici;
            *(ckt->CKTstate0 + here->HICUMibici_Vbici) = Ibici_Vbici;
            *(ckt->CKTstate0 + here->HICUMibici_Vbiei) = Ibici_Vbiei;
            *(ckt->CKTstate0 + here->HICUMibici_Vrth)  = Ibici_Vrth;

            *(ckt->CKTstate0 + here->HICUMibpbi)       = Ibpbi;
            *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi) = Ibpbi_Vbpbi;
            *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei) = Ibpbi_Vbiei;
            *(ckt->CKTstate0 + here->HICUMibpbi_Vbici) = Ibpbi_Vbici;
            *(ckt->CKTstate0 + here->HICUMibpbi_Vrth)  = Ibpbi_Vrth;

            *(ckt->CKTstate0 + here->HICUMibpci)       = Ibpci;
            *(ckt->CKTstate0 + here->HICUMibpci_Vbpci) = Ibpci_Vbpci;
            *(ckt->CKTstate0 + here->HICUMibpci_Vrth)  = Ibpci_Vrth;

            *(ckt->CKTstate0 + here->HICUMisici)       = Isici;
            *(ckt->CKTstate0 + here->HICUMisici_Vsici) = Isici_Vsici;
            *(ckt->CKTstate0 + here->HICUMisici_Vrth)  = Isici_Vrth;

            *(ckt->CKTstate0 + here->HICUMibpsi)       = Ibpsi;
            *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci) = Ibpsi_Vbpci;
            *(ckt->CKTstate0 + here->HICUMibpsi_Vsici) = Ibpsi_Vsici;
            *(ckt->CKTstate0 + here->HICUMibpsi_Vrth)  = Ibpsi_Vrth;

            *(ckt->CKTstate0 + here->HICUMisis_Vsis)   = Isis_Vsis;

            *(ckt->CKTstate0 + here->HICUMieie)        = Ieie;
            *(ckt->CKTstate0 + here->HICUMieie_Vrth)   = Ieie_Vrth;

            *(ckt->CKTstate0 + here->HICUMcqcth)       = Icth;
            *(ckt->CKTstate0 + here->HICUMvrth)        = Vrth;

            *(ckt->CKTstate0 + here->HICUMgqbepar1)    = gqbepar1;
            *(ckt->CKTstate0 + here->HICUMgqbepar2)    = gqbepar2;

            *(ckt->CKTstate0 + here->HICUMgqbcpar1)    = gqbcpar1;
            *(ckt->CKTstate0 + here->HICUMgqbcpar2)    = gqbcpar2;

            *(ckt->CKTstate0 + here->HICUMgqsu)        = gqsu;

            *(ckt->CKTstate0 + here->HICUMith)         = Ith;
            *(ckt->CKTstate0 + here->HICUMith_Vrth)    = Ith_Vrth;
            *(ckt->CKTstate0 + here->HICUMith_Vbiei)   = Ith_Vbiei;
            *(ckt->CKTstate0 + here->HICUMith_Vbici)   = Ith_Vbici;
            *(ckt->CKTstate0 + here->HICUMith_Vbpbi)   = Ith_Vbpbi;
            *(ckt->CKTstate0 + here->HICUMith_Vbpci)   = Ith_Vbpci;
            *(ckt->CKTstate0 + here->HICUMith_Vbpei)   = Ith_Vbpei;
            *(ckt->CKTstate0 + here->HICUMith_Vciei)   = Ith_Vciei;
            *(ckt->CKTstate0 + here->HICUMith_Vsici)   = Ith_Vsici;
            *(ckt->CKTstate0 + here->HICUMith_Vcic)    = Ith_Vcic;
            *(ckt->CKTstate0 + here->HICUMith_Vbbp)    = Ith_Vbbp;
            *(ckt->CKTstate0 + here->HICUMith_Veie)    = Ith_Veie;

            *(ckt->CKTstate0 + here->HICUMixf)         = Ixf;
            *(ckt->CKTstate0 + here->HICUMixf_Vbiei)   = Ixf_Vbiei;
            *(ckt->CKTstate0 + here->HICUMixf_Vbici)   = Ixf_Vbici;
            *(ckt->CKTstate0 + here->HICUMixf_Vxf)     = Ixf_Vxf;
            *(ckt->CKTstate0 + here->HICUMixf_Vrth)    = Ixf_dT;

            *(ckt->CKTstate0 + here->HICUMixf1)        = Ixf1;
            *(ckt->CKTstate0 + here->HICUMixf1_Vbiei)  = Ixf1_Vbiei;
            *(ckt->CKTstate0 + here->HICUMixf1_Vbici)  = Ixf1_Vbici;
            *(ckt->CKTstate0 + here->HICUMixf1_Vxf2)   = Ixf1_Vxf2;
            *(ckt->CKTstate0 + here->HICUMixf1_Vxf1)   = Ixf1_Vxf1;
            *(ckt->CKTstate0 + here->HICUMixf1_Vrth)   = Ixf1_dT;

            *(ckt->CKTstate0 + here->HICUMixf2)        = Ixf2;
            *(ckt->CKTstate0 + here->HICUMixf2_Vbiei)  = Ixf2_Vbiei;
            *(ckt->CKTstate0 + here->HICUMixf2_Vbici)  = Ixf2_Vbici;
            *(ckt->CKTstate0 + here->HICUMixf2_Vxf1)   = Ixf2_Vxf1;
            *(ckt->CKTstate0 + here->HICUMixf2_Vxf2)   = Ixf2_Vxf2;
            *(ckt->CKTstate0 + here->HICUMixf2_Vrth)   = Ixf2_dT;
            *(ckt->CKTstate0 + here->HICUMit)          = it;

load:
//          #############################################################
//          ############### STAMPS NO SH   ##############################
//          ############################################################# 

//          Branch: sc, Stamp element: Cscp 
            rhs_current = model->HICUMtype * (Isc - Isc_Vsc*Vsc);
            *(ckt->CKTrhs + here->HICUMsubsNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollNode) +=  rhs_current;
            // with respect to Vsc
            *(here->HICUMsubsSubsPtr)            +=  Isc_Vsc;
            *(here->HICUMcollCollPtr)            +=  Isc_Vsc;
            *(here->HICUMsubsCollPtr)            += -Isc_Vsc;
            *(here->HICUMcollSubsPtr)            += -Isc_Vsc;

//          Branch: bci, Stamp element: Qbcx
            rhs_current = model->HICUMtype * (Ibci - Ibci_Vbci*Vbci);
            *(ckt->CKTrhs + here->HICUMbaseNode)   += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            //with respect to Vbci
            *(here->HICUMbaseBasePtr)              +=  Ibci_Vbci;
            *(here->HICUMcollCICollCIPtr)          +=  Ibci_Vbci;
            *(here->HICUMbaseCollCIPtr)            += -Ibci_Vbci;
            *(here->HICUMcollCIBasePtr)            += -Ibci_Vbci;

//          Branch: be, Stamp element: Cbepar1 
            rhs_current = model->HICUMtype * (*(ckt->CKTstate0 + here->HICUMcqbepar1) - Vbe * gqbepar1);
            *(ckt->CKTrhs + here->HICUMbaseNode)   += -rhs_current;
            *(ckt->CKTrhs + here->HICUMemitNode)   +=  rhs_current;
            // with respect to Vbe
            *(here->HICUMbaseBasePtr)              +=  gqbepar1;
            *(here->HICUMemitEmitPtr)              +=  gqbepar1;
            *(here->HICUMbaseEmitPtr)              += -gqbepar1;
            *(here->HICUMemitBasePtr)              += -gqbepar1;

//          Branch: bpe, Stamp element: Cbepar2 
            rhs_current = model->HICUMtype * (*(ckt->CKTstate0 + here->HICUMcqbepar2) - Vbpe * gqbepar2);
            *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMemitNode)   +=  rhs_current;
            // with respect to Vbpe
            *(here->HICUMbaseBPBaseBPPtr)          +=  gqbepar2;
            *(here->HICUMemitEmitPtr)              +=  gqbepar2;
            *(here->HICUMbaseBPEmitPtr)            += -gqbepar2;
            *(here->HICUMemitBaseBPPtr)            += -gqbepar2;

//          Branch: bci, Stamp element: Cbcpar1
            rhs_current = model->HICUMtype * (*(ckt->CKTstate0 + here->HICUMcqbcpar1) - Vbci * gqbcpar1);
            *(ckt->CKTrhs + here->HICUMbaseNode)   += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            // with respect to Vbci
            *(here->HICUMbaseBasePtr)              +=  gqbcpar1;
            *(here->HICUMcollCICollCIPtr)          +=  gqbcpar1;
            *(here->HICUMbaseCollCIPtr)            += -gqbcpar1;
            *(here->HICUMcollCIBasePtr)            += -gqbcpar1;

//          Branch: bpci, Stamp element: Cbcpar2
            rhs_current = model->HICUMtype * (*(ckt->CKTstate0 + here->HICUMcqbcpar2) - Vbpci * gqbcpar2);
            *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            // with respect to Vbpci
            *(here->HICUMbaseBPBaseBPPtr)          +=  gqbcpar2;
            *(here->HICUMcollCICollCIPtr)          +=  gqbcpar2;
            *(here->HICUMbaseBPCollCIPtr)          += -gqbcpar2;
            *(here->HICUMcollCIBaseBPPtr)          += -gqbcpar2;

//          Branch: ssi, Stamp element: Csu //Markus: I think rhs sign is wrong here
            rhs_current = model->HICUMtype * (*(ckt->CKTstate0 + here->HICUMcqsu) - Vsis * gqsu);
            *(ckt->CKTrhs + here->HICUMsubsNode)   += +rhs_current;
            *(ckt->CKTrhs + here->HICUMsubsSINode) += -rhs_current;
            // with respect to Vsis
            *(here->HICUMsubsSubsPtr)              +=  gqsu;
            *(here->HICUMsubsSISubsSIPtr)          +=  gqsu;
            *(here->HICUMsubsSubsSIPtr)            += -gqsu;
            *(here->HICUMsubsSISubsPtr)            += -gqsu;

//          Branch: biei, Stamp element: Ibiei = Ibei + Irei ( was Ijbei )
            rhs_current = model->HICUMtype * (Ibiei - Ibiei_Vbiei*Vbiei - Ibiei_Vxf*Vxf - Ibiei_Vbici*Vbici);
            *(ckt->CKTrhs + here->HICUMbaseBINode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
            // with respect to Vbiei
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibiei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ibiei_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr)          += -Ibiei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr)          += -Ibiei_Vbiei;
            // with respect to Vbici
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibiei_Vbici;
            *(here->HICUMemitEICollCIPtr)          +=  Ibiei_Vbici;
            *(here->HICUMbaseBICollCIPtr)          += -Ibiei_Vbici;
            *(here->HICUMemitEIBaseBIPtr)          += -Ibiei_Vbici;
            if (model->HICUMnqs) {
                *(here->HICUMbaseBIXfPtr)          +=  Ibiei_Vxf;
                *(here->HICUMemitEIXfPtr)          += -Ibiei_Vxf;
            }

//          Branch: bpei, Stamp element: Ibpei = Ibep + Irep ( was Ijbep )
            rhs_current = model->HICUMtype * (Ibpei - Ibpei_Vbpei*Vbpei);
            *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
            // with respect to Vbpei
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr)          += -Ibpei_Vbpei;
            *(here->HICUMemitEIBaseBPPtr)          += -Ibpei_Vbpei;

//          Branch: bici, Stamp element: Ibici ( was Ijbci )  f_bi=+ f_ci=-
            rhs_current = model->HICUMtype * (Ibici - Ibici_Vbici*Vbici - Ibici_Vbiei*Vbiei);
            *(ckt->CKTrhs + here->HICUMbaseBINode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            // with respect to Vbici
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCICollCIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbici;
            *(here->HICUMbaseBICollCIPtr)          += -Ibici_Vbici;
            // with respect to Vbiei
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr)          += -Ibici_Vbiei;

//          Branch: ciei, Stamp element: It
            rhs_current = model->HICUMtype * (Iciei - Iciei_Vbiei*Vbiei - Iciei_Vbici*Vbici- Iciei_Vxf2*Vxf2);
            *(ckt->CKTrhs + here->HICUMcollCINode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
            // with respect to Vbiei f_CI = +    f_EI = -
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          += -Iciei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbiei;
            // with respect to Vbici
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbici; 
            *(here->HICUMemitEICollCIPtr)          +=  Iciei_Vbici;
            *(here->HICUMcollCICollCIPtr)          += -Iciei_Vbici;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbici;
            if (model->HICUMnqs) {
                // with respect to Vxf2
                *(here->HICUMcollCIXf2Ptr)         +=  Iciei_Vxf2;
                *(here->HICUMemitEIXf2Ptr)         += -Iciei_Vxf2;
            }

//          Branch: bpci, Stamp element: Ibpci ( was Ijbcx )
            rhs_current = model->HICUMtype * (Ibpci - Ibpci_Vbpci*Vbpci);
            *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            // with respect to Vbpci
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpci_Vbpci;
            *(here->HICUMcollCICollCIPtr)          +=  Ibpci_Vbpci;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpci_Vbpci;
            *(here->HICUMcollCIBaseBPPtr)          += -Ibpci_Vbpci;

//          Branch: cic, Stamp element: Rcx
            // with respect to Vcic
            *(here->HICUMcollCollPtr)     +=  Icic_Vcic;
            *(here->HICUMcollCICollCIPtr) +=  Icic_Vcic;
            *(here->HICUMcollCICollPtr)   += -Icic_Vcic;
            *(here->HICUMcollCollCIPtr)   += -Icic_Vcic;

//          Branch: bbp, Stamp element: Rbx
            // with respect to Vbbp
            *(here->HICUMbaseBasePtr)     +=  Ibbp_Vbbp;
            *(here->HICUMbaseBPBaseBPPtr) +=  Ibbp_Vbbp;
            *(here->HICUMbaseBPBasePtr)   += -Ibbp_Vbbp;
            *(here->HICUMbaseBaseBPPtr)   += -Ibbp_Vbbp;

//          Branch: eie, Stamp element: Re
            // with respect to Veie
            *(here->HICUMemitEmitPtr)     +=  Ieie_Veie;
            *(here->HICUMemitEIEmitEIPtr) +=  Ieie_Veie;
            *(here->HICUMemitEIEmitPtr)   += -Ieie_Veie;
            *(here->HICUMemitEmitEIPtr)   += -Ieie_Veie;

            if (rbi > 0.0) {
//              Branch: bpbi, Stamp element: Rbi, Crbi
                rhs_current = model->HICUMtype * (Ibpbi - Ibpbi_Vbpbi*Vbpbi - Ibpbi_Vbiei*Vbiei - Ibpbi_Vbici*Vbici);
                *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMbaseBINode) +=  rhs_current;
                //f_Bp = +    f_Bi = - 
                // with respect to Vbpbi 
                *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpbi_Vbpbi; 
                *(here->HICUMbaseBIBaseBIPtr)          +=  Ibpbi_Vbpbi;
                *(here->HICUMbaseBPBaseBIPtr)          += -Ibpbi_Vbpbi;
                *(here->HICUMbaseBIBaseBPPtr)          += -Ibpbi_Vbpbi;
                // with respect to Vbiei
                *(here->HICUMbaseBPBaseBIPtr)          +=  Ibpbi_Vbiei; 
                *(here->HICUMbaseBIEmitEIPtr)          +=  Ibpbi_Vbiei;
                *(here->HICUMbaseBPEmitEIPtr)          += -Ibpbi_Vbiei;
                *(here->HICUMbaseBIBaseBIPtr)          += -Ibpbi_Vbiei;
                // with respect to Vbici
                *(here->HICUMbaseBPBaseBIPtr)          +=  Ibpbi_Vbici; 
                *(here->HICUMbaseBICollCIPtr)          +=  Ibpbi_Vbici;
                *(here->HICUMbaseBPCollCIPtr)          += -Ibpbi_Vbici;
                *(here->HICUMbaseBIBaseBIPtr)          += -Ibpbi_Vbici;
            }

//          Branch: sici, Stamp element: Ijsc
            rhs_current = model->HICUMtype * (Isici - Isici_Vsici*Vsici);
            *(ckt->CKTrhs + here->HICUMsubsSINode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
            // with respect to Vsici
            *(here->HICUMsubsSISubsSIPtr)          +=  Isici_Vsici;
            *(here->HICUMcollCICollCIPtr)          +=  Isici_Vsici;
            *(here->HICUMsubsSICollCIPtr)          += -Isici_Vsici;
            *(here->HICUMcollCISubsSIPtr)          += -Isici_Vsici;

//          Branch: bpsi, Stamp element: Its
            rhs_current = model->HICUMtype * (Ibpsi - Ibpsi_Vbpci*Vbpci - Ibpsi_Vsici*Vsici);
            *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
            *(ckt->CKTrhs + here->HICUMsubsSINode) +=  rhs_current;
            // f_Bp = +    f_Si = -
            // with respect to Vsici
            *(here->HICUMbaseBPSubsSIPtr)          +=  Ibpsi_Vsici;
            *(here->HICUMsubsSICollCIPtr)          +=  Ibpsi_Vsici;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpsi_Vsici;
            *(here->HICUMsubsSISubsSIPtr)          += -Ibpsi_Vsici;
            // with respect to Vbpci
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpsi_Vbpci;
            *(here->HICUMsubsSICollCIPtr)          +=  Ibpsi_Vbpci;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpsi_Vbpci;
            *(here->HICUMsubsSIBaseBPPtr)          += -Ibpsi_Vbpci;

//           Branch: sis, Stamp element: Rsu
            // with respect to Vsis
            *(here->HICUMsubsSubsPtr)              +=  Isis_Vsis;
            *(here->HICUMsubsSISubsSIPtr)          +=  Isis_Vsis;
            *(here->HICUMsubsSISubsPtr)            += -Isis_Vsis;
            *(here->HICUMsubsSubsSIPtr)            += -Isis_Vsis;

            if (model->HICUMnqs) {
    //          Branch: xf1-ground, Stamp element: Ixf1   f_xf1=+ 
                rhs_current                          = Ixf1 - Ixf1_Vbici*Vbici - Ixf1_Vbiei*Vbiei -Ixf1_Vxf1*Vxf1 - Ixf1_Vxf2*Vxf2;
                *(ckt->CKTrhs + here->HICUMxf1Node) += -rhs_current; // rhs_current; // into xf1 node
                *(here->HICUMxf1BaseBIPtr)          += +Ixf1_Vbiei;
                *(here->HICUMxf1EmitEIPtr)          += -Ixf1_Vbiei;
                *(here->HICUMxf1BaseBIPtr)          += +Ixf1_Vbici;
                *(here->HICUMxf1CollCIPtr)          += -Ixf1_Vbici;
                *(here->HICUMxf1Xf2Ptr)             += +Ixf1_Vxf2;
                *(here->HICUMxf1Xf1Ptr)             += +Ixf1_Vxf1;

    //          Branch: xf2-ground, Stamp element: Ixf2   f_xf2=+  
                rhs_current                          = Ixf2 - Ixf2_Vbici*Vbici - Ixf2_Vbiei*Vbiei - Ixf2_Vxf1*Vxf1 - Ixf2_Vxf2*Vxf2;
                *(ckt->CKTrhs + here->HICUMxf2Node) += -rhs_current; // rhs_current; // into xf2 node
                *(here->HICUMxf2BaseBIPtr)          += +Ixf2_Vbiei;
                *(here->HICUMxf2EmitEIPtr)          += -Ixf2_Vbiei;
                *(here->HICUMxf2BaseBIPtr)          += +Ixf2_Vbici;
                *(here->HICUMxf2CollCIPtr)          += -Ixf2_Vbici;
                *(here->HICUMxf2Xf2Ptr)             += +Ixf2_Vxf2;
                *(here->HICUMxf2Xf1Ptr)             += +Ixf2_Vxf1;

    //          Branch: xf-ground, Stamp element: Ixf   f_xf=+  
                rhs_current = Ixf - Ixf_Vbici*Vbici - Ixf_Vbiei*Vbiei - Ixf_Vxf*Vxf;
                *(ckt->CKTrhs + here->HICUMxfNode) += -rhs_current; // rhs_current; // into xf node
                *(here->HICUMxfBaseBIPtr)          += +Ixf_Vbiei;
                *(here->HICUMxfEmitEIPtr)          += -Ixf_Vbiei;
                *(here->HICUMxfBaseBIPtr)          += +Ixf_Vbici;
                *(here->HICUMxfCollCIPtr)          += -Ixf_Vbici;
                *(here->HICUMxfXfPtr)              += +Ixf_Vxf;
            }


//          #############################################################
//          ############### FINISH STAMPS NO SH #########################
//          ############################################################# 

            if (model->HICUMselfheat) {

//              #############################################################
//              ############### STAMP WITH SH ADDITIONS #####################
//              #############################################################
                
//              Stamp element: Ibiei  f_Bi = +   f_Ei = -
                rhs_current = -Ibiei_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseBINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMbaseBItempPtr)            +=  Ibiei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Ibiei_Vrth;

//              Stamp element: Ibpei  f_Bp = +   f_Ei = -
                rhs_current = -Ibpei_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMbaseBPtempPtr)            +=  Ibpei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Ibpei_Vrth;

//              Stamp element: Ibici  f_Bi = +   f_Ci = -
                rhs_current = -Ibici_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseBINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMbaseBItempPtr)            +=  Ibici_Vrth;
                *(here->HICUMcollCItempPtr)            += -Ibici_Vrth;

//              Stamp element: Iciei  f_Ci = +   f_Ei = -
                rhs_current = -Iciei_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMcollCINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMemitEINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMcollCItempPtr)            +=  Iciei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Iciei_Vrth;

//              Stamp element: Ibpci  f_Bp = +   f_Ci = -
                rhs_current = -Ibpci_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMbaseBPtempPtr)            +=  Ibpci_Vrth;
                *(here->HICUMcollCItempPtr)            += -Ibpci_Vrth;

//              Stamp element: Rcx  f_Ci = +   f_C = -
                rhs_current = -Icic_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMcollCINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMcollNode)   +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMcollCItempPtr)            +=  Icic_Vrth;
                *(here->HICUMcollTempPtr)              += -Icic_Vrth;

//              Stamp element: Rbx  f_B = +   f_Bp = -
                rhs_current = -Ibbp_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseNode)   += -rhs_current;
                *(ckt->CKTrhs + here->HICUMbaseBPNode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMbaseTempPtr)   +=  Ibbp_Vrth;
                *(here->HICUMbaseBPtempPtr) += -Ibbp_Vrth;

//              Stamp element: Re   f_Ei = +   f_E = -
                rhs_current = - Ieie_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMemitEINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMemitNode)   += +rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMemitEItempPtr) += +Ieie_Vrth;
                *(here->HICUMemitTempPtr)   += -Ieie_Vrth;

                if (rbi > 0.0) {
//                  Stamp element: Rbi    f_Bp = +   f_Bi = -
                    rhs_current = -Ibpbi_Vrth*Vrth;
                    *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
                    *(ckt->CKTrhs + here->HICUMbaseBINode) +=  rhs_current;
                    // with respect to Potential Vrth
                    *(here->HICUMbaseBPtempPtr)            +=  Ibpbi_Vrth;
                    *(here->HICUMbaseBItempPtr)            += -Ibpbi_Vrth;
                };

//              Stamp element: Isici   f_Si = +   f_Ci = -
                rhs_current = -Isici_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMsubsSINode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMcollCINode) +=  rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMsubsSItempPtr)            +=  Isici_Vrth;
                *(here->HICUMcollCItempPtr)            += -Isici_Vrth;

//              Branch: bpsi, Stamp element: Its
                rhs_current = - Ibpsi_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMbaseBPNode) += -rhs_current;
                *(ckt->CKTrhs + here->HICUMsubsSINode) +=  rhs_current;
                // f_Bp = +    f_Si = -
                // with respect to Vrth
                *(here->HICUMbaseBPtempPtr)          +=  Ibpsi_Vrth;
                *(here->HICUMsubsSItempPtr)          += -Ibpsi_Vrth;

                if (model->HICUMnqs) {
    //              Stamp element: Ixf    f_xf = +   
                    rhs_current = -Ixf_dT*Vrth;
                    *(ckt->CKTrhs + here->HICUMxfNode) += -rhs_current;
                    // with respect to Potential Vxf
                    *(here->HICUMxfTempPtr)             +=  Ixf_dT;

    //              Stamp element: Ixf1    f_xf1 = +   
                    rhs_current = -Ixf1_dT*Vrth;
                    *(ckt->CKTrhs + here->HICUMxf1Node) += -rhs_current;
                    // with respect to Potential Vxf1
                    *(here->HICUMxf1TempPtr)             +=  Ixf1_dT;

    //              Stamp element: Ixf2    f_xf2 = +   
                    rhs_current = -Ixf2_dT*Vrth;
                    *(ckt->CKTrhs + here->HICUMxf2Node) += -rhs_current;
                    // with respect to Potential Vxf2
                    *(here->HICUMxf2TempPtr)             +=  Ixf2_dT;
                }



//              Stamp element:    Ith f_T = - Ith 
                // Ith        = -Vrth/here->HICUMrth_t.rpart; //Current from gnd to T
                // Ith_Vrth   = (-1/here->HICUMrth_t.rpart + Vrth/(here->HICUMrth_t.rpart*here->HICUMrth_t.rpart) * here->HICUMrth_t.dpart); 
                rhs_current = Ith 
                              - Ith_Vbiei*Vbiei - Ith_Vbici*Vbici - Ith_Vciei*Vciei
                              - Ith_Vbpei*Vbpei - Ith_Vbpci*Vbpci - Ith_Vsici*Vsici
                              - Ith_Vbpbi*Vbpbi
                              - Ith_Vcic*Vcic - Ith_Vbbp*Vbbp - Ith_Veie*Veie
                              - Ith_Vrth*Vrth;

                *(ckt->CKTrhs + here->HICUMtempNode) += rhs_current;
                // with respect to Potential Vrth
                *(here->HICUMtempTempPtr)   += -Ith_Vrth;
                // with respect to Potential Vbiei
                *(here->HICUMtempBaseBIPtr) += -Ith_Vbiei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vbiei;
                // with respect to Potential Vbici
                *(here->HICUMtempBaseBIPtr) += -Ith_Vbici;
                *(here->HICUMtempCollCIPtr) += +Ith_Vbici;
                // with respect to Potential Vciei
                *(here->HICUMtempCollCIPtr) += -Ith_Vciei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vciei;
                // with respect to Potential Vbpei
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vbpei;
                // with respect to Potential Vbpci
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpci;
                *(here->HICUMtempCollCIPtr) += +Ith_Vbpci;
                // with respect to Potential Vsici
                *(here->HICUMtempSubsSIPtr) += -Ith_Vsici;
                *(here->HICUMtempCollCIPtr) += +Ith_Vsici;
                // with respect to Potential Vbpbi
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpbi;
                *(here->HICUMtempBaseBIPtr) += +Ith_Vbpbi;
                // with respect to Potential Vcic
                *(here->HICUMtempCollCIPtr) += -Ith_Vcic;
                *(here->HICUMtempCollPtr)   += +Ith_Vcic;
                // with respect to Potential Vbbp
                *(here->HICUMtempBasePtr)   += -Ith_Vbbp;
                *(here->HICUMtempBaseBPPtr) += +Ith_Vbbp;
                // with respect to Potential Veie
                *(here->HICUMtempEmitEIPtr) += -Ith_Veie;
                *(here->HICUMtempEmitPtr)   += +Ith_Veie;

//              Cth f_T = - (put this separate to Ith to get signs right)
                rhs_current = Icth - Vrth * Icth_Vrth;
                *(ckt->CKTrhs + here->HICUMtempNode) +=  -rhs_current;
                *(here->HICUMtempTempPtr)            +=  +Icth_Vrth;

//              Stamp element:    Rth f_T = +
                rhs_current                           = Vrth/here->HICUMrth_t.rpart - Irth_Vrth*Vrth;
                *(ckt->CKTrhs + here->HICUMtempNode) += -rhs_current;
                *(here->HICUMtempTempPtr)            += Irth_Vrth;
            }
        }

    }
    return(OK);
}
