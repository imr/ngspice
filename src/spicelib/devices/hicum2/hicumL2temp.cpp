/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/


#include "cmath"
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#include <duals/dual>
#include "hicumL2.hpp"
#include <functional>
#include <fenv.h> //trap NAN

//ngspice header files written in C
#ifdef __cplusplus
extern "C"
{
#endif
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "hicum2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"
#ifdef __cplusplus
}
#endif

using namespace duals::literals;

#define TMAX  326.85
#define TMIN -100.0
#define LN_EXP_LIMIT 11.0

void TMPHICJ(double , double , double , double , double ,
             double , double , double , double , double , double ,
             double *, double *, double *);

// TEMPERATURE UPDATE OF JUNCTION CAPACITANCE RELATED PARAMETERS
// INPUT:
//  mostly model parameters
//  x           : zero bias junction capacitance
//  y           : junction built-in potential
//  z           : grading co-efficient
//  w           : ratio of maximum to zero-bias value of capacitance or punch-through voltage
//  is_al       : condition factor to check what "w" stands for
//  vgeff       : band-gap voltage
// IMPLICIT INPUT:
//  vt          : thermal voltage
//  vt0,qtt0,ln_qtt0,mg : other model variables
// OUTPUT:
//  c_j_t               : temperature update of "c_j"
//  vd_t                : temperature update of "vd0"
//  w_t                 : temperature update of "w"
void TMPHICJ(duals::duald vt, double vt0, duals::duald qtt0, duals::duald ln_qtt0, double mg,
             double c_j, double vd0, double z, double w, double is_al, double vgeff,
             duals::duald *c_j_t, duals::duald *vd_t, duals::duald *w_t)
{
duals::duald vdj0,vdjt,vdt;

if (c_j > 0.0) {
        vdj0    = 2*vt0*log(exp(vd0*0.5/vt0)-exp(-0.5*vd0/vt0));
        vdjt    = vdj0*qtt0+vgeff*(1-qtt0)-mg*vt*ln_qtt0;
        vdt     = vdjt+2*vt*log(0.5*(1+sqrt(1+4*exp(-vdjt/vt))));
        *vd_t    = vdt;
        *c_j_t   = c_j*exp(z*log(vd0/(*vd_t)));
        if (is_al == 1) {
            *w_t = w*(*vd_t)/vd0;
        } else {
            *w_t = w;
        }
    } else {
        *c_j_t   = c_j;
        *vd_t    = vd0;
        *w_t     = w;
    }
}

void hicum_TMPHICJ(duals::duald vt, double vt0, duals::duald qtt0, duals::duald ln_qtt0, double mg,
                   double c_j, double vd0, double z, double w, double is_al, double vgeff,
                   double *c_j_t, double *vd_t, double *w_t,
                   double *c_j_t_dT, double *vd_t_dT, double *w_t_dT)
{
    duals::duald c_j_t_result = 0;
    duals::duald vd_t_result = 0;
    duals::duald w_t_result = 0;
    TMPHICJ(vt, vt0, qtt0, ln_qtt0, mg, c_j, vd0, z, w, is_al, vgeff, &c_j_t_result, &vd_t_result, &w_t_result);
    *c_j_t    = c_j_t_result.rpart();
    *c_j_t_dT = c_j_t_result.dpart();
    *vd_t     = vd_t_result.rpart();
    *vd_t_dT  = vd_t_result.dpart();
    *w_t      = w_t_result.rpart();
    *w_t_dT   = w_t_result.dpart();
}


int
HICUMtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-compute many useful parameters
         */
{
    int iret;
    HICUMmodel *model = (HICUMmodel *)inModel;
    HICUMinstance *here;

    /*  loop through all the bipolar models */
    for( ; model != NULL; model = HICUMnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = HICUMinstances(model); here != NULL ;
                here=HICUMnextInstance(here)) {

            if(!here->HICUMtempGiven) here->HICUMtemp = ckt->CKTtemp;

            if(here->HICUMdtempGiven) here->HICUMtemp = here->HICUMtemp + here->HICUMdtemp;

            iret = hicum_thermal_update(model, here, here -> HICUMtemp);

        }
    }
    return(OK);
}

int hicum_thermal_update(HICUMmodel *inModel, HICUMinstance *inInstance, double HICUMTemp)
{
    HICUMmodel *model = (HICUMmodel *)inModel;
    HICUMinstance *here = (HICUMinstance *)inInstance;

    double mg,k10,k20,avs,vgb_t0,vge_t0,vgbe_t0,vgbe0,vgbc0,vgsc0;
    double zetabci,zetabcxt,zetasct;
    duals::duald temp, dT, vt, qtt0, ln_qtt0;
    duals::duald k1,k2,dvg0,vge_t,vgb_t,vgbe_t,cratio_t,a;
    double cratio_t_real, cratio_t_dual;
    double Tnom, zetatef, cjcx01, cjcx02, C_1;
    duals::duald cjei0_t, vdei_t, cjep0_t, vdep_t;

    //variable for area and m scaling
    double area_times_m;
    double qp0_scaled  ;
    double c10_scaled  ;
    double icbar_scaled;
    double cjei0_scaled;
    double ibeis_scaled;
    double ireis_scaled;
    double ibeps_scaled;
    double ibcxs_scaled;
    double ireps_scaled;
    double cjep0_scaled;
    double cbepar_scaled;
    double ibcis_scaled;
    double cjci0_scaled;
    double cjcx0_scaled;
    double cbcpar_scaled;
    double qavl_scaled ;
    double re_scaled   ;
    double rci0_scaled ;
    double rbx_scaled  ;
    double rcx_scaled  ;
    double rbi0_scaled ;
    double rth_scaled ;

    // double cjci0_t, vdci_t, vptci_t, cjep0_t, vdep_t, ajep_t, vdcx_t, vptcx_t, cscp0_t, vdsp_t, vptsp_t, cjs0_t, vds_t, vpts_t;

    // Warning:
    // The scaling with HICUMm and HICUMarea is done here from model to here variables in order to save memory.
    // Classical spice scaling with "area" is implemented, but it is not recommended to be used. If you want 
    // scaling, more sophisticated expressions should be used. Those can be found in modern PDKs or should be 
    // provided by modeling engineers. 
    // For discrete devices, the multiplication facotor "m" should give reasonable results.
    //
    // The HICUMm device multiplicaton factor can be exected to give good results.
    // The following variables need scaling in HICUM:
    // IT         : qp0 ~ (area m)**2   qp0 ~ area m  icbar ~ area m 
    // BE junction: cjei0  ~ area m      cjep0 ~ m
    //              ibeis  ~ area m      ibeps ~ m
    //              cbepar ~ m -> area scaling not reasonable
    // BC junction: cjci0 ~ area m      cjcx0 ~ m
    //              ibcis ~ area m      ibcxs ~ m 
    //              ireis ~ area m      ireps ~ m 
    //              cbcpar ~ m -> area scaling not reasonable
    //              qavl   ~ area m 
    // re   ~1/(area*m) 
    // rci0 ~1/(area*m) 
    // rbx  ~1/(area*m)  -> assume that scaling with "area" is due to lE0 increase
    // rcx  ~1/(area*m)  -> assume that scaling with "area" is due to lE0 increase
    // rbi0 ~1/(area*m)  -> assume that scaling with "area" is due to lE0 increase
    // rth  ~1/(area*m)  -> bad assumption, but more transistor geometry needs to be known for accurate scaling
    // crth ~ area*m     -> bad assumption, but more transistor geometry needs to be known for accurate scaling
    // Substrate related parameters not scaled on purpose. This is very geometry dependent?

    area_times_m  = here->HICUMm*here->HICUMarea;
    //IT
    qp0_scaled    = model->HICUMqp0   * area_times_m;
    c10_scaled    = model->HICUMc10   * area_times_m*area_times_m;
    icbar_scaled  = model->HICUMicbar * area_times_m;
    rth_scaled    = model->HICUMrth   / area_times_m; //very poor assumption
    cth_scaled    = model->HICUMcth   * area_times_m; //very poor assumption
    //BE junction
    cjei0_scaled  = model->HICUMcjei0  * area_times_m;
    ibeis_scaled  = model->HICUMibeis  * area_times_m;
    ireis_scaled  = model->HICUMireis  * area_times_m;
    ibeps_scaled  = model->HICUMibeps  * here->HICUMm;
    ireps_scaled  = model->HICUMireps  * here->HICUMm;
    cjep0_scaled  = model->HICUMcjep0  * here->HICUMm;
    cbepar_scaled = model->HICUMcbepar * here->HICUMm;
    //BC junction
    ibcis_scaled  = model->HICUMibcis  * area_times_m;
    cjci0_scaled  = model->HICUMcjci0  * area_times_m;
    cjcx0_scaled  = model->HICUMcjcx0  * here->HICUMm;
    cbcpar_scaled = model->HICUMcbcpar * here->HICUMm;
    ibcxs_scaled  = model->HICUMibcxs  * here->HICUMm;
    qavl_scaled   = model->HICUMqavl   * area_times_m;
    //resistances //crth todo
    re_scaled     = model->HICUMre   / area_times_m;
    rci0_scaled   = model->HICUMrci0 / area_times_m;
    rbx_scaled    = model->HICUMrbx  / area_times_m;
    rcx_scaled    = model->HICUMrcx  / area_times_m;
    rbi0_scaled   = model->HICUMrbi0 / area_times_m;

    //these variables depend only on scale, but not on temperature. 
    // They are put into the here struct for usage in load routine.
    here->HICUMicbar_scaled  = icbar_scaled;
    here->HICUMcbepar_scaled = cbepar_scaled;
    here->HICUMcbcpar_scaled = cbcpar_scaled;
    here->HICUMcth_scaled    = cth_scaled;

    Tnom    = model->HICUMtnom;
    k10     = model->HICUMf1vg*Tnom*log(Tnom);
    k20     = model->HICUMf2vg*Tnom;
    avs     = model->HICUMalvs*Tnom;
    vgb_t0  = model->HICUMvgb+k10+k20;
    vge_t0  = model->HICUMvge+k10+k20;
    vgbe_t0 = (vgb_t0+vge_t0)/2;

    vgbe0   = (model->HICUMvgb+model->HICUMvge)/2;
    vgbc0   = (model->HICUMvgb+model->HICUMvgc)/2;
    vgsc0   = (model->HICUMvgs+model->HICUMvgc)/2;

    mg      = 3-model->HICUMf1vg/CONSTKoverQ;
    zetabci = mg+1-model->HICUMzetaci;
    zetabcxt= mg+1-model->HICUMzetacx;
    zetasct = mg-1.5;

    // Limit temperature to avoid FPEs in equations
    if(HICUMTemp < TMIN + CONSTCtoK) {
        HICUMTemp = TMIN + CONSTCtoK;
    } else {
        if (HICUMTemp > TMAX + CONSTCtoK) {
            HICUMTemp = TMAX + CONSTCtoK;
        }
    }
    temp = HICUMTemp+1_e;    //dual number valued temperature
    vt   = temp*CONSTKoverQ; // dual valued temperature voltage

    here->HICUMvt0     = Tnom * CONSTKoverQ;
    here->HICUMvt.rpart = vt.rpart();
    here->HICUMvt.dpart = vt.dpart();
    dT      = temp-Tnom;
    qtt0    = temp/Tnom;
    ln_qtt0 = log(qtt0);

    k1      = model->HICUMf1vg*temp*log(temp);
    k2      = model->HICUMf2vg*temp;
    vgb_t   = model->HICUMvgb+k1+k2;
    vge_t   = model->HICUMvge+k1+k2;
    vgbe_t  = (vgb_t+vge_t)/2;

    here->HICUMtVcrit = here->HICUMvt.rpart *
             log(here->HICUMvt.rpart / (CONSTroot2*ibeis_scaled*here->HICUMarea*here->HICUMm));

    //Internal b-e junction capacitance
    // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,cjei0_scaled,model->HICUMvdei,model->HICUMzei,model->HICUMajei,1,vgbe0,&here->HICUMcjei0_t,&here->HICUMvdei_t,&here->HICUMajei_t);
    hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                  cjei0_scaled, model->HICUMvdei, model->HICUMzei, model->HICUMajei, 1, vgbe0,
                  &here->HICUMcjei0_t.rpart, &here->HICUMvdei_t.rpart, &here->HICUMajei_t.rpart,
                  &here->HICUMcjei0_t.dpart, &here->HICUMvdei_t.dpart, &here->HICUMajei_t.dpart);
    cjei0_t.rpart(here->HICUMcjei0_t.rpart);
    cjei0_t.dpart(here->HICUMcjei0_t.dpart);
    vdei_t.rpart(here->HICUMvdei_t.rpart);
    vdei_t.dpart(here->HICUMvdei_t.dpart);

    if (model->HICUMflcomp == 0.0 || model->HICUMflcomp == 2.1) {
        duals::duald V_gT, r_VgVT, k;
        V_gT     = 3.0*vt*ln_qtt0 + model->HICUMvgb*(qtt0-1.0);
        r_VgVT   = V_gT/vt;
        //Internal b-e diode saturation currents
        a = model->HICUMmcf*r_VgVT/model->HICUMmbei - model->HICUMalb*dT;
        a = ibeis_scaled*exp(a);
        here->HICUMibeis_t.rpart = a.rpart();
        here->HICUMibeis_t.dpart = a.dpart();

        a = model->HICUMmcf*r_VgVT/model->HICUMmrei - model->HICUMalb*dT;
        a = ireis_scaled*exp(a);
        here->HICUMireis_t.rpart = a.rpart();
        here->HICUMireis_t.dpart = a.dpart();

        //Peripheral b-e diode saturation currents
        a = model->HICUMmcf*r_VgVT/model->HICUMmbep - model->HICUMalb*dT;
        a = ibeps_scaled*exp(a);
        here->HICUMibeps_t.rpart = a.rpart();
        here->HICUMibeps_t.dpart = a.dpart();

        a = model->HICUMmcf*r_VgVT/model->HICUMmrep - model->HICUMalb*dT;
        a = ireps_scaled*exp(a);
        here->HICUMireps_t.rpart = a.rpart();
        here->HICUMireps_t.dpart = a.dpart();
        //Internal b-c diode saturation current
        a = r_VgVT/model->HICUMmbci;
        a = ibcis_scaled*exp(a);
        here->HICUMibcis_t.rpart = a.rpart();
        here->HICUMibcis_t.dpart = a.dpart();
        //External b-c diode saturation currents
        a = r_VgVT/model->HICUMmbcx;
        a = ibcxs_scaled*exp(a);
        here->HICUMibcxs_t.rpart = a.rpart();
        here->HICUMibcxs_t.dpart = a.dpart();
        //Saturation transfer current for substrate transistor
        a = r_VgVT/model->HICUMmsf;
        a = model->HICUMitss*exp(a);
        here->HICUMitss_t.rpart = a.rpart();
        here->HICUMitss_t.rpart = a.dpart();
        //Saturation current for c-s diode
        a = r_VgVT/model->HICUMmsc;
        a = model->HICUMiscs*exp(a);
        here->HICUMiscs_t.rpart = a.rpart();
        here->HICUMiscs_t.dpart = a.dpart();
        //Zero bias hole charge
        a = vdei_t/model->HICUMvdei;
        a = qp0_scaled*(1.0+0.5*model->HICUMzei*(1.0-a));
        here->HICUMqp0_t.rpart = a.rpart();
        here->HICUMqp0_t.dpart = a.dpart();
        //Voltage separating ohmic and saturation velocity regime
        a = model->HICUMvlim*(1.0-model->HICUMalvs*dT)*exp(model->HICUMzetaci*ln_qtt0);
        k = (a-vt)/vt;
        if (k.rpart() < LN_EXP_LIMIT) {
            a = vt + vt*log(1.0+exp(k));
        }
        here->HICUMvlim_t.rpart = a.rpart();
        here->HICUMvlim_t.dpart = a.dpart();

        //Neutral emitter storage time
        a = 1.0+model->HICUMalb*dT;
        k = 0.5*(a+sqrt(a*a+0.01));
        a = model->HICUMtef0*qtt0/k;
        here->HICUMtef0_t.rpart = a.rpart();
        here->HICUMtef0_t.dpart = a.dpart();
    } else {
        //Internal b-e diode saturation currents
        a = ibeis_scaled*exp(model->HICUMzetabet*ln_qtt0+model->HICUMvge/vt*(qtt0-1));
        here->HICUMibeis_t.rpart = a.rpart();
        here->HICUMibeis_t.dpart = a.dpart();
        if (model->HICUMflcomp>=2.3) {
            a = ireis_scaled*exp(mg/model->HICUMmrei*ln_qtt0+vgbe0/(model->HICUMmrei*vt)*(qtt0-1));
        } else {
            a = ireis_scaled*exp(0.5*mg*ln_qtt0+0.5*vgbe0/vt*(qtt0-1));
        }
        here->HICUMireis_t.rpart = a.rpart();
        here->HICUMireis_t.dpart = a.dpart();
        //Peripheral b-e diode saturation currents
        a = ibeps_scaled*exp(model->HICUMzetabet*ln_qtt0+model->HICUMvge/vt*(qtt0-1));
        here->HICUMibeps_t.rpart = a.rpart();
        here->HICUMibeps_t.dpart = a.dpart();
        if (model->HICUMflcomp>=2.3) {
            a = ireps_scaled*exp(mg/model->HICUMmrep*ln_qtt0+vgbe0/(model->HICUMmrep*vt)*(qtt0-1));
        } else {
            a = ireps_scaled*exp(0.5*mg*qtt0+0.5*vgbe0/vt*(qtt0-1));
        }
        here->HICUMireps_t.rpart = a.rpart();
        here->HICUMireps_t.dpart = a.dpart();
        //Internal b-c diode saturation currents
        a = ibcis_scaled*exp(zetabci*ln_qtt0+model->HICUMvgc/vt*(qtt0-1));
        here->HICUMibcis_t.rpart = a.rpart();
        here->HICUMibcis_t.dpart = a.dpart();
        //External b-c diode saturation currents
        a = ibcxs_scaled*exp(zetabcxt*ln_qtt0+model->HICUMvgc/vt*(qtt0-1));
        here->HICUMibcxs_t.rpart = a.rpart();
        here->HICUMibcxs_t.dpart = a.dpart();
        //Saturation transfer current for substrate transistor
        a = model->HICUMitss*exp(zetasct*ln_qtt0+model->HICUMvgc/vt*(qtt0-1));
        here->HICUMitss_t.rpart = a.rpart();
        here->HICUMitss_t.dpart = a.dpart();
        //Saturation current for c-s diode
        a = model->HICUMiscs*exp(zetasct*ln_qtt0+model->HICUMvgs/vt*(qtt0-1));
        here->HICUMiscs_t.rpart = a.rpart();
        here->HICUMiscs_t.dpart = a.dpart();
        //Zero bias hole charge
        a = exp(model->HICUMzei*log(vdei_t/model->HICUMvdei));
        a = qp0_scaled*(2.0-a);
        here->HICUMqp0_t.rpart = a.rpart();
        here->HICUMqp0_t.dpart = a.dpart();
        //Voltage separating ohmic and saturation velocity regime
        a = model->HICUMvlim*exp((model->HICUMzetaci-avs)*ln_qtt0);
        here->HICUMvlim_t.rpart = a.rpart();
        here->HICUMvlim_t.dpart = a.dpart();
        //Neutral emitter storage time
        if (model->HICUMflcomp >= 2.3) {
            a = model->HICUMtef0;
        } else {
            zetatef = model->HICUMzetabet-model->HICUMzetact-0.5;
            dvg0    = model->HICUMvgb-model->HICUMvge;
            a       = model->HICUMtef0*exp(zetatef*ln_qtt0-dvg0/vt*(qtt0-1));
        }
        here->HICUMtef0_t.rpart = a.rpart();
        here->HICUMtef0_t.dpart = a.dpart();
    }

    //GICCR prefactor
    a = c10_scaled*exp(model->HICUMzetact*ln_qtt0+model->HICUMvgb/vt*(qtt0-1));
    here->HICUMc10_t.rpart = a.rpart();
    here->HICUMc10_t.dpart = a.dpart();

    // Low-field internal collector resistance
    a = rci0_scaled*exp(model->HICUMzetaci*ln_qtt0);
    here->HICUMrci0_t.rpart = a.rpart();
    here->HICUMrci0_t.dpart = a.dpart();

    //Internal c-e saturation voltage
    a = model->HICUMvces*(1+model->HICUMalces*dT);
    here->HICUMvces_t.rpart = a.rpart();
    here->HICUMvces_t.dpart = a.dpart();

    //Internal b-c junction capacitance
    // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,cjci0_scaled,model->HICUMvdci,model->HICUMzci,model->HICUMvptci,0,vgbc0,&cjci0_t,&vdci_t,&vptci_t);
    hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                  cjci0_scaled, model->HICUMvdci, model->HICUMzci, model->HICUMvptci, 0, vgbc0,
                  &here->HICUMcjci0_t.rpart, &here->HICUMvdci_t.rpart, &here->HICUMvptci_t.rpart,
                  &here->HICUMcjci0_t.dpart, &here->HICUMvdci_t.dpart, &here->HICUMvptci_t.dpart);

    //Low-current forward transit time
    a = model->HICUMt0*(1+model->HICUMalt0*dT+model->HICUMkt0*dT*dT);
    here->HICUMt0_t.rpart = a.rpart();
    here->HICUMt0_t.dpart = a.dpart();

    //Saturation time constant at high current densities
    a = model->HICUMthcs*exp((model->HICUMzetaci-1)*ln_qtt0);
    here->HICUMthcs_t.rpart = a.rpart();
    here->HICUMthcs_t.dpart = a.dpart();

    //Avalanche current factors
    a = model->HICUMfavl*exp(model->HICUMalfav*dT);
    here->HICUMfavl_t.rpart = a.rpart();
    here->HICUMfavl_t.dpart = a.dpart();
    a = qavl_scaled*exp(model->HICUMalqav*dT);
    here->HICUMqavl_t.rpart = a.rpart();
    here->HICUMqavl_t.dpart = a.dpart();
    a = model->HICUMkavl*exp(model->HICUMalkav*dT);
    here->HICUMkavl_t.rpart = a.rpart();
    here->HICUMkavl_t.dpart = a.dpart();

    //Zero bias internal base resistance
    a = rbi0_scaled*exp(model->HICUMzetarbi*ln_qtt0);
    here->HICUMrbi0_t.rpart = a.rpart();
    here->HICUMrbi0_t.dpart = a.dpart();

    //Peripheral b-e junction capacitance
    // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,cjep0_scaled,model->HICUMvdep,model->HICUMzep,model->HICUMajep,1,vgbe0,&cjep0_t,&vdep_t,&ajep_t);
    hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                  cjep0_scaled, model->HICUMvdep, model->HICUMzep, model->HICUMajep, 1, vgbe0,
                  &here->HICUMcjep0_t.rpart, &here->HICUMvdep_t.rpart, &here->HICUMajep_t.rpart,
                  &here->HICUMcjep0_t.dpart, &here->HICUMvdep_t.dpart, &here->HICUMajep_t.dpart);
    cjep0_t.rpart(here->HICUMcjep0_t.rpart);
    cjep0_t.dpart(here->HICUMcjep0_t.dpart);
    vdep_t.rpart(here->HICUMvdep_t.rpart);
    vdep_t.dpart(here->HICUMvdep_t.dpart);

    //Tunneling current factors
    if (model->HICUMibets > 0) { // HICTUN_T
        duals::duald a_eg,ab,aa;
        ab = 1.0;
        aa = 1.0;
        a_eg = vgbe_t0/vgbe_t;
        if(model->HICUMtunode==1 && cjep0_scaled > 0.0 && model->HICUMvdep >0.0) {
            ab = (cjep0_t/cjep0_scaled)*sqrt(a_eg)*vdep_t*vdep_t/(model->HICUMvdep*model->HICUMvdep);
            aa = (model->HICUMvdep/vdep_t)*(cjep0_scaled/cjep0_t)*pow(a_eg,-1.5);
        } else if (model->HICUMtunode==0 && cjei0_scaled > 0.0 && model->HICUMvdei >0.0) {
            ab = (cjei0_t/cjei0_scaled)*sqrt(a_eg)*vdei_t*vdei_t/(model->HICUMvdei*model->HICUMvdei);
            aa = (model->HICUMvdei/vdei_t)*(cjei0_scaled/cjei0_t)*pow(a_eg,-1.5);
        }
        a = model->HICUMibets*ab;
        here->HICUMibets_t.rpart = a.rpart();
        here->HICUMibets_t.dpart = a.dpart();
        a = model->HICUMabet*aa;
        here->HICUMabet_t.rpart = a.rpart();
        here->HICUMabet_t.dpart = a.dpart();
     } else {
        here->HICUMibets_t.rpart = 0;
        here->HICUMibets_t.dpart = 0;
        here->HICUMabet_t.rpart = 1;
        here->HICUMabet_t.dpart = 0;
    }

    //Depletion capacitance splitting at b-c junction
    //Capacitances at peripheral and external base node
    C_1    = (1.0-model->HICUMfbcpar)*(cjcx0_scaled+cbcpar_scaled);
    if (C_1 >= cbcpar_scaled) {
        cjcx01  = C_1-cbcpar_scaled;
        cjcx02  = cjcx0_scaled-cjcx01;
    } else {
        cjcx01  = 0.0;
        cjcx02  = cjcx0_scaled;
    }

    //Temperature mapping for tunneling current is done inside HICTUN
    // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,1.0,model->HICUMvdcx,model->HICUMzcx,model->HICUMvptcx,0,vgbc0,&cratio_t,&vdcx_t,&vptcx_t);
    hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                  1.0, model->HICUMvdcx, model->HICUMzcx, model->HICUMvptcx, 0, vgbc0,
                  &cratio_t_real, &here->HICUMvdcx_t.rpart, &here->HICUMvptcx_t.rpart,
                  &cratio_t_dual, &here->HICUMvdcx_t.dpart, &here->HICUMvptcx_t.dpart);
    cratio_t.rpart(cratio_t_real);
    cratio_t.dpart(cratio_t_dual);
    a = cratio_t*cjcx01;
    here->HICUMcjcx01_t.rpart = a.rpart();
    here->HICUMcjcx01_t.dpart = a.dpart();
    a = cratio_t*cjcx02;
    here->HICUMcjcx02_t.rpart = a.rpart();
    here->HICUMcjcx02_t.dpart = a.dpart();

    //Constant external series resistances
    a = rcx_scaled*exp(model->HICUMzetarcx*ln_qtt0);
    here->HICUMrcx_t.rpart = a.rpart();
    here->HICUMrcx_t.dpart = a.dpart();
    a = rbx_scaled*exp(model->HICUMzetarbx*ln_qtt0);
    here->HICUMrbx_t.rpart = a.rpart();
    here->HICUMrbx_t.dpart = a.dpart();
    a = re_scaled*exp(model->HICUMzetare*ln_qtt0);
    here->HICUMre_t.rpart = a.rpart();
    here->HICUMre_t.dpart = a.dpart();

    //Forward transit time in substrate transistor
    a = model->HICUMtsf*exp((model->HICUMzetacx-1.0)*ln_qtt0);
    here->HICUMtsf_t.rpart = a.rpart();
    here->HICUMtsf_t.dpart = a.dpart();

    //Capacitance for c-s junction
    // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcjs0,model->HICUMvds,model->HICUMzs,model->HICUMvpts,0,vgsc0,&cjs0_t,&vds_t,&vpts_t);
    hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                  model->HICUMcjs0, model->HICUMvds, model->HICUMzs, model->HICUMvpts, 0, vgsc0,
                  &here->HICUMcjs0_t.rpart, &here->HICUMvds_t.rpart, &here->HICUMvpts_t.rpart,
                  &here->HICUMcjs0_t.dpart, &here->HICUMvds_t.dpart, &here->HICUMvpts_t.dpart);
    /*Peripheral s-c capacitance
     * Note, thermal update only required for model->HICUMvds > 0
     * Save computional effort otherwise
     */
    if (model->HICUMvdsp > 0) {
        // TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcscp0,model->HICUMvdsp,model->HICUMzsp,model->HICUMvptsp,0,vgsc0,&cscp0_t,&vdsp_t,&vptsp_t);
        hicum_TMPHICJ(vt, here->HICUMvt0, qtt0, ln_qtt0, mg,
                     model->HICUMcscp0, model->HICUMvdsp, model->HICUMzsp, model->HICUMvptsp, 0, vgsc0,
                     &here->HICUMcscp0_t.rpart, &here->HICUMvdsp_t.rpart, &here->HICUMvptsp_t.rpart,
                     &here->HICUMcscp0_t.dpart, &here->HICUMvdsp_t.dpart, &here->HICUMvptsp_t.dpart);
    } else {
        // Avoid uninitialized variables
        here->HICUMcscp0_t.rpart = model->HICUMcscp0;
        here->HICUMcscp0_t.dpart = 0;
        here->HICUMvdsp_t.rpart = model->HICUMvdsp;
        here->HICUMvdsp_t.dpart = 0;
        here->HICUMvptsp_t.rpart = model->HICUMvptsp;
        here->HICUMvptsp_t.dpart = 0;
    }

    a = model->HICUMahjei*exp(model->HICUMzetahjei*ln_qtt0);
    here->HICUMahjei_t.rpart = a.rpart();
    here->HICUMahjei_t.dpart = a.dpart();
    a = model->HICUMhjei*exp(model->HICUMdvgbe/vt*(exp(model->HICUMzetavgbe*log(qtt0))-1));
    here->HICUMhjei0_t.rpart = a.rpart();
    here->HICUMhjei0_t.dpart = a.dpart();
    a = model->HICUMhf0*exp(model->HICUMdvgbe/vt*(qtt0-1));
    here->HICUMhf0_t.rpart = a.rpart();
    here->HICUMhf0_t.dpart = a.dpart();
    if (model->HICUMflcomp >= 2.3) {
        a = model->HICUMhfe*exp((model->HICUMvgb-model->HICUMvge)/vt*(qtt0-1));
        here->HICUMhfe_t.rpart = a.rpart();
        here->HICUMhfe_t.dpart = a.dpart();
        a = model->HICUMhfc*exp((model->HICUMvgb-model->HICUMvgc)/vt*(qtt0-1));
        here->HICUMhfc_t.rpart = a.rpart();
        here->HICUMhfc_t.dpart = a.dpart();
    } else {
        here->HICUMhfe_t.rpart = model->HICUMhfe;
        here->HICUMhfe_t.dpart = 0;
        here->HICUMhfc_t.rpart = model->HICUMhfc;
        here->HICUMhfc_t.dpart = 0;
    }

    a = rth_scaled*exp(model->HICUMzetarth*ln_qtt0)*(1+model->HICUMalrth*dT);
    here->HICUMrth_t.rpart = a.rpart();
    here->HICUMrth_t.dpart = a.dpart();

    return(0);
}
