/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "hicumdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"

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
void TMPHICJ(double vt, double vt0, double qtt0, double ln_qtt0, double mg, 
             double c_j, double vd0, double z, double w, double is_al, double vgeff, 
             double *c_j_t, double *vd_t, double *w_t)
{
double vdj0,vdjt,vdt;

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

int iret, hicum_thermal_update(HICUMmodel *, HICUMinstance *);

int
HICUMtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-compute many useful parameters
         */
{
    HICUMmodel *model = (HICUMmodel *)inModel;
    HICUMinstance *here;

    /*  loop through all the bipolar models */
    for( ; model != NULL; model = HICUMnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = HICUMinstances(model); here != NULL ;
                here=HICUMnextInstance(here)) {

            if(!here->HICUMtempGiven) here->HICUMtemp = ckt->CKTtemp;

            if(here->HICUMdtempGiven) here->HICUMtemp = here->HICUMtemp + here->HICUMdtemp;

            iret = hicum_thermal_update(model, here);

        }
    }
    return(OK);
}

int hicum_thermal_update(HICUMmodel *inModel, HICUMinstance *inInstance)
{
    HICUMmodel *model = (HICUMmodel *)inModel;
    HICUMinstance *here = (HICUMinstance *)inInstance;

    double k10,k20,avs,vgb_t0,vge_t0,vgbe_t0,vgbe0,vgbc0,vgsc0;
    double zetabci,zetabcxt,zetasct;
    double k1,k2,dvg0,vge_t,vgb_t,vgbe_t,cratio_t,a;
    double Tnom, dT, zetatef, cjcx01, cjcx02, C_1;
    double cjci0_t, vdci_t, vptci_t, cjep0_t, vdep_t, ajep_t, vdcx_t, vptcx_t, cscp0_t, vdsp_t, vptsp_t, cjs0_t, vds_t, vpts_t;

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

    here->HICUMmg      = 3-model->HICUMf1vg/CONSTKoverQ;
    zetabci = here->HICUMmg+1-model->HICUMzetaci;
    zetabcxt= here->HICUMmg+1-model->HICUMzetacx;
    zetasct = here->HICUMmg-1.5;

    // Limit temperature to avoid FPEs in equations
    if(here->HICUMtemp < TMIN + CONSTCtoK) {
        here->HICUMtemp = TMIN + CONSTCtoK;
    } else {
        if (here->HICUMtemp > TMAX + CONSTCtoK) {
            here->HICUMtemp = TMAX + CONSTCtoK;
        }
    }
    here->HICUMvt0     = Tnom * CONSTKoverQ;
    here->HICUMvt      = here->HICUMtemp * CONSTKoverQ;
    dT      = here->HICUMtemp-Tnom;
    here->HICUMqtt0    = here->HICUMtemp/Tnom;
    here->HICUMln_qtt0 = log(here->HICUMqtt0);
    k1      = model->HICUMf1vg*here->HICUMtemp*log(here->HICUMtemp);
    k2      = model->HICUMf2vg*here->HICUMtemp;
    vgb_t   = model->HICUMvgb+k1+k2;
    vge_t   = model->HICUMvge+k1+k2;
    vgbe_t  = (vgb_t+vge_t)/2;

    here->HICUMtVcrit = here->HICUMvt *
             log(here->HICUMvt / (CONSTroot2*model->HICUMibeis*here->HICUMarea*here->HICUMm));

    //Internal b-e junction capacitance
    TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcjei0,model->HICUMvdei,model->HICUMzei,model->HICUMajei,1,vgbe0,&here->HICUMcjei0_t,&here->HICUMvdei_t,&here->HICUMajei_t);

    if (model->HICUMflcomp == 0.0 || model->HICUMflcomp == 2.1) {
        double V_gT, r_VgVT, k;
        V_gT     = 3.0*here->HICUMvt*here->HICUMln_qtt0 + model->HICUMvgb*(here->HICUMqtt0-1.0);
        r_VgVT   = V_gT/here->HICUMvt;
        //Internal b-e diode saturation currents
        a        = model->HICUMmcf*r_VgVT/model->HICUMmbei - model->HICUMalb*dT;
        here->HICUMibeis_t  = model->HICUMibeis*exp(a);
        a        = model->HICUMmcf*r_VgVT/model->HICUMmrei - model->HICUMalb*dT;
        here->HICUMireis_t  = model->HICUMireis*exp(a);
        a        = model->HICUMmcf*r_VgVT/model->HICUMmbep - model->HICUMalb*dT;
        //Peripheral b-e diode saturation currents
        here->HICUMibeps_t  = model->HICUMibeps*exp(a);
        a        = model->HICUMmcf*r_VgVT/model->HICUMmrep - model->HICUMalb*dT;
        here->HICUMireps_t  = model->HICUMireps*exp(a);
        //Internal b-c diode saturation current
        a       = r_VgVT/model->HICUMmbci;
        here->HICUMibcis_t = model->HICUMibcis*exp(a);
        //External b-c diode saturation currents
        a       = r_VgVT/model->HICUMmbcx;
        here->HICUMibcxs_t = model->HICUMibcxs*exp(a);
        //Saturation transfer current for substrate transistor
        a       = r_VgVT/model->HICUMmsf;
        here->HICUMitss_t  = model->HICUMitss*exp(a);
        //Saturation current for c-s diode
        a       = r_VgVT/model->HICUMmsc;
        here->HICUMiscs_t  = model->HICUMiscs*exp(a);
        //Zero bias hole charge
        a        = here->HICUMvdei_t/model->HICUMvdei;
        here->HICUMqp0_t    = model->HICUMqp0*(1.0+0.5*model->HICUMzei*(1.0-a));
        //Voltage separating ohmic and saturation velocity regime
        a = model->HICUMvlim*(1.0-model->HICUMalvs*dT)*exp(model->HICUMzetaci*here->HICUMln_qtt0);
        k = (a-here->HICUMvt)/here->HICUMvt;
        if (k < LN_EXP_LIMIT) {
            here->HICUMvlim_t = here->HICUMvt + here->HICUMvt*log(1.0+exp(k));
        } else {
            here->HICUMvlim_t = a;
        }
        //Neutral emitter storage time
        a        = 1.0+model->HICUMalb*dT;
        k        = 0.5*(a+sqrt(a*a+0.01));
        here->HICUMtef0_t   = model->HICUMtef0*here->HICUMqtt0/k;
    } else {
        //Internal b-e diode saturation currents
        here->HICUMibeis_t  = model->HICUMibeis*exp(model->HICUMzetabet*here->HICUMln_qtt0+model->HICUMvge/here->HICUMvt*(here->HICUMqtt0-1));
        if (model->HICUMflcomp>=2.3) {
            here->HICUMireis_t  = model->HICUMireis*exp(here->HICUMmg/model->HICUMmrei*here->HICUMln_qtt0+vgbe0/(model->HICUMmrei*here->HICUMvt)*(here->HICUMqtt0-1));
        } else {
            here->HICUMireis_t  = model->HICUMireis*exp(0.5*here->HICUMmg*here->HICUMln_qtt0+0.5*vgbe0/here->HICUMvt*(here->HICUMqtt0-1));
        }
        //Peripheral b-e diode saturation currents
        here->HICUMibeps_t  = model->HICUMibeps*exp(model->HICUMzetabet*here->HICUMln_qtt0+model->HICUMvge/here->HICUMvt*(here->HICUMqtt0-1));
        if (model->HICUMflcomp>=2.3) {
            here->HICUMireps_t  = model->HICUMireps*exp(here->HICUMmg/model->HICUMmrep*here->HICUMln_qtt0+vgbe0/(model->HICUMmrep*here->HICUMvt)*(here->HICUMqtt0-1));
        } else {
            here->HICUMireps_t  = model->HICUMireps*exp(0.5*here->HICUMmg*here->HICUMln_qtt0+0.5*vgbe0/here->HICUMvt*(here->HICUMqtt0-1));
        }
        //Internal b-c diode saturation currents
        here->HICUMibcis_t = model->HICUMibcis*exp(zetabci*here->HICUMln_qtt0+model->HICUMvgc/here->HICUMvt*(here->HICUMqtt0-1));
        //External b-c diode saturation currents
        here->HICUMibcxs_t = model->HICUMibcxs*exp(zetabcxt*here->HICUMln_qtt0+model->HICUMvgc/here->HICUMvt*(here->HICUMqtt0-1));
        //Saturation transfer current for substrate transistor
        here->HICUMitss_t  = model->HICUMitss*exp(zetasct*here->HICUMln_qtt0+model->HICUMvgc/here->HICUMvt*(here->HICUMqtt0-1));
        //Saturation current for c-s diode
        here->HICUMiscs_t  = model->HICUMiscs*exp(zetasct*here->HICUMln_qtt0+model->HICUMvgs/here->HICUMvt*(here->HICUMqtt0-1));
        //Zero bias hole charge
        a       = exp(model->HICUMzei*log(here->HICUMvdei_t/model->HICUMvdei));
        here->HICUMqp0_t   = model->HICUMqp0*(2.0-a);
        //Voltage separating ohmic and saturation velocity regime
        here->HICUMvlim_t  = model->HICUMvlim*exp((model->HICUMzetaci-avs)*here->HICUMln_qtt0);
        //Neutral emitter storage time
        if (model->HICUMflcomp >= 2.3) {
            here->HICUMtef0_t  = model->HICUMtef0;
        } else {
            zetatef = model->HICUMzetabet-model->HICUMzetact-0.5;
            dvg0    = model->HICUMvgb-model->HICUMvge;
            here->HICUMtef0_t  = model->HICUMtef0*exp(zetatef*here->HICUMln_qtt0-dvg0/here->HICUMvt*(here->HICUMqtt0-1));
        }
    }

    //GICCR prefactor
    here->HICUMc10_t   = model->HICUMc10*exp(model->HICUMzetact*here->HICUMln_qtt0+model->HICUMvgb/here->HICUMvt*(here->HICUMqtt0-1));

    // Low-field internal collector resistance
    here->HICUMrci0_t  = model->HICUMrci0*exp(model->HICUMzetaci*here->HICUMln_qtt0);

    //Voltage separating ohmic and saturation velocity regime
    //vlim_t  = model->HICUMvlim*exp((model->HICUMzetaci-avs)*here->HICUMln_qtt0);

    //Internal c-e saturation voltage
    here->HICUMvces_t  = model->HICUMvces*(1+model->HICUMalces*dT);


    //Internal b-c diode saturation current
    //ibcis_t = model->HICUMibcis*exp(zetabci*here->HICUMln_qtt0+model->HICUMvgc/here->HICUMvt*(here->HICUMqtt0-1));

    //Internal b-c junction capacitance
    TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcjci0,model->HICUMvdci,model->HICUMzci,model->HICUMvptci,0,vgbc0,&cjci0_t,&vdci_t,&vptci_t);
    here->HICUMcjci0_t = cjci0_t;
    here->HICUMvdci_t = vdci_t;
    here->HICUMvptci_t = vptci_t;

    //Low-current forward transit time
    here->HICUMt0_t    = model->HICUMt0*(1+model->HICUMalt0*dT+model->HICUMkt0*dT*dT);

    //Saturation time constant at high current densities
    here->HICUMthcs_t  = model->HICUMthcs*exp((model->HICUMzetaci-1)*here->HICUMln_qtt0);

    //Avalanche current factors
    here->HICUMfavl_t  = model->HICUMfavl*exp(model->HICUMalfav*dT);
    here->HICUMqavl_t  = model->HICUMqavl*exp(model->HICUMalqav*dT);
    here->HICUMkavl_t  = model->HICUMkavl*exp(model->HICUMalkav*dT);

    //Zero bias internal base resistance
    here->HICUMrbi0_t  = model->HICUMrbi0*exp(model->HICUMzetarbi*here->HICUMln_qtt0);

    //Peripheral b-e junction capacitance
    TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcjep0,model->HICUMvdep,model->HICUMzep,model->HICUMajep,1,vgbe0,&cjep0_t,&vdep_t,&ajep_t);
    here->HICUMcjep0_t = cjep0_t;
    here->HICUMvdep_t = vdep_t;
    here->HICUMajep_t = ajep_t;

    //Tunneling current factors
    if (model->HICUMibets > 0) { // HICTUN_T
        double a_eg,ab,aa;
        ab      = 1.0;
        aa      = 1.0;
        a_eg=vgbe_t0/vgbe_t;
        if(model->HICUMtunode==1 && model->HICUMcjep0 > 0.0 && model->HICUMvdep >0.0) {
            ab      = (here->HICUMcjep0_t/model->HICUMcjep0)*sqrt(a_eg)*vdep_t*vdep_t/(model->HICUMvdep*model->HICUMvdep);
            aa      = (model->HICUMvdep/vdep_t)*(model->HICUMcjep0/here->HICUMcjep0_t)*pow(a_eg,-1.5);
        } else if (model->HICUMtunode==0 && model->HICUMcjei0 > 0.0 && model->HICUMvdei >0.0) {
            ab      = (here->HICUMcjei0_t/model->HICUMcjei0)*sqrt(a_eg)*here->HICUMvdei_t*here->HICUMvdei_t/(model->HICUMvdei*model->HICUMvdei);
            aa      = (model->HICUMvdei/here->HICUMvdei_t)*(model->HICUMcjei0/here->HICUMcjei0_t)*pow(a_eg,-1.5);
        }
        here->HICUMibets_t = model->HICUMibets*ab;
        here->HICUMabet_t  = model->HICUMabet*aa;
    } else {
        here->HICUMibets_t = 0;
        here->HICUMabet_t = 1;
    }

    //Depletion capacitance splitting at b-c junction
    //Capacitances at peripheral and external base node
    C_1    = (1.0-model->HICUMfbcpar)*(model->HICUMcjcx0+model->HICUMcbcpar);
    if (C_1 >= model->HICUMcbcpar) {
        cjcx01  = C_1-model->HICUMcbcpar;
        cjcx02  = model->HICUMcjcx0-cjcx01;
    } else {
        cjcx01  = 0.0;
        cjcx02  = model->HICUMcjcx0;
    }

    //Temperature mapping for tunneling current is done inside HICTUN

    TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,1.0,model->HICUMvdcx,model->HICUMzcx,model->HICUMvptcx,0,vgbc0,&cratio_t,&vdcx_t,&vptcx_t);
    here->HICUMcjcx01_t=cratio_t*cjcx01;
    here->HICUMcjcx02_t=cratio_t*cjcx02;
    here->HICUMvdcx_t = vdcx_t;
    here->HICUMvptcx_t = vptcx_t;

    //External b-c diode saturation currents
    //ibcxs_t       = model->HICUMibcxs*exp(zetabcxt*here->HICUMln_qtt0+model->HICUMvgc/here->HICUMvt*(qtt0-1));

    //Constant external series resistances
    here->HICUMrcx_t   = model->HICUMrcx*exp(model->HICUMzetarcx*here->HICUMln_qtt0);
    here->HICUMrbx_t   = model->HICUMrbx*exp(model->HICUMzetarbx*here->HICUMln_qtt0);
    here->HICUMre_t    = model->HICUMre*exp(model->HICUMzetare*here->HICUMln_qtt0);

    //Forward transit time in substrate transistor
    here->HICUMtsf_t   = model->HICUMtsf*exp((model->HICUMzetacx-1.0)*here->HICUMln_qtt0);

    //Capacitance for c-s junction
    TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcjs0,model->HICUMvds,model->HICUMzs,model->HICUMvpts,0,vgsc0,&cjs0_t,&vds_t,&vpts_t);
    here->HICUMcjs0_t = cjs0_t;
    here->HICUMvds_t = vds_t;
    here->HICUMvpts_t = vpts_t;
    /*Peripheral s-c capacitance
     * Note, thermal update only required for model->HICUMvds > 0
     * Save computional effort otherwise
     */
    if (model->HICUMvdsp > 0) {
        TMPHICJ(here->HICUMvt0,here->HICUMvt,here->HICUMqtt0,here->HICUMln_qtt0,here->HICUMmg,model->HICUMcscp0,model->HICUMvdsp,model->HICUMzsp,model->HICUMvptsp,0,vgsc0,&cscp0_t,&vdsp_t,&vptsp_t);
        here->HICUMcscp0_t = cscp0_t;
        here->HICUMvdsp_t = vdsp_t;
        here->HICUMvptsp_t = vptsp_t;
    } else {
        // Avoid uninitialized variables
        here->HICUMcscp0_t = model->HICUMcscp0;
        here->HICUMvdsp_t = model->HICUMvdsp;
        here->HICUMvptsp_t = model->HICUMvptsp;
    }

    here->HICUMahjei_t = model->HICUMahjei*exp(model->HICUMzetahjei*here->HICUMln_qtt0);
    here->HICUMhjei0_t = model->HICUMhjei*exp(model->HICUMdvgbe/here->HICUMvt*(exp(model->HICUMzetavgbe*log(here->HICUMqtt0))-1));
    here->HICUMhf0_t   = model->HICUMhf0*exp(model->HICUMdvgbe/here->HICUMvt*(here->HICUMqtt0-1));
    if (model->HICUMflcomp >= 2.3) {
        here->HICUMhfe_t   = model->HICUMhfe*exp((model->HICUMvgb-model->HICUMvge)/here->HICUMvt*(here->HICUMqtt0-1));
        here->HICUMhfc_t   = model->HICUMhfc*exp((model->HICUMvgb-model->HICUMvgc)/here->HICUMvt*(here->HICUMqtt0-1));
    } else {
        here->HICUMhfe_t    = model->HICUMhfe;
        here->HICUMhfc_t    = model->HICUMhfc;
    }

    here->HICUMrth_t    = model->HICUMrth*exp(model->HICUMzetarth*here->HICUMln_qtt0)*(1+model->HICUMalrth*dT);

    return(0);
}

