/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This is the function called each iteration to evaluate the
 * VBICs in the circuit and load them into the matrix as appropriate
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"

int vbic_4T_et_cf_fj(
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *);
int vbic_4T_it_cf_fj(
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *);

int
VBICload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the
         * sparse matrix previously provided
         */
{
    VBICmodel *model = (VBICmodel*)inModel;
    VBICinstance *here;
     double p[108]
    ,Vrth=0.0,Vbei,Vbex,Vbci,Vbep,Vbcp
    ,Vrcx,Vbcx,Vrci,Vrbx,Vrbi,Vre,Vrbp
    ,Vrs,Vbe,Vbc,Vcei=0.0,Vcep=0.0,Ibe,Ibe_Vrth=0.0
    ,Ibe_Vbei,Ibex,Ibex_Vrth=0.0,Ibex_Vbex,Itzf,Itzf_Vrth=0.0,Itzf_Vbei
    ,Itzf_Vbci,Itzr,Itzr_Vrth=0.0,Itzr_Vbci,Itzr_Vbei,Ibc,Ibc_Vrth=0.0
    ,Ibc_Vbci,Ibc_Vbei,Ibep,Ibep_Vrth=0.0,Ibep_Vbep,Ircx,Ircx_Vrcx
    ,Ircx_Vrth=0.0,Irci,Irci_Vrci,Irci_Vrth=0.0,Irci_Vbci,Irci_Vbcx,Irbx
    ,Irbx_Vrbx,Irbx_Vrth=0.0,Irbi,Irbi_Vrbi,Irbi_Vrth=0.0,Irbi_Vbei,Irbi_Vbci
    ,Ire,Ire_Vre,Ire_Vrth=0.0,Irbp,Irbp_Vrbp,Irbp_Vrth=0.0,Irbp_Vbep
    ,Irbp_Vbci,Qbe,Qbe_Vrth,Qbe_Vbei,Qbe_Vbci,Qbex,Qbex_Vrth
    ,Qbex_Vbex,Qbc,Qbc_Vrth,Qbc_Vbci,Qbcx,Qbcx_Vrth,Qbcx_Vbcx
    ,Qbep,Qbep_Vrth,Qbep_Vbep,Qbep_Vbci,Qbeo,Qbeo_Vbe,Qbco
    ,Qbco_Vbc,Ibcp,Ibcp_Vrth=0.0,Ibcp_Vbcp,Iccp,Iccp_Vrth=0.0,Iccp_Vbep
    ,Iccp_Vbci,Iccp_Vbcp,Irs,Irs_Vrs,Irs_Vrth=0.0,Qbcp,Qbcp_Vrth
    ,Qbcp_Vbcp,Irth,Irth_Vrth=0.0,Ith=0.0,Ith_Vrth=0.0,Ith_Vbei=0.0,Ith_Vbci=0.0
    ,Ith_Vcei=0.0,Ith_Vbex=0.0,Ith_Vbep=0.0,Ith_Vrs=0.0,Ith_Vbcp=0.0,Ith_Vcep=0.0,Ith_Vrcx=0.0
    ,Ith_Vrci=0.0,Ith_Vbcx=0.0,Ith_Vrbx=0.0,Ith_Vrbi=0.0,Ith_Vre=0.0,Ith_Vrbp=0.0,Qcth=0.0
    ,Qcth_Vrth=0.0,SCALE;
    int iret;
    double vce;
#ifndef PREDICTOR
    double xfact;
#endif
    double vt;
    double delvbei;
    double delvbex;
    double delvbci;
    double delvbcx;
    double delvbep;
    double delvrci;
    double delvrbi;
    double delvrbp;
    double delvbcp;
    double ibehat;
    double ibexhat;
    double itzfhat;
    double itzrhat;
    double ibchat;
    double ibephat;
    double ircihat;
    double irbihat;
    double irbphat;
    double ibcphat;
    double iccphat;
    double ceq, geq, rhs_current;
    int icheck=1;
    int ichk1, ichk2, ichk3, ichk4, ichk5, ichk6;
    int error;
    double gqbeo, cqbeo, gqbco, cqbco, gbcx, cbcx;
    double Icth, Icth_Vrth;

    /*  loop through all the models */
    for( ; model != NULL; model = VBICnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VBICinstances(model); here != NULL ;
                here=VBICnextInstance(here)) {

            vt = here->VBICtemp * CONSTKoverQ;

            gbcx = 0.0;
            cbcx = 0.0;
            gqbeo = 0.0;
            cqbeo = 0.0;
            gqbco = 0.0;
            cqbco = 0.0;
            Icth = 0.0, Icth_Vrth = 0.0;
            /*
             *   model parameters
             */
            memcpy (&p, &model->VBICtnom, sizeof(p));

            p[0] = here->VBICtemp - CONSTCtoK + p[105];
            /* temperature dependent parameter are already calculated */
            p[1] = here->VBICtextCollResist;
            p[2] = here->VBICtintCollResist;
            p[3] = here->VBICtepiSatVoltage;
            p[4] = here->VBICtepiDoping;
            p[6] = here->VBICtextBaseResist;
            p[7] = here->VBICtintBaseResist;
            p[8] = here->VBICtemitterResist;
            p[9] = here->VBICtsubstrateResist;
            p[10] = here->VBICtparBaseResist;
            p[11] = here->VBICtsatCur;
            p[12] = here->VBICtemissionCoeffF;
            p[13] = here->VBICtemissionCoeffR;
            p[16] = here->VBICtdepletionCapBE;
            p[17] = here->VBICtpotentialBE;
            p[21] = here->VBICtdepletionCapBC;
            p[23] = here->VBICtextCapBC;
            p[24] = here->VBICtpotentialBC;
            p[27] = here->VBICtextCapSC;
            p[28] = here->VBICtpotentialSC;
            p[31] = here->VBICtidealSatCurBE;
            p[34] = here->VBICtnidealSatCurBE;
            p[36] = here->VBICtidealSatCurBC;
            p[38] = here->VBICtnidealSatCurBC;
            p[41] = here->VBICtavalanchePar2BC;
            p[42] = here->VBICtparasitSatCur;
            p[45] = here->VBICtidealParasitSatCurBE;
            p[46] = here->VBICtnidealParasitSatCurBE;
            p[47] = here->VBICtidealParasitSatCurBC;
            p[49] = here->VBICtnidealParasitSatCurBC;
            p[53] = here->VBICtrollOffF;
            p[94] = here->VBICtsepISRR;
            p[98] = here->VBICtvbbe;
            p[99] = here->VBICtnbbe;

            SCALE = here->VBICarea * here->VBICm;

            /*
             *   initialization
             */
            icheck=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                Vbe = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICemitNode));
                Vbc = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICcollNode));
                Vbei = *(ckt->CKTstate0 + here->VBICvbei);
                Vbex = *(ckt->CKTstate0 + here->VBICvbex);
                Vbci = *(ckt->CKTstate0 + here->VBICvbci);
                Vbcx = *(ckt->CKTstate0 + here->VBICvbcx);
                Vbep = *(ckt->CKTstate0 + here->VBICvbep);
                Vrci = *(ckt->CKTstate0 + here->VBICvrci);
                Vrbi = *(ckt->CKTstate0 + here->VBICvrbi);
                Vrbp = *(ckt->CKTstate0 + here->VBICvrbp);
                Vrcx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICcollNode)-
                    *(ckt->CKTrhsOld+here->VBICcollCXNode));
                Vrbx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICbaseBXNode));
                Vre = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICemitNode)-
                    *(ckt->CKTrhsOld+here->VBICemitEINode));
                Vbcp = *(ckt->CKTstate0 + here->VBICvbcp);
                Vrs = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICsubsNode)-
                    *(ckt->CKTrhsOld+here->VBICsubsSINode));
                if (here->VBIC_selfheat)
                    Vrth = *(ckt->CKTstate0 + here->VBICvrth);
            } else if(ckt->CKTmode & MODEINITTRAN) {
                Vbe = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICemitNode));
                Vbc = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICcollNode));
                Vbei = *(ckt->CKTstate1 + here->VBICvbei);
                Vbex = *(ckt->CKTstate1 + here->VBICvbex);
                Vbci = *(ckt->CKTstate1 + here->VBICvbci);
                Vbcx = *(ckt->CKTstate1 + here->VBICvbcx);
                Vbep = *(ckt->CKTstate1 + here->VBICvbep);
                Vrci = *(ckt->CKTstate1 + here->VBICvrci);
                Vrbi = *(ckt->CKTstate1 + here->VBICvrbi);
                Vrbp = *(ckt->CKTstate1 + here->VBICvrbp);
                Vrcx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICcollNode)-
                    *(ckt->CKTrhsOld+here->VBICcollCXNode));
                Vrbx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICbaseBXNode));
                Vre = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICemitNode)-
                    *(ckt->CKTrhsOld+here->VBICemitEINode));
                Vbcp = *(ckt->CKTstate1 + here->VBICvbcp);
                Vrs = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICsubsNode)-
                    *(ckt->CKTrhsOld+here->VBICsubsSINode));
                if (here->VBIC_selfheat)
                    Vrth = *(ckt->CKTstate1 + here->VBICvrth);
            } else if((ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)){
                Vbe=model->VBICtype*here->VBICicVBE;
                Vbei=Vbex=Vbe;
                vce=model->VBICtype*here->VBICicVCE;
                Vbc=Vbe-vce;
                Vbci=Vbcx=Vbc;
                Vbep=Vbcp=0.0;
                Vrci=Vrbi=Vrbp=0.0;
                Vrcx=Vrbx=Vre=Vrs=0.0;
                Vrth = 0.0, Icth = 0.0, Icth_Vrth = 0.0;
            } else if((ckt->CKTmode & MODEINITJCT) && (here->VBICoff==0)) {
                Vbe=Vbei=Vbex=model->VBICtype*here->VBICtVcrit;
                Vbc=Vbcx=Vbep=0.0;
                Vbci=-model->VBICtype*here->VBICtVcrit;
                Vbcp=Vbc-Vbe;
                Vrci=Vrbi=Vrbp=0.0;
                Vrcx=Vrbx=Vre=Vrs=0.0;
                Vrth = 0.0, Icth = 0.0, Icth_Vrth = 0.0;
            } else if((ckt->CKTmode & MODEINITJCT) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (here->VBICoff!=0))) {
                Vbe=0.0;
                Vbei=Vbex=Vbe;
                Vbc=0.0;
                Vbci=Vbcx=Vbc;
                Vbep=Vbcp=0.0;
                Vrci=Vrbi=Vrbp=0.0;
                Vrcx=Vrbx=Vre=Vrs=0.0;
                Vrth = 0.0, Icth = 0.0, Icth_Vrth = 0.0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact = ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    Vbei = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbei)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbei);
                    Vbex = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbex)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbex);
                    Vbci = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbci)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbci);
                    Vbcx = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbcx)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbcx);
                    Vbep = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbep)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbep);
                    Vrci = (1+xfact) * *(ckt->CKTstate1 + here->VBICvrci)-
                            xfact * *(ckt->CKTstate2 + here->VBICvrci);
                    Vrbi = (1+xfact) * *(ckt->CKTstate1 + here->VBICvrbi)-
                            xfact * *(ckt->CKTstate2 + here->VBICvrbi);
                    Vrbp = (1+xfact) * *(ckt->CKTstate1 + here->VBICvrbp)-
                            xfact * *(ckt->CKTstate2 + here->VBICvrbp);
                    Vbcp = (1+xfact) * *(ckt->CKTstate1 + here->VBICvbcp)-
                            xfact * *(ckt->CKTstate2 + here->VBICvbcp);
                    if (here->VBIC_selfheat) {
                        Vrth = (1.0 + xfact)* (*(ckt->CKTstate1 + here->VBICvrth))
                          - ( xfact * (*(ckt->CKTstate2 + here->VBICvrth)));
                        *(ckt->CKTstate0 + here->VBICvrth) =
                                *(ckt->CKTstate1 + here->VBICvrth);
                    }
                    *(ckt->CKTstate0 + here->VBICvbei) =
                            *(ckt->CKTstate1 + here->VBICvbei);
                    *(ckt->CKTstate0 + here->VBICvbex) =
                            *(ckt->CKTstate1 + here->VBICvbex);
                    *(ckt->CKTstate0 + here->VBICvbci) =
                            *(ckt->CKTstate1 + here->VBICvbci);
                    *(ckt->CKTstate0 + here->VBICvbcx) =
                            *(ckt->CKTstate1 + here->VBICvbcx);
                    *(ckt->CKTstate0 + here->VBICvbep) =
                            *(ckt->CKTstate1 + here->VBICvbep);
                    *(ckt->CKTstate0 + here->VBICvrci) =
                            *(ckt->CKTstate1 + here->VBICvrci);
                    *(ckt->CKTstate0 + here->VBICvrbi) =
                            *(ckt->CKTstate1 + here->VBICvrbi);
                    *(ckt->CKTstate0 + here->VBICvrbp) =
                            *(ckt->CKTstate1 + here->VBICvrbp);
                    *(ckt->CKTstate0 + here->VBICvbcp) =
                            *(ckt->CKTstate1 + here->VBICvbcp);
                    *(ckt->CKTstate0 + here->VBICibe) =
                            *(ckt->CKTstate1 + here->VBICibe);
                    *(ckt->CKTstate0 + here->VBICibe_Vbei) =
                            *(ckt->CKTstate1 + here->VBICibe_Vbei);
                    *(ckt->CKTstate0 + here->VBICibex) =
                            *(ckt->CKTstate1 + here->VBICibex);
                    *(ckt->CKTstate0 + here->VBICibex_Vbex) =
                            *(ckt->CKTstate1 + here->VBICibex_Vbex);
                    *(ckt->CKTstate0 + here->VBICitzf) =
                            *(ckt->CKTstate1 + here->VBICitzf);
                    *(ckt->CKTstate0 + here->VBICitzf_Vbei) =
                            *(ckt->CKTstate1 + here->VBICitzf_Vbei);
                    *(ckt->CKTstate0 + here->VBICitzf_Vbci) =
                            *(ckt->CKTstate1 + here->VBICitzf_Vbci);
                    *(ckt->CKTstate0 + here->VBICitzr) =
                            *(ckt->CKTstate1 + here->VBICitzr);
                    *(ckt->CKTstate0 + here->VBICitzr_Vbei) =
                            *(ckt->CKTstate1 + here->VBICitzf_Vbei);
                    *(ckt->CKTstate0 + here->VBICitzr_Vbci) =
                            *(ckt->CKTstate1 + here->VBICitzr_Vbci);
                    *(ckt->CKTstate0 + here->VBICibc) =
                            *(ckt->CKTstate1 + here->VBICibc);
                    *(ckt->CKTstate0 + here->VBICibc_Vbci) =
                            *(ckt->CKTstate1 + here->VBICibc_Vbci);
                    *(ckt->CKTstate0 + here->VBICibc_Vbei) =
                            *(ckt->CKTstate1 + here->VBICibc_Vbei);
                    *(ckt->CKTstate0 + here->VBICibep) =
                            *(ckt->CKTstate1 + here->VBICibep);
                    *(ckt->CKTstate0 + here->VBICibep_Vbep) =
                            *(ckt->CKTstate1 + here->VBICibep_Vbep);
                    *(ckt->CKTstate0 + here->VBICirci) =
                            *(ckt->CKTstate1 + here->VBICirci);
                    *(ckt->CKTstate0 + here->VBICirci_Vrci) =
                            *(ckt->CKTstate1 + here->VBICirci_Vrci);
                    *(ckt->CKTstate0 + here->VBICirci_Vbci) =
                            *(ckt->CKTstate1 + here->VBICirci_Vbci);
                    *(ckt->CKTstate0 + here->VBICirci_Vbcx) =
                            *(ckt->CKTstate1 + here->VBICirci_Vbcx);
                    *(ckt->CKTstate0 + here->VBICirbi) =
                            *(ckt->CKTstate1 + here->VBICirbi);
                    *(ckt->CKTstate0 + here->VBICirbi_Vrbi) =
                            *(ckt->CKTstate1 + here->VBICirbi_Vrbi);
                    *(ckt->CKTstate0 + here->VBICirbi_Vbei) =
                            *(ckt->CKTstate1 + here->VBICirbi_Vbei);
                    *(ckt->CKTstate0 + here->VBICirbi_Vbci) =
                            *(ckt->CKTstate1 + here->VBICirbi_Vbci);
                    *(ckt->CKTstate0 + here->VBICirbp) =
                            *(ckt->CKTstate1 + here->VBICirbp);
                    *(ckt->CKTstate0 + here->VBICirbp_Vrbp) =
                            *(ckt->CKTstate1 + here->VBICirbp_Vrbp);
                    *(ckt->CKTstate0 + here->VBICirbp_Vbep) =
                            *(ckt->CKTstate1 + here->VBICirbp_Vbep);
                    *(ckt->CKTstate0 + here->VBICirbp_Vbci) =
                            *(ckt->CKTstate1 + here->VBICirbp_Vbci);
                    *(ckt->CKTstate0 + here->VBICibcp) =
                            *(ckt->CKTstate1 + here->VBICibcp);
                    *(ckt->CKTstate0 + here->VBICibcp_Vbcp) =
                            *(ckt->CKTstate1 + here->VBICibcp_Vbcp);
                    *(ckt->CKTstate0 + here->VBICiccp) =
                            *(ckt->CKTstate1 + here->VBICiccp);
                    *(ckt->CKTstate0 + here->VBICiccp_Vbep) =
                            *(ckt->CKTstate1 + here->VBICiccp_Vbep);
                    *(ckt->CKTstate0 + here->VBICiccp_Vbci) =
                            *(ckt->CKTstate1 + here->VBICiccp_Vbci);
                    *(ckt->CKTstate0 + here->VBICiccp_Vbcp) =
                            *(ckt->CKTstate1 + here->VBICiccp_Vbcp);
                    *(ckt->CKTstate0 + here->VBICgqbeo) =
                            *(ckt->CKTstate1 + here->VBICgqbeo);
                    *(ckt->CKTstate0 + here->VBICgqbco) =
                            *(ckt->CKTstate1 + here->VBICgqbco);
                    *(ckt->CKTstate0 + here->VBICircx_Vrcx) =
                            *(ckt->CKTstate1 + here->VBICircx_Vrcx);
                    *(ckt->CKTstate0 + here->VBICirbx_Vrbx) =
                            *(ckt->CKTstate1 + here->VBICirbx_Vrbx);
                    *(ckt->CKTstate0 + here->VBICirs_Vrs) =
                            *(ckt->CKTstate1 + here->VBICirs_Vrs);
                    *(ckt->CKTstate0 + here->VBICire_Vre) =
                            *(ckt->CKTstate1 + here->VBICire_Vre);
                    if (here->VBIC_selfheat)
                        *(ckt->CKTstate0 + here->VBICqcth) =
                                *(ckt->CKTstate1 + here->VBICqcth);
                } else {
#endif /* PREDICTOR */
                    /*
                     *   compute new nonlinear branch voltages
                     */
                    Vbei = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                        *(ckt->CKTrhsOld+here->VBICemitEINode));
                    Vbex = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                        *(ckt->CKTrhsOld+here->VBICemitEINode));
                    Vbci = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                        *(ckt->CKTrhsOld+here->VBICcollCINode));
                    Vbcx = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                        *(ckt->CKTrhsOld+here->VBICcollCXNode));
                    Vbep = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                        *(ckt->CKTrhsOld+here->VBICbaseBPNode));
                    Vrci = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICcollCXNode)-
                        *(ckt->CKTrhsOld+here->VBICcollCINode));
                    Vrbi = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                        *(ckt->CKTrhsOld+here->VBICbaseBINode));
                    Vrbp = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICbaseBPNode)-
                        *(ckt->CKTrhsOld+here->VBICcollCXNode));
                    Vbcp = model->VBICtype*(
                        *(ckt->CKTrhsOld+here->VBICsubsSINode)-
                        *(ckt->CKTrhsOld+here->VBICbaseBPNode));
                    if (here->VBIC_selfheat)
                        Vrth = *(ckt->CKTrhsOld + here->VBICtempNode);
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvbei = Vbei - *(ckt->CKTstate0 + here->VBICvbei);
                delvbex = Vbex - *(ckt->CKTstate0 + here->VBICvbex);
                delvbci = Vbci - *(ckt->CKTstate0 + here->VBICvbci);
                delvbcx = Vbcx - *(ckt->CKTstate0 + here->VBICvbcx);
                delvbep = Vbep - *(ckt->CKTstate0 + here->VBICvbep);
                delvrci = Vrci - *(ckt->CKTstate0 + here->VBICvrci);
                delvrbi = Vrbi - *(ckt->CKTstate0 + here->VBICvrbi);
                delvrbp = Vrbp - *(ckt->CKTstate0 + here->VBICvrbp);
                delvbcp = Vbcp - *(ckt->CKTstate0 + here->VBICvbcp);

                Vbe = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICemitNode));
                Vbc = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICcollNode));
                Vrcx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICcollNode)-
                    *(ckt->CKTrhsOld+here->VBICcollCXNode));
                Vrbx = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICbaseNode)-
                    *(ckt->CKTrhsOld+here->VBICbaseBXNode));
                Vre = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICemitNode)-
                    *(ckt->CKTrhsOld+here->VBICemitEINode));
                Vrs = model->VBICtype*(
                    *(ckt->CKTrhsOld+here->VBICsubsNode)-
                    *(ckt->CKTrhsOld+here->VBICsubsSINode));
                if (here->VBIC_selfheat)
                    Vrth = *(ckt->CKTrhsOld + here->VBICtempNode);

                ibehat = *(ckt->CKTstate0 + here->VBICibe) +
                         *(ckt->CKTstate0 + here->VBICibe_Vbei)*delvbei;
                ibexhat = *(ckt->CKTstate0 + here->VBICibex) +
                         *(ckt->CKTstate0 + here->VBICibex_Vbex)*delvbex;
                itzfhat = *(ckt->CKTstate0 + here->VBICitzf) +
                         *(ckt->CKTstate0 + here->VBICitzf_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICitzf_Vbci)*delvbci;
                itzrhat = *(ckt->CKTstate0 + here->VBICitzr) +
                         *(ckt->CKTstate0 + here->VBICitzr_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICitzr_Vbci)*delvbci;
                ibchat = *(ckt->CKTstate0 + here->VBICibc) +
                         *(ckt->CKTstate0 + here->VBICibc_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICibc_Vbci)*delvbci;
                ibephat = *(ckt->CKTstate0 + here->VBICibep) +
                         *(ckt->CKTstate0 + here->VBICibep_Vbep)*delvbep;
                ircihat = *(ckt->CKTstate0 + here->VBICirci) + *(ckt->CKTstate0 + here->VBICirci_Vrci)*delvrci +
                         *(ckt->CKTstate0 + here->VBICirci_Vbcx)*delvbcx + *(ckt->CKTstate0 + here->VBICirci_Vbci)*delvbci;
                irbihat = *(ckt->CKTstate0 + here->VBICirbi) + *(ckt->CKTstate0 + here->VBICirbi_Vrbi)*delvrbi +
                         *(ckt->CKTstate0 + here->VBICirbi_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICirbi_Vbci)*delvbci;
                irbphat = *(ckt->CKTstate0 + here->VBICirbp) + *(ckt->CKTstate0 + here->VBICirbp_Vrbp)*delvrbp +
                         *(ckt->CKTstate0 + here->VBICirbp_Vbep)*delvbep + *(ckt->CKTstate0 + here->VBICirbp_Vbci)*delvbci;
                ibcphat = *(ckt->CKTstate0 + here->VBICibcp) +
                         *(ckt->CKTstate0 + here->VBICibcp_Vbcp)*delvbcp;
                iccphat = *(ckt->CKTstate0 + here->VBICiccp) + *(ckt->CKTstate0 + here->VBICiccp_Vbep)*delvbep +
                         *(ckt->CKTstate0 + here->VBICiccp_Vbci)*delvbci + *(ckt->CKTstate0 + here->VBICiccp_Vbcp)*delvbcp;
                /*
                 *    bypass if solution has not changed
                 */
                /* the following collections of if's would be just one
                 * if the average compiler could handle it, but many
                 * find the expression too complicated, thus the split.
                 * ... no bypass in case of selfheating
                 */
                if( (ckt->CKTbypass) && (!(ckt->CKTmode & MODEINITPRED)) && !here->VBIC_selfheat &&
                        (fabs(delvbei) < (ckt->CKTreltol*MAX(fabs(Vbei),
                            fabs(*(ckt->CKTstate0 + here->VBICvbei)))+
                            ckt->CKTvoltTol)) )
                    if( (fabs(delvbex) < ckt->CKTreltol*MAX(fabs(Vbex),
                            fabs(*(ckt->CKTstate0 + here->VBICvbex)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbci) < ckt->CKTreltol*MAX(fabs(Vbci),
                            fabs(*(ckt->CKTstate0 + here->VBICvbci)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbcx) < ckt->CKTreltol*MAX(fabs(Vbcx),
                            fabs(*(ckt->CKTstate0 + here->VBICvbcx)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbep) < ckt->CKTreltol*MAX(fabs(Vbep),
                            fabs(*(ckt->CKTstate0 + here->VBICvbep)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvrci) < ckt->CKTreltol*MAX(fabs(Vrci),
                            fabs(*(ckt->CKTstate0 + here->VBICvrci)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvrbi) < ckt->CKTreltol*MAX(fabs(Vrbi),
                            fabs(*(ckt->CKTstate0 + here->VBICvrbi)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvrbp) < ckt->CKTreltol*MAX(fabs(Vrbp),
                            fabs(*(ckt->CKTstate0 + here->VBICvrbp)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(delvbcp) < ckt->CKTreltol*MAX(fabs(Vbcp),
                            fabs(*(ckt->CKTstate0 + here->VBICvbcp)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(ibehat-*(ckt->CKTstate0 + here->VBICibe)) <
                            ckt->CKTreltol* MAX(fabs(ibehat),
                            fabs(*(ckt->CKTstate0 + here->VBICibe)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibexhat-*(ckt->CKTstate0 + here->VBICibex)) <
                            ckt->CKTreltol* MAX(fabs(ibexhat),
                            fabs(*(ckt->CKTstate0 + here->VBICibex)))+
                            ckt->CKTabstol) )
                    if( (fabs(itzfhat-*(ckt->CKTstate0 + here->VBICitzf)) <
                            ckt->CKTreltol* MAX(fabs(itzfhat),
                            fabs(*(ckt->CKTstate0 + here->VBICitzf)))+
                            ckt->CKTabstol) )
                    if( (fabs(itzrhat-*(ckt->CKTstate0 + here->VBICitzr)) <
                            ckt->CKTreltol* MAX(fabs(itzrhat),
                            fabs(*(ckt->CKTstate0 + here->VBICitzr)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibchat-*(ckt->CKTstate0 + here->VBICibc)) <
                            ckt->CKTreltol* MAX(fabs(ibchat),
                            fabs(*(ckt->CKTstate0 + here->VBICibc)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibephat-*(ckt->CKTstate0 + here->VBICibep)) <
                            ckt->CKTreltol* MAX(fabs(ibephat),
                            fabs(*(ckt->CKTstate0 + here->VBICibep)))+
                            ckt->CKTabstol) )
                    if( (fabs(ircihat-*(ckt->CKTstate0 + here->VBICirci)) <
                            ckt->CKTreltol* MAX(fabs(ircihat),
                            fabs(*(ckt->CKTstate0 + here->VBICirci)))+
                            ckt->CKTabstol) )
                    if( (fabs(irbihat-*(ckt->CKTstate0 + here->VBICirbi)) <
                            ckt->CKTreltol* MAX(fabs(irbihat),
                            fabs(*(ckt->CKTstate0 + here->VBICirbi)))+
                            ckt->CKTabstol) )
                    if( (fabs(irbphat-*(ckt->CKTstate0 + here->VBICirbp)) <
                            ckt->CKTreltol* MAX(fabs(irbphat),
                            fabs(*(ckt->CKTstate0 + here->VBICirbp)))+
                            ckt->CKTabstol) )
                    if( (fabs(ibcphat-*(ckt->CKTstate0 + here->VBICibcp)) <
                            ckt->CKTreltol* MAX(fabs(ibcphat),
                            fabs(*(ckt->CKTstate0 + here->VBICibcp)))+
                            ckt->CKTabstol) )
                    if( (fabs(iccphat-*(ckt->CKTstate0 + here->VBICiccp)) <
                            ckt->CKTreltol* MAX(fabs(iccphat),
                            fabs(*(ckt->CKTstate0 + here->VBICiccp)))+
                            ckt->CKTabstol) ) {
                    /*
                     * bypassing....
                     */
                    Vbei = *(ckt->CKTstate0 + here->VBICvbei);
                    Vbex = *(ckt->CKTstate0 + here->VBICvbex);
                    Vbci = *(ckt->CKTstate0 + here->VBICvbci);
                    Vbcx = *(ckt->CKTstate0 + here->VBICvbcx);
                    Vbep = *(ckt->CKTstate0 + here->VBICvbep);
                    Vrci = *(ckt->CKTstate0 + here->VBICvrci);
                    Vrbi = *(ckt->CKTstate0 + here->VBICvrbi);
                    Vrbp = *(ckt->CKTstate0 + here->VBICvrbp);
                    Vbcp = *(ckt->CKTstate0 + here->VBICvbcp);
                    Ibe       = *(ckt->CKTstate0 + here->VBICibe);
                    Ibe_Vbei  = *(ckt->CKTstate0 + here->VBICibe_Vbei);
                    Ibex      = *(ckt->CKTstate0 + here->VBICibex);
                    Ibex_Vbex = *(ckt->CKTstate0 + here->VBICibex_Vbex);
                    Itzf      = *(ckt->CKTstate0 + here->VBICitzf);
                    Itzf_Vbei = *(ckt->CKTstate0 + here->VBICitzf_Vbei);
                    Itzf_Vbci = *(ckt->CKTstate0 + here->VBICitzf_Vbci);
                    Itzr      = *(ckt->CKTstate0 + here->VBICitzr);
                    Itzr_Vbci = *(ckt->CKTstate0 + here->VBICitzr_Vbci);
                    Itzr_Vbei = *(ckt->CKTstate0 + here->VBICitzr_Vbei);
                    Ibc       = *(ckt->CKTstate0 + here->VBICibc);
                    Ibc_Vbci  = *(ckt->CKTstate0 + here->VBICibc_Vbci);
                    Ibc_Vbei  = *(ckt->CKTstate0 + here->VBICibc_Vbei);
                    Ibep      = *(ckt->CKTstate0 + here->VBICibep);
                    Ibep_Vbep = *(ckt->CKTstate0 + here->VBICibep_Vbep);
                    Irci      = *(ckt->CKTstate0 + here->VBICirci);
                    Irci_Vrci = *(ckt->CKTstate0 + here->VBICirci_Vrci);
                    Irci_Vbci = *(ckt->CKTstate0 + here->VBICirci_Vbci);
                    Irci_Vbcx = *(ckt->CKTstate0 + here->VBICirci_Vbcx);
                    Irbi      = *(ckt->CKTstate0 + here->VBICirbi);
                    Irbi_Vrbi = *(ckt->CKTstate0 + here->VBICirbi_Vrbi);
                    Irbi_Vbei = *(ckt->CKTstate0 + here->VBICirbi_Vbei);
                    Irbi_Vbci = *(ckt->CKTstate0 + here->VBICirbi_Vbci);
                    Irbp      = *(ckt->CKTstate0 + here->VBICirbp);
                    Irbp_Vrbp = *(ckt->CKTstate0 + here->VBICirbp_Vrbp);
                    Irbp_Vbep = *(ckt->CKTstate0 + here->VBICirbp_Vbep);
                    Irbp_Vbci = *(ckt->CKTstate0 + here->VBICirbp_Vbci);
                    Ibcp      = *(ckt->CKTstate0 + here->VBICibcp);
                    Ibcp_Vbcp = *(ckt->CKTstate0 + here->VBICibcp_Vbcp);
                    Iccp      = *(ckt->CKTstate0 + here->VBICiccp);
                    Iccp_Vbep = *(ckt->CKTstate0 + here->VBICiccp_Vbep);
                    Iccp_Vbci = *(ckt->CKTstate0 + here->VBICiccp_Vbci);
                    Iccp_Vbcp = *(ckt->CKTstate0 + here->VBICiccp_Vbcp);
                    gqbeo     = *(ckt->CKTstate0 + here->VBICgqbeo);
                    gqbco     = *(ckt->CKTstate0 + here->VBICgqbco);
                    Ircx_Vrcx = *(ckt->CKTstate0 + here->VBICircx_Vrcx);
                    Irbx_Vrbx = *(ckt->CKTstate0 + here->VBICirbx_Vrbx);
                    Irs_Vrs   = *(ckt->CKTstate0 + here->VBICirs_Vrs);
                    Ire_Vre   = *(ckt->CKTstate0 + here->VBICire_Vre);
                    goto load;
                }
                /*
                 *   limit nonlinear branch voltages
                 */
                ichk1 = 1, ichk2 = 1, ichk3 = 1, ichk4 = 1, ichk5 = 1, ichk6 = 0;
                Vbei = DEVpnjlim(Vbei,*(ckt->CKTstate0 + here->VBICvbei),vt,
                        here->VBICtVcrit,&icheck);
                Vbex = DEVpnjlim(Vbex,*(ckt->CKTstate0 + here->VBICvbex),vt,
                        here->VBICtVcrit,&ichk1);
                Vbci = DEVpnjlim(Vbci,*(ckt->CKTstate0 + here->VBICvbci),vt,
                        here->VBICtVcrit,&ichk2);
                Vbcx = DEVpnjlim(Vbcx,*(ckt->CKTstate0 + here->VBICvbcx),vt,
                        here->VBICtVcrit,&ichk3);
                Vbep = DEVpnjlim(Vbep,*(ckt->CKTstate0 + here->VBICvbep),vt,
                        here->VBICtVcrit,&ichk4);
                Vbcp = DEVpnjlim(Vbcp,*(ckt->CKTstate0 + here->VBICvbcp),vt,
                        here->VBICtVcrit,&ichk5);
                if (here->VBIC_selfheat) {
                    ichk6 = 1;
                    Vrth = DEVlimitlog(Vrth,
                        *(ckt->CKTstate0 + here->VBICvrth),100,&ichk6);
                }
                if ((ichk1 == 1) || (ichk2 == 1) || (ichk3 == 1) || (ichk4 == 1) || (ichk5 == 1) || (ichk6 == 1)) icheck=1;
            }
            /*
             *   determine dc current and derivatives
             */
            Vcei = Vbei - Vbci;
            Vcep = Vbep - Vbcp;
            if (here->VBIC_selfheat) {
                iret = vbic_4T_et_cf_fj(p
                    ,&Vrth, &Vbei, &Vbex, &Vbci, &Vbep, &Vbcp
                    ,&Vrcx, &Vbcx, &Vrci, &Vrbx, &Vrbi, &Vre, &Vrbp
                    ,&Vrs, &Vbe, &Vbc, &Vcei, &Vcep, &Ibe, &Ibe_Vrth
                    ,&Ibe_Vbei, &Ibex, &Ibex_Vrth, &Ibex_Vbex, &Itzf, &Itzf_Vrth, &Itzf_Vbei
                    ,&Itzf_Vbci, &Itzr, &Itzr_Vrth, &Itzr_Vbci, &Itzr_Vbei, &Ibc, &Ibc_Vrth
                    ,&Ibc_Vbci, &Ibc_Vbei, &Ibep, &Ibep_Vrth, &Ibep_Vbep, &Ircx, &Ircx_Vrcx
                    ,&Ircx_Vrth, &Irci, &Irci_Vrci, &Irci_Vrth, &Irci_Vbci, &Irci_Vbcx, &Irbx
                    ,&Irbx_Vrbx, &Irbx_Vrth, &Irbi, &Irbi_Vrbi, &Irbi_Vrth, &Irbi_Vbei, &Irbi_Vbci
                    ,&Ire, &Ire_Vre, &Ire_Vrth, &Irbp, &Irbp_Vrbp, &Irbp_Vrth, &Irbp_Vbep
                    ,&Irbp_Vbci, &Qbe, &Qbe_Vrth, &Qbe_Vbei, &Qbe_Vbci, &Qbex, &Qbex_Vrth
                    ,&Qbex_Vbex, &Qbc, &Qbc_Vrth, &Qbc_Vbci, &Qbcx, &Qbcx_Vrth, &Qbcx_Vbcx
                    ,&Qbep, &Qbep_Vrth, &Qbep_Vbep, &Qbep_Vbci, &Qbeo, &Qbeo_Vbe, &Qbco
                    ,&Qbco_Vbc, &Ibcp, &Ibcp_Vrth, &Ibcp_Vbcp, &Iccp, &Iccp_Vrth, &Iccp_Vbep
                    ,&Iccp_Vbci, &Iccp_Vbcp, &Irs, &Irs_Vrs, &Irs_Vrth, &Qbcp, &Qbcp_Vrth
                    ,&Qbcp_Vbcp, &Irth, &Irth_Vrth, &Ith, &Ith_Vrth, &Ith_Vbei, &Ith_Vbci
                    ,&Ith_Vcei, &Ith_Vbex, &Ith_Vbep, &Ith_Vrs, &Ith_Vbcp, &Ith_Vcep, &Ith_Vrcx
                    ,&Ith_Vrci, &Ith_Vbcx, &Ith_Vrbx, &Ith_Vrbi, &Ith_Vre, &Ith_Vrbp, &Qcth
                    ,&Qcth_Vrth, &SCALE);
            } else {
                iret = vbic_4T_it_cf_fj(p
                    ,&Vbei, &Vbex, &Vbci, &Vbep, &Vbcp, &Vrcx
                    ,&Vbcx, &Vrci, &Vrbx, &Vrbi, &Vre, &Vrbp, &Vrs
                    ,&Vbe, &Vbc, &Ibe, &Ibe_Vbei, &Ibex, &Ibex_Vbex, &Itzf
                    ,&Itzf_Vbei, &Itzf_Vbci, &Itzr, &Itzr_Vbci, &Itzr_Vbei, &Ibc, &Ibc_Vbci
                    ,&Ibc_Vbei, &Ibep, &Ibep_Vbep, &Ircx, &Ircx_Vrcx, &Irci, &Irci_Vrci
                    ,&Irci_Vbci, &Irci_Vbcx, &Irbx, &Irbx_Vrbx, &Irbi, &Irbi_Vrbi, &Irbi_Vbei
                    ,&Irbi_Vbci, &Ire, &Ire_Vre, &Irbp, &Irbp_Vrbp, &Irbp_Vbep, &Irbp_Vbci
                    ,&Qbe, &Qbe_Vbei, &Qbe_Vbci, &Qbex, &Qbex_Vbex, &Qbc, &Qbc_Vbci
                    ,&Qbcx, &Qbcx_Vbcx, &Qbep, &Qbep_Vbep, &Qbep_Vbci, &Qbeo, &Qbeo_Vbe
                    ,&Qbco, &Qbco_Vbc, &Ibcp, &Ibcp_Vbcp, &Iccp, &Iccp_Vbep, &Iccp_Vbci
                    ,&Iccp_Vbcp, &Irs, &Irs_Vrs, &Qbcp, &Qbcp_Vbcp, &SCALE);
            }
            Ibe += ckt->CKTgmin*Vbei;
            Ibe_Vbei += ckt->CKTgmin;
            Ibex += ckt->CKTgmin*Vbex;
            Ibex_Vbex += ckt->CKTgmin;
            Ibc += ckt->CKTgmin*Vbci;
            Ibc_Vbci += ckt->CKTgmin;
            Ibep += ckt->CKTgmin*Vbep;
            Ibep_Vbep += ckt->CKTgmin;
            Irci += ckt->CKTgmin*Vrci;
            Irci_Vrci += ckt->CKTgmin;
            Irci += ckt->CKTgmin*Vbci;
            Irci_Vbci += ckt->CKTgmin;
            Irci += ckt->CKTgmin*Vbcx;
            Irci_Vbcx += ckt->CKTgmin;
            Ibcp += ckt->CKTgmin*Vbcp;
            Ibcp_Vbcp += ckt->CKTgmin;

            if( (ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN | MODEAC)) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ||
                    (ckt->CKTmode & MODEINITSMSIG)) {
                /*
                 *   charge storage elements
                 */

                *(ckt->CKTstate0 + here->VBICqbe)  = Qbe;
                *(ckt->CKTstate0 + here->VBICqbex) = Qbex;
                *(ckt->CKTstate0 + here->VBICqbc)  = Qbc;
                *(ckt->CKTstate0 + here->VBICqbcx) = Qbcx;
                *(ckt->CKTstate0 + here->VBICqbep) = Qbep;
                *(ckt->CKTstate0 + here->VBICqbeo) = Qbeo;
                *(ckt->CKTstate0 + here->VBICqbco) = Qbco;
                *(ckt->CKTstate0 + here->VBICqbcp) = Qbcp;
                if (here->VBIC_selfheat)
                    *(ckt->CKTstate0 + here->VBICqcth) = Qcth;

                here->VBICcapbe = Qbe_Vbei;
                here->VBICcapbex = Qbex_Vbex;
                here->VBICcapbc = Qbc_Vbci;
                here->VBICcapbcx = Qbcx_Vbcx;
                here->VBICcapbep = Qbep_Vbep;
                here->VBICcapbcp = Qbcp_Vbcp;
                if (here->VBIC_selfheat)
                    here->VBICcapcth = Qcth_Vrth;

                /*
                 *   store small-signal parameters
                 */
                if ( (!(ckt->CKTmode & MODETRANOP))||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->VBICcqbe)    = Qbe_Vbei;
                        *(ckt->CKTstate0 + here->VBICcqbeci)  = Qbe_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbex)   = Qbex_Vbex;
                        *(ckt->CKTstate0 + here->VBICcqbc)    = Qbc_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbcx)   = Qbcx_Vbcx;
                        *(ckt->CKTstate0 + here->VBICcqbep)   = Qbep_Vbep;
                        *(ckt->CKTstate0 + here->VBICcqbepci) = Qbep_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbeo)   = Qbeo_Vbe;
                        *(ckt->CKTstate0 + here->VBICcqbco)   = Qbco_Vbc;
                        *(ckt->CKTstate0 + here->VBICcqbcp)   = Qbcp_Vbcp;
                        if (here->VBIC_selfheat)
                            *(ckt->CKTstate0 + here->VBICcqcth)   = Qcth_Vrth;
                        continue; /* go to 1000 */
                    }
                    /*
                     *   transient analysis
                     */

                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->VBICqbe) =
                                *(ckt->CKTstate0 + here->VBICqbe) ;
                        *(ckt->CKTstate1 + here->VBICqbex) =
                                *(ckt->CKTstate0 + here->VBICqbex) ;
                        *(ckt->CKTstate1 + here->VBICqbc) =
                                *(ckt->CKTstate0 + here->VBICqbc) ;
                        *(ckt->CKTstate1 + here->VBICqbcx) =
                                *(ckt->CKTstate0 + here->VBICqbcx) ;
                        *(ckt->CKTstate1 + here->VBICqbep) =
                                *(ckt->CKTstate0 + here->VBICqbep) ;
                        *(ckt->CKTstate1 + here->VBICqbeo) =
                                *(ckt->CKTstate0 + here->VBICqbeo) ;
                        *(ckt->CKTstate1 + here->VBICqbco) =
                                *(ckt->CKTstate0 + here->VBICqbco) ;
                        *(ckt->CKTstate1 + here->VBICqbcp) =
                                *(ckt->CKTstate0 + here->VBICqbcp) ;
                        if (here->VBIC_selfheat)
                            *(ckt->CKTstate1 + here->VBICqcth) =
                                    *(ckt->CKTstate0 + here->VBICqcth) ;
                    }
                    error = NIintegrate(ckt,&geq,&ceq,Qbe_Vbei,here->VBICqbe);
                    if(error) return(error);
                    Ibe_Vbei = Ibe_Vbei + geq;
                    Ibe = Ibe + *(ckt->CKTstate0 + here->VBICcqbe);

                    error = NIintegrate(ckt,&geq,&ceq,Qbex_Vbex,here->VBICqbex);
                    if(error) return(error);
                    Ibex_Vbex = Ibex_Vbex + geq;
                    Ibex = Ibex + *(ckt->CKTstate0 + here->VBICcqbex);

                    error = NIintegrate(ckt,&geq,&ceq,Qbc_Vbci,here->VBICqbc);
                    if(error) return(error);
                    Ibc_Vbci = Ibc_Vbci + geq;
                    Ibc = Ibc + *(ckt->CKTstate0 + here->VBICcqbc);

                    error = NIintegrate(ckt,&geq,&ceq,Qbcx_Vbcx,here->VBICqbcx);
                    if(error) return(error);
                    gbcx = geq;
                    cbcx = *(ckt->CKTstate0 + here->VBICcqbcx);

                    error = NIintegrate(ckt,&geq,&ceq,Qbep_Vbep,here->VBICqbep);
                    if(error) return(error);
                    Ibep_Vbep = Ibep_Vbep + geq;
                    Ibep = Ibep + *(ckt->CKTstate0 + here->VBICcqbep);

                    error = NIintegrate(ckt,&geq,&ceq,Qbcp_Vbcp,here->VBICqbcp);
                    if(error) return(error);
                    Ibcp_Vbcp = Ibcp_Vbcp + geq;
                    Ibcp = Ibcp + *(ckt->CKTstate0 + here->VBICcqbcp);

                    if (here->VBIC_selfheat)
                    {
                        error = NIintegrate(ckt,&geq,&ceq,Qcth_Vrth,here->VBICqcth);
                        if(error) return(error);
                        Icth_Vrth = geq;
                        Icth = *(ckt->CKTstate0 + here->VBICcqcth);
                    }

                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->VBICcqbe) =
                                *(ckt->CKTstate0 + here->VBICcqbe);
                        *(ckt->CKTstate1 + here->VBICcqbex) =
                                *(ckt->CKTstate0 + here->VBICcqbex);
                        *(ckt->CKTstate1 + here->VBICcqbc) =
                                *(ckt->CKTstate0 + here->VBICcqbc);
                        *(ckt->CKTstate1 + here->VBICcqbcx) =
                                *(ckt->CKTstate0 + here->VBICcqbcx);
                        *(ckt->CKTstate1 + here->VBICcqbep) =
                                *(ckt->CKTstate0 + here->VBICcqbep);
                        *(ckt->CKTstate1 + here->VBICcqbcp) =
                                *(ckt->CKTstate0 + here->VBICcqbcp);
                        if (here->VBIC_selfheat)
                            *(ckt->CKTstate1 + here->VBICcqcth) =
                                    *(ckt->CKTstate0 + here->VBICcqcth);
                    }
                }
            }

            /*
             *   check convergence
             */
            if ( (!(ckt->CKTmode & MODEINITFIX))||(!(here->VBICoff))) {
                if (icheck == 1) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }

            /*
             *      charge storage for outer b-e and b-c junctions
             */
            if(ckt->CKTmode & (MODETRAN | MODEAC)) {
                error = NIintegrate(ckt,&gqbeo,&cqbeo,Qbeo_Vbe,here->VBICqbeo);
                if(error) return(error);
                error = NIintegrate(ckt,&gqbco,&cqbco,Qbco_Vbc,here->VBICqbco);
                if(error) return(error);
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->VBICcqbeo) =
                            *(ckt->CKTstate0 + here->VBICcqbeo);
                    *(ckt->CKTstate1 + here->VBICcqbco) =
                            *(ckt->CKTstate0 + here->VBICcqbco);
                }
            }

            *(ckt->CKTstate0 + here->VBICvbei)      = Vbei;
            *(ckt->CKTstate0 + here->VBICvbex)      = Vbex;
            *(ckt->CKTstate0 + here->VBICvbci)      = Vbci;
            *(ckt->CKTstate0 + here->VBICvbcx)      = Vbcx;
            *(ckt->CKTstate0 + here->VBICvbep)      = Vbep;
            *(ckt->CKTstate0 + here->VBICvrci)      = Vrci;
            *(ckt->CKTstate0 + here->VBICvrbi)      = Vrbi;
            *(ckt->CKTstate0 + here->VBICvrbp)      = Vrbp;
            *(ckt->CKTstate0 + here->VBICvbcp)      = Vbcp;

            *(ckt->CKTstate0 + here->VBICibe)       = Ibe;
            *(ckt->CKTstate0 + here->VBICibe_Vbei)  = Ibe_Vbei;
            *(ckt->CKTstate0 + here->VBICibex)      = Ibex;
            *(ckt->CKTstate0 + here->VBICibex_Vbex) = Ibex_Vbex;
            *(ckt->CKTstate0 + here->VBICitzf)      = Itzf;
            *(ckt->CKTstate0 + here->VBICitzf_Vbei) = Itzf_Vbei;
            *(ckt->CKTstate0 + here->VBICitzf_Vbci) = Itzf_Vbci;
            *(ckt->CKTstate0 + here->VBICitzr)      = Itzr;
            *(ckt->CKTstate0 + here->VBICitzr_Vbci) = Itzr_Vbci;
            *(ckt->CKTstate0 + here->VBICitzr_Vbei) = Itzr_Vbei;
            *(ckt->CKTstate0 + here->VBICibc)       = Ibc;
            *(ckt->CKTstate0 + here->VBICibc_Vbci)  = Ibc_Vbci;
            *(ckt->CKTstate0 + here->VBICibc_Vbei)  = Ibc_Vbei;
            *(ckt->CKTstate0 + here->VBICibep)      = Ibep;
            *(ckt->CKTstate0 + here->VBICibep_Vbep) = Ibep_Vbep;
            *(ckt->CKTstate0 + here->VBICirci)      = Irci;
            *(ckt->CKTstate0 + here->VBICirci_Vrci) = Irci_Vrci;
            *(ckt->CKTstate0 + here->VBICirci_Vbci) = Irci_Vbci;
            *(ckt->CKTstate0 + here->VBICirci_Vbcx) = Irci_Vbcx;
            *(ckt->CKTstate0 + here->VBICirbi)      = Irbi;
            *(ckt->CKTstate0 + here->VBICirbi_Vrbi) = Irbi_Vrbi;
            *(ckt->CKTstate0 + here->VBICirbi_Vbei) = Irbi_Vbei;
            *(ckt->CKTstate0 + here->VBICirbi_Vbci) = Irbi_Vbci;
            *(ckt->CKTstate0 + here->VBICirbp)      = Irbp;
            *(ckt->CKTstate0 + here->VBICirbp_Vrbp) = Irbp_Vrbp;
            *(ckt->CKTstate0 + here->VBICirbp_Vbep) = Irbp_Vbep;
            *(ckt->CKTstate0 + here->VBICirbp_Vbci) = Irbp_Vbci;
            *(ckt->CKTstate0 + here->VBICibcp)      = Ibcp;
            *(ckt->CKTstate0 + here->VBICibcp_Vbcp) = Ibcp_Vbcp;
            *(ckt->CKTstate0 + here->VBICiccp)      = Iccp;
            *(ckt->CKTstate0 + here->VBICiccp_Vbep) = Iccp_Vbep;
            *(ckt->CKTstate0 + here->VBICiccp_Vbci) = Iccp_Vbci;
            *(ckt->CKTstate0 + here->VBICiccp_Vbcp) = Iccp_Vbcp;
            *(ckt->CKTstate0 + here->VBICgqbeo)     = gqbeo;
            *(ckt->CKTstate0 + here->VBICgqbco)     = gqbco;
            *(ckt->CKTstate0 + here->VBICircx_Vrcx) = Ircx_Vrcx;
            *(ckt->CKTstate0 + here->VBICirbx_Vrbx) = Irbx_Vrbx;
            *(ckt->CKTstate0 + here->VBICirs_Vrs)   = Irs_Vrs;
            *(ckt->CKTstate0 + here->VBICire_Vre)   = Ire_Vre;
            if (here->VBIC_selfheat)
            {
                *(ckt->CKTstate0 + here->VBICcqcth)     = Icth;
                *(ckt->CKTstate0 + here->VBICicth_Vrth) = Icth_Vrth;
            }
load:
            /*
             *  load current excitation vector and matrix
             */
            rhs_current = model->VBICtype * (*(ckt->CKTstate0 + here->VBICcqbeo) - Vbe * gqbeo);
            *(ckt->CKTrhs + here->VBICbaseNode) += -rhs_current;
            *(ckt->CKTrhs + here->VBICemitNode) +=  rhs_current;
            *(here->VBICbaseBasePtr) +=  gqbeo;
            *(here->VBICemitEmitPtr) +=  gqbeo;
            *(here->VBICbaseEmitPtr) += -gqbeo;
            *(here->VBICemitBasePtr) += -gqbeo;

            rhs_current = model->VBICtype * (*(ckt->CKTstate0 + here->VBICcqbco) - Vbc * gqbco);
            *(ckt->CKTrhs + here->VBICbaseNode) += -rhs_current;
            *(ckt->CKTrhs + here->VBICcollNode) +=  rhs_current;
            *(here->VBICbaseBasePtr) +=  gqbco;
            *(here->VBICcollCollPtr) +=  gqbco;
            *(here->VBICbaseCollPtr) += -gqbco;
            *(here->VBICcollBasePtr) += -gqbco;

            *(ckt->CKTrhs + here->VBICbaseBINode) += -cbcx;
            *(ckt->CKTrhs + here->VBICcollCXNode) +=  cbcx;
            *(here->VBICbaseBIBaseBIPtr) +=  gbcx;
            *(here->VBICcollCXCollCXPtr) +=  gbcx;
            *(here->VBICbaseBICollCXPtr) += -gbcx;
            *(here->VBICcollCXBaseBIPtr) += -gbcx;

/*
c           MNA at internal nodes
c
c           Stamp element: Ibe
*/
            rhs_current = model->VBICtype * (Ibe - Ibe_Vbei*Vbei);
            *(ckt->CKTrhs + here->VBICbaseBINode) += -rhs_current;
            *(here->VBICbaseBIBaseBIPtr) +=  Ibe_Vbei;
            *(here->VBICbaseBIEmitEIPtr) += -Ibe_Vbei;
            *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
            *(here->VBICemitEIBaseBIPtr) += -Ibe_Vbei;
            *(here->VBICemitEIEmitEIPtr) +=  Ibe_Vbei;
/*
c           Stamp element: Ibex
*/
            rhs_current = model->VBICtype * (Ibex - Ibex_Vbex*Vbex);
            *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
            *(here->VBICbaseBXBaseBXPtr) +=  Ibex_Vbex;
            *(here->VBICbaseBXEmitEIPtr) += -Ibex_Vbex;
            *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
            *(here->VBICemitEIBaseBXPtr) += -Ibex_Vbex;
            *(here->VBICemitEIEmitEIPtr) +=  Ibex_Vbex;

/*
c           Stamp element: Itzf
*/
            rhs_current = model->VBICtype * (Itzf - Itzf_Vbei*Vbei - Itzf_Vbci*Vbci);
            *(ckt->CKTrhs + here->VBICcollCINode) += -rhs_current;
            *(here->VBICcollCIBaseBIPtr) +=  Itzf_Vbei;
            *(here->VBICcollCIEmitEIPtr) += -Itzf_Vbei;
            *(here->VBICcollCIBaseBIPtr) +=  Itzf_Vbci;
            *(here->VBICcollCICollCIPtr) += -Itzf_Vbci;
            *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
            *(here->VBICemitEIBaseBIPtr) += -Itzf_Vbei;
            *(here->VBICemitEIEmitEIPtr) +=  Itzf_Vbei;
            *(here->VBICemitEIBaseBIPtr) += -Itzf_Vbci;
            *(here->VBICemitEICollCIPtr) +=  Itzf_Vbci;
/*
c           Stamp element: Itzr
*/
            rhs_current = model->VBICtype * (Itzr - Itzr_Vbei*Vbei - Itzr_Vbci*Vbci);
            *(ckt->CKTrhs + here->VBICemitEINode) += -rhs_current;
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbei;
            *(here->VBICemitEIEmitEIPtr) += -Itzr_Vbei;
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbci;
            *(here->VBICemitEICollCIPtr) += -Itzr_Vbci;
            *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbei;
            *(here->VBICcollCIEmitEIPtr) +=  Itzr_Vbei;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Itzr_Vbci;
/*
c           Stamp element: Ibc
*/
            rhs_current = model->VBICtype * (Ibc - Ibc_Vbci*Vbci - Ibc_Vbei*Vbei);
            *(ckt->CKTrhs + here->VBICbaseBINode) += -rhs_current;
            *(here->VBICbaseBIBaseBIPtr) +=  Ibc_Vbci;
            *(here->VBICbaseBICollCIPtr) += -Ibc_Vbci;
            *(here->VBICbaseBIBaseBIPtr) +=  Ibc_Vbei;
            *(here->VBICbaseBIEmitEIPtr) += -Ibc_Vbei;
            *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
            *(here->VBICcollCIBaseBIPtr) += -Ibc_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Ibc_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Ibc_Vbei;
            *(here->VBICcollCIEmitEIPtr) +=  Ibc_Vbei;
/*
c           Stamp element: Ibep
*/
            rhs_current = model->VBICtype * (Ibep - Ibep_Vbep*Vbep);
            *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
            *(here->VBICbaseBXBaseBXPtr) +=  Ibep_Vbep;
            *(here->VBICbaseBXBaseBPPtr) += -Ibep_Vbep;
            *(ckt->CKTrhs + here->VBICbaseBPNode) +=  rhs_current;
            *(here->VBICbaseBPBaseBXPtr) += -Ibep_Vbep;
            *(here->VBICbaseBPBaseBPPtr) +=  Ibep_Vbep;
/*
c           Stamp element: Rcx
*/
            *(here->VBICcollCollPtr) +=  Ircx_Vrcx;
            *(here->VBICcollCXCollCXPtr) +=  Ircx_Vrcx;
            *(here->VBICcollCXCollPtr) += -Ircx_Vrcx;
            *(here->VBICcollCollCXPtr) += -Ircx_Vrcx;

/*
c           Stamp element: Irci
*/
            rhs_current = model->VBICtype * (Irci - Irci_Vrci*Vrci - Irci_Vbci*Vbci - Irci_Vbcx*Vbcx);
            *(ckt->CKTrhs + here->VBICcollCXNode) += -rhs_current;
            *(here->VBICcollCXCollCXPtr) +=  Irci_Vrci;
            *(here->VBICcollCXCollCIPtr) += -Irci_Vrci;
            *(here->VBICcollCXBaseBIPtr) +=  Irci_Vbci;
            *(here->VBICcollCXCollCIPtr) += -Irci_Vbci;
            *(here->VBICcollCXBaseBIPtr) +=  Irci_Vbcx;
            *(here->VBICcollCXCollCXPtr) += -Irci_Vbcx;
            *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
            *(here->VBICcollCICollCXPtr) += -Irci_Vrci;
            *(here->VBICcollCICollCIPtr) +=  Irci_Vrci;
            *(here->VBICcollCIBaseBIPtr) += -Irci_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Irci_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Irci_Vbcx;
            *(here->VBICcollCICollCXPtr) +=  Irci_Vbcx;
/*
c           Stamp element: Rbx
*/
            *(here->VBICbaseBasePtr) +=  Irbx_Vrbx;
            *(here->VBICbaseBXBaseBXPtr) +=  Irbx_Vrbx;
            *(here->VBICbaseBXBasePtr) += -Irbx_Vrbx;
            *(here->VBICbaseBaseBXPtr) += -Irbx_Vrbx;
/*
c           Stamp element: Irbi
*/
            rhs_current = model->VBICtype * (Irbi - Irbi_Vrbi*Vrbi - Irbi_Vbei*Vbei - Irbi_Vbci*Vbci);
            *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
            *(here->VBICbaseBXBaseBXPtr) +=  Irbi_Vrbi;
            *(here->VBICbaseBXBaseBIPtr) += -Irbi_Vrbi;
            *(here->VBICbaseBXBaseBIPtr) +=  Irbi_Vbei;
            *(here->VBICbaseBXEmitEIPtr) += -Irbi_Vbei;
            *(here->VBICbaseBXBaseBIPtr) +=  Irbi_Vbci;
            *(here->VBICbaseBXCollCIPtr) += -Irbi_Vbci;
            *(ckt->CKTrhs + here->VBICbaseBINode) +=  rhs_current;
            *(here->VBICbaseBIBaseBXPtr) += -Irbi_Vrbi;
            *(here->VBICbaseBIBaseBIPtr) +=  Irbi_Vrbi;
            *(here->VBICbaseBIBaseBIPtr) += -Irbi_Vbei;
            *(here->VBICbaseBIEmitEIPtr) +=  Irbi_Vbei;
            *(here->VBICbaseBIBaseBIPtr) += -Irbi_Vbci;
            *(here->VBICbaseBICollCIPtr) +=  Irbi_Vbci;
/*
c           Stamp element: Re
*/
            *(here->VBICemitEmitPtr) +=  Ire_Vre;
            *(here->VBICemitEIEmitEIPtr) +=  Ire_Vre;
            *(here->VBICemitEIEmitPtr) += -Ire_Vre;
            *(here->VBICemitEmitEIPtr) += -Ire_Vre;
/*
c           Stamp element: Irbp
*/
            rhs_current = model->VBICtype * (Irbp - Irbp_Vrbp*Vrbp - Irbp_Vbep*Vbep - Irbp_Vbci*Vbci);
            *(ckt->CKTrhs + here->VBICbaseBPNode) += -rhs_current;
            *(here->VBICbaseBPBaseBPPtr) +=  Irbp_Vrbp;
            *(here->VBICbaseBPCollCXPtr) += -Irbp_Vrbp;
            *(here->VBICbaseBPBaseBXPtr) +=  Irbp_Vbep;
            *(here->VBICbaseBPBaseBPPtr) += -Irbp_Vbep;
            *(here->VBICbaseBPBaseBIPtr) +=  Irbp_Vbci;
            *(here->VBICbaseBPCollCIPtr) += -Irbp_Vbci;
            *(ckt->CKTrhs + here->VBICcollCXNode) +=  rhs_current;
            *(here->VBICcollCXBaseBPPtr) += -Irbp_Vrbp;
            *(here->VBICcollCXCollCXPtr) +=  Irbp_Vrbp;
            *(here->VBICcollCXBaseBXPtr) += -Irbp_Vbep;
            *(here->VBICcollCXBaseBPPtr) +=  Irbp_Vbep;
            *(here->VBICcollCXBaseBIPtr) += -Irbp_Vbci;
            *(here->VBICcollCXCollCIPtr) +=  Irbp_Vbci;
/*
c           Stamp element: Ibcp
*/
            rhs_current = model->VBICtype * (Ibcp - Ibcp_Vbcp*Vbcp);
            *(ckt->CKTrhs + here->VBICsubsSINode) += -rhs_current;
            *(here->VBICsubsSISubsSIPtr) +=  Ibcp_Vbcp;
            *(here->VBICsubsSIBaseBPPtr) += -Ibcp_Vbcp;
            *(ckt->CKTrhs + here->VBICbaseBPNode) +=  rhs_current;
            *(here->VBICbaseBPSubsSIPtr) += -Ibcp_Vbcp;
            *(here->VBICbaseBPBaseBPPtr) +=  Ibcp_Vbcp;
/*
c           Stamp element: Iccp
*/
            rhs_current = model->VBICtype * (Iccp - Iccp_Vbep*Vbep - Iccp_Vbci*Vbci - Iccp_Vbcp*Vbcp);
            *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
            *(here->VBICbaseBXBaseBXPtr) +=  Iccp_Vbep;
            *(here->VBICbaseBXBaseBPPtr) += -Iccp_Vbep;
            *(here->VBICbaseBXBaseBIPtr) +=  Iccp_Vbci;
            *(here->VBICbaseBXCollCIPtr) += -Iccp_Vbci;
            *(here->VBICbaseBXSubsSIPtr) +=  Iccp_Vbcp;
            *(here->VBICbaseBXBaseBPPtr) += -Iccp_Vbcp;
            *(ckt->CKTrhs + here->VBICsubsSINode) +=  rhs_current;
            *(here->VBICsubsSIBaseBXPtr) += -Iccp_Vbep;
            *(here->VBICsubsSIBaseBPPtr) +=  Iccp_Vbep;
            *(here->VBICsubsSIBaseBIPtr) += -Iccp_Vbci;
            *(here->VBICsubsSICollCIPtr) +=  Iccp_Vbci;
            *(here->VBICsubsSISubsSIPtr) += -Iccp_Vbcp;
            *(here->VBICsubsSIBaseBPPtr) +=  Iccp_Vbcp;
/*
c           Stamp element: Rs
*/
            *(here->VBICsubsSubsPtr) +=  Irs_Vrs;
            *(here->VBICsubsSISubsSIPtr) +=  Irs_Vrs;
            *(here->VBICsubsSISubsPtr) += -Irs_Vrs;
            *(here->VBICsubsSubsSIPtr) += -Irs_Vrs;

            if (here->VBIC_selfheat) {
/*
c               Stamp element: Ibe
*/
                rhs_current = -Ibe_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBINode) += -rhs_current;
                *(here->VBICbaseBItempPtr) +=  Ibe_Vrth;
                *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
                *(here->VBICemitEItempPtr) += -Ibe_Vrth;
/*
c               Stamp element: Ibex
*/
                rhs_current = -Ibex_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
                *(here->VBICbaseBXtempPtr) +=  Ibex_Vrth;
                *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
                *(here->VBICemitEItempPtr) += -Ibex_Vrth;
/*
c               Stamp element: Itzf
*/
                rhs_current = -Itzf_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICcollCINode) += -rhs_current;
                *(here->VBICcollCItempPtr) +=  Itzf_Vrth;
                *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
                *(here->VBICemitEItempPtr) += -Itzf_Vrth;
/*
c               Stamp element: Itzr
*/
                rhs_current = -Itzr_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICemitEINode) += -rhs_current;
                *(here->VBICemitEItempPtr) +=  Itzr_Vrth;
                *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
                *(here->VBICcollCItempPtr) += -Itzr_Vrth;
/*
c               Stamp element: Ibc
*/
                rhs_current = -Ibc_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBINode) += -rhs_current;
                *(here->VBICbaseBItempPtr) +=  Ibc_Vrth;
                *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
                *(here->VBICcollCItempPtr) += -Ibc_Vrth;
/*
c               Stamp element: Ibep
*/
                rhs_current = -Ibep_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
                *(here->VBICbaseBXtempPtr) +=  Ibep_Vrth;
                *(ckt->CKTrhs + here->VBICbaseBPNode) +=  rhs_current;
                *(here->VBICbaseBPtempPtr) += -Ibep_Vrth;
/*
c               Stamp element: Rcx
*/
                rhs_current = -Ircx_Vrth * Vrth;
                *(ckt->CKTrhs + here->VBICcollNode)   += -rhs_current;
                *(here->VBICcollTempPtr)   +=  Ircx_Vrth;
                *(ckt->CKTrhs + here->VBICcollCXNode) +=  rhs_current;
                *(here->VBICcollCXtempPtr) += -Ircx_Vrth;
/*
c               Stamp element: Irci
*/
                rhs_current = -Irci_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICcollCXNode) += -rhs_current;
                *(here->VBICcollCXtempPtr) +=  Irci_Vrth;
                *(ckt->CKTrhs + here->VBICcollCINode) +=  rhs_current;
                *(here->VBICcollCItempPtr) += -Irci_Vrth;
/*
c               Stamp element: Rbx
*/
                rhs_current = -Irbx_Vrth * Vrth;
                *(ckt->CKTrhs + here->VBICbaseNode)   += -rhs_current;
                *(here->VBICbaseTempPtr)   +=  Irbx_Vrth;
                *(ckt->CKTrhs + here->VBICbaseBXNode) +=  rhs_current;
                *(here->VBICbaseBXtempPtr) += -Irbx_Vrth;
/*
c               Stamp element: Irbi
*/
                rhs_current = -Irbi_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
                *(here->VBICbaseBXtempPtr) +=  Irbi_Vrth;
                *(ckt->CKTrhs + here->VBICbaseBINode) +=  rhs_current;
                *(here->VBICbaseBItempPtr) += -Irbi_Vrth;
/*
c               Stamp element: Re
*/
                rhs_current = -Ire_Vrth * Vrth;
                *(ckt->CKTrhs + here->VBICemitNode)   += -rhs_current;
                *(here->VBICemitTempPtr)   +=  Ire_Vrth;
                *(ckt->CKTrhs + here->VBICemitEINode) +=  rhs_current;
                *(here->VBICemitEItempPtr) += -Ire_Vrth;
/*
c               Stamp element: Irbp
*/
                rhs_current = -Irbp_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBPNode) += -rhs_current;
                *(here->VBICbaseBPtempPtr) +=  Irbp_Vrth;
                *(ckt->CKTrhs + here->VBICcollCXNode) +=  rhs_current;
                *(here->VBICcollCXtempPtr) += -Irbp_Vrth;
/*
c               Stamp element: Ibcp
*/
                rhs_current = -Ibcp_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICsubsSINode) += -rhs_current;
                *(here->VBICsubsSItempPtr) +=  Ibcp_Vrth;
                *(ckt->CKTrhs + here->VBICbaseBPNode) +=  rhs_current;
                *(here->VBICbaseBPtempPtr) += -Ibcp_Vrth;
/*
c               Stamp element: Iccp
*/
                rhs_current = -Iccp_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICbaseBXNode) += -rhs_current;
                *(here->VBICbaseBXtempPtr) +=  Iccp_Vrth;
                *(ckt->CKTrhs + here->VBICsubsSINode) +=  rhs_current;
                *(here->VBICsubsSItempPtr) += -Iccp_Vrth;
/*
c               Stamp element: Rs
*/
                rhs_current = -Irs_Vrth * Vrth;
                *(ckt->CKTrhs + here->VBICsubsNode)   += -rhs_current;
                *(here->VBICsubsTempPtr)   +=  Irs_Vrth;
                *(ckt->CKTrhs + here->VBICsubsSINode) +=  rhs_current;
                *(here->VBICsubsSItempPtr) += -Irs_Vrth;
/*
c               Stamp element: Rth
*/
                *(here->VBICtempTempPtr) +=  Irth_Vrth;
/*
c               Stamp element: Cth
*/
                rhs_current = Icth - Icth_Vrth*Vrth;
                *(ckt->CKTrhs + here->VBICtempNode) += -rhs_current;
                *(here->VBICtempTempPtr) +=  Icth_Vrth;
/*
c               Stamp element: Ith
*/
                rhs_current = -Ith - Ith_Vrth*Vrth
                                   - Ith_Vbei*Vbei - Ith_Vbci*Vbci - Ith_Vcei*Vcei
                                   - Ith_Vbex*Vbex - Ith_Vbep*Vbep - Ith_Vbcp*Vbcp
                                   - Ith_Vcep*Vcep - Ith_Vrci*Vrci - Ith_Vbcx*Vbcx
                                   - Ith_Vrbi*Vrbi - Ith_Vrbp*Vrbp
                                   - Ith_Vrcx*Vrcx - Ith_Vrbx*Vrbx - Ith_Vre*Vre - Ith_Vrs*Vrs;

                *(ckt->CKTrhs + here->VBICtempNode) += rhs_current;

                *(here->VBICtempTempPtr)   += -Ith_Vrth;

                *(here->VBICtempBaseBIPtr) += -Ith_Vbei;
                *(here->VBICtempEmitEIPtr) += +Ith_Vbei;
                *(here->VBICtempBaseBIPtr) += -Ith_Vbci;
                *(here->VBICtempCollCIPtr) += +Ith_Vbci;
                *(here->VBICtempCollCIPtr) += -Ith_Vcei;
                *(here->VBICtempEmitEIPtr) += +Ith_Vcei;
                *(here->VBICtempBaseBXPtr) += -Ith_Vbex;
                *(here->VBICtempEmitEIPtr) += +Ith_Vbex;
                *(here->VBICtempBaseBXPtr) += -Ith_Vbep;
                *(here->VBICtempBaseBPPtr) += +Ith_Vbep;
                *(here->VBICtempSubsPtr)   += -Ith_Vbcp;
                *(here->VBICtempBaseBPPtr) += +Ith_Vbcp;
                *(here->VBICtempBaseBXPtr) += -Ith_Vcep;
                *(here->VBICtempSubsPtr)   += +Ith_Vcep;
                *(here->VBICtempCollCXPtr) += -Ith_Vrci;
                *(here->VBICtempCollCIPtr) += +Ith_Vrci;
                *(here->VBICtempBaseBIPtr) += -Ith_Vbcx;
                *(here->VBICtempCollCXPtr) += +Ith_Vbcx;
                *(here->VBICtempBaseBXPtr) += -Ith_Vrbi;
                *(here->VBICtempBaseBIPtr) += +Ith_Vrbi;
                *(here->VBICtempBaseBPPtr) += -Ith_Vrbp;
                *(here->VBICtempCollCXPtr) += +Ith_Vrbp;
                *(here->VBICtempCollPtr)   += -Ith_Vrcx;
                *(here->VBICtempCollCXPtr) += +Ith_Vrcx;
                *(here->VBICtempBasePtr)   += -Ith_Vrbx;
                *(here->VBICtempBaseBXPtr) += +Ith_Vrbx;
                *(here->VBICtempEmitPtr)   += -Ith_Vre;
                *(here->VBICtempEmitEIPtr) += +Ith_Vre;
                *(here->VBICtempSubsPtr)   += -Ith_Vrs;
                *(here->VBICtempSubsSIPtr) += +Ith_Vrs;
            }
        }

    }
    return(OK);
}

int vbic_4T_et_cf_fj(double *p
    ,double *Vrth, double *Vbei, double *Vbex, double *Vbci, double *Vbep, double *Vbcp
    ,double *Vrcx, double *Vbcx, double *Vrci, double *Vrbx, double *Vrbi, double *Vre, double *Vrbp
    ,double *Vrs, double *Vbe, double *Vbc, double *Vcei, double *Vcep, double *Ibe, double *Ibe_Vrth
    ,double *Ibe_Vbei, double *Ibex, double *Ibex_Vrth, double *Ibex_Vbex, double *Itzf, double *Itzf_Vrth, double *Itzf_Vbei
    ,double *Itzf_Vbci, double *Itzr, double *Itzr_Vrth, double *Itzr_Vbci, double *Itzr_Vbei, double *Ibc, double *Ibc_Vrth
    ,double *Ibc_Vbci, double *Ibc_Vbei, double *Ibep, double *Ibep_Vrth, double *Ibep_Vbep, double *Ircx, double *Ircx_Vrcx
    ,double *Ircx_Vrth, double *Irci, double *Irci_Vrci, double *Irci_Vrth, double *Irci_Vbci, double *Irci_Vbcx, double *Irbx
    ,double *Irbx_Vrbx, double *Irbx_Vrth, double *Irbi, double *Irbi_Vrbi, double *Irbi_Vrth, double *Irbi_Vbei, double *Irbi_Vbci
    ,double *Ire, double *Ire_Vre, double *Ire_Vrth, double *Irbp, double *Irbp_Vrbp, double *Irbp_Vrth, double *Irbp_Vbep
    ,double *Irbp_Vbci, double *Qbe, double *Qbe_Vrth, double *Qbe_Vbei, double *Qbe_Vbci, double *Qbex, double *Qbex_Vrth
    ,double *Qbex_Vbex, double *Qbc, double *Qbc_Vrth, double *Qbc_Vbci, double *Qbcx, double *Qbcx_Vrth, double *Qbcx_Vbcx
    ,double *Qbep, double *Qbep_Vrth, double *Qbep_Vbep, double *Qbep_Vbci, double *Qbeo, double *Qbeo_Vbe, double *Qbco
    ,double *Qbco_Vbc, double *Ibcp, double *Ibcp_Vrth, double *Ibcp_Vbcp, double *Iccp, double *Iccp_Vrth, double *Iccp_Vbep
    ,double *Iccp_Vbci, double *Iccp_Vbcp, double *Irs, double *Irs_Vrs, double *Irs_Vrth, double *Qbcp, double *Qbcp_Vrth
    ,double *Qbcp_Vbcp, double *Irth, double *Irth_Vrth, double *Ith, double *Ith_Vrth, double *Ith_Vbei, double *Ith_Vbci
    ,double *Ith_Vcei, double *Ith_Vbex, double *Ith_Vbep, double *Ith_Vrs, double *Ith_Vbcp, double *Ith_Vcep, double *Ith_Vrcx
    ,double *Ith_Vrci, double *Ith_Vbcx, double *Ith_Vrbx, double *Ith_Vrbi, double *Ith_Vre, double *Ith_Vrbp, double *Qcth
    ,double *Qcth_Vrth, double *SCALE)
{
double Tini,Tdev,Tdev_Vrth,Vtv,Vtv_Tdev,Vtv_Vrth,rT;
double rT_Tdev,rT_Vrth,dT,dT_Tdev,dT_Vrth,xvar1,xvar1_rT;
double xvar1_Vrth,IKFatT,IKFatT_xvar1,IKFatT_Vrth,RCXatT,RCXatT_xvar1,RCXatT_Vrth;
double RCIatT,RCIatT_xvar1,RCIatT_Vrth,RBXatT,RBXatT_xvar1,RBXatT_Vrth,RBIatT;
double RBIatT_xvar1,RBIatT_Vrth,REatT,REatT_xvar1,REatT_Vrth,RSatT,RSatT_xvar1;
double RSatT_Vrth,RBPatT,RBPatT_xvar1,RBPatT_Vrth,xvar2,xvar2_rT,xvar2_Vrth;
double xvar3,xvar3_rT,xvar3_Vrth,xvar3_Vtv,xvar4,xvar4_xvar3,xvar4_Vrth;
double xvar1_xvar2,xvar1_xvar4,xvar5,xvar6,xvar6_xvar1,xvar6_Vrth,ISatT;
double ISatT_xvar6,ISatT_Vrth,ISRRatT,ISRRatT_xvar6,ISRRatT_Vrth,ISPatT,ISPatT_xvar6;
double ISPatT_Vrth,IBEIatT,IBEIatT_xvar6,IBEIatT_Vrth,IBENatT,IBENatT_xvar6,IBENatT_Vrth;
double IBCIatT,IBCIatT_xvar6,IBCIatT_Vrth,IBCNatT,IBCNatT_xvar6,IBCNatT_Vrth,IBEIPatT;
double IBEIPatT_xvar6,IBEIPatT_Vrth,IBENPatT,IBENPatT_xvar6,IBENPatT_Vrth,IBCIPatT,IBCIPatT_xvar6;
double IBCIPatT_Vrth,IBCNPatT,IBCNPatT_xvar6,IBCNPatT_Vrth,NFatT,NFatT_dT,NFatT_Vrth;
double NRatT,NRatT_dT,NRatT_Vrth,AVC2atT,AVC2atT_dT,AVC2atT_Vrth,VBBEatT;
double VBBEatT_dT,VBBEatT_Vrth,NBBEatT,NBBEatT_dT,NBBEatT_Vrth,xvar2_Vtv,xvar3_xvar2;
double xvar4_rT,xvar4_Vtv,xvar5_xvar4,xvar5_Vrth,xvar1_xvar3,xvar1_xvar5,psiio;
double psiio_Vtv,psiio_Vrth,psiio_rT,psiio_xvar6,psiin,psiin_psiio,psiin_Vrth;
double psiin_rT,psiin_Vtv,psiin_xvar1,xvar2_psiin,xvar4_xvar1,PEatT,PEatT_psiin;
double PEatT_Vrth,PEatT_Vtv,PEatT_xvar4,PCatT,PCatT_psiin,PCatT_Vrth,PCatT_Vtv;
double PCatT_xvar4,PSatT,PSatT_psiin,PSatT_Vrth,PSatT_Vtv,PSatT_xvar4,xvar1_PEatT;
double xvar2_xvar1,CJEatT,CJEatT_xvar2,CJEatT_Vrth,xvar1_PCatT,CJCatT,CJCatT_xvar2;
double CJCatT_Vrth,CJEPatT,CJEPatT_xvar2,CJEPatT_Vrth,xvar1_PSatT,CJCPatT,CJCPatT_xvar2;
double CJCPatT_Vrth,GAMMatT,GAMMatT_xvar1,GAMMatT_Vrth,GAMMatT_xvar3,VOatT,VOatT_xvar1;
double VOatT_Vrth,xvar1_VBBEatT,xvar1_NBBEatT,xvar1_Vtv,EBBEatT,EBBEatT_xvar1,EBBEatT_Vrth;
double IVEF,IVER,IIKF,IIKF_IKFatT,IIKF_Vrth,IIKR,IIKP;
double IVO,IVO_VOatT,IVO_Vrth,IHRCF,IVTF,IITF,slTF;
double dv0,dv0_PEatT,dv0_Vrth,dvh,dvh_Vbei,dvh_dv0,dvh_Vrth;
double pwq,qlo,qlo_PEatT,qlo_Vrth,qlo_Vbei,qhi,qhi_dvh;
double qhi_Vbei,qhi_Vrth,qhi_PEatT,xvar1_Vbei,xvar3_xvar1,xvar3_Vbei,qlo_xvar3;
double qdbe,qdbe_qlo,qdbe_Vrth,qdbe_Vbei,qdbe_qhi,mv0,mv0_dv0;
double mv0_Vrth,vl0,vl0_dv0,vl0_Vrth,vl0_mv0,xvar1_vl0,q0;
double q0_PEatT,q0_Vrth,q0_xvar3,dv,dv_Vbei,dv_dv0,dv_Vrth;
double mv,mv_dv,mv_Vbei,mv_Vrth,vl,vl_dv,vl_Vbei;
double vl_Vrth,vl_mv,vl_dv0,xvar1_vl,qdbe_vl,qdbe_vl0,qdbe_q0;
double dvh_Vbex,qlo_Vbex,qhi_Vbex,xvar1_Vbex,xvar3_Vbex,qdbex,qdbex_qlo;
double qdbex_Vrth,qdbex_Vbex,qdbex_qhi,dv_Vbex,mv_Vbex,vl_Vbex,qdbex_vl;
double qdbex_vl0,qdbex_q0,dv0_PCatT,dvh_Vbci,qlo_PCatT,qlo_Vbci,qhi_Vbci;
double qhi_PCatT,xvar1_Vbci,xvar3_Vbci,qdbc,qdbc_qlo,qdbc_Vrth,qdbc_Vbci;
double qdbc_qhi,vn0,vn0_dv0,vn0_Vrth,vnl0,vnl0_vn0,vnl0_Vrth;
double vl0_vnl0,qlo0,qlo0_PCatT,qlo0_Vrth,qlo0_xvar3,vn,vn_Vbci;
double vn_dv0,vn_Vrth,vnl,vnl_vn,vnl_Vbci,vnl_Vrth,vl_vnl;
double vl_Vbci,sel,sel_vnl,sel_Vbci,sel_Vrth,crt,crt_xvar1;
double crt_Vrth,xvar1_dv0,cmx,cmx_xvar1,cmx_Vrth,cl,cl_sel;
double cl_Vbci,cl_Vrth,cl_crt,cl_cmx,ql,ql_Vbci,ql_vl;
double ql_Vrth,ql_vl0,ql_cl,qdbc_ql,qdbc_qlo0,q0_PCatT,dv_Vbci;
double mv_Vbci,qdbc_vl,qdbc_vl0,qdbc_q0,dvh_Vbep,qlo_Vbep,qhi_Vbep;
double xvar1_Vbep,xvar3_Vbep,qdbep,qdbep_qlo,qdbep_Vrth,qdbep_Vbep,qdbep_qhi;
double vn_Vbep,vnl_Vbep,vl_Vbep,sel_Vbep,cl_Vbep,ql_Vbep,qdbep_ql;
double qdbep_qlo0,dv_Vbep,mv_Vbep,qdbep_vl,qdbep_vl0,qdbep_q0,dv0_PSatT;
double dvh_Vbcp,qlo_PSatT,qlo_Vbcp,qhi_Vbcp,qhi_PSatT,xvar1_Vbcp,xvar3_Vbcp;
double qdbcp,qdbcp_qlo,qdbcp_Vrth,qdbcp_Vbcp,qdbcp_qhi,q0_PSatT;
double dv_Vbcp,mv_Vbcp,vl_Vbcp,qdbcp_vl,qdbcp_vl0,qdbcp_q0,argi;
double argi_Vbei,argi_NFatT,argi_Vrth,argi_Vtv,expi,expi_argi,expi_Vbei;
double expi_Vrth,Ifi,Ifi_ISatT,Ifi_Vrth,Ifi_expi,Ifi_Vbei,argi_Vbci;
double argi_NRatT,expi_Vbci,Iri,Iri_ISatT,Iri_Vrth,Iri_ISRRatT,Iri_expi;
double Iri_Vbci,q1z,q1z_qdbe,q1z_Vrth,q1z_Vbei,q1z_qdbc,q1z_Vbci;
double q1,q1_q1z,q1_Vrth,q1_Vbei,q1_Vbci,q2,q2_Ifi;
double q2_Vrth,q2_Vbei,q2_IIKF,q2_Iri,q2_Vbci,xvar3_q1,xvar1_q2;
double xvar4_Vbei,xvar4_Vbci,qb,qb_q1,qb_Vrth,qb_Vbei,qb_Vbci;
double qb_xvar4,xvar2_Vbei,xvar2_Vbci,qb_xvar2,Itzr_Iri,Itzr_qb,Itzf_Ifi;
double Itzf_qb,argi_Vbep,expi_Vbep,argx,argx_Vbci,argx_Vtv,argx_Vrth;
double expx,expx_argx,expx_Vbci,expx_Vrth,Ifp,Ifp_ISPatT,Ifp_Vrth;
double Ifp_expi,Ifp_Vbep,Ifp_expx,Ifp_Vbci,q2p,q2p_Ifp,q2p_Vrth;
double q2p_Vbep,q2p_Vbci,qbp,qbp_q2p,qbp_Vrth,qbp_Vbep,qbp_Vbci;
double argi_Vbcp,expi_Vbcp,Irp,Irp_ISPatT,Irp_Vrth,Irp_expi,Irp_Vbcp;
double Iccp_Ifp,Iccp_Irp,Iccp_qbp,argn,argn_Vbei,argn_Vtv,argn_Vrth;
double expn,expn_argn,expn_Vbei,expn_Vrth,argx_VBBEatT,argx_Vbei,argx_NBBEatT;
double expx_Vbei,Ibe_IBEIatT,Ibe_expi,Ibe_IBENatT,Ibe_expn,Ibe_expx,Ibe_EBBEatT;
double argi_Vbex,expi_Vbex,argn_Vbex,expn_Vbex,argx_Vbex,expx_Vbex,Ibex_IBEIatT;
double Ibex_expi,Ibex_IBENatT,Ibex_expn,Ibex_expx,Ibex_EBBEatT,argn_Vbci,expn_Vbci;
double Ibcj,Ibcj_IBCIatT,Ibcj_Vrth,Ibcj_expi,Ibcj_Vbci,Ibcj_IBCNatT,Ibcj_expn;
double argn_Vbep,expn_Vbep,Ibep_IBEIPatT,Ibep_expi,Ibep_IBENPatT,Ibep_expn,vl_PCatT;
double xvar3_vl,xvar1_AVC2atT,avalf,avalf_vl,avalf_Vrth,avalf_Vbci,avalf_xvar4;
double Igc,Igc_Itzf,Igc_Vrth,Igc_Vbei,Igc_Vbci,Igc_Itzr,Igc_Ibcj;
double Igc_avalf,Ibc_Ibcj,Ibc_Igc,Ircx_RCXatT,argx_Vbcx,expx_Vbcx,Kbci;
double Kbci_GAMMatT,Kbci_Vrth,Kbci_expi,Kbci_Vbci,Kbcx,Kbcx_GAMMatT,Kbcx_Vrth;
double Kbcx_expx,Kbcx_Vbcx,rKp1,rKp1_Kbci,rKp1_Vrth,rKp1_Vbci,rKp1_Kbcx;
double rKp1_Vbcx,xvar1_rKp1,xvar1_Vbcx,Iohm,Iohm_Vrci,Iohm_Vtv,Iohm_Vrth;
double Iohm_Kbci,Iohm_Vbci,Iohm_Kbcx,Iohm_Vbcx,Iohm_xvar1,Iohm_RCIatT,derf;
double derf_IVO,derf_Vrth,derf_RCIatT,derf_Iohm,derf_Vrci,derf_Vbci,derf_Vbcx;
double Irci_Iohm,Irci_derf,Irbx_RBXatT,Irbi_qb,Irbi_RBIatT,Ire_REatT,Irbp_qbp;
double Irbp_RBPatT,argn_Vbcp,expn_Vbcp,Ibcp_IBCIPatT,Ibcp_expi,Ibcp_IBCNPatT,Ibcp_expn;
double Irs_RSatT,sgIf,rIf,rIf_Ifi,rIf_Vrth,rIf_Vbei,mIf;
double mIf_rIf,mIf_Vrth,mIf_Vbei,tff,tff_q1,tff_Vrth,tff_Vbei;
double tff_Vbci,tff_xvar2,tff_mIf,Qbe_CJEatT,Qbe_qdbe,Qbe_tff,Qbe_Ifi;
double Qbe_qb,Qbex_CJEatT,Qbex_qdbex,Qbc_CJCatT,Qbc_qdbc,Qbc_Iri,Qbc_Kbci;
double Qbcx_Kbcx,Qbep_CJEPatT,Qbep_qdbep,Qbep_Ifp,Qbcp_CJCPatT,Qbcp_qdbcp,Ith_Ibe;
double Ith_Ibc,Ith_Itzf,Ith_Itzr,Ith_Ibex,Ith_Ibep,Ith_Irs,Ith_Ibcp;
double Ith_Iccp,Ith_Ircx,Ith_Irci,Ith_Irbx,Ith_Irbi,Ith_Ire,Ith_Irbp;

/*  Function and derivative code */

    Tini=2.731500e+02+p[0];
    Tdev=(2.731500e+02+p[0])+(*Vrth);
    Tdev_Vrth=1.0;
    Vtv=1.380662e-23*Tdev/1.602189e-19;
    Vtv_Tdev=8.617347e-5;
    Vtv_Vrth=Vtv_Tdev*Tdev_Vrth;
    rT=Tdev/Tini;
    rT_Tdev=1.0/Tini;
    rT_Vrth=rT_Tdev*Tdev_Vrth;
    dT=Tdev-Tini;
    dT_Tdev=1.0;
    dT_Vrth=dT_Tdev*Tdev_Vrth;
    xvar1=pow(rT,p[90]);
    xvar1_rT=xvar1*p[90]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    IKFatT=p[53]*xvar1;
    IKFatT_xvar1=p[53];
    IKFatT_Vrth=IKFatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[91]);
    xvar1_rT=xvar1*p[91]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RCXatT=p[1]*xvar1;
    RCXatT_xvar1=p[1];
    RCXatT_Vrth=RCXatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[68]);
    xvar1_rT=xvar1*p[68]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RCIatT=p[2]*xvar1;
    RCIatT_xvar1=p[2];
    RCIatT_Vrth=RCIatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[92]);
    xvar1_rT=xvar1*p[92]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RBXatT=p[6]*xvar1;
    RBXatT_xvar1=p[6];
    RBXatT_Vrth=RBXatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[67]);
    xvar1_rT=xvar1*p[67]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RBIatT=p[7]*xvar1;
    RBIatT_xvar1=p[7];
    RBIatT_Vrth=RBIatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[66]);
    xvar1_rT=xvar1*p[66]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    REatT=p[8]*xvar1;
    REatT_xvar1=p[8];
    REatT_Vrth=REatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[69]);
    xvar1_rT=xvar1*p[69]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RSatT=p[9]*xvar1;
    RSatT_xvar1=p[9];
    RSatT_Vrth=RSatT_xvar1*xvar1_Vrth;
    xvar1=pow(rT,p[93]);
    xvar1_rT=xvar1*p[93]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    RBPatT=p[10]*xvar1;
    RBPatT_xvar1=p[10];
    RBPatT_Vrth=RBPatT_xvar1*xvar1_Vrth;
    xvar2=pow(rT,p[78]);
    xvar2_rT=xvar2*p[78]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[71]*(1.0-rT)/Vtv;
    xvar3_rT=p[71]/Vtv;
    xvar3_Vtv=p[71]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[12]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    ISatT=p[11]*xvar6;
    ISatT_xvar6=p[11];
    ISatT_Vrth=ISatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[95]);
    xvar2_rT=xvar2*p[95]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[96]*(1.0-rT)/Vtv;
    xvar3_rT=p[96]/Vtv;
    xvar3_Vtv=p[96]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[13]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    ISRRatT=p[94]*xvar6;
    ISRRatT_xvar6=p[94];
    ISRRatT_Vrth=ISRRatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[78]);
    xvar2_rT=xvar2*p[78]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[97]*(1.0-rT)/Vtv;
    xvar3_rT=p[97]/Vtv;
    xvar3_Vtv=p[97]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[44]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    ISPatT=p[42]*xvar6;
    ISPatT_xvar6=p[42];
    ISPatT_Vrth=ISPatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[79]);
    xvar2_rT=xvar2*p[79]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[72]*(1.0-rT)/Vtv;
    xvar3_rT=p[72]/Vtv;
    xvar3_Vtv=p[72]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[33]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBEIatT=p[31]*xvar6;
    IBEIatT_xvar6=p[31];
    IBEIatT_Vrth=IBEIatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[80]);
    xvar2_rT=xvar2*p[80]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[75]*(1.0-rT)/Vtv;
    xvar3_rT=p[75]/Vtv;
    xvar3_Vtv=p[75]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[35]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBENatT=p[34]*xvar6;
    IBENatT_xvar6=p[34];
    IBENatT_Vrth=IBENatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[79]);
    xvar2_rT=xvar2*p[79]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[73]*(1.0-rT)/Vtv;
    xvar3_rT=p[73]/Vtv;
    xvar3_Vtv=p[73]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[37]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBCIatT=p[36]*xvar6;
    IBCIatT_xvar6=p[36];
    IBCIatT_Vrth=IBCIatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[80]);
    xvar2_rT=xvar2*p[80]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[76]*(1.0-rT)/Vtv;
    xvar3_rT=p[76]/Vtv;
    xvar3_Vtv=p[76]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[39]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBCNatT=p[38]*xvar6;
    IBCNatT_xvar6=p[38];
    IBCNatT_Vrth=IBCNatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[79]);
    xvar2_rT=xvar2*p[79]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[73]*(1.0-rT)/Vtv;
    xvar3_rT=p[73]/Vtv;
    xvar3_Vtv=p[73]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[37]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBEIPatT=p[45]*xvar6;
    IBEIPatT_xvar6=p[45];
    IBEIPatT_Vrth=IBEIPatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[80]);
    xvar2_rT=xvar2*p[80]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[76]*(1.0-rT)/Vtv;
    xvar3_rT=p[76]/Vtv;
    xvar3_Vtv=p[76]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[39]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBENPatT=p[46]*xvar6;
    IBENPatT_xvar6=p[46];
    IBENPatT_Vrth=IBENPatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[79]);
    xvar2_rT=xvar2*p[79]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[74]*(1.0-rT)/Vtv;
    xvar3_rT=p[74]/Vtv;
    xvar3_Vtv=p[74]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[48]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBCIPatT=p[47]*xvar6;
    IBCIPatT_xvar6=p[47];
    IBCIPatT_Vrth=IBCIPatT_xvar6*xvar6_Vrth;
    xvar2=pow(rT,p[80]);
    xvar2_rT=xvar2*p[80]/rT;
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar3=-p[77]*(1.0-rT)/Vtv;
    xvar3_rT=p[77]/Vtv;
    xvar3_Vtv=p[77]*(1.0-rT)/(Vtv*Vtv);
    xvar3_Vrth=xvar3_rT*rT_Vrth;
    xvar3_Vrth=xvar3_Vrth+xvar3_Vtv*Vtv_Vrth;
    xvar4=exp(xvar3);
    xvar4_xvar3=xvar4;
    xvar4_Vrth=xvar4_xvar3*xvar3_Vrth;
    xvar1=(xvar2*xvar4);
    xvar1_xvar2=xvar4;
    xvar1_xvar4=xvar2;
    xvar1_Vrth=xvar1_xvar2*xvar2_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar4*xvar4_Vrth;
    xvar5=(1.0/p[50]);
    xvar6=pow(xvar1,xvar5);
    xvar6_xvar1=xvar6*xvar5/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    IBCNPatT=p[49]*xvar6;
    IBCNPatT_xvar6=p[49];
    IBCNPatT_Vrth=IBCNPatT_xvar6*xvar6_Vrth;
    NFatT=p[12]*(1.0+dT*p[81]);
    NFatT_dT=p[12]*p[81];
    NFatT_Vrth=NFatT_dT*dT_Vrth;
    NRatT=p[13]*(1.0+dT*p[81]);
    NRatT_dT=p[13]*p[81];
    NRatT_Vrth=NRatT_dT*dT_Vrth;
    AVC2atT=p[41]*(1.0+dT*p[82]);
    AVC2atT_dT=p[41]*p[82];
    AVC2atT_Vrth=AVC2atT_dT*dT_Vrth;
    VBBEatT=p[98]*(1.0+dT*(p[101]+dT*p[102]));
    VBBEatT_dT=(2.0*dT*p[102]+p[101])*p[98];
    VBBEatT_Vrth=VBBEatT_dT*dT_Vrth;
    NBBEatT=p[99]*(1.0+dT*p[103]);
    NBBEatT_dT=p[99]*p[103];
    NBBEatT_Vrth=NBBEatT_dT*dT_Vrth;
    xvar2=0.5*p[17]*rT/Vtv;
    xvar2_rT=0.5*p[17]/Vtv;
    xvar2_Vtv=-0.5*p[17]*rT/(Vtv*Vtv);
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar4=-0.5*p[17]*rT/Vtv;
    xvar4_rT=-0.5*p[17]/Vtv;
    xvar4_Vtv=0.5*p[17]*rT/(Vtv*Vtv);
    xvar4_Vrth=xvar4_rT*rT_Vrth;
    xvar4_Vrth=xvar4_Vrth+xvar4_Vtv*Vtv_Vrth;
    xvar5=exp(xvar4);
    xvar5_xvar4=xvar5;
    xvar5_Vrth=xvar5_xvar4*xvar4_Vrth;
    xvar1=xvar3-xvar5;
    xvar1_xvar3=1.0;
    xvar1_xvar5=-1.0;
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar5*xvar5_Vrth;
    xvar6=log(xvar1);
    xvar6_xvar1=1.0/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    psiio=2.0*(Vtv/rT)*xvar6;
    psiio_Vtv=2.0*xvar6/rT;
    psiio_rT=-2.0*Vtv*xvar6/(rT*rT);
    psiio_xvar6=2.0*Vtv/rT;
    psiio_Vrth=psiio_Vtv*Vtv_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_rT*rT_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_xvar6*xvar6_Vrth;
    xvar1=log(rT);
    xvar1_rT=1.0/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    psiin=psiio*rT-3.0*Vtv*xvar1-p[72]*(rT-1.0);
    psiin_psiio=rT;
    psiin_rT=psiio-p[72];
    psiin_Vtv=-3.0*xvar1;
    psiin_xvar1=-3.0*Vtv;
    psiin_Vrth=psiin_psiio*psiio_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_rT*rT_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_Vtv*Vtv_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_xvar1*xvar1_Vrth;
    xvar2=-psiin/Vtv;
    xvar2_psiin=-1.0/Vtv;
    xvar2_Vtv=psiin/(Vtv*Vtv);
    xvar2_Vrth=xvar2_psiin*psiin_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
    xvar1_xvar3=1.0/sqrt(4.0*xvar3+1.0);
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar4=log(xvar1);
    xvar4_xvar1=1.0/xvar1;
    xvar4_Vrth=xvar4_xvar1*xvar1_Vrth;
    PEatT=psiin+2.0*Vtv*xvar4;
    PEatT_psiin=1.0;
    PEatT_Vtv=2.0*xvar4;
    PEatT_xvar4=2.0*Vtv;
    PEatT_Vrth=PEatT_psiin*psiin_Vrth;
    PEatT_Vrth=PEatT_Vrth+PEatT_Vtv*Vtv_Vrth;
    PEatT_Vrth=PEatT_Vrth+PEatT_xvar4*xvar4_Vrth;
    xvar2=0.5*p[24]*rT/Vtv;
    xvar2_rT=0.5*p[24]/Vtv;
    xvar2_Vtv=-0.5*p[24]*rT/(Vtv*Vtv);
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar4=-0.5*p[24]*rT/Vtv;
    xvar4_rT=-0.5*p[24]/Vtv;
    xvar4_Vtv=0.5*p[24]*rT/(Vtv*Vtv);
    xvar4_Vrth=xvar4_rT*rT_Vrth;
    xvar4_Vrth=xvar4_Vrth+xvar4_Vtv*Vtv_Vrth;
    xvar5=exp(xvar4);
    xvar5_xvar4=xvar5;
    xvar5_Vrth=xvar5_xvar4*xvar4_Vrth;
    xvar1=xvar3-xvar5;
    xvar1_xvar3=1.0;
    xvar1_xvar5=-1.0;
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar5*xvar5_Vrth;
    xvar6=log(xvar1);
    xvar6_xvar1=1.0/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    psiio=2.0*(Vtv/rT)*xvar6;
    psiio_Vtv=2.0*xvar6/rT;
    psiio_rT=-2.0*Vtv*xvar6/(rT*rT);
    psiio_xvar6=2.0*Vtv/rT;
    psiio_Vrth=psiio_Vtv*Vtv_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_rT*rT_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_xvar6*xvar6_Vrth;
    xvar1=log(rT);
    xvar1_rT=1.0/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    psiin=psiio*rT-3.0*Vtv*xvar1-p[73]*(rT-1.0);
    psiin_psiio=rT;
    psiin_rT=psiio-p[73];
    psiin_Vtv=-3.0*xvar1;
    psiin_xvar1=-3.0*Vtv;
    psiin_Vrth=psiin_psiio*psiio_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_rT*rT_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_Vtv*Vtv_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_xvar1*xvar1_Vrth;
    xvar2=-psiin/Vtv;
    xvar2_psiin=-1.0/Vtv;
    xvar2_Vtv=psiin/(Vtv*Vtv);
    xvar2_Vrth=xvar2_psiin*psiin_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
    xvar1_xvar3=1.0/sqrt(4.0*xvar3+1.0);
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar4=log(xvar1);
    xvar4_xvar1=1.0/xvar1;
    xvar4_Vrth=xvar4_xvar1*xvar1_Vrth;
    PCatT=psiin+2.0*Vtv*xvar4;
    PCatT_psiin=1.0;
    PCatT_Vtv=2.0*xvar4;
    PCatT_xvar4=2.0*Vtv;
    PCatT_Vrth=PCatT_psiin*psiin_Vrth;
    PCatT_Vrth=PCatT_Vrth+PCatT_Vtv*Vtv_Vrth;
    PCatT_Vrth=PCatT_Vrth+PCatT_xvar4*xvar4_Vrth;
    xvar2=0.5*p[28]*rT/Vtv;
    xvar2_rT=0.5*p[28]/Vtv;
    xvar2_Vtv=-0.5*p[28]*rT/(Vtv*Vtv);
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar4=-0.5*p[28]*rT/Vtv;
    xvar4_rT=-0.5*p[28]/Vtv;
    xvar4_Vtv=0.5*p[28]*rT/(Vtv*Vtv);
    xvar4_Vrth=xvar4_rT*rT_Vrth;
    xvar4_Vrth=xvar4_Vrth+xvar4_Vtv*Vtv_Vrth;
    xvar5=exp(xvar4);
    xvar5_xvar4=xvar5;
    xvar5_Vrth=xvar5_xvar4*xvar4_Vrth;
    xvar1=xvar3-xvar5;
    xvar1_xvar3=1.0;
    xvar1_xvar5=-1.0;
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_xvar5*xvar5_Vrth;
    xvar6=log(xvar1);
    xvar6_xvar1=1.0/xvar1;
    xvar6_Vrth=xvar6_xvar1*xvar1_Vrth;
    psiio=2.0*(Vtv/rT)*xvar6;
    psiio_Vtv=2.0*xvar6/rT;
    psiio_rT=-2.0*Vtv*xvar6/(rT*rT);
    psiio_xvar6=2.0*Vtv/rT;
    psiio_Vrth=psiio_Vtv*Vtv_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_rT*rT_Vrth;
    psiio_Vrth=psiio_Vrth+psiio_xvar6*xvar6_Vrth;
    xvar1=log(rT);
    xvar1_rT=1.0/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    psiin=psiio*rT-3.0*Vtv*xvar1-p[74]*(rT-1.0);
    psiin_psiio=rT;
    psiin_rT=psiio-p[74];
    psiin_Vtv=-3.0*xvar1;
    psiin_xvar1=-3.0*Vtv;
    psiin_Vrth=psiin_psiio*psiio_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_rT*rT_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_Vtv*Vtv_Vrth;
    psiin_Vrth=psiin_Vrth+psiin_xvar1*xvar1_Vrth;
    xvar2=-psiin/Vtv;
    xvar2_psiin=-1.0/Vtv;
    xvar2_Vtv=psiin/(Vtv*Vtv);
    xvar2_Vrth=xvar2_psiin*psiin_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    xvar1=0.5*(1.0+sqrt(1.0+4.0*xvar3));
    xvar1_xvar3=1.0/sqrt(4.0*xvar3+1.0);
    xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
    xvar4=log(xvar1);
    xvar4_xvar1=1.0/xvar1;
    xvar4_Vrth=xvar4_xvar1*xvar1_Vrth;
    PSatT=psiin+2.0*Vtv*xvar4;
    PSatT_psiin=1.0;
    PSatT_Vtv=2.0*xvar4;
    PSatT_xvar4=2.0*Vtv;
    PSatT_Vrth=PSatT_psiin*psiin_Vrth;
    PSatT_Vrth=PSatT_Vrth+PSatT_Vtv*Vtv_Vrth;
    PSatT_Vrth=PSatT_Vrth+PSatT_xvar4*xvar4_Vrth;
    xvar1=p[17]/PEatT;
    xvar1_PEatT=-p[17]/(PEatT*PEatT);
    xvar1_Vrth=xvar1_PEatT*PEatT_Vrth;
    xvar2=pow(xvar1,p[18]);
    xvar2_xvar1=xvar2*p[18]/xvar1;
    xvar2_Vrth=xvar2_xvar1*xvar1_Vrth;
    CJEatT=p[16]*xvar2;
    CJEatT_xvar2=p[16];
    CJEatT_Vrth=CJEatT_xvar2*xvar2_Vrth;
    xvar1=p[24]/PCatT;
    xvar1_PCatT=-p[24]/(PCatT*PCatT);
    xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
    xvar2=pow(xvar1,p[25]);
    xvar2_xvar1=xvar2*p[25]/xvar1;
    xvar2_Vrth=xvar2_xvar1*xvar1_Vrth;
    CJCatT=p[21]*xvar2;
    CJCatT_xvar2=p[21];
    CJCatT_Vrth=CJCatT_xvar2*xvar2_Vrth;
    xvar1=p[24]/PCatT;
    xvar1_PCatT=-p[24]/(PCatT*PCatT);
    xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
    xvar2=pow(xvar1,p[25]);
    xvar2_xvar1=xvar2*p[25]/xvar1;
    xvar2_Vrth=xvar2_xvar1*xvar1_Vrth;
    CJEPatT=p[23]*xvar2;
    CJEPatT_xvar2=p[23];
    CJEPatT_Vrth=CJEPatT_xvar2*xvar2_Vrth;
    xvar1=p[28]/PSatT;
    xvar1_PSatT=-p[28]/(PSatT*PSatT);
    xvar1_Vrth=xvar1_PSatT*PSatT_Vrth;
    xvar2=pow(xvar1,p[29]);
    xvar2_xvar1=xvar2*p[29]/xvar1;
    xvar2_Vrth=xvar2_xvar1*xvar1_Vrth;
    CJCPatT=p[27]*xvar2;
    CJCPatT_xvar2=p[27];
    CJCPatT_Vrth=CJCPatT_xvar2*xvar2_Vrth;
    xvar1=pow(rT,p[78]);
    xvar1_rT=xvar1*p[78]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    xvar2=-p[71]*(1.0-rT)/Vtv;
    xvar2_rT=p[71]/Vtv;
    xvar2_Vtv=p[71]*(1.0-rT)/(Vtv*Vtv);
    xvar2_Vrth=xvar2_rT*rT_Vrth;
    xvar2_Vrth=xvar2_Vrth+xvar2_Vtv*Vtv_Vrth;
    xvar3=exp(xvar2);
    xvar3_xvar2=xvar3;
    xvar3_Vrth=xvar3_xvar2*xvar2_Vrth;
    GAMMatT=p[4]*xvar1*xvar3;
    GAMMatT_xvar1=p[4]*xvar3;
    GAMMatT_xvar3=p[4]*xvar1;
    GAMMatT_Vrth=GAMMatT_xvar1*xvar1_Vrth;
    GAMMatT_Vrth=GAMMatT_Vrth+GAMMatT_xvar3*xvar3_Vrth;
    xvar1=pow(rT,p[70]);
    xvar1_rT=xvar1*p[70]/rT;
    xvar1_Vrth=xvar1_rT*rT_Vrth;
    VOatT=p[3]*xvar1;
    VOatT_xvar1=p[3];
    VOatT_Vrth=VOatT_xvar1*xvar1_Vrth;
    xvar1=-VBBEatT/(NBBEatT*Vtv);
    xvar1_VBBEatT=-1.0/(NBBEatT*Vtv);
    xvar1_NBBEatT=VBBEatT/((NBBEatT*NBBEatT)*Vtv);
    xvar1_Vtv=VBBEatT/(NBBEatT*(Vtv*Vtv));
    xvar1_Vrth=xvar1_VBBEatT*VBBEatT_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_NBBEatT*NBBEatT_Vrth;
    xvar1_Vrth=xvar1_Vrth+xvar1_Vtv*Vtv_Vrth;
    EBBEatT=exp(xvar1);
    EBBEatT_xvar1=EBBEatT;
    EBBEatT_Vrth=EBBEatT_xvar1*xvar1_Vrth;
    if(p[51]>0.0){
        IVEF=1.0/p[51];
    }else{
        IVEF=0.0;
    }
    if(p[52]>0.0){
        IVER=1.0/p[52];
    }else{
        IVER=0.0;
    }
    if(p[53]>0.0){
        IIKF=1.0/IKFatT;
        IIKF_IKFatT=-1.0/(IKFatT*IKFatT);
        IIKF_Vrth=IIKF_IKFatT*IKFatT_Vrth;
    }else{
        IIKF=0.0;
        IIKF_Vrth=0.0;
    }
    if(p[54]>0.0){
        IIKR=1.0/p[54];
    }else{
        IIKR=0.0;
    }
    if(p[55]>0.0){
        IIKP=1.0/p[55];
    }else{
        IIKP=0.0;
    }
    if(p[3]>0.0){
        IVO=1.0/VOatT;
        IVO_VOatT=-1.0/(VOatT*VOatT);
        IVO_Vrth=IVO_VOatT*VOatT_Vrth;
    }else{
        IVO=0.0;
        IVO_Vrth=0.0;
    }
    if(p[5]>0.0){
        IHRCF=1.0/p[5];
    }else{
        IHRCF=0.0;
    }
    if(p[59]>0.0){
        IVTF=1.0/p[59];
    }else{
        IVTF=0.0;
    }
    if(p[60]>0.0){
        IITF=1.0/p[60];
    }else{
        IITF=0.0;
    }
    if(p[60]>0.0){
        slTF=0.0;
    }else{
        slTF=1.0;
    }
    dv0=-PEatT*p[14];
    dv0_PEatT=-p[14];
    dv0_Vrth=dv0_PEatT*PEatT_Vrth;
    if(p[19]<=0.0){
        dvh=(*Vbei)+dv0;
        dvh_Vbei=1.0;
        dvh_dv0=1.0;
        dvh_Vrth=dvh_dv0*dv0_Vrth;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[18]);
            pwq=pow(xvar1,xvar2);
            qlo=PEatT*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[18]);
            qlo_PEatT=(1.0-((1.0-p[14])*(1.0-p[14]))*pwq)/(1.0-p[18]);
            qlo_Vbei=0.0;
            qlo_Vrth=qlo_PEatT*PEatT_Vrth;
            qhi=dvh*(1.0-p[14]+0.5*p[18]*dvh/PEatT)*pwq;
            qhi_dvh=(0.5*dvh*p[18]/PEatT-p[14]+1.0)*pwq+0.5*dvh*p[18]*pwq/PEatT;
            qhi_PEatT=-0.5*(dvh*dvh)*p[18]*pwq/(PEatT*PEatT);
            qhi_Vbei=qhi_dvh*dvh_Vbei;
            qhi_Vrth=qhi_dvh*dvh_Vrth;
            qhi_Vrth=qhi_Vrth+qhi_PEatT*PEatT_Vrth;
        }else{
            xvar1=(1.0-(*Vbei)/PEatT);
            xvar1_Vbei=-1.0/PEatT;
            xvar1_PEatT=(*Vbei)/(PEatT*PEatT);
            xvar1_Vrth=xvar1_PEatT*PEatT_Vrth;
            xvar2=(1.0-p[18]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbei=xvar3_xvar1*xvar1_Vbei;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=PEatT*(1.0-xvar3)/(1.0-p[18]);
            qlo_PEatT=(1.0-xvar3)/(1.0-p[18]);
            qlo_xvar3=-PEatT/(1.0-p[18]);
            qlo_Vrth=qlo_PEatT*PEatT_Vrth;
            qlo_Vbei=qlo_xvar3*xvar3_Vbei;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            qhi=0.0;
            qhi_Vbei=0.0;
            qhi_Vrth=0.0;
        }
        qdbe=qlo+qhi;
        qdbe_qlo=1.0;
        qdbe_qhi=1.0;
        qdbe_Vrth=qdbe_qlo*qlo_Vrth;
        qdbe_Vbei=qdbe_qlo*qlo_Vbei;
        qdbe_Vbei=qdbe_Vbei+qdbe_qhi*qhi_Vbei;
        qdbe_Vrth=qdbe_Vrth+qdbe_qhi*qhi_Vrth;
    }else{
        mv0=sqrt(dv0*dv0+4.0*p[19]*p[19]);
        mv0_dv0=dv0/sqrt((dv0*dv0)+4.0*(p[19]*p[19]));
        mv0_Vrth=mv0_dv0*dv0_Vrth;
        vl0=-0.5*(dv0+mv0);
        vl0_dv0=-0.5;
        vl0_mv0=-0.5;
        vl0_Vrth=vl0_dv0*dv0_Vrth;
        vl0_Vrth=vl0_Vrth+vl0_mv0*mv0_Vrth;
        xvar1=(1.0-vl0/PEatT);
        xvar1_vl0=-1.0/PEatT;
        xvar1_PEatT=vl0/(PEatT*PEatT);
        xvar1_Vrth=xvar1_vl0*vl0_Vrth;
        xvar1_Vrth=xvar1_Vrth+xvar1_PEatT*PEatT_Vrth;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
        q0=-PEatT*xvar3/(1.0-p[18]);
        q0_PEatT=-xvar3/(1.0-p[18]);
        q0_xvar3=-PEatT/(1.0-p[18]);
        q0_Vrth=q0_PEatT*PEatT_Vrth;
        q0_Vrth=q0_Vrth+q0_xvar3*xvar3_Vrth;
        dv=(*Vbei)+dv0;
        dv_Vbei=1.0;
        dv_dv0=1.0;
        dv_Vrth=dv_dv0*dv0_Vrth;
        mv=sqrt(dv*dv+4.0*p[19]*p[19]);
        mv_dv=dv/sqrt((dv*dv)+4.0*(p[19]*p[19]));
        mv_Vbei=mv_dv*dv_Vbei;
        mv_Vrth=mv_dv*dv_Vrth;
        vl=0.5*(dv-mv)-dv0;
        vl_dv=0.5;
        vl_mv=-0.5;
        vl_dv0=-1.0;
        vl_Vbei=vl_dv*dv_Vbei;
        vl_Vrth=vl_dv*dv_Vrth;
        vl_Vbei=vl_Vbei+vl_mv*mv_Vbei;
        vl_Vrth=vl_Vrth+vl_mv*mv_Vrth;
        vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
        xvar1=(1.0-vl/PEatT);
        xvar1_vl=-1.0/PEatT;
        xvar1_PEatT=vl/(PEatT*PEatT);
        xvar1_Vbei=xvar1_vl*vl_Vbei;
        xvar1_Vrth=xvar1_vl*vl_Vrth;
        xvar1_Vrth=xvar1_Vrth+xvar1_PEatT*PEatT_Vrth;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vbei=xvar3_xvar1*xvar1_Vbei;
        xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
        qlo=-PEatT*xvar3/(1.0-p[18]);
        qlo_PEatT=-xvar3/(1.0-p[18]);
        qlo_xvar3=-PEatT/(1.0-p[18]);
        qlo_Vrth=qlo_PEatT*PEatT_Vrth;
        qlo_Vbei=qlo_xvar3*xvar3_Vbei;
        qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
        xvar1=(1.0-p[14]);
        xvar2=(-p[18]);
        xvar3=pow(xvar1,xvar2);
        qdbe=qlo+xvar3*((*Vbei)-vl+vl0)-q0;
        qdbe_qlo=1.0;
        qdbe_Vbei=xvar3;
        qdbe_vl=-xvar3;
        qdbe_vl0=xvar3;
        qdbe_q0=-1.0;
        qdbe_Vrth=qdbe_qlo*qlo_Vrth;
        qdbe_Vbei=qdbe_Vbei+qdbe_qlo*qlo_Vbei;
        qdbe_Vbei=qdbe_Vbei+qdbe_vl*vl_Vbei;
        qdbe_Vrth=qdbe_Vrth+qdbe_vl*vl_Vrth;
        qdbe_Vrth=qdbe_Vrth+qdbe_vl0*vl0_Vrth;
        qdbe_Vrth=qdbe_Vrth+qdbe_q0*q0_Vrth;
    }
    dv0=-PEatT*p[14];
    dv0_PEatT=-p[14];
    dv0_Vrth=dv0_PEatT*PEatT_Vrth;
    if(p[19]<=0.0){
        dvh=(*Vbex)+dv0;
        dvh_Vbex=1.0;
        dvh_dv0=1.0;
        dvh_Vrth=dvh_dv0*dv0_Vrth;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[18]);
            pwq=pow(xvar1,xvar2);
            qlo=PEatT*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[18]);
            qlo_PEatT=(1.0-((1.0-p[14])*(1.0-p[14]))*pwq)/(1.0-p[18]);
            qlo_Vbex=0.0;
            qlo_Vrth=qlo_PEatT*PEatT_Vrth;
            qhi=dvh*(1.0-p[14]+0.5*p[18]*dvh/PEatT)*pwq;
            qhi_dvh=(0.5*dvh*p[18]/PEatT-p[14]+1.0)*pwq+0.5*dvh*p[18]*pwq/PEatT;
            qhi_PEatT=-0.5*(dvh*dvh)*p[18]*pwq/(PEatT*PEatT);
            qhi_Vbex=qhi_dvh*dvh_Vbex;
            qhi_Vrth=qhi_dvh*dvh_Vrth;
            qhi_Vrth=qhi_Vrth+qhi_PEatT*PEatT_Vrth;
        }else{
            xvar1=(1.0-(*Vbex)/PEatT);
            xvar1_Vbex=-1.0/PEatT;
            xvar1_PEatT=(*Vbex)/(PEatT*PEatT);
            xvar1_Vrth=xvar1_PEatT*PEatT_Vrth;
            xvar2=(1.0-p[18]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbex=xvar3_xvar1*xvar1_Vbex;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=PEatT*(1.0-xvar3)/(1.0-p[18]);
            qlo_PEatT=(1.0-xvar3)/(1.0-p[18]);
            qlo_xvar3=-PEatT/(1.0-p[18]);
            qlo_Vrth=qlo_PEatT*PEatT_Vrth;
            qlo_Vbex=qlo_xvar3*xvar3_Vbex;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            qhi=0.0;
            qhi_Vbex=0.0;
            qhi_Vrth=0.0;
        }
        qdbex=qlo+qhi;
        qdbex_qlo=1.0;
        qdbex_qhi=1.0;
        qdbex_Vrth=qdbex_qlo*qlo_Vrth;
        qdbex_Vbex=qdbex_qlo*qlo_Vbex;
        qdbex_Vbex=qdbex_Vbex+qdbex_qhi*qhi_Vbex;
        qdbex_Vrth=qdbex_Vrth+qdbex_qhi*qhi_Vrth;
    }else{
        mv0=sqrt(dv0*dv0+4.0*p[19]*p[19]);
        mv0_dv0=dv0/sqrt((dv0*dv0)+4.0*(p[19]*p[19]));
        mv0_Vrth=mv0_dv0*dv0_Vrth;
        vl0=-0.5*(dv0+mv0);
        vl0_dv0=-0.5;
        vl0_mv0=-0.5;
        vl0_Vrth=vl0_dv0*dv0_Vrth;
        vl0_Vrth=vl0_Vrth+vl0_mv0*mv0_Vrth;
        xvar1=(1.0-vl0/PEatT);
        xvar1_vl0=-1.0/PEatT;
        xvar1_PEatT=vl0/(PEatT*PEatT);
        xvar1_Vrth=xvar1_vl0*vl0_Vrth;
        xvar1_Vrth=xvar1_Vrth+xvar1_PEatT*PEatT_Vrth;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
        q0=-PEatT*xvar3/(1.0-p[18]);
        q0_PEatT=-xvar3/(1.0-p[18]);
        q0_xvar3=-PEatT/(1.0-p[18]);
        q0_Vrth=q0_PEatT*PEatT_Vrth;
        q0_Vrth=q0_Vrth+q0_xvar3*xvar3_Vrth;
        dv=(*Vbex)+dv0;
        dv_Vbex=1.0;
        dv_dv0=1.0;
        dv_Vrth=dv_dv0*dv0_Vrth;
        mv=sqrt(dv*dv+4.0*p[19]*p[19]);
        mv_dv=dv/sqrt((dv*dv)+4.0*(p[19]*p[19]));
        mv_Vbex=mv_dv*dv_Vbex;
        mv_Vrth=mv_dv*dv_Vrth;
        vl=0.5*(dv-mv)-dv0;
        vl_dv=0.5;
        vl_mv=-0.5;
        vl_dv0=-1.0;
        vl_Vbex=vl_dv*dv_Vbex;
        vl_Vrth=vl_dv*dv_Vrth;
        vl_Vbex=vl_Vbex+vl_mv*mv_Vbex;
        vl_Vrth=vl_Vrth+vl_mv*mv_Vrth;
        vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
        xvar1=(1.0-vl/PEatT);
        xvar1_vl=-1.0/PEatT;
        xvar1_PEatT=vl/(PEatT*PEatT);
        xvar1_Vbex=xvar1_vl*vl_Vbex;
        xvar1_Vrth=xvar1_vl*vl_Vrth;
        xvar1_Vrth=xvar1_Vrth+xvar1_PEatT*PEatT_Vrth;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vbex=xvar3_xvar1*xvar1_Vbex;
        xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
        qlo=-PEatT*xvar3/(1.0-p[18]);
        qlo_PEatT=-xvar3/(1.0-p[18]);
        qlo_xvar3=-PEatT/(1.0-p[18]);
        qlo_Vrth=qlo_PEatT*PEatT_Vrth;
        qlo_Vbex=qlo_xvar3*xvar3_Vbex;
        qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
        xvar1=(1.0-p[14]);
        xvar2=(-p[18]);
        xvar3=pow(xvar1,xvar2);
        qdbex=qlo+xvar3*((*Vbex)-vl+vl0)-q0;
        qdbex_qlo=1.0;
        qdbex_Vbex=xvar3;
        qdbex_vl=-xvar3;
        qdbex_vl0=xvar3;
        qdbex_q0=-1.0;
        qdbex_Vrth=qdbex_qlo*qlo_Vrth;
        qdbex_Vbex=qdbex_Vbex+qdbex_qlo*qlo_Vbex;
        qdbex_Vbex=qdbex_Vbex+qdbex_vl*vl_Vbex;
        qdbex_Vrth=qdbex_Vrth+qdbex_vl*vl_Vrth;
        qdbex_Vrth=qdbex_Vrth+qdbex_vl0*vl0_Vrth;
        qdbex_Vrth=qdbex_Vrth+qdbex_q0*q0_Vrth;
    }
    dv0=-PCatT*p[14];
    dv0_PCatT=-p[14];
    dv0_Vrth=dv0_PCatT*PCatT_Vrth;
    if(p[26]<=0.0){
        dvh=(*Vbci)+dv0;
        dvh_Vbci=1.0;
        dvh_dv0=1.0;
        dvh_Vrth=dvh_dv0*dv0_Vrth;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[25]);
            pwq=pow(xvar1,xvar2);
            qlo=PCatT*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[25]);
            qlo_PCatT=(1.0-((1.0-p[14])*(1.0-p[14]))*pwq)/(1.0-p[25]);
            qlo_Vbci=0.0;
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qhi=dvh*(1.0-p[14]+0.5*p[25]*dvh/PCatT)*pwq;
            qhi_dvh=(0.5*dvh*p[25]/PCatT-p[14]+1.0)*pwq+0.5*dvh*p[25]*pwq/PCatT;
            qhi_PCatT=-0.5*(dvh*dvh)*p[25]*pwq/(PCatT*PCatT);
            qhi_Vbci=qhi_dvh*dvh_Vbci;
            qhi_Vrth=qhi_dvh*dvh_Vrth;
            qhi_Vrth=qhi_Vrth+qhi_PCatT*PCatT_Vrth;
        }else{
            if((p[85]>0.0)&&((*Vbci)<-p[85])){
                xvar1=(1.0+p[85]/PCatT);
                xvar1_PCatT=-p[85]/(PCatT*PCatT);
                xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
                qlo=PCatT*(1.0-xvar3*(1.0-((1.0-p[25])*((*Vbci)+p[85]))/(PCatT+p[85])))/(1.0-p[25]);
                qlo_PCatT=(1.0-(1.0-(1.0-p[25])*(p[85]+(*Vbci))/(p[85]+PCatT))*xvar3)/(1.0-p[25])-PCatT*(p[85]+(*Vbci))*xvar3/((p[85]+PCatT)*(p[85]+PCatT));
                qlo_xvar3=PCatT*((1.0-p[25])*(p[85]+(*Vbci))/(p[85]+PCatT)-1.0)/(1.0-p[25]);
                qlo_Vbci=PCatT*xvar3/(p[85]+PCatT);
                qlo_Vrth=qlo_PCatT*PCatT_Vrth;
                qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            }else{
                xvar1=(1.0-(*Vbci)/PCatT);
                xvar1_Vbci=-1.0/PCatT;
                xvar1_PCatT=(*Vbci)/(PCatT*PCatT);
                xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
                xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
                qlo=PCatT*(1.0-xvar3)/(1.0-p[25]);
                qlo_PCatT=(1.0-xvar3)/(1.0-p[25]);
                qlo_xvar3=-PCatT/(1.0-p[25]);
                qlo_Vrth=qlo_PCatT*PCatT_Vrth;
                qlo_Vbci=qlo_xvar3*xvar3_Vbci;
                qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            }
            qhi=0.0;
            qhi_Vbci=0.0;
            qhi_Vrth=0.0;
        }
        qdbc=qlo+qhi;
        qdbc_qlo=1.0;
        qdbc_qhi=1.0;
        qdbc_Vrth=qdbc_qlo*qlo_Vrth;
        qdbc_Vbci=qdbc_qlo*qlo_Vbci;
        qdbc_Vbci=qdbc_Vbci+qdbc_qhi*qhi_Vbci;
        qdbc_Vrth=qdbc_Vrth+qdbc_qhi*qhi_Vrth;
    }else{
        if((p[85]>0.0)&&(p[86]>0.0)){
            vn0=(p[85]+dv0)/(p[85]-dv0);
            vn0_dv0=(p[85]+dv0)/((p[85]-dv0)*(p[85]-dv0))+1.0/(p[85]-dv0);
            vn0_Vrth=vn0_dv0*dv0_Vrth;
            vnl0=2.0*vn0/(sqrt((vn0-1.0)*(vn0-1.0)+4.0*p[26]*p[26])+sqrt((vn0+1.0)*(vn0+1.0)+4.0*p[86]*p[86]));
            vnl0_vn0=2.0/(sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))-2.0*vn0*((vn0+1.0)/sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+(vn0-1.0)/sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26]))));
            vnl0_Vrth=vnl0_vn0*vn0_Vrth;
            vl0=0.5*(vnl0*(p[85]-dv0)-p[85]-dv0);
            vl0_vnl0=0.5*(p[85]-dv0);
            vl0_dv0=0.5*(-vnl0-1.0);
            vl0_Vrth=vl0_vnl0*vnl0_Vrth;
            vl0_Vrth=vl0_Vrth+vl0_dv0*dv0_Vrth;
            xvar1=(1.0-vl0/PCatT);
            xvar1_vl0=-1.0/PCatT;
            xvar1_PCatT=vl0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_vl0*vl0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo0=PCatT*(1.0-xvar3)/(1.0-p[25]);
            qlo0_PCatT=(1.0-xvar3)/(1.0-p[25]);
            qlo0_xvar3=-PCatT/(1.0-p[25]);
            qlo0_Vrth=qlo0_PCatT*PCatT_Vrth;
            qlo0_Vrth=qlo0_Vrth+qlo0_xvar3*xvar3_Vrth;
            vn=(2.0*(*Vbci)+p[85]+dv0)/(p[85]-dv0);
            vn_Vbci=2.0/(p[85]-dv0);
            vn_dv0=(p[85]+2.0*(*Vbci)+dv0)/((p[85]-dv0)*(p[85]-dv0))+1.0/(p[85]-dv0);
            vn_Vrth=vn_dv0*dv0_Vrth;
            vnl=2.0*vn/(sqrt((vn-1.0)*(vn-1.0)+4.0*p[26]*p[26])+sqrt((vn+1.0)*(vn+1.0)+4.0*p[86]*p[86]));
            vnl_vn=2.0/(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))-2.0*vn*((vn+1.0)/sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+(vn-1.0)/sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26]))));
            vnl_Vbci=vnl_vn*vn_Vbci;
            vnl_Vrth=vnl_vn*vn_Vrth;
            vl=0.5*(vnl*(p[85]-dv0)-p[85]-dv0);
            vl_vnl=0.5*(p[85]-dv0);
            vl_dv0=0.5*(-vnl-1.0);
            vl_Vbci=vl_vnl*vnl_Vbci;
            vl_Vrth=vl_vnl*vnl_Vrth;
            vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
            xvar1=(1.0-vl/PCatT);
            xvar1_vl=-1.0/PCatT;
            xvar1_PCatT=vl/(PCatT*PCatT);
            xvar1_Vbci=xvar1_vl*vl_Vbci;
            xvar1_Vrth=xvar1_vl*vl_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=PCatT*(1.0-xvar3)/(1.0-p[25]);
            qlo_PCatT=(1.0-xvar3)/(1.0-p[25]);
            qlo_xvar3=-PCatT/(1.0-p[25]);
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qlo_Vbci=qlo_xvar3*xvar3_Vbci;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            sel=0.5*(vnl+1.0);
            sel_vnl=0.5;
            sel_Vbci=sel_vnl*vnl_Vbci;
            sel_Vrth=sel_vnl*vnl_Vrth;
            xvar1=(1.0+p[85]/PCatT);
            xvar1_PCatT=-p[85]/(PCatT*PCatT);
            xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
            xvar2=(-p[25]);
            crt=pow(xvar1,xvar2);
            crt_xvar1=crt*xvar2/xvar1;
            crt_Vrth=crt_xvar1*xvar1_Vrth;
            xvar1=(1.0+dv0/PCatT);
            xvar1_dv0=1.0/PCatT;
            xvar1_PCatT=-dv0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_dv0*dv0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(-p[25]);
            cmx=pow(xvar1,xvar2);
            cmx_xvar1=cmx*xvar2/xvar1;
            cmx_Vrth=cmx_xvar1*xvar1_Vrth;
            cl=(1.0-sel)*crt+sel*cmx;
            cl_sel=cmx-crt;
            cl_crt=1.0-sel;
            cl_cmx=sel;
            cl_Vbci=cl_sel*sel_Vbci;
            cl_Vrth=cl_sel*sel_Vrth;
            cl_Vrth=cl_Vrth+cl_crt*crt_Vrth;
            cl_Vrth=cl_Vrth+cl_cmx*cmx_Vrth;
            ql=((*Vbci)-vl+vl0)*cl;
            ql_Vbci=cl;
            ql_vl=-cl;
            ql_vl0=cl;
            ql_cl=vl0-vl+(*Vbci);
            ql_Vbci=ql_Vbci+ql_vl*vl_Vbci;
            ql_Vrth=ql_vl*vl_Vrth;
            ql_Vrth=ql_Vrth+ql_vl0*vl0_Vrth;
            ql_Vbci=ql_Vbci+ql_cl*cl_Vbci;
            ql_Vrth=ql_Vrth+ql_cl*cl_Vrth;
            qdbc=ql+qlo-qlo0;
            qdbc_ql=1.0;
            qdbc_qlo=1.0;
            qdbc_qlo0=-1.0;
            qdbc_Vbci=qdbc_ql*ql_Vbci;
            qdbc_Vrth=qdbc_ql*ql_Vrth;
            qdbc_Vrth=qdbc_Vrth+qdbc_qlo*qlo_Vrth;
            qdbc_Vbci=qdbc_Vbci+qdbc_qlo*qlo_Vbci;
            qdbc_Vrth=qdbc_Vrth+qdbc_qlo0*qlo0_Vrth;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[26]*p[26]);
            mv0_dv0=dv0/sqrt((dv0*dv0)+4.0*(p[26]*p[26]));
            mv0_Vrth=mv0_dv0*dv0_Vrth;
            vl0=-0.5*(dv0+mv0);
            vl0_dv0=-0.5;
            vl0_mv0=-0.5;
            vl0_Vrth=vl0_dv0*dv0_Vrth;
            vl0_Vrth=vl0_Vrth+vl0_mv0*mv0_Vrth;
            xvar1=(1.0-vl0/PCatT);
            xvar1_vl0=-1.0/PCatT;
            xvar1_PCatT=vl0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_vl0*vl0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            q0=-PCatT*xvar3/(1.0-p[25]);
            q0_PCatT=-xvar3/(1.0-p[25]);
            q0_xvar3=-PCatT/(1.0-p[25]);
            q0_Vrth=q0_PCatT*PCatT_Vrth;
            q0_Vrth=q0_Vrth+q0_xvar3*xvar3_Vrth;
            dv=(*Vbci)+dv0;
            dv_Vbci=1.0;
            dv_dv0=1.0;
            dv_Vrth=dv_dv0*dv0_Vrth;
            mv=sqrt(dv*dv+4.0*p[26]*p[26]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[26]*p[26]));
            mv_Vbci=mv_dv*dv_Vbci;
            mv_Vrth=mv_dv*dv_Vrth;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_dv0=-1.0;
            vl_Vbci=vl_dv*dv_Vbci;
            vl_Vrth=vl_dv*dv_Vrth;
            vl_Vbci=vl_Vbci+vl_mv*mv_Vbci;
            vl_Vrth=vl_Vrth+vl_mv*mv_Vrth;
            vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
            xvar1=(1.0-vl/PCatT);
            xvar1_vl=-1.0/PCatT;
            xvar1_PCatT=vl/(PCatT*PCatT);
            xvar1_Vbci=xvar1_vl*vl_Vbci;
            xvar1_Vrth=xvar1_vl*vl_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=-PCatT*xvar3/(1.0-p[25]);
            qlo_PCatT=-xvar3/(1.0-p[25]);
            qlo_xvar3=-PCatT/(1.0-p[25]);
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qlo_Vbci=qlo_xvar3*xvar3_Vbci;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            xvar1=(1.0-p[14]);
            xvar2=(-p[25]);
            xvar3=pow(xvar1,xvar2);
            qdbc=qlo+xvar3*((*Vbci)-vl+vl0)-q0;
            qdbc_qlo=1.0;
            qdbc_Vbci=xvar3;
            qdbc_vl=-xvar3;
            qdbc_vl0=xvar3;
            qdbc_q0=-1.0;
            qdbc_Vrth=qdbc_qlo*qlo_Vrth;
            qdbc_Vbci=qdbc_Vbci+qdbc_qlo*qlo_Vbci;
            qdbc_Vbci=qdbc_Vbci+qdbc_vl*vl_Vbci;
            qdbc_Vrth=qdbc_Vrth+qdbc_vl*vl_Vrth;
            qdbc_Vrth=qdbc_Vrth+qdbc_vl0*vl0_Vrth;
            qdbc_Vrth=qdbc_Vrth+qdbc_q0*q0_Vrth;
        }
    }
    dv0=-PCatT*p[14];
    dv0_PCatT=-p[14];
    dv0_Vrth=dv0_PCatT*PCatT_Vrth;
    if(p[26]<=0.0){
        dvh=(*Vbep)+dv0;
        dvh_Vbep=1.0;
        dvh_dv0=1.0;
        dvh_Vrth=dvh_dv0*dv0_Vrth;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[25]);
            pwq=pow(xvar1,xvar2);
            qlo=PCatT*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[25]);
            qlo_PCatT=(1.0-((1.0-p[14])*(1.0-p[14]))*pwq)/(1.0-p[25]);
            qlo_Vbep=0.0;
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qhi=dvh*(1.0-p[14]+0.5*p[25]*dvh/PCatT)*pwq;
            qhi_dvh=(0.5*dvh*p[25]/PCatT-p[14]+1.0)*pwq+0.5*dvh*p[25]*pwq/PCatT;
            qhi_PCatT=-0.5*(dvh*dvh)*p[25]*pwq/(PCatT*PCatT);
            qhi_Vbep=qhi_dvh*dvh_Vbep;
            qhi_Vrth=qhi_dvh*dvh_Vrth;
            qhi_Vrth=qhi_Vrth+qhi_PCatT*PCatT_Vrth;
        }else{
            if((p[85]>0.0)&&((*Vbep)<-p[85])){
                xvar1=(1.0+p[85]/PCatT);
                xvar1_PCatT=-p[85]/(PCatT*PCatT);
                xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
                qlo=PCatT*(1.0-xvar3*(1.0-((1.0-p[25])*((*Vbep)+p[85]))/(PCatT+p[85])))/(1.0-p[25]);
                qlo_PCatT=(1.0-(1.0-(1.0-p[25])*(p[85]+(*Vbep))/(p[85]+PCatT))*xvar3)/(1.0-p[25])-PCatT*(p[85]+(*Vbep))*xvar3/((p[85]+PCatT)*(p[85]+PCatT));
                qlo_xvar3=PCatT*((1.0-p[25])*(p[85]+(*Vbep))/(p[85]+PCatT)-1.0)/(1.0-p[25]);
                qlo_Vbep=PCatT*xvar3/(p[85]+PCatT);
                qlo_Vrth=qlo_PCatT*PCatT_Vrth;
                qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            }else{
                xvar1=(1.0-(*Vbep)/PCatT);
                xvar1_Vbep=-1.0/PCatT;
                xvar1_PCatT=(*Vbep)/(PCatT*PCatT);
                xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
                xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
                qlo=PCatT*(1.0-xvar3)/(1.0-p[25]);
                qlo_PCatT=(1.0-xvar3)/(1.0-p[25]);
                qlo_xvar3=-PCatT/(1.0-p[25]);
                qlo_Vrth=qlo_PCatT*PCatT_Vrth;
                qlo_Vbep=qlo_xvar3*xvar3_Vbep;
                qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            }
            qhi=0.0;
            qhi_Vbep=0.0;
            qhi_Vrth=0.0;
        }
        qdbep=qlo+qhi;
        qdbep_qlo=1.0;
        qdbep_qhi=1.0;
        qdbep_Vrth=qdbep_qlo*qlo_Vrth;
        qdbep_Vbep=qdbep_qlo*qlo_Vbep;
        qdbep_Vbep=qdbep_Vbep+qdbep_qhi*qhi_Vbep;
        qdbep_Vrth=qdbep_Vrth+qdbep_qhi*qhi_Vrth;
    }else{
        if((p[85]>0.0)&&(p[86]>0.0)){
            vn0=(p[85]+dv0)/(p[85]-dv0);
            vn0_dv0=(p[85]+dv0)/((p[85]-dv0)*(p[85]-dv0))+1.0/(p[85]-dv0);
            vn0_Vrth=vn0_dv0*dv0_Vrth;
            vnl0=2.0*vn0/(sqrt((vn0-1.0)*(vn0-1.0)+4.0*p[26]*p[26])+sqrt((vn0+1.0)*(vn0+1.0)+4.0*p[86]*p[86]));
            vnl0_vn0=2.0/(sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))-2.0*vn0*((vn0+1.0)/sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+(vn0-1.0)/sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn0+1.0)*(vn0+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn0-1.0)*(vn0-1.0))+4.0*(p[26]*p[26]))));
            vnl0_Vrth=vnl0_vn0*vn0_Vrth;
            vl0=0.5*(vnl0*(p[85]-dv0)-p[85]-dv0);
            vl0_vnl0=0.5*(p[85]-dv0);
            vl0_dv0=0.5*(-vnl0-1.0);
            vl0_Vrth=vl0_vnl0*vnl0_Vrth;
            vl0_Vrth=vl0_Vrth+vl0_dv0*dv0_Vrth;
            xvar1=(1.0-vl0/PCatT);
            xvar1_vl0=-1.0/PCatT;
            xvar1_PCatT=vl0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_vl0*vl0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo0=PCatT*(1.0-xvar3)/(1.0-p[25]);
            qlo0_PCatT=(1.0-xvar3)/(1.0-p[25]);
            qlo0_xvar3=-PCatT/(1.0-p[25]);
            qlo0_Vrth=qlo0_PCatT*PCatT_Vrth;
            qlo0_Vrth=qlo0_Vrth+qlo0_xvar3*xvar3_Vrth;
            vn=(2.0*(*Vbep)+p[85]+dv0)/(p[85]-dv0);
            vn_Vbep=2.0/(p[85]-dv0);
            vn_dv0=(p[85]+2.0*(*Vbep)+dv0)/((p[85]-dv0)*(p[85]-dv0))+1.0/(p[85]-dv0);
            vn_Vrth=vn_dv0*dv0_Vrth;
            vnl=2.0*vn/(sqrt((vn-1.0)*(vn-1.0)+4.0*p[26]*p[26])+sqrt((vn+1.0)*(vn+1.0)+4.0*p[86]*p[86]));
            vnl_vn=2.0/(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))-2.0*vn*((vn+1.0)/sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+(vn-1.0)/sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26]))));
            vnl_Vbep=vnl_vn*vn_Vbep;
            vnl_Vrth=vnl_vn*vn_Vrth;
            vl=0.5*(vnl*(p[85]-dv0)-p[85]-dv0);
            vl_vnl=0.5*(p[85]-dv0);
            vl_dv0=0.5*(-vnl-1.0);
            vl_Vbep=vl_vnl*vnl_Vbep;
            vl_Vrth=vl_vnl*vnl_Vrth;
            vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
            xvar1=(1.0-vl/PCatT);
            xvar1_vl=-1.0/PCatT;
            xvar1_PCatT=vl/(PCatT*PCatT);
            xvar1_Vbep=xvar1_vl*vl_Vbep;
            xvar1_Vrth=xvar1_vl*vl_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=PCatT*(1.0-xvar3)/(1.0-p[25]);
            qlo_PCatT=(1.0-xvar3)/(1.0-p[25]);
            qlo_xvar3=-PCatT/(1.0-p[25]);
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qlo_Vbep=qlo_xvar3*xvar3_Vbep;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            sel=0.5*(vnl+1.0);
            sel_vnl=0.5;
            sel_Vbep=sel_vnl*vnl_Vbep;
            sel_Vrth=sel_vnl*vnl_Vrth;
            xvar1=(1.0+p[85]/PCatT);
            xvar1_PCatT=-p[85]/(PCatT*PCatT);
            xvar1_Vrth=xvar1_PCatT*PCatT_Vrth;
            xvar2=(-p[25]);
            crt=pow(xvar1,xvar2);
            crt_xvar1=crt*xvar2/xvar1;
            crt_Vrth=crt_xvar1*xvar1_Vrth;
            xvar1=(1.0+dv0/PCatT);
            xvar1_dv0=1.0/PCatT;
            xvar1_PCatT=-dv0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_dv0*dv0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(-p[25]);
            cmx=pow(xvar1,xvar2);
            cmx_xvar1=cmx*xvar2/xvar1;
            cmx_Vrth=cmx_xvar1*xvar1_Vrth;
            cl=(1.0-sel)*crt+sel*cmx;
            cl_sel=cmx-crt;
            cl_crt=1.0-sel;
            cl_cmx=sel;
            cl_Vbep=cl_sel*sel_Vbep;
            cl_Vrth=cl_sel*sel_Vrth;
            cl_Vrth=cl_Vrth+cl_crt*crt_Vrth;
            cl_Vrth=cl_Vrth+cl_cmx*cmx_Vrth;
            ql=((*Vbep)-vl+vl0)*cl;
            ql_Vbep=cl;
            ql_vl=-cl;
            ql_vl0=cl;
            ql_cl=vl0-vl+(*Vbep);
            ql_Vbep=ql_Vbep+ql_vl*vl_Vbep;
            ql_Vrth=ql_vl*vl_Vrth;
            ql_Vrth=ql_Vrth+ql_vl0*vl0_Vrth;
            ql_Vbep=ql_Vbep+ql_cl*cl_Vbep;
            ql_Vrth=ql_Vrth+ql_cl*cl_Vrth;
            qdbep=ql+qlo-qlo0;
            qdbep_ql=1.0;
            qdbep_qlo=1.0;
            qdbep_qlo0=-1.0;
            qdbep_Vbep=qdbep_ql*ql_Vbep;
            qdbep_Vrth=qdbep_ql*ql_Vrth;
            qdbep_Vrth=qdbep_Vrth+qdbep_qlo*qlo_Vrth;
            qdbep_Vbep=qdbep_Vbep+qdbep_qlo*qlo_Vbep;
            qdbep_Vrth=qdbep_Vrth+qdbep_qlo0*qlo0_Vrth;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[26]*p[26]);
            mv0_dv0=dv0/sqrt((dv0*dv0)+4.0*(p[26]*p[26]));
            mv0_Vrth=mv0_dv0*dv0_Vrth;
            vl0=-0.5*(dv0+mv0);
            vl0_dv0=-0.5;
            vl0_mv0=-0.5;
            vl0_Vrth=vl0_dv0*dv0_Vrth;
            vl0_Vrth=vl0_Vrth+vl0_mv0*mv0_Vrth;
            xvar1=(1.0-vl0/PCatT);
            xvar1_vl0=-1.0/PCatT;
            xvar1_PCatT=vl0/(PCatT*PCatT);
            xvar1_Vrth=xvar1_vl0*vl0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            q0=-PCatT*xvar3/(1.0-p[25]);
            q0_PCatT=-xvar3/(1.0-p[25]);
            q0_xvar3=-PCatT/(1.0-p[25]);
            q0_Vrth=q0_PCatT*PCatT_Vrth;
            q0_Vrth=q0_Vrth+q0_xvar3*xvar3_Vrth;
            dv=(*Vbep)+dv0;
            dv_Vbep=1.0;
            dv_dv0=1.0;
            dv_Vrth=dv_dv0*dv0_Vrth;
            mv=sqrt(dv*dv+4.0*p[26]*p[26]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[26]*p[26]));
            mv_Vbep=mv_dv*dv_Vbep;
            mv_Vrth=mv_dv*dv_Vrth;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_dv0=-1.0;
            vl_Vbep=vl_dv*dv_Vbep;
            vl_Vrth=vl_dv*dv_Vrth;
            vl_Vbep=vl_Vbep+vl_mv*mv_Vbep;
            vl_Vrth=vl_Vrth+vl_mv*mv_Vrth;
            vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
            xvar1=(1.0-vl/PCatT);
            xvar1_vl=-1.0/PCatT;
            xvar1_PCatT=vl/(PCatT*PCatT);
            xvar1_Vbep=xvar1_vl*vl_Vbep;
            xvar1_Vrth=xvar1_vl*vl_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PCatT*PCatT_Vrth;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=-PCatT*xvar3/(1.0-p[25]);
            qlo_PCatT=-xvar3/(1.0-p[25]);
            qlo_xvar3=-PCatT/(1.0-p[25]);
            qlo_Vrth=qlo_PCatT*PCatT_Vrth;
            qlo_Vbep=qlo_xvar3*xvar3_Vbep;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            xvar1=(1.0-p[14]);
            xvar2=(-p[25]);
            xvar3=pow(xvar1,xvar2);
            qdbep=qlo+xvar3*((*Vbep)-vl+vl0)-q0;
            qdbep_qlo=1.0;
            qdbep_Vbep=xvar3;
            qdbep_vl=-xvar3;
            qdbep_vl0=xvar3;
            qdbep_q0=-1.0;
            qdbep_Vrth=qdbep_qlo*qlo_Vrth;
            qdbep_Vbep=qdbep_Vbep+qdbep_qlo*qlo_Vbep;
            qdbep_Vbep=qdbep_Vbep+qdbep_vl*vl_Vbep;
            qdbep_Vrth=qdbep_Vrth+qdbep_vl*vl_Vrth;
            qdbep_Vrth=qdbep_Vrth+qdbep_vl0*vl0_Vrth;
            qdbep_Vrth=qdbep_Vrth+qdbep_q0*q0_Vrth;
        }
    }
    if(p[27]>0.0){
        dv0=-PSatT*p[14];
        dv0_PSatT=-p[14];
        dv0_Vrth=dv0_PSatT*PSatT_Vrth;
        if(p[30]<=0.0){
            dvh=(*Vbcp)+dv0;
            dvh_Vbcp=1.0;
            dvh_dv0=1.0;
            dvh_Vrth=dvh_dv0*dv0_Vrth;
            if(dvh>0.0){
                xvar1=(1.0-p[14]);
                xvar2=(-1.0-p[29]);
                pwq=pow(xvar1,xvar2);
                qlo=PSatT*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[29]);
                qlo_PSatT=(1.0-((1.0-p[14])*(1.0-p[14]))*pwq)/(1.0-p[29]);
                qlo_Vbcp=0.0;
                qlo_Vrth=qlo_PSatT*PSatT_Vrth;
                qhi=dvh*(1.0-p[14]+0.5*p[29]*dvh/PSatT)*pwq;
                qhi_dvh=(0.5*dvh*p[29]/PSatT-p[14]+1.0)*pwq+0.5*dvh*p[29]*pwq/PSatT;
                qhi_PSatT=-0.5*(dvh*dvh)*p[29]*pwq/(PSatT*PSatT);
                qhi_Vbcp=qhi_dvh*dvh_Vbcp;
                qhi_Vrth=qhi_dvh*dvh_Vrth;
                qhi_Vrth=qhi_Vrth+qhi_PSatT*PSatT_Vrth;
            }else{
                xvar1=(1.0-(*Vbcp)/PSatT);
                xvar1_Vbcp=-1.0/PSatT;
                xvar1_PSatT=(*Vbcp)/(PSatT*PSatT);
                xvar1_Vrth=xvar1_PSatT*PSatT_Vrth;
                xvar2=(1.0-p[29]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbcp=xvar3_xvar1*xvar1_Vbcp;
                xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
                qlo=PSatT*(1.0-xvar3)/(1.0-p[29]);
                qlo_PSatT=(1.0-xvar3)/(1.0-p[29]);
                qlo_xvar3=-PSatT/(1.0-p[29]);
                qlo_Vrth=qlo_PSatT*PSatT_Vrth;
                qlo_Vbcp=qlo_xvar3*xvar3_Vbcp;
                qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
                qhi=0.0;
                qhi_Vrth=0.0;
                qhi_Vbcp=0.0;
            }
            qdbcp=qlo+qhi;
            qdbcp_qlo=1.0;
            qdbcp_qhi=1.0;
            qdbcp_Vrth=qdbcp_qlo*qlo_Vrth;
            qdbcp_Vbcp=qdbcp_qlo*qlo_Vbcp;
            qdbcp_Vrth=qdbcp_Vrth+qdbcp_qhi*qhi_Vrth;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_qhi*qhi_Vbcp;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[30]*p[30]);
            mv0_dv0=dv0/sqrt((dv0*dv0)+4.0*(p[30]*p[30]));
            mv0_Vrth=mv0_dv0*dv0_Vrth;
            vl0=-0.5*(dv0+mv0);
            vl0_dv0=-0.5;
            vl0_mv0=-0.5;
            vl0_Vrth=vl0_dv0*dv0_Vrth;
            vl0_Vrth=vl0_Vrth+vl0_mv0*mv0_Vrth;
            xvar1=(1.0-vl0/PSatT);
            xvar1_vl0=-1.0/PSatT;
            xvar1_PSatT=vl0/(PSatT*PSatT);
            xvar1_Vrth=xvar1_vl0*vl0_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PSatT*PSatT_Vrth;
            xvar2=(1.0-p[29]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            q0=-PSatT*xvar3/(1.0-p[29]);
            q0_PSatT=-xvar3/(1.0-p[29]);
            q0_xvar3=-PSatT/(1.0-p[29]);
            q0_Vrth=q0_PSatT*PSatT_Vrth;
            q0_Vrth=q0_Vrth+q0_xvar3*xvar3_Vrth;
            dv=(*Vbcp)+dv0;
            dv_Vbcp=1.0;
            dv_dv0=1.0;
            dv_Vrth=dv_dv0*dv0_Vrth;
            mv=sqrt(dv*dv+4.0*p[30]*p[30]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[30]*p[30]));
            mv_Vbcp=mv_dv*dv_Vbcp;
            mv_Vrth=mv_dv*dv_Vrth;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_dv0=-1.0;
            vl_Vbcp=vl_dv*dv_Vbcp;
            vl_Vrth=vl_dv*dv_Vrth;
            vl_Vbcp=vl_Vbcp+vl_mv*mv_Vbcp;
            vl_Vrth=vl_Vrth+vl_mv*mv_Vrth;
            vl_Vrth=vl_Vrth+vl_dv0*dv0_Vrth;
            xvar1=(1.0-vl/PSatT);
            xvar1_vl=-1.0/PSatT;
            xvar1_PSatT=vl/(PSatT*PSatT);
            xvar1_Vbcp=xvar1_vl*vl_Vbcp;
            xvar1_Vrth=xvar1_vl*vl_Vrth;
            xvar1_Vrth=xvar1_Vrth+xvar1_PSatT*PSatT_Vrth;
            xvar2=(1.0-p[29]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbcp=xvar3_xvar1*xvar1_Vbcp;
            xvar3_Vrth=xvar3_xvar1*xvar1_Vrth;
            qlo=-PSatT*xvar3/(1.0-p[29]);
            qlo_PSatT=-xvar3/(1.0-p[29]);
            qlo_xvar3=-PSatT/(1.0-p[29]);
            qlo_Vrth=qlo_PSatT*PSatT_Vrth;
            qlo_Vbcp=qlo_xvar3*xvar3_Vbcp;
            qlo_Vrth=qlo_Vrth+qlo_xvar3*xvar3_Vrth;
            xvar1=(1.0-p[14]);
            xvar2=(-p[29]);
            xvar3=pow(xvar1,xvar2);
            qdbcp=qlo+xvar3*((*Vbcp)-vl+vl0)-q0;
            qdbcp_qlo=1.0;
            qdbcp_Vbcp=xvar3;
            qdbcp_vl=-xvar3;
            qdbcp_vl0=xvar3;
            qdbcp_q0=-1.0;
            qdbcp_Vrth=qdbcp_qlo*qlo_Vrth;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_qlo*qlo_Vbcp;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_vl*vl_Vbcp;
            qdbcp_Vrth=qdbcp_Vrth+qdbcp_vl*vl_Vrth;
            qdbcp_Vrth=qdbcp_Vrth+qdbcp_vl0*vl0_Vrth;
            qdbcp_Vrth=qdbcp_Vrth+qdbcp_q0*q0_Vrth;
        }
    }else{
        qdbcp=0.0;
        qdbcp_Vrth=0.0;
        qdbcp_Vbcp=0.0;
    }
    argi=(*Vbei)/(NFatT*Vtv);
    argi_Vbei=1.0/(NFatT*Vtv);
    argi_NFatT=-(*Vbei)/((NFatT*NFatT)*Vtv);
    argi_Vtv=-(*Vbei)/(NFatT*(Vtv*Vtv));
    argi_Vrth=argi_NFatT*NFatT_Vrth;
    argi_Vrth=argi_Vrth+argi_Vtv*Vtv_Vrth;
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbei=expi_argi*argi_Vbei;
    expi_Vrth=expi_argi*argi_Vrth;
    Ifi=ISatT*(expi-1.0);
    Ifi_ISatT=expi-1.0;
    Ifi_expi=ISatT;
    Ifi_Vrth=Ifi_ISatT*ISatT_Vrth;
    Ifi_Vbei=Ifi_expi*expi_Vbei;
    Ifi_Vrth=Ifi_Vrth+Ifi_expi*expi_Vrth;
    argi=(*Vbci)/(NRatT*Vtv);
    argi_Vbci=1.0/(NRatT*Vtv);
    argi_NRatT=-(*Vbci)/((NRatT*NRatT)*Vtv);
    argi_Vtv=-(*Vbci)/(NRatT*(Vtv*Vtv));
    argi_Vrth=argi_NRatT*NRatT_Vrth;
    argi_Vrth=argi_Vrth+argi_Vtv*Vtv_Vrth;
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    expi_Vrth=expi_argi*argi_Vrth;
    Iri=ISatT*ISRRatT*(expi-1.0);
    Iri_ISatT=(expi-1.0)*ISRRatT;
    Iri_ISRRatT=(expi-1.0)*ISatT;
    Iri_expi=ISatT*ISRRatT;
    Iri_Vrth=Iri_ISatT*ISatT_Vrth;
    Iri_Vrth=Iri_Vrth+Iri_ISRRatT*ISRRatT_Vrth;
    Iri_Vbci=Iri_expi*expi_Vbci;
    Iri_Vrth=Iri_Vrth+Iri_expi*expi_Vrth;
    q1z=1.0+qdbe*IVER+qdbc*IVEF;
    q1z_qdbe=IVER;
    q1z_qdbc=IVEF;
    q1z_Vrth=q1z_qdbe*qdbe_Vrth;
    q1z_Vbei=q1z_qdbe*qdbe_Vbei;
    q1z_Vrth=q1z_Vrth+q1z_qdbc*qdbc_Vrth;
    q1z_Vbci=q1z_qdbc*qdbc_Vbci;
    q1=0.5*(sqrt((q1z-1.0e-4)*(q1z-1.0e-4)+1.0e-8)+q1z-1.0e-4)+1.0e-4;
    q1_q1z=0.5*((q1z-1.0e-4)/sqrt(((q1z-1.0e-4)*(q1z-1.0e-4))+1.0e-8)+1.0);
    q1_Vrth=q1_q1z*q1z_Vrth;
    q1_Vbei=q1_q1z*q1z_Vbei;
    q1_Vbci=q1_q1z*q1z_Vbci;
    q2=Ifi*IIKF+Iri*IIKR;
    q2_Ifi=IIKF;
    q2_IIKF=Ifi;
    q2_Iri=IIKR;
    q2_Vrth=q2_Ifi*Ifi_Vrth;
    q2_Vbei=q2_Ifi*Ifi_Vbei;
    q2_Vrth=q2_Vrth+q2_IIKF*IIKF_Vrth;
    q2_Vrth=q2_Vrth+q2_Iri*Iri_Vrth;
    q2_Vbci=q2_Iri*Iri_Vbci;
    if(p[88]<0.5){
        xvar2=1.0/p[89];
        xvar3=pow(q1,xvar2);
        xvar3_q1=xvar3*xvar2/q1;
        xvar3_Vrth=xvar3_q1*q1_Vrth;
        xvar3_Vbei=xvar3_q1*q1_Vbei;
        xvar3_Vbci=xvar3_q1*q1_Vbci;
        xvar1=(xvar3+4.0*q2);
        xvar1_xvar3=1.0;
        xvar1_q2=4.0;
        xvar1_Vrth=xvar1_xvar3*xvar3_Vrth;
        xvar1_Vbei=xvar1_xvar3*xvar3_Vbei;
        xvar1_Vbci=xvar1_xvar3*xvar3_Vbci;
        xvar1_Vrth=xvar1_Vrth+xvar1_q2*q2_Vrth;
        xvar1_Vbei=xvar1_Vbei+xvar1_q2*q2_Vbei;
        xvar1_Vbci=xvar1_Vbci+xvar1_q2*q2_Vbci;
        xvar4=pow(xvar1,p[89]);
        xvar4_xvar1=xvar4*p[89]/xvar1;
        xvar4_Vrth=xvar4_xvar1*xvar1_Vrth;
        xvar4_Vbei=xvar4_xvar1*xvar1_Vbei;
        xvar4_Vbci=xvar4_xvar1*xvar1_Vbci;
        qb=0.5*(q1+xvar4);
        qb_q1=0.5;
        qb_xvar4=0.5;
        qb_Vrth=qb_q1*q1_Vrth;
        qb_Vbei=qb_q1*q1_Vbei;
        qb_Vbci=qb_q1*q1_Vbci;
        qb_Vrth=qb_Vrth+qb_xvar4*xvar4_Vrth;
        qb_Vbei=qb_Vbei+qb_xvar4*xvar4_Vbei;
        qb_Vbci=qb_Vbci+qb_xvar4*xvar4_Vbci;
    }else{
        xvar1=(1.0+4.0*q2);
        xvar1_q2=4.0;
        xvar1_Vrth=xvar1_q2*q2_Vrth;
        xvar1_Vbei=xvar1_q2*q2_Vbei;
        xvar1_Vbci=xvar1_q2*q2_Vbci;
        xvar2=pow(xvar1,p[89]);
        xvar2_xvar1=xvar2*p[89]/xvar1;
        xvar2_Vrth=xvar2_xvar1*xvar1_Vrth;
        xvar2_Vbei=xvar2_xvar1*xvar1_Vbei;
        xvar2_Vbci=xvar2_xvar1*xvar1_Vbci;
        qb=0.5*q1*(1.0+xvar2);
        qb_q1=0.5*(xvar2+1.0);
        qb_xvar2=0.5*q1;
        qb_Vrth=qb_q1*q1_Vrth;
        qb_Vbei=qb_q1*q1_Vbei;
        qb_Vbci=qb_q1*q1_Vbci;
        qb_Vrth=qb_Vrth+qb_xvar2*xvar2_Vrth;
        qb_Vbei=qb_Vbei+qb_xvar2*xvar2_Vbei;
        qb_Vbci=qb_Vbci+qb_xvar2*xvar2_Vbci;
    }
    (*Itzr)=Iri/qb;
    Itzr_Iri=1.0/qb;
    Itzr_qb=-Iri/(qb*qb);
    *Itzr_Vrth=Itzr_Iri*Iri_Vrth;
    *Itzr_Vbci=Itzr_Iri*Iri_Vbci;
    *Itzr_Vrth=(*Itzr_Vrth)+Itzr_qb*qb_Vrth;
    *Itzr_Vbei=Itzr_qb*qb_Vbei;
    *Itzr_Vbci=(*Itzr_Vbci)+Itzr_qb*qb_Vbci;
    (*Itzf)=Ifi/qb;
    Itzf_Ifi=1.0/qb;
    Itzf_qb=-Ifi/(qb*qb);
    *Itzf_Vrth=Itzf_Ifi*Ifi_Vrth;
    *Itzf_Vbei=Itzf_Ifi*Ifi_Vbei;
    *Itzf_Vrth=(*Itzf_Vrth)+Itzf_qb*qb_Vrth;
    *Itzf_Vbei=(*Itzf_Vbei)+Itzf_qb*qb_Vbei;
    *Itzf_Vbci=Itzf_qb*qb_Vbci;
    if(p[42]>0.0){
        argi=(*Vbep)/(p[44]*Vtv);
        argi_Vbep=1.0/(p[44]*Vtv);
        argi_Vtv=-(*Vbep)/(p[44]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbep=expi_argi*argi_Vbep;
        expi_Vrth=expi_argi*argi_Vrth;
        argx=(*Vbci)/(p[44]*Vtv);
        argx_Vbci=1.0/(p[44]*Vtv);
        argx_Vtv=-(*Vbci)/(p[44]*(Vtv*Vtv));
        argx_Vrth=argx_Vtv*Vtv_Vrth;
        expx=exp(argx);
        expx_argx=expx;
        expx_Vbci=expx_argx*argx_Vbci;
        expx_Vrth=expx_argx*argx_Vrth;
        Ifp=ISPatT*(p[43]*expi+(1.0-p[43])*expx-1.0);
        Ifp_ISPatT=expi*p[43]+expx*(1.0-p[43])-1.0;
        Ifp_expi=ISPatT*p[43];
        Ifp_expx=ISPatT*(1.0-p[43]);
        Ifp_Vrth=Ifp_ISPatT*ISPatT_Vrth;
        Ifp_Vbep=Ifp_expi*expi_Vbep;
        Ifp_Vrth=Ifp_Vrth+Ifp_expi*expi_Vrth;
        Ifp_Vbci=Ifp_expx*expx_Vbci;
        Ifp_Vrth=Ifp_Vrth+Ifp_expx*expx_Vrth;
        q2p=Ifp*IIKP;
        q2p_Ifp=IIKP;
        q2p_Vrth=q2p_Ifp*Ifp_Vrth;
        q2p_Vbep=q2p_Ifp*Ifp_Vbep;
        q2p_Vbci=q2p_Ifp*Ifp_Vbci;
        qbp=0.5*(1.0+sqrt(1.0+4.0*q2p));
        qbp_q2p=1.0/sqrt(4.0*q2p+1.0);
        qbp_Vrth=qbp_q2p*q2p_Vrth;
        qbp_Vbep=qbp_q2p*q2p_Vbep;
        qbp_Vbci=qbp_q2p*q2p_Vbci;
        argi=(*Vbcp)/(p[44]*Vtv);
        argi_Vbcp=1.0/(p[44]*Vtv);
        argi_Vtv=-(*Vbcp)/(p[44]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbcp=expi_argi*argi_Vbcp;
        expi_Vrth=expi_argi*argi_Vrth;
        Irp=ISPatT*(expi-1.0);
        Irp_ISPatT=expi-1.0;
        Irp_expi=ISPatT;
        Irp_Vrth=Irp_ISPatT*ISPatT_Vrth;
        Irp_Vbcp=Irp_expi*expi_Vbcp;
        Irp_Vrth=Irp_Vrth+Irp_expi*expi_Vrth;
        (*Iccp)=(Ifp-Irp)/qbp;
        Iccp_Ifp=1.0/qbp;
        Iccp_Irp=-1.0/qbp;
        Iccp_qbp=-(Ifp-Irp)/(qbp*qbp);
        *Iccp_Vrth=Iccp_Ifp*Ifp_Vrth;
        *Iccp_Vbep=Iccp_Ifp*Ifp_Vbep;
        *Iccp_Vbci=Iccp_Ifp*Ifp_Vbci;
        *Iccp_Vrth=(*Iccp_Vrth)+Iccp_Irp*Irp_Vrth;
        *Iccp_Vbcp=Iccp_Irp*Irp_Vbcp;
        *Iccp_Vrth=(*Iccp_Vrth)+Iccp_qbp*qbp_Vrth;
        *Iccp_Vbep=(*Iccp_Vbep)+Iccp_qbp*qbp_Vbep;
        *Iccp_Vbci=(*Iccp_Vbci)+Iccp_qbp*qbp_Vbci;
    }else{
        Ifp=0.0;
        Ifp_Vrth=0.0;
        Ifp_Vbep=0.0;
        Ifp_Vbci=0.0;
        qbp=1.0;
        qbp_Vrth=0.0;
        qbp_Vbep=0.0;
        qbp_Vbci=0.0;
        (*Iccp)=0.0;
        *Iccp_Vrth=0.0;
        *Iccp_Vbep=0.0;
        *Iccp_Vbci=0.0;
        *Iccp_Vbcp=0.0;
    }
    if(p[32]==1.0){
        argi=(*Vbei)/(p[33]*Vtv);
        argi_Vbei=1.0/(p[33]*Vtv);
        argi_Vtv=-(*Vbei)/(p[33]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbei=expi_argi*argi_Vbei;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbei)/(p[35]*Vtv);
        argn_Vbei=1.0/(p[35]*Vtv);
        argn_Vtv=-(*Vbei)/(p[35]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbei=expn_argn*argn_Vbei;
        expn_Vrth=expn_argn*argn_Vrth;
        if(p[98]>0.0){
            argx=(-VBBEatT-(*Vbei))/(NBBEatT*Vtv);
            argx_VBBEatT=-1.0/(NBBEatT*Vtv);
            argx_Vbei=-1.0/(NBBEatT*Vtv);
            argx_NBBEatT=-(-(*Vbei)-VBBEatT)/((NBBEatT*NBBEatT)*Vtv);
            argx_Vtv=-(-(*Vbei)-VBBEatT)/(NBBEatT*(Vtv*Vtv));
            argx_Vrth=argx_VBBEatT*VBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_NBBEatT*NBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_Vtv*Vtv_Vrth;
            expx=exp(argx);
            expx_argx=expx;
            expx_Vrth=expx_argx*argx_Vrth;
            expx_Vbei=expx_argx*argx_Vbei;
            (*Ibe)=IBEIatT*(expi-1.0)+IBENatT*(expn-1.0)-p[100]*(expx-EBBEatT);
            Ibe_IBEIatT=expi-1.0;
            Ibe_expi=IBEIatT;
            Ibe_IBENatT=expn-1.0;
            Ibe_expn=IBENatT;
            Ibe_expx=-p[100];
            Ibe_EBBEatT=p[100];
            *Ibe_Vrth=Ibe_IBEIatT*IBEIatT_Vrth;
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expi*expi_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_IBENatT*IBENatT_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expn*expn_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expx*expx_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expx*expx_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_EBBEatT*EBBEatT_Vrth;
        }else{
            (*Ibe)=IBEIatT*(expi-1.0)+IBENatT*(expn-1.0);
            Ibe_IBEIatT=expi-1.0;
            Ibe_expi=IBEIatT;
            Ibe_IBENatT=expn-1.0;
            Ibe_expn=IBENatT;
            *Ibe_Vrth=Ibe_IBEIatT*IBEIatT_Vrth;
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expi*expi_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_IBENatT*IBENatT_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expn*expn_Vrth;
        }
        (*Ibex)=0.0;
        *Ibex_Vrth=0.0;
        *Ibex_Vbex=0.0;
    }else if(p[32]==0.0){
        (*Ibe)=0.0;
        *Ibe_Vrth=0.0;
        *Ibe_Vbei=0.0;
        argi=(*Vbex)/(p[33]*Vtv);
        argi_Vbex=1.0/(p[33]*Vtv);
        argi_Vtv=-(*Vbex)/(p[33]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbex=expi_argi*argi_Vbex;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbex)/(p[35]*Vtv);
        argn_Vbex=1.0/(p[35]*Vtv);
        argn_Vtv=-(*Vbex)/(p[35]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbex=expn_argn*argn_Vbex;
        expn_Vrth=expn_argn*argn_Vrth;
        if(p[98]>0.0){
            argx=(-VBBEatT-(*Vbex))/(NBBEatT*Vtv);
            argx_VBBEatT=-1.0/(NBBEatT*Vtv);
            argx_Vbex=-1.0/(NBBEatT*Vtv);
            argx_NBBEatT=-(-(*Vbex)-VBBEatT)/((NBBEatT*NBBEatT)*Vtv);
            argx_Vtv=-(-(*Vbex)-VBBEatT)/(NBBEatT*(Vtv*Vtv));
            argx_Vrth=argx_VBBEatT*VBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_NBBEatT*NBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_Vtv*Vtv_Vrth;
            expx=exp(argx);
            expx_argx=expx;
            expx_Vrth=expx_argx*argx_Vrth;
            expx_Vbex=expx_argx*argx_Vbex;
            (*Ibex)=IBEIatT*(expi-1.0)+IBENatT*(expn-1.0)-p[100]*(expx-EBBEatT);
            Ibex_IBEIatT=expi-1.0;
            Ibex_expi=IBEIatT;
            Ibex_IBENatT=expn-1.0;
            Ibex_expn=IBENatT;
            Ibex_expx=-p[100];
            Ibex_EBBEatT=p[100];
            *Ibex_Vrth=Ibex_IBEIatT*IBEIatT_Vrth;
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expi*expi_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_IBENatT*IBENatT_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expn*expn_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expx*expx_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expx*expx_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_EBBEatT*EBBEatT_Vrth;
        }else{
            (*Ibex)=IBEIatT*(expi-1.0)+IBENatT*(expn-1.0);
            Ibex_IBEIatT=expi-1.0;
            Ibex_expi=IBEIatT;
            Ibex_IBENatT=expn-1.0;
            Ibex_expn=IBENatT;
            *Ibex_Vrth=Ibex_IBEIatT*IBEIatT_Vrth;
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expi*expi_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_IBENatT*IBENatT_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expn*expn_Vrth;
        }
    }else{
        argi=(*Vbei)/(p[33]*Vtv);
        argi_Vbei=1.0/(p[33]*Vtv);
        argi_Vtv=-(*Vbei)/(p[33]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbei=expi_argi*argi_Vbei;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbei)/(p[35]*Vtv);
        argn_Vbei=1.0/(p[35]*Vtv);
        argn_Vtv=-(*Vbei)/(p[35]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbei=expn_argn*argn_Vbei;
        expn_Vrth=expn_argn*argn_Vrth;
        if(p[98]>0.0){
            argx=(-VBBEatT-(*Vbei))/(NBBEatT*Vtv);
            argx_VBBEatT=-1.0/(NBBEatT*Vtv);
            argx_Vbei=-1.0/(NBBEatT*Vtv);
            argx_NBBEatT=-(-(*Vbei)-VBBEatT)/((NBBEatT*NBBEatT)*Vtv);
            argx_Vtv=-(-(*Vbei)-VBBEatT)/(NBBEatT*(Vtv*Vtv));
            argx_Vrth=argx_VBBEatT*VBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_NBBEatT*NBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_Vtv*Vtv_Vrth;
            expx=exp(argx);
            expx_argx=expx;
            expx_Vrth=expx_argx*argx_Vrth;
            expx_Vbei=expx_argx*argx_Vbei;
            (*Ibe)=p[32]*(IBEIatT*(expi-1.0)+IBENatT*(expn-1.0)-p[100]*(expx-EBBEatT));
            Ibe_IBEIatT=(expi-1.0)*p[32];
            Ibe_expi=IBEIatT*p[32];
            Ibe_IBENatT=(expn-1.0)*p[32];
            Ibe_expn=IBENatT*p[32];
            Ibe_expx=-p[100]*p[32];
            Ibe_EBBEatT=p[100]*p[32];
            *Ibe_Vrth=Ibe_IBEIatT*IBEIatT_Vrth;
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expi*expi_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_IBENatT*IBENatT_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expn*expn_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expx*expx_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expx*expx_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_EBBEatT*EBBEatT_Vrth;
        }else{
            (*Ibe)=p[32]*(IBEIatT*(expi-1.0)+IBENatT*(expn-1.0));
            Ibe_IBEIatT=(expi-1.0)*p[32];
            Ibe_expi=IBEIatT*p[32];
            Ibe_IBENatT=(expn-1.0)*p[32];
            Ibe_expn=IBENatT*p[32];
            *Ibe_Vrth=Ibe_IBEIatT*IBEIatT_Vrth;
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expi*expi_Vrth;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_IBENatT*IBENatT_Vrth;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vrth=(*Ibe_Vrth)+Ibe_expn*expn_Vrth;
        }
        argi=(*Vbex)/(p[33]*Vtv);
        argi_Vbex=1.0/(p[33]*Vtv);
        argi_Vtv=-(*Vbex)/(p[33]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbex=expi_argi*argi_Vbex;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbex)/(p[35]*Vtv);
        argn_Vbex=1.0/(p[35]*Vtv);
        argn_Vtv=-(*Vbex)/(p[35]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbex=expn_argn*argn_Vbex;
        expn_Vrth=expn_argn*argn_Vrth;
        if(p[98]>0.0){
            argx=(-VBBEatT-(*Vbex))/(NBBEatT*Vtv);
            argx_VBBEatT=-1.0/(NBBEatT*Vtv);
            argx_Vbex=-1.0/(NBBEatT*Vtv);
            argx_NBBEatT=-(-(*Vbex)-VBBEatT)/((NBBEatT*NBBEatT)*Vtv);
            argx_Vtv=-(-(*Vbex)-VBBEatT)/(NBBEatT*(Vtv*Vtv));
            argx_Vrth=argx_VBBEatT*VBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_NBBEatT*NBBEatT_Vrth;
            argx_Vrth=argx_Vrth+argx_Vtv*Vtv_Vrth;
            expx=exp(argx);
            expx_argx=expx;
            expx_Vrth=expx_argx*argx_Vrth;
            expx_Vbex=expx_argx*argx_Vbex;
            (*Ibex)=(1.0-p[32])*(IBEIatT*(expi-1.0)+IBENatT*(expn-1.0)-p[100]*(expx-EBBEatT));
            Ibex_IBEIatT=(expi-1.0)*(1.0-p[32]);
            Ibex_expi=IBEIatT*(1.0-p[32]);
            Ibex_IBENatT=(expn-1.0)*(1.0-p[32]);
            Ibex_expn=IBENatT*(1.0-p[32]);
            Ibex_expx=-p[100]*(1.0-p[32]);
            Ibex_EBBEatT=p[100]*(1.0-p[32]);
            *Ibex_Vrth=Ibex_IBEIatT*IBEIatT_Vrth;
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expi*expi_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_IBENatT*IBENatT_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expn*expn_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expx*expx_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expx*expx_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_EBBEatT*EBBEatT_Vrth;
        }else{
            (*Ibex)=(1.0-p[32])*(IBEIatT*(expi-1.0)+IBENatT*(expn-1.0));
            Ibex_IBEIatT=(expi-1.0)*(1.0-p[32]);
            Ibex_expi=IBEIatT*(1.0-p[32]);
            Ibex_IBENatT=(expn-1.0)*(1.0-p[32]);
            Ibex_expn=IBENatT*(1.0-p[32]);
            *Ibex_Vrth=Ibex_IBEIatT*IBEIatT_Vrth;
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expi*expi_Vrth;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_IBENatT*IBENatT_Vrth;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vrth=(*Ibex_Vrth)+Ibex_expn*expn_Vrth;
        }
    }
    argi=(*Vbci)/(p[37]*Vtv);
    argi_Vbci=1.0/(p[37]*Vtv);
    argi_Vtv=-(*Vbci)/(p[37]*(Vtv*Vtv));
    argi_Vrth=argi_Vtv*Vtv_Vrth;
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    expi_Vrth=expi_argi*argi_Vrth;
    argn=(*Vbci)/(p[39]*Vtv);
    argn_Vbci=1.0/(p[39]*Vtv);
    argn_Vtv=-(*Vbci)/(p[39]*(Vtv*Vtv));
    argn_Vrth=argn_Vtv*Vtv_Vrth;
    expn=exp(argn);
    expn_argn=expn;
    expn_Vbci=expn_argn*argn_Vbci;
    expn_Vrth=expn_argn*argn_Vrth;
    Ibcj=IBCIatT*(expi-1.0)+IBCNatT*(expn-1.0);
    Ibcj_IBCIatT=expi-1.0;
    Ibcj_expi=IBCIatT;
    Ibcj_IBCNatT=expn-1.0;
    Ibcj_expn=IBCNatT;
    Ibcj_Vrth=Ibcj_IBCIatT*IBCIatT_Vrth;
    Ibcj_Vbci=Ibcj_expi*expi_Vbci;
    Ibcj_Vrth=Ibcj_Vrth+Ibcj_expi*expi_Vrth;
    Ibcj_Vrth=Ibcj_Vrth+Ibcj_IBCNatT*IBCNatT_Vrth;
    Ibcj_Vbci=Ibcj_Vbci+Ibcj_expn*expn_Vbci;
    Ibcj_Vrth=Ibcj_Vrth+Ibcj_expn*expn_Vrth;
    if((p[45]>0.0)||(p[46]>0.0)){
        argi=(*Vbep)/(p[37]*Vtv);
        argi_Vbep=1.0/(p[37]*Vtv);
        argi_Vtv=-(*Vbep)/(p[37]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbep=expi_argi*argi_Vbep;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbep)/(p[39]*Vtv);
        argn_Vbep=1.0/(p[39]*Vtv);
        argn_Vtv=-(*Vbep)/(p[39]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbep=expn_argn*argn_Vbep;
        expn_Vrth=expn_argn*argn_Vrth;
        (*Ibep)=IBEIPatT*(expi-1.0)+IBENPatT*(expn-1.0);
        Ibep_IBEIPatT=expi-1.0;
        Ibep_expi=IBEIPatT;
        Ibep_IBENPatT=expn-1.0;
        Ibep_expn=IBENPatT;
        *Ibep_Vrth=Ibep_IBEIPatT*IBEIPatT_Vrth;
        *Ibep_Vbep=Ibep_expi*expi_Vbep;
        *Ibep_Vrth=(*Ibep_Vrth)+Ibep_expi*expi_Vrth;
        *Ibep_Vrth=(*Ibep_Vrth)+Ibep_IBENPatT*IBENPatT_Vrth;
        *Ibep_Vbep=(*Ibep_Vbep)+Ibep_expn*expn_Vbep;
        *Ibep_Vrth=(*Ibep_Vrth)+Ibep_expn*expn_Vrth;
    }else{
        (*Ibep)=0.0;
        *Ibep_Vrth=0.0;
        *Ibep_Vbep=0.0;
    }
    if(p[40]>0.0){
        vl=0.5*(sqrt((PCatT-(*Vbci))*(PCatT-(*Vbci))+0.01)+(PCatT-(*Vbci)));
        vl_PCatT=0.5*((PCatT-(*Vbci))/sqrt(((PCatT-(*Vbci))*(PCatT-(*Vbci)))+0.01)+1.0);
        vl_Vbci=0.5*(-(PCatT-(*Vbci))/sqrt(((PCatT-(*Vbci))*(PCatT-(*Vbci)))+0.01)-1.0);
        vl_Vrth=vl_PCatT*PCatT_Vrth;
        xvar2=(p[25]-1.0);
        xvar3=pow(vl,xvar2);
        xvar3_vl=xvar3*xvar2/vl;
        xvar3_Vrth=xvar3_vl*vl_Vrth;
        xvar3_Vbci=xvar3_vl*vl_Vbci;
        xvar1=-AVC2atT*xvar3;
        xvar1_AVC2atT=-xvar3;
        xvar1_xvar3=-AVC2atT;
        xvar1_Vrth=xvar1_AVC2atT*AVC2atT_Vrth;
        xvar1_Vrth=xvar1_Vrth+xvar1_xvar3*xvar3_Vrth;
        xvar1_Vbci=xvar1_xvar3*xvar3_Vbci;
        xvar4=exp(xvar1);
        xvar4_xvar1=xvar4;
        xvar4_Vrth=xvar4_xvar1*xvar1_Vrth;
        xvar4_Vbci=xvar4_xvar1*xvar1_Vbci;
        avalf=p[40]*vl*xvar4;
        avalf_vl=p[40]*xvar4;
        avalf_xvar4=p[40]*vl;
        avalf_Vrth=avalf_vl*vl_Vrth;
        avalf_Vbci=avalf_vl*vl_Vbci;
        avalf_Vrth=avalf_Vrth+avalf_xvar4*xvar4_Vrth;
        avalf_Vbci=avalf_Vbci+avalf_xvar4*xvar4_Vbci;
        Igc=((*Itzf)-(*Itzr)-Ibcj)*avalf;
        Igc_Itzf=avalf;
        Igc_Itzr=-avalf;
        Igc_Ibcj=-avalf;
        Igc_avalf=-(*Itzr)+(*Itzf)-Ibcj;
        Igc_Vrth=Igc_Itzf*(*Itzf_Vrth);
        Igc_Vbei=Igc_Itzf*(*Itzf_Vbei);
        Igc_Vbci=Igc_Itzf*(*Itzf_Vbci);
        Igc_Vrth=Igc_Vrth+Igc_Itzr*(*Itzr_Vrth);
        Igc_Vbci=Igc_Vbci+Igc_Itzr*(*Itzr_Vbci);
        Igc_Vbei=Igc_Vbei+Igc_Itzr*(*Itzr_Vbei);
        Igc_Vrth=Igc_Vrth+Igc_Ibcj*Ibcj_Vrth;
        Igc_Vbci=Igc_Vbci+Igc_Ibcj*Ibcj_Vbci;
        Igc_Vrth=Igc_Vrth+Igc_avalf*avalf_Vrth;
        Igc_Vbci=Igc_Vbci+Igc_avalf*avalf_Vbci;
    }else{
        Igc=0.0;
        Igc_Vrth=0.0;
        Igc_Vbei=0.0;
        Igc_Vbci=0.0;
    }
    (*Ibc)=Ibcj-Igc;
    Ibc_Ibcj=1.0;
    Ibc_Igc=-1.0;
    *Ibc_Vrth=Ibc_Ibcj*Ibcj_Vrth;
    *Ibc_Vbci=Ibc_Ibcj*Ibcj_Vbci;
    *Ibc_Vrth=(*Ibc_Vrth)+Ibc_Igc*Igc_Vrth;
    *Ibc_Vbei=Ibc_Igc*Igc_Vbei;
    *Ibc_Vbci=(*Ibc_Vbci)+Ibc_Igc*Igc_Vbci;
    if(p[1]>0.0){
        (*Ircx)=(*Vrcx)/RCXatT;
        *Ircx_Vrcx=1.0/RCXatT;
        Ircx_RCXatT=-(*Vrcx)/(RCXatT*RCXatT);
        *Ircx_Vrth=Ircx_RCXatT*RCXatT_Vrth;
    }else{
        (*Ircx)=0.0;
        *Ircx_Vrcx=0.0;
        *Ircx_Vrth=0.0;
    }
    argi=(*Vbci)/Vtv;
    argi_Vbci=1.0/Vtv;
    argi_Vtv=-(*Vbci)/(Vtv*Vtv);
    argi_Vrth=argi_Vtv*Vtv_Vrth;
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    expi_Vrth=expi_argi*argi_Vrth;
    argx=(*Vbcx)/Vtv;
    argx_Vbcx=1.0/Vtv;
    argx_Vtv=-(*Vbcx)/(Vtv*Vtv);
    argx_Vrth=argx_Vtv*Vtv_Vrth;
    expx=exp(argx);
    expx_argx=expx;
    expx_Vbcx=expx_argx*argx_Vbcx;
    expx_Vrth=expx_argx*argx_Vrth;
    Kbci=sqrt(1.0+GAMMatT*expi);
    Kbci_GAMMatT=expi/(2.0*sqrt(expi*GAMMatT+1.0));
    Kbci_expi=GAMMatT/(2.0*sqrt(expi*GAMMatT+1.0));
    Kbci_Vrth=Kbci_GAMMatT*GAMMatT_Vrth;
    Kbci_Vbci=Kbci_expi*expi_Vbci;
    Kbci_Vrth=Kbci_Vrth+Kbci_expi*expi_Vrth;
    Kbcx=sqrt(1.0+GAMMatT*expx);
    Kbcx_GAMMatT=expx/(2.0*sqrt(expx*GAMMatT+1.0));
    Kbcx_expx=GAMMatT/(2.0*sqrt(expx*GAMMatT+1.0));
    Kbcx_Vrth=Kbcx_GAMMatT*GAMMatT_Vrth;
    Kbcx_Vbcx=Kbcx_expx*expx_Vbcx;
    Kbcx_Vrth=Kbcx_Vrth+Kbcx_expx*expx_Vrth;
    if(p[2]>0.0){
        rKp1=(Kbci+1.0)/(Kbcx+1.0);
        rKp1_Kbci=1.0/(Kbcx+1.0);
        rKp1_Kbcx=-(Kbci+1.0)/((Kbcx+1.0)*(Kbcx+1.0));
        rKp1_Vrth=rKp1_Kbci*Kbci_Vrth;
        rKp1_Vbci=rKp1_Kbci*Kbci_Vbci;
        rKp1_Vrth=rKp1_Vrth+rKp1_Kbcx*Kbcx_Vrth;
        rKp1_Vbcx=rKp1_Kbcx*Kbcx_Vbcx;
        xvar1=log(rKp1);
        xvar1_rKp1=1.0/rKp1;
        xvar1_Vrth=xvar1_rKp1*rKp1_Vrth;
        xvar1_Vbci=xvar1_rKp1*rKp1_Vbci;
        xvar1_Vbcx=xvar1_rKp1*rKp1_Vbcx;
        Iohm=((*Vrci)+Vtv*(Kbci-Kbcx-xvar1))/RCIatT;
        Iohm_Vrci=1.0/RCIatT;
        Iohm_Vtv=(-xvar1-Kbcx+Kbci)/RCIatT;
        Iohm_Kbci=Vtv/RCIatT;
        Iohm_Kbcx=-Vtv/RCIatT;
        Iohm_xvar1=-Vtv/RCIatT;
        Iohm_RCIatT=-(Vtv*(-xvar1-Kbcx+Kbci)+(*Vrci))/(RCIatT*RCIatT);
        Iohm_Vrth=Iohm_Vtv*Vtv_Vrth;
        Iohm_Vrth=Iohm_Vrth+Iohm_Kbci*Kbci_Vrth;
        Iohm_Vbci=Iohm_Kbci*Kbci_Vbci;
        Iohm_Vrth=Iohm_Vrth+Iohm_Kbcx*Kbcx_Vrth;
        Iohm_Vbcx=Iohm_Kbcx*Kbcx_Vbcx;
        Iohm_Vrth=Iohm_Vrth+Iohm_xvar1*xvar1_Vrth;
        Iohm_Vbci=Iohm_Vbci+Iohm_xvar1*xvar1_Vbci;
        Iohm_Vbcx=Iohm_Vbcx+Iohm_xvar1*xvar1_Vbcx;
        Iohm_Vrth=Iohm_Vrth+Iohm_RCIatT*RCIatT_Vrth;
        derf=IVO*RCIatT*Iohm/(1.0+0.5*IVO*IHRCF*sqrt((*Vrci)*(*Vrci)+0.01));
        derf_IVO=Iohm*RCIatT/(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)-0.5*IHRCF*Iohm*IVO*RCIatT*sqrt(((*Vrci)*(*Vrci))+0.01)/((0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)*(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0));
        derf_RCIatT=Iohm*IVO/(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0);
        derf_Iohm=IVO*RCIatT/(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0);
        derf_Vrci=-0.5*IHRCF*Iohm*(IVO*IVO)*RCIatT*(*Vrci)/(sqrt(((*Vrci)*(*Vrci))+0.01)*((0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)*(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)));
        derf_Vrth=derf_IVO*IVO_Vrth;
        derf_Vrth=derf_Vrth+derf_RCIatT*RCIatT_Vrth;
        derf_Vrci=derf_Vrci+derf_Iohm*Iohm_Vrci;
        derf_Vrth=derf_Vrth+derf_Iohm*Iohm_Vrth;
        derf_Vbci=derf_Iohm*Iohm_Vbci;
        derf_Vbcx=derf_Iohm*Iohm_Vbcx;
        (*Irci)=Iohm/sqrt(1.0+derf*derf);
        Irci_Iohm=1.0/sqrt((derf*derf)+1.0);
        Irci_derf=-derf*Iohm/pow(((derf*derf)+1.0),(3.0/2.0));
        *Irci_Vrci=Irci_Iohm*Iohm_Vrci;
        *Irci_Vrth=Irci_Iohm*Iohm_Vrth;
        *Irci_Vbci=Irci_Iohm*Iohm_Vbci;
        *Irci_Vbcx=Irci_Iohm*Iohm_Vbcx;
        *Irci_Vrth=(*Irci_Vrth)+Irci_derf*derf_Vrth;
        *Irci_Vrci=(*Irci_Vrci)+Irci_derf*derf_Vrci;
        *Irci_Vbci=(*Irci_Vbci)+Irci_derf*derf_Vbci;
        *Irci_Vbcx=(*Irci_Vbcx)+Irci_derf*derf_Vbcx;
    }else{
        (*Irci)=0.0;
        *Irci_Vrci=0.0;
        *Irci_Vrth=0.0;
        *Irci_Vbci=0.0;
        *Irci_Vbcx=0.0;
    }
    if(p[6]>0.0){
        (*Irbx)=(*Vrbx)/RBXatT;
        *Irbx_Vrbx=1.0/RBXatT;
        Irbx_RBXatT=-(*Vrbx)/(RBXatT*RBXatT);
        *Irbx_Vrth=Irbx_RBXatT*RBXatT_Vrth;
    }else{
        (*Irbx)=0.0;
        *Irbx_Vrbx=0.0;
        *Irbx_Vrth=0.0;
    }
    if(p[7]>0.0){
        (*Irbi)=(*Vrbi)*qb/RBIatT;
        *Irbi_Vrbi=qb/RBIatT;
        Irbi_qb=(*Vrbi)/RBIatT;
        Irbi_RBIatT=-qb*(*Vrbi)/(RBIatT*RBIatT);
        *Irbi_Vrth=Irbi_qb*qb_Vrth;
        *Irbi_Vbei=Irbi_qb*qb_Vbei;
        *Irbi_Vbci=Irbi_qb*qb_Vbci;
        *Irbi_Vrth=(*Irbi_Vrth)+Irbi_RBIatT*RBIatT_Vrth;
    }else{
        (*Irbi)=0.0;
        *Irbi_Vrbi=0.0;
        *Irbi_Vrth=0.0;
        *Irbi_Vbei=0.0;
        *Irbi_Vbci=0.0;
    }
    if(p[8]>0.0){
        (*Ire)=(*Vre)/REatT;
        *Ire_Vre=1.0/REatT;
        Ire_REatT=-(*Vre)/(REatT*REatT);
        *Ire_Vrth=Ire_REatT*REatT_Vrth;
    }else{
        (*Ire)=0.0;
        *Ire_Vre=0.0;
        *Ire_Vrth=0.0;
    }
    if(p[10]>0.0){
        (*Irbp)=(*Vrbp)*qbp/RBPatT;
        *Irbp_Vrbp=qbp/RBPatT;
        Irbp_qbp=(*Vrbp)/RBPatT;
        Irbp_RBPatT=-qbp*(*Vrbp)/(RBPatT*RBPatT);
        *Irbp_Vrth=Irbp_qbp*qbp_Vrth;
        *Irbp_Vbep=Irbp_qbp*qbp_Vbep;
        *Irbp_Vbci=Irbp_qbp*qbp_Vbci;
        *Irbp_Vrth=(*Irbp_Vrth)+Irbp_RBPatT*RBPatT_Vrth;
    }else{
        (*Irbp)=0.0;
        *Irbp_Vrbp=0.0;
        *Irbp_Vrth=0.0;
        *Irbp_Vbep=0.0;
        *Irbp_Vbci=0.0;
    }
    if((p[47]>0.0)||(p[49]>0.0)){
        argi=(*Vbcp)/(p[48]*Vtv);
        argi_Vbcp=1.0/(p[48]*Vtv);
        argi_Vtv=-(*Vbcp)/(p[48]*(Vtv*Vtv));
        argi_Vrth=argi_Vtv*Vtv_Vrth;
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbcp=expi_argi*argi_Vbcp;
        expi_Vrth=expi_argi*argi_Vrth;
        argn=(*Vbcp)/(p[50]*Vtv);
        argn_Vbcp=1.0/(p[50]*Vtv);
        argn_Vtv=-(*Vbcp)/(p[50]*(Vtv*Vtv));
        argn_Vrth=argn_Vtv*Vtv_Vrth;
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbcp=expn_argn*argn_Vbcp;
        expn_Vrth=expn_argn*argn_Vrth;
        (*Ibcp)=IBCIPatT*(expi-1.0)+IBCNPatT*(expn-1.0);
        Ibcp_IBCIPatT=expi-1.0;
        Ibcp_expi=IBCIPatT;
        Ibcp_IBCNPatT=expn-1.0;
        Ibcp_expn=IBCNPatT;
        *Ibcp_Vrth=Ibcp_IBCIPatT*IBCIPatT_Vrth;
        *Ibcp_Vbcp=Ibcp_expi*expi_Vbcp;
        *Ibcp_Vrth=(*Ibcp_Vrth)+Ibcp_expi*expi_Vrth;
        *Ibcp_Vrth=(*Ibcp_Vrth)+Ibcp_IBCNPatT*IBCNPatT_Vrth;
        *Ibcp_Vbcp=(*Ibcp_Vbcp)+Ibcp_expn*expn_Vbcp;
        *Ibcp_Vrth=(*Ibcp_Vrth)+Ibcp_expn*expn_Vrth;
    }else{
        (*Ibcp)=0.0;
        *Ibcp_Vrth=0.0;
        *Ibcp_Vbcp=0.0;
    }
    if(p[9]>0.0){
        (*Irs)=(*Vrs)/RSatT;
        *Irs_Vrs=1.0/RSatT;
        Irs_RSatT=-(*Vrs)/(RSatT*RSatT);
        *Irs_Vrth=Irs_RSatT*RSatT_Vrth;
    }else{
        (*Irs)=0.0;
        *Irs_Vrs=0.0;
        *Irs_Vrth=0.0;
    }
    if(Ifi>0.0){
        sgIf=1.0;
    }else{
        sgIf=0.0;
    }
    rIf=Ifi*sgIf*IITF;
    rIf_Ifi=IITF*sgIf;
    rIf_Vrth=rIf_Ifi*Ifi_Vrth;
    rIf_Vbei=rIf_Ifi*Ifi_Vbei;
    mIf=rIf/(rIf+1.0);
    mIf_rIf=1.0/(rIf+1.0)-rIf/((rIf+1.0)*(rIf+1.0));
    mIf_Vrth=mIf_rIf*rIf_Vrth;
    mIf_Vbei=mIf_rIf*rIf_Vbei;
    xvar1=(*Vbci)*IVTF/1.44;
    xvar1_Vbci=0.6944444*IVTF;
    xvar2=exp(xvar1);
    xvar2_xvar1=xvar2;
    xvar2_Vbci=xvar2_xvar1*xvar1_Vbci;
    tff=p[56]*(1.0+p[57]*q1)*(1.0+p[58]*xvar2*(slTF+mIf*mIf)*sgIf);
    tff_q1=p[57]*p[56]*(sgIf*(slTF+(mIf*mIf))*p[58]*xvar2+1.0);
    tff_xvar2=(q1*p[57]+1.0)*sgIf*(slTF+(mIf*mIf))*p[56]*p[58];
    tff_mIf=2.0*mIf*(q1*p[57]+1.0)*sgIf*p[56]*p[58]*xvar2;
    tff_Vrth=tff_q1*q1_Vrth;
    tff_Vbei=tff_q1*q1_Vbei;
    tff_Vbci=tff_q1*q1_Vbci;
    tff_Vbci=tff_Vbci+tff_xvar2*xvar2_Vbci;
    tff_Vrth=tff_Vrth+tff_mIf*mIf_Vrth;
    tff_Vbei=tff_Vbei+tff_mIf*mIf_Vbei;
    (*Qbe)=CJEatT*qdbe*p[32]+tff*Ifi/qb;
    Qbe_CJEatT=qdbe*p[32];
    Qbe_qdbe=CJEatT*p[32];
    Qbe_tff=Ifi/qb;
    Qbe_Ifi=tff/qb;
    Qbe_qb=-Ifi*tff/(qb*qb);
    *Qbe_Vrth=Qbe_CJEatT*CJEatT_Vrth;
    *Qbe_Vrth=(*Qbe_Vrth)+Qbe_qdbe*qdbe_Vrth;
    *Qbe_Vbei=Qbe_qdbe*qdbe_Vbei;
    *Qbe_Vrth=(*Qbe_Vrth)+Qbe_tff*tff_Vrth;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_tff*tff_Vbei;
    *Qbe_Vbci=Qbe_tff*tff_Vbci;
    *Qbe_Vrth=(*Qbe_Vrth)+Qbe_Ifi*Ifi_Vrth;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_Ifi*Ifi_Vbei;
    *Qbe_Vrth=(*Qbe_Vrth)+Qbe_qb*qb_Vrth;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_qb*qb_Vbei;
    *Qbe_Vbci=(*Qbe_Vbci)+Qbe_qb*qb_Vbci;
    (*Qbex)=CJEatT*qdbex*(1.0-p[32]);
    Qbex_CJEatT=qdbex*(1.0-p[32]);
    Qbex_qdbex=CJEatT*(1.0-p[32]);
    *Qbex_Vrth=Qbex_CJEatT*CJEatT_Vrth;
    *Qbex_Vrth=(*Qbex_Vrth)+Qbex_qdbex*qdbex_Vrth;
    *Qbex_Vbex=Qbex_qdbex*qdbex_Vbex;
    (*Qbc)=CJCatT*qdbc+p[61]*Iri+p[22]*Kbci;
    Qbc_CJCatT=qdbc;
    Qbc_qdbc=CJCatT;
    Qbc_Iri=p[61];
    Qbc_Kbci=p[22];
    *Qbc_Vrth=Qbc_CJCatT*CJCatT_Vrth;
    *Qbc_Vrth=(*Qbc_Vrth)+Qbc_qdbc*qdbc_Vrth;
    *Qbc_Vbci=Qbc_qdbc*qdbc_Vbci;
    *Qbc_Vrth=(*Qbc_Vrth)+Qbc_Iri*Iri_Vrth;
    *Qbc_Vbci=(*Qbc_Vbci)+Qbc_Iri*Iri_Vbci;
    *Qbc_Vrth=(*Qbc_Vrth)+Qbc_Kbci*Kbci_Vrth;
    *Qbc_Vbci=(*Qbc_Vbci)+Qbc_Kbci*Kbci_Vbci;
    (*Qbcx)=p[22]*Kbcx;
    Qbcx_Kbcx=p[22];
    *Qbcx_Vrth=Qbcx_Kbcx*Kbcx_Vrth;
    *Qbcx_Vbcx=Qbcx_Kbcx*Kbcx_Vbcx;
    (*Qbep)=CJEPatT*qdbep+p[61]*Ifp;
    Qbep_CJEPatT=qdbep;
    Qbep_qdbep=CJEPatT;
    Qbep_Ifp=p[61];
    *Qbep_Vrth=Qbep_CJEPatT*CJEPatT_Vrth;
    *Qbep_Vrth=(*Qbep_Vrth)+Qbep_qdbep*qdbep_Vrth;
    *Qbep_Vbep=Qbep_qdbep*qdbep_Vbep;
    *Qbep_Vrth=(*Qbep_Vrth)+Qbep_Ifp*Ifp_Vrth;
    *Qbep_Vbep=(*Qbep_Vbep)+Qbep_Ifp*Ifp_Vbep;
    *Qbep_Vbci=Qbep_Ifp*Ifp_Vbci;
    (*Qbcp)=CJCPatT*qdbcp+p[87]*(*Vbcp);
    Qbcp_CJCPatT=qdbcp;
    Qbcp_qdbcp=CJCPatT;
    *Qbcp_Vbcp=p[87];
    *Qbcp_Vrth=Qbcp_CJCPatT*CJCPatT_Vrth;
    *Qbcp_Vrth=(*Qbcp_Vrth)+Qbcp_qdbcp*qdbcp_Vrth;
    *Qbcp_Vbcp=(*Qbcp_Vbcp)+Qbcp_qdbcp*qdbcp_Vbcp;
    (*Qbeo)=(*Vbe)*p[15];
    *Qbeo_Vbe=p[15];
    (*Qbco)=(*Vbc)*p[20];
    *Qbco_Vbc=p[20];
    (*Ith)=-((*Ibe)*(*Vbei)+(*Ibc)*(*Vbci)+((*Itzf)-(*Itzr))*(*Vcei)+(*Ibex)*(*Vbex)+(*Ibep)*(*Vbep)+(*Irs)*(*Vrs)+(*Ibcp)*(*Vbcp)+(*Iccp)*(*Vcep)+(*Ircx)*(*Vrcx)+(*Irci)*(*Vrci)+(*Irbx)*(*Vrbx)+(*Irbi)*(*Vrbi)+(*Ire)*(*Vre)+(*Irbp)*(*Vrbp));
    Ith_Ibe=-(*Vbei);
    *Ith_Vbei=-(*Ibe);
    Ith_Ibc=-(*Vbci);
    *Ith_Vbci=-(*Ibc);
    Ith_Itzf=-(*Vcei);
    Ith_Itzr=(*Vcei);
    *Ith_Vcei=(*Itzr)-(*Itzf);
    Ith_Ibex=-(*Vbex);
    *Ith_Vbex=-(*Ibex);
    Ith_Ibep=-(*Vbep);
    *Ith_Vbep=-(*Ibep);
    Ith_Irs=-(*Vrs);
    *Ith_Vrs=-(*Irs);
    Ith_Ibcp=-(*Vbcp);
    *Ith_Vbcp=-(*Ibcp);
    Ith_Iccp=-(*Vcep);
    *Ith_Vcep=-(*Iccp);
    Ith_Ircx=-(*Vrcx);
    *Ith_Vrcx=-(*Ircx);
    Ith_Irci=-(*Vrci);
    *Ith_Vrci=-(*Irci);
    Ith_Irbx=-(*Vrbx);
    *Ith_Vrbx=-(*Irbx);
    Ith_Irbi=-(*Vrbi);
    *Ith_Vrbi=-(*Irbi);
    Ith_Ire=-(*Vre);
    *Ith_Vre=-(*Ire);
    Ith_Irbp=-(*Vrbp);
    *Ith_Vrbp=-(*Irbp);
    *Ith_Vrth=Ith_Ibe*(*Ibe_Vrth);
    *Ith_Vbei=(*Ith_Vbei)+Ith_Ibe*(*Ibe_Vbei);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ibc*(*Ibc_Vrth);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Ibc*(*Ibc_Vbci);
    *Ith_Vbei=(*Ith_Vbei)+Ith_Ibc*(*Ibc_Vbei);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Itzf*(*Itzf_Vrth);
    *Ith_Vbei=(*Ith_Vbei)+Ith_Itzf*(*Itzf_Vbei);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Itzf*(*Itzf_Vbci);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Itzr*(*Itzr_Vrth);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Itzr*(*Itzr_Vbci);
    *Ith_Vbei=(*Ith_Vbei)+Ith_Itzr*(*Itzr_Vbei);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ibex*(*Ibex_Vrth);
    *Ith_Vbex=(*Ith_Vbex)+Ith_Ibex*(*Ibex_Vbex);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ibep*(*Ibep_Vrth);
    *Ith_Vbep=(*Ith_Vbep)+Ith_Ibep*(*Ibep_Vbep);
    *Ith_Vrs=(*Ith_Vrs)+Ith_Irs*(*Irs_Vrs);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Irs*(*Irs_Vrth);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ibcp*(*Ibcp_Vrth);
    *Ith_Vbcp=(*Ith_Vbcp)+Ith_Ibcp*(*Ibcp_Vbcp);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Iccp*(*Iccp_Vrth);
    *Ith_Vbep=(*Ith_Vbep)+Ith_Iccp*(*Iccp_Vbep);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Iccp*(*Iccp_Vbci);
    *Ith_Vbcp=(*Ith_Vbcp)+Ith_Iccp*(*Iccp_Vbcp);
    *Ith_Vrcx=(*Ith_Vrcx)+Ith_Ircx*(*Ircx_Vrcx);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ircx*(*Ircx_Vrth);
    *Ith_Vrci=(*Ith_Vrci)+Ith_Irci*(*Irci_Vrci);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Irci*(*Irci_Vrth);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Irci*(*Irci_Vbci);
    *Ith_Vbcx=Ith_Irci*(*Irci_Vbcx);
    *Ith_Vrbx=(*Ith_Vrbx)+Ith_Irbx*(*Irbx_Vrbx);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Irbx*(*Irbx_Vrth);
    *Ith_Vrbi=(*Ith_Vrbi)+Ith_Irbi*(*Irbi_Vrbi);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Irbi*(*Irbi_Vrth);
    *Ith_Vbei=(*Ith_Vbei)+Ith_Irbi*(*Irbi_Vbei);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Irbi*(*Irbi_Vbci);
    *Ith_Vre=(*Ith_Vre)+Ith_Ire*(*Ire_Vre);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Ire*(*Ire_Vrth);
    *Ith_Vrbp=(*Ith_Vrbp)+Ith_Irbp*(*Irbp_Vrbp);
    *Ith_Vrth=(*Ith_Vrth)+Ith_Irbp*(*Irbp_Vrth);
    *Ith_Vbep=(*Ith_Vbep)+Ith_Irbp*(*Irbp_Vbep);
    *Ith_Vbci=(*Ith_Vbci)+Ith_Irbp*(*Irbp_Vbci);
    if(p[83]>0.0){
        (*Irth)=(*Vrth)/p[83];
        *Irth_Vrth=1.0/p[83];
    }else{
        (*Irth)=0.0;
        *Irth_Vrth=0.0;
    }
    (*Qcth)=(*Vrth)*p[84];
    *Qcth_Vrth=p[84];

/*  Scale outputs */

    if((*SCALE)!=1.0){
        *Ibe=(*SCALE)*(*Ibe);
        *Ibe_Vrth=(*SCALE)*(*Ibe_Vrth);
        *Ibe_Vbei=(*SCALE)*(*Ibe_Vbei);
        *Ibex=(*SCALE)*(*Ibex);
        *Ibex_Vrth=(*SCALE)*(*Ibex_Vrth);
        *Ibex_Vbex=(*SCALE)*(*Ibex_Vbex);
        *Itzf=(*SCALE)*(*Itzf);
        *Itzf_Vrth=(*SCALE)*(*Itzf_Vrth);
        *Itzf_Vbei=(*SCALE)*(*Itzf_Vbei);
        *Itzf_Vbci=(*SCALE)*(*Itzf_Vbci);
        *Itzr=(*SCALE)*(*Itzr);
        *Itzr_Vrth=(*SCALE)*(*Itzr_Vrth);
        *Itzr_Vbci=(*SCALE)*(*Itzr_Vbci);
        *Itzr_Vbei=(*SCALE)*(*Itzr_Vbei);
        *Ibc=(*SCALE)*(*Ibc);
        *Ibc_Vrth=(*SCALE)*(*Ibc_Vrth);
        *Ibc_Vbci=(*SCALE)*(*Ibc_Vbci);
        *Ibc_Vbei=(*SCALE)*(*Ibc_Vbei);
        *Ibep=(*SCALE)*(*Ibep);
        *Ibep_Vrth=(*SCALE)*(*Ibep_Vrth);
        *Ibep_Vbep=(*SCALE)*(*Ibep_Vbep);
        *Ircx=(*SCALE)*(*Ircx);
        *Ircx_Vrcx=(*SCALE)*(*Ircx_Vrcx);
        *Ircx_Vrth=(*SCALE)*(*Ircx_Vrth);
        *Irci=(*SCALE)*(*Irci);
        *Irci_Vrci=(*SCALE)*(*Irci_Vrci);
        *Irci_Vrth=(*SCALE)*(*Irci_Vrth);
        *Irci_Vbci=(*SCALE)*(*Irci_Vbci);
        *Irci_Vbcx=(*SCALE)*(*Irci_Vbcx);
        *Irbx=(*SCALE)*(*Irbx);
        *Irbx_Vrbx=(*SCALE)*(*Irbx_Vrbx);
        *Irbx_Vrth=(*SCALE)*(*Irbx_Vrth);
        *Irbi=(*SCALE)*(*Irbi);
        *Irbi_Vrbi=(*SCALE)*(*Irbi_Vrbi);
        *Irbi_Vrth=(*SCALE)*(*Irbi_Vrth);
        *Irbi_Vbei=(*SCALE)*(*Irbi_Vbei);
        *Irbi_Vbci=(*SCALE)*(*Irbi_Vbci);
        *Ire=(*SCALE)*(*Ire);
        *Ire_Vre=(*SCALE)*(*Ire_Vre);
        *Ire_Vrth=(*SCALE)*(*Ire_Vrth);
        *Irbp=(*SCALE)*(*Irbp);
        *Irbp_Vrbp=(*SCALE)*(*Irbp_Vrbp);
        *Irbp_Vrth=(*SCALE)*(*Irbp_Vrth);
        *Irbp_Vbep=(*SCALE)*(*Irbp_Vbep);
        *Irbp_Vbci=(*SCALE)*(*Irbp_Vbci);
        *Qbe=(*SCALE)*(*Qbe);
        *Qbe_Vrth=(*SCALE)*(*Qbe_Vrth);
        *Qbe_Vbei=(*SCALE)*(*Qbe_Vbei);
        *Qbe_Vbci=(*SCALE)*(*Qbe_Vbci);
        *Qbex=(*SCALE)*(*Qbex);
        *Qbex_Vrth=(*SCALE)*(*Qbex_Vrth);
        *Qbex_Vbex=(*SCALE)*(*Qbex_Vbex);
        *Qbc=(*SCALE)*(*Qbc);
        *Qbc_Vrth=(*SCALE)*(*Qbc_Vrth);
        *Qbc_Vbci=(*SCALE)*(*Qbc_Vbci);
        *Qbcx=(*SCALE)*(*Qbcx);
        *Qbcx_Vrth=(*SCALE)*(*Qbcx_Vrth);
        *Qbcx_Vbcx=(*SCALE)*(*Qbcx_Vbcx);
        *Qbep=(*SCALE)*(*Qbep);
        *Qbep_Vrth=(*SCALE)*(*Qbep_Vrth);
        *Qbep_Vbep=(*SCALE)*(*Qbep_Vbep);
        *Qbep_Vbci=(*SCALE)*(*Qbep_Vbci);
        *Qbeo=(*SCALE)*(*Qbeo);
        *Qbeo_Vbe=(*SCALE)*(*Qbeo_Vbe);
        *Qbco=(*SCALE)*(*Qbco);
        *Qbco_Vbc=(*SCALE)*(*Qbco_Vbc);
        *Ibcp=(*SCALE)*(*Ibcp);
        *Ibcp_Vrth=(*SCALE)*(*Ibcp_Vrth);
        *Ibcp_Vbcp=(*SCALE)*(*Ibcp_Vbcp);
        *Iccp=(*SCALE)*(*Iccp);
        *Iccp_Vrth=(*SCALE)*(*Iccp_Vrth);
        *Iccp_Vbep=(*SCALE)*(*Iccp_Vbep);
        *Iccp_Vbci=(*SCALE)*(*Iccp_Vbci);
        *Iccp_Vbcp=(*SCALE)*(*Iccp_Vbcp);
        *Irs=(*SCALE)*(*Irs);
        *Irs_Vrs=(*SCALE)*(*Irs_Vrs);
        *Irs_Vrth=(*SCALE)*(*Irs_Vrth);
        *Qbcp=(*SCALE)*(*Qbcp);
        *Qbcp_Vrth=(*SCALE)*(*Qbcp_Vrth);
        *Qbcp_Vbcp=(*SCALE)*(*Qbcp_Vbcp);
        *Irth=(*SCALE)*(*Irth);
        *Irth_Vrth=(*SCALE)*(*Irth_Vrth);
        *Ith=(*SCALE)*(*Ith);
        *Ith_Vrth=(*SCALE)*(*Ith_Vrth);
        *Ith_Vbei=(*SCALE)*(*Ith_Vbei);
        *Ith_Vbci=(*SCALE)*(*Ith_Vbci);
        *Ith_Vcei=(*SCALE)*(*Ith_Vcei);
        *Ith_Vbex=(*SCALE)*(*Ith_Vbex);
        *Ith_Vbep=(*SCALE)*(*Ith_Vbep);
        *Ith_Vrs=(*SCALE)*(*Ith_Vrs);
        *Ith_Vbcp=(*SCALE)*(*Ith_Vbcp);
        *Ith_Vcep=(*SCALE)*(*Ith_Vcep);
        *Ith_Vrcx=(*SCALE)*(*Ith_Vrcx);
        *Ith_Vrci=(*SCALE)*(*Ith_Vrci);
        *Ith_Vbcx=(*SCALE)*(*Ith_Vbcx);
        *Ith_Vrbx=(*SCALE)*(*Ith_Vrbx);
        *Ith_Vrbi=(*SCALE)*(*Ith_Vrbi);
        *Ith_Vre=(*SCALE)*(*Ith_Vre);
        *Ith_Vrbp=(*SCALE)*(*Ith_Vrbp);
        *Qcth=(*SCALE)*(*Qcth);
        *Qcth_Vrth=(*SCALE)*(*Qcth_Vrth);
    }
    return(0);
}
int vbic_4T_it_cf_fj(double *p
    ,double *Vbei, double *Vbex, double *Vbci, double *Vbep, double *Vbcp, double *Vrcx
    ,double *Vbcx, double *Vrci, double *Vrbx, double *Vrbi, double *Vre, double *Vrbp, double *Vrs
    ,double *Vbe, double *Vbc, double *Ibe, double *Ibe_Vbei, double *Ibex, double *Ibex_Vbex, double *Itzf
    ,double *Itzf_Vbei, double *Itzf_Vbci, double *Itzr, double *Itzr_Vbci, double *Itzr_Vbei, double *Ibc, double *Ibc_Vbci
    ,double *Ibc_Vbei, double *Ibep, double *Ibep_Vbep, double *Ircx, double *Ircx_Vrcx, double *Irci, double *Irci_Vrci
    ,double *Irci_Vbci, double *Irci_Vbcx, double *Irbx, double *Irbx_Vrbx, double *Irbi, double *Irbi_Vrbi, double *Irbi_Vbei
    ,double *Irbi_Vbci, double *Ire, double *Ire_Vre, double *Irbp, double *Irbp_Vrbp, double *Irbp_Vbep, double *Irbp_Vbci
    ,double *Qbe, double *Qbe_Vbei, double *Qbe_Vbci, double *Qbex, double *Qbex_Vbex, double *Qbc, double *Qbc_Vbci
    ,double *Qbcx, double *Qbcx_Vbcx, double *Qbep, double *Qbep_Vbep, double *Qbep_Vbci, double *Qbeo, double *Qbeo_Vbe
    ,double *Qbco, double *Qbco_Vbc, double *Ibcp, double *Ibcp_Vbcp, double *Iccp, double *Iccp_Vbep, double *Iccp_Vbci
    ,double *Iccp_Vbcp, double *Irs, double *Irs_Vrs, double *Qbcp, double *Qbcp_Vbcp, double *SCALE)
{
double  Vtv,IVEF,IVER,IIKF,IIKR,IIKP,IVO;
double  IHRCF,IVTF,IITF,slTF,dv0,dvh,dvh_Vbei;
double  xvar1,xvar2,pwq,qlo,qlo_Vbei,qhi,qhi_dvh;
double  qhi_Vbei,xvar1_Vbei,xvar3,xvar3_xvar1,xvar3_Vbei,qlo_xvar3,qdbe;
double  qdbe_qlo,qdbe_Vbei,qdbe_qhi,mv0,vl0,q0,dv;
double  dv_Vbei,mv,mv_dv,mv_Vbei,vl,vl_dv,vl_Vbei;
double  vl_mv,xvar1_vl,qdbe_vl,dvh_Vbex,qlo_Vbex,qhi_Vbex,xvar1_Vbex;
double  xvar3_Vbex,qdbex,qdbex_qlo,qdbex_Vbex,qdbex_qhi,dv_Vbex,mv_Vbex;
double  vl_Vbex,qdbex_vl,dvh_Vbci,qlo_Vbci,qhi_Vbci,xvar1_Vbci,xvar3_Vbci;
double  qdbc,qdbc_qlo,qdbc_Vbci,qdbc_qhi,vn0,vnl0,qlo0;
double  vn,vn_Vbci,vnl,vnl_vn,vnl_Vbci,vl_vnl,vl_Vbci;
double  sel,sel_vnl,sel_Vbci,crt,cmx,cl,cl_sel;
double  cl_Vbci,ql,ql_Vbci,ql_vl,ql_cl,qdbc_ql,dv_Vbci;
double  mv_Vbci,qdbc_vl,dvh_Vbep,qlo_Vbep,qhi_Vbep,xvar1_Vbep,xvar3_Vbep;
double  qdbep,qdbep_qlo,qdbep_Vbep,qdbep_qhi,vn_Vbep,vnl_Vbep,vl_Vbep;
double  sel_Vbep,cl_Vbep,ql_Vbep,qdbep_ql,dv_Vbep,mv_Vbep,qdbep_vl;
double  dvh_Vbcp,qlo_Vbcp,qhi_Vbcp,xvar1_Vbcp,xvar3_Vbcp,qdbcp,qdbcp_qlo;
double  qdbcp_Vbcp,qdbcp_Vbep,qdbcp_qhi,dv_Vbcp,mv_Vbcp,vl_Vbcp,qdbcp_vl;
double  argi,argi_Vbei,expi,expi_argi,expi_Vbei,Ifi,Ifi_expi;
double  Ifi_Vbei,argi_Vbci,expi_Vbci,Iri,Iri_expi,Iri_Vbci,q1z;
double  q1z_qdbe,q1z_Vbei,q1z_qdbc,q1z_Vbci,q1,q1_q1z,q1_Vbei;
double  q1_Vbci,q2,q2_Ifi,q2_Vbei,q2_Iri,q2_Vbci,xvar3_q1;
double  xvar1_xvar3,xvar1_q2,xvar4,xvar4_xvar1,xvar4_Vbei,xvar4_Vbci,qb;
double  qb_q1,qb_Vbei,qb_Vbci,qb_xvar4,xvar2_xvar1,xvar2_Vbei,xvar2_Vbci;
double  qb_xvar2,Itzr_Iri,Itzr_qb,Itzf_Ifi,Itzf_qb,argi_Vbep,expi_Vbep;
double  argx,argx_Vbci,expx,expx_argx,expx_Vbci,Ifp,Ifp_expi;
double  Ifp_Vbep,Ifp_expx,Ifp_Vbci,q2p,q2p_Ifp,q2p_Vbep,q2p_Vbci;
double  qbp,qbp_q2p,qbp_Vbep,qbp_Vbci,argi_Vbcp,expi_Vbcp,Irp;
double  Irp_expi,Irp_Vbcp,Iccp_Ifp,Iccp_Irp,Iccp_qbp,argn,argn_Vbei;
double  expn,expn_argn,expn_Vbei,argx_Vbei,expx_Vbei,Ibe_expi,Ibe_expn;
double  Ibe_expx,argi_Vbex,expi_Vbex,argn_Vbex,expn_Vbex,argx_Vbex,expx_Vbex;
double  Ibex_expi,Ibex_expn,Ibex_expx,argn_Vbci,expn_Vbci,Ibcj,Ibcj_expi;
double  Ibcj_Vbci,Ibcj_expn,argn_Vbep,expn_Vbep,Ibep_expi,Ibep_expn,xvar3_vl;
double  avalf,avalf_vl,avalf_Vbci,avalf_xvar4,Igc,Igc_Itzf,Igc_Vbei;
double  Igc_Vbci,Igc_Itzr,Igc_Ibcj,Igc_avalf,Ibc_Ibcj,Ibc_Igc,argx_Vbcx;
double  expx_Vbcx,Kbci,Kbci_expi,Kbci_Vbci,Kbcx,Kbcx_expx,Kbcx_Vbcx;
double  rKp1,rKp1_Kbci,rKp1_Vbci,rKp1_Kbcx,rKp1_Vbcx,xvar1_rKp1,xvar1_Vbcx;
double  Iohm,Iohm_Vrci,Iohm_Kbci,Iohm_Vbci,Iohm_Kbcx,Iohm_Vbcx,Iohm_xvar1;
double  derf,derf_Iohm,derf_Vrci,derf_Vbci,derf_Vbcx,Irci_Iohm,Irci_derf;
double  Irbi_qb,Irbp_qbp,argn_Vbcp,expn_Vbcp,Ibcp_expi,Ibcp_expn,sgIf;
double  rIf,rIf_Ifi,rIf_Vbei,mIf,mIf_rIf,mIf_Vbei,tff;
double  tff_q1,tff_Vbei,tff_Vbci,tff_xvar2,tff_mIf,Qbe_qdbe,Qbe_tff;
double  Qbe_Ifi,Qbe_qb,Qbex_qdbex,Qbc_qdbc,Qbc_Iri,Qbc_Kbci,Qbcx_Kbcx;
double  Qbep_qdbep,Qbep_Ifp,Qbcp_qdbcp;

/*  Function and derivative code */

    Vtv=1.380662e-23*(2.731500e+02+p[0])/1.602189e-19;
    if(p[51]>0.0){
        IVEF=1.0/p[51];
    }else{
        IVEF=0.0;
    }
    if(p[52]>0.0){
        IVER=1.0/p[52];
    }else{
        IVER=0.0;
    }
    if(p[53]>0.0){
        IIKF=1.0/p[53];
    }else{
        IIKF=0.0;
    }
    if(p[54]>0.0){
        IIKR=1.0/p[54];
    }else{
        IIKR=0.0;
    }
    if(p[55]>0.0){
        IIKP=1.0/p[55];
    }else{
        IIKP=0.0;
    }
    if(p[3]>0.0){
        IVO=1.0/p[3];
    }else{
        IVO=0.0;
    }
    if(p[5]>0.0){
        IHRCF=1.0/p[5];
    }else{
        IHRCF=0.0;
    }
    if(p[59]>0.0){
        IVTF=1.0/p[59];
    }else{
        IVTF=0.0;
    }
    if(p[60]>0.0){
        IITF=1.0/p[60];
    }else{
        IITF=0.0;
    }
    if(p[60]>0.0){
        slTF=0.0;
    }else{
        slTF=1.0;
    }
    dv0=-p[17]*p[14];
    if(p[19]<=0.0){
        dvh=(*Vbei)+dv0;
        dvh_Vbei=1.0;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[18]);
            pwq=pow(xvar1,xvar2);
            qlo=p[17]*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[18]);
            qlo_Vbei=0.0;
            qhi=dvh*(1.0-p[14]+0.5*p[18]*dvh/p[17])*pwq;
            qhi_dvh=(0.5*dvh*p[18]/p[17]-p[14]+1.0)*pwq+0.5*dvh*p[18]*pwq/p[17];
            qhi_Vbei=qhi_dvh*dvh_Vbei;
        }else{
            xvar1=(1.0-(*Vbei)/p[17]);
            xvar1_Vbei=-1.0/p[17];
            xvar2=(1.0-p[18]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbei=xvar3_xvar1*xvar1_Vbei;
            qlo=p[17]*(1.0-xvar3)/(1.0-p[18]);
            qlo_xvar3=-p[17]/(1.0-p[18]);
            qlo_Vbei=qlo_xvar3*xvar3_Vbei;
            qhi=0.0;
            qhi_Vbei=0.0;
        }
        qdbe=qlo+qhi;
        qdbe_qlo=1.0;
        qdbe_qhi=1.0;
        qdbe_Vbei=qdbe_qlo*qlo_Vbei;
        qdbe_Vbei=qdbe_Vbei+qdbe_qhi*qhi_Vbei;
    }else{
        mv0=sqrt(dv0*dv0+4.0*p[19]*p[19]);
        vl0=-0.5*(dv0+mv0);
        xvar1=(1.0-vl0/p[17]);
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        q0=-p[17]*xvar3/(1.0-p[18]);
        dv=(*Vbei)+dv0;
        dv_Vbei=1.0;
        mv=sqrt(dv*dv+4.0*p[19]*p[19]);
        mv_dv=dv/sqrt((dv*dv)+4.0*(p[19]*p[19]));
        mv_Vbei=mv_dv*dv_Vbei;
        vl=0.5*(dv-mv)-dv0;
        vl_dv=0.5;
        vl_mv=-0.5;
        vl_Vbei=vl_dv*dv_Vbei;
        vl_Vbei=vl_Vbei+vl_mv*mv_Vbei;
        xvar1=(1.0-vl/p[17]);
        xvar1_vl=-1.0/p[17];
        xvar1_Vbei=xvar1_vl*vl_Vbei;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vbei=xvar3_xvar1*xvar1_Vbei;
        qlo=-p[17]*xvar3/(1.0-p[18]);
        qlo_xvar3=-p[17]/(1.0-p[18]);
        qlo_Vbei=qlo_xvar3*xvar3_Vbei;
        xvar1=(1.0-p[14]);
        xvar2=(-p[18]);
        xvar3=pow(xvar1,xvar2);
        qdbe=qlo+xvar3*((*Vbei)-vl+vl0)-q0;
        qdbe_qlo=1.0;
        qdbe_Vbei=xvar3;
        qdbe_vl=-xvar3;
        qdbe_Vbei=qdbe_Vbei+qdbe_qlo*qlo_Vbei;
        qdbe_Vbei=qdbe_Vbei+qdbe_vl*vl_Vbei;
    }
    dv0=-p[17]*p[14];
    if(p[19]<=0.0){
        dvh=(*Vbex)+dv0;
        dvh_Vbex=1.0;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[18]);
            pwq=pow(xvar1,xvar2);
            qlo=p[17]*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[18]);
            qlo_Vbex=0.0;
            qhi=dvh*(1.0-p[14]+0.5*p[18]*dvh/p[17])*pwq;
            qhi_dvh=(0.5*dvh*p[18]/p[17]-p[14]+1.0)*pwq+0.5*dvh*p[18]*pwq/p[17];
            qhi_Vbex=qhi_dvh*dvh_Vbex;
        }else{
            xvar1=(1.0-(*Vbex)/p[17]);
            xvar1_Vbex=-1.0/p[17];
            xvar2=(1.0-p[18]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbex=xvar3_xvar1*xvar1_Vbex;
            qlo=p[17]*(1.0-xvar3)/(1.0-p[18]);
            qlo_xvar3=-p[17]/(1.0-p[18]);
            qlo_Vbex=qlo_xvar3*xvar3_Vbex;
            qhi=0.0;
            qhi_Vbex=0.0;
        }
        qdbex=qlo+qhi;
        qdbex_qlo=1.0;
        qdbex_qhi=1.0;
        qdbex_Vbex=qdbex_qlo*qlo_Vbex;
        qdbex_Vbex=qdbex_Vbex+qdbex_qhi*qhi_Vbex;
    }else{
        mv0=sqrt(dv0*dv0+4.0*p[19]*p[19]);
        vl0=-0.5*(dv0+mv0);
        xvar1=(1.0-vl0/p[17]);
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        q0=-p[17]*xvar3/(1.0-p[18]);
        dv=(*Vbex)+dv0;
        dv_Vbex=1.0;
        mv=sqrt(dv*dv+4.0*p[19]*p[19]);
        mv_dv=dv/sqrt((dv*dv)+4.0*(p[19]*p[19]));
        mv_Vbex=mv_dv*dv_Vbex;
        vl=0.5*(dv-mv)-dv0;
        vl_dv=0.5;
        vl_mv=-0.5;
        vl_Vbex=vl_dv*dv_Vbex;
        vl_Vbex=vl_Vbex+vl_mv*mv_Vbex;
        xvar1=(1.0-vl/p[17]);
        xvar1_vl=-1.0/p[17];
        xvar1_Vbex=xvar1_vl*vl_Vbex;
        xvar2=(1.0-p[18]);
        xvar3=pow(xvar1,xvar2);
        xvar3_xvar1=xvar3*xvar2/xvar1;
        xvar3_Vbex=xvar3_xvar1*xvar1_Vbex;
        qlo=-p[17]*xvar3/(1.0-p[18]);
        qlo_xvar3=-p[17]/(1.0-p[18]);
        qlo_Vbex=qlo_xvar3*xvar3_Vbex;
        xvar1=(1.0-p[14]);
        xvar2=(-p[18]);
        xvar3=pow(xvar1,xvar2);
        qdbex=qlo+xvar3*((*Vbex)-vl+vl0)-q0;
        qdbex_qlo=1.0;
        qdbex_Vbex=xvar3;
        qdbex_vl=-xvar3;
        qdbex_Vbex=qdbex_Vbex+qdbex_qlo*qlo_Vbex;
        qdbex_Vbex=qdbex_Vbex+qdbex_vl*vl_Vbex;
    }
    dv0=-p[24]*p[14];
    if(p[26]<=0.0){
        dvh=(*Vbci)+dv0;
        dvh_Vbci=1.0;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[25]);
            pwq=pow(xvar1,xvar2);
            qlo=p[24]*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[25]);
            qlo_Vbci=0.0;
            qhi=dvh*(1.0-p[14]+0.5*p[25]*dvh/p[24])*pwq;
            qhi_dvh=(0.5*dvh*p[25]/p[24]-p[14]+1.0)*pwq+0.5*dvh*p[25]*pwq/p[24];
            qhi_Vbci=qhi_dvh*dvh_Vbci;
        }else{
            if((p[85]>0.0)&&((*Vbci)<-p[85])){
                xvar1=(1.0+p[85]/p[24]);
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                qlo=p[24]*(1.0-xvar3*(1.0-((1.0-p[25])*((*Vbci)+p[85]))/(p[24]+p[85])))/(1.0-p[25]);
                qlo_Vbci=p[24]*xvar3/(p[85]+p[24]);
            }else{
                xvar1=(1.0-(*Vbci)/p[24]);
                xvar1_Vbci=-1.0/p[24];
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
                qlo=p[24]*(1.0-xvar3)/(1.0-p[25]);
                qlo_xvar3=-p[24]/(1.0-p[25]);
                qlo_Vbci=qlo_xvar3*xvar3_Vbci;
            }
            qhi=0.0;
            qhi_Vbci=0.0;
        }
        qdbc=qlo+qhi;
        qdbc_qlo=1.0;
        qdbc_qhi=1.0;
        qdbc_Vbci=qdbc_qlo*qlo_Vbci;
        qdbc_Vbci=qdbc_Vbci+qdbc_qhi*qhi_Vbci;
    }else{
        if((p[85]>0.0)&&(p[86]>0.0)){
            vn0=(p[85]+dv0)/(p[85]-dv0);
            vnl0=2.0*vn0/(sqrt((vn0-1.0)*(vn0-1.0)+4.0*p[26]*p[26])+sqrt((vn0+1.0)*(vn0+1.0)+4.0*p[86]*p[86]));
            vl0=0.5*(vnl0*(p[85]-dv0)-p[85]-dv0);
            xvar1=(1.0-vl0/p[24]);
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            qlo0=p[24]*(1.0-xvar3)/(1.0-p[25]);
            vn=(2.0*(*Vbci)+p[85]+dv0)/(p[85]-dv0);
            vn_Vbci=2.0/(p[85]-dv0);
            vnl=2.0*vn/(sqrt((vn-1.0)*(vn-1.0)+4.0*p[26]*p[26])+sqrt((vn+1.0)*(vn+1.0)+4.0*p[86]*p[86]));
            vnl_vn=2.0/(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))-2.0*vn*((vn+1.0)/sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+(vn-1.0)/sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26]))));
            vnl_Vbci=vnl_vn*vn_Vbci;
            vl=0.5*(vnl*(p[85]-dv0)-p[85]-dv0);
            vl_vnl=0.5*(p[85]-dv0);
            vl_Vbci=vl_vnl*vnl_Vbci;
            xvar1=(1.0-vl/p[24]);
            xvar1_vl=-1.0/p[24];
            xvar1_Vbci=xvar1_vl*vl_Vbci;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
            qlo=p[24]*(1.0-xvar3)/(1.0-p[25]);
            qlo_xvar3=-p[24]/(1.0-p[25]);
            qlo_Vbci=qlo_xvar3*xvar3_Vbci;
            sel=0.5*(vnl+1.0);
            sel_vnl=0.5;
            sel_Vbci=sel_vnl*vnl_Vbci;
            xvar1=(1.0+p[85]/p[24]);
            xvar2=(-p[25]);
            crt=pow(xvar1,xvar2);
            xvar1=(1.0+dv0/p[24]);
            xvar2=(-p[25]);
            cmx=pow(xvar1,xvar2);
            cl=(1.0-sel)*crt+sel*cmx;
            cl_sel=cmx-crt;
            cl_Vbci=cl_sel*sel_Vbci;
            ql=((*Vbci)-vl+vl0)*cl;
            ql_Vbci=cl;
            ql_vl=-cl;
            ql_cl=vl0-vl+(*Vbci);
            ql_Vbci=ql_Vbci+ql_vl*vl_Vbci;
            ql_Vbci=ql_Vbci+ql_cl*cl_Vbci;
            qdbc=ql+qlo-qlo0;
            qdbc_ql=1.0;
            qdbc_qlo=1.0;
            qdbc_Vbci=qdbc_ql*ql_Vbci;
            qdbc_Vbci=qdbc_Vbci+qdbc_qlo*qlo_Vbci;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[26]*p[26]);
            vl0=-0.5*(dv0+mv0);
            xvar1=(1.0-vl0/p[24]);
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            q0=-p[24]*xvar3/(1.0-p[25]);
            dv=(*Vbci)+dv0;
            dv_Vbci=1.0;
            mv=sqrt(dv*dv+4.0*p[26]*p[26]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[26]*p[26]));
            mv_Vbci=mv_dv*dv_Vbci;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_Vbci=vl_dv*dv_Vbci;
            vl_Vbci=vl_Vbci+vl_mv*mv_Vbci;
            xvar1=(1.0-vl/p[24]);
            xvar1_vl=-1.0/p[24];
            xvar1_Vbci=xvar1_vl*vl_Vbci;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbci=xvar3_xvar1*xvar1_Vbci;
            qlo=-p[24]*xvar3/(1.0-p[25]);
            qlo_xvar3=-p[24]/(1.0-p[25]);
            qlo_Vbci=qlo_xvar3*xvar3_Vbci;
            xvar1=(1.0-p[14]);
            xvar2=(-p[25]);
            xvar3=pow(xvar1,xvar2);
            qdbc=qlo+xvar3*((*Vbci)-vl+vl0)-q0;
            qdbc_qlo=1.0;
            qdbc_Vbci=xvar3;
            qdbc_vl=-xvar3;
            qdbc_Vbci=qdbc_Vbci+qdbc_qlo*qlo_Vbci;
            qdbc_Vbci=qdbc_Vbci+qdbc_vl*vl_Vbci;
        }
    }
    dv0=-p[24]*p[14];
    if(p[26]<=0.0){
        dvh=(*Vbep)+dv0;
        dvh_Vbep=1.0;
        if(dvh>0.0){
            xvar1=(1.0-p[14]);
            xvar2=(-1.0-p[25]);
            pwq=pow(xvar1,xvar2);
            qlo=p[24]*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[25]);
            qlo_Vbep=0.0;
            qhi=dvh*(1.0-p[14]+0.5*p[25]*dvh/p[24])*pwq;
            qhi_dvh=(0.5*dvh*p[25]/p[24]-p[14]+1.0)*pwq+0.5*dvh*p[25]*pwq/p[24];
            qhi_Vbep=qhi_dvh*dvh_Vbep;
        }else{
            if((p[85]>0.0)&&((*Vbep)<-p[85])){
                xvar1=(1.0+p[85]/p[24]);
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                qlo=p[24]*(1.0-xvar3*(1.0-((1.0-p[25])*((*Vbep)+p[85]))/(p[24]+p[85])))/(1.0-p[25]);
                qlo_Vbep=p[24]*xvar3/(p[85]+p[24]);
            }else{
                xvar1=(1.0-(*Vbep)/p[24]);
                xvar1_Vbep=-1.0/p[24];
                xvar2=(1.0-p[25]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
                qlo=p[24]*(1.0-xvar3)/(1.0-p[25]);
                qlo_xvar3=-p[24]/(1.0-p[25]);
                qlo_Vbep=qlo_xvar3*xvar3_Vbep;
            }
            qhi=0.0;
            qhi_Vbep=0.0;
        }
        qdbep=qlo+qhi;
        qdbep_qlo=1.0;
        qdbep_qhi=1.0;
        qdbep_Vbep=qdbep_qlo*qlo_Vbep;
        qdbep_Vbep=qdbep_Vbep+qdbep_qhi*qhi_Vbep;
    }else{
        if((p[85]>0.0)&&(p[86]>0.0)){
            vn0=(p[85]+dv0)/(p[85]-dv0);
            vnl0=2.0*vn0/(sqrt((vn0-1.0)*(vn0-1.0)+4.0*p[26]*p[26])+sqrt((vn0+1.0)*(vn0+1.0)+4.0*p[86]*p[86]));
            vl0=0.5*(vnl0*(p[85]-dv0)-p[85]-dv0);
            xvar1=(1.0-vl0/p[24]);
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            qlo0=p[24]*(1.0-xvar3)/(1.0-p[25]);
            vn=(2.0*(*Vbep)+p[85]+dv0)/(p[85]-dv0);
            vn_Vbep=2.0/(p[85]-dv0);
            vnl=2.0*vn/(sqrt((vn-1.0)*(vn-1.0)+4.0*p[26]*p[26])+sqrt((vn+1.0)*(vn+1.0)+4.0*p[86]*p[86]));
            vnl_vn=2.0/(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))-2.0*vn*((vn+1.0)/sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+(vn-1.0)/sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))/((sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26])))*(sqrt(((vn+1.0)*(vn+1.0))+4.0*(p[86]*p[86]))+sqrt(((vn-1.0)*(vn-1.0))+4.0*(p[26]*p[26]))));
            vnl_Vbep=vnl_vn*vn_Vbep;
            vl=0.5*(vnl*(p[85]-dv0)-p[85]-dv0);
            vl_vnl=0.5*(p[85]-dv0);
            vl_Vbep=vl_vnl*vnl_Vbep;
            xvar1=(1.0-vl/p[24]);
            xvar1_vl=-1.0/p[24];
            xvar1_Vbep=xvar1_vl*vl_Vbep;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
            qlo=p[24]*(1.0-xvar3)/(1.0-p[25]);
            qlo_xvar3=-p[24]/(1.0-p[25]);
            qlo_Vbep=qlo_xvar3*xvar3_Vbep;
            sel=0.5*(vnl+1.0);
            sel_vnl=0.5;
            sel_Vbep=sel_vnl*vnl_Vbep;
            xvar1=(1.0+p[85]/p[24]);
            xvar2=(-p[25]);
            crt=pow(xvar1,xvar2);
            xvar1=(1.0+dv0/p[24]);
            xvar2=(-p[25]);
            cmx=pow(xvar1,xvar2);
            cl=(1.0-sel)*crt+sel*cmx;
            cl_sel=cmx-crt;
            cl_Vbep=cl_sel*sel_Vbep;
            ql=((*Vbep)-vl+vl0)*cl;
            ql_Vbep=cl;
            ql_vl=-cl;
            ql_cl=vl0-vl+(*Vbep);
            ql_Vbep=ql_Vbep+ql_vl*vl_Vbep;
            ql_Vbep=ql_Vbep+ql_cl*cl_Vbep;
            qdbep=ql+qlo-qlo0;
            qdbep_ql=1.0;
            qdbep_qlo=1.0;
            qdbep_Vbep=qdbep_ql*ql_Vbep;
            qdbep_Vbep=qdbep_Vbep+qdbep_qlo*qlo_Vbep;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[26]*p[26]);
            vl0=-0.5*(dv0+mv0);
            xvar1=(1.0-vl0/p[24]);
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            q0=-p[24]*xvar3/(1.0-p[25]);
            dv=(*Vbep)+dv0;
            dv_Vbep=1.0;
            mv=sqrt(dv*dv+4.0*p[26]*p[26]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[26]*p[26]));
            mv_Vbep=mv_dv*dv_Vbep;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_Vbep=vl_dv*dv_Vbep;
            vl_Vbep=vl_Vbep+vl_mv*mv_Vbep;
            xvar1=(1.0-vl/p[24]);
            xvar1_vl=-1.0/p[24];
            xvar1_Vbep=xvar1_vl*vl_Vbep;
            xvar2=(1.0-p[25]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbep=xvar3_xvar1*xvar1_Vbep;
            qlo=-p[24]*xvar3/(1.0-p[25]);
            qlo_xvar3=-p[24]/(1.0-p[25]);
            qlo_Vbep=qlo_xvar3*xvar3_Vbep;
            xvar1=(1.0-p[14]);
            xvar2=(-p[25]);
            xvar3=pow(xvar1,xvar2);
            qdbep=qlo+xvar3*((*Vbep)-vl+vl0)-q0;
            qdbep_qlo=1.0;
            qdbep_Vbep=xvar3;
            qdbep_vl=-xvar3;
            qdbep_Vbep=qdbep_Vbep+qdbep_qlo*qlo_Vbep;
            qdbep_Vbep=qdbep_Vbep+qdbep_vl*vl_Vbep;
        }
    }
    if(p[27]>0.0){
        dv0=-p[28]*p[14];
        if(p[30]<=0.0){
            dvh=(*Vbcp)+dv0;
            dvh_Vbcp=1.0;
            if(dvh>0.0){
                xvar1=(1.0-p[14]);
                xvar2=(-1.0-p[29]);
                pwq=pow(xvar1,xvar2);
                qlo=p[28]*(1.0-pwq*(1.0-p[14])*(1.0-p[14]))/(1.0-p[29]);
                qlo_Vbep=0.0;
                qlo_Vbcp=0.0;
                qhi=dvh*(1.0-p[14]+0.5*p[29]*dvh/p[28])*pwq;
                qhi_dvh=(0.5*dvh*p[29]/p[28]-p[14]+1.0)*pwq+0.5*dvh*p[29]*pwq/p[28];
                qhi_Vbep=0.0;
                qhi_Vbcp=qhi_dvh*dvh_Vbcp;
            }else{
                xvar1=(1.0-(*Vbcp)/p[28]);
                xvar1_Vbcp=-1.0/p[28];
                xvar2=(1.0-p[29]);
                xvar3=pow(xvar1,xvar2);
                xvar3_xvar1=xvar3*xvar2/xvar1;
                xvar3_Vbcp=xvar3_xvar1*xvar1_Vbcp;
                qlo=p[28]*(1.0-xvar3)/(1.0-p[29]);
                qlo_xvar3=-p[28]/(1.0-p[29]);
                qlo_Vbep=0.0;
                qlo_Vbcp=qlo_xvar3*xvar3_Vbcp;
                qhi=0.0;
                qhi_Vbep=0.0;
                qhi_Vbcp=0.0;
            }
            qdbcp=qlo+qhi;
            qdbcp_qlo=1.0;
            qdbcp_qhi=1.0;
            qdbcp_Vbcp=qdbcp_qlo*qlo_Vbcp;
            qdbcp_Vbep=qdbcp_qlo*qlo_Vbep;
            qdbcp_Vbep=qdbcp_Vbep+qdbcp_qhi*qhi_Vbep;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_qhi*qhi_Vbcp;
        }else{
            mv0=sqrt(dv0*dv0+4.0*p[30]*p[30]);
            vl0=-0.5*(dv0+mv0);
            xvar1=(1.0-vl0/p[28]);
            xvar2=(1.0-p[29]);
            xvar3=pow(xvar1,xvar2);
            q0=-p[28]*xvar3/(1.0-p[29]);
            dv=(*Vbcp)+dv0;
            dv_Vbcp=1.0;
            mv=sqrt(dv*dv+4.0*p[30]*p[30]);
            mv_dv=dv/sqrt((dv*dv)+4.0*(p[30]*p[30]));
            mv_Vbcp=mv_dv*dv_Vbcp;
            vl=0.5*(dv-mv)-dv0;
            vl_dv=0.5;
            vl_mv=-0.5;
            vl_Vbcp=vl_dv*dv_Vbcp;
            vl_Vbcp=vl_Vbcp+vl_mv*mv_Vbcp;
            xvar1=(1.0-vl/p[28]);
            xvar1_vl=-1.0/p[28];
            xvar1_Vbcp=xvar1_vl*vl_Vbcp;
            xvar2=(1.0-p[29]);
            xvar3=pow(xvar1,xvar2);
            xvar3_xvar1=xvar3*xvar2/xvar1;
            xvar3_Vbcp=xvar3_xvar1*xvar1_Vbcp;
            qlo=-p[28]*xvar3/(1.0-p[29]);
            qlo_xvar3=-p[28]/(1.0-p[29]);
            qlo_Vbep=0.0;
            qlo_Vbcp=qlo_xvar3*xvar3_Vbcp;
            xvar1=(1.0-p[14]);
            xvar2=(-p[29]);
            xvar3=pow(xvar1,xvar2);
            qdbcp=qlo+xvar3*((*Vbcp)-vl+vl0)-q0;
            qdbcp_qlo=1.0;
            qdbcp_Vbcp=xvar3;
            qdbcp_vl=-xvar3;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_qlo*qlo_Vbcp;
            qdbcp_Vbep=qdbcp_qlo*qlo_Vbep;
            qdbcp_Vbcp=qdbcp_Vbcp+qdbcp_vl*vl_Vbcp;
        }
    }else{
        qdbcp=0.0;
        qdbcp_Vbcp=0.0;
    }
    argi=(*Vbei)/(p[12]*Vtv);
    argi_Vbei=1.0/(p[12]*Vtv);
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbei=expi_argi*argi_Vbei;
    Ifi=p[11]*(expi-1.0);
    Ifi_expi=p[11];
    Ifi_Vbei=Ifi_expi*expi_Vbei;
    argi=(*Vbci)/(p[13]*Vtv);
    argi_Vbci=1.0/(p[13]*Vtv);
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    Iri=p[11]*p[94]*(expi-1.0);
    Iri_expi=p[11]*p[94];
    Iri_Vbci=Iri_expi*expi_Vbci;
    q1z=1.0+qdbe*IVER+qdbc*IVEF;
    q1z_qdbe=IVER;
    q1z_qdbc=IVEF;
    q1z_Vbei=q1z_qdbe*qdbe_Vbei;
    q1z_Vbci=q1z_qdbc*qdbc_Vbci;
    q1=0.5*(sqrt((q1z-1.0e-4)*(q1z-1.0e-4)+1.0e-8)+q1z-1.0e-4)+1.0e-4;
    q1_q1z=0.5*((q1z-1.0e-4)/sqrt(((q1z-1.0e-4)*(q1z-1.0e-4))+1.0e-8)+1.0);
    q1_Vbei=q1_q1z*q1z_Vbei;
    q1_Vbci=q1_q1z*q1z_Vbci;
    q2=Ifi*IIKF+Iri*IIKR;
    q2_Ifi=IIKF;
    q2_Iri=IIKR;
    q2_Vbei=q2_Ifi*Ifi_Vbei;
    q2_Vbci=q2_Iri*Iri_Vbci;
    if(p[88]<0.5){
        xvar2=1.0/p[89];
        xvar3=pow(q1,xvar2);
        xvar3_q1=xvar3*xvar2/q1;
        xvar3_Vbei=xvar3_q1*q1_Vbei;
        xvar3_Vbci=xvar3_q1*q1_Vbci;
        xvar1=(xvar3+4.0*q2);
        xvar1_xvar3=1.0;
        xvar1_q2=4.0;
        xvar1_Vbei=xvar1_xvar3*xvar3_Vbei;
        xvar1_Vbci=xvar1_xvar3*xvar3_Vbci;
        xvar1_Vbei=xvar1_Vbei+xvar1_q2*q2_Vbei;
        xvar1_Vbci=xvar1_Vbci+xvar1_q2*q2_Vbci;
        xvar4=pow(xvar1,p[89]);
        xvar4_xvar1=xvar4*p[89]/xvar1;
        xvar4_Vbei=xvar4_xvar1*xvar1_Vbei;
        xvar4_Vbci=xvar4_xvar1*xvar1_Vbci;
        qb=0.5*(q1+xvar4);
        qb_q1=0.5;
        qb_xvar4=0.5;
        qb_Vbei=qb_q1*q1_Vbei;
        qb_Vbci=qb_q1*q1_Vbci;
        qb_Vbei=qb_Vbei+qb_xvar4*xvar4_Vbei;
        qb_Vbci=qb_Vbci+qb_xvar4*xvar4_Vbci;
    }else{
        xvar1=(1.0+4.0*q2);
        xvar1_q2=4.0;
        xvar1_Vbei=xvar1_q2*q2_Vbei;
        xvar1_Vbci=xvar1_q2*q2_Vbci;
        xvar2=pow(xvar1,p[89]);
        xvar2_xvar1=xvar2*p[89]/xvar1;
        xvar2_Vbei=xvar2_xvar1*xvar1_Vbei;
        xvar2_Vbci=xvar2_xvar1*xvar1_Vbci;
        qb=0.5*q1*(1.0+xvar2);
        qb_q1=0.5*(xvar2+1.0);
        qb_xvar2=0.5*q1;
        qb_Vbei=qb_q1*q1_Vbei;
        qb_Vbci=qb_q1*q1_Vbci;
        qb_Vbei=qb_Vbei+qb_xvar2*xvar2_Vbei;
        qb_Vbci=qb_Vbci+qb_xvar2*xvar2_Vbci;
    }
    (*Itzr)=Iri/qb;
    Itzr_Iri=1.0/qb;
    Itzr_qb=-Iri/(qb*qb);
    *Itzr_Vbci=Itzr_Iri*Iri_Vbci;
    *Itzr_Vbei=Itzr_qb*qb_Vbei;
    *Itzr_Vbci=(*Itzr_Vbci)+Itzr_qb*qb_Vbci;
    (*Itzf)=Ifi/qb;
    Itzf_Ifi=1.0/qb;
    Itzf_qb=-Ifi/(qb*qb);
    *Itzf_Vbei=Itzf_Ifi*Ifi_Vbei;
    *Itzf_Vbei=(*Itzf_Vbei)+Itzf_qb*qb_Vbei;
    *Itzf_Vbci=Itzf_qb*qb_Vbci;
    if(p[42]>0.0){
        argi=(*Vbep)/(p[44]*Vtv);
        argi_Vbep=1.0/(p[44]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbep=expi_argi*argi_Vbep;
        argx=(*Vbci)/(p[44]*Vtv);
        argx_Vbci=1.0/(p[44]*Vtv);
        expx=exp(argx);
        expx_argx=expx;
        expx_Vbci=expx_argx*argx_Vbci;
        Ifp=p[42]*(p[43]*expi+(1.0-p[43])*expx-1.0);
        Ifp_expi=p[42]*p[43];
        Ifp_expx=p[42]*(1.0-p[43]);
        Ifp_Vbep=Ifp_expi*expi_Vbep;
        Ifp_Vbci=Ifp_expx*expx_Vbci;
        q2p=Ifp*IIKP;
        q2p_Ifp=IIKP;
        q2p_Vbep=q2p_Ifp*Ifp_Vbep;
        q2p_Vbci=q2p_Ifp*Ifp_Vbci;
        qbp=0.5*(1.0+sqrt(1.0+4.0*q2p));
        qbp_q2p=1.0/sqrt(4.0*q2p+1.0);
        qbp_Vbep=qbp_q2p*q2p_Vbep;
        qbp_Vbci=qbp_q2p*q2p_Vbci;
        argi=(*Vbcp)/(p[44]*Vtv);
        argi_Vbcp=1.0/(p[44]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbcp=expi_argi*argi_Vbcp;
        Irp=p[42]*(expi-1.0);
        Irp_expi=p[42];
        Irp_Vbcp=Irp_expi*expi_Vbcp;
        (*Iccp)=(Ifp-Irp)/qbp;
        Iccp_Ifp=1.0/qbp;
        Iccp_Irp=-1.0/qbp;
        Iccp_qbp=-(Ifp-Irp)/(qbp*qbp);
        *Iccp_Vbep=Iccp_Ifp*Ifp_Vbep;
        *Iccp_Vbci=Iccp_Ifp*Ifp_Vbci;
        *Iccp_Vbcp=Iccp_Irp*Irp_Vbcp;
        *Iccp_Vbep=(*Iccp_Vbep)+Iccp_qbp*qbp_Vbep;
        *Iccp_Vbci=(*Iccp_Vbci)+Iccp_qbp*qbp_Vbci;
    }else{
        Ifp=0.0;
        Ifp_Vbep=0.0;
        Ifp_Vbci=0.0;
        qbp=1.0;
        qbp_Vbep=0.0;
        qbp_Vbci=0.0;
        (*Iccp)=0.0;
        *Iccp_Vbep=0.0;
        *Iccp_Vbci=0.0;
        *Iccp_Vbcp=0.0;
    }
    if(p[32]==1.0){
        argi=(*Vbei)/(p[33]*Vtv);
        argi_Vbei=1.0/(p[33]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbei=expi_argi*argi_Vbei;
        argn=(*Vbei)/(p[35]*Vtv);
        argn_Vbei=1.0/(p[35]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbei=expn_argn*argn_Vbei;
        if(p[98]>0.0){
            argx=(-p[98]-(*Vbei))/(p[99]*Vtv);
            argx_Vbei=-1.0/(p[99]*Vtv);
            expx=exp(argx);
            expx_argx=expx;
            expx_Vbei=expx_argx*argx_Vbei;
            (*Ibe)=p[31]*(expi-1.0)+p[34]*(expn-1.0)-p[100]*(expx-p[104]);
            Ibe_expi=p[31];
            Ibe_expn=p[34];
            Ibe_expx=-p[100];
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expx*expx_Vbei;
        }else{
            (*Ibe)=p[31]*(expi-1.0)+p[34]*(expn-1.0);
            Ibe_expi=p[31];
            Ibe_expn=p[34];
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
        }
        (*Ibex)=0.0;
        *Ibex_Vbex=0.0;
    }else if(p[32]==0.0){
        (*Ibe)=0.0;
        *Ibe_Vbei=0.0;
        argi=(*Vbex)/(p[33]*Vtv);
        argi_Vbex=1.0/(p[33]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbex=expi_argi*argi_Vbex;
        argn=(*Vbex)/(p[35]*Vtv);
        argn_Vbex=1.0/(p[35]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbex=expn_argn*argn_Vbex;
        if(p[98]>0.0){
            argx=(-p[98]-(*Vbex))/(p[99]*Vtv);
            argx_Vbex=-1.0/(p[99]*Vtv);
            expx=exp(argx);
            expx_argx=expx;
            expx_Vbex=expx_argx*argx_Vbex;
            (*Ibex)=p[31]*(expi-1.0)+p[34]*(expn-1.0)-p[100]*(expx-p[104]);
            Ibex_expi=p[31];
            Ibex_expn=p[34];
            Ibex_expx=-p[100];
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expx*expx_Vbex;
        }else{
            (*Ibex)=p[31]*(expi-1.0)+p[34]*(expn-1.0);
            Ibex_expi=p[31];
            Ibex_expn=p[34];
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
        }
    }else{
        argi=(*Vbei)/(p[33]*Vtv);
        argi_Vbei=1.0/(p[33]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbei=expi_argi*argi_Vbei;
        argn=(*Vbei)/(p[35]*Vtv);
        argn_Vbei=1.0/(p[35]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbei=expn_argn*argn_Vbei;
        if(p[98]>0.0){
            argx=(-p[98]-(*Vbei))/(p[99]*Vtv);
            argx_Vbei=-1.0/(p[99]*Vtv);
            expx=exp(argx);
            expx_argx=expx;
            expx_Vbei=expx_argx*argx_Vbei;
            (*Ibe)=p[32]*(p[31]*(expi-1.0)+p[34]*(expn-1.0)-p[100]*(expx-p[104]));
            Ibe_expi=p[31]*p[32];
            Ibe_expn=p[34]*p[32];
            Ibe_expx=-p[100]*p[32];
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expx*expx_Vbei;
        }else{
            (*Ibe)=p[32]*(p[31]*(expi-1.0)+p[34]*(expn-1.0));
            Ibe_expi=p[31]*p[32];
            Ibe_expn=p[34]*p[32];
            *Ibe_Vbei=Ibe_expi*expi_Vbei;
            *Ibe_Vbei=(*Ibe_Vbei)+Ibe_expn*expn_Vbei;
        }
        argi=(*Vbex)/(p[33]*Vtv);
        argi_Vbex=1.0/(p[33]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbex=expi_argi*argi_Vbex;
        argn=(*Vbex)/(p[35]*Vtv);
        argn_Vbex=1.0/(p[35]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbex=expn_argn*argn_Vbex;
        if(p[98]>0.0){
            argx=(-p[98]-(*Vbex))/(p[99]*Vtv);
            argx_Vbex=-1.0/(p[99]*Vtv);
            expx=exp(argx);
            expx_argx=expx;
            expx_Vbex=expx_argx*argx_Vbex;
            (*Ibex)=(1.0-p[32])*(p[31]*(expi-1.0)+p[34]*(expn-1.0)-p[100]*(expx-p[104]));
            Ibex_expi=p[31]*(1.0-p[32]);
            Ibex_expn=p[34]*(1.0-p[32]);
            Ibex_expx=-p[100]*(1.0-p[32]);
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expx*expx_Vbex;
        }else{
            (*Ibex)=(1.0-p[32])*(p[31]*(expi-1.0)+p[34]*(expn-1.0));
            Ibex_expi=p[31]*(1.0-p[32]);
            Ibex_expn=p[34]*(1.0-p[32]);
            *Ibex_Vbex=Ibex_expi*expi_Vbex;
            *Ibex_Vbex=(*Ibex_Vbex)+Ibex_expn*expn_Vbex;
        }
    }
    argi=(*Vbci)/(p[37]*Vtv);
    argi_Vbci=1.0/(p[37]*Vtv);
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    argn=(*Vbci)/(p[39]*Vtv);
    argn_Vbci=1.0/(p[39]*Vtv);
    expn=exp(argn);
    expn_argn=expn;
    expn_Vbci=expn_argn*argn_Vbci;
    Ibcj=p[36]*(expi-1.0)+p[38]*(expn-1.0);
    Ibcj_expi=p[36];
    Ibcj_expn=p[38];
    Ibcj_Vbci=Ibcj_expi*expi_Vbci;
    Ibcj_Vbci=Ibcj_Vbci+Ibcj_expn*expn_Vbci;
    if((p[45]>0.0)||(p[46]>0.0)){
        argi=(*Vbep)/(p[37]*Vtv);
        argi_Vbep=1.0/(p[37]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbep=expi_argi*argi_Vbep;
        argn=(*Vbep)/(p[39]*Vtv);
        argn_Vbep=1.0/(p[39]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbep=expn_argn*argn_Vbep;
        (*Ibep)=p[45]*(expi-1.0)+p[46]*(expn-1.0);
        Ibep_expi=p[45];
        Ibep_expn=p[46];
        *Ibep_Vbep=Ibep_expi*expi_Vbep;
        *Ibep_Vbep=(*Ibep_Vbep)+Ibep_expn*expn_Vbep;
    }else{
        (*Ibep)=0.0;
        *Ibep_Vbep=0.0;
    }
    if(p[40]>0.0){
        vl=0.5*(sqrt((p[24]-(*Vbci))*(p[24]-(*Vbci))+0.01)+(p[24]-(*Vbci)));
        vl_Vbci=0.5*(-(p[24]-(*Vbci))/sqrt(((p[24]-(*Vbci))*(p[24]-(*Vbci)))+0.01)-1.0);
        xvar2=(p[25]-1.0);
        xvar3=pow(vl,xvar2);
        xvar3_vl=xvar3*xvar2/vl;
        xvar3_Vbci=xvar3_vl*vl_Vbci;
        xvar1=-p[41]*xvar3;
        xvar1_xvar3=-p[41];
        xvar1_Vbci=xvar1_xvar3*xvar3_Vbci;
        xvar4=exp(xvar1);
        xvar4_xvar1=xvar4;
        xvar4_Vbci=xvar4_xvar1*xvar1_Vbci;
        avalf=p[40]*vl*xvar4;
        avalf_vl=p[40]*xvar4;
        avalf_xvar4=p[40]*vl;
        avalf_Vbci=avalf_vl*vl_Vbci;
        avalf_Vbci=avalf_Vbci+avalf_xvar4*xvar4_Vbci;
        Igc=((*Itzf)-(*Itzr)-Ibcj)*avalf;
        Igc_Itzf=avalf;
        Igc_Itzr=-avalf;
        Igc_Ibcj=-avalf;
        Igc_avalf=-(*Itzr)+(*Itzf)-Ibcj;
        Igc_Vbei=Igc_Itzf*(*Itzf_Vbei);
        Igc_Vbci=Igc_Itzf*(*Itzf_Vbci);
        Igc_Vbci=Igc_Vbci+Igc_Itzr*(*Itzr_Vbci);
        Igc_Vbei=Igc_Vbei+Igc_Itzr*(*Itzr_Vbei);
        Igc_Vbci=Igc_Vbci+Igc_Ibcj*Ibcj_Vbci;
        Igc_Vbci=Igc_Vbci+Igc_avalf*avalf_Vbci;
    }else{
        Igc=0.0;
        Igc_Vbei=0.0;
        Igc_Vbci=0.0;
    }
    (*Ibc)=Ibcj-Igc;
    Ibc_Ibcj=1.0;
    Ibc_Igc=-1.0;
    *Ibc_Vbci=Ibc_Ibcj*Ibcj_Vbci;
    *Ibc_Vbei=Ibc_Igc*Igc_Vbei;
    *Ibc_Vbci=(*Ibc_Vbci)+Ibc_Igc*Igc_Vbci;
    if(p[1]>0.0){
        (*Ircx)=(*Vrcx)/p[1];
        *Ircx_Vrcx=1.0/p[1];
    }else{
        (*Ircx)=0.0;
        *Ircx_Vrcx=0.0;
    }
    argi=(*Vbci)/Vtv;
    argi_Vbci=1.0/Vtv;
    expi=exp(argi);
    expi_argi=expi;
    expi_Vbci=expi_argi*argi_Vbci;
    argx=(*Vbcx)/Vtv;
    argx_Vbcx=1.0/Vtv;
    expx=exp(argx);
    expx_argx=expx;
    expx_Vbcx=expx_argx*argx_Vbcx;
    Kbci=sqrt(1.0+p[4]*expi);
    Kbci_expi=p[4]/(2.0*sqrt(expi*p[4]+1.0));
    Kbci_Vbci=Kbci_expi*expi_Vbci;
    Kbcx=sqrt(1.0+p[4]*expx);
    Kbcx_expx=p[4]/(2.0*sqrt(expx*p[4]+1.0));
    Kbcx_Vbcx=Kbcx_expx*expx_Vbcx;
    if(p[2]>0.0){
        rKp1=(Kbci+1.0)/(Kbcx+1.0);
        rKp1_Kbci=1.0/(Kbcx+1.0);
        rKp1_Kbcx=-(Kbci+1.0)/((Kbcx+1.0)*(Kbcx+1.0));
        rKp1_Vbci=rKp1_Kbci*Kbci_Vbci;
        rKp1_Vbcx=rKp1_Kbcx*Kbcx_Vbcx;
        xvar1=log(rKp1);
        xvar1_rKp1=1.0/rKp1;
        xvar1_Vbci=xvar1_rKp1*rKp1_Vbci;
        xvar1_Vbcx=xvar1_rKp1*rKp1_Vbcx;
        Iohm=((*Vrci)+Vtv*(Kbci-Kbcx-xvar1))/p[2];
        Iohm_Vrci=1.0/p[2];
        Iohm_Kbci=Vtv/p[2];
        Iohm_Kbcx=-Vtv/p[2];
        Iohm_xvar1=-Vtv/p[2];
        Iohm_Vbci=Iohm_Kbci*Kbci_Vbci;
        Iohm_Vbcx=Iohm_Kbcx*Kbcx_Vbcx;
        Iohm_Vbci=Iohm_Vbci+Iohm_xvar1*xvar1_Vbci;
        Iohm_Vbcx=Iohm_Vbcx+Iohm_xvar1*xvar1_Vbcx;
        derf=IVO*p[2]*Iohm/(1.0+0.5*IVO*IHRCF*sqrt((*Vrci)*(*Vrci)+0.01));
        derf_Iohm=IVO*p[2]/(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0);
        derf_Vrci=-0.5*IHRCF*Iohm*(IVO*IVO)*p[2]*(*Vrci)/(sqrt(((*Vrci)*(*Vrci))+0.01)*((0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)*(0.5*IHRCF*IVO*sqrt(((*Vrci)*(*Vrci))+0.01)+1.0)));
        derf_Vrci=derf_Vrci+derf_Iohm*Iohm_Vrci;
        derf_Vbci=derf_Iohm*Iohm_Vbci;
        derf_Vbcx=derf_Iohm*Iohm_Vbcx;
        (*Irci)=Iohm/sqrt(1.0+derf*derf);
        Irci_Iohm=1.0/sqrt((derf*derf)+1.0);
        Irci_derf=-derf*Iohm/pow(((derf*derf)+1.0),(3.0/2.0));
        *Irci_Vrci=Irci_Iohm*Iohm_Vrci;
        *Irci_Vbci=Irci_Iohm*Iohm_Vbci;
        *Irci_Vbcx=Irci_Iohm*Iohm_Vbcx;
        *Irci_Vrci=(*Irci_Vrci)+Irci_derf*derf_Vrci;
        *Irci_Vbci=(*Irci_Vbci)+Irci_derf*derf_Vbci;
        *Irci_Vbcx=(*Irci_Vbcx)+Irci_derf*derf_Vbcx;
    }else{
        (*Irci)=0.0;
        *Irci_Vrci=0.0;
        *Irci_Vbci=0.0;
        *Irci_Vbcx=0.0;
    }
    if(p[6]>0.0){
        (*Irbx)=(*Vrbx)/p[6];
        *Irbx_Vrbx=1.0/p[6];
    }else{
        (*Irbx)=0.0;
        *Irbx_Vrbx=0.0;
    }
    if(p[7]>0.0){
        (*Irbi)=(*Vrbi)*qb/p[7];
        *Irbi_Vrbi=qb/p[7];
        Irbi_qb=(*Vrbi)/p[7];
        *Irbi_Vbei=Irbi_qb*qb_Vbei;
        *Irbi_Vbci=Irbi_qb*qb_Vbci;
    }else{
        (*Irbi)=0.0;
        *Irbi_Vrbi=0.0;
        *Irbi_Vbei=0.0;
        *Irbi_Vbci=0.0;
    }
    if(p[8]>0.0){
        (*Ire)=(*Vre)/p[8];
        *Ire_Vre=1.0/p[8];
    }else{
        (*Ire)=0.0;
        *Ire_Vre=0.0;
    }
    if(p[10]>0.0){
        (*Irbp)=(*Vrbp)*qbp/p[10];
        *Irbp_Vrbp=qbp/p[10];
        Irbp_qbp=(*Vrbp)/p[10];
        *Irbp_Vbep=Irbp_qbp*qbp_Vbep;
        *Irbp_Vbci=Irbp_qbp*qbp_Vbci;
    }else{
        (*Irbp)=0.0;
        *Irbp_Vrbp=0.0;
        *Irbp_Vbep=0.0;
        *Irbp_Vbci=0.0;
    }
    if((p[47]>0.0)||(p[49]>0.0)){
        argi=(*Vbcp)/(p[48]*Vtv);
        argi_Vbcp=1.0/(p[48]*Vtv);
        expi=exp(argi);
        expi_argi=expi;
        expi_Vbcp=expi_argi*argi_Vbcp;
        argn=(*Vbcp)/(p[50]*Vtv);
        argn_Vbcp=1.0/(p[50]*Vtv);
        expn=exp(argn);
        expn_argn=expn;
        expn_Vbcp=expn_argn*argn_Vbcp;
        (*Ibcp)=p[47]*(expi-1.0)+p[49]*(expn-1.0);
        Ibcp_expi=p[47];
        Ibcp_expn=p[49];
        *Ibcp_Vbcp=Ibcp_expi*expi_Vbcp;
        *Ibcp_Vbcp=(*Ibcp_Vbcp)+Ibcp_expn*expn_Vbcp;
    }else{
        (*Ibcp)=0.0;
        *Ibcp_Vbcp=0.0;
    }
    if(p[9]>0.0){
        (*Irs)=(*Vrs)/p[9];
        *Irs_Vrs=1.0/p[9];
    }else{
        (*Irs)=0.0;
        *Irs_Vrs=0.0;
    }
    if(Ifi>0.0){
        sgIf=1.0;
    }else{
        sgIf=0.0;
    }
    rIf=Ifi*sgIf*IITF;
    rIf_Ifi=IITF*sgIf;
    rIf_Vbei=rIf_Ifi*Ifi_Vbei;
    mIf=rIf/(rIf+1.0);
    mIf_rIf=1.0/(rIf+1.0)-rIf/((rIf+1.0)*(rIf+1.0));
    mIf_Vbei=mIf_rIf*rIf_Vbei;
    xvar1=(*Vbci)*IVTF/1.44;
    xvar1_Vbci=0.6944444*IVTF;
    xvar2=exp(xvar1);
    xvar2_xvar1=xvar2;
    xvar2_Vbci=xvar2_xvar1*xvar1_Vbci;
    tff=p[56]*(1.0+p[57]*q1)*(1.0+p[58]*xvar2*(slTF+mIf*mIf)*sgIf);
    tff_q1=p[57]*p[56]*(sgIf*(slTF+(mIf*mIf))*p[58]*xvar2+1.0);
    tff_xvar2=(q1*p[57]+1.0)*sgIf*(slTF+(mIf*mIf))*p[56]*p[58];
    tff_mIf=2.0*mIf*(q1*p[57]+1.0)*sgIf*p[56]*p[58]*xvar2;
    tff_Vbei=tff_q1*q1_Vbei;
    tff_Vbci=tff_q1*q1_Vbci;
    tff_Vbci=tff_Vbci+tff_xvar2*xvar2_Vbci;
    tff_Vbei=tff_Vbei+tff_mIf*mIf_Vbei;
    (*Qbe)=p[16]*qdbe*p[32]+tff*Ifi/qb;
    Qbe_qdbe=p[16]*p[32];
    Qbe_tff=Ifi/qb;
    Qbe_Ifi=tff/qb;
    Qbe_qb=-Ifi*tff/(qb*qb);
    *Qbe_Vbei=Qbe_qdbe*qdbe_Vbei;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_tff*tff_Vbei;
    *Qbe_Vbci=Qbe_tff*tff_Vbci;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_Ifi*Ifi_Vbei;
    *Qbe_Vbei=(*Qbe_Vbei)+Qbe_qb*qb_Vbei;
    *Qbe_Vbci=(*Qbe_Vbci)+Qbe_qb*qb_Vbci;
    (*Qbex)=p[16]*qdbex*(1.0-p[32]);
    Qbex_qdbex=p[16]*(1.0-p[32]);
    *Qbex_Vbex=Qbex_qdbex*qdbex_Vbex;
    (*Qbc)=p[21]*qdbc+p[61]*Iri+p[22]*Kbci;
    Qbc_qdbc=p[21];
    Qbc_Iri=p[61];
    Qbc_Kbci=p[22];
    *Qbc_Vbci=Qbc_qdbc*qdbc_Vbci;
    *Qbc_Vbci=(*Qbc_Vbci)+Qbc_Iri*Iri_Vbci;
    *Qbc_Vbci=(*Qbc_Vbci)+Qbc_Kbci*Kbci_Vbci;
    (*Qbcx)=p[22]*Kbcx;
    Qbcx_Kbcx=p[22];
    *Qbcx_Vbcx=Qbcx_Kbcx*Kbcx_Vbcx;
    (*Qbep)=p[23]*qdbep+p[61]*Ifp;
    Qbep_qdbep=p[23];
    Qbep_Ifp=p[61];
    *Qbep_Vbep=Qbep_qdbep*qdbep_Vbep;
    *Qbep_Vbep=(*Qbep_Vbep)+Qbep_Ifp*Ifp_Vbep;
    *Qbep_Vbci=Qbep_Ifp*Ifp_Vbci;
    (*Qbcp)=p[27]*qdbcp+p[87]*(*Vbcp);
    Qbcp_qdbcp=p[27];
    *Qbcp_Vbcp=p[87];
    *Qbcp_Vbcp=(*Qbcp_Vbcp)+Qbcp_qdbcp*qdbcp_Vbcp;
    (*Qbeo)=(*Vbe)*p[15];
    *Qbeo_Vbe=p[15];
    (*Qbco)=(*Vbc)*p[20];
    *Qbco_Vbc=p[20];

/*  Scale outputs */

    if((*SCALE)!=1.0){
        *Ibe=(*SCALE)*(*Ibe);
        *Ibe_Vbei=(*SCALE)*(*Ibe_Vbei);
        *Ibex=(*SCALE)*(*Ibex);
        *Ibex_Vbex=(*SCALE)*(*Ibex_Vbex);
        *Itzf=(*SCALE)*(*Itzf);
        *Itzf_Vbei=(*SCALE)*(*Itzf_Vbei);
        *Itzf_Vbci=(*SCALE)*(*Itzf_Vbci);
        *Itzr=(*SCALE)*(*Itzr);
        *Itzr_Vbci=(*SCALE)*(*Itzr_Vbci);
        *Itzr_Vbei=(*SCALE)*(*Itzr_Vbei);
        *Ibc=(*SCALE)*(*Ibc);
        *Ibc_Vbci=(*SCALE)*(*Ibc_Vbci);
        *Ibc_Vbei=(*SCALE)*(*Ibc_Vbei);
        *Ibep=(*SCALE)*(*Ibep);
        *Ibep_Vbep=(*SCALE)*(*Ibep_Vbep);
        *Ircx=(*SCALE)*(*Ircx);
        *Ircx_Vrcx=(*SCALE)*(*Ircx_Vrcx);
        *Irci=(*SCALE)*(*Irci);
        *Irci_Vrci=(*SCALE)*(*Irci_Vrci);
        *Irci_Vbci=(*SCALE)*(*Irci_Vbci);
        *Irci_Vbcx=(*SCALE)*(*Irci_Vbcx);
        *Irbx=(*SCALE)*(*Irbx);
        *Irbx_Vrbx=(*SCALE)*(*Irbx_Vrbx);
        *Irbi=(*SCALE)*(*Irbi);
        *Irbi_Vrbi=(*SCALE)*(*Irbi_Vrbi);
        *Irbi_Vbei=(*SCALE)*(*Irbi_Vbei);
        *Irbi_Vbci=(*SCALE)*(*Irbi_Vbci);
        *Ire=(*SCALE)*(*Ire);
        *Ire_Vre=(*SCALE)*(*Ire_Vre);
        *Irbp=(*SCALE)*(*Irbp);
        *Irbp_Vrbp=(*SCALE)*(*Irbp_Vrbp);
        *Irbp_Vbep=(*SCALE)*(*Irbp_Vbep);
        *Irbp_Vbci=(*SCALE)*(*Irbp_Vbci);
        *Qbe=(*SCALE)*(*Qbe);
        *Qbe_Vbei=(*SCALE)*(*Qbe_Vbei);
        *Qbe_Vbci=(*SCALE)*(*Qbe_Vbci);
        *Qbex=(*SCALE)*(*Qbex);
        *Qbex_Vbex=(*SCALE)*(*Qbex_Vbex);
        *Qbc=(*SCALE)*(*Qbc);
        *Qbc_Vbci=(*SCALE)*(*Qbc_Vbci);
        *Qbcx=(*SCALE)*(*Qbcx);
        *Qbcx_Vbcx=(*SCALE)*(*Qbcx_Vbcx);
        *Qbep=(*SCALE)*(*Qbep);
        *Qbep_Vbep=(*SCALE)*(*Qbep_Vbep);
        *Qbep_Vbci=(*SCALE)*(*Qbep_Vbci);
        *Qbeo=(*SCALE)*(*Qbeo);
        *Qbeo_Vbe=(*SCALE)*(*Qbeo_Vbe);
        *Qbco=(*SCALE)*(*Qbco);
        *Qbco_Vbc=(*SCALE)*(*Qbco_Vbc);
        *Ibcp=(*SCALE)*(*Ibcp);
        *Ibcp_Vbcp=(*SCALE)*(*Ibcp_Vbcp);
        *Iccp=(*SCALE)*(*Iccp);
        *Iccp_Vbep=(*SCALE)*(*Iccp_Vbep);
        *Iccp_Vbci=(*SCALE)*(*Iccp_Vbci);
        *Iccp_Vbcp=(*SCALE)*(*Iccp_Vbcp);
        *Irs=(*SCALE)*(*Irs);
        *Irs_Vrs=(*SCALE)*(*Irs_Vrs);
        *Qbcp=(*SCALE)*(*Qbcp);
        *Qbcp_Vbcp=(*SCALE)*(*Qbcp_Vbcp);
    }
    return(0);
}
