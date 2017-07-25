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

int vbic_4T_it_cf_fj(double *,
    double *,double *,double *,double *,double *,double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
    double *,double *,double *,double *,double *,double *, double *,
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
    ,Vbei,Vbex,Vbci,Vbep,Vbcp,Vrcx
    ,Vbcx,Vrci,Vrbx,Vrbi,Vre,Vrbp,Vrs
    ,Vbe,Vbc,Ibe,Ibe_Vbei,Ibex,Ibex_Vbex,Itzf
    ,Itzf_Vbei,Itzf_Vbci,Itzr,Itzr_Vbci,Itzr_Vbei,Ibc,Ibc_Vbci
    ,Ibc_Vbei,Ibep,Ibep_Vbep,Ircx,Ircx_Vrcx,Irci,Irci_Vrci
    ,Irci_Vbci,Irci_Vbcx,Irbx,Irbx_Vrbx,Irbi,Irbi_Vrbi,Irbi_Vbei
    ,Irbi_Vbci,Ire,Ire_Vre,Irbp,Irbp_Vrbp,Irbp_Vbep,Irbp_Vbci
    ,Qbe,Qbe_Vbei,Qbe_Vbci,Qbex,Qbex_Vbex,Qbc,Qbc_Vbci
    ,Qbcx,Qbcx_Vbcx,Qbep,Qbep_Vbep,Qbep_Vbci,Qbeo,Qbeo_Vbe
    ,Qbco,Qbco_Vbc,Ibcp,Ibcp_Vbcp,Iccp,Iccp_Vbep,Iccp_Vbci
    ,Iccp_Vbcp,Irs,Irs_Vrs,Qbcp,Qbcp_Vbcp,SCALE;
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
    int icheck = 0;
    int ichk1, ichk2, ichk3, ichk4, ichk5;
    int error;
    int SenCond=0;
    double gqbeo, cqbeo, gqbco, cqbco, gbcx, cbcx;

    /*  loop through all the models */
    for( ; model != NULL; model = model->VBICnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VBICinstances; here != NULL ;
                here=here->VBICnextInstance) {

            vt = here->VBICtemp * CONSTKoverQ;

            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("VBICload\n");
#endif /* SENSDEBUG */
                if((ckt->CKTsenInfo->SENstatus == PERTURBATION)&&
                    (here->VBICsenPertFlag == OFF)) continue;
                SenCond = here->VBICsenPertFlag;
            }

            gbcx = 0.0;
            cbcx = 0.0;
            gqbeo = 0.0;
            cqbeo = 0.0;
            gqbco = 0.0;
            cqbco = 0.0;
            /*
             *   dc model paramters
             */
            p[0] = here->VBICttnom;
            p[1] = here->VBICtextCollResist;
            p[2] = here->VBICtintCollResist;
            p[3] = here->VBICtepiSatVoltage;
            p[4] = here->VBICtepiDoping;
            p[5] = model->VBIChighCurFac;
            p[6] = here->VBICtextBaseResist;
            p[7] = here->VBICtintBaseResist;
            p[8] = here->VBICtemitterResist;
            p[9] = here->VBICtsubstrateResist;
            p[10] = here->VBICtparBaseResist;
            p[11] = here->VBICtsatCur;
            p[12] = here->VBICtemissionCoeffF;
            p[13] = here->VBICtemissionCoeffR;
            p[14] = model->VBICdeplCapLimitF;
            p[15] = model->VBICextOverlapCapBE;
            p[16] = here->VBICtdepletionCapBE;
            p[17] = here->VBICtpotentialBE;
            p[18] = model->VBICjunctionExpBE;
            p[19] = model->VBICsmoothCapBE;
            p[20] = model->VBICextOverlapCapBC;
            p[21] = here->VBICtdepletionCapBC;
            p[22] = model->VBICepiCharge;
            p[23] = here->VBICtextCapBC;
            p[24] = here->VBICtpotentialBC;
            p[25] = model->VBICjunctionExpBC;
            p[26] = model->VBICsmoothCapBC;
            p[27] = here->VBICtextCapSC;
            p[28] = here->VBICtpotentialSC;
            p[29] = model->VBICjunctionExpSC;
            p[30] = model->VBICsmoothCapSC;
            p[31] = here->VBICtidealSatCurBE;
            p[32] = model->VBICportionIBEI;
            p[33] = model->VBICidealEmissCoeffBE;
            p[34] = here->VBICtnidealSatCurBE;
            p[35] = model->VBICnidealEmissCoeffBE;
            p[36] = here->VBICtidealSatCurBC;
            p[37] = model->VBICidealEmissCoeffBC;
            p[38] = here->VBICtnidealSatCurBC;
            p[39] = model->VBICnidealEmissCoeffBC;
            p[40] = model->VBICavalanchePar1BC;
            p[41] = here->VBICtavalanchePar2BC;
            p[42] = here->VBICtparasitSatCur;
            p[43] = model->VBICportionICCP;
            p[44] = model->VBICparasitFwdEmissCoeff;
            p[45] = here->VBICtidealParasitSatCurBE;
            p[46] = here->VBICtnidealParasitSatCurBE;
            p[47] = here->VBICtidealParasitSatCurBC;
            p[48] = model->VBICidealParasitEmissCoeffBC;
            p[49] = here->VBICtnidealParasitSatCurBC;
            p[50] = model->VBICnidealParasitEmissCoeffBC;
            p[51] = model->VBICearlyVoltF;
            p[52] = model->VBICearlyVoltR;
            p[53] = here->VBICtrollOffF;
            p[54] = model->VBICrollOffR;
            p[55] = model->VBICparRollOff;
            p[56] = model->VBICtransitTimeF;
            p[57] = model->VBICvarTransitTimeF;
            p[58] = model->VBICtransitTimeBiasCoeffF;
            p[59] = model->VBICtransitTimeFVBC;
            p[60] = model->VBICtransitTimeHighCurrentF;
            p[61] = model->VBICtransitTimeR;
            p[62] = model->VBICdelayTimeF;
            p[63] = model->VBICfNcoef;
            p[64] = model->VBICfNexpA;
            p[65] = model->VBICfNexpB;
            p[66] = model->VBICtempExpRE;
            p[67] = model->VBICtempExpRBI;
            p[68] = model->VBICtempExpRCI;
            p[69] = model->VBICtempExpRS;
            p[70] = model->VBICtempExpVO;
            p[71] = model->VBICactivEnergyEA;
            p[72] = model->VBICactivEnergyEAIE;
            p[73] = model->VBICactivEnergyEAIC;
            p[74] = model->VBICactivEnergyEAIS;
            p[75] = model->VBICactivEnergyEANE;
            p[76] = model->VBICactivEnergyEANC;
            p[77] = model->VBICactivEnergyEANS;
            p[78] = model->VBICtempExpIS;
            p[79] = model->VBICtempExpII;
            p[80] = model->VBICtempExpIN;
            p[81] = model->VBICtempExpNF;
            p[82] = model->VBICtempExpAVC;
            p[83] = model->VBICthermalResist;
            p[84] = model->VBICthermalCapacitance;
            p[85] = model->VBICpunchThroughVoltageBC;
            p[86] = model->VBICdeplCapCoeff1;
            p[87] = model->VBICfixedCapacitanceCS;
            p[88] = model->VBICsgpQBselector;
            p[89] = model->VBIChighCurrentBetaRolloff;
            p[90] = model->VBICtempExpIKF;
            p[91] = model->VBICtempExpRCX;
            p[92] = model->VBICtempExpRBX;
            p[93] = model->VBICtempExpRBP;
            p[94] = here->VBICtsepISRR;
            p[95] = model->VBICtempExpXISR;
            p[96] = model->VBICdear;
            p[97] = model->VBICeap;
            p[98] = here->VBICtvbbe;
            p[99] = here->VBICtnbbe;
            p[100] = model->VBICibbe;
            p[101] = model->VBICtvbbe1;
            p[102] = model->VBICtvbbe2;
            p[103] = model->VBICtnbbe;
            p[104] = model->VBICebbe;
            p[105] = model->VBIClocTempDiff;
            p[106] = model->VBICrevVersion;
            p[107] = model->VBICrefVersion;

            SCALE = here->VBICarea * here->VBICm;

            if(SenCond){
#ifdef SENSDEBUG
                printf("VBICsenPertFlag = ON \n");
#endif /* SENSDEBUG */

                if((ckt->CKTsenInfo->SENmode == TRANSEN)&&
                    (ckt->CKTmode & MODEINITTRAN)) {
                    Vbe = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICbaseNode)-
                        *(ckt->CKTrhsOp+here->VBICemitNode));
                    Vbc = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICbaseNode)-
                        *(ckt->CKTrhsOp+here->VBICcollNode));
                    Vbei = *(ckt->CKTstate1 + here->VBICvbei);
                    Vbex = *(ckt->CKTstate1 + here->VBICvbex);
                    Vbci = *(ckt->CKTstate1 + here->VBICvbci);
                    Vbcx = *(ckt->CKTstate1 + here->VBICvbcx);
                    Vbep = *(ckt->CKTstate1 + here->VBICvbep);
                    Vrci = *(ckt->CKTstate1 + here->VBICvrci);
                    Vrbi = *(ckt->CKTstate1 + here->VBICvrbi);
                    Vrbp = *(ckt->CKTstate1 + here->VBICvrbp);
                    Vrcx = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICcollNode)-
                        *(ckt->CKTrhsOp+here->VBICcollCXNode));
                    Vrbx = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICbaseNode)-
                        *(ckt->CKTrhsOp+here->VBICbaseBXNode));
                    Vre = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICemitNode)-
                        *(ckt->CKTrhsOp+here->VBICemitEINode));
                    Vbcp = *(ckt->CKTstate1 + here->VBICvbcp);
                    Vrs = model->VBICtype*(
                        *(ckt->CKTrhsOp+here->VBICsubsNode)-
                        *(ckt->CKTrhsOp+here->VBICsubsSINode));
                }
                else{
                    Vbei = *(ckt->CKTstate0 + here->VBICvbei);
                    Vbex = *(ckt->CKTstate0 + here->VBICvbex);
                    Vbci = *(ckt->CKTstate0 + here->VBICvbci);
                    Vbcx = *(ckt->CKTstate0 + here->VBICvbcx);
                    Vbep = *(ckt->CKTstate0 + here->VBICvbep);
                    Vrci = *(ckt->CKTstate0 + here->VBICvrci);
                    Vrbi = *(ckt->CKTstate0 + here->VBICvrbi);
                    Vrbp = *(ckt->CKTstate0 + here->VBICvrbp);
                    Vbcp = *(ckt->CKTstate0 + here->VBICvbcp);
                    if((ckt->CKTsenInfo->SENmode == DCSEN)||
                        (ckt->CKTsenInfo->SENmode == TRANSEN)){
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
                    }
                    if(ckt->CKTsenInfo->SENmode == ACSEN){
                        Vbe = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICbaseNode)-
                            *(ckt->CKTrhsOp+here->VBICemitNode));
                        Vbc = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICbaseNode)-
                            *(ckt->CKTrhsOp+here->VBICcollNode));
                        Vrcx = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICcollNode)-
                            *(ckt->CKTrhsOp+here->VBICcollCXNode));
                        Vrbx = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICbaseNode)-
                            *(ckt->CKTrhsOp+here->VBICbaseBXNode));
                        Vre = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICemitNode)-
                            *(ckt->CKTrhsOp+here->VBICemitEINode));
                        Vrs = model->VBICtype*(
                            *(ckt->CKTrhsOp+here->VBICsubsNode)-
                            *(ckt->CKTrhsOp+here->VBICsubsSINode));
                    }
                }
                goto next1;
            }

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
                if( (ckt->CKTmode & MODETRAN) && (ckt->CKTmode & MODEUIC) ) {
                    Vbc = model->VBICtype * (here->VBICicVBE-here->VBICicVCE);
                }
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
            } else if((ckt->CKTmode & MODEINITJCT) && (here->VBICoff==0)) {
                Vbe=Vbei=Vbex=model->VBICtype*here->VBICtVcrit;
                Vbc=Vbci=Vbcx=Vbep=0.0;
                Vbcp=Vbc-Vbe;
                Vrci=Vrbi=Vrbp=0.0;
                Vrcx=Vrbx=Vre=Vrs=0.0;
            } else if((ckt->CKTmode & MODEINITJCT) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (here->VBICoff!=0))) {
                Vbe=0.0;
                Vbei=Vbex=Vbe;
                Vbc=0.0;
                Vbci=Vbcx=Vbc;
                Vbep=Vbcp=0.0;
                Vrci=Vrbi=Vrbp=0.0;
                Vrcx=Vrbx=Vre=Vrs=0.0;
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
                 */
                if( (ckt->CKTbypass) &&
                        (!(ckt->CKTmode & MODEINITPRED)) &&
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
                ichk1 = 1;
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
                if ((ichk1 == 1) || (ichk2 == 1) || (ichk3 == 1) || (ichk4 == 1) || (ichk5 == 1)) icheck=1;
            }
            /*
             *   determine dc current and derivitives
             */
next1:      
            iret = vbic_4T_it_cf_fj(p
            ,&Vbei,&Vbex,&Vbci,&Vbep,&Vbcp,&Vrcx
            ,&Vbcx,&Vrci,&Vrbx,&Vrbi,&Vre,&Vrbp,&Vrs
            ,&Vbe,&Vbc,&Ibe,&Ibe_Vbei,&Ibex,&Ibex_Vbex,&Itzf
            ,&Itzf_Vbei,&Itzf_Vbci,&Itzr,&Itzr_Vbci,&Itzr_Vbei,&Ibc,&Ibc_Vbci
            ,&Ibc_Vbei,&Ibep,&Ibep_Vbep,&Ircx,&Ircx_Vrcx,&Irci,&Irci_Vrci
            ,&Irci_Vbci,&Irci_Vbcx,&Irbx,&Irbx_Vrbx,&Irbi,&Irbi_Vrbi,&Irbi_Vbei
            ,&Irbi_Vbci,&Ire,&Ire_Vre,&Irbp,&Irbp_Vrbp,&Irbp_Vbep,&Irbp_Vbci
            ,&Qbe,&Qbe_Vbei,&Qbe_Vbci,&Qbex,&Qbex_Vbex,&Qbc,&Qbc_Vbci
            ,&Qbcx,&Qbcx_Vbcx,&Qbep,&Qbep_Vbep,&Qbep_Vbci,&Qbeo,&Qbeo_Vbe
            ,&Qbco,&Qbco_Vbc,&Ibcp,&Ibcp_Vbcp,&Iccp,&Iccp_Vbep,&Iccp_Vbci
            ,&Iccp_Vbcp,&Irs,&Irs_Vrs,&Qbcp,&Qbcp_Vbcp,&SCALE);
          
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
            Irci_Vbci += ckt->CKTgmin;
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

                here->VBICcapbe = Qbe_Vbei;
                here->VBICcapbex = Qbex_Vbex;
                here->VBICcapbc = Qbc_Vbci;
                here->VBICcapbcx = Qbcx_Vbcx;
                here->VBICcapbep = Qbep_Vbep;
                here->VBICcapbcp = Qbcp_Vbcp;

                /*
                 *   store small-signal parameters
                 */
                if ( (!(ckt->CKTmode & MODETRANOP))||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->VBICcqbe)  = Qbe_Vbei;
                        *(ckt->CKTstate0 + here->VBICcqbeci) = Qbe_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbex) = Qbex_Vbex;
                        *(ckt->CKTstate0 + here->VBICcqbc)  = Qbc_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbcx) = Qbcx_Vbcx;
                        *(ckt->CKTstate0 + here->VBICcqbep) = Qbep_Vbep;
                        *(ckt->CKTstate0 + here->VBICcqbepci) = Qbep_Vbci;
                        *(ckt->CKTstate0 + here->VBICcqbeo) = Qbeo_Vbe;
                        *(ckt->CKTstate0 + here->VBICcqbco) = Qbco_Vbc;
                        *(ckt->CKTstate0 + here->VBICcqbcp) = Qbcp_Vbcp;
                        if(SenCond) {
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
                        }
#ifdef SENSDEBUG
                        printf("storing small signal parameters for op\n");
                        printf("Cbe = %.7e, Cbex = %.7e\n", Qbe_Vbei, Qbex_Vbex);
                        printf("Cbc = %.7e, Cbcx = %.7e\n", Qbc_Vbci, Qbcx_Vbcx);
                        printf("gpi = %.7e\n", Ibe_Vbei);
                        printf("gmu = %.7e, gm = %.7e\n", Ibc_Vbci, Itzf_Vbei);
                        printf("go = %.7e, gx = %.7e\n", Itzf_Vbci, Irbi_Vrbi);
                        printf("cc = %.7e, cb = %.7e\n", Ibe+Itzf, Ibe);
#endif /* SENSDEBUG */
                        continue; /* go to 1000 */
                    }
                    /*
                     *   transient analysis
                     */
                    if(SenCond && ckt->CKTsenInfo->SENmode == TRANSEN) {
                        *(ckt->CKTstate0 + here->VBICibe)  = Ibe;
                        *(ckt->CKTstate0 + here->VBICibex) = Ibex;
                        *(ckt->CKTstate0 + here->VBICitzf) = Itzf;
                        *(ckt->CKTstate0 + here->VBICitzr) = Itzr;
                        *(ckt->CKTstate0 + here->VBICibc)  = Ibc;
                        *(ckt->CKTstate0 + here->VBICibep) = Ibep;
                        *(ckt->CKTstate0 + here->VBICirci) = Irci;
                        *(ckt->CKTstate0 + here->VBICirbi) = Irbi;
                        *(ckt->CKTstate0 + here->VBICirbp) = Irbp;
                        *(ckt->CKTstate0 + here->VBICibcp) = Ibcp;
                        *(ckt->CKTstate0 + here->VBICiccp) = Iccp;
                        continue;
                    }

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
                    }
                }
            }

            if(SenCond) goto next2;

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
next2:
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

            /* Do not load the Jacobian and the rhs if
               perturbation is being carried out */
            if(SenCond) continue;
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
c           KCL at internal nodes
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
            *(here->VBICcollCXCollPtr) +=  -Ircx_Vrcx;
            *(here->VBICcollCollCXPtr) +=  -Ircx_Vrcx;
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

        }

    }
    return(OK);
}

int vbic_4T_it_cf_fj(
        double *p
	, double *Vbei, double *Vbex, double *Vbci, double *Vbep, double *Vbcp, double *Vrcx
	, double *Vbcx, double *Vrci, double *Vrbx, double *Vrbi, double *Vre, double *Vrbp, double *Vrs
	, double *Vbe, double *Vbc, double *Ibe, double *Ibe_Vbei, double *Ibex, double *Ibex_Vbex, double *Itzf
	, double *Itzf_Vbei, double *Itzf_Vbci, double *Itzr, double *Itzr_Vbci, double *Itzr_Vbei, double *Ibc, double *Ibc_Vbci
	, double *Ibc_Vbei, double *Ibep, double *Ibep_Vbep, double *Ircx, double *Ircx_Vrcx, double *Irci, double *Irci_Vrci
	, double *Irci_Vbci, double *Irci_Vbcx, double *Irbx, double *Irbx_Vrbx, double *Irbi, double *Irbi_Vrbi, double *Irbi_Vbei
	, double *Irbi_Vbci, double *Ire, double *Ire_Vre, double *Irbp, double *Irbp_Vrbp, double *Irbp_Vbep, double *Irbp_Vbci
	, double *Qbe, double *Qbe_Vbei, double *Qbe_Vbci, double *Qbex, double *Qbex_Vbex, double *Qbc, double *Qbc_Vbci
	, double *Qbcx, double *Qbcx_Vbcx, double *Qbep, double *Qbep_Vbep, double *Qbep_Vbci, double *Qbeo, double *Qbeo_Vbe
	, double *Qbco, double *Qbco_Vbc, double *Ibcp, double *Ibcp_Vbcp, double *Iccp, double *Iccp_Vbep, double *Iccp_Vbci
    , double *Iccp_Vbcp, double *Irs, double *Irs_Vrs, double *Qbcp, double *Qbcp_Vbcp, double *SCALE)
{
double	Vtv,IVEF,IVER,IIKF,IIKR,IIKP,IVO;
double	IHRCF,IVTF,IITF,slTF,dv0,dvh,dvh_Vbei;
double	xvar1,xvar2,pwq,qlo,qlo_Vbei,qhi,qhi_dvh;
double	qhi_Vbei,xvar1_Vbei,xvar3,xvar3_xvar1,xvar3_Vbei,qlo_xvar3,qdbe;
double	qdbe_qlo,qdbe_Vbei,qdbe_qhi,mv0,vl0,q0,dv;
double	dv_Vbei,mv,mv_dv,mv_Vbei,vl,vl_dv,vl_Vbei;
double	vl_mv,xvar1_vl,qdbe_vl,dvh_Vbex,qlo_Vbex,qhi_Vbex,xvar1_Vbex;
double	xvar3_Vbex,qdbex,qdbex_qlo,qdbex_Vbex,qdbex_qhi,dv_Vbex,mv_Vbex;
double	vl_Vbex,qdbex_vl,dvh_Vbci,qlo_Vbci,qhi_Vbci,xvar1_Vbci,xvar3_Vbci;
double	qdbc,qdbc_qlo,qdbc_Vbci,qdbc_qhi,vn0,vnl0,qlo0;
double	vn,vn_Vbci,vnl,vnl_vn,vnl_Vbci,vl_vnl,vl_Vbci;
double	sel,sel_vnl,sel_Vbci,crt,cmx,cl,cl_sel;
double	cl_Vbci,ql,ql_Vbci,ql_vl,ql_cl,qdbc_ql,dv_Vbci;
double	mv_Vbci,qdbc_vl,dvh_Vbep,qlo_Vbep,qhi_Vbep,xvar1_Vbep,xvar3_Vbep;
double	qdbep,qdbep_qlo,qdbep_Vbep,qdbep_qhi,vn_Vbep,vnl_Vbep,vl_Vbep;
double	sel_Vbep,cl_Vbep,ql_Vbep,qdbep_ql,dv_Vbep,mv_Vbep,qdbep_vl;
double	dvh_Vbcp,qlo_Vbcp,qhi_Vbcp,xvar1_Vbcp,xvar3_Vbcp,qdbcp,qdbcp_qlo;
double	qdbcp_Vbcp,qdbcp_Vbep,qdbcp_qhi,dv_Vbcp,mv_Vbcp,vl_Vbcp,qdbcp_vl;
double	argi,argi_Vbei,expi,expi_argi,expi_Vbei,Ifi,Ifi_expi;
double	Ifi_Vbei,argi_Vbci,expi_Vbci,Iri,Iri_expi,Iri_Vbci,q1z;
double	q1z_qdbe,q1z_Vbei,q1z_qdbc,q1z_Vbci,q1,q1_q1z,q1_Vbei;
double	q1_Vbci,q2,q2_Ifi,q2_Vbei,q2_Iri,q2_Vbci,xvar3_q1;
double	xvar1_xvar3,xvar1_q2,xvar4,xvar4_xvar1,xvar4_Vbei,xvar4_Vbci,qb;
double	qb_q1,qb_Vbei,qb_Vbci,qb_xvar4,xvar2_xvar1,xvar2_Vbei,xvar2_Vbci;
double	qb_xvar2,Itzr_Iri,Itzr_qb,Itzf_Ifi,Itzf_qb,argi_Vbep,expi_Vbep;
double	argx,argx_Vbci,expx,expx_argx,expx_Vbci,Ifp,Ifp_expi;
double	Ifp_Vbep,Ifp_expx,Ifp_Vbci,q2p,q2p_Ifp,q2p_Vbep,q2p_Vbci;
double	qbp,qbp_q2p,qbp_Vbep,qbp_Vbci,argi_Vbcp,expi_Vbcp,Irp;
double	Irp_expi,Irp_Vbcp,Iccp_Ifp,Iccp_Irp,Iccp_qbp,argn,argn_Vbei;
double	expn,expn_argn,expn_Vbei,argx_Vbei,expx_Vbei,Ibe_expi,Ibe_expn;
double	Ibe_expx,argi_Vbex,expi_Vbex,argn_Vbex,expn_Vbex,argx_Vbex,expx_Vbex;
double	Ibex_expi,Ibex_expn,Ibex_expx,argn_Vbci,expn_Vbci,Ibcj,Ibcj_expi;
double	Ibcj_Vbci,Ibcj_expn,argn_Vbep,expn_Vbep,Ibep_expi,Ibep_expn,xvar3_vl;
double	avalf,avalf_vl,avalf_Vbci,avalf_xvar4,Igc,Igc_Itzf,Igc_Vbei;
double	Igc_Vbci,Igc_Itzr,Igc_Ibcj,Igc_avalf,Ibc_Ibcj,Ibc_Igc,argx_Vbcx;
double	expx_Vbcx,Kbci,Kbci_expi,Kbci_Vbci,Kbcx,Kbcx_expx,Kbcx_Vbcx;
double	rKp1,rKp1_Kbci,rKp1_Vbci,rKp1_Kbcx,rKp1_Vbcx,xvar1_rKp1,xvar1_Vbcx;
double	Iohm,Iohm_Vrci,Iohm_Kbci,Iohm_Vbci,Iohm_Kbcx,Iohm_Vbcx,Iohm_xvar1;
double	derf,derf_Iohm,derf_Vrci,derf_Vbci,derf_Vbcx,Irci_Iohm,Irci_derf;
double	Irbi_qb,Irbp_qbp,argn_Vbcp,expn_Vbcp,Ibcp_expi,Ibcp_expn,sgIf;
double	rIf,rIf_Ifi,rIf_Vbei,mIf,mIf_rIf,mIf_Vbei,tff;
double	tff_q1,tff_Vbei,tff_Vbci,tff_xvar2,tff_mIf,Qbe_qdbe,Qbe_tff;
double	Qbe_Ifi,Qbe_qb,Qbex_qdbex,Qbc_qdbc,Qbc_Iri,Qbc_Kbci,Qbcx_Kbcx;
double	Qbep_qdbep,Qbep_Ifp,Qbcp_qdbcp;

/*	Function and derivative code */

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

/*	Scale outputs */

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
