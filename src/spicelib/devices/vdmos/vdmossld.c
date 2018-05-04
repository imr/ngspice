/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;
    double   SaveState[44];
    int    save_mode;
    int    i;
    int    iparmno;
    int    error;
    int    flag;
    double A0;
    double DELA;
    double Apert;
    double DELAinv;
    double gspr0;
    double gspr;
    double gdpr0;
    double gdpr;
    double cdpr0;
    double cspr0;
    double cd0;
    double cbd0;
    double cbs0;
    double cd;
    double cbd;
    double cbs;
    double DcdprDp;
    double DcsprDp;
    double DcbDp;
    double DcdDp;
    double DcbsDp;
    double DcbdDp;
    double DcdprmDp;
    double DcsprmDp;
    double qgs0;
    double qgd0;
    double qgb0;
    double qbd0;
    double qbd;
    double qbs0;
    double qbs;
    double DqgsDp;
    double DqgdDp;
    double DqgbDp;
    double DqbdDp;
    double DqbsDp;
    double Osxpgs;
    double Osxpgd;
    double Osxpgb;
    double Osxpbd;
    double Osxpbs;
    double tag0;
    double tag1;
    double arg;
    double sarg;
    double sargsw;
    int offset;
    double EffectiveLength;
    SENstruct *info;

#ifdef SENSDEBUG
    printf("VDMOSsenload \n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
    printf("CKTorder = %d\n",ckt->CKTorder);
#endif /* SENSDEBUG */

    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;

    tag0 = ckt->CKTag[0]; 
    tag1 = ckt->CKTag[1];
    if(ckt->CKTorder == 1){
        tag1 = 0;
    }

    /*  loop through all the models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

#ifdef SENSDEBUG
            printf("senload instance name %s\n",here->VDMOSname);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->VDMOSgNode,here->VDMOSdNode,here->VDMOSdNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->VDMOSsNode ,here->VDMOSsNodePrime,
                    here->VDMOSbNode,here->VDMOSsenParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++){
                *(SaveState + i) = *(ckt->CKTstate0 + here->VDMOSstates + i);
            }

            *(SaveState + 17) = here->VDMOSsourceConductance;  
            *(SaveState + 18) = here->VDMOSdrainConductance;  
            *(SaveState + 19) = here->VDMOScd;  
            *(SaveState + 20) = here->VDMOScbs;  
            *(SaveState + 21) = here->VDMOScbd;  
            *(SaveState + 22) = here->VDMOSgmbs;  
            *(SaveState + 23) = here->VDMOSgm;  
            *(SaveState + 24) = here->VDMOSgds;  
            *(SaveState + 25) = here->VDMOSgbd;  
            *(SaveState + 26) = here->VDMOSgbs;  
            *(SaveState + 27) = here->VDMOScapbd;  
            *(SaveState + 28) = here->VDMOScapbs;  
            *(SaveState + 29) = here->VDMOSCbd;  
            *(SaveState + 30) = here->VDMOSCbdsw;  
            *(SaveState + 31) = here->VDMOSCbs;  
            *(SaveState + 32) = here->VDMOSCbssw;  
            *(SaveState + 33) = here->VDMOSf2d;  
            *(SaveState + 34) = here->VDMOSf3d;  
            *(SaveState + 35) = here->VDMOSf4d;  
            *(SaveState + 36) = here->VDMOSf2s;  
            *(SaveState + 37) = here->VDMOSf3s;  
            *(SaveState + 38) = here->VDMOSf4s;  
            *(SaveState + 39) = here->VDMOScgs;  
            *(SaveState + 40) = here->VDMOScgd;  
            *(SaveState + 41) = here->VDMOScgb;  
            *(SaveState + 42) = here->VDMOSvdsat;  
            *(SaveState + 43) = here->VDMOSvon;  
            save_mode  = here->VDMOSmode;  


            if(here->VDMOSsenParmNo == 0) goto next1;

#ifdef SENSDEBUG
            printf("without perturbation \n");
            printf("gbd =%.5e\n",here->VDMOSgbd);
            printf("satCur =%.5e\n",here->VDMOStSatCur);
            printf("satCurDens =%.5e\n",here->VDMOStSatCurDens);
            printf("vbd =%.5e\n",*(ckt->CKTstate0 + here->VDMOSvbd));
#endif /* SENSDEBUG */

            cdpr0= here->VDMOScd;
            cspr0= -(here->VDMOScd + here->VDMOScbd + here->VDMOScbs);
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                qgs0 = *(ckt->CKTstate1 + here->VDMOSqgs);
                qgd0 = *(ckt->CKTstate1 + here->VDMOSqgd);
                qgb0 = *(ckt->CKTstate1 + here->VDMOSqgb);
            }
            else{
                qgs0 = *(ckt->CKTstate0 + here->VDMOSqgs);
                qgd0 = *(ckt->CKTstate0 + here->VDMOSqgd);
                qgb0 = *(ckt->CKTstate0 + here->VDMOSqgb);
            }

            here->VDMOSsenPertFlag = ON;
            error =  VDMOSload((GENmodel*)model,ckt);
            if(error) return(error);

            cd0 =  here->VDMOScd ;
            cbd0 = here->VDMOScbd ;
            cbs0 = here->VDMOScbs ;
            gspr0= here->VDMOSsourceConductance ;
            gdpr0= here->VDMOSdrainConductance ;

            qbs0 = *(ckt->CKTstate0 + here->VDMOSqbs);
            qbd0 = *(ckt->CKTstate0 + here->VDMOSqbd);

            for( flag = 0 ; flag <= 1 ; flag++){
                if(here->VDMOSsens_l == 0)
                    if(flag == 0) goto next2;
                if(here->VDMOSsens_w == 0)
                    if(flag == 1) goto next2;
                if(flag == 0){
                    A0 = here->VDMOSl;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->VDMOSl = Apert;
                }
                else{
                    A0 = here->VDMOSw;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->VDMOSw = Apert;
                    here->VDMOSdrainArea *= (1 + info->SENpertfac);
                    here->VDMOSsourceArea *= (1 + info->SENpertfac);
                    here->VDMOSCbd *= (1 + info->SENpertfac);
                    here->VDMOSCbs *= (1 + info->SENpertfac);
                    if(here->VDMOSdrainPerimiter){
                        here->VDMOSCbdsw += here->VDMOSCbdsw *
                            DELA/here->VDMOSdrainPerimiter;
                    }
                    if(here->VDMOSsourcePerimiter){
                        here->VDMOSCbssw += here->VDMOSCbssw *
                            DELA/here->VDMOSsourcePerimiter;
                    }
                    if(*(ckt->CKTstate0 + here->VDMOSvbd) >=
                        here->VDMOStDepCap){
                        arg = 1-model->VDMOSfwdCapDepCoeff;
                        sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->VDMOSf2d = here->VDMOSCbd*
                                (1-model->VDMOSfwdCapDepCoeff*
                                (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                                +  here->VDMOSCbdsw*(1-model->VDMOSfwdCapDepCoeff*
                                (1+model->VDMOSbulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->VDMOSf3d = here->VDMOSCbd * 
                                model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                                here->VDMOStBulkPot
                                + here->VDMOSCbdsw * 
                                model->VDMOSbulkJctSideGradingCoeff * sargsw/arg/
                                here->VDMOStBulkPot;
                        here->VDMOSf4d = here->VDMOSCbd*
                                here->VDMOStBulkPot*(1-arg*sarg)/
                                (1-model->VDMOSbulkJctBotGradingCoeff)
                                + here->VDMOSCbdsw*here->VDMOStBulkPot*
                                (1-arg*sargsw)/
                                (1-model->VDMOSbulkJctSideGradingCoeff)
                                -here->VDMOSf3d/2*
                                (here->VDMOStDepCap*here->VDMOStDepCap)
                                -here->VDMOStDepCap * here->VDMOSf2d;
                    }
                    if(*(ckt->CKTstate0 + here->VDMOSvbs) >=
                        here->VDMOStDepCap){
                        arg = 1-model->VDMOSfwdCapDepCoeff;
                        sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->VDMOSf2s = here->VDMOSCbs*
                                (1-model->VDMOSfwdCapDepCoeff*
                                (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                                +  here->VDMOSCbssw*(1-model->VDMOSfwdCapDepCoeff*
                                (1+model->VDMOSbulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->VDMOSf3s = here->VDMOSCbs * 
                                model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                                here->VDMOStBulkPot
                                + here->VDMOSCbssw * 
                                model->VDMOSbulkJctSideGradingCoeff * sargsw/arg/
                                here->VDMOStBulkPot;
                        here->VDMOSf4s = here->VDMOSCbs*
                                here->VDMOStBulkPot*(1-arg*sarg)/
                                (1-model->VDMOSbulkJctBotGradingCoeff)
                                + here->VDMOSCbssw*here->VDMOStBulkPot*
                                (1-arg*sargsw)/
                                (1-model->VDMOSbulkJctSideGradingCoeff)
                                -here->VDMOSf3s/2*
                                (here->VDMOStDepCap*here->VDMOStDepCap)
                                -here->VDMOStDepCap * here->VDMOSf2s;
                    }
                    here->VDMOSdrainConductance *= Apert/A0;
                    here->VDMOSsourceConductance *= Apert/A0;
                }


#ifdef SENSDEBUG
                if(flag == 0)
                    printf("perturbation of l\n");
                if(flag == 1)
                    printf("perturbation of w\n");
#endif /* SENSDEBUG */

                error =  VDMOSload((GENmodel*)model,ckt);
                if(error) return(error);

                if(flag == 0){
                    here->VDMOSl = A0;
                }
                else{
                    here->VDMOSw = A0;
                    here->VDMOSdrainArea /= (1 + info->SENpertfac);
                    here->VDMOSsourceArea /= (1 + info->SENpertfac);
                    here->VDMOSdrainConductance *= A0/Apert;
                    here->VDMOSsourceConductance *= A0/Apert;
                }
                cd =  here->VDMOScd ;
                cbd = here->VDMOScbd ;
                cbs = here->VDMOScbs ;

                gspr= here->VDMOSsourceConductance ;
                gdpr= here->VDMOSdrainConductance ;

                DcdDp = (cd - cd0) * DELAinv;
                DcbsDp = (cbs - cbs0) * DELAinv;
                DcbdDp = (cbd - cbd0) * DELAinv;
                DcbDp = ( DcbsDp + DcbdDp );

                DcdprDp = 0;
                DcsprDp = 0;
                if(here->VDMOSdNode != here->VDMOSdNodePrime)
                    if(gdpr0) DcdprDp = cdpr0 * (gdpr - gdpr0)/gdpr0 * DELAinv;
                if(here->VDMOSsNode != here->VDMOSsNodePrime)
                    if(gspr0) DcsprDp = cspr0 * (gspr - gspr0)/gspr0 * DELAinv;

                DcdprmDp = ( - DcdprDp + DcdDp);
                DcsprmDp = ( - DcbsDp - DcdDp - DcbdDp  - DcsprDp);

                if(flag == 0){
                    EffectiveLength = here->VDMOSl 
                        - 2*model->VDMOSlatDiff;
                    if(EffectiveLength == 0){ 
                        DqgsDp = 0;
                        DqgdDp = 0;
                        DqgbDp = 0;
                    }
                    else{
                        DqgsDp = model->VDMOStype * qgs0 / EffectiveLength;
                        DqgdDp = model->VDMOStype * qgd0 / EffectiveLength;
                        DqgbDp = model->VDMOStype * qgb0 / EffectiveLength;
                    }
                }
                else{
                    DqgsDp = model->VDMOStype * qgs0 / here->VDMOSw;
                    DqgdDp = model->VDMOStype * qgd0 / here->VDMOSw;
                    DqgbDp = model->VDMOStype * qgb0 / here->VDMOSw;
                }


                qbd = *(ckt->CKTstate0 + here->VDMOSqbd);
                qbs = *(ckt->CKTstate0 + here->VDMOSqbs);

                DqbsDp = model->VDMOStype * (qbs - qbs0)*DELAinv;
                DqbdDp = model->VDMOStype * (qbd - qbd0)*DELAinv;

                if(flag == 0){
                    *(here->VDMOSdphigs_dl) = DqgsDp;
                    *(here->VDMOSdphigd_dl) = DqgdDp;
                    *(here->VDMOSdphibs_dl) = DqbsDp;
                    *(here->VDMOSdphibd_dl) = DqbdDp;
                    *(here->VDMOSdphigb_dl) = DqgbDp;
                }
                else{
                    *(here->VDMOSdphigs_dw) = DqgsDp;
                    *(here->VDMOSdphigd_dw) = DqgdDp;
                    *(here->VDMOSdphibs_dw) = DqbsDp;
                    *(here->VDMOSdphibd_dw) = DqbdDp;
                    *(here->VDMOSdphigb_dw) = DqgbDp;
                }


#ifdef SENSDEBUG
                printf("CKTag[0]=%.7e,CKTag[1]=%.7e,flag= %d\n",
                        ckt->CKTag[0],ckt->CKTag[1],flag);
                printf("cd0 = %.7e ,cd = %.7e,\n",cd0,cd); 
                printf("cbs0 = %.7e ,cbs = %.7e,\n",cbs0,cbs); 
                printf("cbd0 = %.7e ,cbd = %.7e,\n",cbd0,cbd); 
                printf("DcdprmDp = %.7e,\n",DcdprmDp); 
                printf("DcsprmDp = %.7e,\n",DcsprmDp); 
                printf("DcdprDp = %.7e,\n",DcdprDp); 
                printf("DcsprDp = %.7e,\n",DcsprDp); 
                printf("qgs0 = %.7e \n",qgs0); 
                printf("qgd0 = %.7e \n",qgd0); 
                printf("qgb0 = %.7e \n",qgb0); 
                printf("qbs0 = %.7e ,qbs = %.7e,\n",qbs0,qbs); 
                printf("qbd0 = %.7e ,qbd = %.7e,\n",qbd0,qbd); 
                printf("DqgsDp = %.7e \n",DqgsDp); 
                printf("DqgdDp = %.7e \n",DqgdDp); 
                printf("DqgbDp = %.7e \n",DqgbDp); 
                printf("DqbsDp = %.7e \n",DqbsDp); 
                printf("DqbdDp = %.7e \n",DqbdDp); 
                printf("EffectiveLength = %.7e \n",EffectiveLength); 
                printf("tdepCap = %.7e \n",here->VDMOStDepCap); 
                printf("\n");
#endif /* SENSDEBUG*/
                if((info->SENmode == TRANSEN) &&
                    (ckt->CKTmode & MODEINITTRAN))
                    goto next2;

                /*
                                 *   load RHS matrix
                                 */

                if(flag == 0){
                    *(info->SEN_RHS[here->VDMOSbNode] + here->VDMOSsenParmNo) -= 
                            model->VDMOStype * DcbDp;
                    *(info->SEN_RHS[here->VDMOSdNode] + here->VDMOSsenParmNo) -= 
                            model->VDMOStype * DcdprDp;
                    *(info->SEN_RHS[here->VDMOSdNodePrime] +
                            here->VDMOSsenParmNo) -= model->VDMOStype * DcdprmDp;
                    *(info->SEN_RHS[here->VDMOSsNode] + here->VDMOSsenParmNo) -= 
                            model->VDMOStype * DcsprDp;
                    *(info->SEN_RHS[here->VDMOSsNodePrime] + 
                            here->VDMOSsenParmNo) -= model->VDMOStype * DcsprmDp;
                }
                else{  
                    offset = here->VDMOSsens_l;

                    *(info->SEN_RHS[here->VDMOSbNode] + here->VDMOSsenParmNo + 
                            offset) -= model->VDMOStype * DcbDp;
                    *(info->SEN_RHS[here->VDMOSdNode] + here->VDMOSsenParmNo + 
                            offset) -= model->VDMOStype * DcdprDp;
                    *(info->SEN_RHS[here->VDMOSdNodePrime] + here->VDMOSsenParmNo
                            + offset) -= model->VDMOStype * DcdprmDp;
                    *(info->SEN_RHS[here->VDMOSsNode] + here->VDMOSsenParmNo +
                            offset) -= model->VDMOStype * DcsprDp;
                    *(info->SEN_RHS[here->VDMOSsNodePrime] + here->VDMOSsenParmNo
                            + offset) -= model->VDMOStype * DcsprmDp;
                }
#ifdef SENSDEBUG
                printf("after loading\n");
                if(flag == 0){
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSbNode] +
                            here->VDMOSsenParmNo));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSdNode] +
                            here->VDMOSsenParmNo));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSsNode] +
                            here->VDMOSsenParmNo));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSdNodePrime] +
                            here->VDMOSsenParmNo));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSsNodePrime] +
                            here->VDMOSsenParmNo));
                    printf("\n");
                }
                else{
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSbNode] + 
                            here->VDMOSsenParmNo + here->VDMOSsens_l));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSdNode] + 
                            here->VDMOSsenParmNo + here->VDMOSsens_l));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSsNode] + 
                            here->VDMOSsenParmNo + here->VDMOSsens_l));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSdNodePrime] + 
                            here->VDMOSsenParmNo + here->VDMOSsens_l));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->VDMOSsNodePrime] + 
                            here->VDMOSsenParmNo + here->VDMOSsens_l));
                }
#endif /* SENSDEBUG*/
next2:                   
                ;
            }
next1:
            if((info->SENmode == DCSEN) ||
                (ckt->CKTmode&MODETRANOP) ) goto restore;
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)) goto restore;
            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
#ifdef SENSDEBUG
                printf("after conductive currents\n");
                printf("iparmno = %d\n",iparmno);
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSbNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSdNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSdNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSsNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSsNodePrime] + iparmno));
                printf("\n");
#endif /* SENSDEBUG */
                Osxpgs = tag0 * *(ckt->CKTstate1 + here->VDMOSsensxpgs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->VDMOSsensxpgs +
                        10*(iparmno - 1) + 1);

                Osxpgd = tag0 * *(ckt->CKTstate1 + here->VDMOSsensxpgd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->VDMOSsensxpgd +
                        10*(iparmno - 1) + 1);

                Osxpbs = tag0 * *(ckt->CKTstate1 + here->VDMOSsensxpbs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->VDMOSsensxpbs +
                        10*(iparmno - 1) + 1);

                Osxpbd =tag0 * *(ckt->CKTstate1 + here->VDMOSsensxpbd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->VDMOSsensxpbd +
                        10*(iparmno - 1) + 1);
                Osxpgb = tag0 * *(ckt->CKTstate1 + here->VDMOSsensxpgb +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->VDMOSsensxpgb +
                        10*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("sxpgs=%.7e,sdgs=%.7e\n",
                        *(ckt->CKTstate1 + here->VDMOSsensxpgs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->VDMOSsensxpgs + 10*(iparmno - 1) + 1));
                printf("sxpgd=%.7e,sdgd=%.7e\n",
                        *(ckt->CKTstate1 + here->VDMOSsensxpgd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->VDMOSsensxpgd + 10*(iparmno - 1) + 1));
                printf("sxpbs=%.7e,sdbs=%.7e\n",
                        *(ckt->CKTstate1 + here->VDMOSsensxpbs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->VDMOSsensxpbs + 10*(iparmno - 1) + 1));
                printf("sxpbd=%.7e,sdbd=%.7e\n",
                        *(ckt->CKTstate1 + here->VDMOSsensxpbd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->VDMOSsensxpbd + 10*(iparmno - 1) + 1));
                printf("sxpgb=%.7e,sdgb=%.7e\n",
                        *(ckt->CKTstate1 + here->VDMOSsensxpgb + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->VDMOSsensxpgb + 10*(iparmno - 1) + 1));
                printf("before loading DqDp\n");
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
                printf("\n");
#endif /* SENSDEBUG */
                if(here->VDMOSsens_l && (iparmno == here->VDMOSsenParmNo)){
                    Osxpgs -= tag0 * *(here->VDMOSdphigs_dl);
                    Osxpgd -= tag0 * *(here->VDMOSdphigd_dl);
                    Osxpbs -= tag0 * *(here->VDMOSdphibs_dl);
                    Osxpbd -= tag0 * *(here->VDMOSdphibd_dl);
                    Osxpgb -= tag0 * *(here->VDMOSdphigb_dl);
                }
                if(here->VDMOSsens_w && 
                        (iparmno == (here->VDMOSsenParmNo + here->VDMOSsens_l))){
                    Osxpgs -= tag0 * *(here->VDMOSdphigs_dw);
                    Osxpgd -= tag0 * *(here->VDMOSdphigd_dw);
                    Osxpbs -= tag0 * *(here->VDMOSdphibs_dw);
                    Osxpbd -= tag0 * *(here->VDMOSdphibd_dw);
                    Osxpgb -= tag0 * *(here->VDMOSdphigb_dw);
                }
#ifdef SENSDEBUG
                printf("after loading DqDp\n");
                printf("DqgsDp=%.7e",DqgsDp);
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->VDMOSbNode] + iparmno) += 
                        Osxpbs + Osxpbd -Osxpgb;
                *(info->SEN_RHS[here->VDMOSgNode] + iparmno) += 
                        Osxpgs + Osxpgd + Osxpgb;

                *(info->SEN_RHS[here->VDMOSdNodePrime] + iparmno) -= 
                        Osxpgd + Osxpbd ;
                *(info->SEN_RHS[here->VDMOSsNodePrime] + iparmno) -= 
                        Osxpgs + Osxpbs;
#ifdef SENSDEBUG
                printf("after capacitive currents\n");
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSbNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSdNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSdNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSsNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->VDMOSsNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
restore:    /* put the unperturbed values back into the state vector */
            for(i=0; i <= 16; i++)
                *(ckt->CKTstate0 + here->VDMOSstates + i) = *(SaveState + i);
            here->VDMOSsourceConductance = *(SaveState + 17) ;   
            here->VDMOSdrainConductance = *(SaveState + 18) ; 
            here->VDMOScd =  *(SaveState + 19) ;  
            here->VDMOScbs =  *(SaveState + 20) ;  
            here->VDMOScbd =  *(SaveState + 21) ;  
            here->VDMOSgmbs =  *(SaveState + 22) ;  
            here->VDMOSgm =  *(SaveState + 23) ;  
            here->VDMOSgds =  *(SaveState + 24) ;  
            here->VDMOSgbd =  *(SaveState + 25) ;  
            here->VDMOSgbs =  *(SaveState + 26) ;  
            here->VDMOScapbd =  *(SaveState + 27) ;  
            here->VDMOScapbs =  *(SaveState + 28) ;  
            here->VDMOSCbd =  *(SaveState + 29) ;  
            here->VDMOSCbdsw =  *(SaveState + 30) ;  
            here->VDMOSCbs =  *(SaveState + 31) ;  
            here->VDMOSCbssw =  *(SaveState + 32) ;  
            here->VDMOSf2d =  *(SaveState + 33) ;  
            here->VDMOSf3d =  *(SaveState + 34) ;  
            here->VDMOSf4d =  *(SaveState + 35) ;  
            here->VDMOSf2s =  *(SaveState + 36) ;  
            here->VDMOSf3s =  *(SaveState + 37) ;  
            here->VDMOSf4s =  *(SaveState + 38) ;  
            here->VDMOScgs = *(SaveState + 39) ;  
            here->VDMOScgd = *(SaveState + 40) ;  
            here->VDMOScgb = *(SaveState + 41) ;  
            here->VDMOSvdsat = *(SaveState + 42) ;  
            here->VDMOSvon = *(SaveState + 43) ;   
            here->VDMOSmode = save_mode ;  

            here->VDMOSsenPertFlag = OFF;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("VDMOSsenload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

