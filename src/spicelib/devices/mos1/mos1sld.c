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
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1sLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;
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
    printf("MOS1senload \n");
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
    for( ; model != NULL; model = MOS1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
                here=MOS1nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senload instance name %s\n",here->MOS1name);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->MOS1gNode,here->MOS1dNode,here->MOS1dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS1sNode ,here->MOS1sNodePrime,
                    here->MOS1bNode,here->MOS1senParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++){
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS1states + i);
            }

            *(SaveState + 17) = here->MOS1sourceConductance;  
            *(SaveState + 18) = here->MOS1drainConductance;  
            *(SaveState + 19) = here->MOS1cd;  
            *(SaveState + 20) = here->MOS1cbs;  
            *(SaveState + 21) = here->MOS1cbd;  
            *(SaveState + 22) = here->MOS1gmbs;  
            *(SaveState + 23) = here->MOS1gm;  
            *(SaveState + 24) = here->MOS1gds;  
            *(SaveState + 25) = here->MOS1gbd;  
            *(SaveState + 26) = here->MOS1gbs;  
            *(SaveState + 27) = here->MOS1capbd;  
            *(SaveState + 28) = here->MOS1capbs;  
            *(SaveState + 29) = here->MOS1Cbd;  
            *(SaveState + 30) = here->MOS1Cbdsw;  
            *(SaveState + 31) = here->MOS1Cbs;  
            *(SaveState + 32) = here->MOS1Cbssw;  
            *(SaveState + 33) = here->MOS1f2d;  
            *(SaveState + 34) = here->MOS1f3d;  
            *(SaveState + 35) = here->MOS1f4d;  
            *(SaveState + 36) = here->MOS1f2s;  
            *(SaveState + 37) = here->MOS1f3s;  
            *(SaveState + 38) = here->MOS1f4s;  
            *(SaveState + 39) = here->MOS1cgs;  
            *(SaveState + 40) = here->MOS1cgd;  
            *(SaveState + 41) = here->MOS1cgb;  
            *(SaveState + 42) = here->MOS1vdsat;  
            *(SaveState + 43) = here->MOS1von;  
            save_mode  = here->MOS1mode;  


            if(here->MOS1senParmNo == 0) goto next1;

#ifdef SENSDEBUG
            printf("without perturbation \n");
            printf("gbd =%.5e\n",here->MOS1gbd);
            printf("satCur =%.5e\n",here->MOS1tSatCur);
            printf("satCurDens =%.5e\n",here->MOS1tSatCurDens);
            printf("vbd =%.5e\n",*(ckt->CKTstate0 + here->MOS1vbd));
#endif /* SENSDEBUG */

            cdpr0= here->MOS1cd;
            cspr0= -(here->MOS1cd + here->MOS1cbd + here->MOS1cbs);
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                qgs0 = *(ckt->CKTstate1 + here->MOS1qgs);
                qgd0 = *(ckt->CKTstate1 + here->MOS1qgd);
                qgb0 = *(ckt->CKTstate1 + here->MOS1qgb);
            }
            else{
                qgs0 = *(ckt->CKTstate0 + here->MOS1qgs);
                qgd0 = *(ckt->CKTstate0 + here->MOS1qgd);
                qgb0 = *(ckt->CKTstate0 + here->MOS1qgb);
            }

            here->MOS1senPertFlag = ON;
            error =  MOS1load((GENmodel*)model,ckt);
            if(error) return(error);

            cd0 =  here->MOS1cd ;
            cbd0 = here->MOS1cbd ;
            cbs0 = here->MOS1cbs ;
            gspr0= here->MOS1sourceConductance ;
            gdpr0= here->MOS1drainConductance ;

            qbs0 = *(ckt->CKTstate0 + here->MOS1qbs);
            qbd0 = *(ckt->CKTstate0 + here->MOS1qbd);

            for( flag = 0 ; flag <= 1 ; flag++){
                if(here->MOS1sens_l == 0)
                    if(flag == 0) goto next2;
                if(here->MOS1sens_w == 0)
                    if(flag == 1) goto next2;
                if(flag == 0){
                    A0 = here->MOS1l;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS1l = Apert;
                }
                else{
                    A0 = here->MOS1w;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS1w = Apert;
                    here->MOS1drainArea *= (1 + info->SENpertfac);
                    here->MOS1sourceArea *= (1 + info->SENpertfac);
                    here->MOS1Cbd *= (1 + info->SENpertfac);
                    here->MOS1Cbs *= (1 + info->SENpertfac);
                    if(here->MOS1drainPerimiter){
                        here->MOS1Cbdsw += here->MOS1Cbdsw *
                            DELA/here->MOS1drainPerimiter;
                    }
                    if(here->MOS1sourcePerimiter){
                        here->MOS1Cbssw += here->MOS1Cbssw *
                            DELA/here->MOS1sourcePerimiter;
                    }
                    if(*(ckt->CKTstate0 + here->MOS1vbd) >=
                        here->MOS1tDepCap){
                        arg = 1-model->MOS1fwdCapDepCoeff;
                        sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS1f2d = here->MOS1Cbd*
                                (1-model->MOS1fwdCapDepCoeff*
                                (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS1Cbdsw*(1-model->MOS1fwdCapDepCoeff*
                                (1+model->MOS1bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS1f3d = here->MOS1Cbd * 
                                model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                                here->MOS1tBulkPot
                                + here->MOS1Cbdsw * 
                                model->MOS1bulkJctSideGradingCoeff * sargsw/arg/
                                here->MOS1tBulkPot;
                        here->MOS1f4d = here->MOS1Cbd*
                                here->MOS1tBulkPot*(1-arg*sarg)/
                                (1-model->MOS1bulkJctBotGradingCoeff)
                                + here->MOS1Cbdsw*here->MOS1tBulkPot*
                                (1-arg*sargsw)/
                                (1-model->MOS1bulkJctSideGradingCoeff)
                                -here->MOS1f3d/2*
                                (here->MOS1tDepCap*here->MOS1tDepCap)
                                -here->MOS1tDepCap * here->MOS1f2d;
                    }
                    if(*(ckt->CKTstate0 + here->MOS1vbs) >=
                        here->MOS1tDepCap){
                        arg = 1-model->MOS1fwdCapDepCoeff;
                        sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS1f2s = here->MOS1Cbs*
                                (1-model->MOS1fwdCapDepCoeff*
                                (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS1Cbssw*(1-model->MOS1fwdCapDepCoeff*
                                (1+model->MOS1bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS1f3s = here->MOS1Cbs * 
                                model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                                here->MOS1tBulkPot
                                + here->MOS1Cbssw * 
                                model->MOS1bulkJctSideGradingCoeff * sargsw/arg/
                                here->MOS1tBulkPot;
                        here->MOS1f4s = here->MOS1Cbs*
                                here->MOS1tBulkPot*(1-arg*sarg)/
                                (1-model->MOS1bulkJctBotGradingCoeff)
                                + here->MOS1Cbssw*here->MOS1tBulkPot*
                                (1-arg*sargsw)/
                                (1-model->MOS1bulkJctSideGradingCoeff)
                                -here->MOS1f3s/2*
                                (here->MOS1tDepCap*here->MOS1tDepCap)
                                -here->MOS1tDepCap * here->MOS1f2s;
                    }
                    here->MOS1drainConductance *= Apert/A0;
                    here->MOS1sourceConductance *= Apert/A0;
                }


#ifdef SENSDEBUG
                if(flag == 0)
                    printf("perturbation of l\n");
                if(flag == 1)
                    printf("perturbation of w\n");
#endif /* SENSDEBUG */

                error =  MOS1load((GENmodel*)model,ckt);
                if(error) return(error);

                if(flag == 0){
                    here->MOS1l = A0;
                }
                else{
                    here->MOS1w = A0;
                    here->MOS1drainArea /= (1 + info->SENpertfac);
                    here->MOS1sourceArea /= (1 + info->SENpertfac);
                    here->MOS1drainConductance *= A0/Apert;
                    here->MOS1sourceConductance *= A0/Apert;
                }
                cd =  here->MOS1cd ;
                cbd = here->MOS1cbd ;
                cbs = here->MOS1cbs ;

                gspr= here->MOS1sourceConductance ;
                gdpr= here->MOS1drainConductance ;

                DcdDp = (cd - cd0) * DELAinv;
                DcbsDp = (cbs - cbs0) * DELAinv;
                DcbdDp = (cbd - cbd0) * DELAinv;
                DcbDp = ( DcbsDp + DcbdDp );

                DcdprDp = 0;
                DcsprDp = 0;
                if(here->MOS1dNode != here->MOS1dNodePrime)
                    if(gdpr0) DcdprDp = cdpr0 * (gdpr - gdpr0)/gdpr0 * DELAinv;
                if(here->MOS1sNode != here->MOS1sNodePrime)
                    if(gspr0) DcsprDp = cspr0 * (gspr - gspr0)/gspr0 * DELAinv;

                DcdprmDp = ( - DcdprDp + DcdDp);
                DcsprmDp = ( - DcbsDp - DcdDp - DcbdDp  - DcsprDp);

                if(flag == 0){
                    EffectiveLength = here->MOS1l 
                        - 2*model->MOS1latDiff;
                    if(EffectiveLength == 0){ 
                        DqgsDp = 0;
                        DqgdDp = 0;
                        DqgbDp = 0;
                    }
                    else{
                        DqgsDp = model->MOS1type * qgs0 / EffectiveLength;
                        DqgdDp = model->MOS1type * qgd0 / EffectiveLength;
                        DqgbDp = model->MOS1type * qgb0 / EffectiveLength;
                    }
                }
                else{
                    DqgsDp = model->MOS1type * qgs0 / here->MOS1w;
                    DqgdDp = model->MOS1type * qgd0 / here->MOS1w;
                    DqgbDp = model->MOS1type * qgb0 / here->MOS1w;
                }


                qbd = *(ckt->CKTstate0 + here->MOS1qbd);
                qbs = *(ckt->CKTstate0 + here->MOS1qbs);

                DqbsDp = model->MOS1type * (qbs - qbs0)*DELAinv;
                DqbdDp = model->MOS1type * (qbd - qbd0)*DELAinv;

                if(flag == 0){
                    *(here->MOS1dphigs_dl) = DqgsDp;
                    *(here->MOS1dphigd_dl) = DqgdDp;
                    *(here->MOS1dphibs_dl) = DqbsDp;
                    *(here->MOS1dphibd_dl) = DqbdDp;
                    *(here->MOS1dphigb_dl) = DqgbDp;
                }
                else{
                    *(here->MOS1dphigs_dw) = DqgsDp;
                    *(here->MOS1dphigd_dw) = DqgdDp;
                    *(here->MOS1dphibs_dw) = DqbsDp;
                    *(here->MOS1dphibd_dw) = DqbdDp;
                    *(here->MOS1dphigb_dw) = DqgbDp;
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
                printf("tdepCap = %.7e \n",here->MOS1tDepCap); 
                printf("\n");
#endif /* SENSDEBUG*/
                if((info->SENmode == TRANSEN) &&
                    (ckt->CKTmode & MODEINITTRAN))
                    goto next2;

                /*
                                 *   load RHS matrix
                                 */

                if(flag == 0){
                    *(info->SEN_RHS[here->MOS1bNode] + here->MOS1senParmNo) -= 
                            model->MOS1type * DcbDp;
                    *(info->SEN_RHS[here->MOS1dNode] + here->MOS1senParmNo) -= 
                            model->MOS1type * DcdprDp;
                    *(info->SEN_RHS[here->MOS1dNodePrime] +
                            here->MOS1senParmNo) -= model->MOS1type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS1sNode] + here->MOS1senParmNo) -= 
                            model->MOS1type * DcsprDp;
                    *(info->SEN_RHS[here->MOS1sNodePrime] + 
                            here->MOS1senParmNo) -= model->MOS1type * DcsprmDp;
                }
                else{  
                    offset = here->MOS1sens_l;

                    *(info->SEN_RHS[here->MOS1bNode] + here->MOS1senParmNo + 
                            offset) -= model->MOS1type * DcbDp;
                    *(info->SEN_RHS[here->MOS1dNode] + here->MOS1senParmNo + 
                            offset) -= model->MOS1type * DcdprDp;
                    *(info->SEN_RHS[here->MOS1dNodePrime] + here->MOS1senParmNo
                            + offset) -= model->MOS1type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS1sNode] + here->MOS1senParmNo +
                            offset) -= model->MOS1type * DcsprDp;
                    *(info->SEN_RHS[here->MOS1sNodePrime] + here->MOS1senParmNo
                            + offset) -= model->MOS1type * DcsprmDp;
                }
#ifdef SENSDEBUG
                printf("after loading\n");
                if(flag == 0){
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1bNode] +
                            here->MOS1senParmNo));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1dNode] +
                            here->MOS1senParmNo));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1sNode] +
                            here->MOS1senParmNo));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1dNodePrime] +
                            here->MOS1senParmNo));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1sNodePrime] +
                            here->MOS1senParmNo));
                    printf("\n");
                }
                else{
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1bNode] + 
                            here->MOS1senParmNo + here->MOS1sens_l));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1dNode] + 
                            here->MOS1senParmNo + here->MOS1sens_l));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1sNode] + 
                            here->MOS1senParmNo + here->MOS1sens_l));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1dNodePrime] + 
                            here->MOS1senParmNo + here->MOS1sens_l));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS1sNodePrime] + 
                            here->MOS1senParmNo + here->MOS1sens_l));
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
                        *(info->SEN_RHS[here->MOS1bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1sNodePrime] + iparmno));
                printf("\n");
#endif /* SENSDEBUG */
                Osxpgs = tag0 * *(ckt->CKTstate1 + here->MOS1sensxpgs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS1sensxpgs +
                        10*(iparmno - 1) + 1);

                Osxpgd = tag0 * *(ckt->CKTstate1 + here->MOS1sensxpgd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS1sensxpgd +
                        10*(iparmno - 1) + 1);

                Osxpbs = tag0 * *(ckt->CKTstate1 + here->MOS1sensxpbs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS1sensxpbs +
                        10*(iparmno - 1) + 1);

                Osxpbd =tag0 * *(ckt->CKTstate1 + here->MOS1sensxpbd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS1sensxpbd +
                        10*(iparmno - 1) + 1);
                Osxpgb = tag0 * *(ckt->CKTstate1 + here->MOS1sensxpgb +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS1sensxpgb +
                        10*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("sxpgs=%.7e,sdgs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS1sensxpgs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS1sensxpgs + 10*(iparmno - 1) + 1));
                printf("sxpgd=%.7e,sdgd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS1sensxpgd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS1sensxpgd + 10*(iparmno - 1) + 1));
                printf("sxpbs=%.7e,sdbs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS1sensxpbs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS1sensxpbs + 10*(iparmno - 1) + 1));
                printf("sxpbd=%.7e,sdbd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS1sensxpbd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS1sensxpbd + 10*(iparmno - 1) + 1));
                printf("sxpgb=%.7e,sdgb=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS1sensxpgb + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS1sensxpgb + 10*(iparmno - 1) + 1));
                printf("before loading DqDp\n");
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
                printf("\n");
#endif /* SENSDEBUG */
                if(here->MOS1sens_l && (iparmno == here->MOS1senParmNo)){
                    Osxpgs -= tag0 * *(here->MOS1dphigs_dl);
                    Osxpgd -= tag0 * *(here->MOS1dphigd_dl);
                    Osxpbs -= tag0 * *(here->MOS1dphibs_dl);
                    Osxpbd -= tag0 * *(here->MOS1dphibd_dl);
                    Osxpgb -= tag0 * *(here->MOS1dphigb_dl);
                }
                if(here->MOS1sens_w &&
                        (iparmno == (here->MOS1senParmNo +
                            (int) here->MOS1sens_l))){
                    Osxpgs -= tag0 * *(here->MOS1dphigs_dw);
                    Osxpgd -= tag0 * *(here->MOS1dphigd_dw);
                    Osxpbs -= tag0 * *(here->MOS1dphibs_dw);
                    Osxpbd -= tag0 * *(here->MOS1dphibd_dw);
                    Osxpgb -= tag0 * *(here->MOS1dphigb_dw);
                }
#ifdef SENSDEBUG
                printf("after loading DqDp\n");
                printf("DqgsDp=%.7e",DqgsDp);
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->MOS1bNode] + iparmno) += 
                        Osxpbs + Osxpbd -Osxpgb;
                *(info->SEN_RHS[here->MOS1gNode] + iparmno) += 
                        Osxpgs + Osxpgd + Osxpgb;

                *(info->SEN_RHS[here->MOS1dNodePrime] + iparmno) -= 
                        Osxpgd + Osxpbd ;
                *(info->SEN_RHS[here->MOS1sNodePrime] + iparmno) -= 
                        Osxpgs + Osxpbs;
#ifdef SENSDEBUG
                printf("after capacitive currents\n");
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS1sNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
restore:    /* put the unperturbed values back into the state vector */
            for(i=0; i <= 16; i++)
                *(ckt->CKTstate0 + here->MOS1states + i) = *(SaveState + i);
            here->MOS1sourceConductance = *(SaveState + 17) ;   
            here->MOS1drainConductance = *(SaveState + 18) ; 
            here->MOS1cd =  *(SaveState + 19) ;  
            here->MOS1cbs =  *(SaveState + 20) ;  
            here->MOS1cbd =  *(SaveState + 21) ;  
            here->MOS1gmbs =  *(SaveState + 22) ;  
            here->MOS1gm =  *(SaveState + 23) ;  
            here->MOS1gds =  *(SaveState + 24) ;  
            here->MOS1gbd =  *(SaveState + 25) ;  
            here->MOS1gbs =  *(SaveState + 26) ;  
            here->MOS1capbd =  *(SaveState + 27) ;  
            here->MOS1capbs =  *(SaveState + 28) ;  
            here->MOS1Cbd =  *(SaveState + 29) ;  
            here->MOS1Cbdsw =  *(SaveState + 30) ;  
            here->MOS1Cbs =  *(SaveState + 31) ;  
            here->MOS1Cbssw =  *(SaveState + 32) ;  
            here->MOS1f2d =  *(SaveState + 33) ;  
            here->MOS1f3d =  *(SaveState + 34) ;  
            here->MOS1f4d =  *(SaveState + 35) ;  
            here->MOS1f2s =  *(SaveState + 36) ;  
            here->MOS1f3s =  *(SaveState + 37) ;  
            here->MOS1f4s =  *(SaveState + 38) ;  
            here->MOS1cgs = *(SaveState + 39) ;  
            here->MOS1cgd = *(SaveState + 40) ;  
            here->MOS1cgb = *(SaveState + 41) ;  
            here->MOS1vdsat = *(SaveState + 42) ;  
            here->MOS1von = *(SaveState + 43) ;   
            here->MOS1mode = save_mode ;  

            here->MOS1senPertFlag = OFF;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("MOS1senload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

