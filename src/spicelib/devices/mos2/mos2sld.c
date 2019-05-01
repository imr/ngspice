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
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2sLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
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
    printf("MOS2senload \n");
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
    for( ; model != NULL; model = MOS2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senload instance name %s\n",here->MOS2name);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->MOS2gNode,here->MOS2dNode,here->MOS2dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS2sNode ,here->MOS2sNodePrime,
                    here->MOS2bNode,here->MOS2senParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++){
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS2states + i);
            }

            *(SaveState + 17) = here->MOS2sourceConductance;  
            *(SaveState + 18) = here->MOS2drainConductance;  
            *(SaveState + 19) = here->MOS2cd;  
            *(SaveState + 20) = here->MOS2cbs;  
            *(SaveState + 21) = here->MOS2cbd;  
            *(SaveState + 22) = here->MOS2gmbs;  
            *(SaveState + 23) = here->MOS2gm;  
            *(SaveState + 24) = here->MOS2gds;  
            *(SaveState + 25) = here->MOS2gbd;  
            *(SaveState + 26) = here->MOS2gbs;  
            *(SaveState + 27) = here->MOS2capbd;  
            *(SaveState + 28) = here->MOS2capbs;  
            *(SaveState + 29) = here->MOS2Cbd;  
            *(SaveState + 30) = here->MOS2Cbdsw;  
            *(SaveState + 31) = here->MOS2Cbs;  
            *(SaveState + 32) = here->MOS2Cbssw;  
            *(SaveState + 33) = here->MOS2f2d;  
            *(SaveState + 34) = here->MOS2f3d;  
            *(SaveState + 35) = here->MOS2f4d;  
            *(SaveState + 36) = here->MOS2f2s;  
            *(SaveState + 37) = here->MOS2f3s;  
            *(SaveState + 38) = here->MOS2f4s;  
            *(SaveState + 39) = here->MOS2cgs;  
            *(SaveState + 40) = here->MOS2cgd;  
            *(SaveState + 41) = here->MOS2cgb;  
            *(SaveState + 42) = here->MOS2vdsat;  
            *(SaveState + 43) = here->MOS2von;  
            save_mode  = here->MOS2mode;  


            if(here->MOS2senParmNo == 0) goto next1;

#ifdef SENSDEBUG
            printf("without perturbation \n");
            printf("gbd =%.5e\n",here->MOS2gbd);
            printf("satCur =%.5e\n",here->MOS2tSatCur);
            printf("satCurDens =%.5e\n",here->MOS2tSatCurDens);
            printf("vbd =%.5e\n",*(ckt->CKTstate0 + here->MOS2vbd));
#endif /* SENSDEBUG */

            cdpr0= here->MOS2cd;
            cspr0= -(here->MOS2cd + here->MOS2cbd + here->MOS2cbs);
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                qgs0 = *(ckt->CKTstate1 + here->MOS2qgs);
                qgd0 = *(ckt->CKTstate1 + here->MOS2qgd);
                qgb0 = *(ckt->CKTstate1 + here->MOS2qgb);
            }
            else{
                qgs0 = *(ckt->CKTstate0 + here->MOS2qgs);
                qgd0 = *(ckt->CKTstate0 + here->MOS2qgd);
                qgb0 = *(ckt->CKTstate0 + here->MOS2qgb);
            }

            here->MOS2senPertFlag = ON;
            error =  MOS2load((GENmodel*)model,ckt);
            if(error) return(error);

            cd0 =  here->MOS2cd ;
            cbd0 = here->MOS2cbd ;
            cbs0 = here->MOS2cbs ;
            gspr0= here->MOS2sourceConductance ;
            gdpr0= here->MOS2drainConductance ;

            qbs0 = *(ckt->CKTstate0 + here->MOS2qbs);
            qbd0 = *(ckt->CKTstate0 + here->MOS2qbd);

            for( flag = 0 ; flag <= 1 ; flag++){
                if(here->MOS2sens_l == 0)
                    if(flag == 0) goto next2;
                if(here->MOS2sens_w == 0)
                    if(flag == 1) goto next2;
                if(flag == 0){
                    A0 = here->MOS2l;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS2l = Apert;
                }
                else{
                    A0 = here->MOS2w;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS2w = Apert;
                    here->MOS2drainArea *= (1 + info->SENpertfac);
                    here->MOS2sourceArea *= (1 + info->SENpertfac);
                    here->MOS2Cbd *= (1 + info->SENpertfac);
                    here->MOS2Cbs *= (1 + info->SENpertfac);
                    if(here->MOS2drainPerimiter){
                        here->MOS2Cbdsw += here->MOS2Cbdsw *
                            DELA/here->MOS2drainPerimiter;
                    }
                    if(here->MOS2sourcePerimiter){
                        here->MOS2Cbssw += here->MOS2Cbssw *
                            DELA/here->MOS2sourcePerimiter;
                    }
                    if(*(ckt->CKTstate0 + here->MOS2vbd) >=
                        here->MOS2tDepCap){
                        arg = 1-model->MOS2fwdCapDepCoeff;
                        sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS2f2d = here->MOS2Cbd*
                                (1-model->MOS2fwdCapDepCoeff*
                                (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS2Cbdsw*(1-model->MOS2fwdCapDepCoeff*
                                (1+model->MOS2bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS2f3d = here->MOS2Cbd * 
                                model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                                here->MOS2tBulkPot + here->MOS2Cbdsw * 
                                model->MOS2bulkJctSideGradingCoeff * 
                                sargsw/arg / here->MOS2tBulkPot;
                        here->MOS2f4d = here->MOS2Cbd*
                                here->MOS2tBulkPot*(1-arg*sarg)/
                                (1-model->MOS2bulkJctBotGradingCoeff)
                                + here->MOS2Cbdsw*here->MOS2tBulkPot*
                                (1-arg*sargsw)/
                                (1-model->MOS2bulkJctSideGradingCoeff)
                                -here->MOS2f3d/2*
                                (here->MOS2tDepCap*here->MOS2tDepCap)
                                -here->MOS2tDepCap * here->MOS2f2d;
                    }
                    if(*(ckt->CKTstate0 + here->MOS2vbs) >=
                        here->MOS2tDepCap){
                        arg = 1-model->MOS2fwdCapDepCoeff;
                        sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS2f2s = here->MOS2Cbs*
                                (1-model->MOS2fwdCapDepCoeff*
                                (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS2Cbssw*(1-model->MOS2fwdCapDepCoeff*
                                (1+model->MOS2bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS2f3s = here->MOS2Cbs * 
                                model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                                here->MOS2tBulkPot + here->MOS2Cbssw * 
                                model->MOS2bulkJctSideGradingCoeff *sargsw/arg /
                                here->MOS2tBulkPot;
                        here->MOS2f4s = here->MOS2Cbs*here->MOS2tBulkPot*
                                (1-arg*sarg)/
                                (1-model->MOS2bulkJctBotGradingCoeff)
                                + here->MOS2Cbssw*here->MOS2tBulkPot*
                                (1-arg*sargsw)/
                                (1-model->MOS2bulkJctSideGradingCoeff)
                                -here->MOS2f3s/2*
                                (here->MOS2tDepCap*here->MOS2tDepCap)
                                -here->MOS2tDepCap * here->MOS2f2s;
                    }
                    here->MOS2drainConductance *= Apert/A0;
                    here->MOS2sourceConductance *= Apert/A0;
                }


#ifdef SENSDEBUG
                if(flag == 0)
                    printf("perturbation of l\n");
                if(flag == 1)
                    printf("perturbation of w\n");
#endif /* SENSDEBUG */

                error =  MOS2load((GENmodel*)model,ckt);
                if(error) return(error);

                if(flag == 0){
                    here->MOS2l = A0;
                }
                else{
                    here->MOS2w = A0;
                    here->MOS2drainArea /= (1 + info->SENpertfac);
                    here->MOS2sourceArea /= (1 + info->SENpertfac);
                    here->MOS2drainConductance *= A0/Apert;
                    here->MOS2sourceConductance *= A0/Apert;
                }
                cd =  here->MOS2cd ;
                cbd = here->MOS2cbd ;
                cbs = here->MOS2cbs ;

                gspr= here->MOS2sourceConductance ;
                gdpr= here->MOS2drainConductance ;

                DcdDp = (cd - cd0) * DELAinv;
                DcbsDp = (cbs - cbs0) * DELAinv;
                DcbdDp = (cbd - cbd0) * DELAinv;
                DcbDp = ( DcbsDp + DcbdDp );

                DcdprDp = 0;
                DcsprDp = 0;
                if(here->MOS2dNode != here->MOS2dNodePrime)
                    if(gdpr0) DcdprDp = cdpr0 * (gdpr - gdpr0)/gdpr0 * DELAinv;
                if(here->MOS2sNode != here->MOS2sNodePrime)
                    if(gspr0) DcsprDp = cspr0 * (gspr - gspr0)/gspr0 * DELAinv;

                DcdprmDp = ( - DcdprDp + DcdDp);
                DcsprmDp = ( - DcbsDp - DcdDp - DcbdDp  - DcsprDp);

                if(flag == 0){
                    EffectiveLength = here->MOS2l 
                        - 2*model->MOS2latDiff;
                    if(EffectiveLength == 0){ 
                        DqgsDp = 0;
                        DqgdDp = 0;
                        DqgbDp = 0;
                    }
                    else{
                        DqgsDp = model->MOS2type * qgs0 / EffectiveLength;
                        DqgdDp = model->MOS2type * qgd0 / EffectiveLength;
                        DqgbDp = model->MOS2type * qgb0 / EffectiveLength;
                    }
                }
                else{
                    DqgsDp = model->MOS2type * qgs0 / here->MOS2w;
                    DqgdDp = model->MOS2type * qgd0 / here->MOS2w;
                    DqgbDp = model->MOS2type * qgb0 / here->MOS2w;
                }


                qbd = *(ckt->CKTstate0 + here->MOS2qbd);
                qbs = *(ckt->CKTstate0 + here->MOS2qbs);

                DqbsDp = model->MOS2type * (qbs - qbs0)*DELAinv;
                DqbdDp = model->MOS2type * (qbd - qbd0)*DELAinv;

                if(flag == 0){
                    *(here->MOS2dphigs_dl) = DqgsDp;
                    *(here->MOS2dphigd_dl) = DqgdDp;
                    *(here->MOS2dphibs_dl) = DqbsDp;
                    *(here->MOS2dphibd_dl) = DqbdDp;
                    *(here->MOS2dphigb_dl) = DqgbDp;
                }
                else{
                    *(here->MOS2dphigs_dw) = DqgsDp;
                    *(here->MOS2dphigd_dw) = DqgdDp;
                    *(here->MOS2dphibs_dw) = DqbsDp;
                    *(here->MOS2dphibd_dw) = DqbdDp;
                    *(here->MOS2dphigb_dw) = DqgbDp;
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
                printf("tdepCap = %.7e \n",here->MOS2tDepCap); 
                printf("\n");
#endif /* SENSDEBUG*/
                if((info->SENmode == TRANSEN) &&
                    (ckt->CKTmode & MODEINITTRAN))
                    goto next2;

                /*
                 *   load RHS matrix
                 */

                if(flag == 0){
                    *(info->SEN_RHS[here->MOS2bNode] + here->MOS2senParmNo) -= 
                            model->MOS2type * DcbDp;
                    *(info->SEN_RHS[here->MOS2dNode] + here->MOS2senParmNo) -= 
                            model->MOS2type * DcdprDp;
                    *(info->SEN_RHS[here->MOS2dNodePrime] + here->MOS2senParmNo)
                            -= model->MOS2type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS2sNode] + here->MOS2senParmNo) -= 
                            model->MOS2type * DcsprDp;
                    *(info->SEN_RHS[here->MOS2sNodePrime] + here->MOS2senParmNo)
                            -= model->MOS2type * DcsprmDp;
                }
                else{  
                    offset = here->MOS2sens_l;

                    *(info->SEN_RHS[here->MOS2bNode] + here->MOS2senParmNo + 
                            offset) -= model->MOS2type * DcbDp;
                    *(info->SEN_RHS[here->MOS2dNode] + here->MOS2senParmNo + 
                            offset) -= model->MOS2type * DcdprDp;
                    *(info->SEN_RHS[here->MOS2dNodePrime] + here->MOS2senParmNo
                            + offset) -= model->MOS2type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS2sNode] + here->MOS2senParmNo + 
                            offset) -= model->MOS2type * DcsprDp;
                    *(info->SEN_RHS[here->MOS2sNodePrime] + here->MOS2senParmNo
                            + offset) -= model->MOS2type * DcsprmDp;
                }
#ifdef SENSDEBUG
                printf("after loading\n");
                if(flag == 0){
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2bNode] + 
                            here->MOS2senParmNo));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2dNode] + 
                            here->MOS2senParmNo));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2sNode] +
                            here->MOS2senParmNo));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2dNodePrime] +
                            here->MOS2senParmNo));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2sNodePrime] +
                            here->MOS2senParmNo));
                    printf("\n");
                }
                else{
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2bNode] +
                            here->MOS2senParmNo + here->MOS2sens_l));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2dNode] +
                            here->MOS2senParmNo + here->MOS2sens_l));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2sNode] +
                            here->MOS2senParmNo + here->MOS2sens_l));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2dNodePrime] +
                            here->MOS2senParmNo + here->MOS2sens_l));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS2sNodePrime] +
                            here->MOS2senParmNo + here->MOS2sens_l));
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
                        *(info->SEN_RHS[here->MOS2bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2sNodePrime] + iparmno));
                printf("\n");
#endif /* SENSDEBUG */
                Osxpgs = tag0 * *(ckt->CKTstate1 + here->MOS2sensxpgs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS2sensxpgs +
                        10*(iparmno - 1) + 1);

                Osxpgd = tag0 * *(ckt->CKTstate1 + here->MOS2sensxpgd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS2sensxpgd +
                        10*(iparmno - 1) + 1);

                Osxpbs = tag0 * *(ckt->CKTstate1 + here->MOS2sensxpbs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS2sensxpbs +
                        10*(iparmno - 1) + 1);

                Osxpbd =tag0 * *(ckt->CKTstate1 + here->MOS2sensxpbd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS2sensxpbd +
                        10*(iparmno - 1) + 1);
                Osxpgb = tag0 * *(ckt->CKTstate1 + here->MOS2sensxpgb +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS2sensxpgb +
                        10*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("sxpgs=%.7e,sdgs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS2sensxpgs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS2sensxpgs + 10*(iparmno - 1) + 1));
                printf("sxpgd=%.7e,sdgd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS2sensxpgd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS2sensxpgd + 10*(iparmno - 1) + 1));
                printf("sxpbs=%.7e,sdbs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS2sensxpbs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS2sensxpbs + 10*(iparmno - 1) + 1));
                printf("sxpbd=%.7e,sdbd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS2sensxpbd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS2sensxpbd + 10*(iparmno - 1) + 1));
                printf("sxpgb=%.7e,sdgb=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS2sensxpgb + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS2sensxpgb + 10*(iparmno - 1) + 1));
                printf("before loading DqDp\n");
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
                printf("\n");
#endif /* SENSDEBUG */
                if(here->MOS2sens_l && (iparmno == here->MOS2senParmNo)){
                    Osxpgs -= tag0 * *(here->MOS2dphigs_dl);
                    Osxpgd -= tag0 * *(here->MOS2dphigd_dl);
                    Osxpbs -= tag0 * *(here->MOS2dphibs_dl);
                    Osxpbd -= tag0 * *(here->MOS2dphibd_dl);
                    Osxpgb -= tag0 * *(here->MOS2dphigb_dl);
                }
                if(here->MOS2sens_w &&
                        (iparmno == (here->MOS2senParmNo +
                            (int) here->MOS2sens_l))){
                    Osxpgs -= tag0 * *(here->MOS2dphigs_dw);
                    Osxpgd -= tag0 * *(here->MOS2dphigd_dw);
                    Osxpbs -= tag0 * *(here->MOS2dphibs_dw);
                    Osxpbd -= tag0 * *(here->MOS2dphibd_dw);
                    Osxpgb -= tag0 * *(here->MOS2dphigb_dw);
                }
#ifdef SENSDEBUG
                printf("after loading DqDp\n");
                printf("DqgsDp=%.7e",DqgsDp);
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->MOS2bNode] + iparmno) += 
                        Osxpbs + Osxpbd -Osxpgb;
                *(info->SEN_RHS[here->MOS2gNode] + iparmno) += 
                        Osxpgs + Osxpgd + Osxpgb;

                *(info->SEN_RHS[here->MOS2dNodePrime] + iparmno) -= 
                        Osxpgd + Osxpbd ;

                *(info->SEN_RHS[here->MOS2sNodePrime] + iparmno) -= 
                        Osxpgs + Osxpbs;
#ifdef SENSDEBUG
                printf("after capacitive currents\n");
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS2sNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
restore:    /* put the unperturbed values back into the state vector */
            for(i=0; i <= 16; i++)
                *(ckt->CKTstate0 + here->MOS2states + i) = *(SaveState + i);
            here->MOS2sourceConductance = *(SaveState + 17) ;   
            here->MOS2drainConductance = *(SaveState + 18) ; 
            here->MOS2cd =  *(SaveState + 19) ;  
            here->MOS2cbs =  *(SaveState + 20) ;  
            here->MOS2cbd =  *(SaveState + 21) ;  
            here->MOS2gmbs =  *(SaveState + 22) ;  
            here->MOS2gm =  *(SaveState + 23) ;  
            here->MOS2gds =  *(SaveState + 24) ;  
            here->MOS2gbd =  *(SaveState + 25) ;  
            here->MOS2gbs =  *(SaveState + 26) ;  
            here->MOS2capbd =  *(SaveState + 27) ;  
            here->MOS2capbs =  *(SaveState + 28) ;  
            here->MOS2Cbd =  *(SaveState + 29) ;  
            here->MOS2Cbdsw =  *(SaveState + 30) ;  
            here->MOS2Cbs =  *(SaveState + 31) ;  
            here->MOS2Cbssw =  *(SaveState + 32) ;  
            here->MOS2f2d =  *(SaveState + 33) ;  
            here->MOS2f3d =  *(SaveState + 34) ;  
            here->MOS2f4d =  *(SaveState + 35) ;  
            here->MOS2f2s =  *(SaveState + 36) ;  
            here->MOS2f3s =  *(SaveState + 37) ;  
            here->MOS2f4s =  *(SaveState + 38) ;  
            here->MOS2cgs = *(SaveState + 39) ;  
            here->MOS2cgd = *(SaveState + 40) ;  
            here->MOS2cgb = *(SaveState + 41) ;  
            here->MOS2vdsat = *(SaveState + 42) ;  
            here->MOS2von = *(SaveState + 43) ;   
            here->MOS2mode = save_mode ;  

            here->MOS2senPertFlag = OFF;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("MOS2senload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

