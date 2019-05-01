/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS3sLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
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
    double EffectiveWidth;
    SENstruct *info;

#ifdef SENSDEBUG
    printf("MOS3senload \n");
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
    for( ; model != NULL; model = MOS3nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ;
                here=MOS3nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senload instance name %s\n",here->MOS3name);
            printf("gate = %d ,drain = %d, drainprm = %d\n",
                    here->MOS3gNode,here->MOS3dNode,here->MOS3dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS3sNode ,here->MOS3sNodePrime,
                    here->MOS3bNode,here->MOS3senParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++){
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS3states + i);
            }

            *(SaveState + 17) = here->MOS3sourceConductance;  
            *(SaveState + 18) = here->MOS3drainConductance;  
            *(SaveState + 19) = here->MOS3cd;  
            *(SaveState + 20) = here->MOS3cbs;  
            *(SaveState + 21) = here->MOS3cbd;  
            *(SaveState + 22) = here->MOS3gmbs;  
            *(SaveState + 23) = here->MOS3gm;  
            *(SaveState + 24) = here->MOS3gds;  
            *(SaveState + 25) = here->MOS3gbd;  
            *(SaveState + 26) = here->MOS3gbs;  
            *(SaveState + 27) = here->MOS3capbd;  
            *(SaveState + 28) = here->MOS3capbs;  
            *(SaveState + 29) = here->MOS3Cbd;  
            *(SaveState + 30) = here->MOS3Cbdsw;  
            *(SaveState + 31) = here->MOS3Cbs;  
            *(SaveState + 32) = here->MOS3Cbssw;  
            *(SaveState + 33) = here->MOS3f2d;  
            *(SaveState + 34) = here->MOS3f3d;  
            *(SaveState + 35) = here->MOS3f4d;  
            *(SaveState + 36) = here->MOS3f2s;  
            *(SaveState + 37) = here->MOS3f3s;  
            *(SaveState + 38) = here->MOS3f4s;  
            *(SaveState + 39) = here->MOS3cgs;  
            *(SaveState + 40) = here->MOS3cgd;  
            *(SaveState + 41) = here->MOS3cgb;  
            *(SaveState + 42) = here->MOS3vdsat;  
            *(SaveState + 43) = here->MOS3von;  
            save_mode = here->MOS3mode;  


            if(here->MOS3senParmNo == 0) goto next1;

#ifdef SENSDEBUG
            printf("without perturbation \n");
#endif /* SENSDEBUG */

            cdpr0= here->MOS3cd;
            cspr0= -(here->MOS3cd + here->MOS3cbd + here->MOS3cbs);
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                qgs0 = *(ckt->CKTstate1 + here->MOS3qgs);
                qgd0 = *(ckt->CKTstate1 + here->MOS3qgd);
                qgb0 = *(ckt->CKTstate1 + here->MOS3qgb);
            }
            else{
                qgs0 = *(ckt->CKTstate0 + here->MOS3qgs);
                qgd0 = *(ckt->CKTstate0 + here->MOS3qgd);
                qgb0 = *(ckt->CKTstate0 + here->MOS3qgb);
            }

            here->MOS3senPertFlag = ON;
            error =  MOS3load((GENmodel*)model,ckt);
            if(error) return(error);

            cd0 =  here->MOS3cd ;
            cbd0 = here->MOS3cbd ;
            cbs0 = here->MOS3cbs ;
            gspr0= here->MOS3sourceConductance ;
            gdpr0= here->MOS3drainConductance ;

            qbs0 = *(ckt->CKTstate0 + here->MOS3qbs);
            qbd0 = *(ckt->CKTstate0 + here->MOS3qbd);

            for( flag = 0 ; flag <= 1 ; flag++){
                if(here->MOS3sens_l == 0)
                    if(flag == 0) goto next2;
                if(here->MOS3sens_w == 0)
                    if(flag == 1) goto next2;
                if(flag == 0){
                    A0 = here->MOS3l;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS3l = Apert;
                }
                else{
                    A0 = here->MOS3w;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS3w = Apert;
                    here->MOS3drainArea *= (1 + info->SENpertfac);
                    here->MOS3sourceArea *= (1 + info->SENpertfac);
                    here->MOS3Cbd *= (1 + info->SENpertfac);
                    here->MOS3Cbs *= (1 + info->SENpertfac);
                    if(here->MOS3drainPerimiter){
                        here->MOS3Cbdsw += here->MOS3Cbdsw *
                            DELA/here->MOS3drainPerimiter;
                    }
                    if(here->MOS3sourcePerimiter){
                        here->MOS3Cbssw += here->MOS3Cbssw *
                            DELA/here->MOS3sourcePerimiter;
                    }
                    if(*(ckt->CKTstate0 + here->MOS3vbd) >=
                        here->MOS3tDepCap){
                        arg = 1-model->MOS3fwdCapDepCoeff;
                        sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS3f2d = here->MOS3Cbd*
                                (1-model->MOS3fwdCapDepCoeff*
                                (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS3Cbdsw*(1-model->MOS3fwdCapDepCoeff*
                                (1+model->MOS3bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS3f3d = here->MOS3Cbd * 
                                model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                                model->MOS3bulkJctPotential
                                + here->MOS3Cbdsw * 
                                model->MOS3bulkJctSideGradingCoeff *sargsw/arg /
                                model->MOS3bulkJctPotential;
                        here->MOS3f4d = here->MOS3Cbd*
                                model->MOS3bulkJctPotential*(1-arg*sarg)/
                                (1-model->MOS3bulkJctBotGradingCoeff)
                                + here->MOS3Cbdsw*model->MOS3bulkJctPotential*
                                (1-arg*sargsw)/
                                (1-model->MOS3bulkJctSideGradingCoeff)
                                -here->MOS3f3d/2*
                                (here->MOS3tDepCap*here->MOS3tDepCap)
                                -here->MOS3tDepCap * here->MOS3f2d;
                    }
                    if(*(ckt->CKTstate0 + here->MOS3vbs) >=
                        here->MOS3tDepCap){
                        arg = 1-model->MOS3fwdCapDepCoeff;
                        sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS3f2s = here->MOS3Cbs*
                                (1-model->MOS3fwdCapDepCoeff*
                                (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS3Cbssw*(1-model->MOS3fwdCapDepCoeff*
                                (1+model->MOS3bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS3f3s = here->MOS3Cbs * 
                                model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                                model->MOS3bulkJctPotential
                                + here->MOS3Cbssw * 
                                model->MOS3bulkJctSideGradingCoeff * sargsw/arg/
                                model->MOS3bulkJctPotential;
                        here->MOS3f4s = here->MOS3Cbs*
                                model->MOS3bulkJctPotential*(1-arg*sarg)/
                                (1-model->MOS3bulkJctBotGradingCoeff)
                                + here->MOS3Cbssw*model->MOS3bulkJctPotential*
                                (1-arg*sargsw)/
                                (1-model->MOS3bulkJctSideGradingCoeff)
                                -here->MOS3f3s/2*
                                (here->MOS3tBulkPot*here->MOS3tBulkPot)
                                -here->MOS3tBulkPot * here->MOS3f2s;
                    }
                    here->MOS3drainConductance *= Apert/A0;
                    here->MOS3sourceConductance *= Apert/A0;
                }


#ifdef SENSDEBUG
                if(flag == 0)
                    printf("perturbation of l\n");
                if(flag == 1)
                    printf("perturbation of w\n");
#endif /* SENSDEBUG */

                error =  MOS3load((GENmodel*)model,ckt);
                if(error) return(error);

                if(flag == 0){
                    here->MOS3l = A0;
                }
                else{
                    here->MOS3w = A0;
                    here->MOS3drainArea /= (1 + info->SENpertfac);
                    here->MOS3sourceArea /= (1 + info->SENpertfac);
                    here->MOS3drainConductance *= A0/Apert;
                    here->MOS3sourceConductance *= A0/Apert;
                }
                cd =  here->MOS3cd ;
                cbd = here->MOS3cbd ;
                cbs = here->MOS3cbs ;

                gspr= here->MOS3sourceConductance ;
                gdpr= here->MOS3drainConductance ;

                DcdDp = (cd - cd0) * DELAinv;
                DcbsDp = (cbs - cbs0) * DELAinv;
                DcbdDp = (cbd - cbd0) * DELAinv;
                DcbDp = ( DcbsDp + DcbdDp );

                DcdprDp = 0;
                DcsprDp = 0;
                if(here->MOS3dNode != here->MOS3dNodePrime)
                    if(gdpr0) DcdprDp = cdpr0 * (gdpr - gdpr0)/gdpr0 * DELAinv;
                if(here->MOS3sNode != here->MOS3sNodePrime)
                    if(gspr0) DcsprDp = cspr0 * (gspr - gspr0)/gspr0 * DELAinv;

                DcdprmDp = ( - DcdprDp + DcdDp);
                DcsprmDp = ( - DcbsDp - DcdDp - DcbdDp  - DcsprDp);

                if(flag == 0){
                   EffectiveLength = here->MOS3l 
                        - 2*model->MOS3latDiff + model->MOS3lengthAdjust;
                    if(EffectiveLength == 0){ 
                        DqgsDp = 0;
                        DqgdDp = 0;
                        DqgbDp = 0;
                    }
                    else{
                        DqgsDp = model->MOS3type * qgs0 / EffectiveLength;
                        DqgdDp = model->MOS3type * qgd0 / EffectiveLength;
                        DqgbDp = model->MOS3type * qgb0 / EffectiveLength;
                    }
                }
                else{
                    EffectiveWidth = here->MOS3w 
                        - 2*model->MOS3widthNarrow + model->MOS3widthAdjust;
                    DqgsDp = model->MOS3type * qgs0 / EffectiveWidth;
                    DqgdDp = model->MOS3type * qgd0 / EffectiveWidth;
                    DqgbDp = model->MOS3type * qgb0 / EffectiveWidth;
                }


                qbd = *(ckt->CKTstate0 + here->MOS3qbd);
                qbs = *(ckt->CKTstate0 + here->MOS3qbs);

                DqbsDp = model->MOS3type * (qbs - qbs0)*DELAinv;
                DqbdDp = model->MOS3type * (qbd - qbd0)*DELAinv;

                if(flag == 0){
                    *(here->MOS3dphigs_dl) = DqgsDp;
                    *(here->MOS3dphigd_dl) = DqgdDp;
                    *(here->MOS3dphibs_dl) = DqbsDp;
                    *(here->MOS3dphibd_dl) = DqbdDp;
                    *(here->MOS3dphigb_dl) = DqgbDp;
                }
                else{
                    *(here->MOS3dphigs_dw) = DqgsDp;
                    *(here->MOS3dphigd_dw) = DqgdDp;
                    *(here->MOS3dphibs_dw) = DqbsDp;
                    *(here->MOS3dphibd_dw) = DqbdDp;
                    *(here->MOS3dphigb_dw) = DqgbDp;
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
                printf("tdepCap = %.7e \n",here->MOS3tDepCap); 
                printf("\n");
#endif /* SENSDEBUG*/
                if((info->SENmode == TRANSEN) &&
                    (ckt->CKTmode & MODEINITTRAN))
                    goto next2;

                /*
                                 *   load RHS matrix
                                 */

                if(flag == 0){
                    *(info->SEN_RHS[here->MOS3bNode] + here->MOS3senParmNo) 
                            -= model->MOS3type * DcbDp;
                    *(info->SEN_RHS[here->MOS3dNode] + here->MOS3senParmNo) 
                            -= model->MOS3type * DcdprDp;
                    *(info->SEN_RHS[here->MOS3dNodePrime] + here->MOS3senParmNo)
                            -= model->MOS3type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS3sNode] + here->MOS3senParmNo) 
                            -= model->MOS3type * DcsprDp;
                    *(info->SEN_RHS[here->MOS3sNodePrime] + here->MOS3senParmNo)
                            -= model->MOS3type * DcsprmDp;
                }
                else{  
                    offset = here->MOS3sens_l;

                    *(info->SEN_RHS[here->MOS3bNode] + here->MOS3senParmNo + 
                            offset) -= model->MOS3type * DcbDp;
                    *(info->SEN_RHS[here->MOS3dNode] + here->MOS3senParmNo + 
                            offset) -= model->MOS3type * DcdprDp;
                    *(info->SEN_RHS[here->MOS3dNodePrime] + here->MOS3senParmNo
                            + offset) -= model->MOS3type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS3sNode] + here->MOS3senParmNo +
                            offset) -= model->MOS3type * DcsprDp;
                    *(info->SEN_RHS[here->MOS3sNodePrime] + here->MOS3senParmNo
                            + offset) -= model->MOS3type * DcsprmDp;
                }
#ifdef SENSDEBUG
                printf("after loading\n");
                if(flag == 0){
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3bNode] +
                            here->MOS3senParmNo));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3dNode] +
                            here->MOS3senParmNo));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3sNode] +
                            here->MOS3senParmNo));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3dNodePrime] +
                            here->MOS3senParmNo));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3sNodePrime] +
                            here->MOS3senParmNo));
                    printf("\n");
                }
                else{
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3bNode] +
                            here->MOS3senParmNo + here->MOS3sens_l));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3dNode] +
                            here->MOS3senParmNo + here->MOS3sens_l));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3sNode] +
                            here->MOS3senParmNo + here->MOS3sens_l));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3dNodePrime] +
                            here->MOS3senParmNo + here->MOS3sens_l));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS3sNodePrime] +
                            here->MOS3senParmNo + here->MOS3sens_l));
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
                        *(info->SEN_RHS[here->MOS3bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3sNodePrime] + iparmno));
                printf("\n");
#endif /* SENSDEBUG */
                Osxpgs = tag0 * *(ckt->CKTstate1 + here->MOS3sensxpgs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS3sensxpgs +
                        10*(iparmno - 1) + 1);

                Osxpgd = tag0 * *(ckt->CKTstate1 + here->MOS3sensxpgd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS3sensxpgd +
                        10*(iparmno - 1) + 1);

                Osxpbs = tag0 * *(ckt->CKTstate1 + here->MOS3sensxpbs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS3sensxpbs +
                        10*(iparmno - 1) + 1);

                Osxpbd =tag0 * *(ckt->CKTstate1 + here->MOS3sensxpbd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS3sensxpbd +
                        10*(iparmno - 1) + 1);
                Osxpgb = tag0 * *(ckt->CKTstate1 + here->MOS3sensxpgb +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS3sensxpgb +
                        10*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("sxpgs=%.7e,sdgs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS3sensxpgs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS3sensxpgs + 10*(iparmno - 1) + 1));
                printf("sxpgd=%.7e,sdgd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS3sensxpgd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS3sensxpgd + 10*(iparmno - 1) + 1));
                printf("sxpbs=%.7e,sdbs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS3sensxpbs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS3sensxpbs + 10*(iparmno - 1) + 1));
                printf("sxpbd=%.7e,sdbd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS3sensxpbd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS3sensxpbd + 10*(iparmno - 1) + 1));
                printf("sxpgb=%.7e,sdgb=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS3sensxpgb + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS3sensxpgb + 10*(iparmno - 1) + 1));
                printf("before loading DqDp\n");
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
                printf("\n");
#endif /* SENSDEBUG */
                if(here->MOS3sens_l && (iparmno == here->MOS3senParmNo)){
                    Osxpgs -= tag0 * *(here->MOS3dphigs_dl);
                    Osxpgd -= tag0 * *(here->MOS3dphigd_dl);
                    Osxpbs -= tag0 * *(here->MOS3dphibs_dl);
                    Osxpbd -= tag0 * *(here->MOS3dphibd_dl);
                    Osxpgb -= tag0 * *(here->MOS3dphigb_dl);
                }
                if(here->MOS3sens_w &&
                        (iparmno == (here->MOS3senParmNo +
                            (int) here->MOS3sens_l))){
                    Osxpgs -= tag0 * *(here->MOS3dphigs_dw);
                    Osxpgd -= tag0 * *(here->MOS3dphigd_dw);
                    Osxpbs -= tag0 * *(here->MOS3dphibs_dw);
                    Osxpbd -= tag0 * *(here->MOS3dphibd_dw);
                    Osxpgb -= tag0 * *(here->MOS3dphigb_dw);
                }
#ifdef SENSDEBUG
                printf("after loading DqDp\n");
                printf("DqgsDp=%.7e",DqgsDp);
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->MOS3bNode] + iparmno) += 
                        Osxpbs + Osxpbd -Osxpgb;
                *(info->SEN_RHS[here->MOS3gNode] + iparmno) +=
                        Osxpgs + Osxpgd + Osxpgb;

                *(info->SEN_RHS[here->MOS3dNodePrime] + iparmno) -=
                        Osxpgd + Osxpbd ;
                *(info->SEN_RHS[here->MOS3sNodePrime] + iparmno) -=
                        Osxpgs + Osxpbs;
#ifdef SENSDEBUG
                printf("after capacitive currents\n");
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS3sNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
restore:    /* put the unperturbed values back into the state vector */
            for(i=0; i <= 16; i++)
                *(ckt->CKTstate0 + here->MOS3states + i) = *(SaveState + i);
            here->MOS3sourceConductance = *(SaveState + 17) ;   
            here->MOS3drainConductance = *(SaveState + 18) ; 
            here->MOS3cd =  *(SaveState + 19) ;  
            here->MOS3cbs =  *(SaveState + 20) ;  
            here->MOS3cbd =  *(SaveState + 21) ;  
            here->MOS3gmbs =  *(SaveState + 22) ;  
            here->MOS3gm =  *(SaveState + 23) ;  
            here->MOS3gds =  *(SaveState + 24) ;  
            here->MOS3gbd =  *(SaveState + 25) ;  
            here->MOS3gbs =  *(SaveState + 26) ;  
            here->MOS3capbd =  *(SaveState + 27) ;  
            here->MOS3capbs =  *(SaveState + 28) ;  
            here->MOS3Cbd =  *(SaveState + 29) ;  
            here->MOS3Cbdsw =  *(SaveState + 30) ;  
            here->MOS3Cbs =  *(SaveState + 31) ;  
            here->MOS3Cbssw =  *(SaveState + 32) ;  
            here->MOS3f2d =  *(SaveState + 33) ;  
            here->MOS3f3d =  *(SaveState + 34) ;  
            here->MOS3f4d =  *(SaveState + 35) ;  
            here->MOS3f2s =  *(SaveState + 36) ;  
            here->MOS3f3s =  *(SaveState + 37) ;  
            here->MOS3f4s =  *(SaveState + 38) ;  
            here->MOS3cgs = *(SaveState + 39) ;  
            here->MOS3cgd = *(SaveState + 40) ;  
            here->MOS3cgb = *(SaveState + 41) ;  
            here->MOS3vdsat = *(SaveState + 42) ;  
            here->MOS3von = *(SaveState + 43) ;   
            here->MOS3mode = save_mode ;  

            here->MOS3senPertFlag = OFF;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("MOS3senload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

