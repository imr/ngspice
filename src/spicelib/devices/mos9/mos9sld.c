/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9sLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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
    printf("MOS9senload \n");
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
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senload instance name %s\n",here->MOS9name);
            printf("gate = %d ,drain = %d, drainprm = %d\n",
                    here->MOS9gNode,here->MOS9dNode,here->MOS9dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS9sNode ,here->MOS9sNodePrime,
                    here->MOS9bNode,here->MOS9senParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++){
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS9states + i);
            }

            *(SaveState + 17) = here->MOS9sourceConductance;  
            *(SaveState + 18) = here->MOS9drainConductance;  
            *(SaveState + 19) = here->MOS9cd;  
            *(SaveState + 20) = here->MOS9cbs;  
            *(SaveState + 21) = here->MOS9cbd;  
            *(SaveState + 22) = here->MOS9gmbs;  
            *(SaveState + 23) = here->MOS9gm;  
            *(SaveState + 24) = here->MOS9gds;  
            *(SaveState + 25) = here->MOS9gbd;  
            *(SaveState + 26) = here->MOS9gbs;  
            *(SaveState + 27) = here->MOS9capbd;  
            *(SaveState + 28) = here->MOS9capbs;  
            *(SaveState + 29) = here->MOS9Cbd;  
            *(SaveState + 30) = here->MOS9Cbdsw;  
            *(SaveState + 31) = here->MOS9Cbs;  
            *(SaveState + 32) = here->MOS9Cbssw;  
            *(SaveState + 33) = here->MOS9f2d;  
            *(SaveState + 34) = here->MOS9f3d;  
            *(SaveState + 35) = here->MOS9f4d;  
            *(SaveState + 36) = here->MOS9f2s;  
            *(SaveState + 37) = here->MOS9f3s;  
            *(SaveState + 38) = here->MOS9f4s;  
            *(SaveState + 39) = here->MOS9cgs;  
            *(SaveState + 40) = here->MOS9cgd;  
            *(SaveState + 41) = here->MOS9cgb;  
            *(SaveState + 42) = here->MOS9vdsat;  
            *(SaveState + 43) = here->MOS9von;  
            save_mode = here->MOS9mode;  


            if(here->MOS9senParmNo == 0) goto next1;

#ifdef SENSDEBUG
            printf("without perturbation \n");
#endif /* SENSDEBUG */

            cdpr0= here->MOS9cd;
            cspr0= -(here->MOS9cd + here->MOS9cbd + here->MOS9cbs);
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                qgs0 = *(ckt->CKTstate1 + here->MOS9qgs);
                qgd0 = *(ckt->CKTstate1 + here->MOS9qgd);
                qgb0 = *(ckt->CKTstate1 + here->MOS9qgb);
            }
            else{
                qgs0 = *(ckt->CKTstate0 + here->MOS9qgs);
                qgd0 = *(ckt->CKTstate0 + here->MOS9qgd);
                qgb0 = *(ckt->CKTstate0 + here->MOS9qgb);
            }

            here->MOS9senPertFlag = ON;
            error =  MOS9load((GENmodel*)model,ckt);
            if(error) return(error);

            cd0 =  here->MOS9cd ;
            cbd0 = here->MOS9cbd ;
            cbs0 = here->MOS9cbs ;
            gspr0= here->MOS9sourceConductance ;
            gdpr0= here->MOS9drainConductance ;

            qbs0 = *(ckt->CKTstate0 + here->MOS9qbs);
            qbd0 = *(ckt->CKTstate0 + here->MOS9qbd);

            for( flag = 0 ; flag <= 1 ; flag++){
                if(here->MOS9sens_l == 0)
                    if(flag == 0) goto next2;
                if(here->MOS9sens_w == 0)
                    if(flag == 1) goto next2;
                if(flag == 0){
                    A0 = here->MOS9l;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS9l = Apert;
                }
                else{
                    A0 = here->MOS9w;
                    DELA = info->SENpertfac * A0;
                    DELAinv = 1.0/DELA;
                    Apert = A0 + DELA;
                    here->MOS9w = Apert;
                    here->MOS9drainArea *= (1 + info->SENpertfac);
                    here->MOS9sourceArea *= (1 + info->SENpertfac);
                    here->MOS9Cbd *= (1 + info->SENpertfac);
                    here->MOS9Cbs *= (1 + info->SENpertfac);
                    if(here->MOS9drainPerimiter){
                        here->MOS9Cbdsw += here->MOS9Cbdsw *
                            DELA/here->MOS9drainPerimiter;
                    }
                    if(here->MOS9sourcePerimiter){
                        here->MOS9Cbssw += here->MOS9Cbssw *
                            DELA/here->MOS9sourcePerimiter;
                    }
                    if(*(ckt->CKTstate0 + here->MOS9vbd) >=
                        here->MOS9tDepCap){
                        arg = 1-model->MOS9fwdCapDepCoeff;
                        sarg = exp( (-model->MOS9bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS9bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS9f2d = here->MOS9Cbd*
                                (1-model->MOS9fwdCapDepCoeff*
                                (1+model->MOS9bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS9Cbdsw*(1-model->MOS9fwdCapDepCoeff*
                                (1+model->MOS9bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS9f3d = here->MOS9Cbd * 
                                model->MOS9bulkJctBotGradingCoeff * sarg/arg/
                                model->MOS9bulkJctPotential
                                + here->MOS9Cbdsw * 
                                model->MOS9bulkJctSideGradingCoeff *sargsw/arg /
                                model->MOS9bulkJctPotential;
                        here->MOS9f4d = here->MOS9Cbd*
                                model->MOS9bulkJctPotential*(1-arg*sarg)/
                                (1-model->MOS9bulkJctBotGradingCoeff)
                                + here->MOS9Cbdsw*model->MOS9bulkJctPotential*
                                (1-arg*sargsw)/
                                (1-model->MOS9bulkJctSideGradingCoeff)
                                -here->MOS9f3d/2*
                                (here->MOS9tDepCap*here->MOS9tDepCap)
                                -here->MOS9tDepCap * here->MOS9f2d;
                    }
                    if(*(ckt->CKTstate0 + here->MOS9vbs) >=
                        here->MOS9tDepCap){
                        arg = 1-model->MOS9fwdCapDepCoeff;
                        sarg = exp( (-model->MOS9bulkJctBotGradingCoeff) * 
                                log(arg) );
                        sargsw = exp( (-model->MOS9bulkJctSideGradingCoeff) * 
                                log(arg) );
                        here->MOS9f2s = here->MOS9Cbs*
                                (1-model->MOS9fwdCapDepCoeff*
                                (1+model->MOS9bulkJctBotGradingCoeff))* sarg/arg
                                +  here->MOS9Cbssw*(1-model->MOS9fwdCapDepCoeff*
                                (1+model->MOS9bulkJctSideGradingCoeff))*
                                sargsw/arg;
                        here->MOS9f3s = here->MOS9Cbs * 
                                model->MOS9bulkJctBotGradingCoeff * sarg/arg/
                                model->MOS9bulkJctPotential
                                + here->MOS9Cbssw * 
                                model->MOS9bulkJctSideGradingCoeff * sargsw/arg/
                                model->MOS9bulkJctPotential;
                        here->MOS9f4s = here->MOS9Cbs*
                                model->MOS9bulkJctPotential*(1-arg*sarg)/
                                (1-model->MOS9bulkJctBotGradingCoeff)
                                + here->MOS9Cbssw*model->MOS9bulkJctPotential*
                                (1-arg*sargsw)/
                                (1-model->MOS9bulkJctSideGradingCoeff)
                                -here->MOS9f3s/2*
                                (here->MOS9tBulkPot*here->MOS9tBulkPot)
                                -here->MOS9tBulkPot * here->MOS9f2s;
                    }
                    here->MOS9drainConductance *= Apert/A0;
                    here->MOS9sourceConductance *= Apert/A0;
                }


#ifdef SENSDEBUG
                if(flag == 0)
                    printf("perturbation of l\n");
                if(flag == 1)
                    printf("perturbation of w\n");
#endif /* SENSDEBUG */

                error =  MOS9load((GENmodel*)model,ckt);
                if(error) return(error);

                if(flag == 0){
                    here->MOS9l = A0;
                }
                else{
                    here->MOS9w = A0;
                    here->MOS9drainArea /= (1 + info->SENpertfac);
                    here->MOS9sourceArea /= (1 + info->SENpertfac);
                    here->MOS9drainConductance *= A0/Apert;
                    here->MOS9sourceConductance *= A0/Apert;
                }
                cd =  here->MOS9cd ;
                cbd = here->MOS9cbd ;
                cbs = here->MOS9cbs ;

                gspr= here->MOS9sourceConductance ;
                gdpr= here->MOS9drainConductance ;

                DcdDp = (cd - cd0) * DELAinv;
                DcbsDp = (cbs - cbs0) * DELAinv;
                DcbdDp = (cbd - cbd0) * DELAinv;
                DcbDp = ( DcbsDp + DcbdDp );

                DcdprDp = 0;
                DcsprDp = 0;
                if(here->MOS9dNode != here->MOS9dNodePrime)
                    if(gdpr0) DcdprDp = cdpr0 * (gdpr - gdpr0)/gdpr0 * DELAinv;
                if(here->MOS9sNode != here->MOS9sNodePrime)
                    if(gspr0) DcsprDp = cspr0 * (gspr - gspr0)/gspr0 * DELAinv;

                DcdprmDp = ( - DcdprDp + DcdDp);
                DcsprmDp = ( - DcbsDp - DcdDp - DcbdDp  - DcsprDp);

                if(flag == 0){
                    EffectiveLength = here->MOS9l 
                        - 2*model->MOS9latDiff + model->MOS9lengthAdjust;
                    if(EffectiveLength == 0){ 
                        DqgsDp = 0;
                        DqgdDp = 0;
                        DqgbDp = 0;
                    }
                    else{
                        DqgsDp = model->MOS9type * qgs0 / EffectiveLength;
                        DqgdDp = model->MOS9type * qgd0 / EffectiveLength;
                        DqgbDp = model->MOS9type * qgb0 / EffectiveLength;
                    }
                }
                else{
                    EffectiveWidth = here->MOS9w 
                        - 2*model->MOS9widthNarrow + model->MOS9widthAdjust;
                    DqgsDp = model->MOS9type * qgs0 / EffectiveWidth;
                    DqgdDp = model->MOS9type * qgd0 / EffectiveWidth;
                    DqgbDp = model->MOS9type * qgb0 / EffectiveWidth;
                }


                qbd = *(ckt->CKTstate0 + here->MOS9qbd);
                qbs = *(ckt->CKTstate0 + here->MOS9qbs);

                DqbsDp = model->MOS9type * (qbs - qbs0)*DELAinv;
                DqbdDp = model->MOS9type * (qbd - qbd0)*DELAinv;

                if(flag == 0){
                    *(here->MOS9dphigs_dl) = DqgsDp;
                    *(here->MOS9dphigd_dl) = DqgdDp;
                    *(here->MOS9dphibs_dl) = DqbsDp;
                    *(here->MOS9dphibd_dl) = DqbdDp;
                    *(here->MOS9dphigb_dl) = DqgbDp;
                }
                else{
                    *(here->MOS9dphigs_dw) = DqgsDp;
                    *(here->MOS9dphigd_dw) = DqgdDp;
                    *(here->MOS9dphibs_dw) = DqbsDp;
                    *(here->MOS9dphibd_dw) = DqbdDp;
                    *(here->MOS9dphigb_dw) = DqgbDp;
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
                printf("tdepCap = %.7e \n",here->MOS9tDepCap); 
                printf("\n");
#endif /* SENSDEBUG*/
                if((info->SENmode == TRANSEN) &&
                    (ckt->CKTmode & MODEINITTRAN))
                    goto next2;

                /*
                                 *   load RHS matrix
                                 */

                if(flag == 0){
                    *(info->SEN_RHS[here->MOS9bNode] + here->MOS9senParmNo) 
                            -= model->MOS9type * DcbDp;
                    *(info->SEN_RHS[here->MOS9dNode] + here->MOS9senParmNo) 
                            -= model->MOS9type * DcdprDp;
                    *(info->SEN_RHS[here->MOS9dNodePrime] + here->MOS9senParmNo)
                            -= model->MOS9type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS9sNode] + here->MOS9senParmNo) 
                            -= model->MOS9type * DcsprDp;
                    *(info->SEN_RHS[here->MOS9sNodePrime] + here->MOS9senParmNo)
                            -= model->MOS9type * DcsprmDp;
                }
                else{  
                    offset = here->MOS9sens_l;

                    *(info->SEN_RHS[here->MOS9bNode] + here->MOS9senParmNo + 
                            offset) -= model->MOS9type * DcbDp;
                    *(info->SEN_RHS[here->MOS9dNode] + here->MOS9senParmNo + 
                            offset) -= model->MOS9type * DcdprDp;
                    *(info->SEN_RHS[here->MOS9dNodePrime] + here->MOS9senParmNo
                            + offset) -= model->MOS9type * DcdprmDp;
                    *(info->SEN_RHS[here->MOS9sNode] + here->MOS9senParmNo +
                            offset) -= model->MOS9type * DcsprDp;
                    *(info->SEN_RHS[here->MOS9sNodePrime] + here->MOS9senParmNo
                            + offset) -= model->MOS9type * DcsprmDp;
                }
#ifdef SENSDEBUG
                printf("after loading\n");
                if(flag == 0){
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9bNode] +
                            here->MOS9senParmNo));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9dNode] +
                            here->MOS9senParmNo));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9sNode] +
                            here->MOS9senParmNo));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9dNodePrime] +
                            here->MOS9senParmNo));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9sNodePrime] +
                            here->MOS9senParmNo));
                    printf("\n");
                }
                else{
                    printf("DcbDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9bNode] +
                            here->MOS9senParmNo + here->MOS9sens_l));
                    printf("DcdprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9dNode] +
                            here->MOS9senParmNo + here->MOS9sens_l));
                    printf("DcsprDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9sNode] +
                            here->MOS9senParmNo + here->MOS9sens_l));
                    printf("DcdprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9dNodePrime] +
                            here->MOS9senParmNo + here->MOS9sens_l));
                    printf("DcsprmDp=%.7e\n",
                            *(info->SEN_RHS[here->MOS9sNodePrime] +
                            here->MOS9senParmNo + here->MOS9sens_l));
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
                        *(info->SEN_RHS[here->MOS9bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9sNodePrime] + iparmno));
                printf("\n");
#endif /* SENSDEBUG */
                Osxpgs = tag0 * *(ckt->CKTstate1 + here->MOS9sensxpgs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS9sensxpgs +
                        10*(iparmno - 1) + 1);

                Osxpgd = tag0 * *(ckt->CKTstate1 + here->MOS9sensxpgd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS9sensxpgd +
                        10*(iparmno - 1) + 1);

                Osxpbs = tag0 * *(ckt->CKTstate1 + here->MOS9sensxpbs +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS9sensxpbs +
                        10*(iparmno - 1) + 1);

                Osxpbd =tag0 * *(ckt->CKTstate1 + here->MOS9sensxpbd +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS9sensxpbd +
                        10*(iparmno - 1) + 1);
                Osxpgb = tag0 * *(ckt->CKTstate1 + here->MOS9sensxpgb +
                        10*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->MOS9sensxpgb +
                        10*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("sxpgs=%.7e,sdgs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS9sensxpgs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS9sensxpgs + 10*(iparmno - 1) + 1));
                printf("sxpgd=%.7e,sdgd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS9sensxpgd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS9sensxpgd + 10*(iparmno - 1) + 1));
                printf("sxpbs=%.7e,sdbs=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS9sensxpbs + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS9sensxpbs + 10*(iparmno - 1) + 1));
                printf("sxpbd=%.7e,sdbd=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS9sensxpbd + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS9sensxpbd + 10*(iparmno - 1) + 1));
                printf("sxpgb=%.7e,sdgb=%.7e\n",
                        *(ckt->CKTstate1 + here->MOS9sensxpgb + 
                        10*(iparmno - 1)), *(ckt->CKTstate1 + 
                        here->MOS9sensxpgb + 10*(iparmno - 1) + 1));
                printf("before loading DqDp\n");
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
                printf("\n");
#endif /* SENSDEBUG */
                if(here->MOS9sens_l && (iparmno == here->MOS9senParmNo)){
                    Osxpgs -= tag0 * *(here->MOS9dphigs_dl);
                    Osxpgd -= tag0 * *(here->MOS9dphigd_dl);
                    Osxpbs -= tag0 * *(here->MOS9dphibs_dl);
                    Osxpbd -= tag0 * *(here->MOS9dphibd_dl);
                    Osxpgb -= tag0 * *(here->MOS9dphigb_dl);
                }
                if(here->MOS9sens_w &&
                        (iparmno == (here->MOS9senParmNo +
                            (int) here->MOS9sens_l))){
                    Osxpgs -= tag0 * *(here->MOS9dphigs_dw);
                    Osxpgd -= tag0 * *(here->MOS9dphigd_dw);
                    Osxpbs -= tag0 * *(here->MOS9dphibs_dw);
                    Osxpbd -= tag0 * *(here->MOS9dphibd_dw);
                    Osxpgb -= tag0 * *(here->MOS9dphigb_dw);
                }
#ifdef SENSDEBUG
                printf("after loading DqDp\n");
                printf("DqgsDp=%.7e",DqgsDp);
                printf("Osxpgs=%.7e,Osxpgd=%.7e\n",Osxpgs,Osxpgd);
                printf("Osxpbs=%.7e,Osxpbd=%.7e,Osxpgb=%.7e\n",
                        Osxpbs,Osxpbd,Osxpgb);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->MOS9bNode] + iparmno) += 
                        Osxpbs + Osxpbd -Osxpgb;
                *(info->SEN_RHS[here->MOS9gNode] + iparmno) +=
                        Osxpgs + Osxpgd + Osxpgb;

                *(info->SEN_RHS[here->MOS9dNodePrime] + iparmno) -=
                        Osxpgd + Osxpbd ;
                *(info->SEN_RHS[here->MOS9sNodePrime] + iparmno) -=
                        Osxpgs + Osxpbs;
#ifdef SENSDEBUG
                printf("after capacitive currents\n");
                printf("DcbDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9bNode] + iparmno));
                printf("DcdprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9dNode] + iparmno));
                printf("DcdprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9dNodePrime] + iparmno));
                printf("DcsprDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9sNode] + iparmno));
                printf("DcsprmDp=%.7e\n",
                        *(info->SEN_RHS[here->MOS9sNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
restore:    /* put the unperturbed values back into the state vector */
            for(i=0; i <= 16; i++)
                *(ckt->CKTstate0 + here->MOS9states + i) = *(SaveState + i);
            here->MOS9sourceConductance = *(SaveState + 17) ;   
            here->MOS9drainConductance = *(SaveState + 18) ; 
            here->MOS9cd =  *(SaveState + 19) ;  
            here->MOS9cbs =  *(SaveState + 20) ;  
            here->MOS9cbd =  *(SaveState + 21) ;  
            here->MOS9gmbs =  *(SaveState + 22) ;  
            here->MOS9gm =  *(SaveState + 23) ;  
            here->MOS9gds =  *(SaveState + 24) ;  
            here->MOS9gbd =  *(SaveState + 25) ;  
            here->MOS9gbs =  *(SaveState + 26) ;  
            here->MOS9capbd =  *(SaveState + 27) ;  
            here->MOS9capbs =  *(SaveState + 28) ;  
            here->MOS9Cbd =  *(SaveState + 29) ;  
            here->MOS9Cbdsw =  *(SaveState + 30) ;  
            here->MOS9Cbs =  *(SaveState + 31) ;  
            here->MOS9Cbssw =  *(SaveState + 32) ;  
            here->MOS9f2d =  *(SaveState + 33) ;  
            here->MOS9f3d =  *(SaveState + 34) ;  
            here->MOS9f4d =  *(SaveState + 35) ;  
            here->MOS9f2s =  *(SaveState + 36) ;  
            here->MOS9f3s =  *(SaveState + 37) ;  
            here->MOS9f4s =  *(SaveState + 38) ;  
            here->MOS9cgs = *(SaveState + 39) ;  
            here->MOS9cgd = *(SaveState + 40) ;  
            here->MOS9cgb = *(SaveState + 41) ;  
            here->MOS9vdsat = *(SaveState + 42) ;  
            here->MOS9von = *(SaveState + 43) ;   
            here->MOS9mode = save_mode ;  

            here->MOS9senPertFlag = OFF;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("MOS9senload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

