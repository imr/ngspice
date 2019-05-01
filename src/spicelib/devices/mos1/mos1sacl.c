/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

int
MOS1sAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model*)inModel;
    MOS1instance *here;
    int xnrm;
    int xrev;
    double A0;
    double Apert = 0.0;
    double DELA;
    double DELAinv;
    double gdpr0;
    double gspr0;
    double gds0;
    double gbs0;
    double gbd0;
    double gm0;
    double gmbs0;
    double gdpr;
    double gspr;
    double gds;
    double gbs;
    double gbd;
    double gm;
    double gmbs;
    double xcgs0;
    double xcgd0;
    double xcgb0;
    double xbd0;
    double xbs0;
    double xcgs;
    double xcgd;
    double xcgb;
    double xbd;
    double xbs;
    double vbsOp;
    double vbdOp;
    double vspr;
    double vdpr;
    double vgs;
    double vgd;
    double vgb;
    double vbs;
    double vbd;
    double vds;
    double ivspr;
    double ivdpr;
    double ivgs;
    double ivgd;
    double ivgb;
    double ivbs;
    double ivbd;
    double ivds;
    double cspr;
    double cdpr;
    double cgs;
    double cgd;
    double cgb;
    double cbs;
    double cbd;
    double cds;
    double cs0;
    double csprm0;
    double cd0;
    double cdprm0;
    double cg0;
    double cb0;
    double cs;
    double csprm;
    double cd;
    double cdprm;
    double cg;
    double cb;
    double icspr;
    double icdpr;
    double icgs;
    double icgd;
    double icgb;
    double icbs;
    double icbd;
    double icds;
    double ics0;
    double icsprm0;
    double icd0;
    double icdprm0;
    double icg0;
    double icb0;
    double ics;
    double icsprm;
    double icd;
    double icdprm;
    double icg;
    double icb;
    double DvDp = 0.0;
    int i;
    int flag;
    int error;
    int iparmno;
    double arg;
    double sarg;
    double sargsw;
    double SaveState[44];
    int    save_mode;
    SENstruct *info;

#ifdef SENSDEBUG
    printf("MOS1senacload\n");
    printf("CKTomega = %.5e\n",ckt->CKTomega);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;
    for( ; model != NULL; model = MOS1nextModel(model)) {
        for(here = MOS1instances(model); here!= NULL;
                here = MOS1nextInstance(here)) {

            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++)
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS1states + i);

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

            xnrm=1;
            xrev=0;
            if (here->MOS1mode < 0) {
                xnrm=0;
                xrev=1;
            }

            vbsOp = model->MOS1type * ( 
            *(ckt->CKTrhsOp+here->MOS1bNode) -
                *(ckt->CKTrhsOp+here->MOS1sNodePrime));
            vbdOp = model->MOS1type * ( 
            *(ckt->CKTrhsOp+here->MOS1bNode) -
                *(ckt->CKTrhsOp+here->MOS1dNodePrime));
            vspr = *(ckt->CKTrhsOld + here->MOS1sNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS1sNodePrime) ;
            ivspr = *(ckt->CKTirhsOld + here->MOS1sNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS1sNodePrime) ;
            vdpr = *(ckt->CKTrhsOld + here->MOS1dNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS1dNodePrime) ;
            ivdpr = *(ckt->CKTirhsOld + here->MOS1dNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS1dNodePrime) ;
            vgb = *(ckt->CKTrhsOld + here->MOS1gNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS1bNode) ;
            ivgb = *(ckt->CKTirhsOld + here->MOS1gNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS1bNode) ;
            vbs = *(ckt->CKTrhsOld + here->MOS1bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS1sNodePrime) ;
            ivbs = *(ckt->CKTirhsOld + here->MOS1bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS1sNodePrime) ;
            vbd = *(ckt->CKTrhsOld + here->MOS1bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS1dNodePrime) ;
            ivbd = *(ckt->CKTirhsOld + here->MOS1bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS1dNodePrime) ;
            vds = vbs - vbd ;
            ivds = ivbs - ivbd ;
            vgs = vgb + vbs ;
            ivgs = ivgb + ivbs ;
            vgd = vgb + vbd ;
            ivgd = ivgb + ivbd ;

#ifdef SENSDEBUG
            printf("senacload instance name %s\n",here->MOS1name);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->MOS1gNode,here->MOS1dNode,here->MOS1dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS1sNode ,here->MOS1sNodePrime,here->MOS1bNode,
                    here->MOS1senParmNo);
            printf("\n without  perturbation \n");
#endif /* SENSDEBUG */
            /*  without  perturbation  */
            *(ckt->CKTstate0 + here->MOS1vbs) = vbsOp;
            *(ckt->CKTstate0 + here->MOS1vbd) = vbdOp;

            here->MOS1senPertFlag = ON ;
            if(info->SENacpertflag == 1){
                /* store the  unperturbed values of small signal parameters */
                if((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->MOS1senCgs) = here->MOS1cgs;
                *(here->MOS1senCgd) = here->MOS1cgd;
                *(here->MOS1senCgb) = here->MOS1cgb;
                *(here->MOS1senCbd) = here->MOS1capbd;
                *(here->MOS1senCbs) = here->MOS1capbs;
                *(here->MOS1senGds) = here->MOS1gds;
                *(here->MOS1senGbs) = here->MOS1gbs;
                *(here->MOS1senGbd) = here->MOS1gbd;
                *(here->MOS1senGm) = here->MOS1gm;
                *(here->MOS1senGmbs) = here->MOS1gmbs;

            }
            xcgs0= *(here->MOS1senCgs) * ckt->CKTomega;
            xcgd0= *(here->MOS1senCgd) * ckt->CKTomega;
            xcgb0= *(here->MOS1senCgb) * ckt->CKTomega;
            xbd0= *(here->MOS1senCbd) * ckt->CKTomega;
            xbs0= *(here->MOS1senCbs) * ckt->CKTomega;
            gds0= *(here->MOS1senGds);
            gbs0= *(here->MOS1senGbs);
            gbd0= *(here->MOS1senGbd);
            gm0= *(here->MOS1senGm);
            gmbs0= *(here->MOS1senGmbs);
            gdpr0 = here->MOS1drainConductance;
            gspr0 = here->MOS1sourceConductance;


            cspr = gspr0 * vspr ;
            icspr = gspr0 * ivspr ;
            cdpr = gdpr0 * vdpr ;
            icdpr = gdpr0 * ivdpr ;
            cgs = ( - xcgs0 * ivgs );
            icgs =  xcgs0 * vgs ;
            cgd = ( - xcgd0 * ivgd );
            icgd =  xcgd0 * vgd ;
            cgb = ( - xcgb0 * ivgb );
            icgb =  xcgb0 * vgb ;
            cbs = ( gbs0 * vbs  - xbs0 * ivbs );
            icbs = ( xbs0 * vbs  + gbs0 * ivbs );
            cbd = ( gbd0 * vbd  - xbd0 * ivbd );
            icbd = ( xbd0 * vbd  + gbd0 * ivbd );
            cds = ( gds0 * vds  + xnrm * (gm0 * vgs + gmbs0 * vbs) 
                - xrev * (gm0 * vgd + gmbs0 * vbd) );
            icds = ( gds0 * ivds + xnrm * (gm0 * ivgs + gmbs0 * ivbs)
                - xrev * (gm0 * ivgd + gmbs0 * ivbd) );

            cs0 = cspr;
            ics0 = icspr;
            csprm0 = ( -cspr - cgs - cbs - cds ) ;
            icsprm0 = ( -icspr - icgs - icbs - icds ) ;
            cd0 = cdpr;
            icd0 = icdpr;
            cdprm0 = ( -cdpr - cgd - cbd + cds ) ;
            icdprm0 = ( -icdpr - icgd - icbd + icds ) ;
            cg0 = cgs + cgd + cgb ;
            icg0 = icgs + icgd + icgb ;
            cb0 = cbs + cbd - cgb ;
            icb0 = icbs + icbd - icgb ;
#ifdef SENSDEBUG
            printf("gspr0 = %.7e , gdpr0 = %.7e , gds0 = %.7e, gbs0 = %.7e\n",
                    gspr0,gdpr0,gds0,gbs0);
            printf("gbd0 = %.7e , gm0 = %.7e , gmbs0 = %.7e\n",gbd0,gm0,gmbs0);
            printf("xcgs0 = %.7e , xcgd0 = %.7e , xcgb0 = %.7e,",
                    xcgs0,xcgd0,xcgb0);
            printf("xbd0 = %.7e,xbs0 = %.7e\n",xbd0,xbs0);
            printf("vbs = %.7e , vbd = %.7e , vgb = %.7e\n",vbs,vbd,vgb);
            printf("ivbs = %.7e , ivbd = %.7e , ivgb = %.7e\n",ivbs,ivbd,ivgb);
            printf("cbs0 = %.7e , cbd0 = %.7e , cgb0 = %.7e\n",cbs,cbd,cgb);
            printf("cb0 = %.7e , cg0 = %.7e , cs0 = %.7e\n",cb0,cg0,cs0);
            printf("csprm0 = %.7e, cd0 = %.7e, cdprm0 = %.7e\n",
                    csprm0,cd0,cdprm0);
            printf("icb0 = %.7e , icg0 = %.7e , ics0 = %.7e\n",icb0,icg0,ics0);
            printf("icsprm0 = %.7e, icd0 = %.7e, icdprm0 = %.7e\n",
                    icsprm0,icd0,icdprm0);
            printf("\nPerturbation of vbs\n");
#endif /* SENSDEBUG */

            /* Perturbation of vbs */
            flag = 1;
            A0 = vbsOp;
            DELA =  info->SENpertfac * here->MOS1tVto ;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbs */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->MOS1vbs) = Apert;
                *(ckt->CKTstate0 + here->MOS1vbd) = vbdOp;

                if((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS1senCgs + 1) = here->MOS1cgs;
                *(here->MOS1senCgd + 1) = here->MOS1cgd;
                *(here->MOS1senCgb + 1) = here->MOS1cgb;
                *(here->MOS1senCbd + 1) = here->MOS1capbd;
                *(here->MOS1senCbs + 1) = here->MOS1capbs;
                *(here->MOS1senGds + 1) = here->MOS1gds;
                *(here->MOS1senGbs + 1) = here->MOS1gbs;
                *(here->MOS1senGbd + 1) = here->MOS1gbd;
                *(here->MOS1senGm + 1) = here->MOS1gm;
                *(here->MOS1senGmbs + 1) = here->MOS1gmbs;

                *(ckt->CKTstate0 + here->MOS1vbs) = A0;


            }

            goto load;


pertvbd:  /* Perturbation of vbd */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbd\n");
#endif /* SENSDEBUG */

            flag = 2;
            A0 = vbdOp;
            DELA =  info->SENpertfac * here->MOS1tVto + 1e-8;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbd */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->MOS1vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS1vbd) = Apert;

                if((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS1senCgs + 2) = here->MOS1cgs;
                *(here->MOS1senCgd + 2) = here->MOS1cgd;
                *(here->MOS1senCgb + 2) = here->MOS1cgb;
                *(here->MOS1senCbd + 2) = here->MOS1capbd;
                *(here->MOS1senCbs + 2) = here->MOS1capbs;
                *(here->MOS1senGds + 2) = here->MOS1gds;
                *(here->MOS1senGbs + 2) = here->MOS1gbs;
                *(here->MOS1senGbd + 2) = here->MOS1gbd;
                *(here->MOS1senGm + 2) = here->MOS1gm;
                *(here->MOS1senGmbs + 2) = here->MOS1gmbs;

                *(ckt->CKTstate0 + here->MOS1vbd) = A0;

            }

            goto load;


pertvgb:  /* Perturbation of vgb */
#ifdef SENSDEBUG
            printf("\nPerturbation of vgb\n");
#endif /* SENSDEBUG */

            flag = 3;
            A0 = model->MOS1type * (*(ckt->CKTrhsOp + here->MOS1gNode) 
                -  *(ckt->CKTrhsOp + here->MOS1bNode)); 
            DELA =  info->SENpertfac * A0 + 1e-8;
            DELAinv = model->MOS1type * 1.0/DELA;


            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vgb */
                *(ckt->CKTstate0 + here->MOS1vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS1vbd) = vbdOp;
                *(ckt->CKTrhsOp + here->MOS1bNode) -= DELA; 

                if((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS1senCgs + 3) = here->MOS1cgs;
                *(here->MOS1senCgd + 3) = here->MOS1cgd;
                *(here->MOS1senCgb + 3) = here->MOS1cgb;
                *(here->MOS1senCbd + 3) = here->MOS1capbd;
                *(here->MOS1senCbs + 3) = here->MOS1capbs;
                *(here->MOS1senGds + 3) = here->MOS1gds;
                *(here->MOS1senGbs + 3) = here->MOS1gbs;
                *(here->MOS1senGbd + 3) = here->MOS1gbd;
                *(here->MOS1senGm + 3) = here->MOS1gm;
                *(here->MOS1senGmbs + 3) = here->MOS1gmbs;


                *(ckt->CKTrhsOp + here->MOS1bNode) += DELA; 
            }
            goto load;

pertl:    /* Perturbation of length */

            if(here->MOS1sens_l == 0){
                goto pertw;
            }
#ifdef SENSDEBUG
            printf("\nPerturbation of length\n");
#endif /* SENSDEBUG */
            flag = 4;
            A0 = here->MOS1l;
            DELA =  info->SENpertfac * A0;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed length */
                Apert = A0 + DELA;
                here->MOS1l = Apert;

                *(ckt->CKTstate0 + here->MOS1vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS1vbd) = vbdOp;

                if ((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS1senCgs + 4) = here->MOS1cgs;
                *(here->MOS1senCgd + 4) = here->MOS1cgd;
                *(here->MOS1senCgb + 4) = here->MOS1cgb;
                *(here->MOS1senCbd + 4) = here->MOS1capbd;
                *(here->MOS1senCbs + 4) = here->MOS1capbs;
                *(here->MOS1senGds + 4) = here->MOS1gds;
                *(here->MOS1senGbs + 4) = here->MOS1gbs;
                *(here->MOS1senGbd + 4) = here->MOS1gbd;
                *(here->MOS1senGm + 4) = here->MOS1gm;
                *(here->MOS1senGmbs + 4) = here->MOS1gmbs;

                here->MOS1l = A0;

            }

            goto load;

pertw:    /* Perturbation of width */
            if(here->MOS1sens_w == 0)
                goto next;
#ifdef SENSDEBUG
            printf("\nPerturbation of width\n");
#endif /* SENSDEBUG */
            flag = 5;
            A0 = here->MOS1w;
            DELA = info->SENpertfac * A0;
            DELAinv = 1.0/DELA;
            Apert = A0 + DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed width */
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
                if(vbdOp >= here->MOS1tDepCap){
                    arg = 1-model->MOS1fwdCapDepCoeff;
                    sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) * 
                            log(arg) );
                    sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->MOS1f2d = here->MOS1Cbd*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                        +  here->MOS1Cbdsw*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->MOS1f3d = here->MOS1Cbd * 
                            model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                            here->MOS1tBulkPot + here->MOS1Cbdsw * 
                            model->MOS1bulkJctSideGradingCoeff * sargsw/arg /
                            here->MOS1tBulkPot;
                    here->MOS1f4d = here->MOS1Cbd*here->MOS1tBulkPot*
                            (1-arg*sarg)/ (1-model->MOS1bulkJctBotGradingCoeff)
                            + here->MOS1Cbdsw*here->MOS1tBulkPot*(1-arg*sargsw)/
                            (1-model->MOS1bulkJctSideGradingCoeff)
                            -here->MOS1f3d/2*
                            (here->MOS1tDepCap*here->MOS1tDepCap)
                            -here->MOS1tDepCap * here->MOS1f2d;
                }
                if(vbsOp >= here->MOS1tDepCap){
                    arg = 1-model->MOS1fwdCapDepCoeff;
                    sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) *
                            log(arg) );
                    sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->MOS1f2s = here->MOS1Cbs*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                        +  here->MOS1Cbssw*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->MOS1f3s = here->MOS1Cbs * 
                            model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                            here->MOS1tBulkPot + here->MOS1Cbssw * 
                            model->MOS1bulkJctSideGradingCoeff * sargsw/arg /
                            here->MOS1tBulkPot;
                    here->MOS1f4s = here->MOS1Cbs*
                            here->MOS1tBulkPot*(1-arg*sarg)/
                            (1-model->MOS1bulkJctBotGradingCoeff)
                            + here->MOS1Cbssw*here->MOS1tBulkPot*(1-arg*sargsw)/
                            (1-model->MOS1bulkJctSideGradingCoeff)
                            -here->MOS1f3s/2*
                            (here->MOS1tDepCap*here->MOS1tDepCap)
                            -here->MOS1tDepCap * here->MOS1f2s;
                }

                *(ckt->CKTstate0 + here->MOS1vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS1vbd) = vbdOp;

                if ((error = MOS1load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS1senCgs + 5) = here->MOS1cgs;
                *(here->MOS1senCgd + 5) = here->MOS1cgd;
                *(here->MOS1senCgb + 5) = here->MOS1cgb;
                *(here->MOS1senCbd + 5) = here->MOS1capbd;
                *(here->MOS1senCbs + 5) = here->MOS1capbs;
                *(here->MOS1senGds + 5) = here->MOS1gds;
                *(here->MOS1senGbs + 5) = here->MOS1gbs;
                *(here->MOS1senGbd + 5) = here->MOS1gbd;
                *(here->MOS1senGm + 5) = here->MOS1gm;
                *(here->MOS1senGmbs + 5) = here->MOS1gmbs;

                here->MOS1w = A0;
                here->MOS1drainArea /= (1 + info->SENpertfac);
                here->MOS1sourceArea /= (1 + info->SENpertfac);
            }

load:

            gds= *(here->MOS1senGds + flag);
            gbs= *(here->MOS1senGbs + flag);
            gbd= *(here->MOS1senGbd + flag);
            gm= *(here->MOS1senGm + flag);
            gmbs= *(here->MOS1senGmbs + flag);
            if(flag == 5){
                gdpr = here->MOS1drainConductance * Apert/A0;
                gspr = here->MOS1sourceConductance * Apert/A0;
            }
            else{
                gdpr = here->MOS1drainConductance;
                gspr = here->MOS1sourceConductance;
            }

            xcgs= *(here->MOS1senCgs + flag) * ckt->CKTomega;
            xcgd= *(here->MOS1senCgd + flag) * ckt->CKTomega;
            xcgb= *(here->MOS1senCgb + flag) * ckt->CKTomega;
            xbd= *(here->MOS1senCbd + flag) * ckt->CKTomega;
            xbs= *(here->MOS1senCbs + flag) * ckt->CKTomega;

#ifdef SENSDEBUG
            printf("flag = %d \n",flag);
            printf("gspr = %.7e , gdpr = %.7e , gds = %.7e, gbs = %.7e\n",
                    gspr,gdpr,gds,gbs);
            printf("gbd = %.7e , gm = %.7e , gmbs = %.7e\n",gbd,gm,gmbs);
            printf("xcgs = %.7e , xcgd = %.7e , xcgb = %.7e,", xcgs,xcgd,xcgb);
            printf("xbd = %.7e,xbs = %.7e\n",xbd,xbs);
#endif /* SENSDEBUG */
            cspr = gspr * vspr ;
            icspr = gspr * ivspr ;
            cdpr = gdpr * vdpr ;
            icdpr = gdpr * ivdpr ;
            cgs = ( - xcgs * ivgs );
            icgs =  xcgs * vgs ;
            cgd = ( - xcgd * ivgd );
            icgd =  xcgd * vgd ;
            cgb = ( - xcgb * ivgb );
            icgb =  xcgb * vgb ;
            cbs = ( gbs * vbs  - xbs * ivbs );
            icbs = ( xbs * vbs  + gbs * ivbs );
            cbd = ( gbd * vbd  - xbd * ivbd );
            icbd = ( xbd * vbd  + gbd * ivbd );
            cds = ( gds * vds  + xnrm * (gm * vgs + gmbs * vbs) 
                    - xrev * (gm * vgd + gmbs * vbd) );
            icds = ( gds * ivds + xnrm * (gm * ivgs + gmbs * ivbs)
                    - xrev * (gm * ivgd + gmbs * ivbd) );

            cs = cspr;
            ics = icspr;
            csprm = ( -cspr - cgs - cbs - cds ) ;
            icsprm = ( -icspr - icgs - icbs - icds ) ;
            cd = cdpr;
            icd = icdpr;
            cdprm = ( -cdpr - cgd - cbd + cds ) ;
            icdprm = ( -icdpr - icgd - icbd + icds ) ;
            cg = cgs + cgd + cgb ;
            icg = icgs + icgd + icgb ;
            cb = cbs + cbd - cgb ;
            icb = icbs + icbd - icgb ;

#ifdef SENSDEBUG
            printf("vbs = %.7e , vbd = %.7e , vgb = %.7e\n",vbs,vbd,vgb);
            printf("ivbs = %.7e , ivbd = %.7e , ivgb = %.7e\n",ivbs,ivbd,ivgb);
            printf("cbs = %.7e , cbd = %.7e , cgb = %.7e\n",cbs,cbd,cgb);
            printf("cb = %.7e , cg = %.7e , cs = %.7e\n",cb,cg,cs);
            printf("csprm = %.7e, cd = %.7e, cdprm = %.7e\n",csprm,cd,cdprm);
            printf("icb = %.7e , icg = %.7e , ics = %.7e\n",icb,icg,ics);
            printf("icsprm = %.7e, icd = %.7e, icdprm = %.7e\n",
                    icsprm,icd,icdprm);
#endif /* SENSDEBUG */
            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                if((flag == 4) && (iparmno != here->MOS1senParmNo)) continue;
                if( (flag == 5) && (iparmno != (here->MOS1senParmNo +
                        (int) here->MOS1sens_l))) continue;

                switch(flag){
                case 1: 
                    DvDp = model->MOS1type * 
                            (info->SEN_Sap[here->MOS1bNode][iparmno]
                            -  info->SEN_Sap[here->MOS1sNodePrime][iparmno]);
                    break;
                case 2: 
                    DvDp = model->MOS1type * 
                            ( info->SEN_Sap[here->MOS1bNode][iparmno]
                            -  info->SEN_Sap[here->MOS1dNodePrime][iparmno]);
                    break;
                case 3: 
                    DvDp = model->MOS1type * 
                            ( info->SEN_Sap[here->MOS1gNode][iparmno]
                            -  info->SEN_Sap[here->MOS1bNode][iparmno]);
                    break;
                case 4: 
                    DvDp = 1;
                    break;
                case 5: 
                    DvDp = 1;
                    break;
                }
                *(info->SEN_RHS[here->MOS1bNode] + iparmno) -=  
                        ( cb  - cb0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS1bNode] + iparmno) -=  
                        ( icb  - icb0) * DELAinv * DvDp;

                *(info->SEN_RHS[here->MOS1gNode] + iparmno) -=  
                        ( cg  - cg0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS1gNode] + iparmno) -=  
                        ( icg  - icg0) * DELAinv * DvDp;

                if(here->MOS1sNode != here->MOS1sNodePrime){
                    *(info->SEN_RHS[here->MOS1sNode] + iparmno) -=  
                            ( cs  - cs0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS1sNode] + iparmno) -=  
                            ( ics  - ics0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS1sNodePrime] + iparmno) -=  
                        ( csprm  - csprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS1sNodePrime] + iparmno) -=  
                        ( icsprm  - icsprm0) * DELAinv * DvDp;

                if(here->MOS1dNode != here->MOS1dNodePrime){
                    *(info->SEN_RHS[here->MOS1dNode] + iparmno) -=  
                            ( cd  - cd0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS1dNode] + iparmno) -=  
                            ( icd  - icd0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS1dNodePrime] + iparmno) -=  
                        ( cdprm  - cdprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS1dNodePrime] + iparmno) -=  
                        ( icdprm  - icdprm0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");  
                printf("DvDp = %.5e , DELAinv = %.5e ,flag = %d ,",
                        DvDp,DELAinv,flag);
                printf("iparmno = %d,senparmno = %d\n",
                        iparmno,here->MOS1senParmNo);
                printf("A0 = %.5e , Apert = %.5e ,CONSTvt = %.5e \n",
                        A0,Apert,here->MOS1tVto);
                printf("senb = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1bNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS1bNode] + iparmno));
                printf("seng = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1gNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS1gNode] + iparmno));
                printf("sens = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1sNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS1sNode] + iparmno));
                printf("sensprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1sNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS1sNodePrime] + iparmno));
                printf("send = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1dNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS1dNode] + iparmno));
                printf("sendprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS1dNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS1dNodePrime] + iparmno));
#endif /* SENSDEBUG */

            }
            switch(flag){
            case 1: 
                goto pertvbd ;
            case 2: 
                goto pertvgb ; 
            case 3: 
                goto pertl ;
            case 4: 
                goto pertw ;
            case 5: 
                break; 
            }
next:                   
            ;

            /* put the unperturbed values back into the state vector */
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
    printf("MOS1senacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}


