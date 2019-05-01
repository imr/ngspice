/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2sAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
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
    printf("MOS2senacload\n");
    printf("CKTomega = %.5e\n",ckt->CKTomega);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;
    for( ; model != NULL; model = MOS2nextModel(model)) {
        for(here = MOS2instances(model); here!= NULL;
                here = MOS2nextInstance(here)) {

            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++)
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS2states + i);

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

            xnrm=1;
            xrev=0;
            if (here->MOS2mode < 0) {
                xnrm=0;
                xrev=1;
            }

            vbsOp = model->MOS2type * ( 
            *(ckt->CKTrhsOp+here->MOS2bNode) -
                *(ckt->CKTrhsOp+here->MOS2sNodePrime));
            vbdOp = model->MOS2type * ( 
            *(ckt->CKTrhsOp+here->MOS2bNode) -
                *(ckt->CKTrhsOp+here->MOS2dNodePrime));
            vspr = *(ckt->CKTrhsOld + here->MOS2sNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS2sNodePrime) ;
            ivspr = *(ckt->CKTirhsOld + here->MOS2sNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS2sNodePrime) ;
            vdpr = *(ckt->CKTrhsOld + here->MOS2dNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS2dNodePrime) ;
            ivdpr = *(ckt->CKTirhsOld + here->MOS2dNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS2dNodePrime) ;
            vgb = *(ckt->CKTrhsOld + here->MOS2gNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS2bNode) ;
            ivgb = *(ckt->CKTirhsOld + here->MOS2gNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS2bNode) ;
            vbs = *(ckt->CKTrhsOld + here->MOS2bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS2sNodePrime) ;
            ivbs = *(ckt->CKTirhsOld + here->MOS2bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS2sNodePrime) ;
            vbd = *(ckt->CKTrhsOld + here->MOS2bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS2dNodePrime) ;
            ivbd = *(ckt->CKTirhsOld + here->MOS2bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS2dNodePrime) ;
            vds = vbs - vbd ;
            ivds = ivbs - ivbd ;
            vgs = vgb + vbs ;
            ivgs = ivgb + ivbs ;
            vgd = vgb + vbd ;
            ivgd = ivgb + ivbd ;

#ifdef SENSDEBUG
            printf("senacload instance name %s\n",here->MOS2name);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->MOS2gNode,here->MOS2dNode,here->MOS2dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS2sNode ,here->MOS2sNodePrime,
                    here->MOS2bNode,here->MOS2senParmNo);
            printf("\n without  perturbation \n");
#endif /* SENSDEBUG */
            /*  without  perturbation  */
            *(ckt->CKTstate0 + here->MOS2vbs) = vbsOp;
            *(ckt->CKTstate0 + here->MOS2vbd) = vbdOp;

            here->MOS2senPertFlag = ON ;
            if(info->SENacpertflag == 1){
                /* store the  unperturbed values of small signal parameters */
                if((error = MOS2load((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->MOS2senCgs) = here->MOS2cgs;
                *(here->MOS2senCgd) = here->MOS2cgd;
                *(here->MOS2senCgb) = here->MOS2cgb;
                *(here->MOS2senCbd) = here->MOS2capbd;
                *(here->MOS2senCbs) = here->MOS2capbs;
                *(here->MOS2senGds) = here->MOS2gds;
                *(here->MOS2senGbs) = here->MOS2gbs;
                *(here->MOS2senGbd) = here->MOS2gbd;
                *(here->MOS2senGm) = here->MOS2gm;
                *(here->MOS2senGmbs) = here->MOS2gmbs;

            }
            xcgs0= *(here->MOS2senCgs) * ckt->CKTomega;
            xcgd0= *(here->MOS2senCgd) * ckt->CKTomega;
            xcgb0= *(here->MOS2senCgb) * ckt->CKTomega;
            xbd0= *(here->MOS2senCbd) * ckt->CKTomega;
            xbs0= *(here->MOS2senCbs) * ckt->CKTomega;
            gds0= *(here->MOS2senGds);
            gbs0= *(here->MOS2senGbs);
            gbd0= *(here->MOS2senGbd);
            gm0= *(here->MOS2senGm);
            gmbs0= *(here->MOS2senGmbs);
            gdpr0 = here->MOS2drainConductance;
            gspr0 = here->MOS2sourceConductance;


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
            printf("xbd0 = %.7e,xbs0 = %.7e\n", xbd0,xbs0);
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
            DELA =  info->SENpertfac * CONSTvt0 ;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbs */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->MOS2vbs) = Apert;
                *(ckt->CKTstate0 + here->MOS2vbd) = vbdOp;

                if((error = MOS2load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS2senCgs + 1) = here->MOS2cgs;
                *(here->MOS2senCgd + 1) = here->MOS2cgd;
                *(here->MOS2senCgb + 1) = here->MOS2cgb;
                *(here->MOS2senCbd + 1) = here->MOS2capbd;
                *(here->MOS2senCbs + 1) = here->MOS2capbs;
                *(here->MOS2senGds + 1) = here->MOS2gds;
                *(here->MOS2senGbs + 1) = here->MOS2gbs;
                *(here->MOS2senGbd + 1) = here->MOS2gbd;
                *(here->MOS2senGm + 1) = here->MOS2gm;
                *(here->MOS2senGmbs + 1) = here->MOS2gmbs;

                *(ckt->CKTstate0 + here->MOS2vbs) = A0;


            }

            goto load;


pertvbd:  /* Perturbation of vbd */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbd\n");
#endif /* SENSDEBUG */

            flag = 2;
            A0 = vbdOp;
            DELA =  info->SENpertfac * CONSTvt0 + 1e-8;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbd */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->MOS2vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS2vbd) = Apert;

                if((error = MOS2load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS2senCgs + 2) = here->MOS2cgs;
                *(here->MOS2senCgd + 2) = here->MOS2cgd;
                *(here->MOS2senCgb + 2) = here->MOS2cgb;
                *(here->MOS2senCbd + 2) = here->MOS2capbd;
                *(here->MOS2senCbs + 2) = here->MOS2capbs;
                *(here->MOS2senGds + 2) = here->MOS2gds;
                *(here->MOS2senGbs + 2) = here->MOS2gbs;
                *(here->MOS2senGbd + 2) = here->MOS2gbd;
                *(here->MOS2senGm + 2) = here->MOS2gm;
                *(here->MOS2senGmbs + 2) = here->MOS2gmbs;

                *(ckt->CKTstate0 + here->MOS2vbd) = A0;

            }

            goto load;


pertvgb:  /* Perturbation of vgb */
#ifdef SENSDEBUG
            printf("\nPerturbation of vgb\n");
#endif /* SENSDEBUG */

            flag = 3;
            A0 = model->MOS2type * (*(ckt->CKTrhsOp + here->MOS2gNode) 
                -  *(ckt->CKTrhsOp + here->MOS2bNode)); 
            DELA =  info->SENpertfac * A0 + 1e-8;
            DELAinv = model->MOS2type * 1.0/DELA;


            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vgb */
                *(ckt->CKTstate0 + here->MOS2vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS2vbd) = vbdOp;
                *(ckt->CKTrhsOp + here->MOS2bNode) -= DELA; 

                if((error = MOS2load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS2senCgs + 3) = here->MOS2cgs;
                *(here->MOS2senCgd + 3) = here->MOS2cgd;
                *(here->MOS2senCgb + 3) = here->MOS2cgb;
                *(here->MOS2senCbd + 3) = here->MOS2capbd;
                *(here->MOS2senCbs + 3) = here->MOS2capbs;
                *(here->MOS2senGds + 3) = here->MOS2gds;
                *(here->MOS2senGbs + 3) = here->MOS2gbs;
                *(here->MOS2senGbd + 3) = here->MOS2gbd;
                *(here->MOS2senGm + 3) = here->MOS2gm;
                *(here->MOS2senGmbs + 3) = here->MOS2gmbs;


                *(ckt->CKTrhsOp + here->MOS2bNode) += DELA; 
            }
            goto load;

pertl:    /* Perturbation of length */

            if(here->MOS2sens_l == 0){
                goto pertw;
            }
#ifdef SENSDEBUG
            printf("\nPerturbation of length\n");
#endif /* SENSDEBUG */
            flag = 4;
            A0 = here->MOS2l;
            DELA =  info->SENpertfac * A0;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed length */
                Apert = A0 + DELA;
                here->MOS2l = Apert;

                *(ckt->CKTstate0 + here->MOS2vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS2vbd) = vbdOp;

                if((error = MOS2load((GENmodel*)model,ckt)) != 0) return(error);

                *(here->MOS2senCgs + 4) = here->MOS2cgs;
                *(here->MOS2senCgd + 4) = here->MOS2cgd;
                *(here->MOS2senCgb + 4) = here->MOS2cgb;
                *(here->MOS2senCbd + 4) = here->MOS2capbd;
                *(here->MOS2senCbs + 4) = here->MOS2capbs;
                *(here->MOS2senGds + 4) = here->MOS2gds;
                *(here->MOS2senGbs + 4) = here->MOS2gbs;
                *(here->MOS2senGbd + 4) = here->MOS2gbd;
                *(here->MOS2senGm + 4) = here->MOS2gm;
                *(here->MOS2senGmbs + 4) = here->MOS2gmbs;

                here->MOS2l = A0;

            }

            goto load;

pertw:    /* Perturbation of width */
            if(here->MOS2sens_w == 0)
                goto next;
#ifdef SENSDEBUG
            printf("\nPerturbation of width\n");
#endif /* SENSDEBUG */
            flag = 5;
            A0 = here->MOS2w;
            DELA = info->SENpertfac * A0;
            DELAinv = 1.0/DELA;
            Apert = A0 + DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed width */
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
                if(vbdOp >= here->MOS2tDepCap){
                    arg = 1-model->MOS2fwdCapDepCoeff;
                    sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) *
                            log(arg) );
                    sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->MOS2f2d = here->MOS2Cbd*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                        +  here->MOS2Cbdsw*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->MOS2f3d = here->MOS2Cbd * 
                            model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                            here->MOS2tBulkPot + here->MOS2Cbdsw * 
                            model->MOS2bulkJctSideGradingCoeff * sargsw/arg /
                            here->MOS2tBulkPot;
                    here->MOS2f4d = here->MOS2Cbd*here->MOS2tBulkPot*
                            (1-arg*sarg)/(1-model->MOS2bulkJctBotGradingCoeff)
                            + here->MOS2Cbdsw*here->MOS2tBulkPot*(1-arg*sargsw)/
                            (1-model->MOS2bulkJctSideGradingCoeff)
                            -here->MOS2f3d/2*
                            (here->MOS2tDepCap*here->MOS2tDepCap)
                            -here->MOS2tDepCap * here->MOS2f2d;
                }
                if(vbsOp >= here->MOS2tDepCap){
                    arg = 1-model->MOS2fwdCapDepCoeff;
                    sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) * 
                            log(arg) );
                    sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) *
                            log(arg) );
                    here->MOS2f2s = here->MOS2Cbs*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                        +  here->MOS2Cbssw*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->MOS2f3s = here->MOS2Cbs * 
                            model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                            here->MOS2tBulkPot + here->MOS2Cbssw * 
                            model->MOS2bulkJctSideGradingCoeff * sargsw/arg /
                            here->MOS2tBulkPot;
                    here->MOS2f4s = here->MOS2Cbs*here->MOS2tBulkPot*
                            (1-arg*sarg)/(1-model->MOS2bulkJctBotGradingCoeff)
                            + here->MOS2Cbssw*here->MOS2tBulkPot*(1-arg*sargsw)/
                            (1-model->MOS2bulkJctSideGradingCoeff)
                            -here->MOS2f3s/2*
                            (here->MOS2tDepCap*here->MOS2tDepCap)
                            -here->MOS2tDepCap * here->MOS2f2s;
                }

                *(ckt->CKTstate0 + here->MOS2vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS2vbd) = vbdOp;

                if((error = MOS2load((GENmodel*)model,ckt)) != 0) return(error);

                *(here->MOS2senCgs + 5) = here->MOS2cgs;
                *(here->MOS2senCgd + 5) = here->MOS2cgd;
                *(here->MOS2senCgb + 5) = here->MOS2cgb;
                *(here->MOS2senCbd + 5) = here->MOS2capbd;
                *(here->MOS2senCbs + 5) = here->MOS2capbs;
                *(here->MOS2senGds + 5) = here->MOS2gds;
                *(here->MOS2senGbs + 5) = here->MOS2gbs;
                *(here->MOS2senGbd + 5) = here->MOS2gbd;
                *(here->MOS2senGm + 5) = here->MOS2gm;
                *(here->MOS2senGmbs + 5) = here->MOS2gmbs;

                here->MOS2w = A0;
                here->MOS2drainArea /= (1 + info->SENpertfac);
                here->MOS2sourceArea /= (1 + info->SENpertfac);
            }

load:

            gds= *(here->MOS2senGds + flag);
            gbs= *(here->MOS2senGbs + flag);
            gbd= *(here->MOS2senGbd + flag);
            gm= *(here->MOS2senGm + flag);
            gmbs= *(here->MOS2senGmbs + flag);
            if(flag == 5){
                gdpr = here->MOS2drainConductance * Apert/A0;
                gspr = here->MOS2sourceConductance * Apert/A0;
            }
            else{
                gdpr = here->MOS2drainConductance;
                gspr = here->MOS2sourceConductance;
            }

            xcgs= *(here->MOS2senCgs + flag) * ckt->CKTomega;
            xcgd= *(here->MOS2senCgd + flag) * ckt->CKTomega;
            xcgb= *(here->MOS2senCgb + flag) * ckt->CKTomega;
            xbd= *(here->MOS2senCbd + flag) * ckt->CKTomega;
            xbs= *(here->MOS2senCbs + flag) * ckt->CKTomega;

#ifdef SENSDEBUG
            printf("flag = %d \n",flag);
            printf("gspr = %.7e , gdpr = %.7e , gds = %.7e, gbs = %.7e\n",
                    gspr,gdpr,gds,gbs);
            printf("gbd = %.7e , gm = %.7e , gmbs = %.7e\n",gbd,gm,gmbs);
            printf("xcgs = %.7e , xcgd = %.7e , xcgb = %.7e,", xcgs,xcgd,xcgb);
            printf("xbd = %.7e,xbs = %.7e\n", xbd,xbs);
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
                if ((flag == 4) || (iparmno != here->MOS2senParmNo) ) continue;
                if ((flag == 5) || (iparmno != (here->MOS2senParmNo +
                        (int) here->MOS2sens_l)) ) continue;
                switch(flag){
                case 1: 
                    DvDp = model->MOS2type * 
                            (info->SEN_Sap[here->MOS2bNode][iparmno]
                            -  info->SEN_Sap[here->MOS2sNodePrime][iparmno]);
                    break;
                case 2: 
                    DvDp = model->MOS2type * 
                            ( info->SEN_Sap[here->MOS2bNode][iparmno]
                            -  info->SEN_Sap[here->MOS2dNodePrime][iparmno]);
                    break;
                case 3: 
                    DvDp = model->MOS2type * 
                            ( info->SEN_Sap[here->MOS2gNode][iparmno]
                            -  info->SEN_Sap[here->MOS2bNode][iparmno]);
                    break;
                case 4: 
                    DvDp = 1;
                    break;
                case 5: 
                    DvDp = 1;
                    break;
                }
                *(info->SEN_RHS[here->MOS2bNode] + iparmno) -=  
                        ( cb  - cb0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS2bNode] + iparmno) -=  
                        ( icb  - icb0) * DELAinv * DvDp;

                *(info->SEN_RHS[here->MOS2gNode] + iparmno) -=  
                        ( cg  - cg0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS2gNode] + iparmno) -=  
                        ( icg  - icg0) * DELAinv * DvDp;

                if(here->MOS2sNode != here->MOS2sNodePrime){
                    *(info->SEN_RHS[here->MOS2sNode] + iparmno) -=  
                            ( cs  - cs0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS2sNode] + iparmno) -=  
                            ( ics  - ics0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS2sNodePrime] + iparmno) -=  
                        ( csprm  - csprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS2sNodePrime] + iparmno) -=  
                        ( icsprm  - icsprm0) * DELAinv * DvDp;

                if(here->MOS2dNode != here->MOS2dNodePrime){
                    *(info->SEN_RHS[here->MOS2dNode] + iparmno) -=  
                            ( cd  - cd0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS2dNode] + iparmno) -=  
                            ( icd  - icd0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS2dNodePrime] + iparmno) -=  
                        ( cdprm  - cdprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS2dNodePrime] + iparmno) -=  
                        ( icdprm  - icdprm0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");  
                printf("DvDp = %.5e , DELAinv = %.5e ,flag = %d ,",
                        DvDp,DELAinv,flag);
                printf("iparmno = %d,senparmno = %d\n"
                        ,iparmno,here->MOS2senParmNo);
                printf("A0 = %.5e , Apert = %.5e ,CONSTvt0 = %.5e \n",
                        A0,Apert,CONSTvt0);
                printf("senb = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2bNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS2bNode] + iparmno));
                printf("seng = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2gNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS2gNode] + iparmno));
                printf("sens = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2sNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS2sNode] + iparmno));
                printf("sensprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2sNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS2sNodePrime] + iparmno));
                printf("send = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2dNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS2dNode] + iparmno));
                printf("sendprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS2dNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS2dNodePrime] + iparmno));
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
    printf("MOS2senacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}


