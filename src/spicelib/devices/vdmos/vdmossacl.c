/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

int
VDMOSsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel*)inModel;
    VDMOSinstance *here;
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
    printf("VDMOSsenacload\n");
    printf("CKTomega = %.5e\n",ckt->CKTomega);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;
    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here!= NULL;
                here = VDMOSnextInstance(here)) {

            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++)
                *(SaveState + i) = *(ckt->CKTstate0 + here->VDMOSstates + i);

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

            xnrm=1;
            xrev=0;
            if (here->VDMOSmode < 0) {
                xnrm=0;
                xrev=1;
            }

            vbsOp = model->VDMOStype * ( 
            *(ckt->CKTrhsOp+here->VDMOSbNode) -
                *(ckt->CKTrhsOp+here->VDMOSsNodePrime));
            vbdOp = model->VDMOStype * ( 
            *(ckt->CKTrhsOp+here->VDMOSbNode) -
                *(ckt->CKTrhsOp+here->VDMOSdNodePrime));
            vspr = *(ckt->CKTrhsOld + here->VDMOSsNode) 
                - *(ckt->CKTrhsOld +
                    here->VDMOSsNodePrime) ;
            ivspr = *(ckt->CKTirhsOld + here->VDMOSsNode) 
                - *(ckt->CKTirhsOld +
                    here->VDMOSsNodePrime) ;
            vdpr = *(ckt->CKTrhsOld + here->VDMOSdNode) 
                - *(ckt->CKTrhsOld +
                    here->VDMOSdNodePrime) ;
            ivdpr = *(ckt->CKTirhsOld + here->VDMOSdNode) 
                - *(ckt->CKTirhsOld +
                    here->VDMOSdNodePrime) ;
            vgb = *(ckt->CKTrhsOld + here->VDMOSgNode) 
                - *(ckt->CKTrhsOld +
                    here->VDMOSbNode) ;
            ivgb = *(ckt->CKTirhsOld + here->VDMOSgNode) 
                - *(ckt->CKTirhsOld +
                    here->VDMOSbNode) ;
            vbs = *(ckt->CKTrhsOld + here->VDMOSbNode) 
                - *(ckt->CKTrhsOld +
                    here->VDMOSsNodePrime) ;
            ivbs = *(ckt->CKTirhsOld + here->VDMOSbNode) 
                - *(ckt->CKTirhsOld +
                    here->VDMOSsNodePrime) ;
            vbd = *(ckt->CKTrhsOld + here->VDMOSbNode) 
                - *(ckt->CKTrhsOld +
                    here->VDMOSdNodePrime) ;
            ivbd = *(ckt->CKTirhsOld + here->VDMOSbNode) 
                - *(ckt->CKTirhsOld +
                    here->VDMOSdNodePrime) ;
            vds = vbs - vbd ;
            ivds = ivbs - ivbd ;
            vgs = vgb + vbs ;
            ivgs = ivgb + ivbs ;
            vgd = vgb + vbd ;
            ivgd = ivgb + ivbd ;

#ifdef SENSDEBUG
            printf("senacload instance name %s\n",here->VDMOSname);
            printf("gate = %d ,drain = %d, drainprm = %d\n", 
                    here->VDMOSgNode,here->VDMOSdNode,here->VDMOSdNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->VDMOSsNode ,here->VDMOSsNodePrime,here->VDMOSbNode,
                    here->VDMOSsenParmNo);
            printf("\n without  perturbation \n");
#endif /* SENSDEBUG */
            /*  without  perturbation  */
            *(ckt->CKTstate0 + here->VDMOSvbs) = vbsOp;
            *(ckt->CKTstate0 + here->VDMOSvbd) = vbdOp;

            here->VDMOSsenPertFlag = ON ;
            if(info->SENacpertflag == 1){
                /* store the  unperturbed values of small signal parameters */
                if((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->VDMOSsenCgs) = here->VDMOScgs;
                *(here->VDMOSsenCgd) = here->VDMOScgd;
                *(here->VDMOSsenCgb) = here->VDMOScgb;
                *(here->VDMOSsenCbd) = here->VDMOScapbd;
                *(here->VDMOSsenCbs) = here->VDMOScapbs;
                *(here->VDMOSsenGds) = here->VDMOSgds;
                *(here->VDMOSsenGbs) = here->VDMOSgbs;
                *(here->VDMOSsenGbd) = here->VDMOSgbd;
                *(here->VDMOSsenGm) = here->VDMOSgm;
                *(here->VDMOSsenGmbs) = here->VDMOSgmbs;

            }
            xcgs0= *(here->VDMOSsenCgs) * ckt->CKTomega;
            xcgd0= *(here->VDMOSsenCgd) * ckt->CKTomega;
            xcgb0= *(here->VDMOSsenCgb) * ckt->CKTomega;
            xbd0= *(here->VDMOSsenCbd) * ckt->CKTomega;
            xbs0= *(here->VDMOSsenCbs) * ckt->CKTomega;
            gds0= *(here->VDMOSsenGds);
            gbs0= *(here->VDMOSsenGbs);
            gbd0= *(here->VDMOSsenGbd);
            gm0= *(here->VDMOSsenGm);
            gmbs0= *(here->VDMOSsenGmbs);
            gdpr0 = here->VDMOSdrainConductance;
            gspr0 = here->VDMOSsourceConductance;


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
            DELA =  info->SENpertfac * here->VDMOStVto ;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbs */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->VDMOSvbs) = Apert;
                *(ckt->CKTstate0 + here->VDMOSvbd) = vbdOp;

                if((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->VDMOSsenCgs + 1) = here->VDMOScgs;
                *(here->VDMOSsenCgd + 1) = here->VDMOScgd;
                *(here->VDMOSsenCgb + 1) = here->VDMOScgb;
                *(here->VDMOSsenCbd + 1) = here->VDMOScapbd;
                *(here->VDMOSsenCbs + 1) = here->VDMOScapbs;
                *(here->VDMOSsenGds + 1) = here->VDMOSgds;
                *(here->VDMOSsenGbs + 1) = here->VDMOSgbs;
                *(here->VDMOSsenGbd + 1) = here->VDMOSgbd;
                *(here->VDMOSsenGm + 1) = here->VDMOSgm;
                *(here->VDMOSsenGmbs + 1) = here->VDMOSgmbs;

                *(ckt->CKTstate0 + here->VDMOSvbs) = A0;


            }

            goto load;


pertvbd:  /* Perturbation of vbd */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbd\n");
#endif /* SENSDEBUG */

            flag = 2;
            A0 = vbdOp;
            DELA =  info->SENpertfac * here->VDMOStVto + 1e-8;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vbd */
                Apert = A0 + DELA;
                *(ckt->CKTstate0 + here->VDMOSvbs) = vbsOp;
                *(ckt->CKTstate0 + here->VDMOSvbd) = Apert;

                if((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->VDMOSsenCgs + 2) = here->VDMOScgs;
                *(here->VDMOSsenCgd + 2) = here->VDMOScgd;
                *(here->VDMOSsenCgb + 2) = here->VDMOScgb;
                *(here->VDMOSsenCbd + 2) = here->VDMOScapbd;
                *(here->VDMOSsenCbs + 2) = here->VDMOScapbs;
                *(here->VDMOSsenGds + 2) = here->VDMOSgds;
                *(here->VDMOSsenGbs + 2) = here->VDMOSgbs;
                *(here->VDMOSsenGbd + 2) = here->VDMOSgbd;
                *(here->VDMOSsenGm + 2) = here->VDMOSgm;
                *(here->VDMOSsenGmbs + 2) = here->VDMOSgmbs;

                *(ckt->CKTstate0 + here->VDMOSvbd) = A0;

            }

            goto load;


pertvgb:  /* Perturbation of vgb */
#ifdef SENSDEBUG
            printf("\nPerturbation of vgb\n");
#endif /* SENSDEBUG */

            flag = 3;
            A0 = model->VDMOStype * (*(ckt->CKTrhsOp + here->VDMOSgNode) 
                -  *(ckt->CKTrhsOp + here->VDMOSbNode)); 
            DELA =  info->SENpertfac * A0 + 1e-8;
            DELAinv = model->VDMOStype * 1.0/DELA;


            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vgb */
                *(ckt->CKTstate0 + here->VDMOSvbs) = vbsOp;
                *(ckt->CKTstate0 + here->VDMOSvbd) = vbdOp;
                *(ckt->CKTrhsOp + here->VDMOSbNode) -= DELA; 

                if((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->VDMOSsenCgs + 3) = here->VDMOScgs;
                *(here->VDMOSsenCgd + 3) = here->VDMOScgd;
                *(here->VDMOSsenCgb + 3) = here->VDMOScgb;
                *(here->VDMOSsenCbd + 3) = here->VDMOScapbd;
                *(here->VDMOSsenCbs + 3) = here->VDMOScapbs;
                *(here->VDMOSsenGds + 3) = here->VDMOSgds;
                *(here->VDMOSsenGbs + 3) = here->VDMOSgbs;
                *(here->VDMOSsenGbd + 3) = here->VDMOSgbd;
                *(here->VDMOSsenGm + 3) = here->VDMOSgm;
                *(here->VDMOSsenGmbs + 3) = here->VDMOSgmbs;


                *(ckt->CKTrhsOp + here->VDMOSbNode) += DELA; 
            }
            goto load;

pertl:    /* Perturbation of length */

            if(here->VDMOSsens_l == 0){
                goto pertw;
            }
#ifdef SENSDEBUG
            printf("\nPerturbation of length\n");
#endif /* SENSDEBUG */
            flag = 4;
            A0 = here->VDMOSl;
            DELA =  info->SENpertfac * A0;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed length */
                Apert = A0 + DELA;
                here->VDMOSl = Apert;

                *(ckt->CKTstate0 + here->VDMOSvbs) = vbsOp;
                *(ckt->CKTstate0 + here->VDMOSvbd) = vbdOp;

                if ((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->VDMOSsenCgs + 4) = here->VDMOScgs;
                *(here->VDMOSsenCgd + 4) = here->VDMOScgd;
                *(here->VDMOSsenCgb + 4) = here->VDMOScgb;
                *(here->VDMOSsenCbd + 4) = here->VDMOScapbd;
                *(here->VDMOSsenCbs + 4) = here->VDMOScapbs;
                *(here->VDMOSsenGds + 4) = here->VDMOSgds;
                *(here->VDMOSsenGbs + 4) = here->VDMOSgbs;
                *(here->VDMOSsenGbd + 4) = here->VDMOSgbd;
                *(here->VDMOSsenGm + 4) = here->VDMOSgm;
                *(here->VDMOSsenGmbs + 4) = here->VDMOSgmbs;

                here->VDMOSl = A0;

            }

            goto load;

pertw:    /* Perturbation of width */
            if(here->VDMOSsens_w == 0)
                goto next;
#ifdef SENSDEBUG
            printf("\nPerturbation of width\n");
#endif /* SENSDEBUG */
            flag = 5;
            A0 = here->VDMOSw;
            DELA = info->SENpertfac * A0;
            DELAinv = 1.0/DELA;
            Apert = A0 + DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed width */
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
                if(vbdOp >= here->VDMOStDepCap){
                    arg = 1-model->VDMOSfwdCapDepCoeff;
                    sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) * 
                            log(arg) );
                    sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->VDMOSf2d = here->VDMOSCbd*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                        +  here->VDMOSCbdsw*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->VDMOSf3d = here->VDMOSCbd * 
                            model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                            here->VDMOStBulkPot + here->VDMOSCbdsw * 
                            model->VDMOSbulkJctSideGradingCoeff * sargsw/arg /
                            here->VDMOStBulkPot;
                    here->VDMOSf4d = here->VDMOSCbd*here->VDMOStBulkPot*
                            (1-arg*sarg)/ (1-model->VDMOSbulkJctBotGradingCoeff)
                            + here->VDMOSCbdsw*here->VDMOStBulkPot*(1-arg*sargsw)/
                            (1-model->VDMOSbulkJctSideGradingCoeff)
                            -here->VDMOSf3d/2*
                            (here->VDMOStDepCap*here->VDMOStDepCap)
                            -here->VDMOStDepCap * here->VDMOSf2d;
                }
                if(vbsOp >= here->VDMOStDepCap){
                    arg = 1-model->VDMOSfwdCapDepCoeff;
                    sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) *
                            log(arg) );
                    sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->VDMOSf2s = here->VDMOSCbs*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                        +  here->VDMOSCbssw*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->VDMOSf3s = here->VDMOSCbs * 
                            model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                            here->VDMOStBulkPot + here->VDMOSCbssw * 
                            model->VDMOSbulkJctSideGradingCoeff * sargsw/arg /
                            here->VDMOStBulkPot;
                    here->VDMOSf4s = here->VDMOSCbs*
                            here->VDMOStBulkPot*(1-arg*sarg)/
                            (1-model->VDMOSbulkJctBotGradingCoeff)
                            + here->VDMOSCbssw*here->VDMOStBulkPot*(1-arg*sargsw)/
                            (1-model->VDMOSbulkJctSideGradingCoeff)
                            -here->VDMOSf3s/2*
                            (here->VDMOStDepCap*here->VDMOStDepCap)
                            -here->VDMOStDepCap * here->VDMOSf2s;
                }

                *(ckt->CKTstate0 + here->VDMOSvbs) = vbsOp;
                *(ckt->CKTstate0 + here->VDMOSvbd) = vbdOp;

                if ((error = VDMOSload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->VDMOSsenCgs + 5) = here->VDMOScgs;
                *(here->VDMOSsenCgd + 5) = here->VDMOScgd;
                *(here->VDMOSsenCgb + 5) = here->VDMOScgb;
                *(here->VDMOSsenCbd + 5) = here->VDMOScapbd;
                *(here->VDMOSsenCbs + 5) = here->VDMOScapbs;
                *(here->VDMOSsenGds + 5) = here->VDMOSgds;
                *(here->VDMOSsenGbs + 5) = here->VDMOSgbs;
                *(here->VDMOSsenGbd + 5) = here->VDMOSgbd;
                *(here->VDMOSsenGm + 5) = here->VDMOSgm;
                *(here->VDMOSsenGmbs + 5) = here->VDMOSgmbs;

                here->VDMOSw = A0;
                here->VDMOSdrainArea /= (1 + info->SENpertfac);
                here->VDMOSsourceArea /= (1 + info->SENpertfac);
            }

load:

            gds= *(here->VDMOSsenGds + flag);
            gbs= *(here->VDMOSsenGbs + flag);
            gbd= *(here->VDMOSsenGbd + flag);
            gm= *(here->VDMOSsenGm + flag);
            gmbs= *(here->VDMOSsenGmbs + flag);
            if(flag == 5){
                gdpr = here->VDMOSdrainConductance * Apert/A0;
                gspr = here->VDMOSsourceConductance * Apert/A0;
            }
            else{
                gdpr = here->VDMOSdrainConductance;
                gspr = here->VDMOSsourceConductance;
            }

            xcgs= *(here->VDMOSsenCgs + flag) * ckt->CKTomega;
            xcgd= *(here->VDMOSsenCgd + flag) * ckt->CKTomega;
            xcgb= *(here->VDMOSsenCgb + flag) * ckt->CKTomega;
            xbd= *(here->VDMOSsenCbd + flag) * ckt->CKTomega;
            xbs= *(here->VDMOSsenCbs + flag) * ckt->CKTomega;

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
                if((flag == 4) && (iparmno != here->VDMOSsenParmNo)) continue;
                if( (flag == 5) && (iparmno != (here->VDMOSsenParmNo +
                        here->VDMOSsens_l))) continue;

                switch(flag){
                case 1: 
                    DvDp = model->VDMOStype * 
                            (info->SEN_Sap[here->VDMOSbNode][iparmno]
                            -  info->SEN_Sap[here->VDMOSsNodePrime][iparmno]);
                    break;
                case 2: 
                    DvDp = model->VDMOStype * 
                            ( info->SEN_Sap[here->VDMOSbNode][iparmno]
                            -  info->SEN_Sap[here->VDMOSdNodePrime][iparmno]);
                    break;
                case 3: 
                    DvDp = model->VDMOStype * 
                            ( info->SEN_Sap[here->VDMOSgNode][iparmno]
                            -  info->SEN_Sap[here->VDMOSbNode][iparmno]);
                    break;
                case 4: 
                    DvDp = 1;
                    break;
                case 5: 
                    DvDp = 1;
                    break;
                }
                *(info->SEN_RHS[here->VDMOSbNode] + iparmno) -=  
                        ( cb  - cb0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->VDMOSbNode] + iparmno) -=  
                        ( icb  - icb0) * DELAinv * DvDp;

                *(info->SEN_RHS[here->VDMOSgNode] + iparmno) -=  
                        ( cg  - cg0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->VDMOSgNode] + iparmno) -=  
                        ( icg  - icg0) * DELAinv * DvDp;

                if(here->VDMOSsNode != here->VDMOSsNodePrime){
                    *(info->SEN_RHS[here->VDMOSsNode] + iparmno) -=  
                            ( cs  - cs0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->VDMOSsNode] + iparmno) -=  
                            ( ics  - ics0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->VDMOSsNodePrime] + iparmno) -=  
                        ( csprm  - csprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->VDMOSsNodePrime] + iparmno) -=  
                        ( icsprm  - icsprm0) * DELAinv * DvDp;

                if(here->VDMOSdNode != here->VDMOSdNodePrime){
                    *(info->SEN_RHS[here->VDMOSdNode] + iparmno) -=  
                            ( cd  - cd0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->VDMOSdNode] + iparmno) -=  
                            ( icd  - icd0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->VDMOSdNodePrime] + iparmno) -=  
                        ( cdprm  - cdprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->VDMOSdNodePrime] + iparmno) -=  
                        ( icdprm  - icdprm0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");  
                printf("DvDp = %.5e , DELAinv = %.5e ,flag = %d ,",
                        DvDp,DELAinv,flag);
                printf("iparmno = %d,senparmno = %d\n",
                        iparmno,here->VDMOSsenParmNo);
                printf("A0 = %.5e , Apert = %.5e ,CONSTvt = %.5e \n",
                        A0,Apert,here->VDMOStVto);
                printf("senb = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSbNode] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSbNode] + iparmno));
                printf("seng = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSgNode] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSgNode] + iparmno));
                printf("sens = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSsNode] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSsNode] + iparmno));
                printf("sensprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSsNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSsNodePrime] + iparmno));
                printf("send = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSdNode] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSdNode] + iparmno));
                printf("sendprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->VDMOSdNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->VDMOSdNodePrime] + iparmno));
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
    printf("VDMOSsenacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}


