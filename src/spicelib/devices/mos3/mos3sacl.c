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
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS3sAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
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
    printf("MOS3senacload\n");
    printf("CKTomega = %.5e\n",ckt->CKTomega);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;
    for( ; model != NULL; model = MOS3nextModel(model)) {
        for(here = MOS3instances(model); here!= NULL;
                here = MOS3nextInstance(here)) {

            /* save the unperturbed values in the state vector */
            for(i=0; i <= 16; i++)
                *(SaveState + i) = *(ckt->CKTstate0 + here->MOS3states + i);

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
            save_mode  = here->MOS3mode;  

            xnrm=1;
            xrev=0;
            if (here->MOS3mode < 0) {
                xnrm=0;
                xrev=1;
            }

            vbsOp = model->MOS3type * ( 
            *(ckt->CKTrhsOp+here->MOS3bNode) -
                *(ckt->CKTrhsOp+here->MOS3sNodePrime));
            vbdOp = model->MOS3type * ( 
            *(ckt->CKTrhsOp+here->MOS3bNode) -
                *(ckt->CKTrhsOp+here->MOS3dNodePrime));
            vspr = *(ckt->CKTrhsOld + here->MOS3sNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS3sNodePrime) ;
            ivspr = *(ckt->CKTirhsOld + here->MOS3sNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS3sNodePrime) ;
            vdpr = *(ckt->CKTrhsOld + here->MOS3dNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS3dNodePrime) ;
            ivdpr = *(ckt->CKTirhsOld + here->MOS3dNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS3dNodePrime) ;
            vgb = *(ckt->CKTrhsOld + here->MOS3gNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS3bNode) ;
            ivgb = *(ckt->CKTirhsOld + here->MOS3gNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS3bNode) ;
            vbs = *(ckt->CKTrhsOld + here->MOS3bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS3sNodePrime) ;
            ivbs = *(ckt->CKTirhsOld + here->MOS3bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS3sNodePrime) ;
            vbd = *(ckt->CKTrhsOld + here->MOS3bNode) 
                - *(ckt->CKTrhsOld +
                    here->MOS3dNodePrime) ;
            ivbd = *(ckt->CKTirhsOld + here->MOS3bNode) 
                - *(ckt->CKTirhsOld +
                    here->MOS3dNodePrime) ;
            vds = vbs - vbd ;
            ivds = ivbs - ivbd ;
            vgs = vgb + vbs ;
            ivgs = ivgb + ivbs ;
            vgd = vgb + vbd ;
            ivgd = ivgb + ivbd ;

#ifdef SENSDEBUG
            printf("senacload instance name %s\n",here->MOS3name);
            printf("gate = %d ,drain = %d, drainprm = %d\n",
                    here->MOS3gNode,here->MOS3dNode,here->MOS3dNodePrime);
            printf("source = %d , sourceprm = %d ,body = %d, senparmno = %d\n",
                    here->MOS3sNode ,here->MOS3sNodePrime,
                    here->MOS3bNode,here->MOS3senParmNo);
            printf("\n without  perturbation \n");
#endif /* SENSDEBUG */
            /*  without  perturbation  */
            *(ckt->CKTstate0 + here->MOS3vbs) = vbsOp;
            *(ckt->CKTstate0 + here->MOS3vbd) = vbdOp;

            here->MOS3senPertFlag = ON ;
            if(info->SENacpertflag == 1){
                /* store the  unperturbed values of small signal parameters */
                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->MOS3senCgs) = here->MOS3cgs;
                *(here->MOS3senCgd) = here->MOS3cgd;
                *(here->MOS3senCgb) = here->MOS3cgb;
                *(here->MOS3senCbd) = here->MOS3capbd;
                *(here->MOS3senCbs) = here->MOS3capbs;
                *(here->MOS3senGds) = here->MOS3gds;
                *(here->MOS3senGbs) = here->MOS3gbs;
                *(here->MOS3senGbd) = here->MOS3gbd;
                *(here->MOS3senGm) = here->MOS3gm;
                *(here->MOS3senGmbs) = here->MOS3gmbs;

            }
            xcgs0= *(here->MOS3senCgs) * ckt->CKTomega;
            xcgd0= *(here->MOS3senCgd) * ckt->CKTomega;
            xcgb0= *(here->MOS3senCgb) * ckt->CKTomega;
            xbd0= *(here->MOS3senCbd) * ckt->CKTomega;
            xbs0= *(here->MOS3senCbs) * ckt->CKTomega;
            gds0= *(here->MOS3senGds);
            gbs0= *(here->MOS3senGbs);
            gbd0= *(here->MOS3senGbd);
            gm0= *(here->MOS3senGm);
            gmbs0= *(here->MOS3senGmbs);
            gdpr0 = here->MOS3drainConductance;
            gspr0 = here->MOS3sourceConductance;


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
            printf("xcgs0 = %.7e , xcgd0 = %.7e ,", xcgs0,xcgd0);
            printf("xcgb0 = %.7e, xbd0 = %.7e,xbs0 = %.7e\n" ,xcgb0,xbd0,xbs0);
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
                *(ckt->CKTstate0 + here->MOS3vbs) = Apert;
                *(ckt->CKTstate0 + here->MOS3vbd) = vbdOp;

                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS3senCgs + 1) = here->MOS3cgs;
                *(here->MOS3senCgd + 1) = here->MOS3cgd;
                *(here->MOS3senCgb + 1) = here->MOS3cgb;
                *(here->MOS3senCbd + 1) = here->MOS3capbd;
                *(here->MOS3senCbs + 1) = here->MOS3capbs;
                *(here->MOS3senGds + 1) = here->MOS3gds;
                *(here->MOS3senGbs + 1) = here->MOS3gbs;
                *(here->MOS3senGbd + 1) = here->MOS3gbd;
                *(here->MOS3senGm + 1) = here->MOS3gm;
                *(here->MOS3senGmbs + 1) = here->MOS3gmbs;

                *(ckt->CKTstate0 + here->MOS3vbs) = A0;


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
                *(ckt->CKTstate0 + here->MOS3vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS3vbd) = Apert;

                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS3senCgs + 2) = here->MOS3cgs;
                *(here->MOS3senCgd + 2) = here->MOS3cgd;
                *(here->MOS3senCgb + 2) = here->MOS3cgb;
                *(here->MOS3senCbd + 2) = here->MOS3capbd;
                *(here->MOS3senCbs + 2) = here->MOS3capbs;
                *(here->MOS3senGds + 2) = here->MOS3gds;
                *(here->MOS3senGbs + 2) = here->MOS3gbs;
                *(here->MOS3senGbd + 2) = here->MOS3gbd;
                *(here->MOS3senGm + 2) = here->MOS3gm;
                *(here->MOS3senGmbs + 2) = here->MOS3gmbs;

                *(ckt->CKTstate0 + here->MOS3vbd) = A0;

            }

            goto load;


pertvgb:  /* Perturbation of vgb */
#ifdef SENSDEBUG
            printf("\nPerturbation of vgb\n");
#endif /* SENSDEBUG */

            flag = 3;
            A0 = model->MOS3type * (*(ckt->CKTrhsOp + here->MOS3gNode) 
                -  *(ckt->CKTrhsOp + here->MOS3bNode)); 
            DELA =  info->SENpertfac * A0 + 1e-8;
            DELAinv = model->MOS3type * 1.0/DELA;


            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed vgb */
                *(ckt->CKTstate0 + here->MOS3vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS3vbd) = vbdOp;
                *(ckt->CKTrhsOp + here->MOS3bNode) -= DELA; 

                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS3senCgs + 3) = here->MOS3cgs;
                *(here->MOS3senCgd + 3) = here->MOS3cgd;
                *(here->MOS3senCgb + 3) = here->MOS3cgb;
                *(here->MOS3senCbd + 3) = here->MOS3capbd;
                *(here->MOS3senCbs + 3) = here->MOS3capbs;
                *(here->MOS3senGds + 3) = here->MOS3gds;
                *(here->MOS3senGbs + 3) = here->MOS3gbs;
                *(here->MOS3senGbd + 3) = here->MOS3gbd;
                *(here->MOS3senGm + 3) = here->MOS3gm;
                *(here->MOS3senGmbs + 3) = here->MOS3gmbs;


                *(ckt->CKTrhsOp + here->MOS3bNode) += DELA; 
            }
            goto load;

pertl:    /* Perturbation of length */

            if(here->MOS3sens_l == 0){
                goto pertw;
            }
#ifdef SENSDEBUG
            printf("\nPerturbation of length\n");
#endif /* SENSDEBUG */
            flag = 4;
            A0 = here->MOS3l;
            DELA =  info->SENpertfac * A0;
            DELAinv = 1.0/DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed length */
                Apert = A0 + DELA;
                here->MOS3l = Apert;

                *(ckt->CKTstate0 + here->MOS3vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS3vbd) = vbdOp;

                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS3senCgs + 4) = here->MOS3cgs;
                *(here->MOS3senCgd + 4) = here->MOS3cgd;
                *(here->MOS3senCgb + 4) = here->MOS3cgb;
                *(here->MOS3senCbd + 4) = here->MOS3capbd;
                *(here->MOS3senCbs + 4) = here->MOS3capbs;
                *(here->MOS3senGds + 4) = here->MOS3gds;
                *(here->MOS3senGbs + 4) = here->MOS3gbs;
                *(here->MOS3senGbd + 4) = here->MOS3gbd;
                *(here->MOS3senGm + 4) = here->MOS3gm;
                *(here->MOS3senGmbs + 4) = here->MOS3gmbs;

                here->MOS3l = A0;

            }

            goto load;

pertw:    /* Perturbation of width */
            if(here->MOS3sens_w == 0)
                goto next;
#ifdef SENSDEBUG
            printf("\nPerturbation of width\n");
#endif /* SENSDEBUG */
            flag = 5;
            A0 = here->MOS3w;
            DELA = info->SENpertfac * A0;
            DELAinv = 1.0/DELA;
            Apert = A0 + DELA;

            if(info->SENacpertflag == 1){
                /* store the  values of small signal parameters 
                 * corresponding to perturbed width */
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
                if(vbdOp >= here->MOS3tDepCap){
                    arg = 1-model->MOS3fwdCapDepCoeff;
                    sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * 
                            log(arg) );
                    sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->MOS3f2d = here->MOS3Cbd*(1-model->MOS3fwdCapDepCoeff*
                            (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                            +  here->MOS3Cbdsw*(1-model->MOS3fwdCapDepCoeff*
                            (1+model->MOS3bulkJctSideGradingCoeff))*
                            sargsw/arg;
                    here->MOS3f3d = here->MOS3Cbd * 
                            model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                            model->MOS3bulkJctPotential
                            + here->MOS3Cbdsw * 
                            model->MOS3bulkJctSideGradingCoeff * sargsw/arg /
                            model->MOS3bulkJctPotential;
                    here->MOS3f4d = here->MOS3Cbd*model->MOS3bulkJctPotential*
                            (1-arg*sarg)/ (1-model->MOS3bulkJctBotGradingCoeff)
                            + here->MOS3Cbdsw*model->MOS3bulkJctPotential*
                            (1-arg*sargsw)/
                            (1-model->MOS3bulkJctSideGradingCoeff)
                            -here->MOS3f3d/2*
                            (here->MOS3tDepCap*here->MOS3tDepCap)
                            -here->MOS3tDepCap * here->MOS3f2d;
                }
                if(vbsOp >= here->MOS3tDepCap){
                    arg = 1-model->MOS3fwdCapDepCoeff;
                    sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * 
                            log(arg) );
                    sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * 
                            log(arg) );
                    here->MOS3f2s = here->MOS3Cbs*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                        +  here->MOS3Cbssw*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctSideGradingCoeff))*
                        sargsw/arg;
                    here->MOS3f3s = here->MOS3Cbs * 
                            model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                            model->MOS3bulkJctPotential + here->MOS3Cbssw * 
                            model->MOS3bulkJctSideGradingCoeff * sargsw/arg /
                            model->MOS3bulkJctPotential;
                    here->MOS3f4s = here->MOS3Cbs*model->MOS3bulkJctPotential*
                            (1-arg*sarg)/ (1-model->MOS3bulkJctBotGradingCoeff)
                            + here->MOS3Cbssw*model->MOS3bulkJctPotential*
                            (1-arg*sargsw)/
                            (1-model->MOS3bulkJctSideGradingCoeff)
                            -here->MOS3f3s/2*
                            (here->MOS3tBulkPot*here->MOS3tBulkPot)
                            -here->MOS3tBulkPot * here->MOS3f2s;
                }

                *(ckt->CKTstate0 + here->MOS3vbs) = vbsOp;
                *(ckt->CKTstate0 + here->MOS3vbd) = vbdOp;

                if((error = MOS3load((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->MOS3senCgs + 5) = here->MOS3cgs;
                *(here->MOS3senCgd + 5) = here->MOS3cgd;
                *(here->MOS3senCgb + 5) = here->MOS3cgb;
                *(here->MOS3senCbd + 5) = here->MOS3capbd;
                *(here->MOS3senCbs + 5) = here->MOS3capbs;
                *(here->MOS3senGds + 5) = here->MOS3gds;
                *(here->MOS3senGbs + 5) = here->MOS3gbs;
                *(here->MOS3senGbd + 5) = here->MOS3gbd;
                *(here->MOS3senGm + 5) = here->MOS3gm;
                *(here->MOS3senGmbs + 5) = here->MOS3gmbs;

                here->MOS3w = A0;
                here->MOS3drainArea /= (1 + info->SENpertfac);
                here->MOS3sourceArea /= (1 + info->SENpertfac);
            }

load:

            gds= *(here->MOS3senGds + flag);
            gbs= *(here->MOS3senGbs + flag);
            gbd= *(here->MOS3senGbd + flag);
            gm= *(here->MOS3senGm + flag);
            gmbs= *(here->MOS3senGmbs + flag);
            if(flag == 5){
                gdpr = here->MOS3drainConductance * Apert/A0;
                gspr = here->MOS3sourceConductance * Apert/A0;
            }
            else{
                gdpr = here->MOS3drainConductance;
                gspr = here->MOS3sourceConductance;
            }

            xcgs= *(here->MOS3senCgs + flag) * ckt->CKTomega;
            xcgd= *(here->MOS3senCgd + flag) * ckt->CKTomega;
            xcgb= *(here->MOS3senCgb + flag) * ckt->CKTomega;
            xbd= *(here->MOS3senCbd + flag) * ckt->CKTomega;
            xbs= *(here->MOS3senCbs + flag) * ckt->CKTomega;

#ifdef SENSDEBUG
            printf("flag = %d \n",flag);
            printf("gspr = %.7e , gdpr = %.7e , gds = %.7e, gbs = %.7e\n",
                    gspr,gdpr,gds,gbs);
            printf("gbd = %.7e , gm = %.7e , gmbs = %.7e\n",gbd,gm,gmbs);
            printf("xcgs = %.7e , xcgd = %.7e , xcgb = %.7e,", xcgs,xcgd,xcgb);
            printf("xbd = %.7e,xbs = %.7e\n" ,xbd,xbs);
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
                if( (flag == 4) && (iparmno != here->MOS3senParmNo) ) continue;
                if( (flag == 5) && (iparmno != (here->MOS3senParmNo +
                        (int) here->MOS3sens_l)) ) continue;

                switch(flag){
                case 1: 
                    DvDp = model->MOS3type * 
                            (info->SEN_Sap[here->MOS3bNode][iparmno]
                            -  info->SEN_Sap[here->MOS3sNodePrime][iparmno]);
                    break;
                case 2: 
                    DvDp = model->MOS3type * 
                            ( info->SEN_Sap[here->MOS3bNode][iparmno]
                            -  info->SEN_Sap[here->MOS3dNodePrime][iparmno]);
                    break;
                case 3: 
                    DvDp = model->MOS3type * 
                            ( info->SEN_Sap[here->MOS3gNode][iparmno]
                            -  info->SEN_Sap[here->MOS3bNode][iparmno]);
                    break;
                case 4: 
                    DvDp = 1;
                    break;
                case 5: 
                    DvDp = 1;
                    break;
                }
                *(info->SEN_RHS[here->MOS3bNode] + iparmno) -=  
                        ( cb  - cb0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS3bNode] + iparmno) -=  
                        ( icb  - icb0) * DELAinv * DvDp;

                *(info->SEN_RHS[here->MOS3gNode] + iparmno) -=  
                        ( cg  - cg0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS3gNode] + iparmno) -=  
                        ( icg  - icg0) * DELAinv * DvDp;

                if(here->MOS3sNode != here->MOS3sNodePrime){
                    *(info->SEN_RHS[here->MOS3sNode] + iparmno) -=  
                            ( cs  - cs0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS3sNode] + iparmno) -=  
                            ( ics  - ics0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS3sNodePrime] + iparmno) -=  
                        ( csprm  - csprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS3sNodePrime] + iparmno) -=  
                        ( icsprm  - icsprm0) * DELAinv * DvDp;

                if(here->MOS3dNode != here->MOS3dNodePrime){
                    *(info->SEN_RHS[here->MOS3dNode] + iparmno) -=  
                            ( cd  - cd0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->MOS3dNode] + iparmno) -=  
                            ( icd  - icd0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->MOS3dNodePrime] + iparmno) -=  
                        ( cdprm  - cdprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->MOS3dNodePrime] + iparmno) -=  
                        ( icdprm  - icdprm0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");  
                printf("DvDp = %.5e , DELAinv = %.5e ,flag = %d ,",
                        DvDp,DELAinv,flag);
                printf("iparmno = %d,senparmno = %d\n",
                        iparmno,here->MOS3senParmNo);
                printf("A0 = %.5e , Apert = %.5e ,CONSTvt0 = %.5e \n",
                        A0,Apert,CONSTvt0);
                printf("senb = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3bNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS3bNode] + iparmno));
                printf("seng = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3gNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS3gNode] + iparmno));
                printf("sens = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3sNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS3sNode] + iparmno));
                printf("sensprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3sNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS3sNodePrime] + iparmno));
                printf("send = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3dNode] + iparmno),
                        *(info->SEN_iRHS[here->MOS3dNode] + iparmno));
                printf("sendprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->MOS3dNodePrime] + iparmno),
                        *(info->SEN_iRHS[here->MOS3dNodePrime] + iparmno));
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
    printf("MOS3senacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}


