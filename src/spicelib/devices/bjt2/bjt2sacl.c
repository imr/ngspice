/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


int
BJT2sAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{

    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
    double SaveState[25];
    int    error;
    int    flag;
    double vbeOp;
    double vbcOp;
    double A0;
    double DELA = 0.0;
    double Apert;
    double DELAinv;
    double vte = 0.0;
    double gcpr;
    double gepr;
    double gpi;
    double gmu;
    double go;
    double xgm;
    double td;
    double arg;
    double gm;
    double gx;
    double xcpi;
    double xcmu;
    double xcbx;
    double xccs;
    double xcmcb;
    double cx,icx;
    double cbx,icbx;
    double ccs,iccs;
    double cbc,icbc;
    double cbe,icbe;
    double cce,icce;
    double cb,icb;
    double cbprm,icbprm;
    double cc,icc;
    double ccprm,iccprm;
    double ce,ice;
    double ceprm,iceprm;
    double cs,ics;
    double vcpr,ivcpr;
    double vepr,ivepr;
    double vx,ivx;
    double vbx,ivbx;
    double vcs,ivcs;
    double vbc,ivbc;
    double vbe,ivbe;
    double vce,ivce;
    double cb0,icb0;
    double cbprm0,icbprm0;
    double cc0,icc0;
    double ccprm0,iccprm0;
    double ce0,ice0;
    double ceprm0,iceprm0;
    double cs0,ics0;
    double DvDp = 0.0;
    int iparmno,i;
    SENstruct *info;


#ifdef SENSDEBUG
    printf("BJT2senacload \n");
    printf("BJT2senacload \n");
#endif /* SENSDEBUG */

    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;

    /*  loop through all the models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 20; i++) {
                *(SaveState + i) = *(ckt->CKTstate0 + here->BJT2state + i);
            }

            vcpr = *(ckt->CKTrhsOld + here->BJT2colNode) 
                - *(ckt->CKTrhsOld + here->BJT2colPrimeNode) ;
            ivcpr = *(ckt->CKTirhsOld + here->BJT2colNode) 
                - *(ckt->CKTirhsOld + here->BJT2colPrimeNode) ;
            vepr = *(ckt->CKTrhsOld + here->BJT2emitNode) 
                - *(ckt->CKTrhsOld + here->BJT2emitPrimeNode) ;
            ivepr = *(ckt->CKTirhsOld + here->BJT2emitNode) 
                - *(ckt->CKTirhsOld + here->BJT2emitPrimeNode) ;
            vx = *(ckt->CKTrhsOld + here->BJT2baseNode) 
                - *(ckt->CKTrhsOld + here->BJT2basePrimeNode) ;/* vb_bprm */
            ivx = *(ckt->CKTirhsOld + here->BJT2baseNode)
                - *(ckt->CKTirhsOld + here->BJT2basePrimeNode) ;/* ivb_bprm */
            vcs = *(ckt->CKTrhsOld + here->BJT2colPrimeNode) 
                - *(ckt->CKTrhsOld + here->BJT2substNode) ;
            ivcs = *(ckt->CKTirhsOld + here->BJT2colPrimeNode) 
                - *(ckt->CKTirhsOld + here->BJT2substNode) ;
            vbc = *(ckt->CKTrhsOld + here->BJT2basePrimeNode) 
                - *(ckt->CKTrhsOld + here->BJT2colPrimeNode) ;/* vbprm_cprm */
            ivbc = *(ckt->CKTirhsOld + here->BJT2basePrimeNode)
                - *(ckt->CKTirhsOld + here->BJT2colPrimeNode) ;/* ivbprm_cprm */
            vbe = *(ckt->CKTrhsOld + here->BJT2basePrimeNode) 
                - *(ckt->CKTrhsOld + here->BJT2emitPrimeNode) ;/* vbprm_eprm */
            ivbe = *(ckt->CKTirhsOld + here->BJT2basePrimeNode)
                - *(ckt->CKTirhsOld + here->BJT2emitPrimeNode) ;/* ivbprm_eprm */
            vce = vbe - vbc ;
            ivce = ivbe - ivbc ;
            vbx = vx + vbc ;
            ivbx = ivx + ivbc ;



            vbeOp =model->BJT2type * ( *(ckt->CKTrhsOp +  here->BJT2basePrimeNode)
                -  *(ckt->CKTrhsOp +  here->BJT2emitPrimeNode));
            vbcOp =model->BJT2type * ( *(ckt->CKTrhsOp +  here->BJT2basePrimeNode)
                -  *(ckt->CKTrhsOp +  here->BJT2colPrimeNode));

#ifdef SENSDEBUG
            printf("\n without perturbation\n");
#endif /* SENSDEBUG */
            /* without perturbation */
            A0 = here->BJT2area;
            here->BJT2senPertFlag = ON;
            *(ckt->CKTstate0 + here->BJT2vbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbcOp;
            /* info->SENacpertflag == 1 only for first frequency */

            if(info->SENacpertflag == 1){

                /* store the  unperturbed values of small signal parameters */

                if ((error = BJT2load((GENmodel*)model,ckt))) return(error);

                *(here->BJT2senGpi)= *(ckt->CKTstate0 + here->BJT2gpi);
                *(here->BJT2senGmu)= *(ckt->CKTstate0 + here->BJT2gmu);
                *(here->BJT2senGm)= *(ckt->CKTstate0 + here->BJT2gm);
                *(here->BJT2senGo)= *(ckt->CKTstate0 + here->BJT2go);
                *(here->BJT2senGx)= *(ckt->CKTstate0 + here->BJT2gx);
                *(here->BJT2senCpi)= *(ckt->CKTstate0 + here->BJT2cqbe);
                *(here->BJT2senCmu)= *(ckt->CKTstate0 + here->BJT2cqbc);
                *(here->BJT2senCbx)= *(ckt->CKTstate0 + here->BJT2cqbx);
                *(here->BJT2senCsub)= *(ckt->CKTstate0 + here->BJT2cqsub);
                *(here->BJT2senCmcb)= *(ckt->CKTstate0 + here->BJT2cexbc);
            }
            gcpr = here->BJT2tCollectorConduct * A0;
            gepr = here->BJT2tEmitterConduct * A0;
            gpi= *(here->BJT2senGpi);
            gmu= *(here->BJT2senGmu);
            gm= *(here->BJT2senGm);
            go= *(here->BJT2senGo);
            gx= *(here->BJT2senGx);
            xgm=0;
            td=model->BJT2excessPhase;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            xcpi= *(here->BJT2senCpi) * ckt->CKTomega;
            xcmu= *(here->BJT2senCmu) * ckt->CKTomega;
            xcbx= *(here->BJT2senCbx) * ckt->CKTomega;
            xccs= *(here->BJT2senCsub) * ckt->CKTomega;
            xcmcb= *(here->BJT2senCmcb) * ckt->CKTomega;


            cx=gx * vx ;
            icx=gx * ivx;
            cbx=( -xcbx * ivbx) ;
            icbx= xcbx * vbx ;
            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cbc=(gmu * vbc -xcmu * ivbc) ;
            icbc=xcmu * vbc +  gmu * ivbc ;
            cbe=gpi * vbe -xcpi * ivbe - xcmcb * ivbc ;
            icbe=xcpi * vbe +  gpi * ivbe + xcmcb * vbc;
            cce= go * vce + gm * vbe - xgm * ivbe;
            icce=go * ivce + gm * ivbe + xgm * vbe ;

            cc0=gcpr * vcpr ;
            icc0=gcpr * ivcpr ;
            ce0=gepr * vepr;
            ice0=gepr * ivepr ;
            cb0 = cx + cbx;
            icb0 = icx + icbx;
            if(here->BJT2baseNode != here->BJT2basePrimeNode){
                cbprm0 = (- cx + cbe + cbc);
                icbprm0 = (- icx + icbe + icbc);
            }
            else{
                cbprm0 = ( cbx + cbe + cbc);
                icbprm0 = (icbx + icbe + icbc);
            }
            ccprm0 = (- cbx - cc0 + ccs + cce - cbc);
            iccprm0 = (- icbx - icc0 + iccs + icce - icbc);
            ceprm0 = (- cbe - cce - ce0);
            iceprm0 = (- icbe - icce - ice0);
            cs0 = (- ccs) ;
            ics0 = (- iccs) ;

#ifdef SENSDEBUG
            printf("gepr0 = %.7e , gcpr0 = %.7e , gmu0 = %.7e, gpi0 = %.7e\n",
                    gepr,gcpr,gmu,gpi);
            printf("gm0 = %.7e , go0 = %.7e , gx0 = %.7e, xcpi0 = %.7e\n",
                    gm,go,gx,xcpi);
            printf("xcmu0 = %.7e , xcbx0 = %.7e , xccs0 = %.7e, xcmcb0 = %.7e\n"
                    ,xcmu,xcbx,xccs,xcmcb);
            printf("vepr = %.7e + j%.7e , vcpr = %.7e + j%.7e\n",
                    vepr,ivepr,vcpr,ivcpr);
            printf("vbx = %.7e + j%.7e , vx = %.7e + j%.7e\n",
                    vbx,ivbx,vx,ivx);
            printf("vbc = %.7e + j%.7e , vbe = %.7e + j%.7e\n",
                    vbc,ivbc,vbe,ivbe);
            printf("vce = %.7e + j%.7e , vcs = %.7e + j%.7e\n",
                    vce,ivce,vcs,ivcs);
            printf("cce0 = %.7e + j%.7e , cbe0 = %.7e + j%.7e\n",
                    cce,icce,cbe,icbe);
            printf("cbc0 = %.7e + j%.7e\n",
                    cbc,icbc);
            printf("cc0 = %.7e + j%.7e , ce0 = %.7e + j%.7e\n",
                    cc0,icc0,ce0,ice0);
            printf("cb0 = %.7e + j%.7e , cs0 = %.7e + j%.7e\n",
                    cb0,icb0,cs0,ics0);
            printf("cbprm0 = %.7e + j%.7e , ceprm0 = %.7e + j%.7e\n",
                    cbprm0,icbprm0,ceprm0,iceprm0);
            printf("ccprm0 = %.7e + j%.7e \n",
                    ccprm0,iccprm0);
            printf("\nPerturbation of Area\n");
#endif /* SENSDEBUG */
            /* Perturbation of Area */
            if(here->BJT2senParmNo == 0){
                flag = 0;
                goto next1;
            }

            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            here->BJT2area = Apert;

            *(ckt->CKTstate0 + here->BJT2vbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbcOp;
            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed area 
                 */
                if ((error = BJT2load((GENmodel*)model,ckt))) return(error);

                *(here->BJT2senGpi + 1)= *(ckt->CKTstate0 + here->BJT2gpi);
                *(here->BJT2senGmu + 1)= *(ckt->CKTstate0 + here->BJT2gmu);
                *(here->BJT2senGm + 1)= *(ckt->CKTstate0 + here->BJT2gm);
                *(here->BJT2senGo + 1)= *(ckt->CKTstate0 + here->BJT2go);
                *(here->BJT2senGx + 1)= *(ckt->CKTstate0 + here->BJT2gx);
                *(here->BJT2senCpi + 1)= *(ckt->CKTstate0 + here->BJT2cqbe);
                *(here->BJT2senCmu + 1)= *(ckt->CKTstate0 + here->BJT2cqbc);
                *(here->BJT2senCbx + 1)= *(ckt->CKTstate0 + here->BJT2cqbx);
                *(here->BJT2senCsub + 1)= *(ckt->CKTstate0 + here->BJT2cqsub);
                *(here->BJT2senCmcb + 1)= *(ckt->CKTstate0 + here->BJT2cexbc);
            }


            flag = 0;
            goto load;



pertvbx:    /* Perturbation of vbx */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbx\n");
#endif /* SENSDEBUG */
            here->BJT2area = A0;
            A0 = model->BJT2type * (*(ckt->CKTrhsOp + here->BJT2baseNode)
                - *(ckt->CKTrhsOp + here->BJT2colPrimeNode));
            DELA = info->SENpertfac * A0 + 1e-8; 
            Apert = A0 + DELA;
            DELAinv = model->BJT2type * 1.0/DELA;
            *(ckt->CKTrhsOp + here->BJT2baseNode) += DELA;
            *(ckt->CKTstate0 + here->BJT2vbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbx 
                 */
                if((error = BJT2load((GENmodel*)model,ckt))) return(error);

                *(here->BJT2senGpi + 2)= *(ckt->CKTstate0 + here->BJT2gpi);
                *(here->BJT2senGmu + 2)= *(ckt->CKTstate0 + here->BJT2gmu);
                *(here->BJT2senGm + 2)= *(ckt->CKTstate0 + here->BJT2gm);
                *(here->BJT2senGo + 2)= *(ckt->CKTstate0 + here->BJT2go);
                *(here->BJT2senGx + 2)= *(ckt->CKTstate0 + here->BJT2gx);
                *(here->BJT2senCpi + 2)= *(ckt->CKTstate0 + here->BJT2cqbe);
                *(here->BJT2senCmu + 2)= *(ckt->CKTstate0 + here->BJT2cqbc);
                *(here->BJT2senCbx + 2)= *(ckt->CKTstate0 + here->BJT2cqbx);
                *(here->BJT2senCsub + 2)= *(ckt->CKTstate0 + here->BJT2cqsub);
                *(here->BJT2senCmcb + 2)= *(ckt->CKTstate0 + here->BJT2cexbc);
            }


            flag = 1;
            goto load;


pertvbe:    /* Perturbation of vbe */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbe\n");
#endif /* SENSDEBUG */
            if (*(here->BJT2senCbx) != 0){ 
                *(ckt->CKTrhsOp + here ->BJT2baseNode) -= DELA;
            }
            vte=model->BJT2leakBEemissionCoeff*CONSTvt0;
            A0 = vbeOp;
            DELA = info->SENpertfac * vte ;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            *(ckt->CKTstate0 + here->BJT2vbe) = Apert;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbe 
                 */
                if((error = BJT2load((GENmodel*)model,ckt))) return(error);

                *(here->BJT2senGpi + 3)= *(ckt->CKTstate0 + here->BJT2gpi);
                *(here->BJT2senGmu + 3)= *(ckt->CKTstate0 + here->BJT2gmu);
                *(here->BJT2senGm + 3)= *(ckt->CKTstate0 + here->BJT2gm);
                *(here->BJT2senGo + 3)= *(ckt->CKTstate0 + here->BJT2go);
                *(here->BJT2senGx + 3)= *(ckt->CKTstate0 + here->BJT2gx);
                *(here->BJT2senCpi + 3)= *(ckt->CKTstate0 + here->BJT2cqbe);
                *(here->BJT2senCmu + 3)= *(ckt->CKTstate0 + here->BJT2cqbc);
                *(here->BJT2senCbx + 3)= *(ckt->CKTstate0 + here->BJT2cqbx);
                *(here->BJT2senCsub + 3)= *(ckt->CKTstate0 + here->BJT2cqsub);
                *(here->BJT2senCmcb + 3)= *(ckt->CKTstate0 + here->BJT2cexbc);
            }


            flag = 2;
            goto load;


pertvbc:    /* Perturbation of vbc */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbc\n");
#endif /* SENSDEBUG */
            *(ckt->CKTstate0 + here->BJT2vbe) = A0;

            A0 = vbcOp;
            DELA = info->SENpertfac * vte ;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;


            *(ckt->CKTstate0 + here->BJT2vbc) = Apert;

            *(ckt->CKTstate0 + here->BJT2vbe) = vbeOp;



            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbc 
                 */
                if((error = BJT2load((GENmodel*)model,ckt))) return(error);
                *(here->BJT2senGpi + 4)= *(ckt->CKTstate0 + here->BJT2gpi);
                *(here->BJT2senGmu + 4)= *(ckt->CKTstate0 + here->BJT2gmu);
                *(here->BJT2senGm + 4)= *(ckt->CKTstate0 + here->BJT2gm);
                *(here->BJT2senGo + 4)= *(ckt->CKTstate0 + here->BJT2go);
                *(here->BJT2senGx + 4)= *(ckt->CKTstate0 + here->BJT2gx);
                *(here->BJT2senCpi + 4)= *(ckt->CKTstate0 + here->BJT2cqbe);
                *(here->BJT2senCmu + 4)= *(ckt->CKTstate0 + here->BJT2cqbc);
                *(here->BJT2senCbx + 4)= *(ckt->CKTstate0 + here->BJT2cqbx);
                *(here->BJT2senCsub + 4)= *(ckt->CKTstate0 + here->BJT2cqsub);
                *(here->BJT2senCmcb + 4)= *(ckt->CKTstate0 + here->BJT2cexbc);

            }


            flag = 3;
            goto load;



pertvcs:    /* Perturbation of vcs */
#ifdef SENSDEBUG
            printf("\nPerturbation of vcs\n");
#endif /* SENSDEBUG */
            *(ckt->CKTstate0 + here->BJT2vbc) = A0;
            A0 = model->BJT2type * (*(ckt->CKTrhsOp + here->BJT2substNode)
                -  *(ckt->CKTrhsOp + here->BJT2colPrimeNode));
            DELA = info->SENpertfac * A0 + 1e-8; 
            Apert = A0 + DELA;
            DELAinv = model->BJT2type * 1.0/DELA;
            *(ckt->CKTrhsOp + here->BJT2substNode) += DELA;
            *(ckt->CKTstate0 + here->BJT2vbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vcs 
                 */
                if((error = BJT2load((GENmodel*)model,ckt))) return(error);
                *(here->BJT2senCsub + 5)= *(ckt->CKTstate0 + here->BJT2cqsub);

            }


            flag = 4;

            *(ckt->CKTrhsOp + here->BJT2substNode) -= DELA;
            xccs= *(here->BJT2senCsub + 5) * ckt->CKTomega;

            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cs = -ccs;
            ics = -iccs;
            ccprm = ccprm0 +  cs0 - cs;
            iccprm = iccprm0 + ics0 - ics;
            cbprm = cbprm0;
            icbprm = icbprm0;
            ceprm = ceprm0;
            iceprm = iceprm0;
            cc = cc0;
            icc = icc0;
            ce = ce0;
            ice = ice0;
            cb = cb0;
            icb = icb0;
            goto next2;

load:
            gcpr=here->BJT2tCollectorConduct * here->BJT2area;
            gepr=here->BJT2tEmitterConduct * here->BJT2area;
            gpi= *(here->BJT2senGpi + flag+1);
            gmu= *(here->BJT2senGmu + flag+1);
            gm= *(here->BJT2senGm + flag+1);
            go= *(here->BJT2senGo + flag+1);
            gx= *(here->BJT2senGx + flag+1);
            xgm=0;
            td=model->BJT2excessPhase;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            xcpi= *(here->BJT2senCpi + flag+1) * ckt->CKTomega;
            xcmu= *(here->BJT2senCmu + flag+1) * ckt->CKTomega;
            xcbx= *(here->BJT2senCbx + flag+1) * ckt->CKTomega;
            xccs= *(here->BJT2senCsub + flag+1) * ckt->CKTomega;
            xcmcb= *(here->BJT2senCmcb + flag+1) * ckt->CKTomega;


            cc=gcpr * vcpr ;
            icc=gcpr * ivcpr ;
            ce=gepr * vepr;
            ice=gepr * ivepr ;
            cx=gx * vx ;
            icx=gx * ivx;
            cbx=( -xcbx * ivbx) ;
            icbx= xcbx * vbx ;
            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cbc=(gmu * vbc -xcmu * ivbc) ;
            icbc=xcmu * vbc +  gmu * ivbc ;
            cbe=gpi * vbe -xcpi * ivbe - xcmcb * ivbc ;
            icbe=xcpi * vbe +  gpi * ivbe + xcmcb * vbc;
            cce= go * vce + gm * vbe - xgm * ivbe;
            icce=go * ivce + gm * ivbe + xgm * vbe ;


            cb= cx + cbx;
            icb= icx + icbx;
            if(here->BJT2baseNode != here->BJT2basePrimeNode){
                cbprm=(- cx + cbe + cbc);
                icbprm=(- icx + icbe + icbc);
            }
            else{
                cbprm=( cbx + cbe + cbc);
                icbprm=(icbx + icbe + icbc);
            }
            ccprm=(- cbx - cc + ccs + cce - cbc);
            iccprm=(- icbx - icc + iccs + icce - icbc);
            ceprm=(- cbe - cce - ce);
            iceprm=(- icbe - icce - ice);
            cs= (- ccs) ;
            ics= (- iccs) ;

#ifdef SENSDEBUG
            printf("A0 = %.7e , Apert = %.7e , DELA = %.7e\n"
                    ,A0,Apert,DELA);
            printf("gepr = %.7e , gcpr = %.7e , gmu = %.7e, gpi = %.7e\n"
                    ,gepr,gcpr,gmu,gpi);
            printf("gm = %.7e , go = %.7e , gx = %.7e, xcpi = %.7e\n"
                    ,gm,go,gx,xcpi);
            printf("xcmu = %.7e , xcbx = %.7e , xccs = %.7e, xcmcb = %.7e\n"
                    ,xcmu,xcbx,xccs,xcmcb);

            printf("cx = %.7e + j%.7e , cbx = %.7e + j%.7e\n"
                    ,cx,icx,cbx,icbx);
            printf("ccs %.7e + j%.7e , cbc = %.7e + j%.7e"
                    ,ccs,iccs,cbc,icbc);
            printf("cbe %.7e + j%.7e , cce = %.7e + j%.7e\n"
                    ,cbe,icbe,cce,icce);

            printf("cc = %.7e + j%.7e , ce = %.7e + j%.7e,",
                    ,cc,icc,ce,ice);
            printf("ccprm = %.7e + j%.7e , ceprm = %.7e + j%.7e",
                    ccprm,iccprm,ceprm,iceprm);
            printf("cb = %.7e + j%.7e , cbprm = %.7e + j%.7e , ",
                    cb,icb,cbprm,icbprm)
            printf("cs = %.7e + j%.7e\n",
                    cs,ics);
#endif /* SENSDEBUG */


            /* load the RHS matrix */
next2:
            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                if( (!flag) && (iparmno != here->BJT2senParmNo) ) continue;
                switch(flag){

                case 0: 
                    /* area : so no DC sensitivity term involved */
                    DvDp = 1.0; 

                    break;
                    /* calculate the DC sensitivities of operating points */
                case 1: 
                    DvDp = model->BJT2type * 
                            (info->SEN_Sap[here->BJT2baseNode][iparmno]
                            -  info->SEN_Sap[here->BJT2colPrimeNode][iparmno]);
                    break;
                case 2: 
                    DvDp = model->BJT2type * 
                            (info->SEN_Sap[here->BJT2basePrimeNode][iparmno]
                            -  info->SEN_Sap[here->BJT2emitPrimeNode][iparmno]);
                    break;
                case 3: 
                    DvDp = model->BJT2type * 
                            (info->SEN_Sap[here->BJT2basePrimeNode][iparmno]
                            -  info->SEN_Sap[here->BJT2colPrimeNode][iparmno]);
                    break;
                case 4: 
                    DvDp = model->BJT2type * 
                            (info->SEN_Sap[here->BJT2substNode][iparmno]
                            -  info->SEN_Sap[here->BJT2colPrimeNode][iparmno]);
                    break;
                }
#ifdef SENSDEBUG
                printf("before loading\n");  
                printf("BJT2type = %d\n",model->BJT2type);  
                printf("DvDp = %.7e , flag = %d , iparmno = %d,senparmno = %d\n"
                        ,DvDp,flag,iparmno,here->BJT2senParmNo);
                printf("senb = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2baseNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2baseNode] + iparmno));
                printf("senbrm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2basePrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2basePrimeNode] + iparmno));
                printf("senc = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2colNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2colNode] + iparmno));
                printf("sencprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2colPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2colPrimeNode] + iparmno));
                printf("sene = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2emitNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2emitNode] + iparmno));
                printf("seneprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2emitPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2emitPrimeNode] + iparmno));
                printf("sens = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2substNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2substNode] + iparmno));
#endif /* SENSDEBUG */


                if(here->BJT2baseNode != here->BJT2basePrimeNode){
                    *(info->SEN_RHS[here->BJT2baseNode] + iparmno) -=  
                            ( cb  - cb0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJT2baseNode] + iparmno) -=  
                            ( icb  - icb0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJT2basePrimeNode] + iparmno) -=  
                        ( cbprm  - cbprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJT2basePrimeNode] + iparmno) -=  
                        ( icbprm  - icbprm0) * DELAinv * DvDp;

                if(here->BJT2colNode != here->BJT2colPrimeNode){
                    *(info->SEN_RHS[here->BJT2colNode] + iparmno) -=  
                            ( cc  - cc0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJT2colNode] + iparmno) -=  
                            ( icc  - icc0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJT2colPrimeNode] + iparmno) -=  
                        ( ccprm  - ccprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJT2colPrimeNode] + iparmno) -=  
                        ( iccprm  - iccprm0) * DELAinv * DvDp;

                if(here->BJT2emitNode != here->BJT2emitPrimeNode){
                    *(info->SEN_RHS[here->BJT2emitNode] + iparmno) -= 
                            ( ce  - ce0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJT2emitNode] + iparmno) -= 
                            ( ice  - ice0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJT2emitPrimeNode] + iparmno) -= 
                        ( ceprm  - ceprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJT2emitPrimeNode] + iparmno) -= 
                        ( iceprm  - iceprm0) * DELAinv * DvDp;
                *(info->SEN_RHS[here->BJT2substNode] + iparmno) -=  
                        ( cs  - cs0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJT2substNode] + iparmno) -=  
                        ( ics  - ics0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");                  

                printf("senb = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2baseNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2baseNode] + iparmno));
                printf("senbrm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2basePrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2basePrimeNode] + iparmno));
                printf("senc = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2colNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2colNode] + iparmno));
                printf("sencprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2colPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2colPrimeNode] + iparmno));
                printf("sene = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2emitNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2emitNode] + iparmno));
                printf("seneprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2emitPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2emitPrimeNode] + iparmno));
                printf("sens = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJT2substNode] + iparmno),
                        *(info->SEN_iRHS[here->BJT2substNode] + iparmno));
#endif /* SENSDEBUG */

            }


next1:          
            switch(flag){
            case 0: 
                if (*(here->BJT2senCbx) == 0){ 
                    here->BJT2area = A0;
                    goto pertvbe ;
                }
                else{
                    goto pertvbx;
                }
            case 1: 
                goto pertvbe ; 
            case 2: 
                goto pertvbc ;
            case 3: 
                goto pertvcs ;
            case 4: 
                break; 
            }

            /* put the unperturbed values back into the state vector */
            for(i=0; i <= 20; i++) {
                *(ckt->CKTstate0 + here->BJT2state + i) = *(SaveState + i);
            }
            here->BJT2senPertFlag = OFF;
        }

    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("BJT2senacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

