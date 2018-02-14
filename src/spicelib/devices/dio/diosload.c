/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* actually load the current sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    int    iparmno;
    int    error;
    int    i;
    double SaveState[6];
    double A0;
    double Apert;
    double DELA;
    double DELAinv;
    double cspr0;
    double cd0;
    double cd;
    double qd0;
    double qd;
    double DcsprDp;
    double DcdDp;
    double DqdDp = 0.0;
    double tag0;
    double tag1;
    double Osxp;
    SENstruct *info;

    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;

    tag0 = ckt->CKTag[0];
    tag1 = ckt->CKTag[1];
    if(ckt->CKTorder == 1){
        tag1 = 0;
    }

#ifdef SENSDEBUG
    printf("DIOsenload\n");
    fprintf(stdout,"DIOsenload\n");
    fprintf(stdout,"CKTtime = %.5e\n",ckt->CKTtime);
    fprintf(stdout,"CKTorder = %.5e\n",ckt->CKTorder);
    fprintf(stdout,"tag0 = %.5e tag1 = %.5e\n",tag0,tag1);
#endif /* SENSDEBUG */

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

#ifdef SENSDEBUG
            fprintf(stdout,"pos = %d , posprm = %d ,neg = %d, senparmno = %d\n",
                    here->DIOposNode ,here->DIOposPrimeNode,here->DIOnegNode,
                    here->DIOsenParmNo);
#endif /* SENSDEBUG */



            /* save the unperturbed values in the state vector */
            for(i=0; i <= 4; i++) {
                *(SaveState + i) = *(ckt->CKTstate0 + here->DIOstate + i);
            }
            *(SaveState + 5) = here->DIOcap;

            if(here->DIOsenParmNo == 0) goto next;


            cspr0 = *(ckt->CKTstate0 + here->DIOcurrent);
            here->DIOsenPertFlag = ON;
            error = DIOload((GENmodel*)model,ckt);
            cd0 = *(ckt->CKTstate0 + here->DIOcurrent);
            qd0 = *(ckt->CKTstate0 + here->DIOcapCharge );


#ifdef SENSDEBUG
            fprintf(stdout,"cd0 = %.7e \n",cd0);
#endif /* SENSDEBUG */

            A0 = here->DIOarea;
            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            here->DIOarea = Apert;
            error = DIOload((GENmodel*)model,ckt);
            if(error) return(error);
            here->DIOarea = A0;
            here->DIOsenPertFlag = OFF;

            cd = *(ckt->CKTstate0 + here->DIOcurrent) ;
            qd = *(ckt->CKTstate0 + here->DIOcapCharge);

            DcdDp = (cd -cd0) * DELAinv;
            DcsprDp = 0;
            if(here->DIOposNode != here->DIOposPrimeNode) {
                DcsprDp = cspr0 * info->SENpertfac * DELAinv; 
            }
            DqdDp = (qd - qd0)*DELAinv;

            *(here->DIOdphidp) = DqdDp;

#ifdef SENSDEBUG
            fprintf(stdout,"cd0 = %.7e ,cd = %.7e,DcdDp=%.7e\n", cd0,cd,DcdDp);
            fprintf(stdout,"cspr0 = %.7e ,DcsprDp=%.7e\n", cspr0,DcsprDp);
            fprintf(stdout,"qd0 = %.7e ,qd = %.7e,DqdDp=%.7e\n", qd0,qd,DqdDp);
#endif /* SENSDEBUG */

            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                goto restore;
            }

            /*
             *   load RHS matrices
             */
            *(info->SEN_RHS[here->DIOposNode] + here->DIOsenParmNo) -= DcsprDp;
            *(info->SEN_RHS[here->DIOposPrimeNode] + here->DIOsenParmNo) += 
                    DcsprDp - DcdDp ;

            *(info->SEN_RHS[here->DIOnegNode] + here->DIOsenParmNo) += DcdDp ;
next:       
            if((info->SENmode == DCSEN)||(ckt->CKTmode&MODETRANOP))goto restore;
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)){
                goto restore;
            }

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                Osxp = tag0 * *(ckt->CKTstate1 + here->DIOsensxp +
                        2*(iparmno - 1))
                        + tag1 * *(ckt->CKTstate1 + here->DIOsensxp +
                        2*(iparmno - 1) + 1);

#ifdef SENSDEBUG
                fprintf(stdout,"\n iparmno=%d,Osxp=%.7e\n",iparmno,Osxp);
#endif /* SENSDEBUG */

                if(iparmno == here->DIOsenParmNo) Osxp = Osxp - tag0 * DqdDp;
#ifdef SENSDEBUG
                fprintf(stdout,"Osxp=%.7e\n",Osxp);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->DIOposPrimeNode] + iparmno) += Osxp;
                *(info->SEN_RHS[here->DIOnegNode] + iparmno) -= Osxp;
            }
            /* put the unperturbed values back into the state vector */
restore:    
            for(i=0; i <= 4; i++) {
                *(ckt->CKTstate0 + here->DIOstate + i) = *(SaveState + i);
            }
            here->DIOcap = *(SaveState + 5);
        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("DIOsenload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

