/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
DIOsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    double SaveState[5];
    int    error;
    int    i;
    int iparmno;
    int flag;
    double A0;
    double DELA;
    double Apert;
    double DELAinv;
    double vte;
    double gspr0;
    double geq0;
    double xceq0;
    double vspr;
    double ivspr;
    double vd;
    double ivd;
    double vdOp;
    double gspr;
    double geq;
    double xceq;
    double cspr;
    double icspr;
    double cd;
    double icd;
    double cpos0;
    double icpos0;
    double cpos;
    double icpos;
    double cposprm0;
    double icposprm0;
    double cposprm;
    double icposprm;
    double cneg0;
    double icneg0;
    double cneg;
    double icneg;
    double DvdDp;
    SENstruct *info;

#ifdef SENSDEBUG
    printf("DIOsenacload\n");
#endif /* SENSDEBUG */

    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;
    /*  loop through all the models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            /* save the unperturbed values in the state vector */
            for(i=0; i <= 4; i++) {
                *(SaveState + i) = *(ckt->CKTstate0 + here->DIOstate + i);
            }
            vspr = *(ckt->CKTrhsOld + here->DIOposNode) 
                    - *(ckt->CKTrhsOld + here->DIOposPrimeNode) ;
            ivspr = *(ckt->CKTirhsOld + here->DIOposNode) 
                    - *(ckt->CKTirhsOld + here->DIOposPrimeNode) ;
            vd = *(ckt->CKTrhsOld + here->DIOposPrimeNode) 
                    - *(ckt->CKTrhsOld + here->DIOnegNode) ;
            ivd = *(ckt->CKTirhsOld + here->DIOposPrimeNode) 
                    - *(ckt->CKTirhsOld + here->DIOnegNode) ;
            vdOp = *(ckt->CKTrhsOp + here->DIOposPrimeNode)
                    - *(ckt->CKTrhsOp + here->DIOnegNode);

            /* without perturbation  */
#ifdef SENSDEBUG
            printf("without perturbation \n");
#endif /* SENSDEBUG */
            *(ckt->CKTstate0 + here->DIOvoltage) = vdOp;

            here->DIOsenPertFlag = ON;
            if(info->SENacpertflag == 1){
                if ((error = DIOload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->DIOsenGeq) =  *(ckt->CKTstate0 + here->DIOconduct);
                *(here->DIOsenCeq) = *(ckt->CKTstate0 + here->DIOcapCurrent);
            }
            geq0 = *(here->DIOsenGeq);
            xceq0 = *(here->DIOsenCeq) * ckt->CKTomega;
            A0 = here->DIOarea;
            gspr0=here->DIOtConductance;
            cpos0 = gspr0 * vspr;
            icpos0 = gspr0 * ivspr;
            cposprm0 = geq0 * vd - xceq0 * ivd - cpos0;
            icposprm0 = geq0 * ivd + xceq0 * vd - icpos0;
            cneg0 = - geq0 * vd + xceq0 * ivd;
            icneg0 = - geq0 * ivd - xceq0 * vd;


#ifdef SENSDEBUG
            printf("gspr0 = %.7e , geq0 = %.7e ,xceq0 = %.7e\n", 
                    gspr0 ,geq0,xceq0);
            printf("cpos0 = %.7e + j%.7e , cneg0 = %.7e + j%.7e\n",
                    cpos0,icpos0,cneg0,icneg0);
#endif /* SENSDEBUG */
            /* Perturbation of Area */

#ifdef SENSDEBUG
            printf("Perturbation of Area\n");
#endif /* SENSDEBUG */
            if(here->DIOsenParmNo == 0) goto pertvd;

            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            if(info->SENacpertflag == 1){
                here->DIOarea = Apert;
                *(ckt->CKTstate0 + here->DIOvoltage) = vdOp;
                if ((error = DIOload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->DIOsenGeq + 1) =  *(ckt->CKTstate0 + here->DIOconduct);
                *(here->DIOsenCeq + 1)= *(ckt->CKTstate0 + here->DIOcapCurrent);
                here->DIOarea = A0;
            }
            gspr=here->DIOtConductance*Apert; 
            geq = *(here->DIOsenGeq + 1);
            xceq = *(here->DIOsenCeq + 1) * ckt->CKTomega;
            flag = 0;
            goto load;

pertvd:     /* Perturbation of Diode Voltage */
#ifdef SENSDEBUG
            printf("Perturbation of vd\n");
#endif /* SENSDEBUG */

            vte=model->DIOemissionCoeff * CONSTKoverQ * here->DIOtemp;
            A0 = vdOp;
            DELA = info->SENpertfac * vte;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;


            if(info->SENacpertflag == 1){
                *(ckt->CKTstate0 + here->DIOvoltage) = Apert;
                if ((error = DIOload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->DIOsenGeq + 2) =  *(ckt->CKTstate0 + here->DIOconduct);
                *(here->DIOsenCeq + 2)= *(ckt->CKTstate0 + here->DIOcapCurrent);
                *(ckt->CKTstate0 + here->DIOvoltage) = A0;
            }
            gspr=here->DIOtConductance*here->DIOarea; 
            geq = *(here->DIOsenGeq + 2);
            xceq = *(here->DIOsenCeq + 2) * ckt->CKTomega;


            flag = 1;

load: 

            cspr = gspr * vspr;
            icspr = gspr * ivspr;
            cd = geq * vd - xceq * ivd;
            icd = geq * ivd + xceq * vd;

            cpos = cspr;
            icpos  = icspr;
            cposprm = ( - cspr + cd );
            icposprm = ( - icspr + icd );
            cneg = ( - cd );
            icneg = ( - icd );

#ifdef SENSDEBUG
            printf("gspr = %.7e , geq = %.7e , xceq = %.7e\n",
                    gspr,geq,xceq);
            printf("cspr = %.7e + j%.7e , cd = %.7e + j%.7e\n",
                    cspr,icspr,cd,icd);
            printf("cpos = %.7e + j%.7e , cposprm = %.7e + j%.7e",
                    cpos,icpos,cposprm,icposprm);
            printf(", cneg = %.7e + %.7e\n",cneg,icneg);
            printf("senpprm = %.7e  "
                    ,info->SEN_Sap[here->DIOposPrimeNode][here->DIOsenParmNo]);
            printf("senneg = %.7e \n",
                    info->SEN_Sap[here->DIOnegNode][here->DIOsenParmNo]);
            printf("A0 = %.7e , Apert = %.7e ,factor = %.7e\n,vte = %.7e",
                    A0 ,Apert,DELAinv,vte);
#endif /* SENSDEBUG */



            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                /* calculate the DC sensitivities of operating points */
                DvdDp = info->SEN_Sap[here->DIOposPrimeNode][iparmno]
                    - info->SEN_Sap[here->DIOnegNode][iparmno];
                if(flag == 0){
                    if (here->DIOsenParmNo != iparmno) continue;
                    /* area : so no DC sensitivity term involved */
                    DvdDp =1;
                }
                /* load the RHS matrix */

                if(here->DIOposNode != here->DIOposPrimeNode){
                    /* DcposDp */ 
                    *(info->SEN_RHS[here->DIOposNode] + iparmno) -= 
                            (cpos - cpos0) * DELAinv * DvdDp ;
                    /* DicposDp */ 
                    *(info->SEN_iRHS[here->DIOposNode] + iparmno) -= 
                            (icpos - icpos0) * DELAinv * DvdDp ;
                }



                /* DcposprmDp */ 
                *(info->SEN_RHS[here->DIOposPrimeNode] + iparmno) -= 
                        (cposprm - cposprm0) * DELAinv * DvdDp ;
                /* DicposprmDp */ 
                *(info->SEN_iRHS[here->DIOposPrimeNode] + iparmno) -= 
                        (icposprm - icposprm0) * DELAinv * DvdDp ;
                /* DcnegDp */ 
                *(info->SEN_RHS[here->DIOnegNode] + iparmno) -= 
                        (cneg - cneg0) * DELAinv * DvdDp ;
                /* DicnegDp */ 
                *(info->SEN_iRHS[here->DIOnegNode] + iparmno) -= 
                        (icneg - icneg0) * DELAinv * DvdDp ;

#ifdef SENSDEBUG

                printf("senpos = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->DIOposNode] + iparmno),
                        *(info->SEN_iRHS[here->DIOposNode] + iparmno));
                printf("senposprm = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->DIOposPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->DIOposPrimeNode] + iparmno));
                printf("senneg = %.7e + j%.7e ",
                        *(info->SEN_RHS[here->DIOnegNode] + iparmno),
                        *(info->SEN_iRHS[here->DIOnegNode] + iparmno));
                printf("flag = %d ,DvdDp = %.7e ,iparmno = %d,senparmno = %d\n"
                        ,flag,DvdDp,iparmno,here->DIOsenParmNo);

#endif /* SENSDEBUG */
            }

            if(!flag) goto pertvd;

            /* put the unperturbed values back into the state vector */
            for(i=0; i <= 4; i++) {
                *(ckt->CKTstate0 + here->DIOstate + i) = *(SaveState + i);
            }
            here->DIOsenPertFlag = OFF;
        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("DIOsenacload end \n");
#endif /* SENSDEBUG */

    return(OK);
}

