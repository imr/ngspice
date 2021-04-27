/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "bjtdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


int
BJTsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
    double   SaveState0[27];
    int    i;
    int    iparmno;
    int    error;
    double A0;
    double DELA;
    double Apert;
    double DELAinv;
    double cb0;
    double cb;
    double cc0;
    double cc;
    double cx0;
    double ccpr0;
    double cepr0;
    double DcbDp;
    double DccDp;
    double DceDp;
    double DccprDp;
    double DceprDp;
    double DcxDp;
    double DbprmDp;
    double DcprmDp;
    double DeprmDp;
    double gx;
    double gx0;
    double tag0;
    double tag1;
    double qbe0;
    double qbe;
    double qbc0;
    double qbc;
    double qcs0;
    double qcs;
    double qbx0;
    double qbx;
    double DqbeDp = 0.0;
    double DqbcDp = 0.0;
    double DqcsDp = 0.0;
    double DqbxDp = 0.0;
    double Osxpbe;
    double Osxpbc;
    double Osxpcs;
    double Osxpbx;
    SENstruct *info;

    tag0 = ckt->CKTag[0];
    tag1 = ckt->CKTag[1];
    if(ckt->CKTorder == 1){
        tag1 = 0;
    }
#ifdef SENSDEBUG
    printf("BJTsenload \n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
    printf("CKTorder = %.5e\n",ckt->CKTorder);
    printf("tag0=%.7e,tag1=%.7e\n",tag0,tag1);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;

    info->SENstatus = PERTURBATION;

    /*  loop through all the models */
    for( ; model != NULL; model = BJTnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = BJTinstances(model); here != NULL ;
                here=BJTnextInstance(here)) {

#ifdef SENSDEBUG
            printf("base = %d , baseprm = %d ,col = %d, colprm = %d\n",
                    here->BJTbaseNode ,here->BJTbasePrimeNode,
                    here->BJTcolNode,here->BJTcolPrimeNode);
            printf("emit = %d , emitprm = %d ,subst = %d, senparmno = %d\n",
                    here->BJTemitNode ,here->BJTemitPrimeNode,
                    here->BJTsubstNode,here->BJTsenParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 20; i++){
                *(SaveState0 + i) = *(ckt->CKTstate0 + here->BJTstate + i);
            }
            *(SaveState0 + 21) = *(ckt->CKTstate1 + here->BJTcexbc);
            *(SaveState0 + 22) = *(ckt->CKTstate2 + here->BJTcexbc);
            *(SaveState0 + 23) = here->BJTcapbe;
            *(SaveState0 + 24) = here->BJTcapbc;
            *(SaveState0 + 25) = here->BJTcapsub;
            *(SaveState0 + 26) = here->BJTcapbx;

            if(here->BJTsenParmNo == 0) goto next;

            cx0 = model->BJTtype * *(ckt->CKTstate0 + here->BJTcb);
            ccpr0 = model->BJTtype * *(ckt->CKTstate0 + here->BJTcc);
            cepr0 = -cx0 - ccpr0;

            here->BJTsenPertFlag = ON;
            error =  BJTload((GENmodel*)model,ckt);
            if(error) return(error);

            cb0 = model->BJTtype * *(ckt->CKTstate0 + here->BJTcb);
            cc0 = model->BJTtype * *(ckt->CKTstate0 + here->BJTcc);
            gx0 = *(ckt->CKTstate0 + here->BJTgx);

            qbe0 = *(ckt->CKTstate0 + here->BJTqbe);
            qbc0 = *(ckt->CKTstate0 + here->BJTqbc);
            qcs0 = *(ckt->CKTstate0 + here->BJTqsub);
            qbx0 = *(ckt->CKTstate0 + here->BJTqbx);

            /* perturbation of area */

            A0 = here->BJTarea;
            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            here->BJTsenPertFlag = ON;
            here->BJTarea = Apert;
            error =  BJTload((GENmodel*)model,ckt);
            if(error) return(error);
            here->BJTarea = A0;
            here->BJTsenPertFlag = OFF;


            cb = model->BJTtype * *(ckt->CKTstate0 + here->BJTcb);
            cc = model->BJTtype * *(ckt->CKTstate0 + here->BJTcc);
            gx = *(ckt->CKTstate0 + here->BJTgx);

            qbe = *(ckt->CKTstate0 + here->BJTqbe);
            qbc = *(ckt->CKTstate0 + here->BJTqbc);
            qcs = *(ckt->CKTstate0 + here->BJTqsub);
            qbx = *(ckt->CKTstate0 + here->BJTqbx);

            /* compute the gradients of currents */
            DcbDp = (cb - cb0) * DELAinv;
            DccDp = (cc - cc0) * DELAinv;
            DceDp = DcbDp + DccDp;

            DccprDp  = 0;
            DceprDp  = 0;
            DcxDp  = 0;
            if(here->BJTcolNode != here->BJTcolPrimeNode)
                DccprDp = ccpr0 * info->SENpertfac * DELAinv;
            if(here->BJTemitNode != here->BJTemitPrimeNode)
                DceprDp = cepr0 * info->SENpertfac * DELAinv;
            if(here->BJTbaseNode != here->BJTbasePrimeNode){
                if(gx0) DcxDp =  cx0 * DELAinv * (gx-gx0)/gx0;
            }
            DbprmDp = DcbDp - DcxDp;
            DcprmDp = DccDp - DccprDp;
            DeprmDp = - DceDp  - DceprDp;

            DqbeDp = (qbe - qbe0)*DELAinv;
            DqbcDp = (qbc - qbc0)*DELAinv;
            DqcsDp = (qcs - qcs0)*DELAinv;
            DqbxDp = (qbx - qbx0)*DELAinv;

            *(here->BJTdphibedp) = DqbeDp;
            *(here->BJTdphibcdp) = DqbcDp;
            *(here->BJTdphisubdp) = DqcsDp;
            *(here->BJTdphibxdp) = DqbxDp;

#ifdef SENSDEBUG
            printf("cb0 = %.7e ,cb = %.7e,\n",cb0,cb); 
            printf("cc0 = %.7e ,cc = %.7e,\n",cc0,cc); 
            printf("ccpr0 = %.7e \n",ccpr0); 
            printf("cepr0 = %.7e \n",cepr0); 
            printf("cx0 = %.7e \n",cx0); 
            printf("qbe0 = %.7e ,qbe = %.7e,\n",qbe0,qbe); 
            printf("qbc0 = %.7e ,qbc = %.7e,\n",qbc0,qbc); 
            printf("qcs0 = %.7e ,qcs = %.7e,\n",qcs0,qcs); 
            printf("qbx0 = %.7e ,qbx = %.7e,\n",qbx0,qbx); 
            printf("\n");

#endif /* SENSDEBUG */

            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN))
                goto restore;

            /* load the RHS matrix */

            *(info->SEN_RHS[here->BJTbaseNode] + here->BJTsenParmNo)
                -= DcxDp;
            *(info->SEN_RHS[here->BJTbasePrimeNode] + here->BJTsenParmNo)
                -= DbprmDp;
            *(info->SEN_RHS[here->BJTcolNode] + here->BJTsenParmNo)
                -= DccprDp;
            *(info->SEN_RHS[here->BJTcolPrimeNode] + here->BJTsenParmNo)
                -= DcprmDp;
            *(info->SEN_RHS[here->BJTemitNode] + here->BJTsenParmNo)
                -= DceprDp;
            *(info->SEN_RHS[here->BJTemitPrimeNode] + here->BJTsenParmNo)
                -= DeprmDp;
#ifdef SENSDEBUG
            printf("after loading\n");
            printf("DcxDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTbaseNode] + here->BJTsenParmNo));
            printf("DcbprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTbasePrimeNode] + 
                    here->BJTsenParmNo));
            printf("DccprDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTcolNode] + here->BJTsenParmNo));
            printf("DcprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTcolPrimeNode] + 
                    here->BJTsenParmNo));
            printf("DceprDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTemitNode] +
                    here->BJTsenParmNo));
            printf("DceprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJTemitPrimeNode] + 
                    here->BJTsenParmNo));
#endif /* SENSDEBUG */

next:       
            if((info->SENmode == DCSEN)||(ckt->CKTmode&MODETRANOP))goto restore;
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN))
                goto restore;

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                Osxpbe = tag0 * *(ckt->CKTstate1 + here->BJTsensxpbe +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJTsensxpbe +
                        8*(iparmno - 1) + 1);

                Osxpbc = tag0 * *(ckt->CKTstate1 + here->BJTsensxpbc +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJTsensxpbc +
                        8*(iparmno - 1) + 1);

                Osxpcs = tag0 * *(ckt->CKTstate1 + here->BJTsensxpsub +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJTsensxpsub +
                        8*(iparmno - 1) + 1);

                Osxpbx = tag0 * *(ckt->CKTstate1 + here->BJTsensxpbx +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJTsensxpbx +
                        8*(iparmno - 1) + 1);
#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("Osxpbe=%.7e,Osxpbc=%.7e\n",Osxpbe,Osxpbc);
                printf("Osxpcs=%.7e,Osxpbx=%.7e\n",Osxpcs,Osxpbx);
                printf("sxpbe=%.7e,sdbe=%.7e\n",
                        *(ckt->CKTstate1 + here->BJTsensxpbe + 8*(iparmno - 1))
                        ,*(ckt->CKTstate1 + here->BJTsensxpbe + 
                        8*(iparmno - 1) + 1));
                printf("sxpbc=%.7e,sdbc=%.7e\n",
                        *(ckt->CKTstate1 + here->BJTsensxpbc + 8*(iparmno - 1))
                        ,*(ckt->CKTstate1 + here->BJTsensxpbc + 
                        8*(iparmno - 1) + 1));
                printf("\n");
#endif /* SENSDEBUG */

                if(iparmno == here->BJTsenParmNo){ 
                    Osxpbe = Osxpbe -  tag0 * DqbeDp;
                    Osxpbc = Osxpbc -  tag0 * DqbcDp;
                    Osxpcs = Osxpcs -  tag0 * DqcsDp;
                    Osxpbx = Osxpbx -  tag0 * DqbxDp;
                }

#ifdef SENSDEBUG

                printf("Osxpbe=%.7e,Osxpbc=%.7e\n",Osxpbe,Osxpbc);
                printf("Osxpcs=%.7e,Osxpbx=%.7e\n",Osxpcs,Osxpbx);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->BJTbaseNode] + iparmno) 
                    += model->BJTtype * Osxpbx;

                *(info->SEN_RHS[here->BJTbasePrimeNode] + iparmno) 
                    += model->BJTtype * (Osxpbe + Osxpbc);

                *(info->SEN_RHS[here->BJTcolPrimeNode] + iparmno) 
                    -= model->BJTtype * (Osxpbc + Osxpcs + Osxpbx );

                *(info->SEN_RHS[here->BJTemitPrimeNode] + iparmno) 
                    -= model->BJTtype * Osxpbe;

                *(info->SEN_RHS[here->BJTsubstNode] + iparmno) 
                    += model->BJTtype * Osxpcs;

            }


            /* put the unperturbed values back into the state vector */
restore:    
            for(i=0; i <= 20; i++){
                *(ckt->CKTstate0 + here->BJTstate + i) = *(SaveState0 + i);
            }
            *(ckt->CKTstate1 + here->BJTcexbc) = *(SaveState0 + 21);
            here->BJTcapbe = *(SaveState0 + 23) ;
            here->BJTcapbc = *(SaveState0 + 24) ;
            here->BJTcapsub = *(SaveState0 + 25) ;
            here->BJTcapbx = *(SaveState0 + 26) ;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("BJTsenload end\n");
#endif /* SENSDEBUG */
    return(OK);
}
