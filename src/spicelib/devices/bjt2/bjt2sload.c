/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current sensitivity 
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
BJT2sLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
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
    printf("BJT2senload \n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
    printf("CKTorder = %.5e\n",ckt->CKTorder);
    printf("tag0=%.7e,tag1=%.7e\n",tag0,tag1);
#endif /* SENSDEBUG */
    info = ckt->CKTsenInfo;

    info->SENstatus = PERTURBATION;

    /*  loop through all the models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;

#ifdef SENSDEBUG
            printf("base = %d , baseprm = %d ,col = %d, colprm = %d\n",
                    here->BJT2baseNode ,here->BJT2basePrimeNode,
                    here->BJT2colNode,here->BJT2colPrimeNode);
            printf("emit = %d , emitprm = %d ,subst = %d, senparmno = %d\n",
                    here->BJT2emitNode ,here->BJT2emitPrimeNode,
                    here->BJT2substNode,here->BJT2senParmNo);
#endif /* SENSDEBUG */


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 20; i++){
                *(SaveState0 + i) = *(ckt->CKTstate0 + here->BJT2state + i);
            }
            *(SaveState0 + 21) = *(ckt->CKTstate1 + here->BJT2cexbc);
            *(SaveState0 + 22) = *(ckt->CKTstate2 + here->BJT2cexbc);
            *(SaveState0 + 23) = here->BJT2capbe;
            *(SaveState0 + 24) = here->BJT2capbc;
            *(SaveState0 + 25) = here->BJT2capsub;
            *(SaveState0 + 26) = here->BJT2capbx;

            if(here->BJT2senParmNo == 0) goto next;

            cx0 = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cb);
            ccpr0 = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cc);
            cepr0 = -cx0 - ccpr0;

            here->BJT2senPertFlag = ON;
            error =  BJT2load((GENmodel*)model,ckt);
            if(error) return(error);

            cb0 = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cb);
            cc0 = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cc);
            gx0 = *(ckt->CKTstate0 + here->BJT2gx);

            qbe0 = *(ckt->CKTstate0 + here->BJT2qbe);
            qbc0 = *(ckt->CKTstate0 + here->BJT2qbc);
            qcs0 = *(ckt->CKTstate0 + here->BJT2qsub);
            qbx0 = *(ckt->CKTstate0 + here->BJT2qbx);

            /* perturbation of area */

            A0 = here->BJT2area;
            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            here->BJT2senPertFlag = ON;
            here->BJT2area = Apert;
            error =  BJT2load((GENmodel*)model,ckt);
            if(error) return(error);
            here->BJT2area = A0;
            here->BJT2senPertFlag = OFF;


            cb = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cb);
            cc = model->BJT2type * *(ckt->CKTstate0 + here->BJT2cc);
            gx = *(ckt->CKTstate0 + here->BJT2gx);

            qbe = *(ckt->CKTstate0 + here->BJT2qbe);
            qbc = *(ckt->CKTstate0 + here->BJT2qbc);
            qcs = *(ckt->CKTstate0 + here->BJT2qsub);
            qbx = *(ckt->CKTstate0 + here->BJT2qbx);

            /* compute the gradients of currents */
            DcbDp = (cb - cb0) * DELAinv;
            DccDp = (cc - cc0) * DELAinv;
            DceDp = DcbDp + DccDp;

            DccprDp  = 0;
            DceprDp  = 0;
            DcxDp  = 0;
            if(here->BJT2colNode != here->BJT2colPrimeNode)
                DccprDp = ccpr0 * info->SENpertfac * DELAinv;
            if(here->BJT2emitNode != here->BJT2emitPrimeNode)
                DceprDp = cepr0 * info->SENpertfac * DELAinv;
            if(here->BJT2baseNode != here->BJT2basePrimeNode){
                if(gx0) DcxDp =  cx0 * DELAinv * (gx-gx0)/gx0;
            }
            DbprmDp = DcbDp - DcxDp;
            DcprmDp = DccDp - DccprDp;
            DeprmDp = - DceDp  - DceprDp;

            DqbeDp = (qbe - qbe0)*DELAinv;
            DqbcDp = (qbc - qbc0)*DELAinv;
            DqcsDp = (qcs - qcs0)*DELAinv;
            DqbxDp = (qbx - qbx0)*DELAinv;

            *(here->BJT2dphibedp) = DqbeDp;
            *(here->BJT2dphibcdp) = DqbcDp;
            *(here->BJT2dphisubdp) = DqcsDp;
            *(here->BJT2dphibxdp) = DqbxDp;

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

            *(info->SEN_RHS[here->BJT2baseNode] + here->BJT2senParmNo)
                -= DcxDp;
            *(info->SEN_RHS[here->BJT2basePrimeNode] + here->BJT2senParmNo)
                -= DbprmDp;
            *(info->SEN_RHS[here->BJT2colNode] + here->BJT2senParmNo)
                -= DccprDp;
            *(info->SEN_RHS[here->BJT2colPrimeNode] + here->BJT2senParmNo)
                -= DcprmDp;
            *(info->SEN_RHS[here->BJT2emitNode] + here->BJT2senParmNo)
                -= DceprDp;
            *(info->SEN_RHS[here->BJT2emitPrimeNode] + here->BJT2senParmNo)
                -= DeprmDp;
#ifdef SENSDEBUG
            printf("after loading\n");
            printf("DcxDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2baseNode] + here->BJT2senParmNo));
            printf("DcbprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2basePrimeNode] + 
                    here->BJT2senParmNo));
            printf("DccprDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2colNode] + here->BJT2senParmNo));
            printf("DcprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2colPrimeNode] + 
                    here->BJT2senParmNo));
            printf("DceprDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2emitNode] +
                    here->BJT2senParmNo));
            printf("DceprmDp=%.7e\n",
                    *(info->SEN_RHS[here->BJT2emitPrimeNode] + 
                    here->BJT2senParmNo));
#endif /* SENSDEBUG */

next:       
            if((info->SENmode == DCSEN)||(ckt->CKTmode&MODETRANOP))goto restore;
            if((info->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN))
                goto restore;

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                Osxpbe = tag0 * *(ckt->CKTstate1 + here->BJT2sensxpbe +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJT2sensxpbe +
                        8*(iparmno - 1) + 1);

                Osxpbc = tag0 * *(ckt->CKTstate1 + here->BJT2sensxpbc +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJT2sensxpbc +
                        8*(iparmno - 1) + 1);

                Osxpcs = tag0 * *(ckt->CKTstate1 + here->BJT2sensxpsub +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJT2sensxpsub +
                        8*(iparmno - 1) + 1);

                Osxpbx = tag0 * *(ckt->CKTstate1 + here->BJT2sensxpbx +
                        8*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->BJT2sensxpbx +
                        8*(iparmno - 1) + 1);
#ifdef SENSDEBUG
                printf("iparmno=%d\n",iparmno);
                printf("Osxpbe=%.7e,Osxpbc=%.7e\n",Osxpbe,Osxpbc);
                printf("Osxpcs=%.7e,Osxpbx=%.7e\n",Osxpcs,Osxpbx);
                printf("sxpbe=%.7e,sdbe=%.7e\n",
                        *(ckt->CKTstate1 + here->BJT2sensxpbe + 8*(iparmno - 1))
                        ,*(ckt->CKTstate1 + here->BJT2sensxpbe + 
                        8*(iparmno - 1) + 1));
                printf("sxpbc=%.7e,sdbc=%.7e\n",
                        *(ckt->CKTstate1 + here->BJT2sensxpbc + 8*(iparmno - 1))
                        ,*(ckt->CKTstate1 + here->BJT2sensxpbc + 
                        8*(iparmno - 1) + 1));
                printf("\n");
#endif /* SENSDEBUG */

                if(iparmno == here->BJT2senParmNo){ 
                    Osxpbe = Osxpbe -  tag0 * DqbeDp;
                    Osxpbc = Osxpbc -  tag0 * DqbcDp;
                    Osxpcs = Osxpcs -  tag0 * DqcsDp;
                    Osxpbx = Osxpbx -  tag0 * DqbxDp;
                }

#ifdef SENSDEBUG

                printf("Osxpbe=%.7e,Osxpbc=%.7e\n",Osxpbe,Osxpbc);
                printf("Osxpcs=%.7e,Osxpbx=%.7e\n",Osxpcs,Osxpbx);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->BJT2baseNode] + iparmno) 
                    += model->BJT2type * Osxpbx;

                *(info->SEN_RHS[here->BJT2basePrimeNode] + iparmno) 
                    += model->BJT2type * (Osxpbe + Osxpbc);

                *(info->SEN_RHS[here->BJT2colPrimeNode] + iparmno) 
                    -= model->BJT2type * (Osxpbc + Osxpcs + Osxpbx );

                *(info->SEN_RHS[here->BJT2emitPrimeNode] + iparmno) 
                    -= model->BJT2type * Osxpbe;

                *(info->SEN_RHS[here->BJT2substNode] + iparmno) 
                    += model->BJT2type * Osxpcs;

            }


            /* put the unperturbed values back into the state vector */
restore:    
            for(i=0; i <= 20; i++){
                *(ckt->CKTstate0 + here->BJT2state + i) = *(SaveState0 + i);
            }
            *(ckt->CKTstate1 + here->BJT2cexbc) = *(SaveState0 + 21);
            *(ckt->CKTstate1 + here->BJT2cexbc) = *(SaveState0 + 21);
            here->BJT2capbe = *(SaveState0 + 23) ;
            here->BJT2capbc = *(SaveState0 + 24) ;
            here->BJT2capsub = *(SaveState0 + 25) ;
            here->BJT2capbx = *(SaveState0 + 26) ;

        }
    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("BJT2senload end\n");
#endif /* SENSDEBUG */
    return(OK);
}
