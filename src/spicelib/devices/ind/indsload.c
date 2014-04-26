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
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
INDsLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
     INDinstance *here;
    int       iparmno;
    double    cind;
    double    Osxp;
    double    tag0;
    double    tag1;
    SENstruct *info;

    MUTinstance *muthere;
    MUTmodel *mutmodel;
    double   cind1;
    double   cind2;
    double   rootl1;
    double   rootl2;
    int ktype;
    int itype;
    int IND1_brEq;
    int IND2_brEq;

    info = ckt->CKTsenInfo;

    if((info->SENmode == DCSEN)||(ckt->CKTmode&MODETRANOP)) return( OK );
    if((info->SENmode == TRANSEN) && (ckt->CKTmode & MODEINITTRAN))  return(OK);

#ifdef SENSDEBUG
    fprintf(stdout,"INDsenLoad\n");
    fprintf(stdout,"time = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */


    tag0 = ckt->CKTag[0];
    tag1 = ckt->CKTag[1];
    if(ckt->CKTorder == 1){
        tag1 = 0;
    }

    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

        }
    }
    ktype = CKTtypelook("mutual");
    mutmodel = (MUTmodel *)(ckt->CKThead[ktype]);
    /*  loop through all the mutual inductor models */
    for( ; mutmodel != NULL; mutmodel = MUTnextModel(mutmodel)) {

        /* loop through all the instances of the model */
        for (muthere = MUTinstances(mutmodel); muthere != NULL ;
             muthere=MUTnextInstance(muthere)) {

            if(muthere->MUTsenParmNo ||
                muthere->MUTind1->INDsenParmNo ||
                muthere->MUTind2->INDsenParmNo){

                IND1_brEq = muthere->MUTind1->INDbrEq;
                IND2_brEq = muthere->MUTind2->INDbrEq;
                cind1 = *(ckt->CKTrhsOld + IND1_brEq);
                cind2 = *(ckt->CKTrhsOld + IND2_brEq);
                rootl1 = sqrt( muthere->MUTind1->INDinduct );
                rootl2 = sqrt( muthere->MUTind2->INDinduct );

                if(muthere->MUTsenParmNo){
                    *(info->SEN_RHS[IND1_brEq] + muthere->MUTsenParmNo) 
                        += tag0*cind2*rootl2*rootl1;
                    *(info->SEN_RHS[IND2_brEq] + muthere->MUTsenParmNo) 
                        += tag0*cind1*rootl2*rootl1;
                }
                if(muthere->MUTind1->INDsenParmNo){
                    *(info->SEN_RHS[IND1_brEq] + muthere->MUTind1->INDsenParmNo)
                        += tag0*cind2*muthere->MUTcoupling*rootl2 / (2*rootl1);
                    *(info->SEN_RHS[IND2_brEq] + muthere->MUTind1->INDsenParmNo)
                        += tag0*cind1*muthere->MUTcoupling*rootl2 / (2*rootl1);
                }
                if(muthere->MUTind2->INDsenParmNo){
                    *(info->SEN_RHS[IND1_brEq] + muthere->MUTind2->INDsenParmNo)
                        += tag0*cind2*muthere->MUTcoupling*rootl1 / (2*rootl2);
                    *(info->SEN_RHS[IND2_brEq] + muthere->MUTind2->INDsenParmNo)
                        += tag0*cind1*muthere->MUTcoupling*rootl1 / (2*rootl2); 
                }
            }

#ifdef  SENSDEBUG 
            fprintf(stdout,"cind1 = %.5e,cind2 = %.5e\n",cind1,cind2);
#endif /* SENSDEBUG */

        }
    }
    itype = CKTtypelook("Inductor");
    model = (INDmodel *)(ckt->CKThead[itype]);
    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {
        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            cind = *(ckt->CKTrhsOld + here->INDbrEq);
#ifdef SENSDEBUG
            fprintf(stdout,"\n cind=%.5e\n",cind);
            fprintf(stdout,"\n tag0=%.5e,tag1=%.5e\n",tag0,tag1);
#endif /* SENSDEBUG */
            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                Osxp = tag0 * *(ckt->CKTstate1 + here->INDsensxp
                        + 2*(iparmno - 1))
                    + tag1 * *(ckt->CKTstate1 + here->INDsensxp
                        + 2*(iparmno - 1) + 1);
                if(iparmno == here->INDsenParmNo) Osxp = Osxp - tag0 * cind;
#ifdef SENSDEBUG
                fprintf(stdout,"\n Osxp=%.5e\n",Osxp);
#endif /* SENSDEBUG */

                *(info->SEN_RHS[here->INDbrEq] + iparmno) -= Osxp;
            }
        }
    }
    return(OK);
}
