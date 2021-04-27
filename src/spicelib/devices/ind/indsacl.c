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
INDsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    double  cind,icind,val,ival;     
    MUTinstance *muthere;
    MUTmodel *mutmodel;
    double  cind1;
    double  icind1;
    double  cind2;
    double  icind2;
    double  val11;
    double  ival11;
    double  val12;
    double  ival12;
    double  val13;
    double  ival13;
    double  val21;
    double  ival21;
    double  val22;
    double  ival22;
    double  val23;
    double  ival23;
    double  rootl1;
    double  rootl2;
    double  w;
    double  k1;
    double  k2;
    int ktype;
    int itype;
    SENstruct *info;

    info = ckt->CKTsenInfo;
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

                cind1 = *(ckt->CKTrhsOld + muthere->MUTind1->INDbrEq);
                icind1 = *(ckt->CKTirhsOld + muthere->MUTind1->INDbrEq);
                cind2 = *(ckt->CKTrhsOld + muthere->MUTind2->INDbrEq);
                icind2 = *(ckt->CKTirhsOld + muthere->MUTind2->INDbrEq);
                rootl1 =  sqrt(muthere->MUTind1->INDinduct) ;
                rootl2 =  sqrt(muthere->MUTind2->INDinduct) ;
                k1 = 0.5 * muthere->MUTcoupling * rootl2 / rootl1 ; 
                k2 = 0.5 * muthere->MUTcoupling * rootl1 / rootl2 ; 
                w = ckt->CKTomega ;

                /* load the RHS matrix */

                if(muthere->MUTind1->INDsenParmNo){
                    val11 = - (w * (k1 * icind2));
                    ival11 =  w * (k1 * cind2);
                    val21 = - ( w * k1 * icind1) ;
                    ival21 = w * k1 * cind1 ;
                    *(info->SEN_RHS[muthere->MUTind1->INDbrEq] +
                            muthere->MUTind1->INDsenParmNo) += val11;
                    *(info->SEN_iRHS[muthere->MUTind1->INDbrEq] + 
                            muthere->MUTind1->INDsenParmNo) += ival11;
                    *(info->SEN_RHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTind1->INDsenParmNo) += val21;
                    *(info->SEN_iRHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTind1->INDsenParmNo) += ival21;
                }

                if(muthere->MUTind2->INDsenParmNo){
                    val12 = -( w * k2 * icind2) ;
                    ival12 = w * k2 * cind2 ;
                    val22 = - (w * ( k2 * icind1));
                    ival22 =  w * ( k2 * cind1);
                    *(info->SEN_RHS[muthere->MUTind1->INDbrEq] + 
                            muthere->MUTind2->INDsenParmNo) += val12;
                    *(info->SEN_iRHS[muthere->MUTind1->INDbrEq] + 
                            muthere->MUTind2->INDsenParmNo) += ival12;
                    *(info->SEN_RHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTind2->INDsenParmNo) += val22;
                    *(info->SEN_iRHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTind2->INDsenParmNo) += ival22;
                }

                if(muthere->MUTsenParmNo){
                    val13 = - w * rootl1 * rootl2 * icind2;
                    ival13 =  w * rootl1 * rootl2 * cind2;
                    val23 = - (w * rootl1 * rootl2 * icind1);
                    ival23 =  w * rootl1 * rootl2 * cind1;
                    *(info->SEN_RHS[muthere->MUTind1->INDbrEq] + 
                            muthere->MUTsenParmNo) += val13;
                    *(info->SEN_iRHS[muthere->MUTind1->INDbrEq] + 
                            muthere->MUTsenParmNo) += ival13;
                    *(info->SEN_RHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTsenParmNo) += val23;
                    *(info->SEN_iRHS[muthere->MUTind2->INDbrEq] + 
                            muthere->MUTsenParmNo) += ival23;
                }
            }
        }
    }
    itype = CKTtypelook("Inductor");
    model = (INDmodel *)(ckt->CKThead[itype]);
    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {
                if(here->INDsenParmNo){
                cind = *(ckt->CKTrhsOld + here->INDbrEq);
                icind = *(ckt->CKTirhsOld + here->INDbrEq);
                val = icind * ckt->CKTomega ;
                ival = cind * ckt->CKTomega ;

#ifdef SENSDEBUG
                fprintf(stdout,"cind = %.5e,icind = %.5e\n",cind,icind);
                fprintf(stdout,"val = %.5e,ival = %.5e\n",val,ival);
                fprintf(stdout,"brEq = %d,senparmno = %d\n",
                        here->INDbrEq,here->INDsenParmNo);
#endif /* SENSDEBUG */

                /* load the RHS matrix */

                *(info->SEN_RHS[here->INDbrEq] + here->INDsenParmNo) 
                    -= val;
                *(info->SEN_iRHS[here->INDbrEq] + here->INDsenParmNo) 
                    += ival;
            }
        }
    }
    return(OK);
}

