/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* actually load the current inductance value into the 
     * sparse matrix previously provided 
     */

#include "ngspice.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
INDload(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    double veq;
    double req;
    int error;
    
#ifdef MUTUAL
    MUTinstance *muthere;
    MUTmodel *mutmodel;
    int ktype;
    int itype;
#endif

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->INDnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->INDinstances; here != NULL ;
                here=here->INDnextInstance) {
	    
	    if (here->INDowner != ARCHme) continue;

            if(!(ckt->CKTmode & (MODEDC|MODEINITPRED))) {
                if(ckt->CKTmode & MODEUIC && ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate0 + here->INDflux) = here->INDinduct *
                            here->INDinitCond;
                } else {
                    *(ckt->CKTstate0 + here->INDflux) = here->INDinduct *
                            *(ckt->CKTrhsOld + here->INDbrEq);
                }
            }
#ifdef MUTUAL
        }
    }
    ktype = CKTtypelook("mutual");
    mutmodel = (MUTmodel *)(ckt->CKThead[ktype]);
    /*  loop through all the mutual inductor models */
    for( ; mutmodel != NULL; mutmodel = mutmodel->MUTnextModel ) {

        /* loop through all the instances of the model */
        for (muthere = mutmodel->MUTinstances; muthere != NULL ;
                muthere=muthere->MUTnextInstance) {
	    
	    if (muthere->MUTowner != ARCHme) continue;

            if(!(ckt->CKTmode& (MODEDC|MODEINITPRED))) {
                *(ckt->CKTstate0 + muthere->MUTind1->INDflux)  +=
                    muthere->MUTfactor * *(ckt->CKTrhsOld +
                    muthere->MUTind2->INDbrEq);
		    
                *(ckt->CKTstate0 + muthere->MUTind2->INDflux)  +=
                    muthere->MUTfactor * *(ckt->CKTrhsOld +
                    muthere->MUTind1->INDbrEq);
            }
	    
            *(muthere->MUTbr1br2) -= muthere->MUTfactor*ckt->CKTag[0];
            *(muthere->MUTbr2br1) -= muthere->MUTfactor*ckt->CKTag[0];
        }
    }
    itype = CKTtypelook("Inductor");
    model = (INDmodel *)(ckt->CKThead[itype]);
    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->INDnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->INDinstances; here != NULL ;
                here=here->INDnextInstance) {
	    if (here->INDowner != ARCHme) continue;

#endif /*MUTUAL*/
            if(ckt->CKTmode & MODEDC) {
                req = 0.0;
                veq = 0.0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    *(ckt->CKTstate0 + here->INDflux) =
                        *(ckt->CKTstate1 + here->INDflux);
                } else {
#endif /*PREDICTOR*/
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->INDflux) =
                                *(ckt->CKTstate0 + here->INDflux);
                    }
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
                error=NIintegrate(ckt,&req,&veq,here->INDinduct,here->INDflux);
                if(error) return(error);
            }
	    
            *(ckt->CKTrhs+here->INDbrEq) += veq;
            
	    if(ckt->CKTmode & MODEINITTRAN) {
                *(ckt->CKTstate1+here->INDvolt) = 
                    *(ckt->CKTstate0+here->INDvolt);
            }
	    
            *(here->INDposIbrptr) +=  1;
            *(here->INDnegIbrptr) -=  1;
            *(here->INDibrPosptr) +=  1;
            *(here->INDibrNegptr) -=  1;
            *(here->INDibrIbrptr) -=  req;
        }
    }
    return(OK);
}
