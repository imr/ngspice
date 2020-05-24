/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* actually load the current inductance value into the
 * sparse matrix previously provided
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
INDload(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    double veq;
    double req;
    double m;
    int error;

    MUTinstance *muthere;
    MUTmodel *mutmodel;
    int ktype;
    int itype;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            m = (here->INDm);

            if(!(ckt->CKTmode & (MODEDC|MODEINITPRED))) {
                if(ckt->CKTmode & MODEUIC && ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate0 + here->INDflux) = here->INDinduct / m *
                                                        here->INDinitCond;
                } else {
                    *(ckt->CKTstate0 + here->INDflux) = here->INDinduct / m *
                                                        *(ckt->CKTrhsOld + here->INDbrEq);
                }
            }
        }
    }
    ktype = CKTtypelook("mutual");
    mutmodel = (MUTmodel *)(ckt->CKThead[ktype]);
    /*  loop through all the mutual inductor models */
    for( ; mutmodel != NULL; mutmodel = MUTnextModel(mutmodel)) {

        /* loop through all the instances of the model */
        for (muthere = MUTinstances(mutmodel); muthere != NULL ;
                muthere=MUTnextInstance(muthere)) {

            if(!(ckt->CKTmode& (MODEDC|MODEINITPRED))) {
                /* set initial conditions for mutual inductance here, if uic is set */
                if (ckt->CKTmode & MODEUIC && ckt->CKTmode & MODEINITTRAN) {
                   *(ckt->CKTstate0 + muthere->MUTind1->INDflux) +=
                        muthere->MUTfactor * muthere->MUTind2->INDinitCond;

                   *(ckt->CKTstate0 + muthere->MUTind2->INDflux) +=
                        muthere->MUTfactor * muthere->MUTind1->INDinitCond;
                }
                else {
                    *(ckt->CKTstate0 + muthere->MUTind1->INDflux) +=
                        muthere->MUTfactor * *(ckt->CKTrhsOld +
                            muthere->MUTind2->INDbrEq);

                    *(ckt->CKTstate0 + muthere->MUTind2->INDflux) +=
                        muthere->MUTfactor * *(ckt->CKTrhsOld +
                            muthere->MUTind1->INDbrEq);
                }
            }
            *(muthere->MUTbr1br2Ptr) -= muthere->MUTfactor*ckt->CKTag[0];
            *(muthere->MUTbr2br1Ptr) -= muthere->MUTfactor*ckt->CKTag[0];
        }
    }
    itype = CKTtypelook("Inductor");
    model = (INDmodel *)(ckt->CKThead[itype]);
    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            if(ckt->CKTmode & MODEDC) {
                req = 0.0;
                veq = 0.0;
            } else {
                double newmind;
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
                m = (here->INDm);
                newmind = here->INDinduct/m;
                error=NIintegrate(ckt,&req,&veq,newmind,here->INDflux);
                if(error) return(error);
            }

            *(ckt->CKTrhs+here->INDbrEq) += veq;

            if(ckt->CKTmode & MODEINITTRAN) {
                *(ckt->CKTstate1+here->INDvolt) =
                    *(ckt->CKTstate0+here->INDvolt);
            }

            *(here->INDposIbrPtr) +=  1;
            *(here->INDnegIbrPtr) -=  1;
            *(here->INDibrPosPtr) +=  1;
            *(here->INDibrNegPtr) -=  1;
            *(here->INDibrIbrPtr) -=  req;
        }
    }
    return(OK);
}
