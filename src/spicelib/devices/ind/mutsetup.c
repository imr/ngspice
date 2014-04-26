/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* load the inductor structure with those pointers needed later
 * for fast matrix loading
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

#define TSTALLOC(ptr, first, second)                                    \
    do {                                                                \
        if ((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL) { \
            return(E_NOMEM);                                            \
        }                                                               \
    } while(0)


int
MUTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    MUTmodel *model = (MUTmodel*) inModel;
    MUTinstance *here;

    NG_IGNORE(states);

    for (; model; model = MUTnextModel(model))
        for (here = MUTinstances(model); here; here = MUTnextInstance(here)) {

            int ktype = CKTtypelook("Inductor");

            if (ktype <= 0) {
                SPfrontEnd->IFerrorf (ERR_PANIC,
                                      "mutual inductor, but inductors not available!");
                return(E_INTERN);
            }

            if (!here->MUTind1)
                here->MUTind1 = (INDinstance *) CKTfndDev(ckt, here->MUTindName1);
            if (!here->MUTind1) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                      "%s: coupling to non-existant inductor %s.",
                                      here->MUTname, here->MUTindName1);
            }
            if (!here->MUTind2)
                here->MUTind2 = (INDinstance *) CKTfndDev(ckt, here->MUTindName2);
            if (!here->MUTind2) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                      "%s: coupling to non-existant inductor %s.",
                                      here->MUTname, here->MUTindName2);
            }

            TSTALLOC(MUTbr1br2Ptr, MUTind1->INDbrEq, MUTind2->INDbrEq);
            TSTALLOC(MUTbr2br1Ptr, MUTind2->INDbrEq, MUTind1->INDbrEq);
        }

#ifdef USE_CUSPICE
    int i, j, status;
    INDmodel *indmodel;
    INDinstance *indhere;

    /* Counting the instances */
    for (model = (MUTmodel *)inModel; model; model = MUTnextModel(model)) {
        i = 0;

        for (here = MUTinstances(model); here; here = MUTnextInstance(here))
            i++;

        /* How much instances we have */
        model->n_instances = i;
    }

    /*  loop through all the mutual inductor models */
    for (model = (MUTmodel *)inModel; model; model = MUTnextModel(model)) {
        model->offset = ckt->total_n_values;

        j = 0;

        /* loop through all the instances of the model */
        for (here = MUTinstances(model); here; here = MUTnextInstance(here)) {
            /* For the Matrix */
            if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0))
                j++;

            if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0))
                j++;
        }

        model->n_values = model->n_instances;
        ckt->total_n_values += model->n_values;

        model->n_Ptr = j;
        ckt->total_n_Ptr += model->n_Ptr;


        /* Position Vector assignment */
        model->PositionVector = TMALLOC(int, model->n_instances);

        for (j = 0; j < model->n_instances; j++)
            model->PositionVector [j] = model->offset + j;


        /* PARTICULAR SITUATION */
        /* Pick up the IND model from one of the two IND instances */
        indmodel = INDmodPtr(MUTinstances(model)->MUTind1);
        model->n_instancesRHS = indmodel->n_instances;

        /* Position Vector assignment for the RHS */
        model->PositionVectorRHS = TMALLOC(int, model->n_instancesRHS);

        for (j = 0 ; j < model->n_instancesRHS ; j++)
            model->PositionVectorRHS [j] = indmodel->PositionVectorRHS [j];

        /* InstanceID assignment for every IND instance */
        j = 0;
        for (indhere = INDinstances(indmodel); indhere; indhere = INDnextInstance(indhere))
            indhere->instanceID = j++;

        /* InstanceID storing for every MUT instance */
        model->MUTparamCPU.MUTinstanceIND1Array = TMALLOC(int,
                                                          model->n_instances);
        model->MUTparamCPU.MUTinstanceIND2Array = TMALLOC(int,
                                                          model->n_instances);
        j = 0;
        for (here = MUTinstances(model); here; here = MUTnextInstance(here)) {
            model->MUTparamCPU.MUTinstanceIND1Array [j] = here->MUTind1->instanceID;
            model->MUTparamCPU.MUTinstanceIND2Array [j++] = here->MUTind2->instanceID;
        }
    }

    /*  loop through all the mutual inductor models */
    for (model = (MUTmodel *)inModel; model; model = MUTnextModel(model)) {
        status = cuMUTsetup ((GENmodel *)model);
        if (status != 0)
            return (E_NOMEM);
    }
#endif

    return(OK);
}
