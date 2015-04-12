/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* load the voltage source structure with those
 * pointers needed later for fast matrix loading
 */

int
ASRCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    ASRCinstance *here;
    ASRCmodel *model = (ASRCmodel*) inModel;
    int error, i, j;
    CKTnode *tmp;

    NG_IGNORE(states);

    for (; model; model = model->ASRCnextModel) {
        for (here = model->ASRCinstances; here; here=here->ASRCnextInstance) {

            if (!here->ASRCtc1Given)
                here->ASRCtc1 = 0.0;
            if (!here->ASRCtc2Given)
                here->ASRCtc2 = 0.0;
            if (!here->ASRCreciproctcGiven)
                here->ASRCreciproctc = 0;

            here->ASRCposptr = NULL;
            j = 0; /* strchr of the array holding ptrs to SMP */

            if (here->ASRCtype == ASRC_VOLTAGE)
                if (here->ASRCbranch == 0) {
                    error = CKTmkCur(ckt, &tmp, here->ASRCname, "branch");
                    if (error)
                        return(error);
                    here->ASRCbranch = tmp->number;
                }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr, first, second)                                    \
            do { if ((here->ptr = SMPmakeElt(matrix, first, second)) == NULL) { \
                    return(E_NOMEM);                                    \
                } } while(0)

#define MY_TSTALLOC(ptr, first, second)                                 \
            do { if ((here->ptr = SMPmakeElt(matrix, here->first, (second)->number)) == NULL) { \
                    return(E_NOMEM);                                    \
                } } while(0)

            /* For each controlling variable set the entries
               in the vector of the positions of the SMP */
            if (!here->ASRCtree)
                return E_PARMVAL;

            if (here->ASRCtype == ASRC_VOLTAGE) {

                if (here->ASRCposNode == here->ASRCnegNode) {
                    SPfrontEnd->IFerrorf(ERR_FATAL, "instance %s is a shorted ASRC", here->ASRCname);
                    return(E_UNSUPP);
                }

                here->ASRCposptr = TREALLOC(double *, here->ASRCposptr, j + 4);

                TSTALLOC(ASRCposptr[j++], here->ASRCposNode, here->ASRCbranch);
                TSTALLOC(ASRCposptr[j++], here->ASRCnegNode, here->ASRCbranch);
                TSTALLOC(ASRCposptr[j++], here->ASRCbranch,  here->ASRCnegNode);
                TSTALLOC(ASRCposptr[j++], here->ASRCbranch,  here->ASRCposNode);
            }

            for (i = 0; i < here->ASRCtree->numVars; i++) {
                switch (here->ASRCtree->varTypes[i]) {
                case IF_INSTANCE:
                    here->ASRCcont_br = CKTfndBranch(ckt, here->ASRCtree->vars[i].uValue);
                    if (here->ASRCcont_br == 0) {
                        SPfrontEnd->IFerrorf(ERR_FATAL, "%s: unknown controlling source %s",
                                             here->ASRCname, here->ASRCtree->vars[i].uValue);
                        return(E_BADPARM);
                    }
                    if (here->ASRCtype == ASRC_VOLTAGE) {
                        /* CCVS */
                        here->ASRCposptr = TREALLOC(double *, here->ASRCposptr, j + 1);
                        TSTALLOC(ASRCposptr[j++], here->ASRCbranch, here->ASRCcont_br);
                    } else if (here->ASRCtype == ASRC_CURRENT) {
                        /* CCCS */
                        here->ASRCposptr = TREALLOC(double *, here->ASRCposptr, j + 2);
                        TSTALLOC(ASRCposptr[j++], here->ASRCposNode, here->ASRCcont_br);
                        TSTALLOC(ASRCposptr[j++], here->ASRCnegNode, here->ASRCcont_br);
                    } else {
                        return (E_BADPARM);
                    }
                    break;
                case IF_NODE:
                    if (here->ASRCtype == ASRC_VOLTAGE) {
                        /* VCVS */
                        here->ASRCposptr = TREALLOC(double *, here->ASRCposptr, j + 1);
                        TSTALLOC(ASRCposptr[j++], here->ASRCbranch, here->ASRCtree->vars[i].nValue->number);
                    } else if (here->ASRCtype == ASRC_CURRENT) {
                        /* VCCS */
                        here->ASRCposptr = TREALLOC(double *, here->ASRCposptr, j + 2);
                        TSTALLOC(ASRCposptr[j++], here->ASRCposNode, here->ASRCtree->vars[i].nValue->number);
                        TSTALLOC(ASRCposptr[j++], here->ASRCnegNode, here->ASRCtree->vars[i].nValue->number);
                    } else {
                        return (E_BADPARM);
                    }
                    break;
                default:
                    return (E_BADPARM);
                }
            }
        }
    }

    return(OK);
}


int
ASRCunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model = (ASRCmodel *) inModel;
    ASRCinstance *here;

    for (; model; model = model->ASRCnextModel)
        for (here = model->ASRCinstances; here; here = here->ASRCnextInstance)
            if (here->ASRCbranch) {
                CKTdltNNum(ckt, here->ASRCbranch);
                here->ASRCbranch = 0;
            }

    return OK;
}
