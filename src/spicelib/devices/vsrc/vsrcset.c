/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
VSRCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *state)
        /* load the voltage source structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    CKTnode *tmp;
    int error;

    NG_IGNORE(state);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {
            
            here->VSRCbreak_time = -1.0;        // To set initial breakpoint
            if(here->VSRCposNode == here->VSRCnegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted VSRC", here->VSRCname);
                return(E_UNSUPP);
            }

            if(here->VSRCbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->VSRCname,"branch");
                if(error) return(error);
                here->VSRCbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

#ifdef RFSPICE
            if (here->VSRCisPort)
            {
                error = CKTmkVolt(ckt, &tmp, here->VSRCname, "res");
                if (error) return(error);
                here->VSRCresNode = tmp->number;
                if (ckt->CKTcopyNodesets) {
                    CKTnode* tmpNode;
                    IFuid tmpName;
                    if (CKTinst2Node(ckt, here, 1, &tmpNode, &tmpName) == OK) {
                        if (tmpNode->nsGiven) {
                            tmp->nodeset = tmpNode->nodeset;
                            tmp->nsGiven = tmpNode->nsGiven;
                        }
                    }
                }

                TSTALLOC(VSRCposPosPtr, VSRCposNode, VSRCposNode);
                TSTALLOC(VSRCnegNegPtr, VSRCresNode, VSRCresNode);
                TSTALLOC(VSRCposNegPtr, VSRCposNode, VSRCresNode);
                TSTALLOC(VSRCnegPosPtr, VSRCresNode, VSRCposNode);

                TSTALLOC(VSRCposIbrPtr, VSRCresNode, VSRCbranch);
                TSTALLOC(VSRCnegIbrPtr, VSRCnegNode, VSRCbranch);
                TSTALLOC(VSRCibrNegPtr, VSRCbranch, VSRCnegNode);
                TSTALLOC(VSRCibrPosPtr, VSRCbranch, VSRCresNode);
            }
            else
            {
                TSTALLOC(VSRCposIbrPtr, VSRCposNode, VSRCbranch);
                TSTALLOC(VSRCnegIbrPtr, VSRCnegNode, VSRCbranch);
                TSTALLOC(VSRCibrNegPtr, VSRCbranch, VSRCnegNode);
                TSTALLOC(VSRCibrPosPtr, VSRCbranch, VSRCposNode);
            }
#else
            TSTALLOC(VSRCposIbrPtr, VSRCposNode, VSRCbranch);
            TSTALLOC(VSRCnegIbrPtr, VSRCnegNode, VSRCbranch);
            TSTALLOC(VSRCibrNegPtr, VSRCbranch, VSRCnegNode);
            TSTALLOC(VSRCibrPosPtr, VSRCbranch, VSRCposNode);
#endif

        }
    }
    return(OK);
}

int
VSRCunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model;
    VSRCinstance *here;

    for (model = (VSRCmodel *)inModel; model != NULL;
	    model = VSRCnextModel(model))
    {
        for (here = VSRCinstances(model); here != NULL;
                here=VSRCnextInstance(here))
	{
	    if (here->VSRCbranch > 0)
		CKTdltNNum(ckt, here->VSRCbranch);
            here->VSRCbranch = 0;
#ifdef RFSPICE
            if ((here->VSRCresNode > 0) & (here->VSRCisPort))
                CKTdltNNum(ckt, here->VSRCresNode);
            here->VSRCresNode = 0;

#endif
	}
    }
    return OK;
}
