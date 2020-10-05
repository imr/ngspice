/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* load the voltage source structure with those pointers needed later 
 * for fast matrix loading 
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#include "ngspice/devdefs.h"

/*ARGSUSED*/
int
NULAsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    NULAmodel *model = (NULAmodel *)inModel;
    NULAinstance *here;
    GENinstance *nora;
    GENmodel *noramod;
    
    int i, error;
    CKTnode *tmp;

    NG_IGNORE(states);
    
    noramod = NULL;
    nora = NULL;
    for (i=0;i<DEVmaxnum;i++)
    if (DEVices[i])
    if (strcmp(DEVices[i]->DEVpublic.name,"Norator")==0) {
        noramod = ckt->CKThead[i];
	break;
    }
    if(i==DEVmaxnum)
    if(model && NULAinstances(model)) /* emit warning only if circuit actualy has a nullator */
        SPfrontEnd->IFerrorf (ERR_WARNING,
                        "Norator device needed for nullator, missing.");
    if(noramod)
	nora = noramod->GENinstances;
    
    /*  loop through all the nullator models */
    for( ; model != NULL; model = NULAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NULAinstances(model); here != NULL ;
                here=NULAnextInstance(here)) {
            
            if(here->NULAcontPosNode == here->NULAcontNegNode) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "instance %s is a shorted nullator", here->NULAname);
                return(E_UNSUPP);
            }
	    
	    if(!here->NULAoffsetGiven)
	        here->NULAoffset = 0.0;
	    
            if(here->NULAbranch == 0) {
	        /* assign a branch from a norator device ... */
		if(nora) {
		  here->NULAbranch = DEVices[i]->DEVfindBranch(ckt, noramod, nora->GENname);
		} else {
		  SPfrontEnd->IFerrorf (ERR_FATAL,
                        "excess nullators in respect to number of norator");
		  return(E_UNSUPP);
		}
		
		/* point to next norator instance */
		nora = nora->GENnextInstance;
		if(!nora) {
		   noramod = noramod->GENnextModel;
		   if(noramod)
		      nora = noramod->GENinstances;
		}
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)
            if(here->NULAbranch>0) {
                TSTALLOC(NULAibrContPosPtr, NULAbranch, NULAcontPosNode);
                TSTALLOC(NULAibrContNegPtr, NULAbranch, NULAcontNegNode);
	    } else {
                  SPfrontEnd->IFerrorf (ERR_FATAL,
                        "failed to find a norator branch for nullator %s", here->NULAname);
		  return(E_UNSUPP);
            }
        }
    }
    return(OK);
}

int
NULAunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    NULAmodel *model;
    NULAinstance *here;

    for (model = (NULAmodel *)inModel; model != NULL;
	    model = NULAnextModel(model))
    {
        for (here = NULAinstances(model); here != NULL;
                here=NULAnextInstance(here))
	{
            here->NULAbranch = 0;
	}
    }
    return OK;
}
