/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTload(ckt)
     * this is a driver program to iterate through all the various
     * load functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"


static int ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum);

int
CKTload(register CKTcircuit *ckt)
{
    extern SPICEdev *DEVices[];
    register int i;
    register int size;
    double startTime;
    CKTnode *node;
    int error;
#ifdef PARALLEL_ARCH
    int ibuf[2];
    long type = MT_LOAD, length = 2;
#endif /* PARALLEL_ARCH */
#ifdef STEPDEBUG
    int noncon;
#endif /* STEPDEBUG */

    startTime = (*(SPfrontEnd->IFseconds))();
    size = SMPmatSize(ckt->CKTmatrix);
    for (i=0;i<=size;i++) {
        *(ckt->CKTrhs+i)=0;
    }
    SMPclear(ckt->CKTmatrix);
#ifdef STEPDEBUG
    noncon = ckt->CKTnoncon;
#endif /* STEPDEBUG */

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVload != NULL) && (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVload))(ckt->CKThead[i],ckt);
	    if (ckt->CKTnoncon)
		ckt->CKTtroubleNode = 0;
#ifdef STEPDEBUG
            if(noncon != ckt->CKTnoncon) {
                printf("device type %s nonconvergence\n",
                        (*DEVices[i]).DEVpublic.name);
                noncon = ckt->CKTnoncon;
            }
#endif /* STEPDEBUG */
#ifdef PARALLEL_ARCH
	    if (error) goto combine;
#else
            if(error) return(error);
#endif /* PARALLEL_ARCH */
        }
    }
    if(ckt->CKTmode & MODEDC) {
        /* consider doing nodeset & ic assignments */
        if(ckt->CKTmode & (MODEINITJCT | MODEINITFIX)) {
            /* do nodesets */
            for(node=ckt->CKTnodes;node;node=node->next) {
                if(node->nsGiven) {
		    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
			node->number))
		    {
			*(ckt->CKTrhs+node->number) = 1.0e10 * node->nodeset;
			*(node->ptr) = 1e10;
		    } else {
			*(ckt->CKTrhs+node->number) = node->nodeset;
			*(node->ptr) = 1;
		    }
		    /* DAG: Original CIDER fix. If above fix doesn't work,
		     * revert to this.
		     */
		    /*
                    *(ckt->CKTrhs+node->number) += 1.0e10 * node->nodeset;
                    *(node->ptr) += 1.0e10;
		    */
		}
            }
        }
        if( (ckt->CKTmode & MODETRANOP) && (!(ckt->CKTmode & MODEUIC))) {
            for(node=ckt->CKTnodes;node;node=node->next) {
                if(node->icGiven) {
		    if (ZeroNoncurRow(ckt->CKTmatrix, ckt->CKTnodes,
			node->number))
		    {
			*(ckt->CKTrhs+node->number) += 1.0e10 * node->ic;
			*(node->ptr) += 1.0e10;
		    } else {
			*(ckt->CKTrhs+node->number) = node->ic;
			*(node->ptr) = 1;
		    }
		    /* DAG: Original CIDER fix. If above fix doesn't work,
		     * revert to this.
		     */
		    /*
                    *(ckt->CKTrhs+node->number) += 1.0e10 * node->ic;
                    *(node->ptr) += 1.0e10;
		    */
                }
            }
        }
    }
    /* SMPprint(ckt->CKTmatrix, stdout); if you want to debug, this is a
	good place to start ... */

#ifdef PARALLEL_ARCH
combine:
    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds() - startTime;
    startTime  = SPfrontEnd->IFseconds();
    /* See if any of the DEVload functions bailed. If not, proceed. */
    ibuf[0] = error;
    ibuf[1] = ckt->CKTnoncon;
    IGOP_( &type, ibuf, &length, "+" );
    ckt->CKTnoncon = ibuf[1];
    ckt->CKTstat->STATsyncTime += SPfrontEnd->IFseconds() - startTime;
    if (ibuf[0] == OK) {
      startTime  = SPfrontEnd->IFseconds();
      SMPcombine( ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare );
      ckt->CKTstat->STATcombineTime += SPfrontEnd->IFseconds() - startTime;
      return(OK);
    } else {
      if ( ibuf[0] != error ) {
	error = E_MULTIERR;
      }
      return(error);
    }
#else
    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds()-startTime;
    return(OK);
#endif /* PARALLEL_ARCH */
}

static int
ZeroNoncurRow(SMPmatrix *matrix, CKTnode *nodes, int rownum)
{
	CKTnode		*n;
	double		*x;
	int		currents;

	currents = 0;
	for (n = nodes; n; n = n->next) {
 	    x = (double *) SMPfindElt(matrix, rownum, n->number, 0);
 	    if (x) {
 		if (n->type == SP_CURRENT)
 		    currents = 1;
 		else
 		    *x = 0.0;
 	    }
	}
	return currents;
}
