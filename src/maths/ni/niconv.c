/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * NIconvTest(ckt)
 *  perform the convergence test - returns 1 if any of the 
 *  values in the old and new arrays have changed by more 
 *  than absTol + relTol*(max(old,new)), otherwise returns 0
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"


int
NIconvTest(CKTcircuit *ckt)
{
    int i; /* generic loop variable */
    int size;  /* size of the matrix */
    CKTnode *node; /* current matrix entry */
    double old;
    double new;
    double tol;

    size = SMPmatSize (ckt->CKTmatrix) ;

#ifdef STEPDEBUG
    for (i=1;i<=size;i++) {
        new =  ckt->CKTrhs [i] ;
        old =  ckt->CKTrhsOld [i] ;
	printf("chk for convergence:   %s    new: %g    old: %g\n",CKTnodName(ckt,i),new,old);
    }
#endif /* STEPDEBUG */

    node = ckt->CKTnodes ;
    for (i=1;i<=size;i++) {
        node = node->next;
        new =  ckt->CKTrhs [i] ;
        old =  ckt->CKTrhsOld [i] ;
        if(node->type == SP_VOLTAGE) {
            tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                    ckt->CKTvoltTol;
            if (fabs(new-old) >tol ) {
#ifdef STEPDEBUG
                printf(" non-convergence at node (type=3) %s (fabs(new-old)>tol --> fabs(%g-%g)>%g)\n",CKTnodName(ckt,i),new,old,tol);
		printf("    reltol: %g    voltTol: %g   (tol=reltol*(MAX(fabs(old),fabs(new))) + voltTol)\n",ckt->CKTreltol,ckt->CKTvoltTol);
#endif /* STEPDEBUG */
		ckt->CKTtroubleNode = i;
		ckt->CKTtroubleElt = NULL;
                return(1);
            }
        }
    }

    /* KCL Verification */
    double maximum = 0 ;

    node = ckt->CKTnodes ;
    for (i = 1 ; i <= size ; i++)
    {
        node = node->next ;
        if (node->type == SP_CURRENT)
        {
            if (maximum < fabs (ckt->CKTrhsOld [i]))
                maximum = fabs (ckt->CKTrhsOld [i]) ;
        }
    }

    node = ckt->CKTnodes ;
    for (i = 1 ; i <= size ; i++)
    {
        node = node->next ;
        if (node->type == SP_VOLTAGE)
        {
//            printf ("Valore: %-.9g\tSoglia: %-.9g\n", fabs (ckt->CKTfvk [i] + ckt->CKTdiagGmin * ckt->CKTrhsOld [i]), (ckt->CKTreltol * maximum + ckt->CKTabstol)) ;
//            if (fabs (ckt->CKTfvk [i]) > (ckt->CKTreltol * maximum + ckt->CKTabstol))
            if (fabs (ckt->CKTfvk [i] + ckt->CKTdiagGmin * ckt->CKTrhsOld [i]) > (ckt->CKTreltol * maximum + ckt->CKTabstol))
                return 1 ;
        }
    }

#ifdef NEWCONV
    i = CKTconvTest(ckt);
    if (i)
	ckt->CKTtroubleNode = 0;
    return(i);
#else /* NEWCONV */
    return(0);
#endif /* NEWCONV */
}
