/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
**********/

/* CKTdltNod
*/

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"

int CKTdltNNum(CKTcircuit *cktp, int num);

/* ARGSUSED */
int
CKTdltNod(CKTcircuit *ckt, void *node)
{
    return CKTdltNNum(ckt, ((CKTnode *) node)->number);
}

int
CKTdltNNum(CKTcircuit *cktp, int num)
{
    CKTcircuit *ckt = /* fixme, drop that */  cktp;
    CKTnode *n, *prev, *node, *sprev;
    int	error;

    prev  = NULL;
    node  = NULL;
    sprev = NULL;

    for (n = ckt->CKTnodes; n; n = n->next) {
	if (n->number == num) {
	    node = n;
	    sprev = prev;
	}
	prev = n;
    }

    if (!node)
	return OK;

    ckt->CKTmaxEqNum -= 1;

    if (!sprev) {
	ckt->CKTnodes = node->next;
    } else {
	sprev->next = node->next;
    }
    if (node == ckt->CKTlastNode)
	ckt->CKTlastNode = sprev;

    error = (*(SPfrontEnd->IFdelUid))(ckt,node->name, UID_SIGNAL);
    tfree(node);

    return error;
}
