/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
**********/

/* CKTdltNod
*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"

/* ARGSUSED */
int
CKTdltNod(CKTcircuit* ckt, CKTnode* node)
{
    return CKTdltNNum(ckt, node->number);
}

int
CKTdltNNum(CKTcircuit* ckt, int num)
{
    CKTnode* n, * prev, * node;
    int	error;

    if (!ckt->prev_CKTlastNode->number || num <= ckt->prev_CKTlastNode->number) {
        fprintf(stderr, "Internal Error: CKTdltNNum() removing a non device-local node, this will cause serious problems, please report this issue !\n");
        controlled_exit(EXIT_FAILURE);
    }

    prev = NULL;
    node = NULL;

    for (n = ckt->CKTnodes; n; n = n->next) {
        if (n->number == num) {
            node = n;
            break;
        }
        prev = n;
    }

    if (!node)
        return OK;

    ckt->CKTmaxEqNum -= 1;

    if (!prev) {
        ckt->CKTnodes = node->next;
    }
    else {
        prev->next = node->next;
    }
    if (node == ckt->CKTlastNode)
        ckt->CKTlastNode = prev;

    error = SPfrontEnd->IFdelUid(ckt, node->name, UID_SIGNAL);
    tfree(node);

    return error;
}
