/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"

int
CKTmkCurKCL (CKTcircuit *ckt, int i, double **node)
{
    CKTmkCurKCLnode *tempNode ;

    tempNode = TMALLOC (CKTmkCurKCLnode, 1) ;
    tempNode->KCLcurrent = 0.0 ;
    tempNode->next = ckt->CKTmkCurKCLarray [i] ;
    ckt->CKTmkCurKCLarray [i] = tempNode ;
    *node = &(tempNode->KCLcurrent) ;

    return (OK) ;
}
