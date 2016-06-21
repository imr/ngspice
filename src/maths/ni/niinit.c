/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */
    /*
     * NIinit(nistruct,loadpkg)
     *
     *  Initialize the Numerical iteration package to perform Newton-Raphson
     *  iterations on a sparse matrix filled by the specified load package,
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/smpdefs.h"

#ifdef KLU
#include "ngspice/klu.h"
#endif

int
NIinit(CKTcircuit *ckt)
{
#ifdef SPARSE
/* a concession to Ken Kundert's sparse matrix package - SMP doesn't need this*/
    int Error;
#endif /* SPARSE */

/* Allocation of the new SMPmatrix structure - Francesco Lannutti (2012-02) */
    ckt->CKTmatrix = TMALLOC (SMPmatrix, 1) ;

#ifdef KLU
    ckt->CKTmatrix->CKTkluCommon = TMALLOC (klu_common, 1) ;
    ckt->CKTmatrix->CKTkluSymbolic = NULL ;
    ckt->CKTmatrix->CKTkluNumeric = NULL ;
    ckt->CKTmatrix->CKTkluAp = NULL ;
    ckt->CKTmatrix->CKTkluAi = NULL ;
    ckt->CKTmatrix->CKTkluAx = NULL ;
    ckt->CKTmatrix->CKTkluMatrixIsComplex = CKTkluMatrixReal ;
    ckt->CKTmatrix->CKTkluIntermediate = NULL ;
    ckt->CKTmatrix->CKTkluIntermediate_Complex = NULL ;
    ckt->CKTmatrix->CKTbindStruct = NULL ;
    ckt->CKTmatrix->CKTdiag_CSC = NULL ;
    ckt->CKTmatrix->CKTkluN = 0 ;
    ckt->CKTmatrix->CKTklunz = 0 ;
    ckt->CKTmatrix->CKTkluMODE = ckt->CKTkluMODE ; /* TO BE SUBSTITUTED WITH THE HEURISTICS */

    klu_defaults (ckt->CKTmatrix->CKTkluCommon) ;
#endif

    ckt->CKTniState = NIUNINITIALIZED;
    return SMPnewMatrix(ckt->CKTmatrix, 0);
}
