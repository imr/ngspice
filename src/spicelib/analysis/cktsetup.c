/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTsetup(ckt)
     * this is a driver program to iterate through all the various
     * setup functions provided for the circuit elements in the
     * given circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/enh.h"
#endif

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#include "cusparse_v2.h"
#endif

#ifdef USE_OMP
#include <omp.h>
#include "ngspice/cpextern.h"
int nthreads;
#endif

#define CKALLOC(var,size,type) \
    if(size && ((var = TMALLOC(type, size)) == NULL)){\
            return(E_NOMEM);\
}

#ifdef KLU
#include <stdlib.h>

static
int
BindCompare (const void *a, const void *b)
{
    BindElement *A, *B ;
    A = (BindElement *)a ;
    B = (BindElement *)b ;

    return ((int)(A->Sparse - B->Sparse)) ;
}
#endif

#ifdef USE_CUSPICE
typedef struct sElement {
    int row ;
    int col ;
    double val ;
} Element ;

static
int
Compare (const void *a, const void *b)
{
    Element *A, *B ;
    A = (Element *)a ;
    B = (Element *)b ;
    return (A->row - B->row) ;
}

static
int
Compress (int *Ai, int *Bp, int num_rows, int n_COO)
{
    int i, j ;

    for (i = 0 ; i <= Ai [0] ; i++)
        Bp [i] = 0 ;

    j = Ai [0] + 1 ;
    for (i = 1 ; i < n_COO ; i++)
    {
        if (Ai [i] == Ai [i - 1] + 1)
        {
            Bp [j] = i ;
            j++ ;
        }
        else if (Ai [i] > Ai [i - 1] + 1)
        {
            for ( ; j <= Ai [i] ; j++)
                Bp [j] = i ;
        }
    }

    for ( ; j <= num_rows ; j++)
        Bp [j] = i ;

    return 0 ;
}
#endif

int
CKTsetup(CKTcircuit *ckt)
{
#ifdef USE_CUSPICE
    int status ;
    cusparseStatus_t cusparseStatus ;

    ckt->total_n_values = 0 ;
    ckt->total_n_Ptr = 0 ;

    ckt->total_n_valuesRHS = 0 ;
    ckt->total_n_PtrRHS = 0 ;

    ckt->total_n_timeSteps = 0 ;
#endif

    int i;
    int error;
#ifdef XSPICE
 /* gtri - begin - Setup for adding rshunt option resistors */
    CKTnode *node;
    int     num_nodes;
 /* gtri - end - Setup for adding rshunt option resistors */
#endif
    SMPmatrix *matrix;
    ckt->CKTnumStates=0;

#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        error = CKTsenSetup(ckt);
        if (error)
            return(error);
    }
#endif

    if (ckt->CKTisSetup)
        return E_NOCHANGE;

    error = NIinit(ckt);
    if (error) return(error);
    ckt->CKTisSetup = 1;

    matrix = ckt->CKTmatrix;

#ifdef USE_OMP
    if (!cp_getvar("num_threads", CP_NUM, &nthreads))
        nthreads = 2;

    omp_set_num_threads(nthreads);
/*    if (nthreads == 1)
      printf("OpenMP: %d thread is requested in ngspice\n", nthreads);
    else
      printf("OpenMP: %d threads are requested in ngspice\n", nthreads);*/
#endif

#ifdef HAS_PROGREP
    SetAnalyse("Device Setup", 0);
#endif

    /* preserve CKTlastNode before invoking DEVsetup()
     * so we can check for incomplete CKTdltNNum() invocations
     * during DEVunsetup() causing an erronous circuit matrix
     *   when reinvoking CKTsetup()
     */
    ckt->prev_CKTlastNode = ckt->CKTlastNode;

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVsetup && ckt->CKThead[i] ) {
            error = DEVices[i]->DEVsetup (matrix, ckt->CKThead[i], ckt,
                    &ckt->CKTnumStates);
            if(error) return(error);
        }
    }

#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE)
    {
        fprintf (stderr, "Using KLU as Direct Linear Solver\n") ;

        int i ;
        int n = SMPmatSize (ckt->CKTmatrix) ;
        ckt->CKTmatrix->CKTkluN = n ;

        SMPnnz (ckt->CKTmatrix) ;
        int nz = ckt->CKTmatrix->CKTklunz ;

        ckt->CKTmatrix->CKTkluAp           = TMALLOC (int, n + 1) ;
        ckt->CKTmatrix->CKTkluAi           = TMALLOC (int, nz) ;
        ckt->CKTmatrix->CKTkluAx           = TMALLOC (double, nz) ;
        ckt->CKTmatrix->CKTkluIntermediate = TMALLOC (double, n) ;

        ckt->CKTmatrix->CKTbindStruct      = TMALLOC (BindElement, nz) ;

        ckt->CKTmatrix->CKTdiag_CSC        = TMALLOC (double *, n) ;

        /* Complex Stuff needed for AC Analysis */
        ckt->CKTmatrix->CKTkluAx_Complex = TMALLOC (double, 2 * nz) ;
        ckt->CKTmatrix->CKTkluIntermediate_Complex = TMALLOC (double, 2 * n) ;

        /* Binding Table from Sparse to CSC Format Creation */
        SMPmatrix_CSC (ckt->CKTmatrix) ;

        /* Binding Table Sorting */
        qsort (ckt->CKTmatrix->CKTbindStruct, (size_t)nz, sizeof(BindElement), BindCompare) ;

        /* KLU Pointers Assignment */
        for (i = 0 ; i < DEVmaxnum ; i++)
            if (DEVices [i] && DEVices [i]->DEVbindCSC && ckt->CKThead [i])
                DEVices [i]->DEVbindCSC (ckt->CKThead [i], ckt) ;

        ckt->CKTmatrix->CKTkluMatrixIsComplex = CKTkluMatrixReal ;

#ifdef USE_CUSPICE
        fprintf (stderr, "Using CUSPICE (NGSPICE on CUDA Platforms)\n") ;

        /* In the DEVsetup the Position Vectors must be assigned and copied to the GPU */
        int j, k, u, TopologyNNZ ;
        int uRHS, TopologyNNZRHS ;
        int ret ;


        /* CKTloadOutput Vector allocation - DIRECTLY in the GPU memory */

        /* CKTloadOutput for the RHS Vector allocation - DIRECTLY in the GPU memory */


        /* Diagonal Elements Counting */
        j = 0 ;
        for (i = 0 ; i < n ; i++)
            if (ckt->CKTmatrix->CKTdiag_CSC [i] != NULL)
                j++ ;

        ckt->CKTdiagElements = j ;

        /* Topology Matrix Pre-Allocation in COO format */
        TopologyNNZ = ckt->total_n_Ptr + ckt->CKTdiagElements ; // + ckt->CKTdiagElements because of CKTdiagGmin
                                                                // without the zeroes along the diagonal
        ckt->CKTtopologyMatrixCOOi = TMALLOC (int, TopologyNNZ) ;
        ckt->CKTtopologyMatrixCOOj = TMALLOC (int, TopologyNNZ) ;
        ckt->CKTtopologyMatrixCOOx = TMALLOC (double, TopologyNNZ) ;

        /* Topology Matrix for the RHS Pre-Allocation in COO format */
        TopologyNNZRHS = ckt->total_n_PtrRHS ;
        ckt->CKTtopologyMatrixCOOiRHS = TMALLOC (int, TopologyNNZRHS) ;
        ckt->CKTtopologyMatrixCOOjRHS = TMALLOC (int, TopologyNNZRHS) ;
        ckt->CKTtopologyMatrixCOOxRHS = TMALLOC (double, TopologyNNZRHS) ;


        /* Topology Matrix Pre-Allocation in CSR format */
        ckt->CKTtopologyMatrixCSRp = TMALLOC (int, nz + 1) ;

        /* Topology Matrix for the RHS Pre-Allocation in CSR format */
        ckt->CKTtopologyMatrixCSRpRHS = TMALLOC (int, (n + 1) + 1) ;


        /* Topology Matrix Construction & Topology Matrix for the RHS Construction */

        u = 0 ;
        uRHS = 0 ;
        for (i = 0 ; i < DEVmaxnum ; i++)
            if (DEVices [i] && DEVices [i]->DEVtopology && ckt->CKThead [i])
                DEVices [i]->DEVtopology (ckt->CKThead [i], ckt, &u, &uRHS) ;


        /* CKTdiagGmin Contribute Addition to the Topology Matrix */
        k = u ;
        for (j = 0 ; j < n ; j++)
        {
            if (ckt->CKTmatrix->CKTdiag_CSC [j] >= ckt->CKTmatrix->CKTkluAx)
            {
                ckt->CKTtopologyMatrixCOOi [k] = (int)(ckt->CKTmatrix->CKTdiag_CSC [j] - ckt->CKTmatrix->CKTkluAx) ;
                ckt->CKTtopologyMatrixCOOj [k] = ckt->total_n_values ;
                ckt->CKTtopologyMatrixCOOx [k] = 1 ;
                k++ ;
            }
        }

        /* Copy the Topology Matrix to the GPU in COO format */


        /* COO format to CSR format Conversion using Quick Sort */

        Element *TopologyStruct ;
        TopologyStruct = TMALLOC (Element, TopologyNNZ) ;

        for (i = 0 ; i < TopologyNNZ ; i++)
        {
            TopologyStruct [i].row = ckt->CKTtopologyMatrixCOOi [i] ;
            TopologyStruct [i].col = ckt->CKTtopologyMatrixCOOj [i] ;
            TopologyStruct [i].val = ckt->CKTtopologyMatrixCOOx [i] ;
        }

        qsort (TopologyStruct, (size_t)TopologyNNZ, sizeof(Element), Compare) ;

        for (i = 0 ; i < TopologyNNZ ; i++)
        {
            ckt->CKTtopologyMatrixCOOi [i] = TopologyStruct [i].row ;
            ckt->CKTtopologyMatrixCOOj [i] = TopologyStruct [i].col ;
            ckt->CKTtopologyMatrixCOOx [i] = TopologyStruct [i].val ;
        }

        ret = Compress (ckt->CKTtopologyMatrixCOOi, ckt->CKTtopologyMatrixCSRp, nz, TopologyNNZ) ;

        /* COO format to CSR format Conversion for the RHS using Quick Sort */

        Element *TopologyStructRHS ;
        TopologyStructRHS = TMALLOC (Element, TopologyNNZRHS) ;

        for (i = 0 ; i < TopologyNNZRHS ; i++)
        {
            TopologyStructRHS [i].row = ckt->CKTtopologyMatrixCOOiRHS [i] ;
            TopologyStructRHS [i].col = ckt->CKTtopologyMatrixCOOjRHS [i] ;
            TopologyStructRHS [i].val = ckt->CKTtopologyMatrixCOOxRHS [i] ;
        }

        qsort (TopologyStructRHS, (size_t)TopologyNNZRHS, sizeof(Element), Compare) ;

        for (i = 0 ; i < TopologyNNZRHS ; i++)
        {
            ckt->CKTtopologyMatrixCOOiRHS [i] = TopologyStructRHS [i].row ;
            ckt->CKTtopologyMatrixCOOjRHS [i] = TopologyStructRHS [i].col ;
            ckt->CKTtopologyMatrixCOOxRHS [i] = TopologyStructRHS [i].val ;
        }

        ret = Compress (ckt->CKTtopologyMatrixCOOiRHS, ckt->CKTtopologyMatrixCSRpRHS, n + 1, TopologyNNZRHS) ;

        /* Multiply the Topology Matrix by the M Vector to build the Final CSC Matrix - after the CKTload Call */
#endif

    } else {
        fprintf (stderr, "Using SPARSE 1.3 as Direct Linear Solver\n") ;
    }
#endif

    for(i=0;i<=MAX(2,ckt->CKTmaxOrder)+1;i++) { /* dctran needs 3 states as minimum */
        CKALLOC(ckt->CKTstates[i],ckt->CKTnumStates,double);
    }

#ifdef USE_CUSPICE
        ckt->d_MatrixSize = SMPmatSize (ckt->CKTmatrix) ;
        status = cuCKTsetup (ckt) ;
        if (status != 0)
            return (E_NOMEM) ;

        /* CUSPARSE Handle Creation */
        cusparseStatus = cusparseCreate ((cusparseHandle_t *)(&(ckt->CKTmatrix->CKTcsrmvHandle))) ;
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            fprintf (stderr, "CUSPARSE Handle Setup Error\n") ;
            return (E_NOMEM) ;
        }

        /* CUSPARSE Matrix Descriptor Creation */
        cusparseStatus = cusparseCreateMatDescr ((cusparseMatDescr_t *)(&(ckt->CKTmatrix->CKTcsrmvDescr))) ;
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        {
            fprintf (stderr, "CUSPARSE Matrix Descriptor Setup Error\n") ;
            return (E_NOMEM) ;
        }

        /* CUSPARSE Matrix Properties Definition */
        cusparseSetMatType ((cusparseMatDescr_t)(ckt->CKTmatrix->CKTcsrmvDescr), CUSPARSE_MATRIX_TYPE_GENERAL) ;
        cusparseSetMatIndexBase ((cusparseMatDescr_t)(ckt->CKTmatrix->CKTcsrmvDescr), CUSPARSE_INDEX_BASE_ZERO) ;
#endif

#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        /* to allocate memory to sensitivity structures if
         * it is not done before */

        error = NIsenReinit(ckt);
        if(error) return(error);
    }
#endif
    if(ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt);
        if(error) return(error);
    }
#ifdef XSPICE
  /* gtri - begin - Setup for adding rshunt option resistors */

    if(ckt->enh->rshunt_data.enabled) {

        /* Count number of voltage nodes in circuit */
        for(num_nodes = 0, node = ckt->CKTnodes; node; node = node->next)
            if((node->type == SP_VOLTAGE) && (node->number != 0))
                num_nodes++;

        /* Allocate space for the matrix diagonal data */
        if(num_nodes > 0) {
            ckt->enh->rshunt_data.diag =
                 TMALLOC(double *, num_nodes);
        }

        /* Set the number of nodes in the rshunt data */
        ckt->enh->rshunt_data.num_nodes = num_nodes;

        /* Get/create matrix diagonal entry following what RESsetup does */
        for(i = 0, node = ckt->CKTnodes; node; node = node->next) {
            if((node->type == SP_VOLTAGE) && (node->number != 0)) {
                ckt->enh->rshunt_data.diag[i] =
                      SMPmakeElt(matrix,node->number,node->number);
                i++;
            }
        }

    }

    /* gtri - end - Setup for adding rshunt option resistors */
#endif
    return(OK);
}

int
CKTunsetup(CKTcircuit *ckt)
{
    int i, error, e2;
    CKTnode *node;

    error = OK;
    if (!ckt->CKTisSetup)
        return OK;

    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
        tfree(ckt->CKTstates[i]);
    }

    /* added by HT 050802*/
    for(node=ckt->CKTnodes;node;node=node->next){
        if(node->icGiven || node->nsGiven) {
            node->ptr=NULL;
        }
    }

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVunsetup && ckt->CKThead[i] ) {
            e2 = DEVices[i]->DEVunsetup (ckt->CKThead[i], ckt);
            if (!error && e2)
                error = e2;
        }
    }

    if (ckt->prev_CKTlastNode != ckt->CKTlastNode) {
        fprintf(stderr, "Internal Error: incomplete CKTunsetup(), this will cause serious problems, please report this issue !\n");
        controlled_exit(EXIT_FAILURE);
    }
    ckt->prev_CKTlastNode = NULL;

    ckt->CKTisSetup = 0;
    if(error) return(error);

    NIdestroy(ckt);
    /*
    if (ckt->CKTmatrix)
        SMPdestroy(ckt->CKTmatrix);
    ckt->CKTmatrix = NULL;
    */

    return OK;
}
