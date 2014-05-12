/*
 * Copyright (c) 2014, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 *    this list of conditions and the following disclaimer in the documentation and/or 
 *    other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
 *    endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ngspice/config.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "cuda_runtime_api.h"
#include "ngspice/CUSPICE/CUSPICE.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCKTsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCKTsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuCKTsetup
(
CKTcircuit *ckt
)
{
    int i ;
    long unsigned int m, mRHS, n, nz, TopologyNNZ, TopologyNNZRHS, size1, size2, size3 ;
    cudaError_t status ;

    n = (long unsigned int)ckt->CKTmatrix->CKTkluN ;
    nz = (long unsigned int)ckt->CKTmatrix->CKTklunz ;

    m = (long unsigned int)(ckt->total_n_values + 1) ; // + 1 because of CKTdiagGmin

    TopologyNNZ = (long unsigned int)(ckt->total_n_Ptr + ckt->CKTdiagElements) ; // + n because of CKTdiagGmin
                                                                                 // without the zeroes along the diagonal

    mRHS = (long unsigned int)ckt->total_n_valuesRHS ;
    TopologyNNZRHS = (long unsigned int)ckt->total_n_PtrRHS ;

    size1 = (long unsigned int)(ckt->d_MatrixSize + 1) ;
    size2 = (long unsigned int)ckt->CKTnumStates ;
    size3 = (long unsigned int)ckt->total_n_timeSteps ;

    /* Topology Matrix Handling */
    status = cudaMalloc ((void **)&(ckt->CKTmatrix->d_CKTrhs), (n + 1) * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->CKTmatrix->d_CKTrhs, (n + 1), double, status)

    status = cudaMalloc ((void **)&(ckt->CKTmatrix->d_CKTkluAx), nz * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->CKTmatrix->d_CKTkluAx, nz, double, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTloadOutput), m * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTloadOutput, m, double, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTloadOutputRHS), mRHS * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTloadOutputRHS, mRHS, double, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRp), (nz + 1) * sizeof(int)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRp, (nz + 1), int, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRj), TopologyNNZ * sizeof(int)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRj, TopologyNNZ, int, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRx), TopologyNNZ * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRx, TopologyNNZ, double, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRpRHS), ((n + 1) + 1) * sizeof(int)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRpRHS, ((n + 1) + 1), int, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRjRHS), TopologyNNZRHS * sizeof(int)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRjRHS, TopologyNNZRHS, int, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTtopologyMatrixCSRxRHS), TopologyNNZRHS * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtopologyMatrixCSRxRHS, TopologyNNZRHS, double, status)


    cudaMemset (ckt->d_CKTloadOutput + ckt->total_n_values, 0, sizeof(double)) ; //DiagGmin is 0 at the beginning


    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRp, ckt->CKTtopologyMatrixCSRp, (nz + 1) * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRp, (nz + 1), int, status)

    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRj, ckt->CKTtopologyMatrixCOOj, TopologyNNZ * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRj, TopologyNNZ, int, status)

    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRx, ckt->CKTtopologyMatrixCOOx, TopologyNNZ * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRx, TopologyNNZ, double, status)

    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRpRHS, ckt->CKTtopologyMatrixCSRpRHS, ((n + 1) + 1) * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRpRHS, ((n + 1) + 1), int, status)

    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRjRHS, ckt->CKTtopologyMatrixCOOjRHS, TopologyNNZRHS * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRjRHS, TopologyNNZRHS, int, status)

    status = cudaMemcpy (ckt->d_CKTtopologyMatrixCSRxRHS, ckt->CKTtopologyMatrixCOOxRHS, TopologyNNZRHS * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTtopologyMatrixCSRxRHS, TopologyNNZRHS, double, status)
    /* ------------------------ */

    status = cudaMalloc ((void **)&(ckt->d_CKTnoncon), sizeof(int)) ;
    CUDAMALLOCCHECK (ckt->d_CKTnoncon, 1, int, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTrhsOld), size1 * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTrhsOld, size1, double, status)

    for (i = 0 ; i <= MAX (2, ckt->CKTmaxOrder) + 1 ; i++) /* dctran needs 3 states at least */
    {
        status = cudaMalloc ((void **)&(ckt->d_CKTstates[i]), size2 * sizeof(double)) ;
        CUDAMALLOCCHECK (ckt->d_CKTstates[i], size2, double, status)
    }


    /* Truncation Error */
    status = cudaMalloc ((void **)&(ckt->dD_CKTstates), 8 * sizeof(double *)) ;
    CUDAMALLOCCHECK (ckt->dD_CKTstates, 8, double *, status)

    status = cudaMemcpy (ckt->dD_CKTstates, ckt->d_CKTstates, 8 * sizeof(double *), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->dD_CKTstates, 8, double *, status)

    status = cudaMalloc ((void **)&(ckt->d_CKTdeltaOld), 7 * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTdeltaOld, 7, double, status)

//    ckt->CKTtimeSteps = (double *) malloc (size3 * sizeof(double)) ;
    status = cudaMalloc ((void **)&(ckt->d_CKTtimeSteps), size3 * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtimeSteps, size3, double, status)
    status = cudaMalloc ((void **)&(ckt->d_CKTtimeStepsOut), size3 * sizeof(double)) ;
    CUDAMALLOCCHECK (ckt->d_CKTtimeStepsOut, size3, double, status)

    return (OK) ;
}
