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

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCKTstatesUpdate routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuCKTstatesUpdateDtoH
(
CKTcircuit *ckt
)
{
    int i ;
    long unsigned int size ;
    cudaError_t status ;

    size = (long unsigned int)ckt->CKTnumStates ;

    for (i = 0 ; i < 8 ; i++)
    {
        if (ckt->CKTstates[i] != NULL)
        {
            status = cudaMemcpy (ckt->CKTstates[i], ckt->d_CKTstates[i], size * sizeof(double), cudaMemcpyDeviceToHost) ;
            CUDAMEMCPYCHECK (ckt->CKTstates[i], size, double, status)
        }
    }

    return (OK) ;
}

int
cuCKTstatesFlush
(
CKTcircuit *ckt
)
{
    long unsigned int size ;

    size = (long unsigned int)ckt->CKTnumStates ;
    cudaMemset (ckt->d_CKTstate0, 0, size * sizeof(double)) ;

    return (OK) ;
}

int
cuCKTstate0UpdateHtoD
(
CKTcircuit *ckt
)
{
    long unsigned int size ;
    cudaError_t status ;

    size = (long unsigned int)ckt->CKTnumStates ;
    status = cudaMemcpy (ckt->d_CKTstate0, ckt->CKTstate0, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTstate0, size, double, status)

    return (OK) ;
}

int
cuCKTstate0UpdateDtoH
(
CKTcircuit *ckt
)
{
    long unsigned int size ;
    cudaError_t status ;

    size = (long unsigned int)ckt->CKTnumStates ;
    status = cudaMemcpy (ckt->CKTstate0, ckt->d_CKTstate0, size * sizeof(double), cudaMemcpyDeviceToHost) ;
    CUDAMEMCPYCHECK (ckt->CKTstate0, size, double, status)

    return (OK) ;
}

int
cuCKTstate01copy
(
CKTcircuit *ckt
)
{
    long unsigned int size ;
    cudaError_t status ;

    size = (long unsigned int)ckt->CKTnumStates ;
    status = cudaMemcpy (ckt->d_CKTstate1, ckt->d_CKTstate0, size * sizeof(double), cudaMemcpyDeviceToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTstate1, size, double, status)

    return (OK) ;
}

int
cuCKTstatesCircularBuffer
(
CKTcircuit *ckt
)
{
    int i ;
    double *temp ;

    temp = ckt->d_CKTstates [ckt->CKTmaxOrder + 1] ;
    for (i = ckt->CKTmaxOrder ; i >= 0 ; i--)
        ckt->d_CKTstates [i + 1] = ckt->d_CKTstates [i] ;

    ckt->d_CKTstates [0] = temp ;

    return (OK) ;
}

int
cuCKTstate123copy
(
CKTcircuit *ckt
)
{
    long unsigned int size ;
    cudaError_t status ;

    size = (long unsigned int)ckt->CKTnumStates ;

    status = cudaMemcpy (ckt->d_CKTstate2, ckt->d_CKTstate1, size * sizeof(double), cudaMemcpyDeviceToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTstate2, size, double, status)

    status = cudaMemcpy (ckt->d_CKTstate3, ckt->d_CKTstate1, size * sizeof(double), cudaMemcpyDeviceToDevice) ;
    CUDAMEMCPYCHECK (ckt->d_CKTstate3, size, double, status)

    return (OK) ;
}
