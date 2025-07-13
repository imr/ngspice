/* tline_common.c
 * common definitions for all transmission lines
 */

/* ===========================================================================
 FILE   tline_common.c
 Copyright 2025 Vadim Kuznetsov

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#include <stdlib.h>
#include <string.h>

#include "tline_common.h"

void append_state(tline_state_t **first, double time, double V1, double V2,
		double I1, double I2, double tmax)
{
    tline_state_t *pp = (tline_state_t *) malloc(sizeof(tline_state_t));

    pp->next = NULL;
    pp->time =  time;
    pp->V1 = V1; pp->I1 = I1;
    pp->V2 = V2; pp->I2 = I2;

    if (*first == NULL) {
        *first = pp;
    } else {
        tline_state_t *pn  = *first;
        while (pn->next != NULL) {
            pn = pn->next;
        }
        pn->next = pp;

		double t0 = (*first)->time;

		if ((time - t0) > tmax) {
			tline_state_t *new_first = (*first)->next;
			free(*first);
			*first = new_first;
		}
    }
}

tline_state_t *get_state(tline_state_t *first, double time)
{
    tline_state_t *pp = first;
    while (pp != NULL && pp->time < time) {
        pp = pp->next;
    }
    return pp;
}

tline_state_t *get_tline_last_state(tline_state_t *first)
{
    tline_state_t *pp = first;
	if (first == NULL) return NULL;
    while (pp->next != NULL) {
        pp = pp->next;
    }
    return pp;
}


void delete_tline_last_state(tline_state_t **first)
{
	tline_state_t *pn  = *first;
	if (*first == NULL) return;

	if ((*first)->next == NULL) {
		free (*first);
		*first = NULL;
		return;
	}

	while (pn->next->next != NULL) {
		pn = pn->next;
	}
	free(pn->next);
	pn->next = NULL;
}

void delete_tline_states(tline_state_t **first)
{
	if (*first == NULL) return;
	tline_state_t *pn;
	tline_state_t *pc = *first;
	while (pc != NULL) {
		pn = pc->next;
		free (pc);
		pc = pn;
	}
	*first = NULL;
}

void append_cpline_state(cpline_state_t **first, double time, double *Vp, double *Ip, double tmax)
{
    cpline_state_t *pp = (cpline_state_t *) malloc(sizeof(cpline_state_t));

    pp->next = NULL;
    pp->time =  time;
    memcpy(pp->Vp, Vp, PORT_NUM*sizeof(double));
    memcpy(pp->Ip, Ip, PORT_NUM*sizeof(double));

    if (*first == NULL) {
        *first = pp;
    } else {
        cpline_state_t *pn  = *first;
        while (pn->next != NULL) {
            pn = pn->next;
        }
        pn->next = pp;

		double t0 = (*first)->time;

		if ((time - t0) > tmax) {
			cpline_state_t *new_first = (*first)->next;
			free(*first);
			*first = new_first;
		}
    }
}

cpline_state_t *find_cpline_state(cpline_state_t *first, double time)
{
    cpline_state_t *pp = first;
    while (pp != NULL && pp->time < time) {
        pp = pp->next;
    }
    return pp;
}

cpline_state_t *get_cpline_last_state(cpline_state_t *first)
{
    cpline_state_t *pp = first;
	if (first == NULL) return NULL;
    while (pp->next != NULL) {
        pp = pp->next;
    }
    return pp;
}


void delete_cpline_last_state(cpline_state_t **first)
{
	cpline_state_t *pn  = *first;
	if (*first == NULL) return;

	if ((*first)->next == NULL) {
		free (*first);
		*first = NULL;
		return;
	}

	while (pn->next->next != NULL) {
		pn = pn->next;
	}
	free(pn->next);
	pn->next = NULL;
}

void delete_cpline_states(cpline_state_t **first)
{
	if (*first == NULL) return;
	cpline_state_t *pn;
	cpline_state_t *pc = *first;
	while (pc != NULL) {
		pn = pc->next;
		free (pc);
		pc = pn;
	}
	*first = NULL;
}


