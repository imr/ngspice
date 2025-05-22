/* tline_common.c
 * common definitions for all transmission lines
 * (c) Vadim Kuznetsov 2025
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

