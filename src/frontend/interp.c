/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Polynomial interpolation code.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "interp.h"


void
lincopy(struct dvec *ov, double *newscale, int newlen, struct dvec *oldscale)
{
    if (!isreal(ov)) {
        fprintf(cp_err, "Warning: vector %s is a complex vector - "
                "complex vectors cannot be interpolated\n",
                ov->v_name);
        return;
    }

    if (ov->v_length == 1) {
        fprintf(cp_err, "Warning: %s is a scalar - "
                "interpolation is not possible\n",
                ov->v_name);
        return;
    }

    if (ov->v_length < oldscale->v_length) {
        fprintf(cp_err, "Warning: %s only contains %d points - "
                "interpolation is not performed unless there are "
                "at least as many points as the scale vector (%d)\n",
                ov->v_name, ov->v_length, oldscale->v_length);
        return;
    }

    /* Allocate the vector to receive the linearized data */
    struct dvec * const v = dvec_alloc(copy(ov->v_name),
                   ov->v_type,
                   ov->v_flags | VF_PERMANENT,
                   newlen, NULL);

    /* Do interpolation and then add the vector to the current plot. If
     * interpolation fails, the vector must be freed. */
    if (!ft_interpolate(ov->v_realdata, v->v_realdata, oldscale->v_realdata,
            oldscale->v_length, newscale, newlen, 1)) {
        fprintf(cp_err, "Error: can't interpolate %s\n", ov->v_name);
        dvec_free(v);
        return;
    }

    vec_new(v);
} /* end of function lincopy */



