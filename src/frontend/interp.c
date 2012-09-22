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
    struct dvec *v;
    double *nd;

    if (!isreal(ov)) {
        fprintf(cp_err, "Warning: %s is not real\n", ov->v_name);
        return;
    }

    if (ov->v_length < oldscale->v_length) {
        fprintf(cp_err, "Warning: %s is too short\n", ov->v_name);
        return;
    }

    v = alloc(struct dvec);
    v->v_name = copy(ov->v_name);
    v->v_type = ov->v_type;
    v->v_flags = ov->v_flags;
    v->v_flags |= VF_PERMANENT;
    v->v_length = newlen;

    nd = TMALLOC(double, newlen);
    if (!ft_interpolate(ov->v_realdata, nd, oldscale->v_realdata,
                        oldscale->v_length, newscale, newlen, 1))
    {
        fprintf(cp_err, "Error: can't interpolate %s\n", ov->v_name);
        return;
    }
    v->v_realdata = nd;

    vec_new(v);
}

