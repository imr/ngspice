/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Curve plotting routines and general (non-graphics) plotting things.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/graph.h"
#include "graf.h"
#include "ngspice/ftedbgra.h"

#include "plotcurv.h"


static void plotinterval(struct dvec *v, double lo, double hi, register double *coeffs,
                         int degree, bool rotated);


/* Plot the vector v, with scale xs.  If we are doing curve-fitting, then
 * do some tricky stuff.
 *
 * This function may be called multiple times to plot a single vector so that
 * each call is short, keeping the UI responsive.  The *incr structure
 * describes the progress made in previous calls.  If incr is NULL
 * a complete vector is plotted in one call.
 *
 * These macros allow some local variables to be replaced by members of *incr.
 */

#define i   ip->index // int
#define end ip->end   // int
#define dir ip->dir   // int
#define lx  ip->lx    // double
#define ly  ip->ly    // double

/* Macros for early returns during incremental plotting and
 * adjusting number of loops.
 */

#define INCR_RETURN \
    if (incr) { \
        if (i >= end) \
            incr->istate = Done; \
        else \
            return; \
    }

#define INCR_RETURN_MORE \
    if (incr && i < end) \
            return; \
    i = 0;

#define INCR_SCALE_LIMIT(s) \
    if (incr) {  \
        limit *= s; \
        if (limit > end) \
            limit = end; \
    }

void
ft_graf(struct dvec *v, struct dvec *xs, bool nostart,
        struct incremental_plot *incr)
{
    struct incremental_plot  local, *ip;
    int                      degree, gridsize, length, limit;
    register int             j, l;
    double                  *scratch, *result, *gridbuf, *mm;
    register double         *xdata, *ydata;
    bool                     rot, increasing = FALSE;
    double                   dx, dy;

    if (incr) {
        /* Using multiple calls. In the Active state jump back to an
         * active plotting loop using previous values for local variables.
         *
         * Sensitive readers should stop here and avoid all blocks
         * beginning "if (incr)".
         */

        ip = incr;                      // Use passed context for variables
        switch(incr->istate) {
        case Start:
            i = 0;
            incr->istate = Active;
            end = v->v_length;
            if (xs && xs->v_length < end)
                end = xs->v_length;
            limit = (incr->maximum > end) ? end : incr->maximum;
            break;
        case Active:
            limit = incr->index + incr->maximum;
            if (limit > end)
                limit = end;

            switch (incr->type) {
            case Scan_Monotonic:
                goto resume_mono;
                break;
            case Oneval:
                goto resume_oneval;
                break;
            case Simple:
                goto resume_simple;
                break;
            case Regrid:
                goto resume_regrid;
                break;
            case Poly:
                goto resume_poly;
                break;
            }
            break;
        case Done:
            return; // Not expected!
            break;
        case Abort:
            incr->type = Done;
            switch (incr->type) {
            case Regrid:
                goto resume_regrid;    // Dummy for now.
                break;
            case Poly:
                goto resume_poly;      // Dummy for now.
                break;
            default:
                return;                // No memory to be freed.
            }
            break;
        }
    } else {
        ip = &local;                    // Use automatic variable
        i = 0;
        end = v->v_length;
        if (xs && xs->v_length < end)
            end = xs->v_length;
        limit = end;
    }

    if (xs && v->v_length != xs->v_length) {
        fprintf(stderr,
                "Warning: length of vector %s and its scale %s "
                "do not match, plot may be truncated!\n",
                v->v_name, xs->v_name);
    }

    /* if already started, use saved degree */
    if (nostart) {
        degree = currentgraph->degree;
    } else {
        if (!cp_getvar("polydegree", CP_NUM, &degree, 0))
            degree = 1;
        currentgraph->degree = degree;
    }

    if (degree > end)
        degree = end;

    if (degree < 1) {
        fprintf(cp_err, "Error: polydegree is %d, can't plot...\n",
                degree);
        return;
    }

    if (!cp_getvar("gridsize", CP_NUM, &gridsize, 0))
        gridsize = 0;

    if ((gridsize < 0) || (gridsize > 10000)) {
        fprintf(cp_err, "Error: bad grid size %d\n", gridsize);
        return;
    }

    if (gridsize && xs && end > 1) {
        ip->type = Scan_Monotonic;
        INCR_SCALE_LIMIT(32);
    resume_mono:
        if (isreal(xs)) {
            increasing = (xs->v_realdata[0] < xs->v_realdata[1]);
            for (; i < end - 1; i++)
                if (increasing != (xs->v_realdata[i] < xs->v_realdata[i + 1])) {
                    fprintf(cp_err,
                            "Warning: scale not monotonic, gridsize not relevant.\n");
                    gridsize = 0;
                    i = end;
                    break;
                }
        } else {
            increasing = (realpart(xs->v_compdata[0]) < realpart(xs->v_compdata[1]));
            for (; i < end - 1; i++)
                if (increasing != (realpart(xs->v_compdata[i]) < realpart(xs->v_compdata[i + 1]))) {
                    fprintf(cp_err,
                            "Warning: scale not monotonic, gridsize not relevant.\n");
                    gridsize = 0;
                    i = end;
                    break;
                }
        }
        INCR_RETURN_MORE
    }

    if (!nostart)
        gr_start(v);

    /* Do the one value case */

    if (!xs) {
        ip->type = Oneval;
    resume_oneval:
        for (; i < limit; i++) {

            /* We should do the one - point case too!
             *      Important for pole-zero for example
             */
            if (end == 1) {
                j = 0;
            } else {
                j = i-1;
                if (i == 0)
                    continue;
            }

            if (isreal(v)) {
                /* This isn't good but we may as well do
                 * something useful.
                 */
                gr_point(v, v->v_realdata[i],
                         0.0, /* v->v_realdata[i], */
                         v->v_realdata[j],
                         0.0, /* v->v_realdata[j], */
                         (j == i ? 1 : i));
            } else {
                gr_point(v, realpart(v->v_compdata[i]),
                         imagpart(v->v_compdata[i]),
                         realpart(v->v_compdata[j]),
                         imagpart(v->v_compdata[j]), (j == i ? 1 : i));
            }
        }
        INCR_RETURN
        gr_end(v);
        return;
    }

    xs->v_flags |= VF_PERMANENT;

    /* First check the simple case, where we don't have to do any
     * interpolation.
     */
    if ((degree == 1) && (gridsize == 0)) {
        /* We have to take care of non-monotonic x-axis values.
         * If they occur, plotting is suppressed, except for mono is set
         * to FALSE by flag 'retraceplot' in command 'plot'.
         * Then everything is plotted.
         */
        bool mono;

        ip->type = Simple;
        dir = 0;
        lx = 0.0;
        ly = 0.0;
    resume_simple:

        mono = (currentgraph->plottype != PLOT_RETLIN);
        for (; i < limit; i++) {
            dx = isreal(xs) ? xs->v_realdata[i] :
                realpart(xs->v_compdata[i]);
            dy = isreal(v) ? v->v_realdata[i] :
                realpart(v->v_compdata[i]);
            if ((i == 0 || (dir > 0 ? lx > dx : (dir < 0 ? lx < dx : 0))) &&
                (mono || (xs->v_plot && xs->v_plot->pl_scale == xs)))
            {
                gr_point(v, dx, dy, lx, ly, 0);
            } else {
                gr_point(v, dx, dy, lx, ly, i);
                if (!dir)
                    dir = lx > dx ? -1 : lx < dx ? 1 : 0;
            }
            lx = dx;
            ly = dy;
        }
        if (end == 1) {
            gr_point(v, dx, dy, lx, ly, 1);
        } else
            INCR_RETURN
        gr_end(v);
        if (incr)
            incr->istate = Done;
        return;
    }

    if (gridsize < degree + 1)
        gridsize = 0;

    if (gridsize) {
        /* This is done quite differently from what we do below... */
        gridbuf = TMALLOC(double, gridsize);
        result = TMALLOC(double, gridsize);
        if (isreal(v)) {
            ydata = v->v_realdata;
        } else {
            ydata = TMALLOC(double, end);
            for (i = 0; i < end; i++)
                ydata[i] = realpart(v->v_compdata[i]);
        }

        if (isreal(xs)) {
            xdata = xs->v_realdata;
        } else {
            xdata = TMALLOC(double, xs->v_length);
            for (i = 0; i < xs->v_length; i++)
                xdata[i] = realpart(xs->v_compdata[i]);
        }

        mm = ft_minmax(xs, TRUE);
        dx = (mm[1] - mm[0]) / gridsize;
        if (increasing)
            for (i = 0, dy = mm[0]; i < gridsize; i++, dy += dx)
                gridbuf[i] = dy;
        else
            for (i = 0, dy = mm[1]; i < gridsize; i++, dy -= dx)
                gridbuf[i] = dy;
        if (!ft_interpolate(ydata, result, xdata, end, gridbuf,
                            gridsize, degree)) {
            fprintf(cp_err, "Error: can't put %s on gridsize %d\n",
                    v->v_name, gridsize);
            return;
        }
        /* Now this is a problem.  There's no way that we can
         * figure out where to put the tic marks to correspond with
         * the actual data...
         */
    resume_regrid:
        for (i = 0; i < gridsize; i++)
            gr_point(v, gridbuf[i], result[i],
                     gridbuf[i ? (i - 1) : i], result[i ? (i - 1) : i], -1);
        gr_end(v);
        tfree(gridbuf);
        tfree(result);
        if (!isreal(v))
            tfree(ydata);
        if (!isreal(xs))
            tfree(xdata);
        return;
    }

    /* We need to do curve fitting now. First get some scratch
     * space
     */
    scratch = TMALLOC(double, (degree + 1) * (degree + 2));
    result = TMALLOC(double, degree + 1);
    xdata = TMALLOC(double, degree + 1);
    ydata = TMALLOC(double, degree + 1);


    /* Plot the first degree segments... */
    if (isreal(v))
        memcpy(ydata, v->v_realdata, (size_t)(degree + 1) * sizeof(double));
    else
        for (i = 0; i <= degree; i++)
            ydata[i] = realpart(v->v_compdata[i]);

    if (isreal(xs))
        memcpy(xdata, xs->v_realdata, (size_t)(degree + 1) * sizeof(double));
    else
        for (i = 0; i <= degree; i++)
            xdata[i] = realpart(xs->v_compdata[i]);

    rot = FALSE;
    while (!ft_polyfit(xdata, ydata, result, degree, scratch)) {
        /* Rotate the coordinate system 90 degrees and try again.
         * If it doesn't work this time, bump the interpolation
         * degree down by one...
         */
        if (ft_polyfit(ydata, xdata, result, degree, scratch)) {
            rot = TRUE;
            break;
        }
        if (--degree == 0) {
            fprintf(cp_err, "plotcurve: Internal Error: ack...\n");
            return;
        }
    }

 resume_poly:
    /* Plot this part of the curve... */
    for (i = 0; i < degree; i++)
        if (rot)
            plotinterval(v, ydata[i], ydata[i + 1], result, degree, TRUE);
        else
            plotinterval(v, xdata[i], xdata[i + 1], result, degree, FALSE);

    /* Now plot the rest, piece by piece... l is the
     * last element under consideration.
     */
    length = end;
    for (l = degree + 1; l < length; l++) {

        /* Shift the old stuff by one and get another value. */
        for (i = 0; i < degree; i++) {
            xdata[i] = xdata[i + 1];
            ydata[i] = ydata[i + 1];
        }

        if (isreal(v))
            ydata[i] = v->v_realdata[l];
        else
            ydata[i] = realpart(v->v_compdata[l]);

        if (isreal(xs))
            xdata[i] = xs->v_realdata[l];
        else
            xdata[i] = realpart(xs->v_compdata[l]);

        rot = FALSE;
        while (!ft_polyfit(xdata, ydata, result, degree, scratch)) {
            if (ft_polyfit(ydata, xdata, result, degree, scratch)) {
                rot = TRUE;
                break;
            }
            if (--degree == 0) {
                fprintf(cp_err,
                        "plotcurve: Internal Error: ack...\n");
                return;
            }
        }

        if (rot)
            plotinterval(v, ydata[degree - 1], ydata[degree],
                         result, degree, TRUE);
        else
            plotinterval(v, xdata[degree - 1], xdata[degree],
                         result, degree, FALSE);
        if (incr)
            incr->istate = Done;

    }

    tfree(scratch);
    tfree(xdata);
    tfree(ydata);
    tfree(result);

    gr_end(v);
}

#undef i
#undef end
#undef dir
#undef lx
#undef ly

#define GRANULARITY 10

static void
plotinterval(struct dvec *v, double lo, double hi, register double *coeffs, int degree, bool rotated)
{
    double incr, dx, dy, lx, ly;
    register int i;
    int steps;

    /*
      fprintf(cp_err, "plotinterval(%s, %G, %G, [ ", v->v_name, lo, hi);
      for (i = 0; i <= degree; i++)
      fprintf(cp_err, "%G ", coeffs[i]);
      fprintf(cp_err, "], %d, %s)\n\r", degree, rotated ? "TRUE" : "FALSE");
    */

    /* This is a problem -- how do we know what granularity to use?  If
     * the guy cares about this he will use gridsize.
     */
    if (!cp_getvar("polysteps", CP_NUM, &steps, 0))
        steps = GRANULARITY;

    incr = (hi - lo) / (double) (steps + 1);
    dx = lo + incr;
    lx = lo;
    ly = ft_peval(lo, coeffs, degree);
    for (i = 0; i <= steps; i++, dx += incr) {
        dy = ft_peval(dx, coeffs, degree);
        if (rotated)
            gr_point(v, dy, dx, ly, lx, -1);
        else
            gr_point(v, dx, dy, lx, ly, -1);
        lx = dx;
        ly = dy;
        /* fprintf(cp_err, "plot (%G, %G)\n\r", dx, dy); */
    }
}
