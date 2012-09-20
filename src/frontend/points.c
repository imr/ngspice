/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/graph.h"
#include "ngspice/ftedbgra.h"

#include "points.h"


/* Returns the minimum and maximum values of a dvec. Returns a pointer
 * to static data.  If real is TRUE look at the real parts, otherwise
 * the imag parts.  */


double *
ft_minmax(struct dvec *v, bool real)
{
    static double res[2];
    register int i;
    double d;

    res[0] = HUGE;
    res[1] = - res[0];

    for (i = 0; i < v->v_length; i++) {
        if (isreal(v))
            d = v->v_realdata[i];
        else if (real)
            d = realpart(v->v_compdata[i]);
        else
            d = imagpart(v->v_compdata[i]);
        if (d < res[0])
            res[0] = d;
        if (d > res[1])
            res[1] = d;
    }
    return (res);
}


/* Figure out where a point should go, given the limits of the plotting
 * area and the type of scale (log or linear).
 */

int
ft_findpoint(double pt, double *lims, int maxp, int minp, bool islog)
{
    double tl, th;

    if (pt < lims[0])
        pt = lims[0];
    if (pt > lims[1])
        pt = lims[1];
    if (islog) {
        tl = mylog10(lims[0]);
        th = mylog10(lims[1]);
        return (int)(((mylog10(pt) - tl) / (th - tl)) *
                     (maxp - minp) + minp);
    } else {
        return (int)(((pt - lims[0]) / (lims[1] - lims[0])) *
                     (maxp - minp) + minp);
    }
}


/* Will report the minimum and maximum in "reflection coefficient" space
 */

double *
ft_SMITHminmax(struct dvec *v, bool yval)
{
    static double res[2];
    register int i;
    double d, d2;

    res[0] = HUGE;
    res[1] = - res[0];

    for (i = 0; i < v->v_length; i++) {
        if (isreal(v))
            SMITH_tfm(v->v_realdata[i], 0.0, &d, &d2);
        else
            SMITH_tfm(realpart(v->v_compdata[i]), imagpart(v->v_compdata[i]),
                      &d, &d2);
/* Are we are looking for min/max X or Y ralue
 */
        if (yval)
            d = d2;

        if (d < res[0])
            res[0] = d;
        if (d > res[1])
            res[1] = d;
    }
    return (res);
}


int
SMITH_tfm(double re, double im, double *x, double *y)
{
    double  dnom;

    dnom = (re + 1) * (re + 1) + im * im;
    *x = (re * re + im * im - 1) / dnom;
    *y = 2 * im / dnom;

    return 0;
}
