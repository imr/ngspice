#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include "interpolate.h"
#include "polyeval.h"
#include "polyfit.h"

/* Returns thestrchr of the last element that was calculated. oval is
 * the value of the old scale at the end of the interval that is being
 * interpolated from, and sign is 1 if the old scale was increasing,
 * and -1 if it was decreasing.  */
static int
putinterval(double *poly, int degree, double *nvec,
	    int last, double *nscale, int nlen, double oval, int sign)
{
    int end, i;

    /* See how far we have to go. */
    for (end = last + 1; end < nlen; end++)
        if (nscale[end] * sign > oval * sign)
            break;
    end--;

    for (i = last + 1; i <= end; i++)
        nvec[i] = ft_peval(nscale[i], poly, degree);
    return (end);
}


/* Interpolate data from oscale to nscale. data is assumed to be olen long,
 * ndata will be nlen long. Returns FALSE if the scales are too strange
 * to deal with.  Note that we are guaranteed that either both scales are
 * strictly increasing or both are strictly decreasing.
 */
bool
ft_interpolate(double *data, double *ndata, double *oscale, int olen,
	       double *nscale, int nlen, int degree)
{
    double *result, *scratch, *xdata, *ydata;
    int sign, lastone, i, l;

    if ((olen < 2) || (nlen < 2)) {
        fprintf(cp_err, "Error: lengths too small to interpolate.\n");
        return (FALSE);
    }
    if ((degree < 1) || (degree > olen)) {
        fprintf(cp_err, "Error: degree is %d, can't interpolate.\n",
                degree);
        return (FALSE);
    }

    if (oscale[1] < oscale[0])
        sign = -1;
    else
        sign = 1;

    scratch = TMALLOC(double, (degree + 1) * (degree + 2));
    result = TMALLOC(double, degree + 1);
    xdata = TMALLOC(double, degree + 1);
    ydata = TMALLOC(double, degree + 1);

    /* Deal with the first degree pieces. */
    memcpy(ydata, data, (size_t) (degree + 1) * sizeof (double));
    memcpy(xdata, oscale, (size_t) (degree + 1) * sizeof (double));

    while (!ft_polyfit(xdata, ydata, result, degree, scratch)) {
        /* If it doesn't work this time, bump the interpolation
         * degree down by one.
         */

        if (--degree == 0) {
            fprintf(cp_err, "ft_interpolate: Internal Error.\n");
            return (FALSE);
        }

    }

    /* Add this part of the curve. What we do is evaluate the polynomial
     * at those points between the last one and the one that is greatest,
     * without being greater than the leftmost old scale point, or least
     * if the scale is decreasing at the end of the interval we are looking
     * at.
     */
    lastone = -1;
    for (i = 0; i < degree; i++) {
        lastone = putinterval(result, degree, ndata, lastone, 
                    nscale, nlen, xdata[i], sign);
    }

    /* Now plot the rest, piece by piece. l is the 
     * last element under consideration.
     */
    for (l = degree + 1; l < olen; l++) {

        /* Shift the old stuff by one and get another value. */
        for (i = 0; i < degree; i++) {
            xdata[i] = xdata[i + 1];
            ydata[i] = ydata[i + 1];
        }
        ydata[i] = data[l];
        xdata[i] = oscale[l];

        while (!ft_polyfit(xdata, ydata, result, degree, scratch)) {
            if (--degree == 0) {
                fprintf(cp_err, 
                    "interpolate: Internal Error.\n");
                return (FALSE);
            }
        }
        lastone = putinterval(result, degree, ndata, lastone, 
                    nscale, nlen, xdata[i], sign);
    }
    if (lastone < nlen - 1) /* ??? */
	ndata[nlen - 1] = data[olen - 1];
    tfree(scratch);
    tfree(xdata);
    tfree(ydata);
    tfree(result);
    return (TRUE);
}
