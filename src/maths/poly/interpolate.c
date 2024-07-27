#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include "interpolate.h"
#include "polyeval.h"
#include "polyfit.h"

/* Returns the strchr of the last element that was calculated. oval is
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
 * increasing or both are decreasing.
 */

#define EDGE_FACTOR 1e-3

bool
ft_interpolate(double *data, double *ndata, double *oscale, int olen,
	       double *nscale, int nlen, int degree)
{
    double *result, *scratch, *xdata, *ydata, diff;
    int sign = 1, lastone, i, l, middle, tdegree;

    if ((olen < 2) || (nlen < 2)) {
        fprintf(cp_err, "Error: lengths too small to interpolate.\n");
        return (FALSE);
    }
    if ((degree < 1) || (degree > olen)) {
        fprintf(cp_err, "Error: degree is %d, can't interpolate.\n",
                degree);
        return (FALSE);
    }

    for (i = 0; i < olen - 1; ++i) {
        if (oscale[i + 1] < oscale[i]) {
            sign = -1;
            break;
        } else if (oscale[i + 1] > oscale[i]) {
            sign = 1;
            break;
        }
    }
    if (i >= olen) {
        fprintf(cp_err, "Error: bad scale, can't interpolate.\n");
        return FALSE;
    }

    scratch = TMALLOC(double, (degree + 1) * (degree + 2));
    result = TMALLOC(double, degree + 1);
    xdata = TMALLOC(double, degree + 1);
    ydata = TMALLOC(double, degree + 1);

    /* Initial load of the values to be analysed by ft_polyfit(),
     * skipping irrelevant points and checking for and fudging vertical edges.
     */

    i = l = 0;
    middle = (degree + 1) / 2;
    if (sign > 0) {
        while (l < olen - degree && oscale[l + middle] < nscale[0])
            ++l;
    } else {
        while (l < olen - degree && oscale[l + middle] > nscale[0])
            ++l;
    }
    ydata[0] = data[l];
    xdata[0] = oscale[l];
    do {
        if (oscale[l + 1] == oscale[l]) {
            if (i == 0) {
                ydata[0] = data[++l];  // Ignore first point.
            } else {
                /* Push the previous x value back, making edge a slope. */

                diff = xdata[i] - xdata[i - 1];
                xdata[i] -= sign * diff * EDGE_FACTOR;
            }
        }
        xdata[++i] = oscale[++l];
        ydata[i] = data[l];
    } while (i < degree && l < olen - 1);

    if (i < degree) {
        fprintf(cp_err, "Error: too few points to calculate polynomial\n");
        return FALSE;
    }

    i = 0;
    tdegree = degree;
    while (!ft_polyfit(xdata + i, ydata + i, result, tdegree, scratch)) {
        /* If it doesn't work this time, bump the interpolation
         * degree down by one.
         */
        if (--tdegree == 0) {
            fprintf(cp_err, "ft_interpolate: Internal Error.\n");
            return (FALSE);
        }
        if (tdegree % 2)
            ++i;            // Drop left point.
    }

    /* Add this part of the curve. What we do is evaluate the polynomial
     * at those points between the last one and the one that is greatest,
     * without being greater than the leftmost old scale point, or least
     * if the scale is decreasing at the end of the interval we are looking
     * at.
     */

    lastone = putinterval(result, tdegree, ndata, -1,
                          nscale, nlen, xdata[middle], sign);

    /* Now plot the rest, piece by piece. l is the 
     * last element under consideration.
     */
    for (++l; l < olen && lastone < nlen - 1; l++) {
        double out;

        /* Shift the old stuff by one and get another value. */

        out = xdata[0];
        for (i = 0; i < degree; i++) {
            xdata[i] = xdata[i + 1];
            ydata[i] = ydata[i + 1];
        }
        ydata[i] = data[l];
        xdata[i] = oscale[l];

        /* Check for vertical edge. */

        if (oscale[l] == xdata[i - 1]) {
            if (degree == 1)
                diff = xdata[0] - out;
            else
                diff = xdata[i - 1] - xdata[i - 2];
            xdata[i - 1] -= sign * diff * EDGE_FACTOR;
        }

        /* Skip input points until the next output point is framed. */

        if (l < olen - degree) {
            if (sign > 0 && xdata[middle] < nscale[lastone + 1])
                continue;
            else if (sign < 0 && xdata[middle] > nscale[lastone + 1])
                continue;
        }

        i = 0;
        tdegree = degree;
        while (!ft_polyfit(xdata + i, ydata + i, result, tdegree, scratch)) {
            /* If it doesn't work this time, bump the interpolation
             * degree down by one.
             */

            if (--tdegree == 0) {
                fprintf(cp_err, "ft_interpolate: Internal Error.\n");
                return (FALSE);
            }
            if (!((degree - tdegree) & 1))
                ++i;            // Drop left point after right.
        }
        lastone = putinterval(result, tdegree, ndata, lastone,
                              nscale, nlen, xdata[middle], sign);
    }
    lastone = putinterval(result, degree, ndata, lastone,
                          nscale, nlen, oscale[olen - 1], sign);
    if (lastone < nlen - 1) /* ??? */
	ndata[nlen - 1] = data[olen - 1];
    tfree(scratch);
    tfree(xdata);
    tfree(ydata);
    tfree(result);
    return (TRUE);
}
