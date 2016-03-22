#include "ngspice/ngspice.h"

#include "polyfit.h"
#include "polyeval.h"

/* Takes n = (degree+1) doubles, and fills in result with the n
 * coefficients of the polynomial that will fit them. It also takes a
 * pointer to an array of n ^ 2 + n doubles to use for scratch -- we
 * want to make this fast and avoid doing tmallocs for each call.  */
bool
ft_polyfit(double *xdata, double *ydata, double *result,
	   int degree, double *scratch)
{
    double *mat1 = scratch;
    int l, k, j, i;
    int n = degree + 1;
    double *mat2 = scratch + n * n; /* XXX These guys are hacks! */
    double d;

    /* speed up fitting process, e.g. for command 'linearize' */
    if (degree == 1) {
        result[0] = (xdata[1] * ydata[0] - xdata[0] * ydata[1]) / (xdata[1] - xdata[0]);
        result[1] = (ydata[1] - ydata[0]) / (xdata[1] - xdata[0]);
        return (TRUE);
    }

    memset(result, 0,   (size_t) (n)     * sizeof(double));
    memset(mat1, 0,     (size_t) (n * n) * sizeof(double));
    memcpy(mat2, ydata, (size_t) (n)     * sizeof(double));

    /* Fill in the matrix with x^k for 0 <= k <= degree for each point */
    l = 0;
    for (i = 0; i < n; i++) {
	d = 1.0;
        for (j = 0; j < n; j++) {
            mat1[l] = d;
	    d *= xdata[i];
            l += 1;
        }
    }

    /* Do Gauss-Jordan elimination on mat1. */
    for (i = 0; i < n; i++) {
      int lindex;
      double largest;
        /* choose largest pivot */
        for (j=i, largest = mat1[i * n + i], lindex = i; j < n; j++) {
          if (fabs(mat1[j * n + i]) > largest) {
            largest = fabs(mat1[j * n + i]);
            lindex = j;
          }
        }
        if (lindex != i) {
          /* swap rows i and lindex */
          for (k = 0; k < n; k++) {
            SWAP(double, mat1[i * n + k], mat1[lindex * n + k]);
          }
          SWAP(double, mat2[i], mat2[lindex]);
        }
        /* Make sure we have a non-zero pivot. */
        if (mat1[i * n + i] == 0.0) {
            /* this should be rotated. */
            return (FALSE);
        }
        for (j = i + 1; j < n; j++) {
            d = mat1[j * n + i] / mat1[i * n + i];
            for (k = 0; k < n; k++)
                mat1[j * n + k] -= d * mat1[i * n + k];
            mat2[j] -= d * mat2[i];
        }
    }

    for (i = n - 1; i > 0; i--)
        for (j = i - 1; j >= 0; j--) {
            d = mat1[j * n + i] / mat1[i * n + i];
            for (k = 0; k < n; k++)
                mat1[j * n + k] -= 
                        d * mat1[i * n + k];
            mat2[j] -= d * mat2[i];
        }
    
    /* Now write the stuff into the result vector. */
    for (i = 0; i < n; i++) {
        result[i] = mat2[i] / mat1[i * n + i];
        /* printf(cp_err, "result[%d] = %G\n", i, result[i]);*/
    }

#define ABS_TOL 0.001
#define REL_TOL 0.001

    /* Let's check and make sure the coefficients are ok.  If they aren't,
     * just return FALSE.  This is not the best way to do it.
     */
    for (i = 0; i < n; i++) {
        d = ft_peval(xdata[i], result, degree);
        if (fabs(d - ydata[i]) > ABS_TOL) {
            /*
            fprintf(cp_err,
                "Error: polyfit: x = %e, y = %le, int = %e\n",
                    xdata[i], ydata[i], d);
            printmat("mat1", mat1, n, n);
            printmat("mat2", mat2, n, 1);
            */
            return (FALSE);
        } else if (fabs(d - ydata[i]) / (fabs(d) > ABS_TOL ? fabs(d) :
                ABS_TOL) > REL_TOL) {
            /*
            fprintf(cp_err,
                "Error: polyfit: x = %e, y = %le, int = %e\n",
                    xdata[i], ydata[i], d);
            printmat("mat1", mat1, n, n);
            printmat("mat2", mat2, n, 1);
            */
            return (FALSE);
        }
    }

    return (TRUE);
}
