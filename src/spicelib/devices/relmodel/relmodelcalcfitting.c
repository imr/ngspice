/**********
Author: Francesco Lannutti - June 2016
**********/

#include "ngspice/ngspice.h"
#include "relmodeldefs.h"
#include "ngspice/sperror.h"

#include <gsl/gsl_fit.h>
#include <gsl/gsl_linalg.h>

int
RELMODELcalculateFitting (unsigned int rows, unsigned int number_of_periods, double target, double *timeFit, double *deltaVthFit, double *result)
{
    double f, factor_for_2pi, *fitting_matrix ;
    unsigned int columns, i, j, number_of_modes, size ;

    /* Generate the fitting matrix */

    number_of_modes = 10 * (number_of_periods + 1) ;

    columns = 2 * number_of_modes + 1 ;
    size = rows * columns ;
    fitting_matrix = TMALLOC (double, size) ;

    factor_for_2pi = 2 * M_PI / (timeFit [rows - 1] - timeFit [0]) ;

    for (i = 0 ; i < rows ; i++)
    {
        /* The first element of every row is equal to 1 */
        fitting_matrix [columns * i] = 1 ;

        /* The odd elements of every row are cos(x) */
        for (j = 0 ; j < number_of_modes ; j++)
        {
            fitting_matrix [columns * i + 2 * j + 1] = cos ((j + 1) * timeFit [i] * factor_for_2pi) ;
        }

        /* The even elements of every row are sin(x) */
        for (j = 1 ; j <= number_of_modes ; j++)
        {
            fitting_matrix [columns * i + 2 * j] = sin (j * timeFit [i] * factor_for_2pi) ;
        }
    }

    gsl_matrix_view m = gsl_matrix_view_array (fitting_matrix, rows, columns) ;
    gsl_vector_view b = gsl_vector_view_array (deltaVthFit, rows) ;
    gsl_vector *tau = gsl_vector_alloc (MIN (rows, columns)) ;

    gsl_vector *x = gsl_vector_alloc (columns) ;
    gsl_vector *residual = gsl_vector_alloc (rows) ;
    gsl_linalg_QR_decomp (&m.matrix, tau) ;
    gsl_linalg_QR_lssolve (&m.matrix, tau, &b.vector, x, residual) ;

    f = gsl_vector_get (x, 0) ;

    /* The odd elements of every row are cos(x) */
    for (j = 0 ; j < number_of_modes ; j++)
    {
        f += gsl_vector_get (x, 2 * j + 1) * cos ((j + 1) * target * factor_for_2pi) ;
    }

    /* The even elements of every row are sin(x) */
    for (j = 1 ; j <= number_of_modes ; j++)
    {
        f += gsl_vector_get (x, 2 * j) * sin (j * target * factor_for_2pi) ;
    }

    fprintf (stderr, "\n\nExtrapolation at %-.9g seconds:\n\t\t\t\tDeltaVth: %-.9gmV\n\n\n\n", target, f * 1000) ;

    *result = f ;

    gsl_vector_free (tau) ;
    gsl_vector_free (x) ;
    gsl_vector_free (residual) ;
    FREE (fitting_matrix) ;

    return (OK) ;
}
