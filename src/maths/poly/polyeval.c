#include "polyeval.h"

double
ft_peval(double x, double *coeffs, int degree)
{
	double	y;
	int	i;

	if (!coeffs)
		return 0.0;	/* XXX Should not happen */

	y = coeffs[degree];	/* there are (degree+1) coeffs */

	for (i = degree - 1; i >= 0; i--) {
		y *= x;
		y += coeffs[i];
	}

	return y;
}
