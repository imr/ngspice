#include "polyderiv.h"

void
ft_polyderiv(double *coeffs, int degree)
{
	int	i;

	for (i = 0; i < degree; i++) {
		coeffs[i] = (i + 1) * coeffs[i + 1];
	}
}
