#ifndef _POLYFIT_H
#define _POLYFIT_H

#include <ngspice/bool.h>

bool ft_polyfit(double *xdata, double *ydata, double *result,
		int degree, double *scratch);

#endif
