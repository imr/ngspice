#ifndef ngspice_POLYFIT_H
#define ngspice_POLYFIT_H

#include "ngspice/bool.h"

bool ft_polyfit(double *xdata, double *ydata, double *result,
		int degree, double *scratch);

#endif
