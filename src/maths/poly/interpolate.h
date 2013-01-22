#ifndef ngspice_INTERPOLATE_H
#define ngspice_INTERPOLATE_H

#include "ngspice/bool.h"

bool ft_interpolate(double *data, double *ndata, double *oscale, int olen, double *nscale, int nlen, int degree);

#endif
