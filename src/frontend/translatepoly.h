#ifndef TRANSLATEPOLY_H
#define TRANSLATEPOLY_H

#include "ngspice/ftedefs.h"

/* Translate a polynomial controlled Source line to an
  arbitrary behavioural modelling source line.
*/

struct card * translatepoly(struct card * input_line);

#endif