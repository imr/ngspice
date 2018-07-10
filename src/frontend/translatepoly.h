#ifndef TRANSLATEPOLY_H
#define TRANSLATEPOLY_H

#include "ngspice/ftedefs.h"

/* Translate a polynomial controlled Source line to an
  arbitrary behavioural modelling source line.
*/

struct line * translatepoly(struct line * input_line);

#endif