/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"



int
NDEVtrunc(
  GENmodel *inModel,
  register CKTcircuit *ckt,
  double *timeStep )

{
  NG_IGNORE(inModel);
  NG_IGNORE(ckt);
  NG_IGNORE(timeStep);

  return (OK);
}
