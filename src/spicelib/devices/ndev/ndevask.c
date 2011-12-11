/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ndevdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
NDEVask(
  CKTcircuit *ckt,
  GENinstance *inInst,
  int which,
  IFvalue *value,
  IFvalue *select )
{
  NG_IGNORE(ckt);
  NG_IGNORE(inInst);
  NG_IGNORE(which);
  NG_IGNORE(value);
  NG_IGNORE(select);

  return (OK);
  /* NOTREACHED */
}
