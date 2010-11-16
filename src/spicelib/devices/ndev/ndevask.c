/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ndevdefs.h"
#include "complex.h"
#include "sperror.h"
#include "suffix.h"


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
