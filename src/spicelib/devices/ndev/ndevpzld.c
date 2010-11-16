/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "ndevdefs.h"
#include "suffix.h"


int
NDEVpzLoad(
  GENmodel *inModel,
  register CKTcircuit *ckt,
  SPcomplex *s )
{
  NG_IGNORE(inModel);
  NG_IGNORE(ckt);
  NG_IGNORE(s);
 
  return (OK);
}
