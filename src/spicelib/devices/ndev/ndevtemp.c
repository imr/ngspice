/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "ndevdefs.h"
#include "numenum.h"
#include "carddefs.h"
#include "sperror.h"
#include "suffix.h"

#define NIL(type)   ((type *)0)

int 
NDEVtemp(inModel, ckt)
  GENmodel *inModel;
  register CKTcircuit *ckt;
/*
 * perform the temperature update to the diode
 */
{

  return (OK);
}
