/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ndevdefs.h"
#include "ngspice/numenum.h"
#include "ngspice/carddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int 
NDEVtemp(
  GENmodel *inModel,
  register CKTcircuit *ckt )
/*
 * perform the temperature update to the diode
 */
{
  NG_IGNORE(inModel);
  NG_IGNORE(ckt);

  return (OK);
}
