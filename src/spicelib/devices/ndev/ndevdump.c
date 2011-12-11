/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * This is a simple routine to dump the internal device states. It produces
 * states for .OP, .DC, & .TRAN simulations.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ndevdefs.h"
#include "ngspice/suffix.h"


/* State Counter
 * static int state_numOP = 0;
 * static int state_numDC = 0;
 * static int state_numTR = 0;
 */

void
NDEV_dump(
  GENmodel *inModel,
  CKTcircuit *ckt )
{
    NG_IGNORE(inModel);
    NG_IGNORE(ckt);
  
}



void
NDEV_acct(
  GENmodel *inModel,
  CKTcircuit *ckt,
  FILE *file )
{
    NG_IGNORE(inModel);
    NG_IGNORE(ckt);
    NG_IGNORE(file);
  
}
