/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * Function to load the COMPLEX circuit matrix using the small signal
 * parameters saved during a previous DC operating point analysis.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "ndevdefs.h"
#include "sperror.h"
#include "complex.h"
#include "suffix.h"


int
NDEVacLoad(inModel, ckt)
  GENmodel *inModel;
  CKTcircuit *ckt;
{
 
  return (OK);
}
