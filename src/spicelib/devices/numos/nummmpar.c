/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NUMOSs in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "numosdef.h"
#include "sperror.h"
#include "suffix.h"

int
NUMOSmParam(param, value, inModel)
  int param;
  IFvalue *value;
  GENmodel *inModel;
{
  switch (param) {
  case NUMOS_MOD_NUMOS:
    /* no action - already know it is a 2d-numerical MOS, but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
