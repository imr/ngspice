/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NUMOSs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMOSmParam(int param, IFvalue *value, GENmodel *inModel)
{
  NG_IGNORE(value);
  NG_IGNORE(inModel);

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
