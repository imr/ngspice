/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NUMD2s in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "numd2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMD2mParam(int param, IFvalue *value, GENmodel *inModel)
{
  NG_IGNORE(value);
  NG_IGNORE(inModel);

  switch (param) {
  case NUMD2_MOD_NUMD:
    /* no action - already know it is a 2d-numerical diode, but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
