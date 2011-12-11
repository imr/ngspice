/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NBJT2s in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NBJT2mParam(int param, IFvalue *value, GENmodel *inModel)
{
  NG_IGNORE(value);
  NG_IGNORE(inModel);

  switch (param) {
  case NBJT2_MOD_NBJT:
    /* no action - already know it is a 2d-numerical bjt, but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
