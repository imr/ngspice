/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NBJTs in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "nbjtdefs.h"
#include "sperror.h"
#include "suffix.h"

int
NBJTmParam(param, value, inModel)
  int param;
  IFvalue *value;
  GENmodel *inModel;
{
  switch (param) {
  case NBJT_MOD_NBJT:
    /* no action - already know it is a numerical bjt, but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
