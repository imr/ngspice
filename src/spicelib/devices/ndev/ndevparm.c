/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "const.h"
#include "ndevdefs.h"
#include "sperror.h"
#include "suffix.h"

int
NDEVparam(
  int param,
  IFvalue *value,
  GENinstance *inInst,
  IFvalue *select )
{
  IGNORE(value);
  IGNORE(inInst);
  IGNORE(select);

  switch (param) {
  case NDEV_MOD_NDEV:
    /* no action , but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
