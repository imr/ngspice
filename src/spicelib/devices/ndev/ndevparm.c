/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/const.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NDEVparam(
  int param,
  IFvalue *value,
  GENinstance *inInst,
  IFvalue *select )
{
  NG_IGNORE(value);
  NG_IGNORE(inInst);
  NG_IGNORE(select);

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
