/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/const.h"
#include "numddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMDparam(int param, IFvalue *value, GENinstance *inInst, IFvalue *select)
{
  NUMDinstance *inst = (NUMDinstance *) inInst;

  NG_IGNORE(select);

  switch (param) {
  case NUMD_AREA:
    inst->NUMDarea = value->rValue;
    inst->NUMDareaGiven = TRUE;
    break;
  case NUMD_OFF:
    inst->NUMDoff = TRUE;
    break;
  case NUMD_IC_FILE:
    inst->NUMDicFile = value->sValue;
    inst->NUMDicFileGiven = TRUE;
    break;
  case NUMD_PRINT:
    inst->NUMDprint = value->iValue;
    inst->NUMDprintGiven = TRUE;
    break;
  case NUMD_TEMP:
    inst->NUMDtemp = value->rValue + CONSTCtoK;
    inst->NUMDtempGiven = TRUE;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
