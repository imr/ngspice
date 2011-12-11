/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets instance parameters for NUMOSs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NUMOSparam(int param, IFvalue *value, GENinstance *inInst, IFvalue *select)
{
  register NUMOSinstance *inst = (NUMOSinstance *) inInst;

  NG_IGNORE(select);

  switch (param) {
  case NUMOS_AREA:
    inst->NUMOSarea = value->rValue;
    inst->NUMOSareaGiven = TRUE;
    break;
  case NUMOS_WIDTH:
    inst->NUMOSwidth = value->rValue;
    inst->NUMOSwidthGiven = TRUE;
    break;
  case NUMOS_LENGTH:
    inst->NUMOSlength = value->rValue;
    inst->NUMOSlengthGiven = TRUE;
    break;
  case NUMOS_OFF:
    inst->NUMOSoff = TRUE;
    break;
  case NUMOS_IC_FILE:
    inst->NUMOSicFile = value->sValue;
    inst->NUMOSicFileGiven = TRUE;
    break;
  case NUMOS_PRINT:
    inst->NUMOSprint = value->iValue;
    inst->NUMOSprintGiven = TRUE;
    break;
  case NUMOS_TEMP:
    inst->NUMOStemp = value->rValue + CONSTCtoK;
    inst->NUMOStempGiven = TRUE;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
