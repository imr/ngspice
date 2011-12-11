/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets instance parameters for NBJTs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NBJTparam(int param, IFvalue *value, GENinstance *inInst, IFvalue *select)
{
  register NBJTinstance *inst = (NBJTinstance *) inInst;

  NG_IGNORE(select);

  switch (param) {
  case NBJT_AREA:
    inst->NBJTarea = value->rValue;
    inst->NBJTareaGiven = TRUE;
    break;
  case NBJT_OFF:
    inst->NBJToff = TRUE;
    break;
  case NBJT_IC_FILE:
    inst->NBJTicFile = value->sValue;
    inst->NBJTicFileGiven = TRUE;
    break;
  case NBJT_PRINT:
    inst->NBJTprint = value->iValue;
    inst->NBJTprintGiven = TRUE;
    break;
  case NBJT_TEMP:
    inst->NBJTtemp = value->rValue + CONSTCtoK;
    inst->NBJTtempGiven = TRUE;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
