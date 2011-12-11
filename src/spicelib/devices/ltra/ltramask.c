/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

/*
 * This routine sets model parameters for LTRA lines in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
LTRAmAsk(CKTcircuit *ckt, GENmodel *inModel, int param, IFvalue *value)
{
  LTRAmodel *mods = (LTRAmodel *) inModel;

  NG_IGNORE(ckt);

  switch (param) {
  case LTRA_MOD_LTRA:
    value->iValue = 1;
    break;
  case LTRA_MOD_RELTOL:
    value->rValue = mods->LTRAreltol;
    break;
  case LTRA_MOD_ABSTOL:
    value->rValue = mods->LTRAabstol;
    break;
  case LTRA_MOD_STLINEREL:
    value->rValue = mods->LTRAstLineReltol;
    break;
  case LTRA_MOD_STLINEABS:
    value->rValue = mods->LTRAstLineAbstol;
    break;
  case LTRA_MOD_CHOPREL:
    value->rValue = mods->LTRAchopReltol;
    break;
  case LTRA_MOD_CHOPABS:
    value->rValue = mods->LTRAchopAbstol;
    break;
  case LTRA_MOD_TRUNCNR:
    value->iValue = mods->LTRAtruncNR;
    break;
  case LTRA_MOD_TRUNCDONTCUT:
    value->iValue = mods->LTRAtruncDontCut;
    break;
  case LTRA_MOD_R:
    value->rValue = mods->LTRAresist;
    break;
  case LTRA_MOD_L:
    value->rValue = mods->LTRAinduct;
    break;
  case LTRA_MOD_G:
    value->rValue = mods->LTRAconduct;
    break;
  case LTRA_MOD_C:
    value->rValue = mods->LTRAcapac;
    break;
  case LTRA_MOD_LEN:
    value->rValue = mods->LTRAlength;
    break;
  case LTRA_MOD_NL:
    value->rValue = mods->LTRAnl;
    break;
  case LTRA_MOD_FREQ:
    value->rValue = mods->LTRAf;
    break;
  case LTRA_MOD_FULLCONTROL:
    value->iValue = mods->LTRAlteConType;
    break;
  case LTRA_MOD_HALFCONTROL:
    value->iValue = mods->LTRAlteConType;
    break;
  case LTRA_MOD_NOCONTROL:
    value->iValue = mods->LTRAlteConType;
    break;
  case LTRA_MOD_PRINT:
    value->iValue = mods->LTRAprintFlag;
    break;
  case LTRA_MOD_NOPRINT:
    mods->LTRAprintFlag = FALSE;
    break;
    /*
     * case LTRA_MOD_RONLY: mods->LTRArOnly= TRUE; break;
     */
  case LTRA_MOD_STEPLIMIT:
    value->iValue = mods->LTRAstepLimit;
    break;
  case LTRA_MOD_NOSTEPLIMIT:
    value->iValue = mods->LTRAstepLimit;
    break;
  case LTRA_MOD_LININTERP:
    value->iValue = mods->LTRAhowToInterp;
    break;
  case LTRA_MOD_QUADINTERP:
    value->iValue = mods->LTRAhowToInterp;
    break;
  case LTRA_MOD_MIXEDINTERP:
    value->iValue = mods->LTRAhowToInterp;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
