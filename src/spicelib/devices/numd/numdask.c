/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "numddefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* Check out this one */
extern int NUMDinitSmSig(NUMDinstance *);


/* ARGSUSED */
int
NUMDask(CKTcircuit *ckt, GENinstance *inInst, int which, IFvalue *value, IFvalue *select)
{
  NUMDinstance *inst = (NUMDinstance *) inInst;

  NG_IGNORE(select);

  switch (which) {
  case NUMD_AREA:
    value->rValue = inst->NUMDarea;
    return (OK);
  case NUMD_TEMP:
    value->rValue = inst->NUMDtemp - CONSTCtoK;
    return (OK);
  case NUMD_VD:
    value->rValue = *(ckt->CKTstate0 + inst->NUMDvoltage);
    return (OK);
  case NUMD_ID:
    value->rValue = *(ckt->CKTstate0 + inst->NUMDid);
    return (OK);
  case NUMD_G11:
    value->rValue = *(ckt->CKTstate0 + inst->NUMDconduct);
    return (OK);
  case NUMD_G12:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMDconduct);
    return (OK);
  case NUMD_G21:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMDconduct);
    return (OK);
  case NUMD_G22:
    value->rValue = *(ckt->CKTstate0 + inst->NUMDconduct);
    return (OK);
  case NUMD_C11:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->rValue = inst->NUMDc11;
    return (OK);
  case NUMD_C12:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->rValue = -inst->NUMDc11;
    return (OK);
  case NUMD_C21:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->rValue = -inst->NUMDc11;
    return (OK);
  case NUMD_C22:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->rValue = inst->NUMDc11;
    return (OK);
  case NUMD_Y11:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->cValue.real = inst->NUMDy11r;
    value->cValue.imag = inst->NUMDy11i;
    return (OK);
  case NUMD_Y12:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMDy11r;
    value->cValue.imag = -inst->NUMDy11i;
    return (OK);
  case NUMD_Y21:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMDy11r;
    value->cValue.imag = -inst->NUMDy11i;
    return (OK);
  case NUMD_Y22:
    if (!inst->NUMDsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMDinitSmSig(inst);
    }
    value->cValue.real = inst->NUMDy11r;
    value->cValue.imag = inst->NUMDy11i;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
