/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* Externals Declarations */
extern int NUMOSinitSmSig(NUMOSinstance *);

/* ARGSUSED */
int
NUMOSask(CKTcircuit *ckt, GENinstance *inInst, int which, IFvalue *value, IFvalue *select)
{
  NUMOSinstance *inst = (NUMOSinstance *) inInst;

  NG_IGNORE(select);

  switch (which) {
  case NUMOS_AREA:
    value->rValue = inst->NUMOSarea;
    return (OK);
  case NUMOS_WIDTH:
    value->rValue = inst->NUMOSwidth;
    return (OK);
  case NUMOS_LENGTH:
    value->rValue = inst->NUMOSlength;
    return (OK);
  case NUMOS_TEMP:
    value->rValue = inst->NUMOStemp - CONSTCtoK;
    return (OK);
  case NUMOS_G11:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIdDVdb);
    return (OK);
  case NUMOS_G12:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIdDVgb);
    return (OK);
  case NUMOS_G13:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIdDVsb);
    return (OK);
  case NUMOS_G14:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIdDVdb)
	- *(ckt->CKTstate0 + inst->NUMOSdIdDVgb)
	- *(ckt->CKTstate0 + inst->NUMOSdIdDVsb);
    return (OK);
  case NUMOS_G21:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIgDVdb);
    return (OK);
  case NUMOS_G22:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIgDVgb);
    return (OK);
  case NUMOS_G23:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIgDVsb);
    return (OK);
  case NUMOS_G24:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIgDVdb)
	- *(ckt->CKTstate0 + inst->NUMOSdIgDVgb)
	- *(ckt->CKTstate0 + inst->NUMOSdIgDVsb);
    return (OK);
  case NUMOS_G31:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIsDVdb);
    return (OK);
  case NUMOS_G32:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIsDVgb);
    return (OK);
  case NUMOS_G33:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIsDVsb);
    return (OK);
  case NUMOS_G34:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIsDVdb)
	- *(ckt->CKTstate0 + inst->NUMOSdIsDVgb)
	- *(ckt->CKTstate0 + inst->NUMOSdIsDVsb);
    return (OK);
  case NUMOS_G41:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIdDVdb)
	- *(ckt->CKTstate0 + inst->NUMOSdIgDVdb)
	- *(ckt->CKTstate0 + inst->NUMOSdIsDVdb);
    return (OK);
  case NUMOS_G42:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIdDVgb)
	- *(ckt->CKTstate0 + inst->NUMOSdIgDVgb)
	- *(ckt->CKTstate0 + inst->NUMOSdIsDVgb);
    return (OK);
  case NUMOS_G43:
    value->rValue = -*(ckt->CKTstate0 + inst->NUMOSdIdDVsb)
	- *(ckt->CKTstate0 + inst->NUMOSdIgDVsb)
	- *(ckt->CKTstate0 + inst->NUMOSdIsDVsb);
    return (OK);
  case NUMOS_G44:
    value->rValue = *(ckt->CKTstate0 + inst->NUMOSdIdDVdb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIgDVdb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIsDVdb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIdDVgb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIgDVgb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIsDVgb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIdDVsb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIgDVsb)
	+ *(ckt->CKTstate0 + inst->NUMOSdIsDVsb);
    return (OK);
  case NUMOS_C11:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc11;
    return (OK);
  case NUMOS_C12:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc12;
    return (OK);
  case NUMOS_C13:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc13;
    return (OK);
  case NUMOS_C14:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc11 - inst->NUMOSc12 - inst->NUMOSc13;
    return (OK);
  case NUMOS_C21:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc21;
    return (OK);
  case NUMOS_C22:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc22;
    return (OK);
  case NUMOS_C23:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc23;
    return (OK);
  case NUMOS_C24:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc21 - inst->NUMOSc22 - inst->NUMOSc23;
    return (OK);
  case NUMOS_C31:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc31;
    return (OK);
  case NUMOS_C32:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc32;
    return (OK);
  case NUMOS_C33:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc33;
    return (OK);
  case NUMOS_C34:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc31 - inst->NUMOSc32 - inst->NUMOSc33;
    return (OK);
  case NUMOS_C41:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc11 - inst->NUMOSc21 - inst->NUMOSc31;
    return (OK);
  case NUMOS_C42:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc12 - inst->NUMOSc22 - inst->NUMOSc32;
    return (OK);
  case NUMOS_C43:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = -inst->NUMOSc13 - inst->NUMOSc23 - inst->NUMOSc33;
    return (OK);
  case NUMOS_C44:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->rValue = inst->NUMOSc11 + inst->NUMOSc12 + inst->NUMOSc13
	+ inst->NUMOSc21 + inst->NUMOSc22 + inst->NUMOSc23
	+ inst->NUMOSc31 + inst->NUMOSc32 + inst->NUMOSc33;
    return (OK);
  case NUMOS_Y11:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy11r;
    value->cValue.imag = inst->NUMOSy11i;
    return (OK);
  case NUMOS_Y12:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy12r;
    value->cValue.imag = inst->NUMOSy12i;
    return (OK);
  case NUMOS_Y13:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy13r;
    value->cValue.imag = inst->NUMOSy13i;
    return (OK);
  case NUMOS_Y14:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy11r - inst->NUMOSy11r - inst->NUMOSy11r;
    value->cValue.imag = -inst->NUMOSy11i - inst->NUMOSy11i - inst->NUMOSy11i;
    return (OK);
  case NUMOS_Y21:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy21r;
    value->cValue.imag = inst->NUMOSy21i;
    return (OK);
  case NUMOS_Y22:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy22r;
    value->cValue.imag = inst->NUMOSy22i;
    return (OK);
  case NUMOS_Y23:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy23r;
    value->cValue.imag = inst->NUMOSy23i;
    return (OK);
  case NUMOS_Y24:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy21r - inst->NUMOSy22r - inst->NUMOSy23r;
    value->cValue.imag = -inst->NUMOSy21i - inst->NUMOSy22i - inst->NUMOSy23i;
    return (OK);
  case NUMOS_Y31:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy31r;
    value->cValue.imag = inst->NUMOSy31i;
    return (OK);
  case NUMOS_Y32:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy32r;
    value->cValue.imag = inst->NUMOSy32i;
    return (OK);
  case NUMOS_Y33:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy33r;
    value->cValue.imag = inst->NUMOSy33i;
    return (OK);
  case NUMOS_Y34:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy31r - inst->NUMOSy32r - inst->NUMOSy33r;
    value->cValue.imag = -inst->NUMOSy31i - inst->NUMOSy32i - inst->NUMOSy33i;
    return (OK);
  case NUMOS_Y41:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy11r - inst->NUMOSy21r - inst->NUMOSy31r;
    value->cValue.imag = -inst->NUMOSy11i - inst->NUMOSy21i - inst->NUMOSy31i;
    return (OK);
  case NUMOS_Y42:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy12r - inst->NUMOSy22r - inst->NUMOSy32r;
    value->cValue.imag = -inst->NUMOSy12i - inst->NUMOSy22i - inst->NUMOSy32i;
    return (OK);
  case NUMOS_Y43:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = -inst->NUMOSy13r - inst->NUMOSy23r - inst->NUMOSy33r;
    value->cValue.imag = -inst->NUMOSy13i - inst->NUMOSy23i - inst->NUMOSy33i;
    return (OK);
  case NUMOS_Y44:
    if (!inst->NUMOSsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NUMOSinitSmSig(inst);
    }
    value->cValue.real = inst->NUMOSy11r + inst->NUMOSy21r + inst->NUMOSy31r
	+ inst->NUMOSy12r + inst->NUMOSy22r + inst->NUMOSy32r
	+ inst->NUMOSy13r + inst->NUMOSy23r + inst->NUMOSy33r;
    value->cValue.imag = inst->NUMOSy11i + inst->NUMOSy21i + inst->NUMOSy31i
	+ inst->NUMOSy12i + inst->NUMOSy22i + inst->NUMOSy32i
	+ inst->NUMOSy13i + inst->NUMOSy23i + inst->NUMOSy33i;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
