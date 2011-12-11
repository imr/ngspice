/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* Check out this one */
extern int NBJT2initSmSig(NBJT2instance *);


/* ARGSUSED */
int
NBJT2ask(CKTcircuit *ckt, GENinstance *inInst, int which, IFvalue *value, IFvalue *select)
{
  NBJT2instance *inst = (NBJT2instance *) inInst;

  NG_IGNORE(select);

  switch (which) {
  case NBJT2_WIDTH:
    value->rValue = inst->NBJT2width;
    return (OK);
  case NBJT2_AREA:
    value->rValue = inst->NBJT2area;
    return (OK);
  case NBJT2_TEMP:
    value->rValue = inst->NBJT2temp - CONSTCtoK;
    return (OK);
  case NBJT2_G11:
    value->rValue = *(ckt->CKTstate0 + inst->NBJT2dIcDVce);
    return (OK);
  case NBJT2_G12:
    value->rValue = *(ckt->CKTstate0 + inst->NBJT2dIcDVbe);
    return (OK);
  case NBJT2_G13:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJT2dIcDVce)
	- *(ckt->CKTstate0 + inst->NBJT2dIcDVbe);
    return (OK);
  case NBJT2_G21:
    value->rValue = *(ckt->CKTstate0 + inst->NBJT2dIeDVce)
	- *(ckt->CKTstate0 + inst->NBJT2dIcDVce);
    return (OK);
  case NBJT2_G22:
    value->rValue = *(ckt->CKTstate0 + inst->NBJT2dIeDVbe)
	- *(ckt->CKTstate0 + inst->NBJT2dIcDVbe);
    return (OK);
  case NBJT2_G23:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJT2dIeDVce)
	+ *(ckt->CKTstate0 + inst->NBJT2dIcDVce)
    -*(ckt->CKTstate0 + inst->NBJT2dIeDVbe)
	+ *(ckt->CKTstate0 + inst->NBJT2dIcDVbe);
    return (OK);
  case NBJT2_G31:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJT2dIeDVce);
    return (OK);
  case NBJT2_G32:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJT2dIeDVbe);
    return (OK);
  case NBJT2_G33:
    value->rValue = *(ckt->CKTstate0 + inst->NBJT2dIeDVce)
	+ *(ckt->CKTstate0 + inst->NBJT2dIeDVbe);
    return (OK);
  case NBJT2_C11:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = inst->NBJT2c11;
    return (OK);
  case NBJT2_C12:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = inst->NBJT2c12;
    return (OK);
  case NBJT2_C13:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = -inst->NBJT2c11 - inst->NBJT2c12;
    return (OK);
  case NBJT2_C21:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = inst->NBJT2c21;
    return (OK);
  case NBJT2_C22:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = inst->NBJT2c22;
    return (OK);
  case NBJT2_C23:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = -inst->NBJT2c21 - inst->NBJT2c22;
    return (OK);
  case NBJT2_C31:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = -inst->NBJT2c11 - inst->NBJT2c21;
    return (OK);
  case NBJT2_C32:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = -inst->NBJT2c12 - inst->NBJT2c22;
    return (OK);
  case NBJT2_C33:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->rValue = inst->NBJT2c11 + inst->NBJT2c21
	+ inst->NBJT2c12 + inst->NBJT2c22;
    return (OK);
  case NBJT2_Y11:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = inst->NBJT2y11r;
    value->cValue.imag = inst->NBJT2y11i;
    return (OK);
  case NBJT2_Y12:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = inst->NBJT2y12r;
    value->cValue.imag = inst->NBJT2y12i;
    return (OK);
  case NBJT2_Y13:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = -inst->NBJT2y11r - inst->NBJT2y12r;
    value->cValue.imag = -inst->NBJT2y11i - inst->NBJT2y12i;
    return (OK);
  case NBJT2_Y21:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = inst->NBJT2y21r;
    value->cValue.imag = inst->NBJT2y21i;
    return (OK);
  case NBJT2_Y22:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = inst->NBJT2y22r;
    value->cValue.imag = inst->NBJT2y22i;
    return (OK);
  case NBJT2_Y23:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = -inst->NBJT2y21r - inst->NBJT2y22r;
    value->cValue.imag = -inst->NBJT2y21i - inst->NBJT2y22i;
    return (OK);
  case NBJT2_Y31:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = -inst->NBJT2y11r - inst->NBJT2y21r;
    value->cValue.imag = -inst->NBJT2y11i - inst->NBJT2y21i;
    return (OK);
  case NBJT2_Y32:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = -inst->NBJT2y12r - inst->NBJT2y22r;
    value->cValue.imag = -inst->NBJT2y12i - inst->NBJT2y22i;
    return (OK);
  case NBJT2_Y33:
    if (!inst->NBJT2smSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJT2initSmSig(inst);
    }
    value->cValue.real = inst->NBJT2y11r + inst->NBJT2y21r
	+ inst->NBJT2y12r + inst->NBJT2y22r;
    value->cValue.imag = inst->NBJT2y11i + inst->NBJT2y21i
	+ inst->NBJT2y12i + inst->NBJT2y22i;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
