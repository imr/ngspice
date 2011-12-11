/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* Check out this one */
extern int NBJTinitSmSig(NBJTinstance *);


/* ARGSUSED */
int
NBJTask(CKTcircuit *ckt, GENinstance *inInst, int which, IFvalue *value, IFvalue *select)
{
  NBJTinstance *inst = (NBJTinstance *) inInst;

  NG_IGNORE(select);

  switch (which) {
  case NBJT_AREA:
    value->rValue = inst->NBJTarea;
    return (OK);
  case NBJT_TEMP:
    value->rValue = inst->NBJTtemp - CONSTCtoK;
    return (OK);
  case NBJT_G11:
    value->rValue = *(ckt->CKTstate0 + inst->NBJTdIcDVce);
    return (OK);
  case NBJT_G12:
    value->rValue = *(ckt->CKTstate0 + inst->NBJTdIcDVbe);
    return (OK);
  case NBJT_G13:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJTdIcDVce)
	- *(ckt->CKTstate0 + inst->NBJTdIcDVbe);
    return (OK);
  case NBJT_G21:
    value->rValue = *(ckt->CKTstate0 + inst->NBJTdIeDVce)
	- *(ckt->CKTstate0 + inst->NBJTdIcDVce);
    return (OK);
  case NBJT_G22:
    value->rValue = *(ckt->CKTstate0 + inst->NBJTdIeDVbe)
	- *(ckt->CKTstate0 + inst->NBJTdIcDVbe);
    return (OK);
  case NBJT_G23:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJTdIeDVce)
	+ *(ckt->CKTstate0 + inst->NBJTdIcDVce)  /* XXX  there was a ;*/
    -*(ckt->CKTstate0 + inst->NBJTdIeDVbe)
	+ *(ckt->CKTstate0 + inst->NBJTdIcDVbe);
    return (OK);
  case NBJT_G31:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJTdIeDVce);
    return (OK);
  case NBJT_G32:
    value->rValue = -*(ckt->CKTstate0 + inst->NBJTdIeDVbe);
    return (OK);
  case NBJT_G33:
    value->rValue = *(ckt->CKTstate0 + inst->NBJTdIeDVce)
	+ *(ckt->CKTstate0 + inst->NBJTdIeDVbe);
    return (OK);
  case NBJT_C11:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = inst->NBJTc11;
    return (OK);
  case NBJT_C12:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = inst->NBJTc12;
    return (OK);
  case NBJT_C13:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = -inst->NBJTc11 - inst->NBJTc12;
    return (OK);
  case NBJT_C21:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = inst->NBJTc21;
    return (OK);
  case NBJT_C22:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = inst->NBJTc22;
    return (OK);
  case NBJT_C23:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = -inst->NBJTc21 - inst->NBJTc22;
    return (OK);
  case NBJT_C31:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = -inst->NBJTc11 - inst->NBJTc21;
    return (OK);
  case NBJT_C32:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = -inst->NBJTc12 - inst->NBJTc22;
    return (OK);
  case NBJT_C33:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->rValue = inst->NBJTc11 + inst->NBJTc21
	+ inst->NBJTc12 + inst->NBJTc22;
    return (OK);
  case NBJT_Y11:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = inst->NBJTy11r;
    value->cValue.imag = inst->NBJTy11i;
    return (OK);
  case NBJT_Y12:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = inst->NBJTy12r;
    value->cValue.imag = inst->NBJTy12i;
    return (OK);
  case NBJT_Y13:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = -inst->NBJTy11r - inst->NBJTy12r;
    value->cValue.imag = -inst->NBJTy11i - inst->NBJTy12i;
    return (OK);
  case NBJT_Y21:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = inst->NBJTy21r;
    value->cValue.imag = inst->NBJTy21i;
    return (OK);
  case NBJT_Y22:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = inst->NBJTy22r;
    value->cValue.imag = inst->NBJTy22i;
    return (OK);
  case NBJT_Y23:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = -inst->NBJTy21r - inst->NBJTy22r;
    value->cValue.imag = -inst->NBJTy21i - inst->NBJTy22i;
    return (OK);
  case NBJT_Y31:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = -inst->NBJTy11r - inst->NBJTy21r;
    value->cValue.imag = -inst->NBJTy11i - inst->NBJTy21i;
    return (OK);
  case NBJT_Y32:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = -inst->NBJTy12r - inst->NBJTy22r;
    value->cValue.imag = -inst->NBJTy12i - inst->NBJTy22i;
    return (OK);
  case NBJT_Y33:
    if (!inst->NBJTsmSigAvail
	&& ckt->CKTcurrentAnalysis != DOING_TRAN) {
      NBJTinitSmSig(inst);
    }
    value->cValue.real = inst->NBJTy11r + inst->NBJTy21r
	+ inst->NBJTy12r + inst->NBJTy22r;
    value->cValue.imag = inst->NBJTy11i + inst->NBJTy21i
	+ inst->NBJTy12i + inst->NBJTy22i;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
