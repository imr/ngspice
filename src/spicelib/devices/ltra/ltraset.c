/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "ltradefs.h"
#include "sperror.h"
#include "suffix.h"

int
LTRAsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *state)
/*
 * load the transmission line structure with those pointers needed later for
 * fast matrix loading
 */
{
  LTRAmodel *model = (LTRAmodel *) inModel;
  LTRAinstance *here;
  int error;
  CKTnode *tmp;

  /* loop through all the transmission line models */
  for (; model != NULL; model = model->LTRAnextModel) {

    if (!model->LTRAnlGiven) {
      model->LTRAnl = .25;
    }
    if (!model->LTRAfGiven) {
      model->LTRAf = 1e9;
    }
    if (!model->LTRAreltolGiven) {
      model->LTRAreltol = 1;
    }
    if (!model->LTRAabstolGiven) {
      model->LTRAabstol = 1;
    }
    if (!model->LTRAresistGiven) {
      (*(SPfrontEnd->IFerror)) (ERR_WARNING,
	  "%s: lossy line series resistance not given, assumed zero",
	  &(model->LTRAmodName));
      model->LTRAresist = 0.0;
      /* return(E_BADPARM); */
    }
    if (model->LTRAstLineReltol == 0.0)
      model->LTRAstLineReltol = ckt->CKTreltol;
    if (model->LTRAstLineAbstol == 0.0)
      model->LTRAstLineAbstol = ckt->CKTabstol;
    /* LTRAchopReltol and LTRAchopAbstol default zero */

    if ((model->LTRAhowToInterp != LTRA_MOD_LININTERP) &&
	(model->LTRAhowToInterp != LTRA_MOD_QUADINTERP) &&
	(model->LTRAhowToInterp != LTRA_MOD_MIXEDINTERP)) {

      /*
       * (*(SPfrontEnd->IFerror))(ERR_FATAL, "%s: have to specify one of
       * lininterp, quadinterp or mixedinterp", &(model->LTRAmodName));
       * return(E_BADPARM);
       */
      if (ckt->CKTtryToCompact) {
	model->LTRAhowToInterp = LTRA_MOD_LININTERP;
	(*(SPfrontEnd->IFerror)) (ERR_WARNING,
	    "%s: using linear interpolation because trytocompact option specified",
	    &(model->LTRAmodName));
      } else {
	model->LTRAhowToInterp = LTRA_MOD_QUADINTERP;
      }
    }
    if ((model->LTRAstepLimit != LTRA_MOD_NOSTEPLIMIT))
      model->LTRAstepLimit = LTRA_MOD_STEPLIMIT;
    if ((model->LTRAlteConType != LTRA_MOD_FULLCONTROL) &&
	(model->LTRAlteConType != LTRA_MOD_HALFCONTROL))
      model->LTRAlteConType = LTRA_MOD_NOCONTROL;

    if (!model->LTRAconductGiven) {
      /*
       * (*(SPfrontEnd->IFerror))(ERR_WARNING, "%s: lossy line parallel
       * conductance not given, assumed zero", &(model->LTRAmodName));
       */
      model->LTRAconduct = 0.0;
      /* return(E_BADPARM); */
    }
    if (!model->LTRAinductGiven) {
      (*(SPfrontEnd->IFerror)) (ERR_WARNING,
	  "%s: lossy line series inductance not given, assumed zero",
	  &(model->LTRAmodName));
      model->LTRAinduct = 0.0;
      /* return(E_BADPARM); */
    }
    if (!model->LTRAcapacGiven) {
      (*(SPfrontEnd->IFerror)) (ERR_FATAL,
	  "%s: lossy line parallel capacitance not given, assumed zero",
	  &(model->LTRAmodName));
      model->LTRAcapac = 0.0;
      /* return(E_BADPARM); */
    }
    if (!model->LTRAlengthGiven) {
      (*(SPfrontEnd->IFerror)) (ERR_FATAL,
	  "%s: lossy line length must be given",
	  &(model->LTRAmodName));
      return (E_BADPARM);
    }
    if ((model->LTRAresist == 0) && (model->LTRAconduct == 0) &&
	(model->LTRAcapac != 0) && (model->LTRAinduct != 0)) {
      model->LTRAspecialCase = LTRA_MOD_LC;
#ifdef LTRADEBUG
      (*(SPfrontEnd->IFerror)) (ERR_INFO,
	  "%s: lossless line",
	  &(model->LTRAmodName));
#endif
    }
    if ((model->LTRAresist != 0) && (model->LTRAconduct == 0) &&
	(model->LTRAcapac != 0) && (model->LTRAinduct != 0)) {
      model->LTRAspecialCase = LTRA_MOD_RLC;
#ifdef LTRADEBUG
      (*(SPfrontEnd->IFerror)) (ERR_INFO,
	  "%s: RLC line",
	  &(model->LTRAmodName));
#endif
    }
    if ((model->LTRAresist != 0) && (model->LTRAconduct == 0) &&
	(model->LTRAcapac != 0) && (model->LTRAinduct == 0)) {
      model->LTRAspecialCase = LTRA_MOD_RC;
#ifdef LTRADEBUG
      (*(SPfrontEnd->IFerror)) (ERR_INFO,
	  "%s: RC line",
	  &(model->LTRAmodName));
#endif
    }
    if ((model->LTRAresist != 0) && (model->LTRAconduct == 0) &&
	(model->LTRAcapac == 0) && (model->LTRAinduct != 0)) {
      model->LTRAspecialCase = LTRA_MOD_RL;
      (*(SPfrontEnd->IFerror)) (ERR_FATAL,
	  "%s: RL line not supported yet",
	  &(model->LTRAmodName));
      return (E_BADPARM);
#ifdef LTRADEBUG
#endif
    }
    if ((model->LTRAresist != 0) && (model->LTRAconduct != 0) &&
	(model->LTRAcapac == 0) && (model->LTRAinduct == 0)) {
      model->LTRAspecialCase = LTRA_MOD_RG;
#ifdef LTRADEBUG
      (*(SPfrontEnd->IFerror)) (ERR_INFO,
	  "%s: RG line",
	  &(model->LTRAmodName));
#endif
    }
    if ((model->LTRAconduct != 0) && ((model->LTRAcapac != 0) ||
	    (model->LTRAinduct != 0))) {
      model->LTRAspecialCase = LTRA_MOD_LTRA;
      (*(SPfrontEnd->IFerror)) (ERR_FATAL,
	  "%s: Nonzero G (except RG) line not supported yet",
	  &(model->LTRAmodName));
      return (E_BADPARM);
#ifdef LTRADEBUG
#endif
    }
    if ((model->LTRAresist == 0.0 ? 0 : 1) + (model->LTRAconduct
	    == 0.0 ? 0 : 1) + (model->LTRAinduct == 0.0 ? 0 : 1) +
	(model->LTRAcapac == 0.0 ? 0 : 1) <= 1) {
      (*(SPfrontEnd->IFerror)) (ERR_FATAL,
	  "%s: At least two of R,L,G,C must be specified and nonzero",
	  &(model->LTRAmodName));
      return (E_BADPARM);
    }
    /* loop through all the instances of the model */
    for (here = model->LTRAinstances; here != NULL;
	here = here->LTRAnextInstance) {

      if (here->LTRAbrEq1 == 0) {
	error = CKTmkVolt(ckt, &tmp, here->LTRAname, "i1");
	if (error)
	  return (error);
	here->LTRAbrEq1 = tmp->number;
      }
      if (here->LTRAbrEq2 == 0) {
	error = CKTmkVolt(ckt, &tmp, here->LTRAname, "i2");
	if (error)
	  return (error);
	here->LTRAbrEq2 = tmp->number;
      }
      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

      TSTALLOC(LTRAibr1Pos1Ptr, LTRAbrEq1, LTRAposNode1)
      TSTALLOC(LTRAibr1Neg1Ptr, LTRAbrEq1, LTRAnegNode1)
      TSTALLOC(LTRAibr1Pos2Ptr, LTRAbrEq1, LTRAposNode2)
      TSTALLOC(LTRAibr1Neg2Ptr, LTRAbrEq1, LTRAnegNode2)
      TSTALLOC(LTRAibr1Ibr1Ptr, LTRAbrEq1, LTRAbrEq1)
      TSTALLOC(LTRAibr1Ibr2Ptr, LTRAbrEq1, LTRAbrEq2)
      TSTALLOC(LTRAibr2Pos1Ptr, LTRAbrEq2, LTRAposNode1)
      TSTALLOC(LTRAibr2Neg1Ptr, LTRAbrEq2, LTRAnegNode1)
      TSTALLOC(LTRAibr2Pos2Ptr, LTRAbrEq2, LTRAposNode2)
      TSTALLOC(LTRAibr2Neg2Ptr, LTRAbrEq2, LTRAnegNode2)
      TSTALLOC(LTRAibr2Ibr1Ptr, LTRAbrEq2, LTRAbrEq1)
      TSTALLOC(LTRAibr2Ibr2Ptr, LTRAbrEq2, LTRAbrEq2)
      TSTALLOC(LTRApos1Ibr1Ptr, LTRAposNode1, LTRAbrEq1)
      TSTALLOC(LTRAneg1Ibr1Ptr, LTRAnegNode1, LTRAbrEq1)
      TSTALLOC(LTRApos2Ibr2Ptr, LTRAposNode2, LTRAbrEq2)
      TSTALLOC(LTRAneg2Ibr2Ptr, LTRAnegNode2, LTRAbrEq2)
      /*
       * the following are done so that SMPpreOrder does not screw up on
       * occasion - for example, when one end of the lossy line is hanging
       */
      TSTALLOC(LTRApos1Pos1Ptr, LTRAposNode1, LTRAposNode1)
      TSTALLOC(LTRAneg1Neg1Ptr, LTRAnegNode1, LTRAnegNode1)
      TSTALLOC(LTRApos2Pos2Ptr, LTRAposNode2, LTRAposNode2)
      TSTALLOC(LTRAneg2Neg2Ptr, LTRAnegNode2, LTRAnegNode2)
    }
  }
  return (OK);
}

int
LTRAunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
  LTRAmodel *model;
  LTRAinstance *here;

  for (model = (LTRAmodel *) inModel; model != NULL;
      model = model->LTRAnextModel) {
    for (here = model->LTRAinstances; here != NULL;
	here = here->LTRAnextInstance) {
      if (here->LTRAbrEq1) {
	CKTdltNNum(ckt, here->LTRAbrEq1);
	here->LTRAbrEq1 = 0;
      }
      if (here->LTRAbrEq2) {
	CKTdltNNum(ckt, here->LTRAbrEq2);
	here->LTRAbrEq2 = 0;
      }
    }
  }
  return OK;
}
