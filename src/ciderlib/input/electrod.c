/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/elctdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define UM_TO_CM 1.0e-4

extern int ELCTnewCard(GENcard**,GENmodel*);
extern int ELCTparam(int,IFvalue*,GENcard*);


IFparm ELCTpTable[] = {
  IP("x.low", 	ELCT_X_LOW,	IF_REAL,	"Location of left edge"),
  IP("x.high",	ELCT_X_HIGH,	IF_REAL,	"Location of right edge"),
  IP("y.low",	ELCT_Y_LOW,	IF_REAL,	"Location of top edge"),
  IP("y.high",	ELCT_Y_HIGH,	IF_REAL,	"Location of bottom edge"),
  IP("ix.low", 	ELCT_IX_LOW,	IF_INTEGER,	"Index of left edge"),
  IP("ix.high",	ELCT_IX_HIGH,	IF_INTEGER,	"Index of right edge"),
  IP("iy.low",	ELCT_IY_LOW,	IF_INTEGER,	"Index of top edge"),
  IP("iy.high",	ELCT_IY_HIGH,	IF_INTEGER,	"Index of bottom edge"),
  IP("number",	ELCT_NUMBER,	IF_INTEGER,	"Electrode ID number")
};

IFcardInfo ELCTinfo = {
  "electrode",
  "Location of a contact to the device",
  NUMELEMS(ELCTpTable),
  ELCTpTable,

  ELCTnewCard,
  ELCTparam,
  NULL
};

int
ELCTnewCard(GENcard **inCard, GENmodel *inModel)
{
    ELCTcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(ELCTcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->ELCTnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENelectrodes;
    if (!tmpCard) { /* First in list */
        model->GENelectrodes = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->ELCTnextCard) tmpCard = tmpCard->ELCTnextCard;
	/* And add new card */
	tmpCard->ELCTnextCard = newCard;
    }
    return(OK);
}


int
ELCTparam(int param, IFvalue *value, GENcard *inCard)
{
    ELCTcard *card = (ELCTcard *)inCard;

    switch (param) {
	case ELCT_X_LOW:
	    card->ELCTxLow = value->rValue * UM_TO_CM;
	    card->ELCTxLowGiven = TRUE;
	    break;
	case ELCT_X_HIGH:
	    card->ELCTxHigh = value->rValue * UM_TO_CM;
	    card->ELCTxHighGiven = TRUE;
	    break;
	case ELCT_Y_LOW:
	    card->ELCTyLow = value->rValue * UM_TO_CM;
	    card->ELCTyLowGiven = TRUE;
	    break;
	case ELCT_Y_HIGH:
	    card->ELCTyHigh = value->rValue * UM_TO_CM;
	    card->ELCTyHighGiven = TRUE;
	    break;
	case ELCT_IX_LOW:
	    card->ELCTixLow = value->iValue;
	    card->ELCTixLowGiven = TRUE;
	    break;
	case ELCT_IX_HIGH:
	    card->ELCTixHigh = value->iValue;
	    card->ELCTixHighGiven = TRUE;
	    break;
	case ELCT_IY_LOW:
	    card->ELCTiyLow = value->iValue;
	    card->ELCTiyLowGiven = TRUE;
	    break;
	case ELCT_IY_HIGH:
	    card->ELCTiyHigh = value->iValue;
	    card->ELCTiyHighGiven = TRUE;
	    break;
	case ELCT_NUMBER:
	    card->ELCTnumber = value->iValue;
	    card->ELCTnumberGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
