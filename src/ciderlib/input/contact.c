/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/contdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

extern int CONTnewCard(GENcard**,GENmodel*);
extern int CONTparam(int,IFvalue*,GENcard*);


IFparm CONTpTable[] = {
  IP("neutral",		CONT_NEUTRAL,	IF_FLAG,	"Charge neutral"),
  IP("aluminum",	CONT_ALUMINUM,	IF_FLAG,	"Aluminum contact"),
  IP("p.polysilicon",	CONT_P_POLY,	IF_FLAG,	"P poly contact"),
  IP("n.polysilicon",	CONT_N_POLY,	IF_FLAG,	"N poly contact"),
  IP("workfunction",	CONT_WORKFUN,	IF_REAL,	"Metal work function"),
  IP("number",		CONT_NUMBER,	IF_INTEGER,	"Electrode ID number")
};

IFcardInfo CONTinfo = {
  "contact",
  "Properties of a contact to the device",
  NUMELEMS(CONTpTable),
  CONTpTable,

  CONTnewCard,
  CONTparam,
  NULL
};

int
CONTnewCard(GENcard **inCard, GENmodel *inModel)
{
    CONTcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(CONTcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->CONTnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENcontacts;
    if (!tmpCard) { /* First in list */
        model->GENcontacts = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->CONTnextCard) tmpCard = tmpCard->CONTnextCard;
	/* And add new card */
	tmpCard->CONTnextCard = newCard;
    }
    return(OK);
}

int
CONTparam(int param, IFvalue *value, GENcard *inCard)
{
    CONTcard *card = (CONTcard *)inCard;

    switch (param) {
	case CONT_NEUTRAL:
	    if ( value->iValue ) {
	        card->CONTtype = CONT_NEUTRAL;
	        card->CONTtypeGiven = TRUE;
	    } else {
		if ( card->CONTtype == CONT_NEUTRAL ) {
		    card->CONTtype = -1;
		    card->CONTtypeGiven = FALSE;
		}
	    }
	    break;
	case CONT_ALUMINUM:
	    if ( value->iValue ) {
	        card->CONTtype = CONT_ALUMINUM;
	        card->CONTtypeGiven = TRUE;
	    } else {
		if ( card->CONTtype == CONT_ALUMINUM ) {
		    card->CONTtype = -1;
		    card->CONTtypeGiven = FALSE;
		}
	    }
	    break;
	case CONT_P_POLY:
	    if ( value->iValue ) {
	        card->CONTtype = CONT_P_POLY;
	        card->CONTtypeGiven = TRUE;
	    } else {
		if ( card->CONTtype == CONT_P_POLY ) {
		    card->CONTtype = -1;
		    card->CONTtypeGiven = FALSE;
		}
	    }
	    break;
	case CONT_N_POLY:
	    if ( value->iValue ) {
	        card->CONTtype = CONT_N_POLY;
	        card->CONTtypeGiven = TRUE;
	    } else {
		if ( card->CONTtype == CONT_N_POLY ) {
		    card->CONTtype = -1;
		    card->CONTtypeGiven = FALSE;
		}
	    }
	    break;
	case CONT_WORKFUN:
	    card->CONTtype = CONT_WORKFUN;
	    card->CONTtypeGiven = TRUE;
	    card->CONTworkfun = value->rValue;
	    card->CONTworkfunGiven = TRUE;
	    break;
	case CONT_NUMBER:
	    card->CONTnumber = value->iValue;
	    card->CONTnumberGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
