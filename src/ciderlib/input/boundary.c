/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/bdrydefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define UM_TO_CM 1.0e-4

extern int BDRYnewCard(GENcard**,GENmodel*);
extern int BDRYparam(int,IFvalue*,GENcard*);

IFparm BDRYpTable[] = {
  IP("domain", 	BDRY_DOMAIN,	IF_INTEGER,	"Primary domain"),
  IP("neighbor",BDRY_NEIGHBOR,	IF_INTEGER,	"Neighboring domain"),
  IP("x.low", 	BDRY_X_LOW,	IF_REAL,	"Location of left edge"),
  IP("x.high",	BDRY_X_HIGH,	IF_REAL,	"Location of right edge"),
  IP("y.low",	BDRY_Y_LOW,	IF_REAL,	"Location of top edge"),
  IP("y.high",	BDRY_Y_HIGH,	IF_REAL,	"Location of bottom edge"),
  IP("ix.low", 	BDRY_IX_LOW,	IF_INTEGER,	"Index of left edge"),
  IP("ix.high",	BDRY_IX_HIGH,	IF_INTEGER,	"Index of right edge"),
  IP("iy.low",	BDRY_IY_LOW,	IF_INTEGER,	"Index of top edge"),
  IP("iy.high",	BDRY_IY_HIGH,	IF_INTEGER,	"Index of bottom edge"),
  IP("nss",	BDRY_QF,	IF_REAL,	"Fixed charge"),
  IP("qss",	BDRY_QF,	IF_REAL,	"Fixed charge"),
  IP("qf",	BDRY_QF,	IF_REAL,	"Fixed charge"),
  IP("sn",	BDRY_SN,	IF_REAL,	"Electron recomb velocity"),
  IP("srvn",	BDRY_SN,	IF_REAL,	"Electron recomb velocity"),
  IP("vsrfn",	BDRY_SN,	IF_REAL,	"Electron recomb velocity"),
  IP("sp",	BDRY_SP,	IF_REAL,	"Hole recomb velocity"),
  IP("srvp",	BDRY_SP,	IF_REAL,	"Hole recomb velocity"),
  IP("vsrfp",	BDRY_SP,	IF_REAL,	"Hole recomb velocity"),
  IP("layer.width",BDRY_LAYER,	IF_REAL,	"Width of surface charge layer")
};

IFcardInfo BDRYinfo = {
  "boundary",
  "Specify properties of a domain boundary",
  NUMELEMS(BDRYpTable),
  BDRYpTable,

  BDRYnewCard,
  BDRYparam,
  NULL
};
IFcardInfo INTFinfo = {
  "interface",
  "Specify properties of an interface between two domains",
  NUMELEMS(BDRYpTable),
  BDRYpTable,

  BDRYnewCard,
  BDRYparam,
  NULL
};

int
BDRYnewCard(GENcard **inCard, GENmodel *inModel)
{
    BDRYcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(BDRYcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->BDRYnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENboundaries;
    if (!tmpCard) { /* First in list */
        model->GENboundaries = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->BDRYnextCard) tmpCard = tmpCard->BDRYnextCard;
	/* And add new card */
	tmpCard->BDRYnextCard = newCard;
    }
    return(OK);
}

int
BDRYparam(int param, IFvalue *value, GENcard *inCard)
{
    BDRYcard *card = (BDRYcard *)inCard;

    switch (param) {
	case BDRY_DOMAIN:
	    card->BDRYdomain = value->iValue;
	    card->BDRYdomainGiven = TRUE;
	    break;
	case BDRY_NEIGHBOR:
	    card->BDRYneighbor = value->iValue;
	    card->BDRYneighborGiven = TRUE;
	    break;
	case BDRY_X_LOW:
	    card->BDRYxLow = value->rValue * UM_TO_CM;
	    card->BDRYxLowGiven = TRUE;
	    break;
	case BDRY_X_HIGH:
	    card->BDRYxHigh = value->rValue * UM_TO_CM;
	    card->BDRYxHighGiven = TRUE;
	    break;
	case BDRY_Y_LOW:
	    card->BDRYyLow = value->rValue * UM_TO_CM;
	    card->BDRYyLowGiven = TRUE;
	    break;
	case BDRY_Y_HIGH:
	    card->BDRYyHigh = value->rValue * UM_TO_CM;
	    card->BDRYyHighGiven = TRUE;
	    break;
	case BDRY_IX_LOW:
	    card->BDRYixLow = value->iValue;
	    card->BDRYixLowGiven = TRUE;
	    break;
	case BDRY_IX_HIGH:
	    card->BDRYixHigh = value->iValue;
	    card->BDRYixHighGiven = TRUE;
	    break;
	case BDRY_IY_LOW:
	    card->BDRYiyLow = value->iValue;
	    card->BDRYiyLowGiven = TRUE;
	    break;
	case BDRY_IY_HIGH:
	    card->BDRYiyHigh = value->iValue;
	    card->BDRYiyHighGiven = TRUE;
	    break;
	case BDRY_QF:
	    card->BDRYqf = value->rValue;
	    card->BDRYqfGiven = TRUE;
	    break;
	case BDRY_SN:
	    card->BDRYsn = value->rValue;
	    card->BDRYsnGiven = TRUE;
	    break;
	case BDRY_SP:
	    card->BDRYsp = value->rValue;
	    card->BDRYspGiven = TRUE;
	    break;
	case BDRY_LAYER:
	    card->BDRYlayer = value->rValue;
	    card->BDRYlayerGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
