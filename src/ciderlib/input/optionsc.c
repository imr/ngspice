/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/optndefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define M_TO_CM 1.0e2
#define M2_TO_CM2 (M_TO_CM * M_TO_CM)
#define UM_TO_CM 1.0e-4
#define UM2_TO_CM2 (UM_TO_CM * UM_TO_CM)

extern int OPTNnewCard(GENcard**,GENmodel*);
extern int OPTNparam(int,IFvalue*,GENcard*);



IFparm OPTNpTable[] = {
/* Supported Types of Devices. Ideally should be automatically extracted */
  IP("resistor",OPTN_RESISTOR,   IF_FLAG,    "Resistor"),
  IP("capacitor",OPTN_CAPACITOR, IF_FLAG,    "Capacitor"),
  IP("diode",   OPTN_DIODE,	IF_FLAG,    "Diode"),
  IP("bipolar", OPTN_BIPOLAR,	IF_FLAG,    "Bipolar Transistor"),
  IP("bjt",     OPTN_BIPOLAR,	IF_FLAG,    "Bipolar Transistor"),
  IP("soibjt",   OPTN_SOIBJT,	IF_FLAG,    "SOI Bipolar"),
  IP("moscap",  OPTN_MOSCAP,     IF_FLAG,    "MOS Capacitor"),
  IP("mosfet",	OPTN_MOSFET,     IF_FLAG,    "MOSFET"),
  IP("soimos",  OPTN_SOIMOS,     IF_FLAG,    "SOI MOSFET"),
  IP("jfet",    OPTN_JFET,	IF_FLAG,    "Junction FET"),
  IP("mesfet",  OPTN_MESFET,	IF_FLAG,    "MESFET"),
/* Various layout dimensions */
  IP("defa",	OPTN_DEFA,	IF_REAL,    "Default Mask Area"),
  IP("defw",	OPTN_DEFW,	IF_REAL,    "Default Mask Width"),
  IP("defl",	OPTN_DEFL,	IF_REAL,    "Default Mask Length"),
  IP("base.area",OPTN_BASE_AREA, IF_REAL,    "1D BJT Base Area"),
  IP("base.length",OPTN_BASE_LENGTH,IF_REAL, "1D BJT Base Length"),
  IP("base.depth",OPTN_BASE_DEPTH, IF_REAL,  "1D BJT Base Depth"),
/* Values */
  IP("tnom",	OPTN_TNOM,	IF_REAL,    "Nominal Temperature"),
/* Device Initial Condition File */
  IP("ic.file", OPTN_IC_FILE,   IF_STRING,  "Initial condition file"),
  IP("unique",  OPTN_UNIQUE,    IF_FLAG,    "Generate unique filename")
};

IFcardInfo OPTNinfo = {
  "options",
  "Provide optional information and hints",
  NUMELEMS(OPTNpTable),
  OPTNpTable,

  OPTNnewCard,
  OPTNparam,
  NULL
};

int
OPTNnewCard(GENcard **inCard, GENmodel *inModel)
{
    OPTNcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    tmpCard = model->GENoptions;
    if (!tmpCard) { /* First in list */
        newCard = TMALLOC(OPTNcard, 1);
        if (!newCard) {
            *inCard = NULL;
            return(E_NOMEM);
        }
        newCard->OPTNnextCard = NULL;
        *inCard = (GENcard *) newCard;
        model->GENoptions = newCard;
    } else { /* Only one card of this type allowed */
	*inCard = (GENcard *) tmpCard;
    }
    return(OK);
}

int
OPTNparam(int param, IFvalue *value, GENcard *inCard)
{
    OPTNcard *card = (OPTNcard *)inCard;

    switch (param) {
	case OPTN_RESISTOR:
	    card->OPTNdeviceType = OPTN_RESISTOR;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_CAPACITOR:
	    card->OPTNdeviceType = OPTN_CAPACITOR;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_DIODE:
	    card->OPTNdeviceType = OPTN_DIODE;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_MOSCAP:
	    card->OPTNdeviceType = OPTN_MOSCAP;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_BIPOLAR:
	case OPTN_SOIBJT:			/* XXX Treat SOI as normal */
	    card->OPTNdeviceType = OPTN_BIPOLAR;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_MOSFET:
	case OPTN_SOIMOS:			/* XXX Treat SOI as normal */
	    card->OPTNdeviceType = OPTN_MOSFET;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_JFET:
	case OPTN_MESFET:			/* XXX Treat MES as junction */
	    card->OPTNdeviceType = OPTN_JFET;
	    card->OPTNdeviceTypeGiven = TRUE;
	    break;
	case OPTN_DEFA:
	    card->OPTNdefa = value->rValue * M2_TO_CM2;
	    card->OPTNdefaGiven = TRUE;
	    break;
	case OPTN_DEFW:
	    card->OPTNdefw = value->rValue * M_TO_CM;
	    card->OPTNdefwGiven = TRUE;
	    break;
	case OPTN_DEFL:
	    card->OPTNdefl = value->rValue * M_TO_CM;
	    card->OPTNdeflGiven = TRUE;
	    break;
	case OPTN_BASE_AREA:
	    card->OPTNbaseArea = value->rValue;
	    card->OPTNbaseAreaGiven = TRUE;
	    break;
	case OPTN_BASE_LENGTH:
	    card->OPTNbaseLength = value->rValue * UM_TO_CM;
	    card->OPTNbaseLengthGiven = TRUE;
	    break;
	case OPTN_BASE_DEPTH:
	    card->OPTNbaseDepth = value->rValue * UM_TO_CM;
	    card->OPTNbaseDepthGiven = TRUE;
	    break;
	case OPTN_TNOM:
	    card->OPTNtnom = value->rValue;
	    card->OPTNtnomGiven = TRUE;
	    break;
	case OPTN_IC_FILE:
	    card->OPTNicFile = value->sValue;
	    card->OPTNicFileGiven = TRUE;
	    break;
	case OPTN_UNIQUE:
	    card->OPTNunique = value->iValue;
	    card->OPTNuniqueGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
