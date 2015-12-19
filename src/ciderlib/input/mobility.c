/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/numenum.h"
#include "ngspice/mobdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

extern int MOBnewCard(GENcard**,GENmodel*);
extern int MOBparam(int,IFvalue*,GENcard*);



IFparm MOBpTable[] = {
  IP("material",MOB_MATERIAL,	IF_INTEGER, "Material index"),
  IP("electron",MOB_ELEC,	IF_FLAG,    "Electron mobility flag"),
  IP("hole",    MOB_HOLE,	IF_FLAG,    "Hole mobility flag"),
  IP("majority",MOB_MAJOR,	IF_FLAG,    "Majority carrier flag"),
  IP("minority",MOB_MINOR,	IF_FLAG,    "Minority carrier flag"),
  IP("mumax",	MOB_MUMAX,	IF_REAL,    "Maximum bulk mobility"),
  IP("mumin",	MOB_MUMIN,	IF_REAL,    "Minimum bulk mobility"),
  IP("ntref",   MOB_NTREF,	IF_REAL,    "Ionized impurity ref. conc."),
  IP("ntexp",   MOB_NTEXP,	IF_REAL,    "Ionized impurity exponent"),
  IP("vsat",	MOB_VSAT,	IF_REAL,    "Saturation velocity"),
  IP("vwarm",   MOB_VWARM,	IF_REAL,    "Warm carrier ref. velocity"),
  IP("mus",	MOB_MUS,	IF_REAL,    "Initial surface mobility"),
  IP("ec.a",	MOB_EC_A,	IF_REAL,    "Surf. mobility critical field"),
  IP("ec.b",	MOB_EC_B,	IF_REAL,    "Surf. mobility critical field #2"),
  IP("concmodel",MOB_CONC_MOD,	IF_STRING,  "Concentration dep. model"),
  IP("fieldmodel",MOB_FIELD_MOD,IF_STRING,  "Driving field dep. model"),
  IP("init",    MOB_INIT,	IF_FLAG,    "Initialize with defaults")
};

IFcardInfo MOBinfo = {
  "mobility",
  "Specify parameters and types of mobility models",
  NUMELEMS(MOBpTable),
  MOBpTable,

  MOBnewCard,
  MOBparam,
  NULL
};

int
MOBnewCard(GENcard **inCard, GENmodel *inModel)
{
    MOBcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(MOBcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->MOBnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENmobility;
    if (!tmpCard) { /* First in list */
        model->GENmobility = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->MOBnextCard) tmpCard = tmpCard->MOBnextCard;
	/* And add new card */
	tmpCard->MOBnextCard = newCard;
    }
    return(OK);
}

int
MOBparam(int param, IFvalue *value, GENcard *inCard)
{
    MOBcard *card = (MOBcard *)inCard;

    switch (param) {
	case MOB_MATERIAL:
	    card->MOBmaterial = value->iValue;
	    card->MOBmaterialGiven = TRUE;
	    break;
	case MOB_ELEC:
	    if ( value->iValue ) {
	        card->MOBcarrier = ELEC;
	        card->MOBcarrierGiven = TRUE;
	    } else {
		if ( card->MOBcarrier == ELEC ) {
		    card->MOBcarrier = -1;
		    card->MOBcarrierGiven = FALSE;
		}
	    }
	    break;
	case MOB_HOLE:
	    if ( value->iValue ) {
	        card->MOBcarrier = HOLE;
	        card->MOBcarrierGiven = TRUE;
	    } else {
		if ( card->MOBcarrier == HOLE ) {
		    card->MOBcarrier = -1;
		    card->MOBcarrierGiven = FALSE;
		}
	    }
	    break;
	case MOB_MAJOR:
	    if ( value->iValue ) {
	        card->MOBcarrType = MAJOR;
	        card->MOBcarrTypeGiven = TRUE;
	    } else {
		if ( card->MOBcarrType == MAJOR ) {
		    card->MOBcarrType = -1;
		    card->MOBcarrTypeGiven = FALSE;
		}
	    }
	    break;
	case MOB_MINOR:
	    if ( value->iValue ) {
	        card->MOBcarrType = MINOR;
	        card->MOBcarrTypeGiven = TRUE;
	    } else {
		if ( card->MOBcarrType == MINOR ) {
		    card->MOBcarrType = -1;
		    card->MOBcarrTypeGiven = FALSE;
		}
	    }
	    break;
	case MOB_MUMAX:
	    card->MOBmuMax = value->rValue;
	    card->MOBmuMaxGiven = TRUE;
	    break;
	case MOB_MUMIN:
	    card->MOBmuMin = value->rValue;
	    card->MOBmuMinGiven = TRUE;
	    break;
	case MOB_NTREF:
	    card->MOBntRef = value->rValue;
	    card->MOBntRefGiven = TRUE;
	    break;
	case MOB_NTEXP:
	    card->MOBntExp = value->rValue;
	    card->MOBntExpGiven = TRUE;
	    break;
	case MOB_VSAT:
	    card->MOBvSat = value->rValue;
	    card->MOBvSatGiven = TRUE;
	    break;
	case MOB_VWARM:
	    card->MOBvWarm = value->rValue;
	    card->MOBvWarmGiven = TRUE;
	    break;
	case MOB_MUS:
	    card->MOBmus = value->rValue;
	    card->MOBmusGiven = TRUE;
	    break;
	case MOB_EC_A:
	    card->MOBecA = value->rValue;
	    card->MOBecAGiven = TRUE;
	    break;
	case MOB_EC_B:
	    card->MOBecB = value->rValue;
	    card->MOBecBGiven = TRUE;
	    break;
	case MOB_CONC_MOD:
	    if ( cinprefix( value->sValue, "ct", 1 ) ) {
		card->MOBconcModel = CT;
	        card->MOBconcModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "ar", 1 ) ) {
		card->MOBconcModel = AR;
	        card->MOBconcModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "uf", 1 ) ) {
		card->MOBconcModel = UF;
	        card->MOBconcModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "sg", 1 ) ) {
		card->MOBconcModel = SG;
	        card->MOBconcModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "ga", 1 ) ) {
		card->MOBconcModel = GA;
	        card->MOBconcModelGiven = TRUE;
	    }
	    break;
	case MOB_FIELD_MOD:
	    if ( cinprefix( value->sValue, "ct", 1 ) ) {
		card->MOBfieldModel = CT;
	        card->MOBfieldModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "ar", 1 ) ) {
		card->MOBfieldModel = AR;
	        card->MOBfieldModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "sg", 1 ) ) {
		card->MOBfieldModel = SG;
	        card->MOBfieldModelGiven = TRUE;
	    } else if ( cinprefix( value->sValue, "ga", 1 ) ) {
		card->MOBfieldModel = GA;
	        card->MOBfieldModelGiven = TRUE;
	    }
	    break;
	case MOB_INIT:
	    card->MOBinit = value->iValue;
	    card->MOBinitGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
