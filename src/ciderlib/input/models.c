/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice.h"
#include "numcards.h"
#include "numgen.h"
#include "modldefs.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"

extern int MODLnewCard(void**,void*);
extern int MODLparam(int,IFvalue*,void*);


IFparm MODLpTable[] = {
  IP("bgn",	MODL_BGNW,	IF_FLAG,    "Bandgap narrowing"),
  IP("bgnw",	MODL_BGNW,	IF_FLAG,    "Bandgap narrowing"),
  IP("tmpmob",	MODL_TEMPMOB,	IF_FLAG,    "Temp-dependent mobility"),
  IP("tempmob",	MODL_TEMPMOB,	IF_FLAG,    "Temp-dependent mobility"),
  IP("conmob",	MODL_CONCMOB,	IF_FLAG,    "Conc-dependent mobility"),
  IP("concmob",	MODL_CONCMOB,	IF_FLAG,    "Conc-dependent mobility"),
  IP("fldmob",	MODL_FIELDMOB,	IF_FLAG,
     "Lateral-field-dependent mobility"),
  IP("fieldmob",MODL_FIELDMOB,	IF_FLAG,
     "Lateral-field-dependent mobility"),
  IP("trfmob",MODL_TRANSMOB,	IF_FLAG,
      "Transverse-field-dependent surface mobility"),
  IP("transmob",MODL_TRANSMOB,	IF_FLAG,
      "Transverse-field-dependent surface mobility"),
  IP("srfmob",  MODL_SURFMOB,	IF_FLAG,    "Activate surface mobility"),
  IP("surfmob", MODL_SURFMOB,	IF_FLAG,    "Activate surface mobility"),
  IP("matchmob",MODL_MATCHMOB,	IF_FLAG,
      "Matching low-field surface/bulk mobilities"),
  IP("srh",	MODL_SRH,	IF_FLAG,    "SRH recombination"),
  IP("consrh",	MODL_CONCTAU,	IF_FLAG,    "Conc-dependent SRH recomb"),
  IP("conctau",	MODL_CONCTAU,	IF_FLAG,    "Conc-dependent SRH recomb"),
  IP("auger",	MODL_AUGER,	IF_FLAG,    "Auger recombination"),
  IP("avalanche",MODL_AVAL,	IF_FLAG,    "Local avalanche generation")
};

IFcardInfo MODLinfo = {
  "models",
  "Specify which physical models should be simulated",
  NUMELEMS(MODLpTable),
  MODLpTable,

  MODLnewCard,
  MODLparam,
  NULL
};

int
MODLnewCard(void **inCard, void *inModel)
{
    MODLcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    tmpCard = model->GENmodels;
    if (!tmpCard) { /* First in list */
        newCard = NEW( MODLcard );
        if (!newCard) {
            *inCard = (void *)NULL;
            return(E_NOMEM);
        }
        newCard->MODLnextCard = (MODLcard *)NULL;
        *inCard = (void *)newCard;
        model->GENmodels = newCard;
    } else { /* Only one card of this type allowed */
	*inCard = (void *)tmpCard;
    }
    return(OK);
}

int
MODLparam(int param, IFvalue *value, void *inCard)
{
    MODLcard *card = (MODLcard *)inCard;

    switch (param) {
	case MODL_BGNW:
	    card->MODLbandGapNarrowing = value->iValue;
	    card->MODLbandGapNarrowingGiven = TRUE;
	    break;
	case MODL_TEMPMOB:
	    card->MODLtempDepMobility = value->iValue;
	    card->MODLtempDepMobilityGiven = TRUE;
	    break;
	case MODL_CONCMOB:
	    card->MODLconcDepMobility = value->iValue;
	    card->MODLconcDepMobilityGiven = TRUE;
	    break;
	case MODL_TRANSMOB:
	    card->MODLtransDepMobility = value->iValue;
	    card->MODLtransDepMobilityGiven = TRUE;
	    break;
	case MODL_FIELDMOB:
	    card->MODLfieldDepMobility = value->iValue;
	    card->MODLfieldDepMobilityGiven = TRUE;
	    break;
	case MODL_SURFMOB:
	    card->MODLsurfaceMobility = value->iValue;
	    card->MODLsurfaceMobilityGiven = TRUE;
	    break;
	case MODL_MATCHMOB:
	    card->MODLmatchingMobility = value->iValue;
	    card->MODLmatchingMobilityGiven = TRUE;
	    break;
	case MODL_SRH:
	    card->MODLsrh = value->iValue;
	    card->MODLsrhGiven = TRUE;
	    break;
	case MODL_CONCTAU:
	    card->MODLconcDepLifetime = value->iValue;
	    card->MODLconcDepLifetimeGiven = TRUE;
	    break;
	case MODL_AUGER:
	    card->MODLauger = value->iValue;
	    card->MODLaugerGiven = TRUE;
	    break;
	case MODL_AVAL:
	    card->MODLavalancheGen = value->iValue;
	    card->MODLavalancheGenGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
