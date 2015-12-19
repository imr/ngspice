/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/dopdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define UM_TO_CM 1.0e-4

extern int DOPnewCard(GENcard**,GENmodel*);
extern int DOPparam(int,IFvalue*,GENcard*);


IFparm DOPpTable[] = {
  IP("domains",	DOP_DOMAIN,	IF_INTVEC,	"Domain(s) to dope"),
  IP("uniform",	DOP_UNIF,	IF_FLAG,	"Uniform doping"),
  IP("linear",	DOP_LINEAR,	IF_FLAG,	"Linear graded doping"),
  IP("gaussian",DOP_GAUSS,	IF_FLAG,	"Gaussian doping"),
  IP("gdiff",	DOP_GAUSS,	IF_FLAG,	"Gaussian diffusion"),
  IP("gimp",	DOP_GAUSS,	IF_FLAG,	"Gaussian implantation"),
  IP("erfc",	DOP_ERFC,	IF_FLAG,	"Error-function doping"),
  IP("errfc",	DOP_ERFC,	IF_FLAG,	"Error-function doping"),
  IP("exponential",DOP_EXP,	IF_FLAG,	"Exponential doping"),
  IP("suprem3", DOP_SUPREM3,	IF_FLAG,	"Suprem3 doping"),
  IP("ascii",	DOP_ASCII,	IF_FLAG,	"Ascii-input doping"),
  IP("infile",	DOP_INFILE,	IF_STRING,	"Input-file name"),
  IP("lat.rotate",DOP_ROTATE_LAT,IF_FLAG,	"Rotate principal profile"),
  IP("lat.unif",DOP_UNIF_LAT,	IF_FLAG,	"Uniform lateral falloff"),
  IP("lat.linf",DOP_LINEAR_LAT,	IF_FLAG,	"Linear lateral falloff"),
  IP("lat.gauss",DOP_GAUSS_LAT,	IF_FLAG,	"Gaussian lateral falloff"),
  IP("lat.erfc",DOP_ERFC_LAT,	IF_FLAG,	"Erfc lateral falloff"),
  IP("lat.exp", DOP_EXP_LAT,	IF_FLAG,	"Exponential lateral falloff"),
  IP("boron",	DOP_BORON,	IF_FLAG,	"Boron impurities"),
  IP("phosphorus",DOP_PHOSP,	IF_FLAG,	"Phosphorus impurities"),
  IP("arsenic",	DOP_ARSEN,	IF_FLAG,	"Arsenic impurities"),
  IP("antimony",DOP_ANTIM,	IF_FLAG,	"Antimony impurities"),
  IP("p.type",	DOP_P_TYPE,	IF_FLAG,	"P-type impurities"),
  IP("acceptor",DOP_P_TYPE,	IF_FLAG,	"Acceptor impurities"),
  IP("n.type",	DOP_N_TYPE,	IF_FLAG,	"N-type impurities"),
  IP("donor",	DOP_N_TYPE,	IF_FLAG,	"Donor impurities"),
  IP("x.axis",	DOP_X_AXIS,	IF_FLAG,	"X principal axis"),
  IP("y.axis",	DOP_Y_AXIS,	IF_FLAG,	"Y principal axis"),
  IP("x.low", 	DOP_X_LOW,	IF_REAL,	"Low X edge"),
  IP("x.high",	DOP_X_HIGH,	IF_REAL,	"High X edge"),
  IP("y.low",	DOP_Y_LOW,	IF_REAL,	"Low Y edge"),
  IP("y.high",	DOP_Y_HIGH,	IF_REAL,	"High Y edge"),
  IP("conc",	DOP_CONC,	IF_REAL,	"Concentration"),
  IP("peak.conc",DOP_CONC,	IF_REAL,	"Peak concentration"),
  IP("location",DOP_LOCATION,	IF_REAL,	"Peak location"),
  IP("range",   DOP_LOCATION,	IF_REAL,	"Peak location"),
  IP("char.length",DOP_CHAR_LEN,IF_REAL,	"Characteristic length"),
  IP("ratio.lat",DOP_RATIO_LAT,	IF_REAL,	"Lateral ratio")
};

IFcardInfo DOPinfo = {
  "doping",
  "Add dopant to domains of a device",
  NUMELEMS(DOPpTable),
  DOPpTable,

  DOPnewCard,
  DOPparam,
  NULL
};

int
DOPnewCard(GENcard **inCard, GENmodel *inModel)
{
    DOPcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(DOPcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->DOPnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENdopings;
    if (!tmpCard) { /* First in list */
        model->GENdopings = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->DOPnextCard) tmpCard = tmpCard->DOPnextCard;
	/* And add new card */
	tmpCard->DOPnextCard = newCard;
    }
    return(OK);
}


int
DOPparam(int param, IFvalue *value, GENcard *inCard)
{
    int i;
    DOPcard *card = (DOPcard *)inCard;

    switch (param) {
	case DOP_DOMAIN:
	    if ( !card->DOPdomainsGiven ) {
	      card->DOPnumDomains = value->v.numValue;
	      card->DOPdomains = TMALLOC(int, value->v.numValue);
	      for ( i=0; i < card->DOPnumDomains; i++ ) {
		  card->DOPdomains[i] = value->v.vec.iVec[i];
	      }
	      card->DOPdomainsGiven = TRUE;
	    } /* else: do nothing, can only define domains once */
	    break;
	case DOP_ROTATE_LAT:
	    card->DOProtateLat = TRUE;
	    card->DOProtateLatGiven = TRUE;
	    break;
	case DOP_UNIF:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_UNIF;
	        card->DOPprofileTypeGiven = TRUE;
	    }
	    break;
	case DOP_UNIF_LAT:
	    if ( !card->DOPlatProfileTypeGiven ) {
	        card->DOPlatProfileType = DOP_UNIF;
	        card->DOPlatProfileTypeGiven = TRUE;
	    }
	    break;
	case DOP_LINEAR:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_LINEAR;
	        card->DOPprofileTypeGiven = TRUE;
	    }
	    break;
	case DOP_LINEAR_LAT:
	    if ( !card->DOPlatProfileTypeGiven ) {
	        card->DOPlatProfileType = DOP_LINEAR_LAT;
	        card->DOPlatProfileTypeGiven = TRUE;
	    }
	    break;
	case DOP_GAUSS:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_GAUSS;
	        card->DOPprofileTypeGiven = TRUE;
	    }
	    break;
	case DOP_GAUSS_LAT:
	    if ( !card->DOPlatProfileTypeGiven ) {
	        card->DOPlatProfileType = DOP_GAUSS;
	        card->DOPlatProfileTypeGiven = TRUE;
	    }
	    break;
	case DOP_ERFC:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_ERFC;
	        card->DOPprofileTypeGiven = TRUE;
	    }
	    break;
	case DOP_ERFC_LAT:
	    if ( !card->DOPlatProfileTypeGiven ) {
	        card->DOPlatProfileType = DOP_ERFC;
	        card->DOPlatProfileTypeGiven = TRUE;
	    }
	    break;
	case DOP_EXP:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_EXP;
	        card->DOPprofileTypeGiven = TRUE;
	    }
	    break;
	case DOP_EXP_LAT:
	    if ( !card->DOPlatProfileTypeGiven ) {
	        card->DOPlatProfileType = DOP_EXP;
	        card->DOPlatProfileTypeGiven = TRUE;
	    }
	    break;
	case DOP_SUPREM3:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_SUPREM3;
	        card->DOPprofileTypeGiven = TRUE;
	    } else if ( card->DOPprofileType == DOP_ASCII ) {
	        card->DOPprofileType = DOP_SUPASCII;
	    }
	    break;
	case DOP_ASCII:
	    if ( !card->DOPprofileTypeGiven ) {
	        card->DOPprofileType = DOP_ASCII;
	        card->DOPprofileTypeGiven = TRUE;
	    } else if ( card->DOPprofileType == DOP_SUPREM3 ) {
	        card->DOPprofileType = DOP_SUPASCII;
	    }
	    break;
	case DOP_BORON:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_BORON;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_PHOSP:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_PHOSP;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_ARSEN:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_ARSEN;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_ANTIM:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_ANTIM;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_P_TYPE:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_P_TYPE;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_N_TYPE:
	    if ( !card->DOPimpurityTypeGiven ) {
	        card->DOPimpurityType = DOP_N_TYPE;
	        card->DOPimpurityTypeGiven = TRUE;
	    }
	    break;
	case DOP_X_AXIS:
	    if ( !card->DOPaxisTypeGiven ) {
	        card->DOPaxisType = DOP_X_AXIS;
	        card->DOPaxisTypeGiven = TRUE;
	    }
	    break;
	case DOP_Y_AXIS:
	    if ( !card->DOPaxisTypeGiven ) {
	        card->DOPaxisType = DOP_Y_AXIS;
	        card->DOPaxisTypeGiven = TRUE;
	    }
	    break;
	case DOP_INFILE:
	    card->DOPinFile = value->sValue;
	    card->DOPinFileGiven = TRUE;
	    break;
	case DOP_X_LOW:
	    card->DOPxLow = value->rValue * UM_TO_CM;
	    card->DOPxLowGiven = TRUE;
	    break;
	case DOP_X_HIGH:
	    card->DOPxHigh = value->rValue * UM_TO_CM;
	    card->DOPxHighGiven = TRUE;
	    break;
	case DOP_Y_LOW:
	    card->DOPyLow = value->rValue * UM_TO_CM;
	    card->DOPyLowGiven = TRUE;
	    break;
	case DOP_Y_HIGH:
	    card->DOPyHigh = value->rValue * UM_TO_CM;
	    card->DOPyHighGiven = TRUE;
	    break;
	case DOP_CONC:
	    card->DOPconc = fabs(value->rValue);
	    card->DOPconcGiven = TRUE;
	    break;
	case DOP_LOCATION:
	    card->DOPlocation = value->rValue * UM_TO_CM;
	    card->DOPlocationGiven = TRUE;
	    break;
	case DOP_CHAR_LEN:
	    card->DOPcharLen = value->rValue * UM_TO_CM;
	    card->DOPcharLenGiven = TRUE;
	    break;
	case DOP_RATIO_LAT:
	    card->DOPratioLat = value->rValue;
	    card->DOPratioLatGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
