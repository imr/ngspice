/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/numenum.h"
#include "ngspice/matldefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


extern int MATLnewCard(GENcard**,GENmodel *);
extern int MATLparam(int,IFvalue*,GENcard *);


IFparm MATLpTable[] = {
  IP("number",	MATL_NUMBER,	IF_INTEGER, "Material ID number"),
  IP("insulator",MATL_INSULATOR,IF_FLAG,    "Insulator"),
  IP("oxide",	MATL_OXIDE,	IF_FLAG,    "Oxide"),
  IP("sio2",	MATL_OXIDE,	IF_FLAG,    "Oxide"),
  IP("nitride",	MATL_NITRIDE,	IF_FLAG,    "Nitride"),
  IP("si3n4",	MATL_NITRIDE,	IF_FLAG,    "Nitride"),
  IP("semiconductor",MATL_SEMICON,IF_FLAG,  "Semiconductor"),
  IP("silicon",	MATL_SILICON,	IF_FLAG,    "Silicon"),
  IP("polysilicon",MATL_POLYSIL,IF_FLAG,    "Polysilicon"),
  IP("gaas",    MATL_GAAS,      IF_FLAG,    "Gallium-Arsenide"),
  IP("nc",	MATL_NC0,	IF_REAL,    "Conduction band density"),
  IP("nc0",	MATL_NC0,	IF_REAL,    "Conduction band density"),
  IP("nc300",	MATL_NC0,	IF_REAL,    "Conduction band density"),
  IP("nv",	MATL_NV0,	IF_REAL,    "Valence band density"),
  IP("nv0",	MATL_NV0,	IF_REAL,    "Valence band density"),
  IP("nv300",	MATL_NV0,	IF_REAL,    "Valence band density"),
  IP("eg",	MATL_EG0,	IF_REAL,    "Energy gap"),
  IP("eg0",	MATL_EG0,	IF_REAL,    "Energy gap"),
  IP("eg300",	MATL_EG0,	IF_REAL,    "Energy gap"),
  IP("deg.dt",	MATL_DEGDT,	IF_REAL,    "Bandgap narrowing w/ temp"),
  IP("egalpha",	MATL_DEGDT,	IF_REAL,    "Bandgap narrowing w/ temp"),
  IP("eg.tref",	MATL_TREF_EG,	IF_REAL,    "E-gap reference temperature"),
  IP("egbeta",	MATL_TREF_EG,	IF_REAL,    "E-gap reference temperature"),
  IP("deg.dc",	MATL_DEGDC,	IF_REAL,    "Bandgap narrowing w/ N&P doping"),
  IP("eg.cref",	MATL_CREF_EG,	IF_REAL,    "E-gap reference conc (N&P type)"),
  IP("nbgn",	MATL_CREF_EG,	IF_REAL,    "E-gap reference conc (N&P type)"),
  IP("deg.dn",	MATL_DEGDN,	IF_REAL,    "Bandgap narrowing w/ N doping"),
  IP("eg.nref",	MATL_NREF_EG,	IF_REAL,    "E-gap reference conc (N type)"),
  IP("nbgnn",	MATL_NREF_EG,	IF_REAL,    "E-gap reference conc (N type)"),
  IP("deg.dp",	MATL_DEGDP,	IF_REAL,    "Bandgap narrowing w/ P doping"),
  IP("eg.pref",	MATL_PREF_EG,	IF_REAL,    "E-gap reference conc (P type)"),
  IP("nbgnp",	MATL_PREF_EG,	IF_REAL,    "E-gap reference conc (P type)"),
  IP("affinity",MATL_AFFIN,	IF_REAL,    "Electron affinity"),
  IP("permittivity",MATL_PERMIT,IF_REAL,    "Dielectric permittivity"),
  IP("epsilon", MATL_PERMIT,    IF_REAL,    "Dielectric permittivity"),
  IP("tn",	MATL_TAUN0,	IF_REAL,    "SRH electron lifetime"),
  IP("tn0",	MATL_TAUN0,	IF_REAL,    "SRH electron lifetime"),
  IP("taun0",	MATL_TAUN0,	IF_REAL,    "SRH electron lifetime"),
  IP("tp",	MATL_TAUP0,	IF_REAL,    "SRH hole lifetime"),
  IP("tp0",	MATL_TAUP0,	IF_REAL,    "SRH hole lifetime"),
  IP("taup0",	MATL_TAUP0,	IF_REAL,    "SRH hole lifetime"),
  IP("nsrhn",	MATL_NSRHN,	IF_REAL,    "SRH reference conc (electrons)"),
  IP("srh.nref",MATL_NSRHN,	IF_REAL,    "SRH reference conc (electrons)"),
  IP("nsrhp",	MATL_NSRHP,	IF_REAL,    "SRH reference conc (holes)"),
  IP("srh.pref",MATL_NSRHP,	IF_REAL,    "SRH reference conc (holes)"),
  IP("cn",	MATL_CNAUG,	IF_REAL,    "Auger coefficient (electrons)"),
  IP("cnaug",	MATL_CNAUG,	IF_REAL,    "Auger coefficient (electrons)"),
  IP("augn",	MATL_CNAUG,	IF_REAL,    "Auger coefficient (electrons)"),
  IP("cp",	MATL_CPAUG,	IF_REAL,    "Auger coefficient (holes)"),
  IP("cpaug",	MATL_CPAUG,	IF_REAL,    "Auger coefficient (holes)"),
  IP("augp",	MATL_CPAUG,	IF_REAL,    "Auger coefficient (holes)"),
  IP("arichn",	MATL_ARICHN,	IF_REAL,    "Richardson constant (electrons)"),
  IP("arichp",	MATL_ARICHP,	IF_REAL,    "Richardson constant (holes)")
};

IFcardInfo MATLinfo = {
  "material",
  "Specify physical properties of a material",
  NUMELEMS(MATLpTable),
  MATLpTable,

  MATLnewCard,
  MATLparam,
  NULL
};

IFcardInfo PHYSinfo = {
  "physics",
  "Specify physical properties of a material",
  NUMELEMS(MATLpTable),
  MATLpTable,

  MATLnewCard,
  MATLparam,
  NULL
};

int
MATLnewCard(GENcard **inCard, GENmodel *inModel)
{
    MATLcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    newCard = TMALLOC(MATLcard, 1);
    if (!newCard) {
        *inCard = NULL;
        return(E_NOMEM);
    }
    newCard->MATLnextCard = NULL;
    *inCard = (GENcard *) newCard;

    tmpCard = model->GENmaterials;
    if (!tmpCard) { /* First in list */
        model->GENmaterials = newCard;
    } else {
	/* Go to end of list */
        while (tmpCard->MATLnextCard) tmpCard = tmpCard->MATLnextCard;
	/* And add new card */
	tmpCard->MATLnextCard = newCard;
    }
    return(OK);
}

int
MATLparam(int param, IFvalue *value, GENcard *inCard)
{
    MATLcard *card = (MATLcard *)inCard;

    switch (param) {
	case MATL_NUMBER:
	    card->MATLnumber = value->iValue;
	    card->MATLnumberGiven = TRUE;
	    break;
	case MATL_NC0:
	    card->MATLnc0 = value->rValue;
	    card->MATLnc0Given = TRUE;
	    break;
	case MATL_NV0:
	    card->MATLnv0 = value->rValue;
	    card->MATLnv0Given = TRUE;
	    break;
	case MATL_EG0:
	    card->MATLeg0 = value->rValue;
	    card->MATLeg0Given = TRUE;
	    break;
	case MATL_DEGDT:
	    card->MATLdEgdT = value->rValue;
	    card->MATLdEgdTGiven = TRUE;
	    break;
	case MATL_TREF_EG:
	    card->MATLtrefEg = value->rValue;
	    card->MATLtrefEgGiven = TRUE;
	    break;
	case MATL_DEGDC:
	    card->MATLdEgdN = value->rValue;
	    card->MATLdEgdNGiven = TRUE;
	    card->MATLdEgdP = value->rValue;
	    card->MATLdEgdPGiven = TRUE;
	    break;
	case MATL_CREF_EG:
	    card->MATLnrefEg = value->rValue;
	    card->MATLnrefEgGiven = TRUE;
	    card->MATLprefEg = value->rValue;
	    card->MATLprefEgGiven = TRUE;
	    break;
	case MATL_DEGDN:
	    card->MATLdEgdN = value->rValue;
	    card->MATLdEgdNGiven = TRUE;
	    break;
	case MATL_NREF_EG:
	    card->MATLnrefEg = value->rValue;
	    card->MATLnrefEgGiven = TRUE;
	    break;
	case MATL_DEGDP:
	    card->MATLdEgdP = value->rValue;
	    card->MATLdEgdPGiven = TRUE;
	    break;
	case MATL_PREF_EG:
	    card->MATLprefEg = value->rValue;
	    card->MATLprefEgGiven = TRUE;
	    break;
	case MATL_AFFIN:
	    card->MATLaffinity = value->rValue;
	    card->MATLaffinityGiven = TRUE;
	    break;
	case MATL_PERMIT:
	    card->MATLpermittivity = value->rValue;
	    card->MATLpermittivityGiven = TRUE;
	    break;
	case MATL_TAUN0:
	    card->MATLtaun0 = value->rValue;
	    card->MATLtaun0Given = TRUE;
	    break;
	case MATL_TAUP0:
	    card->MATLtaup0 = value->rValue;
	    card->MATLtaup0Given = TRUE;
	    break;
	case MATL_NSRHN:
	    card->MATLnrefSRHn = value->rValue;
	    card->MATLnrefSRHnGiven = TRUE;
	    break;
	case MATL_NSRHP:
	    card->MATLnrefSRHp = value->rValue;
	    card->MATLnrefSRHpGiven = TRUE;
	    break;
	case MATL_CNAUG:
	    card->MATLcnAug = value->rValue;
	    card->MATLcnAugGiven = TRUE;
	    break;
	case MATL_CPAUG:
	    card->MATLcpAug = value->rValue;
	    card->MATLcpAugGiven = TRUE;
	    break;
	case MATL_ARICHN:
	    card->MATLaRichN = value->rValue;
	    card->MATLaRichNGiven = TRUE;
	    break;
	case MATL_ARICHP:
	    card->MATLaRichP = value->rValue;
	    card->MATLaRichPGiven = TRUE;
	    break;
	case MATL_INSULATOR:
	    if ( value->iValue ) {
	        card->MATLmaterial = INSULATOR;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == INSULATOR ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_OXIDE:
	    if ( value->iValue ) {
	        card->MATLmaterial = OXIDE;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == OXIDE ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_NITRIDE:
	    if ( value->iValue ) {
	        card->MATLmaterial = NITRIDE;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == NITRIDE ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_SEMICON:
	    if ( value->iValue ) {
	        card->MATLmaterial = SEMICON;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == SEMICON ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_SILICON:
	    if ( value->iValue ) {
	        card->MATLmaterial = SILICON;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == SILICON ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_POLYSIL:
	    if ( value->iValue ) {
	        card->MATLmaterial = POLYSILICON;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == POLYSILICON ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	case MATL_GAAS:
	    if ( value->iValue ) {
	        card->MATLmaterial = GAAS;
	        card->MATLmaterialGiven = TRUE;
	    } else {
		if ( card->MATLmaterial == GAAS ) {
	            card->MATLmaterial = -1;
	            card->MATLmaterialGiven = FALSE;
		}
	    }
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
