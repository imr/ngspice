/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"
#include "ngspice/numgen.h"
#include "ngspice/numenum.h"
#include "ngspice/outpdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "../misc/tilde.h"

extern int OUTPnewCard(GENcard**,GENmodel*);
extern int OUTPparam(int,IFvalue*,GENcard*);


IFparm OUTPpTable[] = {
/* Debugging Flags */
  IP("all.debug",OUTP_ALL_DEBUG,IF_FLAG,    "Debug All Analyses"),
  IP("op.debug",OUTP_DC_DEBUG,	IF_FLAG,    "Debug DC/OP Analyses"),
  IP("dc.debug",OUTP_DC_DEBUG,	IF_FLAG,    "Debug DC/OP Analyses"),
  IP("tran.debug",OUTP_TRAN_DEBUG,IF_FLAG,  "Debug TRAN Analysis"),
  IP("ac.debug",OUTP_AC_DEBUG,	IF_FLAG,    "Debug AC/PZ Analyses"),
  IP("pz.debug",OUTP_AC_DEBUG,	IF_FLAG,    "Debug AC/PZ Analyses"),
/* General Information */
  IP("geometry",OUTP_GEOM,	IF_FLAG,    "Geometric information"),
  IP("mesh",    OUTP_MESH,	IF_FLAG,    "Mesh information"),
  IP("material",OUTP_MATERIAL,	IF_FLAG,    "Material information"),
  IP("globals", OUTP_GLOBALS,	IF_FLAG,    "Global information"),
  IP("statistics", OUTP_STATS,	IF_FLAG,    "Resource usage information"),
  IP("resources", OUTP_STATS,	IF_FLAG,    "Resource usage information"),
/* Solution Information */
  IP("rootfile", OUTP_ROOTFILE,	IF_STRING,  "Root of output file names"),
  IP("rawfile", OUTP_RAWFILE,	IF_FLAG,    "SPICE rawfile data format"),
  IP("hdf",     OUTP_HDF,	IF_FLAG,    "HDF data format"),
  IP("doping",  OUTP_DOPING,	IF_FLAG,    "Net doping"),
  IP("psi",	OUTP_PSI,	IF_FLAG,    "Potential"),
  IP("equ.psi",	OUTP_EQU_PSI,	IF_FLAG,    "Equilibrium potential"),
  IP("vac.psi",	OUTP_VAC_PSI,	IF_FLAG,    "Vacuum potential"),
  IP("n.conc",  OUTP_N_CONC,	IF_FLAG,    "Electron concentration"),
  IP("electrons", OUTP_N_CONC,	IF_FLAG,    "Electron concentration"),
  IP("p.conc",  OUTP_P_CONC,	IF_FLAG,    "Hole concentration"),
  IP("holes",   OUTP_P_CONC,	IF_FLAG,    "Hole concentration"),
  IP("phin",    OUTP_PHIN,	IF_FLAG,    "Electron quasi-fermi potential"),
  IP("qfn",     OUTP_PHIN,	IF_FLAG,    "Electron quasi-fermi potential"),
  IP("phip",    OUTP_PHIP,	IF_FLAG,    "Hole quasi-fermi potential"),
  IP("qfp",     OUTP_PHIP,	IF_FLAG,    "Hole quasi-fermi potential"),
  IP("phic",    OUTP_PHIC,	IF_FLAG,    "Conduction band potential"),
  IP("band.con",OUTP_PHIC,	IF_FLAG,    "Conduction band potential"),
  IP("phiv",    OUTP_PHIV,	IF_FLAG,    "Valence band potential"),
  IP("band.val",OUTP_PHIV,	IF_FLAG,    "Valence band potential"),
  IP("e.field",	OUTP_E_FIELD,	IF_FLAG,    "Electric field"),
  IP("jc",      OUTP_J_C,	IF_FLAG,    "Conduction current density"),
  IP("j.conduc",OUTP_J_C,	IF_FLAG,    "Conduction current density"),
  IP("jd",      OUTP_J_D,	IF_FLAG,    "Displacement current density"),
  IP("j.disp",  OUTP_J_D,	IF_FLAG,    "Displacement current density"),
  IP("jn",      OUTP_J_N,	IF_FLAG,    "Electron current density"),
  IP("j.electr",OUTP_J_N,	IF_FLAG,    "Electron current density"),
  IP("jp",      OUTP_J_P,	IF_FLAG,    "Hole current density"),
  IP("j.hole",  OUTP_J_P,	IF_FLAG,    "Hole current density"),
  IP("jt",      OUTP_J_T,	IF_FLAG,    "Total current density"),
  IP("j.total", OUTP_J_T,	IF_FLAG,    "Total current density"),
  IP("unet",    OUTP_U_NET,	IF_FLAG,    "Net recombination"),
  IP("recomb",  OUTP_U_NET,	IF_FLAG,    "Net recombination"),
  IP("mun",     OUTP_MUN,	IF_FLAG,    "Elctron mobility"),
  IP("mob.elec",OUTP_MUN,	IF_FLAG,    "Electron mobility"),
  IP("mup",     OUTP_MUP,	IF_FLAG,    "Hole mobility"),
  IP("mob.hole",OUTP_MUP,	IF_FLAG,    "Hole mobility")
};

IFcardInfo OUTPinfo = {
  "output",
  "Identify information to be output to user",
  NUMELEMS(OUTPpTable),
  OUTPpTable,

  OUTPnewCard,
  OUTPparam,
  NULL
};

int
OUTPnewCard(GENcard **inCard, GENmodel *inModel)
{
    OUTPcard *tmpCard, *newCard;
    GENnumModel *model = (GENnumModel *)inModel;

    tmpCard = model->GENoutputs;
    if (!tmpCard) { /* First in list */
        newCard = TMALLOC(OUTPcard, 1);
        if (!newCard) {
            *inCard = NULL;
            return(E_NOMEM);
        }
        newCard->OUTPnextCard = NULL;
        *inCard = (GENcard *) newCard;
        model->GENoutputs = newCard;
    } else { /* Only one card of this type allowed */
	*inCard = (GENcard *) tmpCard;
    }
    return(OK);
}

int
OUTPparam(int param, IFvalue *value, GENcard *inCard)
{
    OUTPcard *card = (OUTPcard *)inCard;

    switch (param) {
	case OUTP_ALL_DEBUG:
	    card->OUTPdcDebug = value->iValue;
	    card->OUTPdcDebugGiven = TRUE;
	    card->OUTPtranDebug = value->iValue;
	    card->OUTPtranDebugGiven = TRUE;
	    card->OUTPacDebug = value->iValue;
	    card->OUTPacDebugGiven = TRUE;
	    break;
	case OUTP_DC_DEBUG:
	    card->OUTPdcDebug = value->iValue;
	    card->OUTPdcDebugGiven = TRUE;
	    break;
	case OUTP_TRAN_DEBUG:
	    card->OUTPtranDebug = value->iValue;
	    card->OUTPtranDebugGiven = TRUE;
	    break;
	case OUTP_AC_DEBUG:
	    card->OUTPacDebug = value->iValue;
	    card->OUTPacDebugGiven = TRUE;
	    break;
	case OUTP_GEOM:
	    card->OUTPgeom = value->iValue;
	    card->OUTPgeomGiven = TRUE;
	    break;
	case OUTP_MESH:
	    card->OUTPmesh = value->iValue;
	    card->OUTPmeshGiven = TRUE;
	    break;
	case OUTP_MATERIAL:
	    card->OUTPmaterial = value->iValue;
	    card->OUTPmaterialGiven = TRUE;
	    break;
	case OUTP_GLOBALS:
	    card->OUTPglobals = value->iValue;
	    card->OUTPglobalsGiven = TRUE;
	    break;
	case OUTP_STATS:
	    card->OUTPstats = value->iValue;
	    card->OUTPstatsGiven = TRUE;
	    break;
	case OUTP_ROOTFILE:
	    card->OUTProotFile = tildexpand(value->sValue); /*xxx*/
	    card->OUTProotFileGiven = TRUE;
	    break;
	case OUTP_RAWFILE:
	    card->OUTPfileType = RAWFILE;
	    card->OUTPfileTypeGiven = TRUE;
	    break;
	case OUTP_HDF:
	    return(E_UNSUPP);
	    break;
	case OUTP_DOPING:
	    card->OUTPdoping = value->iValue;
	    card->OUTPdopingGiven = TRUE;
	    break;
	case OUTP_PSI:
	    card->OUTPpsi = value->iValue;
	    card->OUTPpsiGiven = TRUE;
	    break;
	case OUTP_EQU_PSI:
	    card->OUTPequPsi = value->iValue;
	    card->OUTPequPsiGiven = TRUE;
	    break;
	case OUTP_VAC_PSI:
	    card->OUTPvacPsi = value->iValue;
	    card->OUTPvacPsiGiven = TRUE;
	    break;
	case OUTP_N_CONC:
	    card->OUTPnConc = value->iValue;
	    card->OUTPnConcGiven = TRUE;
	    break;
	case OUTP_P_CONC:
	    card->OUTPpConc = value->iValue;
	    card->OUTPpConcGiven = TRUE;
	    break;
	case OUTP_PHIN:
	    card->OUTPphin = value->iValue;
	    card->OUTPphinGiven = TRUE;
	    break;
	case OUTP_PHIP:
	    card->OUTPphip = value->iValue;
	    card->OUTPphipGiven = TRUE;
	    break;
	case OUTP_PHIC:
	    card->OUTPphic = value->iValue;
	    card->OUTPphicGiven = TRUE;
	    break;
	case OUTP_PHIV:
	    card->OUTPphiv = value->iValue;
	    card->OUTPphivGiven = TRUE;
	    break;
	case OUTP_J_C:
	    card->OUTPjc = value->iValue;
	    card->OUTPjcGiven = TRUE;
	    break;
	case OUTP_J_D:
	    card->OUTPjd = value->iValue;
	    card->OUTPjdGiven = TRUE;
	    break;
	case OUTP_J_N:
	    card->OUTPjn = value->iValue;
	    card->OUTPjnGiven = TRUE;
	    break;
	case OUTP_J_P:
	    card->OUTPjp = value->iValue;
	    card->OUTPjpGiven = TRUE;
	    break;
	case OUTP_J_T:
	    card->OUTPjt = value->iValue;
	    card->OUTPjtGiven = TRUE;
	    break;
	case OUTP_U_NET:
	    card->OUTPuNet = value->iValue;
	    card->OUTPuNetGiven = TRUE;
	    break;
	case OUTP_MUN:
	    card->OUTPmun = value->iValue;
	    card->OUTPmunGiven = TRUE;
	    break;
	case OUTP_MUP:
	    card->OUTPmup = value->iValue;
	    card->OUTPmupGiven = TRUE;
	    break;
	default:
	    return(E_BADPARM);
	    break;
    }
    return(OK);
}
