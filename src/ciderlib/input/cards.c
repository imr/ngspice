/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numcards.h"

extern IFcardInfo CONTinfo;
extern IFcardInfo DOPinfo;
extern IFcardInfo ELCTinfo;
extern IFcardInfo BDRYinfo;
extern IFcardInfo INTFinfo;
extern IFcardInfo XMSHinfo;
extern IFcardInfo YMSHinfo;
extern IFcardInfo METHinfo;
extern IFcardInfo MOBinfo;
extern IFcardInfo MODLinfo;
extern IFcardInfo PHYSinfo;
extern IFcardInfo MATLinfo;
extern IFcardInfo DOMNinfo;
extern IFcardInfo REGNinfo;
extern IFcardInfo OPTNinfo;
extern IFcardInfo OUTPinfo;

IFcardInfo *INPcardTab[] = {
    &CONTinfo,
    &DOPinfo,
    &ELCTinfo,
    &BDRYinfo,
    &INTFinfo,
    &XMSHinfo,
    &YMSHinfo,
    &METHinfo,
    &MOBinfo,
    &MODLinfo,
    &PHYSinfo,
    &MATLinfo,
    &DOMNinfo,
    &REGNinfo,
    &OPTNinfo,
    &OUTPinfo
};

int INPnumCards = NUMELEMS(INPcardTab);
