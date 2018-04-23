/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
#ifndef _URCEXT_H
#define _URCEXT_H

extern int URCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int URCmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int URCmParam(int,IFvalue*,GENmodel*);
extern int URCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int URCsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int URCunsetup(GENmodel*,CKTcircuit*);

#endif
