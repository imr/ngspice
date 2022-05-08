/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int VSRCaccept(CKTcircuit*,GENmodel*);
extern int VSRCacLoad(GENmodel*,CKTcircuit*);
extern int VSRCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VSRCdelete(GENinstance*);
extern int VSRCfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int VSRCload(GENmodel*,CKTcircuit*);
extern int VSRCmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int VSRCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VSRCpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VSRCsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VSRCunsetup(GENmodel*,CKTcircuit*);
extern int VSRCpzSetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VSRCtemp(GENmodel*,CKTcircuit*);
#ifdef RFSPICE
extern int VSRCspupdate(GENmodel*, CKTcircuit*);
extern int VSRCgetActivePorts(GENmodel* inModel, CKTcircuit* ckt, VSRCinstance** ports);
extern int VSRCspinit(GENmodel* inModel, CKTcircuit* ckt, CMat* zref, CMat* gn, CMat* gninv);
#endif
