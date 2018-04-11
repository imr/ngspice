/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int RESask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int RESload(GENmodel*,CKTcircuit*);
extern int RESacload(GENmodel*,CKTcircuit*);
extern int RESmodAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int RESmParam(int,IFvalue*,GENmodel*);
extern int RESparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int RESpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int RESsAcLoad(GENmodel*,CKTcircuit*);
extern int RESsLoad(GENmodel*,CKTcircuit*);
extern int RESsSetup(SENstruct*,GENmodel*);
extern void RESsPrint(GENmodel*,CKTcircuit*);
extern int RESsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int REStemp(GENmodel*,CKTcircuit*);
extern int RESnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int RESsoaCheck(CKTcircuit *, GENmodel *);
