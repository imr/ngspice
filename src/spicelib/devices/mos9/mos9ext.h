/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

extern int MOS9acLoad(GENmodel*,CKTcircuit*);
extern int MOS9ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS9convTest(GENmodel *,CKTcircuit *);
extern int MOS9delete(GENinstance*);
extern int MOS9getic(GENmodel*,CKTcircuit*);
extern int MOS9load(GENmodel*,CKTcircuit*);
extern int MOS9mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MOS9mParam(int,IFvalue*,GENmodel*);
extern int MOS9param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS9pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MOS9sAcLoad(GENmodel*,CKTcircuit*);
extern int MOS9sLoad(GENmodel*,CKTcircuit*);
extern void MOS9sPrint(GENmodel*,CKTcircuit*);
extern int MOS9sSetup(SENstruct*,GENmodel*);
extern int MOS9sUpdate(GENmodel*,CKTcircuit*);
extern int MOS9setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS9unsetup(GENmodel*,CKTcircuit*);
extern int MOS9temp(GENmodel*,CKTcircuit*);
extern int MOS9trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS9disto(int,GENmodel*,CKTcircuit*);
extern int MOS9noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int MOS9dSetup(GENmodel*,CKTcircuit*);
