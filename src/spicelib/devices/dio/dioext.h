/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

extern int DIOacLoad(GENmodel*,CKTcircuit*);
extern int DIOask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int DIOconvTest(GENmodel *,CKTcircuit*);
extern int DIOgetic(GENmodel*,CKTcircuit*);
extern int DIOload(GENmodel*,CKTcircuit*);
extern int DIOmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int DIOmParam(int,IFvalue*,GENmodel*);
extern int DIOparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int DIOpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int DIOsAcLoad(GENmodel*,CKTcircuit*);
extern int DIOsLoad(GENmodel*,CKTcircuit*);
extern int DIOsSetup(SENstruct*,GENmodel*);
extern void DIOsPrint(GENmodel*,CKTcircuit*);
extern int DIOsUpdate(GENmodel*,CKTcircuit*);
extern int DIOsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int DIOunsetup(GENmodel*,CKTcircuit*);
extern int DIOtemp(GENmodel*,CKTcircuit*);
extern int DIOtrunc(GENmodel*,CKTcircuit*,double*);
extern int DIOdisto(int,GENmodel*,CKTcircuit*);
extern int DIOnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int DIOdSetup(DIOmodel*,CKTcircuit*);
extern int DIOsoaCheck(CKTcircuit *, GENmodel *);

