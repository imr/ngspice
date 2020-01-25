/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

extern int VDMOSacLoad(GENmodel *,CKTcircuit*);
extern int VDMOSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VDMOSgetic(GENmodel*,CKTcircuit*);
extern int VDMOSload(GENmodel*,CKTcircuit*);
extern int VDMOSmAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int VDMOSmParam(int,IFvalue*,GENmodel*);
extern int VDMOSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VDMOSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VDMOSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VDMOSunsetup(GENmodel*,CKTcircuit*);
extern int VDMOStemp(GENmodel*,CKTcircuit*);
extern int VDMOStrunc(GENmodel*,CKTcircuit*,double*);
extern int VDMOSconvTest(GENmodel*,CKTcircuit*);
extern int VDMOSdisto(int,GENmodel*,CKTcircuit*);
extern int VDMOSnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int VDMOSdSetup(GENmodel*,CKTcircuit*);
extern int VDMOSsoaCheck(CKTcircuit *, GENmodel *);
