/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int MOS6acLoad(GENmodel *,CKTcircuit*);
extern int MOS6ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS6delete(GENmodel*,IFuid,GENinstance**);
extern void MOS6destroy(GENmodel**);
extern int MOS6getic(GENmodel*,CKTcircuit*);
extern int MOS6load(GENmodel*,CKTcircuit*);
extern int MOS6mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int MOS6mDelete(GENmodel**,IFuid,GENmodel*);
extern int MOS6mParam(int,IFvalue*,GENmodel*);
extern int MOS6param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS6pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
#ifdef notdef
extern int MOS6sAcLoad(GENmodel*,CKTcircuit*);
extern int MOS6sLoad(GENmodel*,CKTcircuit*);
extern void MOS6sPrint(GENmodel*,CKTcircuit*);
extern int MOS6sSetup(SENstruct*,GENmodel*);
extern int MOS6sUpdate(GENmodel*,CKTcircuit*);
#endif
extern int MOS6setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS6unsetup(GENmodel*,CKTcircuit*);
extern int MOS6temp(GENmodel*,CKTcircuit*);
extern int MOS6trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS6convTest(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int MOS6acLoad();
extern int MOS6ask();
extern int MOS6delete();
extern void MOS6destroy();
extern int MOS6getic();
extern int MOS6load();
extern int MOS6mAsk();
extern int MOS6mDelete();
extern int MOS6mParam();
extern int MOS6param();
extern int MOS6pzLoad();
extern int MOS6sAcLoad();
extern int MOS6sLoad();
extern void MOS6sPrint();
extern int MOS6sSetup();
extern int MOS6sUpdate();
extern int MOS6setup();
extern int MOS6unsetup();
extern int MOS6temp();
extern int MOS6trunc();
extern int MOS6convTest();
#endif /* stdc */
