/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int BJTacLoad(GENmodel *,CKTcircuit*);
extern int BJTask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BJTconvTest(GENmodel*,CKTcircuit*);
extern int BJTdelete(GENmodel*,IFuid,GENinstance**);
extern void BJTdestroy(GENmodel**);
extern int BJTgetic(GENmodel*,CKTcircuit*);
extern int BJTload(GENmodel*,CKTcircuit*);
extern int BJTmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int BJTmDelete(GENmodel**,IFuid,GENmodel*);
extern int BJTmParam(int,IFvalue*,GENmodel*);
extern int BJTparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BJTpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BJTsAcLoad(GENmodel*,CKTcircuit*);
extern int BJTsLoad(GENmodel*,CKTcircuit*);
extern void BJTsPrint(GENmodel*,CKTcircuit*);
extern int BJTsSetup(SENstruct*,GENmodel*);
extern int BJTsUpdate(GENmodel*,CKTcircuit*);
extern int BJTsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BJTunsetup(GENmodel*,CKTcircuit*);
extern int BJTtemp(GENmodel*,CKTcircuit*);
extern int BJTtrunc(GENmodel*,CKTcircuit*,double*);
extern int BJTdisto(int,GENmodel*,CKTcircuit*);
extern int BJTnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
#else /* stdc */
extern int BJTacLoad();
extern int BJTask();
extern int BJTconvTest();
extern int BJTdelete();
extern void BJTdestroy();
extern int BJTgetic();
extern int BJTload();
extern int BJTmAsk();
extern int BJTmDelete();
extern int BJTmParam();
extern int BJTparam();
extern int BJTpzLoad();
extern int BJTsAcLoad();
extern int BJTsLoad();
extern void BJTsPrint();
extern int BJTsSetup();
extern int BJTsUpdate();
extern int BJTsetup();
extern int BJTunsetup();
extern int BJTtemp();
extern int BJTtrunc();
extern int BJTdisto();
extern int BJTnoise();
#endif /* stdc */

