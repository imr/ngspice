/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int INDacLoad(GENmodel*,CKTcircuit*);
extern int INDask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int INDdelete(GENmodel*,IFuid,GENinstance**);
extern void INDdestroy(GENmodel**);
extern int INDload(GENmodel*,CKTcircuit*);
extern int INDmDelete(GENmodel**,IFuid,GENmodel*);
extern int INDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int INDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int INDsAcLoad(GENmodel*,CKTcircuit*);
extern int INDsLoad(GENmodel*,CKTcircuit*);
extern void INDsPrint(GENmodel*,CKTcircuit*);
extern int INDsSetup(SENstruct*,GENmodel*);
extern int INDsUpdate(GENmodel*,CKTcircuit*);
extern int INDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int INDunsetup(GENmodel*,CKTcircuit*);
extern int INDtrunc(GENmodel*,CKTcircuit*,double*);
#else /* stdc */
extern int INDacLoad();
extern int INDask();
extern int INDdelete();
extern void INDdestroy();
extern int INDload();
extern int INDmDelete();
extern int INDparam();
extern int INDpzLoad();
extern int INDsAcLoad();
extern int INDsLoad();
extern void INDsPrint();
extern int INDsSetup();
extern int INDsUpdate();
extern int INDsetup();
extern int INDunsetup();
extern int INDtrunc();
#endif /* stdc */

#ifdef MUTUAL

#ifdef __STDC__
extern int MUTacLoad(GENmodel*,CKTcircuit*);
extern int MUTask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MUTdelete(GENmodel*,IFuid,GENinstance**);
extern void MUTdestroy(GENmodel**);
extern int MUTmDelete(GENmodel**,IFuid,GENmodel*);
extern int MUTparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int MUTpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern void MUTsPrint(GENmodel*,CKTcircuit*);
extern int MUTsSetup(SENstruct*,GENmodel*);
extern int MUTsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
#else /* stdc */
extern int MUTacLoad();
extern int MUTask();
extern int MUTdelete();
extern void MUTdestroy();
extern int MUTmDelete();
extern int MUTparam();
extern int MUTpzLoad();
extern void MUTsPrint();
extern int MUTsSetup();
extern int MUTsetup();
#endif /* stdc */

#endif
