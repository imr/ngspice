/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/

#ifdef __STDC__
extern int MESacLoad(GENmodel*,CKTcircuit*);
extern int MESask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MESdelete(GENmodel*,IFuid,GENinstance**);
extern void MESdestroy(GENmodel**);
extern int MESgetic(GENmodel*,CKTcircuit*);
extern int MESload(GENmodel*,CKTcircuit*);
extern int MESmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MESmDelete(GENmodel**,IFuid,GENmodel*);
extern int MESmParam(int,IFvalue*,GENmodel*);
extern int MESparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int MESpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MESsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MESunsetup(GENmodel*,CKTcircuit*);
extern int MEStemp(GENmodel*,CKTcircuit*);
extern int MEStrunc(GENmodel*,CKTcircuit*,double*);
extern int MESdisto(int,GENmodel*,CKTcircuit*);
extern int MESnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int MESacLoad();
extern int MESask();
extern int MESdelete();
extern void MESdestroy();
extern int MESgetic();
extern int MESload();
extern int MESmAsk();
extern int MESmDelete();
extern int MESmParam();
extern int MESparam();
extern int MESpzLoad();
extern int MESsetup();
extern int MESunsetup();
extern int MEStemp();
extern int MEStrunc();
extern int MESdisto();
extern int MESnoise();
#endif /* stdc */
