/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int JFETacLoad(GENmodel*,CKTcircuit*);
extern int JFETask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int JFETdelete(GENmodel*,IFuid,GENinstance**);
extern void JFETdestroy(GENmodel**);
extern int JFETgetic(GENmodel*,CKTcircuit*);
extern int JFETload(GENmodel*,CKTcircuit*);
extern int JFETmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int JFETmDelete(GENmodel**,IFuid,GENmodel*);
extern int JFETmParam(int,IFvalue*,GENmodel*);
extern int JFETparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int JFETpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int JFETsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int JFETunsetup(GENmodel*,CKTcircuit*);
extern int JFETtemp(GENmodel*,CKTcircuit*);
extern int JFETtrunc(GENmodel*,CKTcircuit*,double*);
extern int JFETdisto(int,GENmodel*,CKTcircuit*);
extern int JFETnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int JFETacLoad();
extern int JFETask();
extern int JFETdelete();
extern void JFETdestroy();
extern int JFETgetic();
extern int JFETload();
extern int JFETmAsk();
extern int JFETmDelete();
extern int JFETmParam();
extern int JFETparam();
extern int JFETpzLoad();
extern int JFETsetup();
extern int JFETunsetup();
extern int JFETtemp();
extern int JFETtrunc();
extern int JFETdisto();
extern int JFETnoise();
#endif /* stdc */
