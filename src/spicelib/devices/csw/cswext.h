/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
**********/

#ifdef __STDC__
extern int CSWask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CSWacLoad(GENmodel*,CKTcircuit*);
extern int CSWdelete(GENmodel*,IFuid,GENinstance**);
extern void CSWdestroy(GENmodel**);
extern int CSWload(GENmodel*,CKTcircuit*);
extern int CSWmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int CSWmDelete(GENmodel**,IFuid,GENmodel*);
extern int CSWmParam(int,IFvalue*,GENmodel*);
extern int CSWparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CSWpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CSWsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int CSWnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
#else /* stdc */
extern int CSWask();
extern int CSWacLoad();
extern int CSWdelete();
extern void CSWdestroy();
extern int CSWload();
extern int CSWmAsk();
extern int CSWmDelete();
extern int CSWmParam();
extern int CSWparam();
extern int CSWpzLoad();
extern int CSWsetup();
extern int CSWnoise();
#endif /* stdc */

