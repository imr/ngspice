/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
**********/

/* extern int CPLaccept(CKTcircuit*,GENmodel*); */
extern int CPLdelete(GENmodel*,IFuid,GENinstance**);
extern void CPLdestroy(GENmodel**);
extern int CPLload(GENmodel*,CKTcircuit*);
extern int CPLmDelete(GENmodel**,IFuid,GENmodel*);
extern int CPLmParam(int,IFvalue*,GENmodel*);
extern int CPLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CPLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
