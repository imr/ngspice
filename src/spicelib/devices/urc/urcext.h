/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int URCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int URCdelete(GENmodel*,IFuid,GENinstance**);
extern void URCdestroy(GENmodel**);
extern int URCmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int URCmDelete(GENmodel**,IFuid,GENmodel*);
extern int URCmParam(int,IFvalue*,GENmodel*);
extern int URCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int URCsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int URCunsetup(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int URCask();
extern int URCdelete();
extern void URCdestroy();
extern int URCmAsk();
extern int URCmDelete();
extern int URCmParam();
extern int URCparam();
extern int URCsetup();
extern int URCunsetup();
#endif /* stdc */
