/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int ISRCaccept(CKTcircuit*,GENmodel*);
extern int ISRCacLoad(GENmodel*,CKTcircuit*);
extern int ISRCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int ISRCdelete(GENmodel*,IFuid,GENinstance**);
extern void ISRCdestroy(GENmodel**);
extern int ISRCload(GENmodel*,CKTcircuit*);
extern int ISRCmDelete(GENmodel**,IFuid,GENmodel*);
extern int ISRCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int ISRCpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int ISRCtemp(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int ISRCaccept();
extern int ISRCacLoad();
extern int ISRCask();
extern int ISRCdelete();
extern void ISRCdestroy();
extern int ISRCload();
extern int ISRCmDelete();
extern int ISRCparam();
extern int ISRCpzLoad();
extern int ISRCtemp();
#endif /* stdc */
