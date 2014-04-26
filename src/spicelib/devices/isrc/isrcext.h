/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int ISRCaccept(CKTcircuit*,GENmodel*);
extern int ISRCacLoad(GENmodel*,CKTcircuit*);
extern int ISRCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int ISRCdelete(GENinstance*);
extern void ISRCdestroy(void);
extern int ISRCload(GENmodel*,CKTcircuit*);
extern int ISRCmDelete(GENmodel*);
extern int ISRCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int ISRCpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int ISRCtemp(GENmodel*,CKTcircuit*);

#ifdef USE_CUSPICE
extern int ISRCsetup (SMPmatrix *, GENmodel *, CKTcircuit *, int *) ;
extern int ISRCtopology (GENmodel *, CKTcircuit *, int *, int *) ;
#endif
