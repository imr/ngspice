/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int ASRCask(CKTcircuit*,GENinstance *,int,IFvalue *,IFvalue*);
extern int ASRCconvTest(GENmodel *,CKTcircuit*);
extern int ASRCdelete(GENmodel *,IFuid,GENinstance **);
extern void ASRCdestroy(GENmodel**);
extern int ASRCfindBr(CKTcircuit *,GENmodel *,IFuid);
extern int ASRCload(GENmodel *,CKTcircuit*);
extern int ASRCmDelete(GENmodel**,IFuid,GENmodel*);
extern int ASRCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int ASRCpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int ASRCacLoad(GENmodel*,CKTcircuit*);
extern int ASRCsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int ASRCunsetup(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int ASRCask();
extern int ASRCconvTest();
extern int ASRCdelete();
extern void ASRCdestroy();
extern int ASRCfindBr();
extern int ASRCload();
extern int ASRCmDelete();
extern int ASRCparam();
extern int ASRCpzLoad();
extern int ASRCacLoad();
extern int ASRCsetup();
extern int ASRCunsetup();
#endif /* stdc */

