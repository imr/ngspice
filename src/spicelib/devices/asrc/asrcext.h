/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int ASRCask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue*);
extern int ASRCconvTest(GENmodel *, CKTcircuit *);
extern int ASRCdelete(GENinstance *);
extern void ASRCdestroy(void);
extern int ASRCfindBr(CKTcircuit *, GENmodel *, IFuid);
extern int ASRCload(GENmodel *, CKTcircuit *);
extern int ASRCparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int ASRCpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int ASRCacLoad(GENmodel *, CKTcircuit *);
extern int ASRCsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int ASRCunsetup(GENmodel *, CKTcircuit *);
extern int ASRCtemp(GENmodel *, CKTcircuit *);
