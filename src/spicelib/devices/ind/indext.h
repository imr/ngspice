/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
#ifndef _INDEXT_H
#define _INDEXT_H

extern int INDacLoad(GENmodel*,CKTcircuit*);
extern int INDask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int INDmAsk(CKTcircuit*, GENmodel*, int, IFvalue*);
extern int INDload(GENmodel*,CKTcircuit*);
extern int INDmParam(int, IFvalue*, GENmodel*);
extern int INDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int INDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int INDsAcLoad(GENmodel*,CKTcircuit*);
extern int INDsLoad(GENmodel*,CKTcircuit*);
extern void INDsPrint(GENmodel*,CKTcircuit*);
extern int INDsSetup(SENstruct*,GENmodel*);
extern int INDsUpdate(GENmodel*,CKTcircuit*);
extern int INDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int INDunsetup(GENmodel*,CKTcircuit*);
extern int INDtemp(GENmodel*, CKTcircuit*);
extern int INDtrunc(GENmodel*,CKTcircuit*,double*);

extern int MUTacLoad(GENmodel*,CKTcircuit*);
extern int MUTask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MUTparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int MUTpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern void MUTsPrint(GENmodel*,CKTcircuit*);
extern int MUTsSetup(SENstruct*,GENmodel*);
extern int MUTsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MUTtemp(GENmodel *inModel, CKTcircuit *ckt);
#endif
