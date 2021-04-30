/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
#ifndef __BJTEXT_H
#define __BJTEXT_H


extern int BJTacLoad(GENmodel *,CKTcircuit*);
extern int BJTask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BJTconvTest(GENmodel*,CKTcircuit*);
extern int BJTdelete(GENinstance*);
extern int BJTgetic(GENmodel*,CKTcircuit*);
extern int BJTload(GENmodel*,CKTcircuit*);
extern int BJTmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int BJTmParam(int,IFvalue*,GENmodel*);
extern int BJTparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BJTpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BJTsAcLoad(GENmodel*,CKTcircuit*);
extern int BJTsLoad(GENmodel*,CKTcircuit*);
extern void BJTsPrint(GENmodel*,CKTcircuit*);
extern int BJTsSetup(SENstruct*,GENmodel*);
extern int BJTsUpdate(GENmodel*,CKTcircuit*);
extern int BJTsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BJTunsetup(GENmodel*,CKTcircuit*);
extern int BJTtemp(GENmodel*,CKTcircuit*);
extern int BJTtrunc(GENmodel*,CKTcircuit*,double*);
extern int BJTdisto(int,GENmodel*,CKTcircuit*);
extern int BJTnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BJTdSetup(GENmodel*, CKTcircuit*);
extern int BJTsoaCheck(CKTcircuit *, GENmodel *);

#endif
