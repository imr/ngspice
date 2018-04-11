/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soiddext.h
Modifed by Paolo Nenzi 2002
**********/

extern int B3SOIDDacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIDDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIDDconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIDDgetic(GENmodel*,CKTcircuit*);
extern int B3SOIDDload(GENmodel*,CKTcircuit*);
extern int B3SOIDDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOIDDmParam(int,IFvalue*,GENmodel*);
extern void B3SOIDDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B3SOIDDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B3SOIDDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B3SOIDDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B3SOIDDtemp(GENmodel*,CKTcircuit*);
extern int B3SOIDDtrunc(GENmodel*,CKTcircuit*,double*);
extern int B3SOIDDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B3SOIDDunsetup(GENmodel*,CKTcircuit*);
