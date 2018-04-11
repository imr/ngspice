/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soipdext.h
Modified by Paolo Nenzi 2002
**********/

extern int B3SOIPDacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIPDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIPDconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIPDgetic(GENmodel*,CKTcircuit*);
extern int B3SOIPDload(GENmodel*,CKTcircuit*);
extern int B3SOIPDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOIPDmParam(int,IFvalue*,GENmodel*);
extern void B3SOIPDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B3SOIPDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B3SOIPDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B3SOIPDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B3SOIPDtemp(GENmodel*,CKTcircuit*);
extern int B3SOIPDtrunc(GENmodel*,CKTcircuit*,double*);
extern int B3SOIPDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B3SOIPDunsetup(GENmodel*,CKTcircuit*);
