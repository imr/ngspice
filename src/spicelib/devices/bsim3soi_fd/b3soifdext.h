/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
Modified by Paolo Nenzi 2002
File: b3soifdext.h
**********/

extern int B3SOIFDacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIFDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIFDconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIFDgetic(GENmodel*,CKTcircuit*);
extern int B3SOIFDload(GENmodel*,CKTcircuit*);
extern int B3SOIFDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOIFDmParam(int,IFvalue*,GENmodel*);
extern void B3SOIFDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B3SOIFDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B3SOIFDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B3SOIFDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B3SOIFDtemp(GENmodel*,CKTcircuit*);
extern int B3SOIFDtrunc(GENmodel*,CKTcircuit*,double*);
extern int B3SOIFDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B3SOIFDunsetup(GENmodel*,CKTcircuit*);

