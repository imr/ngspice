/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soifdext.h
**********/

#ifdef __STDC__
extern int B3SOIFDacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIFDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIFDconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIFDdelete(GENmodel*,IFuid,GENinstance**);
extern void B3SOIFDdestroy(GENmodel**);
extern int B3SOIFDgetic(GENmodel*,CKTcircuit*);
extern int B3SOIFDload(GENmodel*,CKTcircuit*);
extern int B3SOIFDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOIFDmDelete(GENmodel**,IFuid,GENmodel*);
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

#else /* stdc */
extern int B3SOIFDacLoad();
extern int B3SOIFDdelete();
extern void B3SOIFDdestroy();
extern int B3SOIFDgetic();
extern int B3SOIFDload();
extern int B3SOIFDmDelete();
extern int B3SOIFDask();
extern int B3SOIFDmAsk();
extern int B3SOIFDconvTest();
extern int B3SOIFDtemp();
extern int B3SOIFDmParam();
extern void B3SOIFDmosCap();
extern int B3SOIFDparam();
extern int B3SOIFDpzLoad();
extern int B3SOIFDsetup();
extern int B3SOIFDtrunc();
extern int B3SOIFDnoise();
extern int B3SOIFDunsetup();

#endif /* stdc */

