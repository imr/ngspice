/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

extern int HFETAacLoad(GENmodel*,CKTcircuit*);
extern int HFETAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int HFETAdelete(GENinstance*);
extern void HFETAdestroy(void);
extern int HFETAgetic(GENmodel*,CKTcircuit*);
extern int HFETAload(GENmodel*,CKTcircuit*);
extern int HFETAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int HFETAmDelete(GENmodel*);
extern int HFETAmParam(int,IFvalue*,GENmodel*);
extern int HFETAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int HFETApzLoad(GENmodel*, CKTcircuit*, SPcomplex*);
extern int HFETAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HFETAtemp(GENmodel*,CKTcircuit*);
extern int HFETAtrunc(GENmodel*,CKTcircuit*,double*);
extern int HFETAunsetup(GENmodel*,CKTcircuit*);

#ifdef KLU
extern int HFETAbindCSC (GENmodel*, CKTcircuit*) ;
extern int HFETAbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int HFETAbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
