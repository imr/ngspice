
extern int TXLaccept(CKTcircuit*,GENmodel*);
extern int TXLask(CKTcircuit*, GENinstance*, int, IFvalue*, IFvalue*);
extern int TXLdelete(GENinstance*);
extern void TXLdestroy(void);
extern int TXLfindBr(CKTcircuit*, GENmodel*, IFuid);
extern int TXLload(GENmodel*,CKTcircuit*);
extern int TXLmodAsk(CKTcircuit*, GENmodel*, int, IFvalue*);

extern int TXLmDelete(GENmodel*);
extern int TXLmParam(int,IFvalue*,GENmodel*);
extern int TXLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TXLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TXLunsetup(GENmodel*, CKTcircuit*);

#ifdef KLU
extern int TXLbindCSC (GENmodel*, CKTcircuit*) ;
extern int TXLbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int TXLbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
