
extern int TXLaccept(CKTcircuit*,GENmodel*);
extern int TXLask(CKTcircuit*, GENinstance*, int, IFvalue*, IFvalue*);
extern int TXLfindBr(CKTcircuit*, GENmodel*, IFuid);
extern int TXLload(GENmodel*,CKTcircuit*);
extern int TXLmodAsk(CKTcircuit*, GENmodel*, int, IFvalue*);
extern int TXLmParam(int,IFvalue*,GENmodel*);
extern int TXLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TXLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TXLunsetup(GENmodel*, CKTcircuit*);
extern int TXLdevDelete(GENinstance*);
