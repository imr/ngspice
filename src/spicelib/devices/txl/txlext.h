
extern int TXLaccept(CKTcircuit*,GENmodel*);
extern int TXLask(CKTcircuit*, GENinstance*, int, IFvalue*, IFvalue*);
extern int TXLdelete(GENmodel*,IFuid,GENinstance**);
extern void TXLdestroy(GENmodel**);
extern int TXLfindBr(CKTcircuit*, GENmodel*, IFuid);
extern int TXLload(GENmodel*,CKTcircuit*);
extern int TXLmodAsk(CKTcircuit*, GENmodel*, int, IFvalue*);

extern int TXLmDelete(GENmodel**,IFuid,GENmodel*);
extern int TXLmParam(int,IFvalue*,GENmodel*);
extern int TXLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TXLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TXLunsetup(GENmodel*, CKTcircuit*);
