/* Enhancement-242: native n-port device -- extern declarations */
#ifndef ngspice_NPORTEXT_H
#define ngspice_NPORTEXT_H

extern int NPORTacLoad(GENmodel *, CKTcircuit *);
extern int NPORTdelete(GENinstance *);
extern int NPORTload(GENmodel *, CKTcircuit *);
extern int NPORTmParam(int, IFvalue *, GENmodel *);
extern int NPORTparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int NPORTsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NPORTunsetup(GENmodel *, CKTcircuit *);
extern int NPORTtemp(GENmodel *, CKTcircuit *);
extern int NPORTask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NPORTmAsk(CKTcircuit *, GENmodel *, int, IFvalue *);
extern int NPORTbindCSC(GENmodel *, CKTcircuit *);
extern int NPORTbindCSCComplex(GENmodel *, CKTcircuit *);
extern int NPORTbindCSCComplexToReal(GENmodel *, CKTcircuit *);

#endif
