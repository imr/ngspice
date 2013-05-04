#ifndef _CAPINIT_H
#define _CAPINIT_H

extern IFparm CAPpTable[ ];
extern IFparm CAPmPTable[ ];
extern char *CAPnames[ ];
extern int CAPpTSize;
extern int CAPmPTSize;
extern int CAPnSize;
extern int CAPiSize;
extern int CAPmSize;

#ifdef KLU
extern int CAPisLinear ;
extern int CAPisLinearStatic ;
#endif

#endif
