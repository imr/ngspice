/*************
 * Header file for signal_handler.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SIGNAL_HANDLER_H
#define ngspice_SIGNAL_HANDLER_H

RETSIGTYPE ft_sigintr(void);
RETSIGTYPE sigfloat(int code);
RETSIGTYPE sigstop(void);
RETSIGTYPE sigcont(void);
RETSIGTYPE sigill(void);
RETSIGTYPE sigbus(void);
RETSIGTYPE sigsegv(void);
RETSIGTYPE sigsegvsh(void);
RETSIGTYPE sig_sys(void);

extern JMP_BUF jbuf;

extern void ft_sigintr_cleanup(void);

#endif
