/*************
 * Header file for signal_handler.c
 * 1999 E. Rouat
 ************/

#ifndef SIGNAL_HANDLER_H_INCLUDED
#define SIGNAL_HANDLER_H_INCLUDED

RETSIGTYPE ft_sigintr(void);
RETSIGTYPE sigfloat(int sig, int code);
RETSIGTYPE sigstop(void);
RETSIGTYPE sigcont(void);
RETSIGTYPE sigill(void);
RETSIGTYPE sigbus(void);
RETSIGTYPE sigsegv(void);
RETSIGTYPE sig_sys(void);

#endif
