/*************
 * Header file for signal_handler.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_SIGNAL_HANDLER_H
#define ngspice_SIGNAL_HANDLER_H

void ft_sigintr(void);
void sigfloat(int code);
void sigstop(void);
void sigcont(void);
void sigill(void);
void sigbus(void);
void sigsegv(void);
void sigsegvsh(void);
void sig_sys(void);

extern JMP_BUF jbuf;

extern void ft_sigintr_cleanup(void);

#endif
