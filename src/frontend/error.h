/*************
 * Header file for error.c
 * 1999 E. Rouat
 ************/

#ifndef ERROR_H_INCLUDED
#define ERROR_H_INCLUDED

void fperror(char *mess, int code);
void ft_sperror(int code, char *mess);
void fatal(void);
void internalerror(char *message);




#endif
