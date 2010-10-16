/*************
 * Header file for error.c
 * 1999 E. Rouat
 ************/

#ifndef ERROR_H_INCLUDED
#define ERROR_H_INCLUDED

void controlled_exit(int status);
void fperror(char *mess, int code);
void ft_sperror(int code, char *mess);
void fatal(void);

#endif
