/*************
 * Header file for inpcom.c
 * 1999 E. Rouat
 ************/

#ifndef INPCOM_H_INCLUDED
#define INPCOM_H_INCLUDED

FILE * inp_pathopen(char *name, char *mode);
void inp_readall(FILE *fp, struct line **data);
void inp_casefix(register char *string);

#endif
