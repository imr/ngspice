/*************
 * Header file for inpcom.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_INPCOM_H
#define ngspice_INPCOM_H

struct card *insert_new_line(struct card *card, char *line,
                             int linenum, int linenum_orig, char *linesource);
char *inp_pathresolve(const char *name);

extern char* inp_remove_ws(char* s);
extern char* search_plain_identifier(char* str, const char* identifier);

extern int readdegparams(struct card* deck);
extern int adddegmonitors(struct card* deck);
extern int quote_degmons(struct card* deck);
extern int remsqrbra(struct card* deck);

#endif
