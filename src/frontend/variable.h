/*
  variable.h
  $Id$
*/

#ifndef _VARIABLE_H
#define _VARIABLE_H

#include "cpextern.h"

/* Variables that are accessible to the parser via $varname
 * expansions.  If the type is VT_LIST the value is a pointer to a
 * list of the elements.  */
struct variable {
    enum vt_types va_type;
    char *va_name;
    union {
        bool vV_bool;
        int vV_num;
        double vV_real;
        char *vV_string;
        struct variable *vV_list;
    } va_V;
    struct variable *va_next;      /* Link. */
} ;

#define va_bool   va_V.vV_bool
#define va_num    va_V.vV_num
#define va_real   va_V.vV_real
#define va_string va_V.vV_string
#define va_vlist  va_V.vV_list

struct xxx {
    struct variable *x_v;
    char x_char;
} ;


extern struct variable *variables;
extern bool cp_noglob;
extern bool cp_nonomatch;
extern bool cp_noclobber;
extern bool cp_ignoreeof;
extern bool cp_echo;

/* extern struct variable *variables; */
wordlist * cp_varwl(struct variable *var);
void cp_vset(char *varname, enum vt_types type, char *value);
struct variable * cp_setparse(wordlist *wl);
void cp_remvar(char *varname);
bool cp_getvar(char *name, enum vt_types type, void *retval);
wordlist * cp_variablesubst(wordlist *wlist);
wordlist * vareval(char *string);
void cp_vprint(void);
void free_struct_variable(struct variable *v);

#endif /* _VARIABLE_H */
