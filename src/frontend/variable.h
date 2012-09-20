/*
  variable.h
*/

#ifndef _VARIABLE_H
#define _VARIABLE_H

#include "ngspice/cpextern.h"

/* Variables that are accessible to the parser via $varname
 * expansions.  If the type is CP_LIST the value is a pointer to a
 * list of the elements.  */
struct variable {
    enum cp_types va_type;
    char *va_name;
    union {
        bool vV_bool;
        int vV_num;
        double vV_real;
        char *vV_string;
        struct variable *vV_list;
    } va_V;
    struct variable *va_next;      /* Link. */
};

#define va_bool   va_V.vV_bool
#define va_num    va_V.vV_num
#define va_real   va_V.vV_real
#define va_string va_V.vV_string
#define va_vlist  va_V.vV_list

struct xxx {
    struct variable *x_v;
    char x_char;
};


extern struct variable *variables;
extern bool cp_echo;

/* extern struct variable *variables; */
wordlist *cp_varwl(struct variable *var);
wordlist *cp_variablesubst(wordlist *wlist);
void free_struct_variable(struct variable *v);

#endif
