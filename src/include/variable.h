#ifndef _VARIABLE_H
#define _VARIABLE_H

/* Variables that are accessible to the parser via $varname expansions. 
 * If the type is VT_LIST the value is a pointer to a list of the elements.
 */

struct variable {
    char va_type;
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

#define va_bool  va_V.vV_bool
#define va_num    va_V.vV_num
#define va_real  va_V.vV_real
#define va_string   va_V.vV_string
#define va_vlist     va_V.vV_list

enum vt_types {
  VT_BOOL,
  VT_NUM,
  VT_REAL,
  VT_STRING,
  VT_LIST
};


#endif
