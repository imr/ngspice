#include <config.h>
#include <ngspice.h>

#include <bool.h>
#include <wordlist.h>

#include "variable.h"


/* The set command. Syntax is set [opt ...] [opt = val ...]. Val may
 * be a string, an int, a float, or a list of the form (elt1 elt2
 * ...).  */
void
com_set(wordlist *wl)
{
    struct variable *vars, *oldvar;
    char *s;

    if (wl == NULL) {
        cp_vprint();
        return;
    }
    vars = cp_setparse(wl);

    /* This is sort of a hassle... */
    while (vars) { 
        switch (vars->va_type) {
	case VT_BOOL:
            s = (char *) &vars->va_bool;
            break;
	case VT_NUM:
            s = (char *) &vars->va_num;
            break;
	case VT_REAL:
            s = (char *) &vars->va_real;
            break;
	case VT_STRING:
            s = vars->va_string;
            break;
	case VT_LIST:
            s = (char *) vars->va_vlist;
            break;
	default:
	    s = (char *) NULL;
        }
        cp_vset(vars->va_name, vars->va_type, s);
	oldvar = vars;
        vars = vars->va_next;
	/* va: avoid memory leak: free oldvar carefully */
        tfree(oldvar->va_name);
        if (oldvar->va_type==VT_STRING) 
            tfree(oldvar->va_string); /* copied in cp_vset */
        /* don't free oldvar->va_list! This structure is used furthermore! */
        tfree(oldvar);
    }
	
    return;
}

