#include "ngspice/config.h"
#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "variable.h"
#include "com_set.h"


/* The set command. Syntax is set [opt ...] [opt = val ...]. Val may
 * be a string, an int, a float, or a list of the form (elt1 elt2
 * ...).  */
void
com_set(wordlist *wl)
{
    struct variable *vars, *oldvar;

    if (wl == NULL) {
        cp_vprint();
        return;
    }
    vars = cp_setparse(wl);

    /* This is sort of a hassle... */
    while (vars) {
        void *s;
        switch (vars->va_type) {
        case CP_BOOL:
            s = &vars->va_bool;
            break;
        case CP_NUM:
            s = &vars->va_num;
            break;
        case CP_REAL:
            s = &vars->va_real;
            break;
        case CP_STRING:
            s = vars->va_string;
            break;
        case CP_LIST:
            s = vars->va_vlist;
            break;
        default:
            s = NULL;
        }
        cp_vset(vars->va_name, vars->va_type, s);
        oldvar = vars;
        vars = vars->va_next;
        /* va: avoid memory leak: free oldvar carefully */
        tfree(oldvar->va_name);
        if (oldvar->va_type == CP_STRING)
            tfree(oldvar->va_string); /* copied in cp_vset */
        /* don't free oldvar->va_list! This structure is used furthermore! */
        tfree(oldvar);
    }

    return;
}
