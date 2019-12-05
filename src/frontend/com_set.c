#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "variable.h"
#include "com_set.h"


/* The set command.
 *
 * Syntax is set [var [= val] ...]
 *
 * var is the name of the variable to be defined. Quoting allows special
 *      characters such as = to be included as part of the name.
 * val may be a string, an int, a float, a bool, or a list of the form
 * ( elt1 ... ).
 *
 * With no var value, all variables that are currently defined are printed
 * Without the "= val" portion, the variable becomes a Boolean set to true.
 * Lists must have spaces both after the leading '(' and before the
 *      trailing ')'. Individual elements may be of any type.
 * Quoted expressions are taken to be strings in all cases and quoting a
 *      grouping character ("(" or ")") suppresses its special properties.
 *      Further, words "(" and ")" within a list are ordinary words.
 *
 * This function may alter the input wordlist, but on return its resources
 * can be freed in the normal manner.
 */
void com_set(wordlist *wl)
{
    /* Handle case of printing defined variables */
    if (wl == (wordlist *) NULL) {
        cp_vprint();
        return;
    }

    struct variable *oldvar;

    struct variable *vars = cp_setparse(wl);

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

        /* Free allocations associated with the current variable */
        txfree(oldvar->va_name);
        if (oldvar->va_type == CP_STRING){
            txfree(oldvar->va_string); /* copied in cp_vset */
        }
        /* don't free oldvar->va_list! This structure is used furthermore! */
        txfree(oldvar);
    }
}
