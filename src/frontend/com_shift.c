/*************
* com_shift.c
************/

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "com_shift.h"
#include "variable.h"
#include "ngspice/cpextern.h"


/* Shift a list variable, by default argv, one to the left (or more if
 * a second argument is given.  */
void
com_shift(wordlist *wl)
{
    struct variable *v, *vv;
    char *n = "argv";
    int num = 1;

    if (wl) {
        n = wl->wl_word;
        wl = wl->wl_next;
    }
    if (wl)
        num = scannum(wl->wl_word);

    for (v = variables; v; v = v->va_next)
        if (eq(v->va_name, n))
            break;
    if (!v) {
        fprintf(cp_err, "Error: %s: no such variable\n", n);
        return;
    }
    if (v->va_type != CP_LIST) {
        fprintf(cp_err, "Error: %s not of type list\n", n);
        return;
    }
    for (vv = v->va_vlist; vv && (num > 0); num--)
        vv = vv->va_next;
    if (num) {
        fprintf(cp_err, "Error: variable %s not long enough\n", n);
        return;
    }

    v->va_vlist = vv;
}
