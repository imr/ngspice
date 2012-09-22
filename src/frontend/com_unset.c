/*************
* com_unset.c
************/

#include "ngspice/config.h"
#include "ngspice/ngspice.h"

#include "ngspice/macros.h"
#include "ngspice/bool.h"

#include "com_unset.h"
#include "variable.h"

void
com_unset(wordlist *wl)
{
    char *name;
    struct variable *var, *nv;

    if (eq(wl->wl_word, "*")) {
        for (var = variables; var; var = nv) {
            nv = var->va_next;
            cp_remvar(var->va_name);
        }
        wl = wl->wl_next;
    }
    while (wl != NULL) {
        name = wl->wl_word;
        cp_remvar(name);
        wl = wl->wl_next;
    }
}
