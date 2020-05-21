/*************
* com_unset.c
************/

#include "ngspice/ngspice.h"

#include "ngspice/macros.h"
#include "ngspice/bool.h"

#include "com_unset.h"
#include "variable.h"

/* clear variables (by name or all) */
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

/* clear all variables (called from com_quit) */
void unset_all(void){
    struct variable *var, *nv;
    for (var = variables; var; var = nv) {
        nv = var->va_next;
        cp_remvar(var->va_name);
    }
}
