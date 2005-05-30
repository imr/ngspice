/*************
* com_unset.c
* $Id$
************/

#include <config.h>
#include <ngspice.h>

#include <macros.h>
#include <bool.h>

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
    return;
}
