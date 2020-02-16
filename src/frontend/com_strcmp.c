#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "com_strcmp.h"
#include "variable.h"


/* This is a truly evil thing */
void
com_strcmp(wordlist *wl)
{
    char *var, *s1, *s2;
    int i;

    var = wl->wl_word;
    s1 = cp_unquote(wl->wl_next->wl_word);
    s2 = cp_unquote(wl->wl_next->wl_next->wl_word);

    i = strcmp(s1, s2);
    tfree(s1);/*DG  cp_unquote memory leak*/
    tfree(s2);
    cp_vset(var, CP_NUM, &i);
}
