#include <bool.h>
#include <wordlist.h>

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

    cp_vset(var, VT_NUM, (char *) &i);
    return;
}
