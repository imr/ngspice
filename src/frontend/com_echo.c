/*************
* com_echo.c
************/

#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"
#include "ngspice/bool.h"

#include "com_echo.h"
#include "ngspice/cpextern.h"


void
com_echo(wordlist *wlist)
{   char*copyword;
    bool nl = TRUE;

    if (wlist && eq(wlist->wl_word, "-n")) {
        wlist = wlist->wl_next;
        nl = FALSE;
    }

    while (wlist) {
        /* fputs(cp_unquote(wlist->wl_word), cp_out); very bad the string allocated by cp_unquote could not be freed: memory leak*/
        copyword = cp_unquote(wlist->wl_word);
        fputs(copyword, cp_out);
        tfree(copyword);
        if (wlist->wl_next)
            fputs(" ", cp_out);
        wlist = wlist->wl_next;
    }

    if (nl)
        fputs("\n", cp_out);
}
