#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "variable.h"
#include "com_set.h"

wordlist* readifile(wordlist*);


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

    /* special case input redirection*/
    wordlist *ww = wl->wl_next;
    if (ww && eq(ww->wl_word, "<"))
        wl = readifile(wl);

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
}

/* read a file from cp_in, add the tokens to a wordlist and
   create an input for a string list like set ccc = ( 3 5 7 ).
   Comment lines in input file (starting with '*') are ignored. */
wordlist*
readifile(wordlist* win)
{
    /* max file size */
    char intoken[4096];
    /* save start address */
    wordlist* tw = win;
    char *tmpstr;

    /* delete wordlist from '<' on */
    wl_free(win->wl_next);
    wl_append_word(&win, &win, copy("="));
    wl_append_word(&win, &win, copy("("));
    /* read a line. If it starts with '*', ignore it */
    while (fgets(intoken, 4096, cp_in) != NULL) {
        if (intoken[0] == '*')
            continue;
        char* delstr;
        char* instr = delstr = copy(intoken);
        /* get all tokens, ignoring '\n' 
           and add to string list */
        while ((tmpstr = gettok(&instr)) != NULL) {
            wl_append_word(&win, &win, tmpstr);
        }
        tfree(delstr);
    }
    wl_append_word(&win, &win, copy(")"));
    /* close and reset cp_in 
    (was opened in streams.c:84) */
    cp_ioreset();
    return tw;
#if 0
    size_t retval = fread(intoken, 1, 4096, cp_in);
    intoken[retval] = '\0';
    char* delstr;
    char* instr = delstr = copy(intoken);
    /* get all tokens, ignoring '\n' 
    and add to string list */
    while ((tmpstr = gettok(&instr)) != NULL) {
        wl_append_word(&win, &win, tmpstr);
    }
    tfree(delstr);
    wl_append_word(&win, &win, copy(")"));
    /* close and reset cp_in 
    (was opened in streams.c:84) */
    cp_ioreset();
    return tw;
#endif
}
