/*************
* com_chdir.c
************/

#include "ngspice/ngspice.h"

#include "ngspice/wordlist.h"

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#include "com_chdir.h"
#include "ngspice/cpextern.h"


void
com_chdir(wordlist *wl)
{
    char *s;
#ifdef HAVE_PWD_H
    struct passwd *pw;
#endif
#ifdef HAVE_GETCWD
    char localbuf[257];
#endif
    int copied = 0;

    s = NULL;

    if (wl == NULL) {

        s = getenv("HOME");
        if (!s)
            s = getenv("USERPROFILE");

#ifdef HAVE_PWD_H
        if (s == NULL) {
            pw = getpwuid(getuid());
            if (pw == NULL) {
                fprintf(cp_err, "Can't get your password entry\n");
                return;
            }
            s = pw->pw_dir;
        }
#endif
    } else {
        s = cp_unquote(wl->wl_word);
        copied = 1;
    }


    if (s != NULL)
        if (chdir(s) == -1)
            perror(s);

    if (copied)
        tfree(s);

#ifdef HAVE_GETCWD
    s = getcwd(localbuf, sizeof(localbuf));
    if (s)
        printf("Current directory: %s\n", s);
    else
        fprintf(cp_err, "Can't get current working directory.\n");
#endif

}

/* just print the current working directory */
void
com_getcwd(wordlist *wl)
{
    NG_IGNORE(wl);
#ifdef HAVE_GETCWD
    char *s;
    char localbuf[257];
    s = getcwd(localbuf, sizeof(localbuf));
    if (s)
        printf("Current directory: %s\n", s);
    else
        fprintf(cp_err, "Can't get current working directory.\n");
#else
    fprintf(cp_err, "Error, function getcwd not available\n");
#endif
}
