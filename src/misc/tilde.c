/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2002 R. Oktas, <roktas@omu.edu.tr>
**********/

#include "ngspice/ngspice.h"
#include "tilde.h"
#include "ngspice/stringskip.h"

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#if defined(__MINGW32__) || defined(_MSC_VER)
#undef BOOLEAN
#include <windows.h> /* win32 functions */
#include "shlobj.h"  /* SHGetFolderPath */
#endif

/* XXX To prevent a name collision with `readline's `tilde_expand',
   the original name: `tilde_expand' has changed to `tildexpand'. This
   situation naturally brings to mind that `tilde_expand' could be used
   directly from `readline' (since it will already be included if we
   wish to activate the `readline' support). Following implementation of
   'tilde expanding' has some problems which constitutes another good
   reason why it should be replaced: eg. it returns NULL which should
   not behave this way, IMHO.  Anyway... Don't care for the moment, may
   be in the future. -- ro */


char *
tildexpand(char *string)
{
#ifdef HAVE_PWD_H
    char buf[BSIZE_SP];
    char *k, c;
#endif
#if defined(__MINGW32__) || defined(_MSC_VER)
    char buf2[BSIZE_SP];
#endif
    char *result = NULL;

    if (!string)
	return NULL;

    TEMPORARY_SKIP_WS_X1(string);

    if (*string != '~')
        return copy(string);

    string += 1;

    if (!*string || *string == '/') {
       /* First try the environment setting. May also make life easier
          for non-unix platforms, eg. MS-DOS. -- ro */
       result = getenv("HOME");
#ifdef HAVE_PWD_H
       /* Can't find a result from the environment, let's try
          the other stuff. -- ro */
       if (!result) {
           struct passwd *pw;
           pw = getpwuid(getuid());
           if (pw)
             result = pw->pw_dir;
           *buf = '\0';
       }

    } else {
       struct passwd *pw;
	k = buf;
	while ((c = *string) && c != '/')
		*k++ = c, string++;
	*k = '\0';
	pw = getpwnam(buf);
       if (pw)
         result = pw->pw_dir;
#endif
    }
    if (result) {
#ifdef HAVE_PWD_H
       strcpy(buf, result);
	if (*string)
	    strcat(buf, string);
	return copy(buf);

    } else
	return NULL;

#else
       /* Emulate the old behavior to prevent side effects. -- ro */
       return copy(string);
    }
#if defined(__MINGW32__) || defined(_MSC_VER)
    else if(SUCCEEDED(SHGetFolderPath(NULL,
                             CSIDL_PERSONAL,
                             NULL,
                             0,
                             buf2)))
    {
        if (*string)
            strcat(buf2, string);
        return copy(buf2);
    }
#endif
    return NULL;
#endif
}

