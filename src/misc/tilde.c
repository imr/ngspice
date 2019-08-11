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
#include <shlobj.h>  /* SHGetFolderPath */
#endif

/* For Windows MINGW and Visual Studio: Expand the tilde '~' to the
   environmental variable HOME, if set by the user. If not set, it
   expands to the env variable USERPROFILE, that typically returns
   C:\Users\<user name>. If this is not available, a folder path led
   to by CSIDL_PERSONAL, i.e.  C:\Users\<user name>\Documents.
   If not MINGW or Visual Studio, and if  HAVE_PWD_H is defined,
   then '~' is expanded to HOME, else string is returned.
*/
char *
tildexpand(char *string)
{
    char *result = NULL;

    if (!string)
        return NULL;

    string = skip_ws(string);

    if (*string != '~')
        return copy(string);

#if defined(__MINGW32__) || defined(_MSC_VER)
    char buf2[BSIZE_SP];
    string += 1;
    result = getenv("HOME");
    if (!result)
        result = getenv("USERPROFILE");
    if (!result)
        if (SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, buf2))) {
            if (*string)
                strcat(buf2, string);
            return copy(buf2);
        }
        else
            return NULL;
    else {
        strcpy(buf2, result);
        if (*string)
            strcat(buf2, string);
        return copy(buf2);
    }
#else

#ifdef HAVE_PWD_H
    char buf[BSIZE_SP];
    char *k, c;
#endif

    string += 1;

    if (!*string || *string == '/') {
       /* First try the environment setting. */
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

    return NULL;
#endif
#endif
}

