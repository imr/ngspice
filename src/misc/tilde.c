/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/


#include <config.h>
#include "ngspice.h"
#include <stdio.h>
#include "tilde.h"

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif



char *
tilde_expand(char *string)
{
#ifdef HAVE_PWD_H
    struct passwd *pw;
    /*extern struct passwd *getpwuid( );*/
    char	buf[BSIZE_SP];
    char	*k, c;

    if (!string)
	return NULL;

    while (*string && isspace(*string))
	string++;

    if (*string != '~')
        return copy(string);

    string += 1;

    if (!*string || *string == '/') {
        pw = getpwuid(getuid());
	*buf = 0;
    } else {
	k = buf;
	while ((c = *string) && c != '/')
		*k++ = c, string++;
	*k = 0;
	pw = getpwnam(buf);
    }

    if (pw) {
	strcpy(buf, pw->pw_dir);
	if (*string)
	    strcat(buf, string);
    } else
	return NULL;

    return copy(buf);

#else
    return copy(string);
#endif
}
