/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2002 R. Oktas, <roktas@omu.edu.tr>
**********/

#include <config.h>
#include <stdio.h> 
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#include "ngspice.h"
#include "tilde.h"

#ifdef HAVE_PWD_H
#include <pwd.h>
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

/* PN: Since readline patch is not compiled in by default the behaviour
   described above is not acceptable. I will make two different version
   of tilde expansion routine, the one choosen depends on HAVE_GNUREADLINE
   Note: since a readline replacement called libedit is under 
         development at libedit.sourceforge.net, I will switch to it
	 ASAP, since libedit is covered by BSD licnse.
         
 */  
 
#ifndef HAVE_GNUREADLINE  
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

#else  /* HAVE_GNUREADLINE */

char *
tildexpand(char *string)
{

    char buf[BSIZE_SP];
    char *result, *k, c;
    if (!string)
	return NULL;

    while (*string && isspace(*string))
	string++;

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
	    *buf = 0;
	}
    } else {
	struct passwd *pw;    
	k = buf;
	while ((c = *string) && c != '/')
		*k++ = c, string++;
	*k = 0;
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
#endif
}
#endif /* HAVE_GNUREADLINE */
