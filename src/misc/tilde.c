/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2002 R. Oktas, <roktas@omu.edu.tr>
**********/

#include "ngspice/defines.h"
#include "ngspice/ngspice.h"
#include "ngspice/stringskip.h"
#include "tilde.h"

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef _WIN32
#undef BOOLEAN
#include <windows.h> /* win32 functions */
#include "shlobj.h"  /* SHGetFolderPath */
#endif

static inline int copy_home_to_buf(size_t n_byte_dst, char **p_dst,
        const char *src);

/* XXX To prevent a name collision with `readline's `tilde_expand',
   the original name: `tilde_expand' has changed to `tildexpand'. This
   situation naturally brings to mind that `tilde_expand' could be used
   directly from `readline' (since it will already be included if we
   wish to activate the `readline' support). Following implementation of
   'tilde expanding' has some problems which constitutes another good
   reason why it should be replaced: eg. it returns NULL which should
   not behave this way, IMHO.  Anyway... Don't care for the moment, may
   be in the future. -- ro */
char *tildexpand(const char *string)
{
    /* If no string passed, return NULL */
    if (!string) {
        return NULL;
    }

    string = skip_ws(string); /* step past leading whitespace */

    /* If the string does not begin with a tilde, there is no ~ to expand */
    if (*string != '~') {
        return copy(string);
    }

    ++string; /* step past tilde */

    /* Test for home of current user */
    if (*string == '\0' || *string ==  DIR_TERM) {
        char *sz_home;
        const int n_char_home = get_local_home(0, &sz_home);
        if (n_char_home < 0) {
            return copy(string); /* Strip the ~ and return the rest */
        }
        const size_t n_char_rest = strlen(string);
        sz_home = TREALLOC(char, sz_home, (size_t) n_char_home + n_char_rest + 1);
        strcpy(sz_home + n_char_home, string);
        return sz_home;
    }

#ifdef HAVE_PWD_H
    /* ~bob -- Get name of user and find home for that user */
    {
        char buf_fixed[100];
        char *buf = buf_fixed;
        const char * const usr_start = string;
        char c;
        while ((c = *string) && c != '/') {
            string++;
        }
        const char * const usr_end = string;
        const size_t n_char_usr = (size_t) (usr_end - usr_start);
        const size_t n_byte_usr = n_char_usr + 1;
        if (n_byte_usr > sizeof buf_fixed) {
            buf = TMALLOC(char, n_byte_usr);
        }
        (void) memcpy(buf, usr_start, n_char_usr);
        buf[n_char_usr] = '\0';


        char *sz_home;
        const int n_char_home = get_usr_home(buf, 0, &sz_home);
        if (buf != buf_fixed) { /* free allocated buffer for user name */
            txfree(buf);
        }
        if (n_char_home < 0) {
            return copy(usr_start); /* Strip the ~ and return the rest */
        }
        const size_t n_char_rest = strlen(string);
        sz_home = TREALLOC(char, sz_home, (size_t) n_char_home + n_char_rest + 1);
        strcpy(sz_home + n_char_home, string);
        return sz_home;
    }

#else
    /* ~abc is meaningless */
    return copy(string); /* Again strip ~ and return rest */
#endif
} /* end of function tildexpand */




/* Get value of "HOME" for the current user and copy to *p_buf if
 * the value is less than n_byte_buf characters long. Otherwise
 * allocate a buffer of the minimum required size.
 *
 * Return values
 * >0: Number of characters copied to *p_buf, excluding the trailing null
 * -1: A value for HOME could not be obtained.
 *
 * Remarks:
 * This function does not free any allocation at *p_buf, allowing a
 * fixed buffer to be passed to it.
 */
int get_local_home(size_t n_byte_buf, char **p_buf)
{
    char *sz_home = (char *) NULL;
#ifdef _WIN32
    char buf_sh_path[MAX_PATH];
#endif

    do {
        /* First thing to try is an environment variable HOME */
        if ((sz_home = getenv("HOME")) != (char *) NULL) {
            break;
        }

#if defined(_WIN32)
        /* If Windows, try an env var USERPROFILE next */
        if ((sz_home = getenv("USERPROFILE")) != (char *) NULL) {
            break;
        }

        /* For Windows, the folder path CSIDL_PERSONAL is tried next */
        if (SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0,
                buf_sh_path))) {
            sz_home = buf_sh_path;
            break;
        }
#elif defined(HAVE_PWD_H) /* _WIN32 and HAVE_PWD_H are mutually exclusive */
        /* Get home information for the current user */
        {
            struct passwd *pw;
            pw = getpwuid(getuid());
            if (pw) {
                sz_home = pw->pw_dir;
            }
        }
#endif
    } while (0);

    if (sz_home == (char *) NULL) { /* did not find a HOME value */
        return -1;
    }

    /* Copy home value to buffer */
    return copy_home_to_buf(n_byte_buf, p_buf, sz_home);
} /* end of function get_local_home */



#ifdef HAVE_PWD_H
/* Get value of "HOME" for usr and copy to *p_buf if
 * the value is less than n_byte_buf characters long. Otherwise
 * allocate a buffer of the minimum required size.
 *
 * Return values
 * >0: Number of characters copied to *pp_buf, excluding the trailing null
 * -1: A value for HOME could not be obtained.
 *
 * Remarks:
 * This function does not free any allocation at *p_buf, allowing a
 * fixed buffer to be passed to it.
 */
int get_usr_home(const char *usr, size_t n_byte_buf, char **p_buf)
{
    struct passwd * const pw = getpwnam(usr);
    if (pw) {
        /* Copy home value to buffer */
        return copy_home_to_buf(n_byte_buf, p_buf, pw->pw_dir);
    }

    return -1;
} /* end of function get_usr_home */
#endif /* HAVE_PWD_H */



/* This function copies home value src to the buffer at *p_dst, allocating
 * a larger buffer if required.
 *
 * Parameters
 * n_byte_dst: Size of supplied destination buffer
 * p_dst: Address containing address of supplied buffer on input. May be
 *      given the address of a larger buffer allocation if the input buffer
 *      is too small.
 * src: Address of HOME value
 *
 * Return values
 * number of characters copied excluding terminating null
 *
 * Remarks:
 * This function does not free any allocation at *p_dst, allowing a
 * fixed buffer to be passed to it.
 */
static inline int copy_home_to_buf(size_t n_byte_dst, char **p_dst,
        const char *src)
{
    const size_t n_char_src = strlen(src); /* Size of HOME value */
    const size_t n_byte_src = n_char_src + 1;

    /* Allocate dst if input buffer too small */
    if (n_byte_src > n_byte_dst) { /* too big */
        *p_dst = TMALLOC(char, n_byte_src);
    }

    (void) memcpy(*p_dst, src, n_byte_src);

    return (int) n_char_src;
} /* end of function copy_home_to_buf */



