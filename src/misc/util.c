/*************
 * Various utility functions.
 * 2002 R. Oktas, <roktas@omu.edu.tr>
 ************/

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "ngspice.h"
#include "util.h"


/* **************************************************************** */
/*                                                                  */
/*                     Stuff for Filename Handling                  */
/*                                                                  */
/* **************************************************************** */

/* Canonicalize PATH, and return a new path.  The new path differs from PATH
   in that:
       Multple `/'s are collapsed to a single `/'.
       Leading `./'s and trailing `/.'s are removed.
       Trailing `/'s are removed.
       Non-leading `../'s and trailing `..'s are handled by removing
       portions of the path.

    Stolen from Bash source (slightly modified).
    Credit goes to Chet Ramey, et al. -- ro */

char *
canonicalize_pathname(char *path)
{
    int i, start;
    char stub_char;
    char *result;

    /* The result cannot be larger than the input PATH. */
    result = copy(path);

    stub_char = (*path == '/') ? '/' : '.';

    /* Walk along RESULT looking for things to compact. */
    i = 0;
    while (1) {
       if (!result[i])
           break;

       while (result[i] && result[i] != '/')
           i++;

       start = i++;

       /* If we didn't find any slashes, then there is nothing left to do. */
       if (!result[start])
           break;

       /* Handle multiple `/'s in a row. */
       while (result[i] == '/')
           i++;

#if !defined (apollo)
       if ((start + 1) != i)
#else
       if ((start + 1) != i && (start != 0 || i != 2))
#endif /* apollo */
       {
           strcpy (result + start + 1, result + i);
           i = start + 1;
       }

#if 0
       /* Handle backslash-quoted `/'. */
       if (start > 0 && result[start - 1] == '\\')
           continue;
#endif

       /* Check for trailing `/'. */
       if (start && !result[i]) {
           zero_last:
           result[--i] = '\0';
           break;
       }

       /* Check for `../', `./' or trailing `.' by itself. */
       if (result[i] == '.') {
           /* Handle trailing `.' by itself. */
           if (!result[i + 1])
               goto zero_last;

           /* Handle `./'. */
           if (result[i + 1] == '/') {
               strcpy(result + i, result + i + 1);
               i = (start < 0) ? 0 : start;
               continue;
           }

           /* Handle `../' or trailing `..' by itself. */
           if (result[i + 1] == '.' &&
              (result[i + 2] == '/' || !result[i + 2])) {
               while (--start > -1 && result[start] != '/');
               strcpy(result + start + 1, result + i + 2);
               i = (start < 0) ? 0 : start;
               continue;
           }
       }
    }

    if (!*result) {
       *result = stub_char;
       result[1] = '\0';
    }
    return (result);
}


/* Turn STRING (a pathname) into an absolute pathname, assuming that
   DOT_PATH contains the symbolic location of `.'.  This always
   returns a new string, even if STRING was an absolute pathname to
   begin with.

   Stolen from Bash source (slightly modified).
   Credit goes to Chet Ramey, et al. -- ro */

char * absolute_pathname(char *string, char *dot_path)
{
  char *result;
  int result_len;

  if (!dot_path || *string == '/')
      result = copy(string);
  else {
      if (dot_path && dot_path[0]) {
         result = tmalloc(2 + strlen(dot_path) + strlen(string));
         strcpy(result, dot_path);
         result_len = strlen(result);
         if (result[result_len - 1] != '/') {
             result[result_len++] = '/';
             result[result_len] = '\0';
         }
      } else {
         result = tmalloc(3 + strlen (string));
         result[0] = '.'; result[1] = '/'; result[2] = '\0';
         result_len = 2;
      }

      strcpy(result + result_len, string);
  }

  return (result);
}
