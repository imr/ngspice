/*************
 * Various utility functions.
 * 2002 R. Oktas, <roktas@omu.edu.tr>
 ************/

#include "ngspice/ngspice.h"
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

    stub_char = '.';
    if(*path == '/')
        stub_char = '/';

    /* Walk along RESULT looking for things to compact. */
    i = 0;
    for (;;) {
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
               while (--start > -1 && result[start] != '/')
                   ;
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
  size_t result_len;

  if (!dot_path || *string == '/')
      result = copy(string);
  else {
      if (dot_path && dot_path[0]) {
         result = TMALLOC(char, 2 + strlen(dot_path) + strlen(string));
         strcpy(result, dot_path);
         result_len = strlen(result);
         if (result[result_len - 1] != '/') {
             result[result_len++] = '/';
             result[result_len] = '\0';
         }
      } else {
         result = TMALLOC(char, 3 + strlen (string));
         result[0] = '.'; result[1] = '/'; result[2] = '\0';
         result_len = 2;
      }

      strcpy(result + result_len, string);
  }

  return (result);
}


/*

char *
basename(const char *name)
{
    char *base;
    char *p;
    static char *tmp = NULL;
    int len; 

    if (tmp) {
        free(tmp);
        tmp = NULL;
    }

    if (!name || !strcmp(name, ""))
        return "";

    if (!strcmp(name, "/"))
        return "/";

    len = strlen(name);
    if (name[len - 1] == '/') {
        // ditch the trailing '/'
        p = tmp = TMALLOC(char, len);
        strncpy(p, name, len - 1); 
    } else {
        p = (char *) name;
    }

    for (base = p; *p; p++) 
        if (*p == '/') 
            base = p + 1;
    
    return base;
}
*/


#if defined(HAS_WINDOWS) || defined(_MSC_VER) || defined(__MINGW32__)

char *
ngdirname(const char *name)
{
    char *ret;
    const char *end = NULL;
    int start = 0;

    if(name && ((name[0] >= 'a' && name[0] <= 'z') ||
                (name[0] >= 'A' && name[0] <= 'Z')) && name[1] == ':')
        start = 2;

    if(name) {
        const char *p = name + start;
        for(; *p; p++)
            if(*p == '/' || *p == '\\')
                end = p;
    }

    if(end && end == name + start)
        end++;

    if(end)
        ret = copy_substring(name, end);
    else {
        char *p = TMALLOC(char, 4);
        ret = p;
        if(start) {
            *p++ = name[0];
            *p++ = name[1];
        }
        *p++ = '.';
        *p++ = '\0';
    }

    return ret;
}

#else

char *
ngdirname(const char *name)
{
    char *ret;
    const char *end;

    end = name ? strrchr(name, '/') : NULL;

    if(end && end == name)
        end++;

    if(end)
        ret = copy_substring(name, end);
    else
        ret = copy(".");

    return ret;
}

#endif
