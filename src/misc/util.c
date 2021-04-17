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
        tfree(tmp);
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


#if defined(HAS_WINGUI) || defined(_MSC_VER) || defined(__MINGW32__)
/* This function returns the path portion of name or "." if it is NULL.
 * The returned string is a new allocation that must be freed by the
 * caller */
char *ngdirname(const char *name)
{
    /* If name not given, return "." */
    if (name == (char *) NULL) {
        return dup_string(".", 1);
    }

    /* Offset to start of path name after any drive letter present */
    const int start = (((name[0] >= 'a' && name[0] <= 'z') ||
            (name[0] >= 'A' && name[0] <= 'Z')) && name[1] == ':') ? 2 : 0;

    /* Find last dir separator, if any, and return the input directory up to
     * that separator or including it if it is the 1st char after any drive
     *specification. */
    {
        const char *p0 = name + start; /* 1st char past drive */
        const char *p;
        for (p = p0 + strlen(name + start) - 1; p >= p0; --p) {
            const char ch_cur = *p;
            if (ch_cur == '/' || ch_cur == '\\') { /* at last dir sep */
                /* Stop copy at last dir sep or right after if
                 * it is the first char after any drive spec.
                 * In the second case return "[drive]<dir sep>" */
                const char * const end = p + (p == p0);
                return copy_substring(name, end);
            }
        }
    }

    /* No directory separator found so return "[drive]." */
    {
        char * const ret = TMALLOC(char, 2 + start);
        char *p = ret;
        if (start) { /* Add drive letter if found */
            *p++ = name[0];
            *p++ = name[1];
        }
        *p++ = '.';
        *p = '\0';
        return ret;
    }
} /* end of function ngdirname */

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

/* Replacement for fopen, when using wide chars (utf-16) */
#ifndef EXT_ASC
#if defined(__MINGW32__) || defined(_MSC_VER)
#undef BOOLEAN
#include <windows.h>
FILE *
newfopen(const char *fn, const char* md)
{
    FILE* fp;
    if (fn == NULL)
        return NULL;
    wchar_t wfn[BSIZE_SP];
    wchar_t wmd[16];
    MultiByteToWideChar(CP_UTF8, 0, md, -1, wmd, 15);
    if (MultiByteToWideChar(CP_UTF8, 0, fn, -1, wfn, BSIZE_SP - 1) == 0) {
        fprintf(stderr, "UTF-8 to UTF-16 conversion failed with 0x%x\n", GetLastError());
        fprintf(stderr, "%s could not be converted\n", fn);
        return NULL;
    }
    fp = _wfopen(wfn, wmd);

/* If wide char fails, at least fopen may try the potentially ANSI encoded special characters like á */
#undef fopen
    if (fp == NULL)
        fp = fopen(fn, md);
#define fopen newfopen

    return fp;
}
#endif
#endif

