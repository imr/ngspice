/*       mystring.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt
 *  Free software under the terms of the GNU Lesser General Public License
 * $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <memory.h>
#include <math.h>
#include <stdarg.h>

#include "config.h"
#ifdef HAS_WINDOWS
#include "wstdio.h"
#endif

#include "general.h"

#include "../error.h" /* controlled_exit() */

/***** primitive input-output ***/

bool
ci_prefix (register char *p, register char *s)
{
    while (*p)
    {
        if ((isupper (*p) ? tolower (*p) : *p) !=
                (isupper (*s) ? tolower (*s) : *s))
            return (0);
        p++;
        s++;
    }
    return (1);
}

void
wc (char c)
{
    fputc (c, stdout);
}

void
wln (void)
{
    fputc ('\n', stdout);
}

void
ws (char *s)
{
    fputs(s, stdout);
}

void
wi (long i)
{
    SPICE_DSTRING s ;
    spice_dstring_init(&s) ;
    nadd (&s, i);
    ws ( spice_dstring_value(&s)) ;
    spice_dstring_free(&s) ;
}

void
rs ( SPICE_DSTRINGPTR dstr_p)
{
    /* basic line input, limit= 80 chars */
    char c;

    spice_dstring_reinit(dstr_p) ;
    do
    {
        c = fgetc (stdin);
        cadd (dstr_p, c);
    }
    while (!((c == '\r') || (c == '\n')));
}

char
rc (void)
{
    int ls;
    char val ;
    char *s_p ;
    SPICE_DSTRING dstr ;

    spice_dstring_init(&dstr) ;
    rs (&dstr);
    ls = spice_dstring_length (&dstr);
    if (ls > 0)
    {
        s_p = spice_dstring_value(&dstr) ;
        val = s_p[ls - 1] ;
    }
    else
    {
        val = 0 ;
    }
    spice_dstring_free(&dstr) ;
    return val ;
}

/*******  Strings ************
 *  are 0-terminated char arrays with a 2-byte trailer: max length.
 *  the string mini-library is "overflow-safe" under these conditions:
 *    use Str(n,s) macro: define and initialize a string s of maxlen n<255
 *    use sini() to initialize empty strings;  sfix() for non-empty ones.
 *    the Sini() macro does automatic sizing, for automatic char arrays
 *    to allocate a string on the heap, use newstring(n).
 *    use maxlen() and length() to retrieve string max and actual length
 *    use: cadd, cins, sadd, sins, scopy, pscopy to manipulate them
 *    never put '\x0' characters inside strings !
 *
 *    the 'killer idea' is the following:
 *    on string overflow and/or on heap allocation failure, a program
 *    MUST die.   Now we only die on a heap failure as with dynamic
 *    string we cannot have a string overflow.
 */

void
sfix ( SPICE_DSTRINGPTR dstr_p, int len)
/* suppose s is allocated and filled with non-zero stuff */
{
    /* This function will now eliminate the max field.   The length of
     * the string is going to be i-1 and a null is going to be added
     * at the ith position to be compatible with old codel.  Also no
     * null characters will be present in the string leading up to the
     * NULL so this will make it a valid string. */
    int j;
    char *s ;

    spice_dstring_setlength( dstr_p, len ) ;
    s = spice_dstring_value( dstr_p ) ;
    for (j = 0; j < len; j++)	/* eliminate null characters ! */
        if (s[j] == 0)
            s[j] = 1;

}

int
length (char *s)
{
    return (int) strlen(s);
}

/* -----------------------------------------------------------------
 * Function: add string t to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
bool
sadd ( SPICE_DSTRINGPTR dstr_p, char *t)
{
    spice_dstring_append( dstr_p, t, -1 ) ;
    return 1 ;
}

/* -----------------------------------------------------------------
 * Function: add character c to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
bool
cadd ( SPICE_DSTRINGPTR dstr_p, char c)
{
    char tmp_str[2] ;
    tmp_str[0] = c ;
    tmp_str[1] = 0 ;
    spice_dstring_append( dstr_p, tmp_str, -1 ) ;
    return 1 ;
}

/* -----------------------------------------------------------------
 * Function: insert character c at front of dynamic string dstr_p
 * ----------------------------------------------------------------- */
bool
cins ( SPICE_DSTRINGPTR dstr_p, char c)
{
    int i ;
    int ls ;
    char *s_p ;

    ls = spice_dstring_length(dstr_p) ;
    spice_dstring_setlength(dstr_p,ls+2) ; /* make sure we have space for char + EOS */
    s_p = spice_dstring_value(dstr_p) ;
    for (i = ls + 1; i >= 0; i--)
        s_p[i + 1] = s_p[i];
    s_p[0] = c;
    return 1 ;
}

/* -----------------------------------------------------------------
 * Function: insert string t at front of dynamic string dstr_p
 * ----------------------------------------------------------------- */
bool
sins ( SPICE_DSTRINGPTR dstr_p, char *t)
{
    int i ;
    int ls ;
    int lt ;
    char *s_p ;

    ls = spice_dstring_length(dstr_p) ;
    lt = length (t) ;
    spice_dstring_setlength(dstr_p,ls+lt+1) ; /* make sure we have space for string + EOS */
    s_p = spice_dstring_value(dstr_p) ;
    for (i = ls + 1; i >= 0; i--)
        s_p[i + lt] = s_p[i];

    for (i = 0; i < lt; i++)
        s_p[i] = t[i];
    return 1 ;

}

int
cpos (char c, char *s)
/* return position of c in s, or 0 if not found.
 * BUG, Pascal inherited: first char is at 1, not 0 !
 * No longer!  Now position is C-based to make life easier.
 */
{
    int i = 0;
    while ((s[i] != c) && (s[i] != 0))
        i++;

    if (s[i] == c)
        return i ;
    else
        return -1 ;
}

char
upcase (char c)
{
    if ((c >= 'a') && (c <= 'z'))
        return c + 'A' - 'a';
    else
        return c;
}

/* -----------------------------------------------------------------
 * Create copy of the dynamic string.  Dynamic strings are always NULL
 * terminated.
 * ----------------------------------------------------------------- */
bool
scopyd(SPICE_DSTRINGPTR s, SPICE_DSTRINGPTR t)	/* returns success flag */
{
    spice_dstring_reinit( s ) ;
    spice_dstring_append( s, spice_dstring_value(t), -1 ) ;
    return 1 ; /* Dstrings expand to any length */
}

/* -----------------------------------------------------------------
 * Create copy of the string in the dynamic string.  Dynamic strings
 * are always NULLterminated.
 * ----------------------------------------------------------------- */
bool
scopys(SPICE_DSTRINGPTR s, char *t)	/* returns success flag */
{
    spice_dstring_reinit( s ) ;
    spice_dstring_append( s, t, -1 ) ;
    return 1 ; /* Dstrings expand to any length */
}

/* -----------------------------------------------------------------
 * Create an upper case copy of a string and store it in a dynamic string.
 * Dynamic strings are always NULL * terminated.
 * ----------------------------------------------------------------- */
bool
scopy_up (SPICE_DSTRINGPTR dstr_p, char *str)	/* returns success flag */
{
    char up[2] ;			/* short string */
    char *ptr ;			/* position in string */

    spice_dstring_reinit( dstr_p ) ;
    up[1] = 0 ;
    for( ptr = str ; ptr && *ptr ; ptr++ )
    {
        up[0] = upcase ( *ptr );
        spice_dstring_append( dstr_p, up, 1 ) ;
    }
    return 1 ; /* Dstrings expand to any length */
}

/* -----------------------------------------------------------------
 * Create a lower case copy of a string and store it in a dynamic string.
 * Dynamic strings are always NULL * terminated.
 * ----------------------------------------------------------------- */
bool
scopy_lower (SPICE_DSTRINGPTR dstr_p, char *str)	/* returns success flag */
{
    char low[2] ;			/* short string */
    char *ptr ;			/* position in string */

    spice_dstring_reinit( dstr_p ) ;
    low[1] = 0 ;
    for( ptr = str ; ptr && *ptr ; ptr++ )
    {
        low[0] = lowcase ( *ptr );
        spice_dstring_append( dstr_p, low, 1 ) ;
    }
    return 1 ; /* Dstrings expand to any length */
}

bool
ccopy ( SPICE_DSTRINGPTR dstr_p, char c)	/* returns success flag */
{
    char *s_p ;			/* current string */

    sfix ( dstr_p, 1);
    s_p = spice_dstring_value(dstr_p) ;
    s_p[0] = c ;
    return 1 ;
}

char *
pscopy (SPICE_DSTRINGPTR dstr_p, char *t, int start, int leng)
/* partial string copy, with C-based start - Because we now have a 0 based
 * start and string may copy outselves, we may need to restore the first
 * character of the original dstring because resetting string will wipe
 * out first character. */
{
    int i;					/* counter */
    int stop ;					/* stop value */
    char *s_p ;					/* value of dynamic string */

    stop = length(t) ;

    if (start < stop)  				/* nothing! */
    {
        if ((start + leng - 1) > stop)
        {
//      leng = stop - start + 1;
            leng = stop - start ;
        }
        _spice_dstring_setlength(dstr_p,leng) ;
        s_p = spice_dstring_value(dstr_p) ;
        for (i = 0; i < leng; i++)
            s_p[i] = t[start + i];
        s_p[leng] = '\0' ;
    }
    else
    {
        s_p = spice_dstring_reinit(dstr_p) ;
    }
    return s_p ;
}

char *
pscopy_up (SPICE_DSTRINGPTR dstr_p, char *t, int start, int leng)
/* partial string copy to upper case, with C convention for start. */
{
    int i;					/* counter */
    int stop ;					/* stop value */
    char *s_p ;					/* value of dynamic string */

    stop = length(t) ;

    if (start < stop)  				/* nothing! */
    {
        if ((start + leng - 1) > stop)
        {
//      leng = stop - start + 1;
            leng = stop - start ;
        }
        _spice_dstring_setlength(dstr_p,leng) ;
        s_p = spice_dstring_value(dstr_p) ;
        for (i = 0; i < leng; i++)
            s_p[i] = upcase ( t[start + i] ) ;
        s_p[leng] = '\0' ;
    }
    else
    {
        s_p = spice_dstring_reinit(dstr_p) ;
    }
    return s_p ;
}

int
ord (char c)
{
    return (c & 0xff);
}				/* strip high byte */

int
pred (int i)
{
    return (--i);
}

int
succ (int i)
{
    return (++i);
}

bool
nadd ( SPICE_DSTRINGPTR dstr_p, long n)
/* append a decimal integer to a string */
{
    int d[25];
    int j, k ;
    char sg;			/* the sign */
    char load_str[2] ;		/* used to load dstring */
    k = 0;

    if (n < 0)
    {
        n = -n;
        sg = '-';
    }
    else
        sg = '+';

    while (n > 0)
    {
        d[k] = n % 10;
        k++;
        n = n / 10;
    }

    if (k == 0)
        cadd (dstr_p, '0');
    else
    {
        load_str[1] = 0 ;
        if (sg == '-')
        {
            load_str[0] = sg ;
            spice_dstring_append( dstr_p, load_str, 1 ) ;
        }
        for (j = k - 1; j >= 0; j--)
        {
            load_str[0] = d[j] + '0';
            spice_dstring_append( dstr_p, load_str, 1 ) ;
        }
    }

    return 1 ;
}

bool
naddll (SPICE_DSTRINGPTR dstr_p, long long n)
/* append a decimal integer (but a long long) to a string */
{
    int d[25];
    int j, k ;
    char sg;			/* the sign */
    char load_str[2] ;		/* used to load dstring */
    k = 0;

    if (n < 0)
    {
        n = -n;
        sg = '-';
    }
    else
        sg = '+';

    while (n > 0)
    {
        d[k] = n % 10;
        k++;
        n = n / 10;
    }

    if (k == 0)
        cadd (dstr_p, '0');
    else
    {
        load_str[1] = 0 ;
        if (sg == '-')
        {
            load_str[0] = sg ;
            spice_dstring_append( dstr_p, load_str, 1 ) ;
        }
        for (j = k - 1; j >= 0; j--)
        {
            load_str[0] = d[j] + '0';
            spice_dstring_append( dstr_p, load_str, 1 ) ;
        }
    }

    return 1 ;
}

void
stri (long n, SPICE_DSTRINGPTR dstr_p)
/* convert integer to string */
{
    spice_dstring_reinit( dstr_p ) ;
    nadd (dstr_p, n) ;
}

void
rawcopy (void *a, void *b, int la, int lb)
/* dirty binary copy */
{
    int j, n;
    if (lb < la)
        n = lb;
    else
        n = la;

    for (j = 0; j < n; j++)
        ((char *) a)[j] = ((char *) b)[j];
}

int
scompare (char *a, char *b)
{
    unsigned short j = 0;
    int k = 0;
    while ((a[j] == b[j]) && (a[j] != 0) && (b[j] != 0))
        j++;

    if (a[j] < b[j])
        k = -1;
    else if (a[j] > b[j])
        k = 1;

    return k;
}

bool
steq (char *a, char *b)		/* string a==b test */
{
    return strcmp (a, b) == 0;
}

bool
stne (char *s, char *t)
{
    return strcmp (s, t) != 0;
}

int
hi (long w)
{
    return (w & 0xff00) >> 8;
}

int
lo (long w)
{
    return (w & 0xff);
}

char
lowcase (char c)
{
    if ((c >= 'A') && (c <= 'Z'))
        return (char) (c - 'A' + 'a');
    else
        return c;
}

bool
alfa (char c)
{
    return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')) || c == '_'
           || c == '[' || c == ']';
}

bool
num (char c)
{
    return (c >= '0') && (c <= '9');
}

bool
alfanum (char c)
{
    return alfa (c) || ((c >= '0') && (c <= '9'));
}

int
freadstr (FILE * f, SPICE_DSTRINGPTR dstr_p)
/* read a line from a file.
   was BUG: long lines truncated without warning, ctrl chars are dumped.
   Bug no more as we can only run out of memory.  Removed max argument.
*/
{
    char c;
    char str_load[2] ;
    int len = 0 ;

    str_load[0] = 0 ;
    str_load[1] = 0 ;
    spice_dstring_reinit(dstr_p) ;
    do
    {
        c = fgetc (f);		/*  tab is the only control char accepted */
        if (((c >= ' ') || (c < 0) || (c == '\t')))
        {
            str_load[0] = c;
            spice_dstring_append( dstr_p, str_load, 1 ) ;
        }
    }
    while (!(feof (f) || (c == '\n')));


    return len ;
}

char
freadc (FILE * f)
{
    return fgetc (f);
}

long
freadi (FILE * f)
/* reads next integer, but returns 0 if none found. */
{
    long z = 0;
    bool minus = 0;
    char c;

    do
    {
        c = fgetc (f);
    }
    while (!(feof (f) || !((c > 0) && (c <= ' '))));	/* skip space */

    if (c == '-')
    {
        minus = 1;
        c = fgetc (f);
    }

    while (num (c))
    {
        z = 10 * z + c - '0';
        c = fgetc (f);
    }

    ungetc (c, f);		/* re-push character lookahead */

    if (minus)
        z = -z;

    return z;
}

char *
stupcase (char *s)
{
    int i = 0;

    while (s[i] != 0)
    {
        s[i] = upcase (s[i]);
        i++;
    }

    return s;
}

/*****  pointer tricks: app won't use naked malloc(), free() ****/

void
dispose (void *p)
{
    if (p != NULL)
        free (p);
}

void *
new (size_t sz)
{
    void *p = tmalloc (sz);
    if (p == NULL)
    {			/* fatal error */
        ws (" new() failure. Program halted.\n");
        controlled_exit(EXIT_FAILURE);
    }
    return p;
}

/***** elementary math *******/

double
sqr (double x)
{
    return x * x;
}

double
absf (double x)
{
    if (x < 0.0)
        return -x;
    else
        return x;
}

long
absi (long i)
{
    if (i >= 0)
        return (i);
    else
        return (-i);
}

void
strif (long i, int f, SPICE_DSTRINGPTR dstr_p)
/* formatting like str(i:f,s) in Turbo Pascal */
{
    int j, k ;
    char cs;
    char t[32];
    char load_str[2] ;			/* load dstring */
    k = 0;

    if (i < 0)
    {
        i = -i;
        cs = '-';
    }
    else
    {
        cs = ' ';
    }

    while (i > 0)
    {
        j = (int) (i % 10);
        i = (long) (i / 10);
        t[k] = (char)('0' + j);
        k++;
    }

    if (k == 0)
    {
        t[k] = '0';
        k++;
    }

    if (cs == '-')
        t[k] = cs;
    else
        k--;

    /* now the string  is in 0...k in reverse order */
    for (j = 1; j <= k; j++)
        t[k + j] = t[k - j];	/* mirror image */

    t[2 * k + 1] = 0;		/* null termination */
    load_str[1] = 0 ;		/* not really needed */
    spice_dstring_reinit(dstr_p) ;

    if ((f > k) && (f < 40))
    {
        /* reasonable format */
        for (j = k + 2; j <= f; j++)
        {
            load_str[0] = ' ';
            spice_dstring_append( dstr_p, load_str, 1 ) ;
        }
    }

    for (j = 0; j <= k + 1; j++)
    {
        load_str[0] = t[k + j];	/* shift t down */
        spice_dstring_append( dstr_p, load_str, 1 ) ;
    }
}

bool
odd (long x)
{
    return (x & 1);
}

static bool
match (char *s, char *t, int n, int tstart, bool testcase)
{
    /* returns 0 if ( tstart is out of range. But n may be 0 ? */
    /* 1 if s matches t[tstart...tstart+n]  */
    int i, j, lt;
    bool ok;
    char a, b;
    i = 0;
    j = tstart;
    lt = length (t);
    ok = (tstart < lt);

    while (ok && (i < n))
    {
        a = s[i];
        b = t[j];
        if (!testcase)
        {
            a = upcase (a);
            b = upcase (b);
        }
        ok = (j < lt) && (a == b);
        i++;
        j++;
    }
    return ok;
}

int
posi (char *sub, char *s, int opt)
/* find position of substring in s */
{
    /* opt=0: like Turbo Pascal */
    /* opt=1: like Turbo Pascal Pos, but case insensitive */
    /* opt=2: position in space separated wordlist for scanners */
    int a, b, k, j;
    bool ok, tstcase;
    SPICE_DSTRING tstr ;

    ok = 0;
    spice_dstring_init(&tstr) ;
    tstcase = (opt == 0);

    if (opt <= 1)
        scopys (&tstr, sub);
    else
    {
        cadd (&tstr, ' ');
        sadd (&tstr, sub);
        cadd (&tstr, ' ');
    }

    a = spice_dstring_length(&tstr) ;
    b = (int) (length (s) - a);
    k = 0;
    j = 1;

    if (a > 0)			/* ;} else { return 0 */
        while ((k <= b) && (!ok))
        {
            ok = match ( spice_dstring_value(&tstr), s, a, k, tstcase);	/* we must start at k=0 ! */
            k++;
            if (s[k] == ' ')
                j++; /* word counter */ ;
        }

    if (opt == 2)
        k = j;

    if (ok)
        return k;
    else
        return 0;

}

int
spos_(char *sub, char *s)
/* equivalent to Turbo Pascal pos().
   BUG: counts 1 ... length(s), not from 0 like C
*/
{
    char *ptr;

    if ((ptr = strstr (s, sub)) != NULL)
        return (int) (strlen (s) - strlen (ptr));
    else
        return -1 ;

}

void
strf (double x, int f1, int f2, SPICE_DSTRINGPTR dstr_p)
/* e-format if f2<0, else f2 digits after the point, total width=f1 */
/* if f1=0, also e-format with f2 digits */
{
    /* ngspice default f1=17, f2=10 */
    int dlen ;			/* length of digits */
    char *dbuf_p ;		/* beginning of sprintf buffer */
    SPICE_DSTRING fmt ;		/* format string */
    SPICE_DSTRING dformat ;	/* format float */

    spice_dstring_init(&fmt) ;
    spice_dstring_init(&dformat) ;
    cadd (&fmt, '%');
    if (f1 > 0)
    {
        nadd (&fmt, f1);		/* f1 is the total width */
        if (f2 < 0)
            sadd (&fmt, "lE");	/* exponent format */
        else
        {
            cadd (&fmt, '.');
            nadd (&fmt, f2);
            sadd (&fmt, "g");
        }
    }
    else
    {
        cadd (&fmt, '.');
        nadd (&fmt, absi (f2 - 6));	/* note the 6 surplus positions */
        cadd (&fmt, 'e');
    }

    dlen = 2 * (f1 + f2) + 1 ;   /* be conservative */
    dbuf_p = spice_dstring_setlength(&dformat, dlen)  ;
    sprintf (dbuf_p, spice_dstring_value(&fmt), x);
    scopys( dstr_p, dbuf_p ) ;

    spice_dstring_free(&fmt) ;
    spice_dstring_free(&dformat) ;
}

double
rval (char *s, int *err)
/* returns err=0 if ok, else length of partial string ? */
{
    double r = 0.0;
    int n = sscanf (s, "%lG", &r);

    if (n == 1)
        (*err) = 0;
    else
        (*err) = 1;

    return r;
}

long
ival (char *s, int *err)
/* value of s as integer string.  error code err= 0 if Ok */
{
    int k = 0, digit = 0, ls;
    long z = 0;
    bool minus = 0, ok = 1;
    char c;
    ls = length (s);

    do
    {
        c = s[k];
        k++;
    }
    while (!((k >= ls) || !((c > 0) && (c <= ' '))));	/* skip space */

    if (c == '-')
    {
        minus = 1;
        c = s[k];
        k++;
    }

    while (num (c))
    {
        z = 10 * z + c - '0';
        c = s[k];
        k++;
        digit++;
    }

    if (minus)
        z = -z;

    ok = (digit > 0) && (c == 0);	/* successful end of string */

    if (ok)
        (*err) = 0;
    else
        (*err) = k;			/* one beyond error position */

    return z;
}

#ifndef HAVE_LIBM

long
np_round (double x)
/* using <math.h>, it would be simpler: floor(x+0.5), see below */
{
    double u;
    long z;
    int n;
//  Str (40, s);
    SPICE_DSTRING s ;
    spice_dstring_init(&s) ;
    u = 2e9;
    if (x > u)
        x = u;
    else if (x < -u)
        x = -u;

    n = sprintf (s, "%-12.0f", x);
    s[n] = 0;
    sscanf (s, "%ld", &z);
    return z;
}

long
np_trunc (double x)
{
    long n = np_round (x);
    if ((n > x) && (x >= 0.0))
        n--;
    else if ((n < x) && (x < 0.0))
        n++;

    return n;
}

double
frac (double x)
{
    return x - np_trunc (x);
}

#else /* use floor() and ceil() */

long
np_round (double r)
{
    return (long) floor (r + 0.5);
}

long
np_trunc (double r)
{
    if (r >= 0.0)
        return (long) floor (r);
    else
        return (long) ceil (r);
}

double
frac (double x)
{
    if (x >= 0.0)
        return (x - floor (x));
    else
        return (x - ceil (x));
}

#endif
