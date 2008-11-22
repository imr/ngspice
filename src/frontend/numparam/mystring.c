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

#include "config.h"
#ifdef HAS_WINDOWS
#include "wstdio.h"
#endif

#include "general.h"

#define Getmax(s,ls)  (((unsigned char)(s[ls+1])) << 8) + (unsigned char)(s[ls+2])

/***** primitive input-output ***/

int
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
  wc ('\n');
}

void
ws (char *s)
{
  int k = 0;

  while (s[k] != 0)
    {
      wc (s[k]);
      k++;
    }
}

void
wi (long i)
{
  Str (16, s);
  nadd (s, i);
  ws (s);
}

void
rs (char *s)
{				/*basic line input, limit= 80 chars */
  int max, i;
  char c;
  exit (-1);
  max = maxlen (s);
  i = 0;
  sini (s, max);
  if (max > 80)
    max = 80;

  do
    {
      c = fgetc (stdin);
      if ((i < max) && (c >= ' '))
	{
	  cadd (s, c);
	  i++;
	}
    }
  while (!((c == Cr) || (c == '\n')));
  /* return i */ ;
}

char
rc (void)
{
  int ls;
  Str (80, s);
  rs (s);
  ls = length (s);
  if (ls > 0)
    return s[ls - 1];
  else
    return 0;

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
 *    MUST die. 
 */

static void
stringbug (char *op, char *s, char *t, char c)
/* we brutally stop the program on string overflow */
{
  char rep = ' ';
  fprintf (stderr, " STRING overflow %s\n", op);
  fprintf (stderr, " Operand1: %s\n", s);
  if (t != NULL)
    fprintf (stderr, " Operand2: %s\n", t);

  if (c != 0)
    fprintf (stderr, "{%c}\n", c);

  fprintf (stderr, "Aborting...\n");
  exit (1);

/* The code below cannot be reached */
/* Remnants of old interface ?*/

  ws (" [A]bort [I]gnore ? ");
  rep = rc ();
  if (upcase (rep) == 'A')
    exit (1);
}

void
sini (char *s, int max)		/* suppose s is allocated */
{
  if (max < 1)
    max = 1;
  else if (max > Maxstr)
    max = Maxstr;

  s[0] = 0;
  s[1] = Hi (max);
  s[2] = Lo (max);
}

void
sfix (char *s, int i, int max)
/* suppose s is allocated and filled with non-zero stuff */
{
  int j;
  if (max < 1)
    max = 1;
  else if (max > Maxstr)
    max = Maxstr;

  if (i > max)
    i = max;
  else if (i < 0)
    i = 0;

  s[i] = 0;
  s[i + 1] = Hi (max);
  s[i + 2] = Lo (max);

  for (j = 0; j < i; j++)	/* eliminate null characters ! */
    if (s[j] == 0)
      s[j] = 1;

}

static void
inistring (char *s, char c, int max)
/* suppose s is allocated. empty it if c is zero ! */
{
  int i = 0;
  s[i] = c;
  if (c != 0)
    {
      i++;
      s[i] = 0;
    }

  if (max < 1)
    max = 1;
  else if (max > Maxstr)
    max = Maxstr;

  s[i + 1] = Hi (max);
  s[i + 2] = Lo (max);
}

int
length (char *s)
{
  int lg = 0;

  while (s[lg])
    lg++;

  return lg;
}

int
maxlen (char *s)
{
  int ls = length (s);

  return Getmax (s, ls);
}

unsigned char
sadd (char *s, char *t)
{
  unsigned char ok;
  int i = 0, max, ls = length (s);
  max = Getmax (s, ls);

  while ((t[i] != 0) && (ls < max))
    {
      s[ls] = t[i];
      i++;
      ls++;
    }

  s[ls] = 0;
  s[ls + 1] = Hi (max);
  s[ls + 2] = Lo (max);
  ok = (t[i] == 0);		/* end of t is reached */

  if (!ok)
    stringbug ("sadd", s, t, 0);

  return ok;
}

unsigned char
cadd (char *s, char c)
{
  int max, ls = length (s);
  unsigned char ok;
  max = Getmax (s, ls);
  ok = (ls < max);
  if (ok)
    {
      s[ls + 3] = s[ls + 2];
      s[ls + 2] = s[ls + 1];
      s[ls + 1] = 0;
      s[ls] = c;
    }
  if (!ok)
    stringbug ("cadd", s, NULL, c);

  return ok;
}

unsigned char
cins (char *s, char c)
{
  int i, max, ls = length (s);
  unsigned char ok;
  max = Getmax (s, ls);
  ok = (ls < max);

  if (ok)
    {
      for (i = ls + 2; i >= 0; i--)
	s[i + 1] = s[i];
      s[0] = c;
    }

  if (!ok)
    stringbug ("cins", s, NULL, c);

  return ok;
}

unsigned char
sins (char *s, char *t)
{
  int i, max, ls = length (s), lt = length (t);
  unsigned char ok;
  max = Getmax (s, ls);
  ok = ((ls + lt) < max);

  if (ok)
    {
      for (i = ls + 2; i >= 0; i--)
	s[i + lt] = s[i];

      for (i = 0; i < lt; i++)
	s[i] = t[i];
    }

  if (!ok)
    stringbug ("sins", s, t, 0);

  return ok;
}

int
cpos (char c, char *s)
/* return position of c in s, or 0 if not found.
 * BUG, Pascal inherited: first char is at 1, not 0 !
 */
{
  int i = 0;
  while ((s[i] != c) && (s[i] != 0))
    i++;

  if (s[i] == c)
    return (i + 1);
  else
    return 0;
}

char
upcase (char c)
{
  if ((c >= 'a') && (c <= 'z'))
    return c + 'A' - 'a';
  else
    return c;
}

unsigned char
scopy (char *s, char *t)	/* returns success flag */
{
  unsigned char ok;
  int i, max, ls = length (s);
  max = Getmax (s, ls);
  i = 0;

  while ((t[i] != 0) && (i < max))
    {
      s[i] = t[i];
      i++;
    }

  s[i] = 0;
  s[i + 1] = Hi (max);
  s[i + 2] = Lo (max);
  ok = (t[i] == 0);		/* end of t is reached */

  if (!ok)
    stringbug ("scopy", s, t, 0);

  return ok;
}

unsigned char
scopy_up (char *s, char *t)	/* returns success flag */
{
  unsigned char ok;
  int i, max, ls = length (s);
  max = Getmax (s, ls);
  i = 0;
  while ((t[i] != 0) && (i < max))
    {
      s[i] = upcase (t[i]);
      i++;
    }

  s[i] = 0;
  s[i + 1] = Hi (max);
  s[i + 2] = Lo (max);
  ok = (t[i] == 0);		/* end of t is reached */

  if (!ok)
    stringbug ("scopy_up", s, t, 0);

  return ok;
}

unsigned char
ccopy (char *s, char c)		/* returns success flag */
{
  int max, ls = length (s);
  unsigned char ok = 0;
  max = Getmax (s, ls);

  if (max > 0)
    {
      s[0] = c;
      sfix (s, 1, max);
      ok = 1;
    }

  if (!ok)
    stringbug ("ccopy", s, NULL, c);

  return ok;
}

char *
pscopy (char *s, char *t, int start, int leng)
/* partial string copy, with Turbo Pascal convention for "start" */
/* BUG: position count starts at 1, not 0 ! */
{
  int max = maxlen (s);		/* keep it for later */
  int stop = length (t);
  int i;
  unsigned char ok = (max >= 0) && (max <= Maxstr);

  if (!ok)
    stringbug ("copy target non-init", s, t, 0);

  if (leng > max)
    {
      leng = max;
      ok = 0;
    }

  if (start > stop)
    {				/* nothing! */
      ok = 0;
      inistring (s, 0, max);
    }
  else
    {
      if ((start + leng - 1) > stop)
	{
	  leng = stop - start + 1;
	  ok = 0;
	}
      for (i = 0; i < leng; i++)
	s[i] = t[start + i - 1];

      i = leng;
      s[i] = 0;
      s[i + 1] = Hi (max);
      s[i + 2] = Lo (max);
    }
  /* if ( ! ok ) { stringbug("copy",s, t, 0) ;} */
  /* if ( ok ) { return s ;} else { return NULL ;} */
  ok = ok;
  return s;
}

char *
pscopy_up (char *s, char *t, int start, int leng)
/* partial string copy, with Turbo Pascal convention for "start" */
/* BUG: position count starts at 1, not 0 ! */
{
  int max = maxlen (s);		/* keep it for later */
  int stop = length (t);
  int i;
  unsigned char ok = (max >= 0) && (max <= Maxstr);

  if (!ok)
    stringbug ("copy target non-init", s, t, 0);

  if (leng > max)
    {
      leng = max;
      ok = 0;
    }

  if (start > stop)
    {				/* nothing! */
      ok = 0;
      inistring (s, 0, max);
    }
  else
    {
      if ((start + leng - 1) > stop)
	{
	  leng = stop - start + 1;
	  ok = 0;
	}
      for (i = 0; i < leng; i++)
	s[i] = upcase (t[start + i - 1]);

      i = leng;
      s[i] = 0;
      s[i + 1] = Hi (max);
      s[i + 2] = Lo (max);
    }
  /* if ( ! ok ) { stringbug("copy",s, t, 0) ;} */
  /* if ( ok ) { return s ;} else { return NULL ;} */
  ok = ok;
  return s;
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

unsigned char
nadd (char *s, long n)
/* append a decimal integer to a string */
{
  int d[25];
  int j, k, ls, len;
  char sg;			/* the sign */
  unsigned char ok;
  k = 0;
  len = maxlen (s);

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
    ok = cadd (s, '0');
  else
    {
      ls = length (s);
      ok = (len - ls) > k;
      if (ok)
	{
	  if (sg == '-')
	    {
	      s[ls] = sg;
	      ls++;
	    }
	  for (j = k - 1; j >= 0; j--)
	    {
	      s[ls] = d[j] + '0';
	      ls++;
	    }
	  sfix (s, ls, len);
	}
    }

  if (!ok)
    stringbug ("nadd", s, NULL, sg);

  return ok;
}

void
stri (long n, char *s)
/* convert integer to string */
{
  sini (s, maxlen (s));
  nadd (s, n);
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

unsigned char
steq (char *a, char *b)		/* string a==b test */
{
  unsigned short j = 0;
  while ((a[j] == b[j]) && (a[j] != 0) && (b[j] != 0))
    j++;

  return ((a[j] == 0) && (b[j] == 0)) /* string equality test */ ;
}

unsigned char
stne (char *s, char *t)
{
  return scompare (s, t) != 0;
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

unsigned char
alfa (char c)
{
  return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')) || c == '_'
    || c == '[' || c == ']';
}

unsigned char
num (char c)
{
  return (c >= '0') && (c <= '9');
}

unsigned char
alfanum (char c)
{
  return alfa (c) || ((c >= '0') && (c <= '9'));
}

int
freadstr (FILE * f, char *s, int max)
/* read a line from a file. 
   BUG: long lines truncated without warning, ctrl chars are dumped.
*/
{
  char c;
  int i = 0, mxlen = maxlen (s);

  if (mxlen < max)
    max = mxlen;

  do
    {
      c = fgetc (f);		/*  tab is the only control char accepted */
      if (((c >= ' ') || (c < 0) || (c == Tab)) && (i < max))
	{
	  s[i] = c;
	  i++;
	}
    }
  while (!(feof (f) || (c == '\n')));

  s[i] = 0;
  s[i + 1] = Hi (mxlen);
  s[i + 2] = Lo (mxlen);

  return i;
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
  unsigned char minus = 0;
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
new (long sz)
{
  void *p;
  if (sz <= 0)
    return NULL;
  else
    {
      p = tmalloc (sz);
      if (p == NULL)
	{			/* fatal error */
	  ws (" new() failure. Program halted.\n");
	  exit (1);
	}
      return p;
    }
}

char *
newstring (int n)
{
  char *s = (char *) new (n + 4);

  sini (s, n);
  return s;
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
strif (long i, int f, char *s)
/* formatting like str(i:f,s) in Turbo Pascal */
{
  int j, k, n, max;
  char cs;
  char t[32];
  k = 0;
  max = maxlen (s);

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
  n = 0;

  if ((f > k) && (f < 40))
    {				/* reasonable format */
      for (j = k + 2; j <= f; j++)
	{
	  s[n] = ' ';
	  n++;
	}
    }

  for (j = 0; j <= k + 1; j++)
    s[n + j] = t[k + j];	/* shift t down */

  k = length (s);
  sfix (s, k, max);
}

unsigned char
odd (long x)
{
  return (x & 1);
}

int
vali (char *s, long *i)
/* convert s to integer i. returns error code 0 if Ok */
/* BUG: almost identical to ival() with arg/return value swapped ... */
{
  int k = 0, digit = 0, ls;
  long z = 0;
  unsigned char minus = 0, ok = 1;
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

  *i = z;
  ok = (digit > 0) && (c == 0);	/* successful end of string */

  if (ok)
    return 0;
  else
    return k;			/* one beyond error position */
}

static unsigned char
match (char *s, char *t, int n, int tstart, unsigned char testcase)
{
/* returns 0 if ( tstart is out of range. But n may be 0 ? */
/* 1 if s matches t[tstart...tstart+n]  */
  int i, j, lt;
  unsigned char ok;
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
  /*  opt=1: like Turbo Pascal Pos, but case insensitive */
  /*  opt=2: position in space separated wordlist for scanners */
  int a, b, k, j;
  unsigned char ok, tstcase;
  Str (250, t);
  ok = 0;
  tstcase = (opt == 0);

  if (opt <= 1)
    scopy (t, sub);
  else
    {
      cadd (t, ' ');
      sadd (t, sub);
      cadd (t, ' ');
    }

  a = length (t);
  b = (int) (length (s) - a);
  k = 0;
  j = 1;

  if (a > 0)			/*;} else { return 0 */
    while ((k <= b) && (!ok))
      {
	ok = match (t, s, a, k, tstcase);	/* we must start at k=0 ! */
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
spos (char *sub, char *s)
/* equivalent to Turbo Pascal pos().
   BUG: counts 1 ... length(s), not from 0 like C  
*/
{
  char *ptr;

  if ((ptr = strstr (s, sub)))
    return strlen (s) - strlen (ptr) + 1;
  else
    return 0;

}

/**** float formatting with printf/scanf ******/

int
valr (char *s, double *r)
/* returns 0 if ok, else length of partial string ? */
{
  int n = sscanf (s, "%lG", r);
  if (n == 1)
    return (0);
  else
    return (1);
}

void
strf (double x, int f1, int f2, char *t)
/* e-format if f2<0, else f2 digits after the point, total width=f1 */
/* if f1=0, also e-format with f2 digits */
{				/*default f1=17, f2=-1 */
  Str (30, fmt);
  int n, mlt;
  mlt = maxlen (t);
  cadd (fmt, '%');
  if (f1 > 0)
    {
      nadd (fmt, f1);		/* f1 is the total width */
      if (f2 < 0)
	sadd (fmt, "lE");	/* exponent format */
      else
	{
	  cadd (fmt, '.');
	  nadd (fmt, f2);
	  sadd (fmt, "lg");
	}
    }
  else
    {
      cadd (fmt, '.');
      nadd (fmt, absi (f2 - 6));	/* note the 6 surplus positions */
      cadd (fmt, 'e');
    }
  n = sprintf (t, fmt, x);
  sfix (t, n, mlt);
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
  unsigned char minus = 0, ok = 1;
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

#ifndef _MATH_H

long
np_round (double x)
/* using <math.h>, it would be simpler: floor(x+0.5) */
{
  double u;
  long z;
  int n;
  Str (40, s);
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

double
intp (double x)
{
  double u = 2e9;
  if ((x > u) || (x < -u))
    return x;
  else
    return np_trunc (x);
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

double
intp (double x)			/* integral part */
{
  if (x >= 0.0)
    return floor (x);
  else
    return ceil (x);
}

#endif /* _MATH_H */
