/*       xpressn.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt
 *  Free software under the terms of the GNU Lesser General Public License
 */

#include <stdio.h>                /* for function message() only. */

#include "general.h"
#include "numparam.h"
#include "ngspice.h"

/* random numbers in /maths/misc/randnumb.c */
extern double gauss();

/************ keywords ************/

/* SJB - 150 chars is ample for this - see initkeys() */
static Str (150, keys);      /* all my keywords */
static Str (150, fmath);     /* all math functions */

extern char *nupa_inst_name; /* see spicenum.c */
extern long dynsubst;        /* see inpcom.c */
extern int dynLlen;

static double
ternary_fcn (int conditional, double if_value, double else_value)
{
  if (conditional)
    return if_value;
  else
    return else_value;
}


static double
agauss (double nominal_val, double variation, double sigma)
{
  double stdvar;
  stdvar=variation/sigma;
  return (nominal_val+stdvar*gauss());
}

static void
initkeys (void)
/* the list of reserved words */
{
  scopy_up (keys,
            "and or not div mod if else end while macro funct defined"
            " include for to downto is var");
  scopy_up (fmath,
            "sqr sqrt sin cos exp ln arctan abs pow pwr max min int log sinh cosh tanh ternary_fcn agauss");
}

static double
mathfunction (int f, double z, double x)
/* the list of built-in functions. Patch 'fmath', here and near line 888 to get more ...*/
{
  double y;
  switch (f) {
    case 1:
        y = x * x;
        break;
    case 2:
        y = sqrt (x);
        break;
    case 3:
        y = sin (x);
        break;
    case 4:
        y = cos (x);
        break;
    case 5:
        y = exp (x);
        break;
    case 6:
        y = ln (x);
        break;
    case 7:
        y = atan (x);
        break;
    case 8:
        y = fabs (x);
        break;
    case 9:
        y = pow (z, x);
        break;
    case 10:
        y = exp (x * ln (fabs (z)));
        break;
    case 11:
        y = MAX (x, z);
        break;
    case 12:
        y = MIN (x, z);
        break;
    case 13:
        y = trunc (x);
        break;
    case 14:
        y = log (x);
        break;
    case 15:
        y = sinh (x);
        break;
    case 16:
        y = cosh (x);
        break;
    case 17: 
        y=sinh(x)/cosh(x);
        break;
    default:
        y = x;
        break;
    }
  return y;
}

static unsigned char 
message (tdico * dic, char *s)
/* record 'dic' should know about source file and line */
{
   Strbig (dynLlen, t);
   dic->errcount++;
   if ((dic->srcfile != NULL) && dic->srcfile[0])
   {
      scopy (t, dic->srcfile);
      cadd (t, ':');
   }
   if (dic->srcline >= 0)
   {
      nadd (t, dic->srcline);
      sadd (t, ": ");
   }
   sadd (t, s);
   cadd (t, '\n');
   fputs (t, stderr);
   Strrem(t);

   return 1 /*error! */ ;
}

void
debugwarn (tdico * d, char *s)
{
   message (d, s);
   d->errcount--;
}


/************ the input text symbol table (dictionary) *************/

void
initdico (tdico * dico)
{
   int i;
   dico->nbd = 0;
   sini(dico->option,sizeof(dico->option)-4);
   sini(dico->srcfile,sizeof(dico->srcfile)-4);
   dico->srcline = -1;
   dico->errcount = 0;

   dico->dyndat = (entry*)tmalloc(3 * sizeof(entry));
  
   for (i = 0; i < 3; i++)
      sini (dico->dyndat[i].nom, 100);

   dico->tos = 0;
   dico->stack[dico->tos] = 0;        /* global data beneath */
   initkeys ();
}

/*  local semantics for parameters inside a subckt */
/*  arguments as wll as .param expressions  */
/* to do:  scope semantics ?
   "params:" and all new symbols should have local scope inside subcircuits.
   redefinition of old symbols gives a warning message.
*/

typedef enum {Push='u'} _nPush;
typedef enum {Pop='o'} _nPop;

static void
dicostack (tdico * dico, char op)
/* push or pop operation for nested subcircuit locals */
{
   char *param_name, *inst_name;
   int i, current_stack_size, old_stack_size;

   if (op == Push)
   {
      if (dico->tos < (20 - 1))
         dico->tos++;
      else
         message (dico, " Subckt Stack overflow");

      dico->stack[dico->tos] = dico->nbd;
      dico->inst_name[dico->tos] = nupa_inst_name;
   }
   else if (op == Pop)
   {
      if (dico->tos > 0)
      {
         /* keep instance parameters around */
         current_stack_size = dico->nbd;
         old_stack_size = dico->stack[dico->tos];
         inst_name = dico->inst_name[dico->tos];

         for (i = old_stack_size + 1; i <= current_stack_size; i++)
         {
            param_name =
               tmalloc (strlen (inst_name) + strlen (dico->dyndat[i].nom) + 2);
            sprintf (param_name, "%s.%s", inst_name, dico->dyndat[i].nom);
            nupa_add_inst_param (param_name, dico->dyndat[i].vl); 
            tfree (param_name);
         }
         tfree (inst_name);

         dico->nbd = dico->stack[dico->tos];        /* simply kill all local items */
         dico->tos--;
      }
      else
      {
         message (dico, " Subckt Stack underflow.");
      }
   }
}

int
donedico (tdico * dico)
{
   int sze = dico->nbd;
   return sze;
}

static int
entrynb (tdico * d, char *s)
/* symbol lookup from end to start,  for stacked local symbols .*/
/* bug: sometimes we need access to same-name symbol, at lower level? */
{
   int i;
   unsigned char ok;
   ok = 0;
   i = d->nbd + 1;

   while ((!ok) && (i > 1))
   {
      i--;
      ok = steq (d->dyndat[i].nom, s);
   }
   if (!ok)
      return 0;
   else
      return i;
}

char
getidtype (tdico * d, char *s)
/* test if identifier s is known. Answer its type, or '?' if not in list */
{
   char itp = '?';                /* assume unknown */
   int i = entrynb (d, s);

   if (i > 0)
      itp = d->dyndat[i].tp;
   return itp;
}

static double
fetchnumentry (tdico * dico, char *t, unsigned char *perr)
{
   unsigned char err = *perr;
   unsigned short k;
   double u;
   Strbig (dynLlen, s);
   k = entrynb (dico, t);        /*no keyword */
   /*dbg -- if ( k<=0 ) { ws("Dico num lookup fails. ") ;} */

   while ((k > 0) && (dico->dyndat[k].tp == 'P'))
      k = dico->dyndat[k].ivl;        /*pointer chain */

   if (k > 0)
      if (dico->dyndat[k].tp != 'R')
         k = 0;

   if (k > 0)
      u = dico->dyndat[k].vl;
   else
   {
      u = 0.0;
      scopy (s, "Undefined number [");
      sadd (s, t);
      cadd (s, ']');
      err = message (dico, s);
   }

   *perr = err;

   Strrem(s);

   return u;
}

/*******  writing dictionary entries *********/

int
attrib (tdico * dico, char *t, char op)
{
/* seek or attribute dico entry number for string t.
   Option  op='N' : force a new entry, if tos>level and old is  valid.
*/
   int i;
   unsigned char ok;
   i = dico->nbd + 1;
   ok = 0;
   while ((!ok) && (i > 1))
   {                                /*search old */
      i--;
      ok = steq (dico->dyndat[i].nom, t);
   }

   if (ok && (op == 'N')
     && (dico->dyndat[i].level < dico->tos) && (dico->dyndat[i].tp != '?'))
   {
      ok = 0;
   }

   if (!ok)
   {
      dico->nbd++;
      i = dico->nbd;
      dico->dyndat = trealloc(dico->dyndat, (i+1) * sizeof(entry));
      sini (dico->dyndat[i].nom, 100);
      scopy (dico->dyndat[i].nom, t);
      dico->dyndat[i].tp = '?';        /*signal Unknown */
      dico->dyndat[i].level = dico->tos;
   }
   return i;
}

static unsigned char
define (tdico * dico, 
        char *t,        /* identifier to define */
        char op,        /* option */
        char tpe,       /* type marker */
        double z,       /* float value if any */
        int w,          /* integer value if any */
        char *base)     /* string pointer if any */
{
/*define t as real or integer,
  opcode= 'N' impose a new item under local conditions.
  check for pointers, too, in full macrolanguage version:
     Call with 'N','P',0.0, ksymbol ... for VAR parameter passing.
  Overwrite warning, beware: During 1st pass (macro definition),
      we already make symbol entries which are dummy globals !
      we mark each id with its subckt level, and warn if write at higher one.
*/
   int i;
   char c;
   unsigned char err, warn;
   Strbig (dynLlen, v);
   i = attrib (dico, t, op);
   err = 0;
   if (i <= 0)
      err = message (dico, " Symbol table overflow");
   else
   {
      if (dico->dyndat[i].tp == 'P')
         i = dico->dyndat[i].ivl; /*pointer indirection */

      if (i > 0)
         c = dico->dyndat[i].tp;
      else
         c = ' ';

      if ((c == 'R') || (c == 'S') || (c == '?'))
      {
         dico->dyndat[i].vl = z;
         dico->dyndat[i].tp = tpe;
         dico->dyndat[i].ivl = w;
         dico->dyndat[i].sbbase = base;
         /* if ( (c !='?') && (i<= dico->stack[dico->tos]) ) {  */
         if (c == '?')
            dico->dyndat[i].level = dico->tos; /* promote! */

         if (dico->dyndat[i].level < dico->tos)
         {
            /* warn about re-write to a global scope! */
            scopy (v, t);
            cadd (v, ':');
            nadd (v, dico->dyndat[i].level);
            sadd (v, " overwritten.");
            warn = message (dico, v);
         }
      }
      else
      {
         scopy (v, t);
         sadd (v, ": cannot redefine");
         /* suppress error message, resulting from multiple definition of
         symbols (devices) in .model lines with same name, but in different subcircuits.
         Subcircuit expansion is o.k., we have to deal with this numparam
         behaviour later. (H. Vogt 090426) 
         */
         /*err = message (dico, v);*/
      }
   }
   Strrem(v);
   return err;
}

unsigned char
defsubckt (tdico * dico, char *s, int w, char categ)
/* called on 1st pass of spice source code,
   to enter subcircuit (categ=U) and model (categ=O) names
*/
{
   Str (80, u);
   unsigned char err;
   int i, j, ls;
   ls = length (s);
   i = 0;

   while ((i < ls) && (s[i] != '.'))
      i++;                        /* skip 1st dotword */

   while ((i < ls) && (s[i] > ' '))
      i++;

   while ((i < ls) && (s[i] <= ' '))
      i++;                        /* skip blank */

   j = i;

   while ((j < ls) && (s[j] > ' '))
      j++;

   if ((j > i))
   {
      pscopy_up (u, s, i + 1, j - i);
      err = define (dico, u, ' ', categ, 0.0, w, NULL);
   }
   else
      err = message (dico, "Subcircuit or Model without name.");

   return err;
}

int
findsubckt (tdico * dico, char *s, char *subname)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value
   and the name in string subname  */
{
   Str (80, u);                        /* u= subckt name is last token in string s */
   int i, j, k;
   k = length (s);

   while ((k >= 0) && (s[k] <= ' '))
      k--;

   j = k;

   while ((k >= 0) && (s[k] > ' '))
      k--;

   pscopy_up (u, s, k + 2, j - k);
   i = entrynb (dico, u);

   if ((i > 0) && (dico->dyndat[i].tp == 'U'))
   {
      i = dico->dyndat[i].ivl;
      scopy (subname, u);
   }
   else
   {
      i = 0;
      scopy (subname, "");
      message (dico, "Cannot find subcircuit.");
   }

   return i;
}

#if 0                                /* unused, from the full macro language... */
static int
deffuma (                        /* define function or macro entry. */
          tdico * dico, char *t, char tpe, unsigned short bufstart,
          unsigned char *pjumped, unsigned char *perr)
{
  unsigned char jumped = *pjumped;
  unsigned char err = *perr;
/* if not jumped, define new function or macro, returns index to buffferstart
   if jumped, return index to existing function
*/
  int i, j;
  Strbig (Llen, v);
  i = attrib (dico, t, ' ');
  j = 0;
  if (i <= 0)
    {
      err = message (dico, " Symbol table overflow");
    }
  else
    {
      if (dico->dat[i].tp != '?')
        {                        /*old item! */
          if (jumped)
            {
              j = dico->dat[i].ivl;
            }
          else
            {
              scopy (v, t);
              sadd (v, " already defined");
              err = message (dico, v);
            }
        }
      else
        {
          dico->dat[i].tp = tpe;
          dico->nfms++;
          j = dico->nfms;
          dico->dat[i].ivl = j;
          dico->fms[j].start = bufstart;
            /* =ibf->bufaddr = start addr in buffer */ ;
        }
    }
  *pjumped = jumped;
  *perr = err;
  return j;
}
#endif

/************ input scanner stuff **************/

static unsigned char
keyword (char *keys, char *t)
{
/* return 0 if t not found in list keys, else the ordinal number */
   unsigned char i, j, k;
   int lt, lk;
   unsigned char ok;
   lt = length (t);
   lk = length (keys);
   k = 0;
   j = 0;

   do {
      j++;
      i = 0;
      ok = 1;

      do {
         i++;
         k++;
         ok = (k <= lk) && (t[i - 1] == keys[k - 1]);
      } while (!((!ok) || (i >= lt)));

      if (ok)
         ok = (k == lk) || (keys[k] <= ' ');

      if (!ok && (k < lk))    /*skip to next item */
         while ((k <= lk) && (keys[k - 1] > ' '))
            k++;
   } while (!(ok || (k >= lk)));

   if (ok)
      return j;
   else
      return 0;
}

static double
parseunit (double x, char *s)
/* the Spice suffixes */
{
  double u = 0;
  Str (20, t);
  unsigned char isunit;
  isunit = 1;
  pscopy (t, s, 1, 3);

  if (steq (t, "MEG"))
      u = 1e6;
  else if (s[0] == 'G')
      u = 1e9;
  else if (s[0] == 'K')
      u = 1e3;
  else if (s[0] == 'M')
      u = 0.001;
  else if (s[0] == 'U')
      u = 1e-6;
  else if (s[0] == 'N')
      u = 1e-9;
  else if (s[0] == 'P')
      u = 1e-12;
  else if (s[0] == 'F')
      u = 1e-15;
  else
      isunit = 0;

  if (isunit)
    x = x * u;

  return x;
}

static int
fetchid (char *s, char *t, int ls, int i)
/* copy next identifier from s into t, advance and return scan index i */
{
  char c;
  unsigned char ok;
  c = s[i - 1];

  while ((!alfa (c)) && (i < ls))
    {
      i++;
      c = s[i - 1];
    }

  scopy (t, "");
  cadd (t, upcase (c));

  do {
      i++;
      if (i <= ls)
          c = s[i - 1];
      else
          c = Nul;

      c = upcase (c);
      ok = alfanum (c) || c == '.';

      if (ok)
        cadd (t, c);

    } while (ok);
  return i /*return updated i */ ;
}

static double
exists (tdico * d, char *s, int *pi, unsigned char *perror)
/* check if s in simboltable 'defined': expect (ident) and return 0 or 1 */
{
  unsigned char error = *perror;
  int i = *pi;
  double x;
  int ls;
  char c;
  unsigned char ok;
  Strbig (dynLlen, t);
  ls = length (s);
  x = 0.0;

  do {
      i++;
      if (i > ls)
          c = Nul;
      else
          c = s[i - 1];

      ok = (c == '(');
    } while (!(ok || (c == Nul)));

  if (ok)
    {
      i = fetchid (s, t, ls, i);
      i--;
      if (entrynb (d, t) > 0)
        x = 1.0;

      do {
          i++;

          if (i > ls)
              c = Nul;
          else
              c = s[i - 1];

          ok = (c == ')');
        } while (!(ok || (c == Nul)));
    }
  if (!ok)
      error = message (d, " Defined() syntax");

/*keep pointer on last closing ")" */

  *perror = error;
  *pi = i;
  Strrem(t);
  return x;
}

static double
fetchnumber (tdico * dico, char *s, int ls, int *pi, unsigned char *perror)
/* parse a Spice number in string s */
{
  unsigned char error = *perror;
  int i = *pi;
  int k, err;
  char d;
  Str (20, t);
//  Strbig (Llen, v);
  double u;
  Strbig (dynLlen, v);
  k = i;

  do {
      k++;
      if (k > ls)
          d = (char)(0);
      else
          d = s[k - 1];
    } while (!(!((d == '.') || ((d >= '0') && (d <= '9')))));

  if ((d == 'e') || (d == 'E'))
    {                                /*exponent follows */
      k++;
      d = s[k - 1];

      if ((d == '+') || (d == '-'))
        k++;

      do {
          k++;
          if (k > ls)
              d = (char)(0);
          else
              d = s[k - 1];
        } while (!(!((d >= '0') && (d <= '9'))));
    }

  pscopy (t, s, i, k - i);

  if (t[0] == '.')
      cins (t, '0');
  else if (t[length (t) - 1] == '.')
      cadd (t, '0');

  u = rval (t, &err);

  if (err != 0)
    {
      scopy (v, "Number format error: ");
      sadd (v, t);
      error = message (dico, v);
    }
  else
    {
      scopy (t, "");
      while (alfa (d))
        {
          cadd (t, upcase (d));
          k++;

          if (k > ls)
              d = Nul;
          else
              d = s[k - 1];
        }

      u = parseunit (u, t);
    }

  i = k - 1;
  *perror = error;
  *pi = i;
  Strrem(v);
  return u;
}

static char
fetchoperator (tdico * dico,
               char *s, int ls,
               int *pi,
               unsigned char *pstate, unsigned char *plevel,
               unsigned char *perror)
/* grab an operator from string s and advance scan index pi.
   each operator has: one-char alias, precedence level, new interpreter state.
*/
{
  int i = *pi;
  unsigned char state = *pstate;
  unsigned char level = *plevel;
  unsigned char error = *perror;
  char c, d;
  Strbig (dynLlen, v);
  c = s[i - 1];

  if (i < ls)
      d = s[i];
  else
      d = Nul;

  if ((c == '!') && (d == '='))
    {
      c = '#';
      i++;
    }
  else if ((c == '<') && (d == '>'))
    {
      c = '#';
      i++;
    }
  else if ((c == '<') && (d == '='))
    {
      c = 'L';
      i++;
    }
  else if ((c == '>') && (d == '='))
    {
      c = 'G';
      i++;
    }
  else if ((c == '*') && (d == '*'))
    {
      c = '^';
      i++;
    }
  else if ((c == '=') && (d == '='))
    {
      i++;
    }
  else if ((c == '&') && (d == '&'))
    {
      i++;
    }
  else if ((c == '|') && (d == '|'))
    {
      i++;
    }
  if ((c == '+') || (c == '-'))
    {
      state = 2;                /*pending operator */
      level = 4;
    }
  else if ((c == '*') || (c == '/') || (c == '%') || (c == '\\'))
    {
      state = 2;
      level = 3;
    }
  else if (c == '^')
    {
      state = 2;
      level = 2;
    }
  else if (cpos (c, "=<>#GL") > 0)
    {
      state = 2;
      level = 5;
    }
  else if (c == '&')
    {
      state = 2;
      level = 6;
    }
  else if (c == '|')
    {
      state = 2;
      level = 7;
    }
  else if (c == '!')
    {
      state = 3;
    }
  else
    {
      state = 0;
      if (c > ' ')
        {
          scopy (v, "Syntax error: letter [");
          cadd (v, c);
          cadd (v, ']');
          error = message (dico, v);
        }
    }
  *pi = i;
  *pstate = state;
  *plevel = level;
  *perror = error;
  Strrem(v);
  return c;
}

static char
opfunctkey (tdico * dico,
            unsigned char kw, char c,
            unsigned char *pstate, unsigned char *plevel,
            unsigned char *perror)
/* handle operator and built-in keywords */
{
  unsigned char state = *pstate;
  unsigned char level = *plevel;
  unsigned char error = *perror;
/*if kw operator keyword, c=token*/
  switch (kw)
    {                                /*& | ~ DIV MOD  Defined */
    case 1:
      c = '&';
      state = 2;
      level = 6;
      break;
    case 2:
      c = '|';
      state = 2;
      level = 7;
      break;
    case 3:
      c = '!';
      state = 3;
      level = 1;
      break;
    case 4:
      c = '\\';
      state = 2;
      level = 3;
      break;
    case 5:
      c = '%';
      state = 2;
      level = 3;
      break;
    case Defd:
      c = '?';
      state = 1;
      level = 0;
      break;
    default:
      state = 0;
      error = message (dico, " Unexpected Keyword");
      break;
    }                                /*case */

  *pstate = state;
  *plevel = level;
  *perror = error;
  return c;
}

static double
operate (char op, double x, double y)
{
/* execute operator op on a pair of reals */
/* bug:   x:=x op y or simply x:=y for empty op?  No error signalling! */
  double u = 1.0;
  double z = 0.0;
  double epsi = 1e-30;
  double t;
  switch (op)
    {
    case ' ':
      x = y; /*problem here: do type conversions ?! */ ;
      break;
    case '+':
      x = x + y;
      break;
    case '-':
      x = x - y;
      break;
    case '*':
      x = x * y;
      break;
    case '/':
      if (absf (y) > epsi)
        x = x / y;
      break;
    case '^':                        /*power */
      t = absf (x);
      if (t < epsi)
        x = z;
      else
        x = exp (y * ln (t));
      break;
    case '&':                        /*&& */
      if (y < x)
        x = y; /*=Min*/ ;
      break;
    case '|':                        /*|| */
      if (y > x)
        x = y; /*=Max*/ ;
      break;
    case '=':
      if (x == y)
        x = u;
      else
        x = z;
      break;
    case '#':                        /*<> */
      if (x != y)
        x = u;
      else
        x = z;
      break;
    case '>':
      if (x > y)
        x = u;
      else
        x = z;
      break;
    case '<':
      if (x < y)
        x = u;
      else
        x = z;
      break;
    case 'G':                        /*>= */
      if (x >= y)
        x = u;
      else
        x = z;
      break;
    case 'L':                        /*<= */
      if (x <= y)
        x = u;
      else
        x = z;
      break;
    case '!':                        /*! */
      if (y == z)
        x = u;
      else
        x = z;
      break;
    case '%':                        /*% */
      t = np_trunc (x / y);
      x = x - y * t;
      break;
    case '\\':                        /*/ */
      x = np_trunc (absf (x / y));
      break;
    }                                /*case */
  return x;
}

static double
formula (tdico * dico, char *s, unsigned char *perror)
{
/* Expression parser.
  s is a formula with parentheses and math ops +-* / ...
  State machine and an array of accumulators handle operator precedence.
  Parentheses handled by recursion.
  Empty expression is forbidden: must find at least 1 atom.
  Syntax error if no toggle between binoperator && (unop/state1) !
  States : 1=atom, 2=binOp, 3=unOp, 4= stop-codon.
  Allowed transitions:  1->2->(3,1) and 3->(3,1).
*/
  typedef enum {nprece=9} _nnprece;                /*maximal nb of precedence levels */
  unsigned char error = *perror;
  unsigned char negate = 0;
  unsigned char state, oldstate, topop, ustack, level, kw, fu;
  double u = 0.0, v, w = 0.0;
  double accu[nprece + 1];
  char oper[nprece + 1];
  char uop[nprece + 1];
  int i, k, ls, natom, arg2, arg3;
  char c, d;
//  Strbig (Llen, t);
  unsigned char ok;
  Strbig (dynLlen, t);

  for (i = 0; i <= nprece; i++)
    {
      accu[i] = 0.0;
      oper[i] = ' ';
    }
  i = 0;
  ls = length (s);

  while ((ls > 0) && (s[ls - 1] <= ' '))
    ls--;                        /*clean s */

  state = 0;
  natom = 0;
  ustack = 0;
  topop = 0;
  oldstate = 0;
  fu = 0;
  error = 0;
  level = 0;

  while ((i < ls) && (!error))
    {
      i++;
      c = s[i - 1];
      if (c == '(')
        {                        /*sub-formula or math function */
          level = 1;
          /* new: must support multi-arg functions */
          k = i;
          arg2 = 0;
          v = 1.0;
          arg3 = 0;

          do {
              k++;
              if (k > ls)
                  d = (char)(0);
              else
                  d = s[k - 1];

              if (d == '(')
                  level++;
              else if (d == ')')
                  level--;

              if ((d == ',') && (level == 1))
                {
                  if (arg2 == 0)
                    arg2 = k;
                  else
                    arg3 = k;        // kludge for more than 2 args (ternary expression);
                } /* comma list? */ ;
            }
          while (!((k > ls) || ((d == ')') && (level <= 0))));

          if (k > ls)
            {
              error = message (dico, "Closing \")\" not found.");
              natom++; /*shut up other error message */ ;
            }
          else
            {
              if (arg2 > i)
                {
                  pscopy (t, s, i + 1, arg2 - i - 1);
                  v = formula (dico, t, &error);
                  i = arg2;
                }
              if (arg3 > i)
                {
                  pscopy (t, s, i + 1, arg3 - i - 1);
                  w = formula (dico, t, &error);
                  i = arg3;
                }
              pscopy (t, s, i + 1, k - i - 1);
              u = formula (dico, t, &error);
              state = 1;        /*atom */
              if (fu > 0)
                {
                  if ((fu == 18))
                      u = ternary_fcn ((int) v, w, u);
                  else if ((fu == 19))
                      u = agauss (v, w, u);
                  else
                      u = mathfunction (fu, v, u);

                }
            }
          i = k;
          fu = 0;
        }
      else if (alfa (c))
        {
          i = fetchid (s, t, ls, i);        /*user id, but sort out keywords */
          state = 1;
          i--;
          kw = keyword (keys, t);        /*debug ws('[',kw,']'); */
          if (kw == 0)
            {
              fu = keyword (fmath, t);        /* numeric function? */
              if (fu == 0)
                  u = fetchnumentry (dico, t, &error);
              else
                  state = 0; /* state==0 means: ignore for the moment */
            }
          else
              c = opfunctkey (dico, kw, c, &state, &level, &error);

          if (kw == Defd)
              u = exists (dico, s, &i, &error);
        }
      else if (((c == '.') || ((c >= '0') && (c <= '9'))))
        {
          u = fetchnumber (dico, s, ls, &i, &error);
          if (negate)
            {
              u = -1 * u;
              negate = 0;
            }
          state = 1;
        }
      else
          c = fetchoperator (dico, s, ls, &i, &state, &level, &error);
          /* may change c to some other operator char! */
          /* control chars <' '  ignored */

      ok = (oldstate == 0) || (state == 0) ||
        ((oldstate == 1) && (state == 2)) || ((oldstate != 1)
                                              && (state != 2));
      if (oldstate == 2 && state == 2 && c == '-')
        {
          ok = 1;
          negate = 1;
          continue;
        }

      if (!ok)
          error = message (dico, " Misplaced operator");

      if (state == 3)
        {                        /*push unary operator */
          ustack++;
          uop[ustack] = c;
        }
      else if (state == 1)
        {                        /*atom pending */
          natom++;
          if (i >= ls)
            {
              state = 4;
              level = topop;
            }                        /*close all ops below */
          for (k = ustack; k >= 1; k--)
              u = operate (uop[k], u, u);

          ustack = 0;
          accu[0] = u; /* done: all pending unary operators */ ;
        }

      if ((state == 2) || (state == 4))
        {
          /* do pending binaries of priority Upto "level" */
          for (k = 1; k <= level; k++)
            {                        /* not yet speed optimized! */
              accu[k] = operate (oper[k], accu[k], accu[k - 1]);
              accu[k - 1] = 0.0;
              oper[k] = ' '; /*reset intermediates */ ;
            }
          oper[level] = c;

          if (level > topop)
              topop = level;
        }
      if ((state > 0))
        {
          oldstate = state;
        }
    } /*while */ ;
  if ((natom == 0) || (oldstate != 4))
    {
      scopy (t, " Expression err: ");
      sadd (t, s);
      error = message (dico, t);
    }

  if (negate == 1)
    {
      error =
        message (dico,
                 " Problem with formula eval -- wrongly determined negation!");
    }

  *perror = error;

  Strrem(t);

  if (error)
      return 1.0;
  else
      return accu[topop];
}                                /*formula */

static char
fmttype (double x)
{
/* I=integer, P=fixedpoint F=floatpoint*/
/*  find out the "natural" type of format for number x*/
  double ax, dx;
  int rx;
  unsigned char isint, astronomic;
  ax = absf (x);
  isint = 0;
  astronomic = 0;

  if (ax < 1e-30)
      isint = 1;
  else if (ax < 32000)
    {                                /*detect integers */
      rx = np_round (x);
      dx = (x - rx) / ax;
      isint = (absf (dx) < 1e-6);
    }

  if (!isint)
      astronomic = (ax >= 1e6) || (ax < 0.01);

  if (isint)
      return 'I';
  else if (astronomic)
      return 'F';
  else
      return 'P';
}

static unsigned char
evaluate (tdico * dico, char *q, char *t, unsigned char mode)
{
/* transform t to result q. mode 0: expression, mode 1: simple variable */
  double u = 0.0;
  int k, j, lq;
  char dt, fmt;
  unsigned char numeric, done, nolookup;
  unsigned char err;
  Strbig (dynLlen, v);
  scopy (q, "");
  numeric = 0;
  err = 0;

  if (mode == 1)
    {                                /*string? */
      stupcase (t);
      k = entrynb (dico, t);
      nolookup = (k <= 0);
      while ((k > 0) && (dico->dyndat[k].tp == 'P'))
          k = dico->dyndat[k].ivl;

      /*pointer chain */
      if (k > 0)
          dt = dico->dyndat[k].tp;
      else
          dt = ' ';

      /*data type: Real or String */
      if (dt == 'R')
        {
          u = dico->dyndat[k].vl;
          numeric = 1;
        }
      else if (dt == 'S')
        {                        /*suppose source text "..." at */
          j = dico->dyndat[k].ivl;
          lq = 0;
          do {
              j++;
              lq++;
              dt = /*ibf->bf[j]; */ dico->dyndat[k].sbbase[j];

              if (cpos ('3', dico->option) <= 0)
                  dt = upcase (dt); /* spice-2 */

              done = (dt == '\"') || (dt < ' ') || (lq > 99);

              if (!done)
                  cadd (q, dt);
            } while (!(done));
        }
      else
        k = 0;

      if (k <= 0)
        {
          scopy (v, "");
          cadd (v, '\"');
          sadd (v, t);
          sadd (v, "\" not evaluated. ");

          if (nolookup)
              sadd (v, "Lookup failure.");

          err = message (dico, v);
        }
    }
  else
    {
      u = formula (dico, t, &err);
      numeric = 1;
    }
  if (numeric)
    {
      fmt = fmttype (u);
      if (fmt == 'I')
          stri (np_round (u), q);
      else
        {
          //strf(u,6,-1,q);
          strf (u, 17, 10, q);
        } /* strf() arg 2 doesnt work: always >10 significant digits ! */ ;
    }
  Strrem(v);
  return err;
}

#if 0
static unsigned char
scanline (tdico * dico, char *s, char *r, unsigned char err)
/* scan host code line s for macro substitution.  r=result line */
{
  int i, k, ls, level, nd, nnest;
  unsigned char spice3;
  char c, d;
  Strbig (Llen, q);
  Strbig (Llen, t);
  Str (20, u);
  spice3 = cpos ('3', dico->option) > 0;        /* we had -3 on the command line */
  i = 0;
  ls = length (s);
  scopy (r, "");
  err = 0;
  pscopy (u, s, 1, 3);
  if ((ls > 7) && steq (u, "**&"))
    {                                /*special Comment **&AC #... */
      pscopy (r, s, 1, 7);
      i = 7;
    }
  while ((i < ls) && (!err))
    {
      i++;
      c = s[i - 1];
      if (c == Pspice)
        {                        /* try pspice expression syntax */
          k = i;
          nnest = 1;
          do
            {
              k++;
              d = s[k - 1];
              if (d == '{')
                {
                  nnest++;
                }
              else if (d == '}')
                {
                  nnest--;
                }
            }
          while (!((nnest == 0) || (d == 0)));
          if (d == 0)
            {
              err = message (dico, "Closing \"}\" not found.");
            }
          else
            {
              pscopy (t, s, i + 1, k - i - 1);
              err = evaluate (dico, q, t, 0);
            }
          i = k;
          if (!err)
            {                        /*insert number */
              sadd (r, q);
            }
          else
            {
              err = message (dico, s);
            }
        }
      else if (c == Intro)
        {
          Inc (i);
          while ((i < ls) && (s[i - 1] <= ' '))
            i++;
          k = i;
          if (s[k - 1] == '(')
            {                        /*sub-formula */
              level = 1;
              do
                {
                  k++;
                  if (k > ls)
                    {
                      d = chr (0);
                    }
                  else
                    {
                      d = s[k - 1];
                    }
                  if (d == '(')
                    {
                      level++;
                    }
                  else if (d == ')')
                    {
                      level--;
                    }
                }
              while (!((k > ls) || ((d == ')') && (level <= 0))));
              if (k > ls)
                {
                  err = message (dico, "Closing \")\" not found.");
                }
              else
                {
                  pscopy (t, s, i + 1, k - i - 1);
                  err = evaluate (dico, q, t, 0);
                }
              i = k;
            }
          else
            {                        /*simple identifier may also be string */
              do
                {
                  k++;
                  if (k > ls)
                    {
                      d = chr (0);
                    }
                  else
                    {
                      d = s[k - 1];
                    }
                }
              while (!((k > ls) || (d <= ' ')));
              pscopy (t, s, i, k - i);
              err = evaluate (dico, q, t, 1);
              i = k - 1;
            }
          if (!err)
            {                        /*insert the number */
              sadd (r, q);
            }
          else
            {
              message (dico, s);
            }
        }
      else if (c == Nodekey)
        {                        /*follows: a node keyword */
          do
            {
              i++;
            }
          while (!(s[i - 1] > ' '));
          k = i;
          do
            {
              k++;
            }
          while (!((k > ls) || !alfanum (s[k - 1])));
          pscopy (q, s, i, k - i);
          nd = parsenode (Addr (dico->nodetab), q);
          if (!spice3)
            {
              stri (nd, q);
            }                        /* substitute by number */
          sadd (r, q);
          i = k - 1;
        }
      else
        {
          if (!spice3)
            {
              c = upcase (c);
            }
          cadd (r, c); /*c<>Intro */ ;
        }
    }                                /*while */
  return err;
}
#endif

/********* interface functions for spice3f5 extension ***********/

static void
compactfloatnb (char *v)
/* try to squeeze a floating pt format to 10 characters */
/* erase superfluous 000 digit streams before E */
/* bug: truncating, no rounding */
{
  int n, k, m, lex, lem;
  Str (20, expo);
  Str (10, expn);
  n = cpos ('E', v);            /* if too long, try to delete digits */
  if (n==0) n = cpos ('e', v);

  if (n > 0) {
    pscopy (expo, v, n, length (v));
    lex = length (expo);
    if (lex > 4) {            /* exponent only 2 digits */
      pscopy (expn, expo, 2, 4);
      if (atoi(expn) < -99) scopy(expo, "e-099"); /* brutal */
      if (atoi(expn) > +99) scopy(expo, "e+099");
      expo[2] = expo[3];
      expo[3] = expo[4];
      expo[4] = '\0';
      lex = 4;
    }
    k = n - 1;                /* mantissa is 0...k */

    m = 17;
    while (v[m] != ' ')
      m--;
    m++;
    while ((v[k] == '0') && (v[k - 1] == '0'))
      k--;

    lem = k - m;

    if ((lem + lex) > 10)
      lem = 10 - lex;

    pscopy (v, v, m+1, lem);
    if (cpos('.', v) > 0) {
      while (lem < 6) {
        cadd(v, '0');
        lem++;
      }
    } else {
      cadd(v, '.');
      lem++;
      while (lem < 6) {
        cadd(v, '0');
        lem++;
      }
    }
    sadd (v, expo);
  } else {
    m = 0;
    while (v[m] == ' ')
      m++;

    lem = length(v) - m;
    if (lem > 10) lem = 10;
    pscopy (v, v, m+1, lem);
  }
}

static int
insertnumber (tdico * dico, int i, char *s, char *u)
/* insert u in string s in place of the next placeholder number */
{
  Str (40, v);
  Str (80, msg);
  unsigned char found;
  int ls, k;
  long accu;
  ls = length (s);

  scopy (v, u);
  compactfloatnb (v);

  while (length (v) < 17)
      cadd (v, ' ');

  if (length (v) > 17)
    {
      scopy (msg, " insertnumber fails: ");
      sadd (msg, u);
      message (dico, msg);
    }

  found = 0;

  while ((!found) && (i < ls))
    {
      found = (s[i] == '1');
      k = 0;
      accu = 0;

      while (found && (k < 10))
        {                        /* parse a 10-digit number */
          found = num (s[i + k]);

          if (found)
              accu = 10 * accu + s[i + k] - '0';

          k++;
        }

      if (found)
        {
          accu = accu - 1000000000L;        /* plausibility test */
          found = (accu > 0) && (accu < dynsubst + 1); /* dynsubst numbers have been allocated */
        }
      i++;
    }

  if (found)
    {                                /* substitute at i-1 */
      i--;
      for (k = 0; k < 11; k++)
        s[i + k] = v[k];

      i = i + 17;

    }
  else
    {
      i = ls;
      fprintf (stderr, "xpressn.c--insertnumber:  i=%d  s=%s  u=%s\n", i, s,
               u);
      message (dico, "insertnumber: missing slot ");
    }
  return i;
}

unsigned char
nupa_substitute (tdico * dico, char *s, char *r, unsigned char err)
/* s: pointer to original source line.
   r: pointer to result line, already heavily modified wrt s
   anywhere we find a 10-char numstring in r, substitute it.
  bug: wont flag overflow!
*/
{
  int i, k, ls, level, nnest, ir;
  char c, d;
//  Strbig (Llen, q);
//  Strbig (Llen, t);
  Strdbig (dynLlen, q, t);
  i = 0;
  ls = length (s);
  err = 0;
  ir = 0;

  while ((i < ls) && (!err))
    {
      i++;
      c = s[i - 1];
      if (c == Pspice)
        {                        /* try pspice expression syntax */
          k = i;
          nnest = 1;
          do {
              k++;
              d = s[k - 1];
              if (d == '{')
                  nnest++;
              else if (d == '}')
                  nnest--;
            } while (!((nnest == 0) || (d == 0)));

          if (d == 0)
              err = message (dico, "Closing \"}\" not found.");
          else
            {
              pscopy (t, s, i + 1, k - i - 1);
              err = evaluate (dico, q, t, 0);
            }

          i = k;
          if (!err)
              ir = insertnumber (dico, ir, r, q);
          else
              err = message (dico, "Cannot compute substitute");
        }
      else if (c == Intro)
        {
          i++;
          while ((i < ls) && (s[i - 1] <= ' '))
            i++;

          k = i;

          if (s[k - 1] == '(')
            {                        /*sub-formula */
              level = 1;
              do {
                  k++;
                  if (k > ls)
                      d = (char)(0);
                  else
                      d = s[k - 1];

                  if (d == '(')
                      level++;
                  else if (d == ')')
                      level--;
                } while (!((k > ls) || ((d == ')') && (level <= 0))));

              if (k > ls)
                  err = message (dico, "Closing \")\" not found.");
              else
                {
                  pscopy (t, s, i + 1, k - i - 1);
                  err = evaluate (dico, q, t, 0);
                }
              i = k;
            }
          else
            {                        /*simple identifier may also be string? */
              do {
                  k++;
                  if (k > ls)
                      d = (char)(0);
                  else
                      d = s[k - 1];
                } while (!((k > ls) || (d <= ' ')));

              pscopy (t, s, i, k - i);
              err = evaluate (dico, q, t, 1);
              i = k - 1;
            }

          if (!err)
              ir = insertnumber (dico, ir, r, q);
          else
              message (dico, "Cannot compute &(expression)");
        }
    } 
                               /*while */
  Strdrem(q,t);
  return err;
}

static unsigned char
getword (char *s, char *t, int after, int *pi)
/* isolate a word from s after position "after". return i= last read+1 */
{
  int i = *pi;
  int ls;
  unsigned char key;
  i = after;
  ls = length (s);

 do
    {
      i++;
    } while (!((i >= ls) || alfa (s[i - 1])));

  scopy (t, "");

  while ((i <= ls) && (alfa (s[i - 1]) || num (s[i - 1])))
    {
      cadd (t, upcase (s[i - 1]));
      i++;
    }

  if (t[0])
      key = keyword (keys, t);
  else
      key = 0;

  *pi = i;
  return key;
}

static char
getexpress (char *s, char *t, int *pi)
/* returns expression-like string until next separator
 Input  i=position before expr, output  i=just after expr, on separator.
 returns tpe=='R' if ( numeric, 'S' if ( string only
*/
{
  int i = *pi;
  int ia, ls, level;
  char c, d, tpe;
  unsigned char comment = 0;
  ls = length (s);
  ia = i + 1;

  while ((ia < ls) && (s[ia - 1] <= ' '))
      ia++; /*white space ? */

  if (s[ia - 1] == '"')
    {                                /*string constant */
      ia++;
      i = ia;

      while ((i < ls) && (s[i - 1] != '"'))
        i++;

      tpe = 'S';

      do {
          i++;
        } while (!((i > ls) || (s[i - 1] > ' ')));
    }
  else
    {

      if (s[ia - 1] == '{')
        ia++;

      i = ia - 1;

      do {
          i++;

          if (i > ls)
              c = ';';
          else
              c = s[i - 1];

          if (c == '(')
            {                        /*sub-formula */
              level = 1;
              do {
                  i++;

                  if (i > ls)
                      d = Nul;
                  else
                      d = s[i - 1];

                  if (d == '(')
                      level++;
                  else if (d == ')')
                      level--;
                } while (!((i > ls) || ((d == ')') && (level <= 0))));
            }
          /* buggy? */ if ((c == '/') || (c == '-'))
              comment = (s[i] == c);
        } while (!((cpos (c, ",;)}") > 0) || comment));        /*legal separators */

      tpe = 'R';

    }

  pscopy (t, s, ia, i - ia);

  if (s[i - 1] == '}')
    i++;

  if (tpe == 'S')
    i++;                        /* beyond quote */

  *pi = i;
  return tpe;
}

unsigned char
nupa_assignment (tdico * dico, char *s, char mode)
/* is called for all 'Param' lines of the input file.
   is also called for the params: section of a subckt .
   mode='N' define new local variable, else global...
   bug: we cannot rely on the transformed line, must re-parse everything!
*/
{
/* s has the format: ident = expression; ident= expression ...  */
//  Strbig (Llen, t);
//  Strbig (Llen, u);
  int i, j, ls;
  unsigned char key;
  unsigned char error, err;
  char dtype;
  int wval = 0;
  double rval = 0.0;
  Strdbig (dynLlen, t, u);
  ls = length (s);
  error = 0;
  i = 0;
  j = spos ("//", s);                /* stop before comment if any */

  if (j > 0)
    ls = j - 1;
  /* bug: doesnt work. need to  revise getexpress ... !!! */
  i = 0;

  while ((i < ls) && (s[i] <= ' '))
    i++;

  if (s[i] == Intro)
    i++;

  if (s[i] == '.')
    {                                /* skip any dot keyword */
      while (s[i] > ' ')
        i++;
    }

  while ((i < ls) && (!error))
    {
      key = getword (s, t, i, &i);
      if ((t[0] == 0) || (key > 0))
          error = message (dico, " Identifier expected");

      if (!error)
        {                        /* assignment expressions */
          while ((i <= ls) && (s[i - 1] != '='))
            i++;

          if (i > ls)
              error = message (dico, " = sign expected .");

          dtype = getexpress (s, u, &i);

          if (dtype == 'R')
            {
              rval = formula (dico, u, &error);
              if (error)
                {
                  message (dico, " Formula() error.");
                  fprintf (stderr, "      %s\n", s);
                }
            }
          else if (dtype == 'S')
              wval = i;

          err = define (dico, t, mode /*was ' ' */ , dtype, rval, wval, NULL);
          error = error || err;
        }

      if ((i < ls) && (s[i - 1] != ';'))
          error = message (dico, " ; sign expected.");
      else
        /* i++ */;
    }
  Strdrem(t,u);
  return error;
}

unsigned char
nupa_subcktcall (tdico * dico, char *s, char *x, unsigned char err)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params
*/
{
  int n, m, i, j, k, g, h, narg = 0, ls, nest;
//  Strbig (Llen, t);
//  Strbig (Llen, u);
//  Strbig (Llen, v);
//  Strbig (Llen, idlist);
  Str (80, subname);
  char *buf, *token;
  unsigned char found;
  Strfbig (dynLlen, t, u, v, idlist);
  /*
     skip over instance name -- fixes bug where instance 'x1' is
     same name as subckt 'x1'
   */
  while (*x != ' ')
    x++;

  /***** first, analyze the subckt definition line */
  n = 0;                        /* number of parameters if any */
  ls = length (s);
  j = spos ("//", s);

  if (j > 0)
    pscopy_up (t, s, 1, j - 1);
  else
    scopy_up (t, s);

  j = spos ("SUBCKT", t);

  if (j > 0)
    {
      j = j + 6;                /* fetch its name */
      while ((j < ls) && (t[j] <= ' '))
        j++;

      while (t[j] != ' ')
        {
          cadd (subname, t[j]);
          j++;
        }
    }
  else
    err = message (dico, " ! a subckt line!");

  i = spos ("PARAMS:", t);

  if (i > 0)
    {
      pscopy (t, t, i + 7, length (t));
      while (j = cpos ('=', t), j > 0)
        {                        /* isolate idents to the left of =-signs */
          k = j - 2;
          while ((k >= 0) && (t[k] <= ' '))
            k--;

          h = k;

          while ((h >= 0) && alfanum (t[h]))
            h--;

          if (alfa (t[h + 1]) && (k > h))
            {                        /* we have some id */
              for (m = (h + 1); m <= k; m++)
                cadd (idlist, t[m]);

              sadd (idlist, "=$;");
              n++;
            }
          else
            message (dico, "identifier expected.");

          pscopy (t, t, j + 1, length (t));
        }
    }
  /***** next, analyze the circuit call line */
  if (!err)
    {
      narg = 0;
      j = spos ("//", x);

      if (j > 0)
        pscopy_up (t, x, 1, j - 1);
      else
        scopy_up (t, x);

      ls = length (t);

      buf = (char*) tmalloc(strlen(t) + 1);
      strcpy(buf, t);

      found = 0;
      token = strtok(buf, " ");       /* a bit more exact - but not sufficient everytime */
      j = j + strlen(token) + 1;
      if (strcmp(token, subname)) {
        while ((token = strtok(NULL, " "))) {
          if (!strcmp(token, subname)) {
            found = 1;
            break;
          }
          j = j + strlen(token) + 1;
        }
      }
      free(buf);

      /*  make sure that subname followed by space */
      if (found)
        {
          j = j + length (subname) + 1;        /* 1st position of arglist: j */

          while ((j < ls) && ((t[j] <= ' ') || (t[j] == ',')))
            j++;

          while (j < ls)
            {                        /* try to fetch valid arguments */
              k = j;
              scopy (u, "");
              if ((t[k] == Intro))
                {                /* handle historical syntax... */
                  if (alfa (t[k + 1]))
                      k++;
                  else if (t[k + 1] == '(')
                    {                /* transform to braces... */
                      k++;
                      t[k] = '{';
                      g = k;
                      nest = 1;
                      while ((nest > 0) && (g < ls))
                        {
                          g++;
                          if (t[g] == '(')
                            nest++;
                          else if (t[g] == ')')
                            nest--;
                        }

                      if ((g < ls) && (nest == 0))
                        t[g] = '}';
                    }
                }

              if (alfanum (t[k]) || t[k] == '.')
                {                /* number, identifier */
                  h = k;
                  while (t[k] > ' ')
                    k++;

                  pscopy (u, t, h + 1, k - h);
                  j = k;
                }
              else if (t[k] == '{')
                {
                  getexpress (t, u, &j);
                  j--; /* confusion: j was in Turbo Pascal convention */ ;
                }
              else
                {
                  j++;
                  if (t[k] > ' ')
                    {
                      scopy (v, "Subckt call, symbol ");
                      cadd (v, t[k]);
                      sadd (v, " not understood");
                      message (dico, v);
                    }
                }

              if (u[0])
                {
                  narg++;
                  k = cpos ('$', idlist);

                  if (k > 0)
                    {                /* replace dollar with expression string u */
                      pscopy (v, idlist, 1, k - 1);
                      sadd (v, u);
                      pscopy (u, idlist, k + 1, length (idlist));
                      scopy (idlist, v);
                      sadd (idlist, u);
                    }
                }
            }
        }
      else
          message (dico, "Cannot find called subcircuit");
    }
  /***** finally, execute the multi-assignment line */
  dicostack (dico, Push);        /* create local symbol scope */
  if (narg != n)
    {
      scopy (t, " Mismatch: ");
      nadd (t, n);
      sadd (t, "  formal but ");
      nadd (t, narg);
      sadd (t, " actual params.");
      err = message (dico, t);
      message (dico, idlist);
      /* ;} else { debugwarn(dico, idlist) */ ;
    }
  err = nupa_assignment (dico, idlist, 'N');
  Strfrem(t,u,v,idlist);
  return err;
}

void
nupa_subcktexit (tdico * dico)
{
  dicostack (dico, Pop);
}
