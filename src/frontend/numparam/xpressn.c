/*       xpressn.c                Copyright (C)  2002    Georg Post
 
   This file is part of Numparam, see:  readme.txt
   Free software under the terms of the GNU Lesser General Public License
   $Id$
 */

#include <stdio.h>                /* for function message() only. */

#include "general.h"
#include "numparam.h"
#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "../frontend/variable.h"
#include "compatmode.h"

/* random numbers in /maths/misc/randnumb.c */
extern double gauss();
extern COMPATMODE_T ngspice_compat_mode(void) ;

/************ keywords ************/

/* SJB - 150 chars is ample for this - see initkeys() */
static SPICE_DSTRING keyS ;      /* all my keywords */
static SPICE_DSTRING fmathS ;     /* all math functions */

extern char *nupa_inst_name; /* see spicenum.c */
extern long dynsubst;        /* see inpcom.c */
extern unsigned int dynLlen;

#define MAX_STRING_INSERT 17 /* max. string length to be inserted and replaced */
#define ACT_CHARACTS 15      /* actual string length to be inserted and replaced */
                             /* was 10, needs to be less or equal to MAX_STRING_INSERT - 2 */

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
  spice_dstring_init(&keyS) ;
  scopy_up (&keyS,
            "and or not div mod if else end while macro funct defined"
            " include for to downto is var");
  scopy_up (&fmathS,
            "sqr sqrt sin cos exp ln arctan abs pow pwr max min int log sinh cosh tanh ternary_fcn v agauss sgn");
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
    case 21:
        if (x>0) y=1.;
        else if (x == 0) y=0.;
        else y = -1.; 
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
   char *srcfile ;			/* src file name */
   SPICE_DSTRING t ;			/* temp dstring */	

   spice_dstring_init(&t) ;
   dic->errcount++;
   srcfile = spice_dstring_value( &(dic->srcfile) ) ;
   if ((srcfile != NULL) && srcfile[0])
   {
      scopyd(&t, &(dic->srcfile)) ;
      cadd (&t, ':');
   }
   if (dic->srcline >= 0)
   {
      sadd (&t, "Original line no.: ");
      nadd (&t, dic->oldline);
      sadd (&t, ", new internal line no.: ");
      nadd (&t, dic->srcline);
      sadd (&t, ":\n");
   }
   sadd (&t, s);
   cadd (&t, '\n');
   fputs ( spice_dstring_value(&t), stderr);
   spice_dstring_free(&t) ;

   return 1 /* error! */ ;
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
   COMPATMODE_T compat_mode;

   dico->nbd = 0;
   spice_dstring_init( &(dico->option) ) ;
   spice_dstring_init( &(dico->srcfile) ) ;
   
   dico->srcline = -1;
   dico->errcount = 0;

   dico->symbol_table = nghash_init( NGHASH_MIN_SIZE ) ;
   nghash_unique( dico->symbol_table, FALSE ) ;
   spice_dstring_init( &(dico->lookup_buf) ) ;
  
   dico->tos = 0;
   dico->stack[dico->tos] = 0;        /* global data beneath */
   initkeys ();

   compat_mode = ngspice_compat_mode() ;

   if( compat_mode == COMPATMODE_HSPICE )
     dico->hspice_compatibility = 1 ;
   else 
     dico->hspice_compatibility = 0 ;
}

static void dico_free_entry( entry *entry_p )
{
    if( entry_p->symbol ){
      txfree(entry_p->symbol ) ;
    }
    txfree(entry_p) ;
} /* end dico_free_entry() */

static
entry **dico_rebuild_symbol_array( tdico * dico, int *num_entries_ret )
{
    int i ;				/* counter */
    int size ;				/* number of entries in symbol table */
    entry *entry_p ;			/* current entry */
   NGHASHITER iter ;			/* hash iterator - thread safe */

    size = *num_entries_ret = nghash_get_size( dico->symbol_table ) ;
    if( dico->num_symbols == size ){
      /* no work to do up to date */
      return( dico->symbol_array ) ;
    }
    if( size <= 0 ){
      size = 1 ;
    }
    dico->symbol_array = trealloc( dico->symbol_array, (size+1) * sizeof(entry *) ) ;
    i = 0 ;
    for (entry_p = nghash_enumerateRE(dico->symbol_table,NGHASH_FIRST(&iter)) ;
	 entry_p ;
	 entry_p = nghash_enumerateRE(dico->symbol_table,&iter)){
      dico->symbol_array[i++] = entry_p ;
    }
    dico->num_symbols = *num_entries_ret ;
    return dico->symbol_array ;
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
   char *param_p, *inst_name;
   int i, current_stack_size, old_stack_size;
   int num_entries ;				/* number of entries */
   entry **entry_array ;			/* entry array */
   entry *entry_p ;				/* current entry */
   SPICE_DSTRING param_name ;

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
	 spice_dstring_init(&param_name) ;
	 entry_array = dico_rebuild_symbol_array( dico, &num_entries ) ;

         for (i = old_stack_size + 1; i <= current_stack_size; i++)
         {
	    spice_dstring_reinit(&param_name) ;
	    if( i < num_entries ){
	      entry_p = entry_array[i] ;
	      param_p = spice_dstring_print( &param_name,  "%s.%s", 
						inst_name, 
						entry_p->symbol ) ;
	      nupa_add_inst_param (param_p, entry_p->vl); 
/*	      nghash_deleteItem( dico->symbol_table, entry_p->symbol, entry_p ) ;
	      dico_free_entry( entry_p ) ; */
	    }
         }
         tfree (inst_name);
	 spice_dstring_free(&param_name) ;

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

/* FIXME : WPS this should be a hash table */
static entry *
entrynb (tdico * d, char *s)
/* symbol lookup from end to start,  for stacked local symbols .*/
/* bug: sometimes we need access to same-name symbol, at lower level? */
{
   entry *entry_p ;			/* search hash table */

   entry_p = nghash_find( d->symbol_table, s ) ;
   return( entry_p ) ;
}

char
getidtype (tdico * d, char *s)
/* test if identifier s is known. Answer its type, or '?' if not in table */
{
   entry *entry_p ;		  /* hash table entry */
   char itp = '?';                /* assume unknown */

   entry_p = entrynb (d, s) ;
   if( entry_p ){
      itp = entry_p->tp ;
   }
   return (itp) ;
}

static double
fetchnumentry (tdico * dico, char *t, unsigned char *perr)
{
   unsigned char err = *perr;
   double u;
   entry *entry_p ;			/* hash table entry */
   SPICE_DSTRING s ;			/* dynamic string */

   spice_dstring_init(&s) ;
   entry_p = entrynb (dico, t);        /* no keyword */
   /*dbg -- if ( k<=0 ) { ws("Dico num lookup fails. ") ;} */

   while ( entry_p && (entry_p->tp == 'P') ){
      entry_p = entry_p->pointer ;
   }

   if ( entry_p )
      if (entry_p->tp != 'R')
         entry_p = NULL ;

   if ( entry_p )
      u = entry_p->vl ;
   else
   {
      u = 0.0;
      scopys(&s, "Undefined number [") ;
      sadd (&s, t);
      cadd (&s, ']');
      err = message (dico, spice_dstring_value(&s) ) ;
   }

   *perr = err;

   spice_dstring_free(&s) ;

   return u;
}

/*******  writing dictionary entries *********/

entry *
attrib (tdico * dico, char *t, char op)
{
/* seek or attribute dico entry number for string t.
   Option  op='N' : force a new entry, if tos>level and old is  valid.
*/
   int i;
   entry *entry_p ;			/* symbol table entry */

   entry_p = nghash_find( dico->symbol_table, t ) ;
   if ( entry_p && (op == 'N')
     && ( entry_p->level < dico->tos) && ( entry_p->tp != '?'))
   {
      entry_p = NULL ;
   }

   if (!(entry_p))
   {
      dico->nbd++;
      i = dico->nbd;
      entry_p = tmalloc( sizeof(entry) ) ;
      entry_p->symbol = strdup( t ) ;
      entry_p->tp = '?';        /* signal Unknown */
      entry_p->level = dico->tos ;
      nghash_insert( dico->symbol_table, t, entry_p ) ;
   }
   return entry_p ;
}

static unsigned char
define (tdico * dico, 
        char *t,        /* identifier to define */
        char op,        /* option */
        char tpe,       /* type marker */
        double z,       /* float value if any */
        int w,          /* integer value if any */
	entry *pval,    /* pointer value if any */
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
   char c;
   unsigned char err, warn;
   entry *entry_p ;			/* spice table entry */
   SPICE_DSTRING vartemp ;		/* vairable temp */

   spice_dstring_init(&vartemp) ;
   entry_p = attrib (dico, t, op);
   err = 0;
   if (!(entry_p))
      err = message (dico, " Symbol table overflow");
   else
   {
      if ( entry_p->tp == 'P')
         entry_p = entry_p->pointer ; /* pointer indirection */

      if (entry_p)
         c = entry_p->tp ;
      else
         c = ' ';

      if ((c == 'R') || (c == 'S') || (c == '?'))
      {
         entry_p->vl = z;
         entry_p->tp = tpe;
         entry_p->ivl = w ;
         entry_p->sbbase = base ;
         /* if ( (c !='?') && (i<= dico->stack[dico->tos]) ) {  */
         if (c == '?')
            entry_p->level = dico->tos; /* promote! */

         if ( entry_p->level < dico->tos)
         {
            /* warn about re-write to a global scope! */
            scopys(&vartemp, t) ;
            cadd (&vartemp, ':');
            nadd (&vartemp, entry_p->level);
            sadd (&vartemp, " overwritten.");
            warn = message (dico, spice_dstring_value(&vartemp));
         }
      }
      else
      {
 	 scopys( &vartemp, t) ;
         sadd ( &vartemp, ": cannot redefine");
         /* suppress error message, resulting from multiple definition of
         symbols (devices) in .model lines with same name, but in different subcircuits.
         Subcircuit expansion is o.k., we have to deal with this numparam
         behaviour later. (H. Vogt 090426) 
         */
         /*err = message (dico, v);*/
      }
   }
   spice_dstring_free(&vartemp) ;
   return err;
}

unsigned char
defsubckt (tdico * dico, char *s, int w, char categ)
/* called on 1st pass of spice source code,
   to enter subcircuit (categ=U) and model (categ=O) names
*/
{
   SPICE_DSTRING ustr ;			/* temp user string */
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
      spice_dstring_init(&ustr) ;
      pscopy_up ( &ustr, s, i, j - i) ;
      err = define (dico, spice_dstring_value(&ustr), ' ', categ, 0.0, w, NULL, NULL);
      spice_dstring_free(&ustr) ;
   }
   else
      err = message (dico, "Subcircuit or Model without name.");

   return err;
}

int
findsubckt (tdico * dico, char *s, SPICE_DSTRINGPTR subname)
/* input: s is a subcircuit invocation line.
   returns 0 if not found, else the stored definition line number value
   and the name in string subname  */
{
   entry *entry_p ;		      /* symbol table entry */
   SPICE_DSTRING ustr ;		      /* u= subckt name is last token in string s */
   int j, k;
   int line ;			      /* stored line number */
   spice_dstring_init(&ustr) ;
   k = length (s);

   while ((k >= 0) && (s[k] <= ' '))
      k--;

   j = k;

   while ((k >= 0) && (s[k] > ' '))
      k--;

   pscopy_up ( &ustr, s, k + 1, j - k) ;
   entry_p = entrynb (dico, spice_dstring_value(&ustr) ) ;

   if ((entry_p) && ( entry_p->tp == 'U'))
   {
      line = entry_p->ivl;
      scopyd ( subname, &ustr ) ;
   }
   else
   {
      line = 0;
      spice_dstring_reinit(subname);
      message (dico, "Cannot find subcircuit.");
   }

   return line ;
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
        {                        /* old item! */
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
keyword ( SPICE_DSTRINGPTR keys_p, SPICE_DSTRINGPTR tstr_p)
{
/* return 0 if t not found in list keys, else the ordinal number */
   unsigned char i, j, k;
   int lt, lk;
   unsigned char ok;
   char *t ;
   char *keys ;
   lt = spice_dstring_length(tstr_p) ;
   t = spice_dstring_value(tstr_p) ;
   lk = spice_dstring_length (keys_p);
   keys = spice_dstring_value(keys_p);
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

      if (!ok && (k < lk))    /* skip to next item */
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
  SPICE_DSTRING t ;
  unsigned char isunit;
  isunit = 1;
  spice_dstring_init(&t) ;

  pscopy (&t, s, 0, 3);

  if (steq ( spice_dstring_value(&t), "MEG"))
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

  spice_dstring_free(&t) ;

  return x;
}

static int
fetchid (char *s, SPICE_DSTRINGPTR t, int ls, int i)
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

  spice_dstring_reinit(t) ;
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
  return i /* return updated i */ ;
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
  SPICE_DSTRING t ;

  ls = length (s);
  spice_dstring_init(&t) ;
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
      i = fetchid (s, &t, ls, i);
      i--;
      if (entrynb(d, spice_dstring_value(&t)))
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

/* keep pointer on last closing ")" */

  *perror = error;
  *pi = i;
  spice_dstring_free(&t) ;
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
  char *t ;
  SPICE_DSTRING tstr ;
  SPICE_DSTRING vstr ;
  double u;
  spice_dstring_init(&tstr) ;
  spice_dstring_init(&vstr) ;
  k = i;

  do {
      k++;
      if (k > ls)
          d = (char)(0);
      else
          d = s[k - 1];
    } while (!(!((d == '.') || ((d >= '0') && (d <= '9')))));

  if ((d == 'e') || (d == 'E'))
    {                                /* exponent follows */
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

  pscopy (&tstr, s, i-1, k - i) ;

  t = spice_dstring_value(&tstr) ;
  if (t[0] == '.')
      cins ( &tstr, '0');
  else if (t[length (t) - 1] == '.')
      cadd (&tstr, '0');

  t = spice_dstring_value(&tstr) ;
  u = rval (t, &err);  /* extract real value from string here */

  if (err != 0)
    {
      scopys(&vstr, "Number format error: ") ;
      sadd (&vstr, t);
      error = message (dico, spice_dstring_value(&vstr)) ;
    }
  else
    {
      spice_dstring_reinit(&tstr);
      while (alfa (d))
        {
          cadd (&tstr, upcase (d));
          k++;

          if (k > ls)
              d = Nul;
          else
              d = s[k - 1];
        }

      t = spice_dstring_value(&tstr) ;
      u = parseunit (u, t);
    }

  i = k - 1;
  *perror = error;
  *pi = i;
  spice_dstring_free(&tstr) ;
  spice_dstring_free(&vstr) ;
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
  SPICE_DSTRING vstr ;
  c = s[i - 1];
  spice_dstring_init(&vstr) ;

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
      state = 2;                /* pending operator */
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
          spice_dstring_append(&vstr, "Syntax error: letter [", -1 );
          cadd (&vstr, c);
          cadd (&vstr, ']');
          error = message (dico, spice_dstring_value(&vstr) );
        }
    }
  *pi = i;
  *pstate = state;
  *plevel = level;
  *perror = error;
  spice_dstring_free(&vstr) ;
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
    {                                /* & | ~ DIV MOD  Defined */
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
    }                                /* case */

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
      x = y; /* problem here: do type conversions ?! */ ;
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
    case '^':                        /* power */
      t = absf (x);
      if (t < epsi)
        x = z;
      else
        x = exp (y * ln (t));
      break;
    case '&':                        /* && */
      if (y < x)
        x = y; /*=Min*/ ;
      break;
    case '|':                        /* || */
      if (y > x)
        x = y; /*=Max*/ ;
      break;
    case '=':
      if (x == y)
        x = u;
      else
        x = z;
      break;
    case '#':                        /* <> */
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
    case 'G':                        /* >= */
      if (x >= y)
        x = u;
      else
        x = z;
      break;
    case 'L':                        /* <= */
      if (x <= y)
        x = u;
      else
        x = z;
      break;
    case '!':                        /* ! */
      if (y == z)
        x = u;
      else
        x = z;
      break;
    case '%':                        /* % */
      t = np_trunc (x / y);
      x = x - y * t;
      break;
    case '\\':                        /* / */
      x = np_trunc (absf (x / y));
      break;
    }                                /* case */
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
  typedef enum {nprece=9} _nnprece;                /* maximal nb of precedence levels */
  unsigned char error = *perror;
  unsigned char negate = 0;
  unsigned char state, oldstate, topop, ustack, level, kw, fu;
  double u = 0.0, v, w = 0.0;
  double accu[nprece + 1];
  char oper[nprece + 1];
  char uop[nprece + 1];
  int i, k, ls, natom, arg2, arg3;
  char c, d;
  unsigned char ok;
  SPICE_DSTRING tstr ;

  spice_dstring_init(&tstr) ;
  for (i = 0; i <= nprece; i++)
    {
      accu[i] = 0.0;
      oper[i] = ' ';
    }
  i = 0;
  ls = length (s);

  while ((ls > 0) && (s[ls - 1] <= ' '))
    ls--;                        /* clean s */

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
        {                        /* sub-formula or math function */
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
                    arg3 = k;        /* kludge for more than 2 args (ternary expression) */
                } /* comma list? */ ;
            }
          while (!((k > ls) || ((d == ')') && (level <= 0))));

          if (k > ls)
            {
              error = message (dico, "Closing \")\" not found.");
              natom++; /* shut up other error message */ ;
            }
          else
            {
              if (arg2 > i)
                {
                  pscopy (&tstr, s, i, arg2 - i - 1);
                  v = formula (dico, spice_dstring_value(&tstr), &error);
                  i = arg2;
                }
              if (arg3 > i)
                {
                  pscopy (&tstr, s, i, arg3 - i - 1);
                  w = formula (dico, spice_dstring_value(&tstr), &error);
                  i = arg3;
                }
              pscopy (&tstr, s, i, k - i - 1);
              u = formula (dico, spice_dstring_value(&tstr), &error);
              state = 1;        /* atom */
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
          i = fetchid (s, &tstr, ls, i);   /* user id, but sort out keywords */
          state = 1;
          i--;
          kw = keyword (&keyS, &tstr);            /* debug ws('[',kw,']'); */
          if (kw == 0)
            {
              fu = keyword (&fmathS, &tstr);      /* numeric function? */
              if (fu == 0)
                  u = fetchnumentry (dico, spice_dstring_value(&tstr), &error);
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
        {                        /* push unary operator */
          ustack++;
          uop[ustack] = c;
        }
      else if (state == 1)
        {                        /* atom pending */
          natom++;
          if (i >= ls)
            {
              state = 4;
              level = topop;
            }                        /* close all ops below */
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
              oper[k] = ' '; /* reset intermediates */ ;
            }
          oper[level] = c;

          if (level > topop)
              topop = level;
        }
      if ((state > 0))
        {
          oldstate = state;
        }
    } /* while */ ;
  if ((natom == 0) || (oldstate != 4))
    {
      spice_dstring_reinit(&tstr) ;
      sadd( &tstr, " Expression err: ");
      sadd (&tstr, s);
      error = message (dico, spice_dstring_value(&tstr));
    }

  if (negate == 1)
    {
      error =
        message (dico,
                 " Problem with formula eval -- wrongly determined negation!");
    }

  *perror = error;

  spice_dstring_free(&tstr) ;

  if (error)
      return 1.0;
  else
      return accu[topop];
}                                /* formula */

static char
fmttype (double x)
{
/* I=integer, P=fixedpoint, F=floatpoint */
/* find out the "natural" type of format for number x */
  double ax, dx;
  int rx;
  unsigned char isint, astronomic;
  ax = absf (x);
  isint = 0;
  astronomic = 0;

  if (ax < 1e-39)                    /* smaller then 1e-39 is 0 */
      isint = 1;                     /* and seen as an integer */
  else if (ax < 64000)
    {                                /* detect integers */
      rx = np_round (x);
      dx = (x - rx) / ax;
      isint = (absf (dx) < 1e-06);
    }

  if (!isint)
      astronomic = (ax >= 1e+06) || (ax < 0.01); /* astronomic for 10 digits */

  if (isint)
      return 'I';
  else if (astronomic)
      return 'F';
  else
      return 'P';
}

static unsigned char
evaluate (tdico * dico, SPICE_DSTRINGPTR qstr_p, char *t, unsigned char mode)
{
/* transform t to result q. mode 0: expression, mode 1: simple variable */
  double u = 0.0;
  int j, lq;
  char dt, fmt;
  entry *entry_p ;
  unsigned char numeric, done, nolookup;
  unsigned char err;
  SPICE_DSTRING vstr ;

  spice_dstring_init(&vstr) ;
  spice_dstring_reinit(qstr_p) ;
  numeric = 0;
  err = 0;

  if (mode == 1)
    {                                /* string? */
      stupcase (t);
      entry_p = entrynb (dico, t);
      nolookup = (!(entry_p));
      while ((entry_p) && (entry_p->tp == 'P')){
          entry_p = entry_p->pointer ;		/* follow pointer chain */
      }

      /* pointer chain */
      if (entry_p)
          dt = entry_p->tp;
      else
          dt = ' ';

      /* data type: Real or String */
      if (dt == 'R')
        {
          u = entry_p->vl;
          numeric = 1;
        }
      else if (dt == 'S')
        {                        /* suppose source text "..." at */
          j = entry_p->ivl;
          lq = 0;
          do {
              j++;
              lq++;
              dt = /* ibf->bf[j]; */ entry_p->sbbase[j];

              if (cpos ('3', spice_dstring_value(&dico->option)) <= 0)
                  dt = upcase (dt); /* spice-2 */

              done = (dt == '\"') || (dt < ' ') || (lq > 99);

              if (!done)
                  cadd (qstr_p, dt);
            } while (!(done));
        }

      if (!(entry_p))
        {
          spice_dstring_reinit(&vstr) ;
          cadd (&vstr, '\"');
          sadd (&vstr, t);
          sadd (&vstr, "\" not evaluated. ");

          if (nolookup)
              sadd (&vstr, "Lookup failure.");

          err = message (dico, spice_dstring_value(&vstr));
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
          stri (np_round (u), qstr_p);
      else
        {
          strf (u, 17, 10, qstr_p);
        }
    }
  spice_dstring_free(&vstr) ;
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
    {                                /* special Comment **&AC #... */
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
	      if( dico->hspice_compatibility && (strcasecmp(t,"LAST")==0) ) {
		strcpy(q,"last") ;
		err=0;
	      } else 
		err = evaluate (dico, q, t, 0);
            }
          i = k;
          if (!err)
            {                        /* insert number */
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
            {                        /* sub-formula */
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
            {                        /* simple identifier may also be string */
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
            {                        /* insert the number */
              sadd (r, q);
            }
          else
            {
              message (dico, s);
            }
        }
      else if (c == Nodekey)
        {                        /* follows: a node keyword */
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
          cadd (r, c); /* c<>Intro */ ;
        }
    }                                /* while */
  return err;
}
#endif

/********* interface functions for spice3f5 extension ***********/

static void
compactfloatnb (SPICE_DSTRINGPTR vstr_p)
/* try to squeeze a floating pt format to ACT_CHARACTS characters */
/* erase superfluous 000 digit streams before E */
/* bug: truncating, no rounding */
{
  int n, k, m, lex, lem;
  char *expov ;
  char *expnv ;
  char *v_p ;
  SPICE_DSTRING expo_str ;
  SPICE_DSTRING expn_str ;

  spice_dstring_init(&expo_str) ;
  spice_dstring_init(&expn_str) ;
  n = cpos ('E', spice_dstring_value(vstr_p)) ; /* if too long, try to delete digits */
  if (n<0) n = cpos ('e', spice_dstring_value(vstr_p));

  if (n >= 0) {
    pscopy (&expo_str, spice_dstring_value(vstr_p), n, 
	    spice_dstring_length(vstr_p));
    lex = spice_dstring_length (&expo_str) ;
    if (lex > 4) {            /* exponent only 2 digits */
      pscopy (&expn_str, spice_dstring_value(&expo_str), 1, 4);
      expnv = spice_dstring_value(&expn_str) ;
      if (atoi(expnv) < -99){
	spice_dstring_reinit(&expo_str) ;
	sadd(&expo_str, "e-099"); /* brutal */
      }
      if (atoi(expnv) > +99){
	spice_dstring_reinit(&expo_str) ;
	sadd(&expo_str, "e+099");
      }
      expov = spice_dstring_value(&expo_str) ;
      expov[2] = expov[3];
      expov[3] = expov[4];
      expov[4] = '\0';
      spice_dstring_setlength(&expo_str,4) ;
      lex = 4;
    }
    k = n ;                /* mantissa is 0...k */

    m = MAX_STRING_INSERT;
    v_p = spice_dstring_value(vstr_p) ;
    while (v_p[m] != ' ')
      m--;
    m++;
    while ((v_p[k] == '0') && (v_p[k - 1] == '0'))
      k--;

    lem = k - m;

    if ((lem + lex) > ACT_CHARACTS)
      lem = ACT_CHARACTS - lex;

    pscopy (vstr_p, spice_dstring_value(vstr_p), m, lem);
    if (cpos('.', spice_dstring_value(vstr_p)) >= 0) {
      while (lem < ACT_CHARACTS - 4) {
        cadd(vstr_p, '0');
        lem++;
      }
    } else {
      cadd(vstr_p, '.');
      lem++;
      while (lem < ACT_CHARACTS - 4) {
        cadd(vstr_p, '0');
        lem++;
      }
    }
    sadd (vstr_p, spice_dstring_value(&expo_str) );
  } else {
    m = 0;
    v_p = spice_dstring_value(vstr_p) ;
    while (v_p[m] == ' ')
      m++;

    lem = spice_dstring_length(vstr_p) - m;
    if (lem > ACT_CHARACTS) lem = ACT_CHARACTS;
    pscopy (vstr_p, spice_dstring_value(vstr_p), m, lem);
  }
}

static int
insertnumber (tdico * dico, int i, char *s, SPICE_DSTRINGPTR ustr_p)
/* insert u in string s in place of the next placeholder number */
{
  SPICE_DSTRING vstr ;			/* dynamic string */
  SPICE_DSTRING mstr ;			/* dynamic string */
  char *v_p ;				/* value of vstr dyna string */
  unsigned char found;
  int ls, k;
  long long accu;
  ls = length (s);

  spice_dstring_init(&vstr) ;
  spice_dstring_init(&mstr) ;
  scopyd (&vstr, ustr_p) ;
  compactfloatnb (&vstr) ;

  while ( spice_dstring_length (&vstr) < MAX_STRING_INSERT)
      cadd (&vstr, ' ');

  if ( spice_dstring_length (&vstr) > MAX_STRING_INSERT)
    {
      spice_dstring_append( &mstr, " insertnumber fails: ", -1);
      sadd (&mstr, spice_dstring_value(ustr_p));
      message (dico, spice_dstring_value(&mstr)) ;
    }

  found = 0;

  while ((!found) && (i < ls))
    {
      found = (s[i] == '1');
      k = 0;
      accu = 0;

      while (found && (k < 15))
        {                        	/* parse a 15-digit number */
          found = num (s[i + k]);

          if (found)
              accu = 10 * accu + s[i + k] - '0';

          k++;
        }

      if (found)
        {
	  accu = accu - 100000000000000LL;	/* plausibility test */

          found = (accu > 0) && (accu < dynsubst + 1); /* dynsubst numbers have been allocated */
        }
      i++;
    }

  if (found)
    {                                /* substitute at i-1 ongoing */
      i--;
      v_p = spice_dstring_value(&vstr) ;
      for (k = 0; k < ACT_CHARACTS; k++)
        s[i + k] = v_p[k];

      i = i + MAX_STRING_INSERT;

    }
  else
    {
      i = ls;
      fprintf (stderr, "xpressn.c--insertnumber:  i=%d  s=%s  u=%s\n", i, s,
               spice_dstring_value(ustr_p)) ;
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
  SPICE_DSTRING qstr ;			/* temp result dynamic string */
  SPICE_DSTRING tstr ;			/* temp dynamic string */

  spice_dstring_init(&qstr) ;
  spice_dstring_init(&tstr) ;
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
              pscopy (&tstr, s, i , k - i - 1);
              /* exeption made for .meas */
              if( strcasecmp( spice_dstring_value(&tstr),"LAST")==0) {
                 spice_dstring_reinit(&qstr) ;
		 sadd(&qstr,"last") ;
                 err=0;
              } else 
		err = evaluate (dico, &qstr, spice_dstring_value(&tstr), 0);
          }

          i = k;
          if (!err)
              ir = insertnumber (dico, ir, r, &qstr) ;
          else
              err = message (dico, "Cannot compute substitute");
        }
      else if (c == Intro)
        {
        /* skip "&&" which may occur in B source */
            if ((i + 1 < ls) && (s[i] == Intro)) {
                i++;
                continue;
            }

        i++;
          while ((i < ls) && (s[i - 1] <= ' '))
            i++;

          k = i;

          if (s[k - 1] == '(')
            {                        /* sub-formula */
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
                  pscopy (&tstr, s, i, k - i - 1);
                  err = evaluate (dico, &qstr, spice_dstring_value(&tstr), 0);
                }
              i = k;
            }
          else
            {                        /* simple identifier may also be string? */
              do {
                  k++;
                  if (k > ls)
                      d = (char)(0);
                  else
                      d = s[k - 1];
                } while (!((k > ls) || (d <= ' ')));

              pscopy (&tstr, s, i-1, k - i);
              err = evaluate (dico, &qstr, spice_dstring_value(&tstr), 1);
              i = k - 1;
            }

          if (!err)
              ir = insertnumber (dico, ir, r, &qstr);
          else
              message (dico, "Cannot compute &(expression)");
        }
    } 
                               /* while */
  spice_dstring_free(&qstr) ;
  spice_dstring_free(&tstr) ;
  return err;
}

static unsigned char
getword (char *s, SPICE_DSTRINGPTR tstr_p, int after, int *pi)
/* isolate a word from s after position "after". return i= last read+1 */
{
  int i = *pi;
  int ls;
  unsigned char key;
  char *t_p ;
  i = after;
  ls = length (s);

 do
    {
      i++;
    } while (!((i >= ls) || alfa (s[i - 1])));

  spice_dstring_reinit(tstr_p) ;

  while ((i <= ls) && (alfa (s[i - 1]) || num (s[i - 1])))
    {
      cadd (tstr_p, upcase (s[i - 1]));
      i++;
    }

  t_p = spice_dstring_value(tstr_p) ;
  if (t_p[0])
      key = keyword (&keyS, tstr_p);
  else
      key = 0;

  *pi = i;
  return key;
}

static char
getexpress (char *s, SPICE_DSTRINGPTR tstr_p, int *pi)
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
    {                                /* string constant */
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
            {                        /* sub-formula */
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
        } while (!((cpos (c, ",;)}") >= 0) || comment));        /* legal separators */

      tpe = 'R';

    }

  pscopy (tstr_p, s, ia-1, i - ia);

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
  int i, j, ls;
  unsigned char key;
  unsigned char error, err;
  char dtype;
  int wval = 0;
  double rval = 0.0;
  char *t_p ;					/* dstring contents value */
  SPICE_DSTRING tstr ;				/* temporary dstring */
  SPICE_DSTRING ustr ;				/* temporary dstring */
  spice_dstring_init(&tstr) ;
  spice_dstring_init(&ustr) ;
  ls = length (s);
  error = 0;
  i = 0;
  j = spos_ ("//", s);                /* stop before comment if any */

  if (j >= 0)
    ls = j ;
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
      key = getword (s, &tstr, i, &i);
      t_p = spice_dstring_value(&tstr) ;
      if ((t_p[0] == 0) || (key > 0))
          error = message (dico, " Identifier expected");

      if (!error)
        {                        /* assignment expressions */
          while ((i <= ls) && (s[i - 1] != '='))
            i++;

          if (i > ls)
              error = message (dico, " = sign expected .");

          dtype = getexpress (s, &ustr, &i);

          if (dtype == 'R')
            {
              rval = formula (dico, spice_dstring_value(&ustr), &error);
              if (error)
                {
                  message (dico, " Formula() error.");
                  fprintf (stderr, "      %s\n", s);
                }
            }
          else if (dtype == 'S')
              wval = i;

          err = define (dico, spice_dstring_value(&tstr), mode /* was ' ' */ , 
	                dtype, rval, wval, NULL, NULL);
          error = error || err;
        }

      if ((i < ls) && (s[i - 1] != ';'))
          error = message (dico, " ; sign expected.");
      else
        /* i++ */;
    }
  spice_dstring_free(&tstr) ;
  spice_dstring_free(&ustr) ;
  return error;
}

unsigned char
nupa_subcktcall (tdico * dico, char *s, char *x, unsigned char err)
/* s= a subckt define line, with formal params.
   x= a matching subckt call line, with actual params
*/
{
  int n, m, i, j, k, g, h, narg = 0, ls, nest;
  SPICE_DSTRING subname ;
  SPICE_DSTRING tstr ;
  SPICE_DSTRING ustr ;
  SPICE_DSTRING vstr ;
  SPICE_DSTRING idlist ;
  SPICE_DSTRING parsebuf ;
  char *buf, *token;
  char *t_p ;
  char *u_p ;
  unsigned char found;
  spice_dstring_init(&subname) ;
  spice_dstring_init(&tstr) ;
  spice_dstring_init(&ustr) ;
  spice_dstring_init(&vstr) ;
  spice_dstring_init(&idlist) ;
  /*
     skip over instance name -- fixes bug where instance 'x1' is
     same name as subckt 'x1'
   */
  while (*x != ' ')
    x++;

  /***** first, analyze the subckt definition line */
  n = 0;                        /* number of parameters if any */
  ls = length (s);
  j = spos_ ("//", s);

  if (j >= 0)
    pscopy_up (&tstr, s, 0, j );
  else
    scopy_up (&tstr, s);

  j = spos_ ("SUBCKT", spice_dstring_value(&tstr) ) ;

  if (j >= 0)
    {
      j = j + 6;                /* fetch its name - skip subckt */
      t_p = spice_dstring_value(&tstr) ;
      while ((j < ls) && (t_p[j] <= ' '))
        j++;

      while (t_p[j] != ' ')
        {
          cadd (&subname, t_p[j]);
          j++;
        }
    }
  else
    err = message (dico, " ! a subckt line!");

  i = spos_ ("PARAMS:", spice_dstring_value(&tstr));

  if (i >= 0)
    {
      pscopy (&tstr, spice_dstring_value(&tstr), i + 7, spice_dstring_length (&tstr));
      while (j = cpos ('=', spice_dstring_value(&tstr)), j >= 0)
        {                        /* isolate idents to the left of =-signs */
          k = j - 1;
	  t_p = spice_dstring_value(&tstr) ;
          while ((k >= 0) && (t_p[k] <= ' '))
            k--;

          h = k;

          while ((h >= 0) && alfanum (t_p[h]))
            h--;

          if (alfa (t_p[h + 1]) && (k > h))
            {                        /* we have some id */
              for (m = (h + 1); m <= k; m++)
                cadd (&idlist, t_p[m]);

              sadd (&idlist, "=$;");
              n++;
            }
          else
            message (dico, "identifier expected.");

	  /* It is j+1 to skip over the '=' */
          pscopy (&tstr, spice_dstring_value(&tstr), j+1, spice_dstring_length (&tstr));
        }
    }
  /***** next, analyze the circuit call line */
  if (!err)
    {
      narg = 0;
      j = spos_ ("//", x);

      if (j >= 0)
        pscopy_up ( &tstr, x, 0, j );
      else {
        scopy_up (&tstr, x);
	j = 0 ;
      }

      ls = spice_dstring_length (&tstr);

      spice_dstring_init(&parsebuf) ;
      scopyd(&parsebuf, &tstr) ;
      buf = spice_dstring_value(&parsebuf) ;

      found = 0;
      token = strtok(buf, " ");       /* a bit more exact - but not sufficient everytime */
      j = j + strlen(token) + 1;
      if (strcmp(token, spice_dstring_value(&subname))) {
        while ((token = strtok(NULL, " "))) {
          if (!strcmp(token, spice_dstring_value(&subname))) {
            found = 1;
            break;
          }
          j = j + strlen(token) + 1;
        }
      }
      spice_dstring_free(&parsebuf) ;

      /*  make sure that subname followed by space */
      if (found)
        {
          j = j + spice_dstring_length (&subname) + 1;        /* 1st position of arglist: j */

	  t_p = spice_dstring_value(&tstr) ;
          while ((j < ls) && ((t_p[j] <= ' ') || (t_p[j] == ',')))
            j++;

          while (j < ls)
            {                        /* try to fetch valid arguments */
              k = j;
              spice_dstring_reinit(&ustr) ;
              if ((t_p[k] == Intro))
                {                /* handle historical syntax... */
                  if (alfa (t_p[k + 1]))
                      k++;
                  else if (t_p[k + 1] == '(')
                    {                /* transform to braces... */
                      k++;
                      t_p[k] = '{';
                      g = k;
                      nest = 1;
                      while ((nest > 0) && (g < ls))
                        {
                          g++;
                          if (t_p[g] == '(')
                            nest++;
                          else if (t_p[g] == ')')
                            nest--;
                        }

                      if ((g < ls) && (nest == 0))
                        t_p[g] = '}';
                    }
                }

              if (alfanum (t_p[k]) || t_p[k] == '.')
                {                /* number, identifier */
                  h = k;
                  while (t_p[k] > ' ')
                    k++;

                  pscopy (&ustr, spice_dstring_value(&tstr), h, k - h);
                  j = k;
                }
              else if (t_p[k] == '{')
                {
                  getexpress ( spice_dstring_value(&tstr), &ustr, &j);
                  j--; /* confusion: j was in Turbo Pascal convention */ ;
                }
              else
                {
                  j++;
                  if (t_p[k] > ' ')
                    {
                      spice_dstring_append(&vstr, "Subckt call, symbol ",-1) ;
                      cadd (&vstr, t_p[k]);
                      sadd (&vstr, " not understood");
                      message (dico, spice_dstring_value(&vstr) ) ;
                    }
                }

	      u_p = spice_dstring_value(&ustr) ;
              if (u_p[0])
                {
                  narg++;
                  k = cpos ('$', spice_dstring_value(&idlist)) ;

                  if (k >= 0)
                    {                /* replace dollar with expression string u */
                      pscopy (&vstr, spice_dstring_value(&idlist), 0, k);
                      sadd ( &vstr, spice_dstring_value(&ustr)) ;
                      pscopy (&ustr, spice_dstring_value(&idlist), k+1, spice_dstring_length (&idlist));
                      scopyd (&idlist, &vstr);
                      sadd (&idlist, spice_dstring_value(&ustr));
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
      scopys(&tstr, " Mismatch: ");
      nadd (&tstr, n);
      sadd (&tstr, "  formal but ");
      nadd (&tstr, narg);
      sadd (&tstr, " actual params.");
      err = message (dico, spice_dstring_value(&tstr));
      message (dico, spice_dstring_value(&idlist));
      /* ;} else { debugwarn(dico, idlist) */ ;
    }
  err = nupa_assignment (dico, spice_dstring_value(&idlist), 'N');

  spice_dstring_free(&subname) ;
  spice_dstring_free(&tstr) ;
  spice_dstring_free(&ustr) ;
  spice_dstring_free(&vstr) ;
  spice_dstring_free(&idlist) ;
  return err;
}

void
nupa_subcktexit (tdico * dico)
{
  dicostack (dico, Pop);
}
