/*       spicenum.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt  
 *  Free software under the terms of the GNU Lesser General Public License 
 */

/* number parameter add-on for Spice.
   to link with mystring.o, xpressn.o (math formula interpreter), 
   and with Spice frontend src/lib/fte.a . 
   Interface function nupa_signal to tell us about automaton states.
Buglist (some are 'features'):
  blank lines get category '*' 
  inserts conditional blanks before or after  braces 
  between .control and .endc, flags all lines as 'category C', dont touch. 
  there are reserved magic numbers (1e9 + n) as placeholders
  control lines must not contain {} . 
  ignores the '.option numparam' line planned to trigger the actions
  operation of .include certainly doesnt work 
  there are frozen maxima for source and expanded circuit size.
Todo:
  add support for nested .if .elsif .else .endif controls.
*/

#include <stdio.h>

#include "general.h"
#include "numparam.h"
#include "ngspice.h"

extern void txfree (void *ptr);

/* Uncomment this line to allow debug tracing */
/* #define TRACE_NUMPARAMS */

/*  the nupa_signal arguments sent from Spice:

   sig=1: Start of the subckt expansion.
   sig=2: Stop of the subckt expansion.
   sig=3: Stop of the evaluation phase.
   sig=0: Start of a deck copy operation

  After sig=1 until sig=2, nupa_copy does no transformations.
  At sig=2, we prepare for nupa_eval loop.
  After sig=3, we assume the initial state (clean).

  In Clean state, a lot of deckcopy operations come in and we
  overwrite any line pointers, or we start a new set after each sig=0 ?
  Anyway, we neutralize all & and .param lines  (category[] array!)
  and we substitute all {} &() and &id placeholders by dummy numbers. 
  The placeholders are long integers 1000000000+n (10 digits, n small).

*/
/**********  string handling ***********/

#define PlaceHold 1000000000L
static long placeholder = 0;

#ifdef NOT_REQUIRED		/* SJB - not required as front-end now does stripping */
static int
stripcomment (char *s)
/* allow end-of-line comments in Spice, like C++ */
{
  int i, ls;
  char c, d;
  unsigned char stop;
  ls = length (s);
  c = ' ';
  i = 0;
  stop = 0;

  while ((i < ls) && !stop)
    {
      d = c;
      i++;
      c = s[i - 1];
      stop = (c == d) && ((c == '/') || (c == '-'));
      /* comments after // or -- */ ;
    }
  if (stop)
    {
      i = i - 2;		/*last valid character before Comment */
      while ((i > 0) && (s[i - 1] <= ' '))
	i--;			/*strip blank space */

      if (i <= 0)
	scopy (s, "");
      else
	pscopy (s, s, 1, i);
    }
  else
    i = -1;

  return i /* i>=0  if comment stripped at that position */ ;
}
#endif /* NOT_REQUIRED */

static void
stripsomespace (char *s, unsigned char incontrol)
{
/* iff s starts with one of some markers, strip leading space */
  Str (12, markers);
  int i, ls;
  scopy (markers, "*.&+#$");

  if (!incontrol)
    sadd (markers, "xX");

  ls = length (s);
  i = 0;
  while ((i < ls) && (s[i] <= ' '))
    i++;

  if ((i > 0) && (i < ls) && (cpos (s[i], markers) > 0))
    pscopy (s, s, i + 1, ls);

}

#if 0				/* unused? */
void
partition (char *t)
/* t is a list val=expr val=expr .... Insert Lf-& before any val= */
/* the Basic preprocessor doesnt understand multiple cmd/line */
/* bug:  strip trailing spaces */
{
  Strbig (Llen, u);
  int i, lt, state;
  char c;
  cadd (u, Intro);
  state = 0;			/* a trivial 3-state machine */
  lt = length (t);
  while (t[lt - 1] <= ' ')
    lt--;

  for (i = 0; i < lt; i++)
    {
      c = t[i];
      if (c == '=')
	{
	  state = 1;
	}
      else if ((state == 1) && (c == ' '))
	{
	  state = 2;
	}
      if (state == 2)
	{
	  cadd (u, Lf);
	  cadd (u, Intro);
	  state = 0;
	}
      cadd (u, c);
    }
  scopy (t, u);
  for (i = 0; i < length (t); i++)
    {				/* kill braces inside */
      if ((t[i] == '{') || (t[i] == '}'))
	{
	  t[i] = ' ';
	}
    }
}
#endif

static int
stripbraces (char *s)
/* puts the funny placeholders. returns the number of {...} substitutions */
{
  int n, i, nest, ls, j;
  Strbig (Llen, t);
  n = 0;
  ls = length (s);
  i = 0;

  while (i < ls)
    {
      if (s[i] == '{')
	{			/* something to strip */
	  j = i + 1;
	  nest = 1;
	  n++;

	  while ((nest > 0) && (j < ls))
	    {
	      if (s[j] == '{')
		nest++;
	      else if (s[j] == '}')
		nest--;
	      j++;
	    }
	  pscopy (t, s, 1, i);
	  placeholder++;

	  if (t[i - 1] > ' ')
	    cadd (t, ' ');

          cadd (t, ' ');        // add extra character to increase number significant digits for evaluated numbers
          cadd (t, ' ');
          cadd (t, ' ');
          cadd (t, ' ');
          nadd (t, PlaceHold + placeholder);
          cadd (t, ' ');

	  if (s[j] >= ' ')
	    cadd (t, ' ');

	  i = length (t);
	  pscopy (s, s, j + 1, ls);
	  sadd (t, s);
	  scopy (s, t);
	}
      else
	i++;

      ls = length (s);
    }

  return n;
}

static int
findsubname (tdico * dico, char *s)
/* truncate the parameterized subckt call to regular old Spice */
/* scan a string from the end, skipping non-idents and {expressions} */
/* then truncate s after the last subckt(?) identifier */
{
  Str (80, name);
  int h, j, k, nest, ls;
  unsigned char found;
  h = 0;
  ls = length (s);
  k = ls;
  found = 0;

  while ((k >= 0) && (!found))
    {				/* skip space, then non-space */
      while ((k >= 0) && (s[k] <= ' '))
	k--;

      h = k + 1;		/* at h: space */
      while ((k >= 0) && (s[k] > ' '))
	{

	  if (s[k] == '}')
	    {
	      nest = 1;
	      k--;

	      while ((nest > 0) && (k >= 0))
		{
		  if (s[k] == '{')
		    nest--;
		  else if (s[k] == '}')
		    nest++;

		  k--;
		}
	      h = k + 1; /* h points to '{' */ ;
	    }
	  else
	    k--;
	}

      found = (k >= 0) && alfanum (s[k + 1]);	/* suppose an identifier */
      if (found)
	{			/* check for known subckt name */
	  scopy (name, "");
	  j = k + 1;
	  while (alfanum (s[j]))
	    {
	      cadd (name, upcase (s[j]));
	      j++;
	    }
	  found = (getidtype (dico, name) == 'U');
	}
    }
  if (found && (h < ls))
    pscopy (s, s, 1, h);

  return h;
}

static void
modernizeex (char *s)
/* old style expressions &(..) and &id --> new style with braces. */
{
  Strbig (Llen, t);
  int i, state, ls;
  char c, d;
  i = 0;
  state = 0;
  ls = length (s);

  while (i < ls)
    {
      c = s[i];
      d = s[i + 1];
      if ((!state) && (c == Intro) && (i > 0))
	{
	  if (d == '(')
	    {
	      state = 1;
	      i++;
	      c = '{';
	    }
	  else if (alfa (d))
	    {
	      cadd (t, '{');
	      i++;
	      while (alfanum (s[i]))
		{
		  cadd (t, s[i]);
		  i++;
		}
	      c = '}';
	      i--;
	    }
	}
      else if (state)
	{
	  if (c == '(')
	    state++;
	  else if (c == ')')
	    state--;

	  if (!state)		/* replace--) by terminator */
	    c = '}';

	}

      cadd (t, c);
      i++;
    }
  scopy (s, t);
}

static char
transform (tdico * dico, char *s, unsigned char nostripping, char *u)
/*         line s is categorized and crippled down to basic Spice
 *         returns in u control word following dot, if any 
 * 
 * any + line is copied as-is.
 * any & or .param line is commented-out.
 * any .subckt line has params section stripped off
 * any X line loses its arguments after sub-circuit name
 * any &id or &() or {} inside line gets a 10-digit substitute.
 *
 * strip  the new syntax off the codeline s, and
 * return the line category as follows:
 *   '*'  comment line
 *   '+'  continuation line
 *   ' '  other untouched netlist or command line
 *   'P'  parameter line, commented-out; (name,linenr)-> symbol table.
 *   'S'  subckt entry line, stripped;   (name,linenr)-> symbol table.
 *   'U'  subckt exit line
 *   'X'  subckt call line, stripped
 *   'C'  control entry line
 *   'E'  control exit line
 *   '.'  any other dot line
 *   'B'  netlist (or .model ?) line that had Braces killed 
 */
{
  Strbig (Llen, t);
  char category;
  int i, k, a, n;

  stripsomespace (s, nostripping);
  modernizeex (s);		/* required for stripbraces count */
  scopy (u, "");

  if (s[0] == '.')
    {				/* check Pspice parameter format */
      scopy_up (t, s);
      k = 1;

      while (t[k] > ' ')
	{
	  cadd (u, t[k]);
	  k++;
	}

      if (ci_prefix (".PARAM", t) == 1)
	{			/* comment it out */
	  /*s[0]='*'; */
	  category = 'P';
	}
      else if (ci_prefix (".SUBCKT", t) == 1)
	{			/* split off any "params" tail */
	  a = spos ("PARAMS:", t);
	  if (a > 0)
	    pscopy (s, s, 1, a - 1);

	  category = 'S';
	}
      else if (ci_prefix (".CONTROL", t) == 1)
	category = 'C';
      else if (ci_prefix (".ENDC", t) == 1)
	category = 'E';
      else if (ci_prefix (".ENDS", t) == 1)
	category = 'U';
      else
	{
	  category = '.';
	  n = stripbraces (s);
	  if (n > 0)
	    category = 'B';	/* priority category ! */
	}
    }
  else if (s[0] == Intro)
    {				/* private style preprocessor line */
      s[0] = '*';
      category = 'P';
    }
  else if (upcase (s[0]) == 'X')
    {				/* strip actual parameters */
      i = findsubname (dico, s);	/* i= index following last identifier in s */
      category = 'X';
    }
  else if (s[0] == '+')		/* continuation line */
    category = '+';
  else if (cpos (s[0], "*$#") <= 0)
    {				/* not a comment line! */
      n = stripbraces (s);
      if (n > 0)
	category = 'B';		/* line that uses braces */
      else
	category = ' ';		/* ordinary code line */
    }
  else
    category = '*';

  return category;
}

/************ core of numparam **************/

/* some day, all these nasty globals will go into the tdico structure
   and everything will get hidden behind some "handle" ...
*/

static int linecount = 0;	/* global: number of lines received via nupa_copy */
static int evalcount = 0;	/* number of lines through nupa_eval() */
static int nblog = 0;		/* serial number of (debug) logfile */
static unsigned char inexpansion = 0;	/* flag subckt expansion phase */
static unsigned char incontrol = 0;	/* flag control code sections */
static unsigned char dologfile = 0;	/* for debugging */
static unsigned char firstsignal = 1;
static FILE *logfile = NULL;
static tdico *dico = NULL;

/*  already part of dico : */
/*  Str(80, srcfile);   source file */
/*  Darray(refptr, char *, Maxline)   pointers to source code lines */
/*  Darray(category, char, Maxline)  category of each line */

/*
   Open ouput to a log file.
   takes no action if logging is disabled.
   Open the log if not already open.
*/
static void
putlogfile (char c, int num, char *t)
{
  Strbig (Llen, u);
  Str (20, fname);

  if (dologfile)
    {
      if ((logfile == NULL))
	{
	  scopy (fname, "logfile.");
	  nblog++;
	  nadd (fname, nblog);
	  logfile = fopen (fname, "w");
	}

      if ((logfile != NULL))
	{
	  cadd (u, c);
	  nadd (u, num);
	  cadd (u, ':');
	  cadd (u, ' ');
	  sadd (u, t);
	  cadd (u, '\n');
	  fputs (u, logfile);
	}
    }
}

static void
nupa_init (char *srcfile)
{
  int i;
  /* init the symbol table and so on, before the first  nupa_copy. */
  evalcount = 0;
  linecount = 0;
  incontrol = 0;
  placeholder = 0;
  dico = (tdico *)new(sizeof(tdico));
  inst_dico = (tdico *)new(sizeof(tdico));
  initdico (dico);
  initdico (inst_dico);

  for (i = 0; i < Maxline; i++)
    {
      dico->refptr[i] = NULL;
      dico->category[i] = '?';
    }
  sini (dico->srcfile, sizeof (dico->srcfile) - 4);

  if (srcfile != NULL)
    scopy (dico->srcfile, srcfile);
}

static void
nupa_done (void)
{
  int i;
  Str (80, rep);
  int dictsize, nerrors;

  if (logfile != NULL)
    {
      fclose (logfile);
      logfile = NULL;
    }
  nerrors = dico->errcount;
  dictsize = donedico (dico);

  for (i = Maxline - 1; i >= 0; i--)
    dispose ((void *) dico->refptr[i]);

  dispose ((void *) dico);
  dico = NULL;
  if (nerrors)
    {
      /* debug: ask if spice run really wanted */
      scopy (rep, " Copies=");
      nadd (rep, linecount);
      sadd (rep, " Evals=");
      nadd (rep, evalcount);
      sadd (rep, " Placeholders=");
      nadd (rep, placeholder);
      sadd (rep, " Symbols=");
      nadd (rep, dictsize);
      sadd (rep, " Errors=");
      nadd (rep, nerrors);
      cadd (rep, '\n');
      ws (rep);
      ws ("Numparam expansion errors: Run Spice anyway? y/n ? \n");
      rs (rep);
      if (upcase (rep[0]) != 'Y')
	exit (-1);
    }

  linecount = 0;
  evalcount = 0;
  placeholder = 0;
  /* release symbol table data */ ;
}

/* SJB - Scan the line for subcircuits */
void
nupa_scan (char *s, int linenum, int is_subckt)
{

  if (is_subckt)
    defsubckt (dico, s, linenum, 'U');
  else
    defsubckt (dico, s, linenum, 'O');

}

static char *
lower_str (char *str)
{
  char *s;

  for (s = str; *s; s++)
    *s = tolower (*s);

  return str;
}

static char *
upper_str (char *str)
{
  char *s;

  for (s = str; *s; s++)
    *s = toupper (*s);

  return str;
}

void
nupa_list_params (FILE * cp_out)
{
  char *name;
  int i;

  fprintf (cp_out, "\n\n");
  for (i = 1; i <= dico->nbd + 1; i++)
    {
      if (dico->dat[i].tp == 'R')
	{
	  name = lower_str (strdup (dico->dat[i].nom));
	  fprintf (cp_out, "       ---> %s = %g\n", name, dico->dat[i].vl);
	  txfree (name);
	}
    }
}

double
nupa_get_param (char *param_name, int *found)
{
  char *name = upper_str (strdup (param_name));
  double result = 0;
  int i;

  *found = 0;

  for (i = 1; i <= dico->nbd + 1; i++)
    {
      if (strcmp (dico->dat[i].nom, name) == 0)
	{
	  result = dico->dat[i].vl;
	  *found = 1;
	  break;
	}
    }

  txfree (name);
  return result;
}

void
nupa_add_param (char *param_name, double value)
{
  char *up_name = upper_str (strdup (param_name));
  int i = attrib (dico, up_name, 'N');

  dico->dat[i].vl = value;
  dico->dat[i].tp = 'R';
  dico->dat[i].ivl = 0;
  dico->dat[i].sbbase = NULL;

  txfree (up_name);
}

void
nupa_add_inst_param (char *param_name, double value)
{
  char *up_name = upper_str (strdup (param_name));
  int i = attrib (inst_dico, up_name, 'N');

  inst_dico->dat[i].vl = value;
  inst_dico->dat[i].tp = 'R';
  inst_dico->dat[i].ivl = 0;
  inst_dico->dat[i].sbbase = NULL;

  txfree (up_name);
}

void
nupa_copy_inst_dico ()
{
  int i;

  for (i = 1; i <= inst_dico->nbd; i++)
    nupa_add_param (inst_dico->dat[i].nom, inst_dico->dat[i].vl);
}

char *
nupa_copy (char *s, int linenum)
/* returns a copy (not quite) of s in freshly allocated memory.
   linenum, for info only, is the source line number. 
   origin pointer s is kept, memory is freed later in nupa_done.
  must abort all Spice if malloc() fails.  
  :{ called for the first time sequentially for all spice deck lines.
  :{ then called again for all X invocation lines, top-down for
    subckts defined at the outer level, but bottom-up for local
    subcircuit expansion, but has no effect in that phase.    
  we steal a copy of the source line pointer.
  - comment-out a .param or & line
  - substitute placeholders for all {..} --> 10-digit numeric values.
*/
{
  Strbig (Llen, u);
  Strbig (Llen, keywd);
  char *t;
  int ls;
  char c, d;
  ls = length (s);

  while ((ls > 0) && (s[ls - 1] <= ' '))
    ls--;

  pscopy (u, s, 1, ls);		/* strip trailing space, CrLf and so on */
  dico->srcline = linenum;

  if ((!inexpansion) && (linenum >= 0) && (linenum < Maxline))
    {
      linecount++;
      dico->refptr[linenum] = s;
      c = transform (dico, u, incontrol, keywd);
      if (c == 'C')
	incontrol = 1;
      else if (c == 'E')
	incontrol = 0;

      if (incontrol)
	c = 'C';		/* force it */

      d = dico->category[linenum];	/* warning if already some strategic line! */

      if ((d == 'P') || (d == 'S') || (d == 'X'))
	fprintf (stderr,
		 " Numparam warning: overwriting P,S or X line (linenum == %d).\n",
		 linenum);

      dico->category[linenum] = c;
    }				/* keep a local copy and mangle the string */

  ls = length (u);
  t = strdup (u);

  if (t == NULL)
    {
      fputs ("Fatal: String malloc crash in nupa_copy()\n", stderr);
      exit (-1);
    }
  else
    {
      if (!inexpansion)
	{
	  putlogfile (dico->category[linenum], linenum, t);
	};
    }
  return t;
}

int
nupa_eval (char *s, int linenum)
/* s points to a partially transformed line.
   compute variables if linenum points to a & or .param line.
   if ( the original is an X line,  compute actual params.;
   } else {  substitute any &(expr) with the current values.
   All the X lines are preserved (commented out) in the expanded circuit.
*/
{
  int idef;			/* subckt definition line */
  char c, keep, *ptr;
  unsigned int i;
  Str (80, subname);
  unsigned char err = 1;

  dico->srcline = linenum;
  c = dico->category[linenum];
#ifdef TRACE_NUMPARAMS
  fprintf (stderr, "** SJB - in nupa_eval()\n");
  fprintf (stderr, "** SJB - processing line %3d: %s\n", linenum, s);
  fprintf (stderr, "** SJB - category '%c'\n", c);
#endif /* TRACE_NUMPARAMS */
  if (c == 'P')			/* evaluate parameters */
    nupa_assignment (dico, dico->refptr[linenum], 'N');
  else if (c == 'B')		/* substitute braces line */
    err = nupa_substitute (dico, dico->refptr[linenum], s, 0);
  else if (c == 'X')
    {				/* compute args of subcircuit, if required */
      ptr = s;
      while (!isspace (*ptr))
	ptr++;
      keep = *ptr;
      *ptr = '\0';
      nupa_inst_name = strdup (s);
      *nupa_inst_name = 'x';
      *ptr = keep;

      for (i = 0; i < strlen (nupa_inst_name); i++)
	nupa_inst_name[i] = toupper (nupa_inst_name[i]);

      idef = findsubckt (dico, s, subname);
      if (idef > 0)
	nupa_subcktcall (dico, dico->refptr[idef], dico->refptr[linenum], 0);
      else
	putlogfile ('?', linenum, "  illegal subckt call.");
    }
  else if (c == 'U')		/*  release local symbols = parameters */
    nupa_subcktexit (dico);

  putlogfile ('e', linenum, s);
  evalcount++;
#ifdef TRACE_NUMPARAMS
  fprintf (stderr, "** SJB - leaving nupa_eval(): %s   %d\n", s, err);
  ws ("** SJB -                  --> ");
  ws (s);
  wln ();
  ws ("** SJB - leaving nupa_eval()");
  wln ();
  wln ();
#endif /* TRACE_NUMPARAMS */
  if (err)
    return 0;
  else
    return 1;
}

int
nupa_signal (int sig, char *info)
/* warning: deckcopy may come inside a recursion ! substart no! */
/* info is context-dependent string data */
{
  putlogfile ('!', sig, " Nupa Signal");
  if (sig == NUPADECKCOPY)
    {
      if (firstsignal)
	{
	  nupa_init (info);
	  firstsignal = 0;
	}
    }
  else if (sig == NUPASUBSTART)
    inexpansion = 1;
  else if (sig == NUPASUBDONE)
    {
      inexpansion = 0;
      nupa_inst_name = NULL;
    }
  else if (sig == NUPAEVALDONE)
    {
      nupa_done ();
      firstsignal = 1;
    }
  return 1;
}

#ifdef USING_NUPATEST
/* This is use only by the nupatest program */
tdico *
nupa_fetchinstance (void)
{
  return dico;
}
#endif /* USING_NUPATEST */
