/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group 
Modified: 2000 AlansFixes
**********/

/*------------------------------------------------------------------------------
 * re-written by SDB during 4.2003 to enable SPICE2 POLY statements to be processed
 * properly.  This is particularly important for dependent sources, whose argument
 * list changes when POLY is used.
 * Major changes include:
 * -- Added lots of comments which (hopefully) elucidate the steps taken
 *    by the program during its processing.
 * -- Re-wrote translate, which does the processing of each card.
 * Please direct comments/questions/complaints to Stuart Brorson:
 * mailto:sdb@cloud9.net
 *-----------------------------------------------------------------------------*/

/*
 * Expand subcircuits. This is very spice-dependent. Bug fixes by Norbert 
 * Jeske on 10/5/85.
 */

/*======================================================================*
 * Expand all subcircuits in the deck. This handles imbedded .subckt
 * definitions. The variables substart, subend, and subinvoke can be used
 * to redefine the controls used. The syntax is invariant though.
 * NOTE: the deck must be passed without the title line.
 * What we do is as follows: first make one pass through the circuit
 * and collect all of the subcircuits. Then, whenever a line that starts
 * with 'x' is found, copy the subcircuit associated with that name and
 * splice it in. A few of the problems: the nodes in the spliced-in
 * stuff must be unique, so when we copy it, append "subcktname:" to
 * each node. If we are in a nested subcircuit, use foo:bar:...:node.
 * Then we have to systematically change all references to the renamed
 * nodes. On top of that, we have to know how many args BJT's have,
 * so we have to keep track of model names.
 *======================================================================*/

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "fteinp.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include MIF function prototypes */
#include "mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif

#include "subckt.h"
#include "variable.h"


/* ----- static declarations ----- */
static struct line * doit(struct line *deck);
static int translate(struct line *deck, char *formal, char *actual, char *scname, 
		     char *subname);
static void finishLine(char *dst, char *src, char *scname);
static int settrans(char *formal, char *actual, char *subname);
static char * gettrans(char *name);
static int numnodes(char *name);
static int  numdevs(char *s);
static bool modtranslate(struct line *deck, char *subname);
static void devmodtranslate(struct line *deck, char *subname);

/*---------------------------------------------------------------------
 * table is used in settrans and gettrans -- it holds the netnames used 
 * in the .subckt definition (t_old), and in the subcircuit invocation
 * (t_new)
 *--------------------------------------------------------------------*/
static struct tab {
    char *t_old;
    char *t_new;
} table[512];   /* That had better be enough. */


/*---------------------------------------------------------------------
 *  subs is the linked list which holds the .subckt definitions 
 *  found during processing. 
 *--------------------------------------------------------------------*/
struct subs {
    char *su_name;          /* The .subckt name. */
    char *su_args;          /* The .subckt arguments, space seperated. */
    int su_numargs;
    struct line *su_def;    /* Pointer to the .subckt definition. */
    struct subs *su_next;
} ;


static wordlist *modnames, *submod;
static struct subs *subs = NULL;
static bool nobjthack = FALSE;

static char start[32], sbend[32], invoke[32], model[32];

/*-------------------------------------------------------------------*/
/* inp_subcktexpand is the top level function which translates       */
/* .subckts into mainlined code.   Note that there are two things    */
/* we need to do:  1. Find all .subckt definitions & stick them      */
/* into a list.  2. Find all subcircuit invocations (refdes X)       */
/* and replace them with the .subckt definition stored earlier.      */
/*                                                                   */
/* The algorithm is as follows:                                      */
/* 1.  Define some aliases for .subckt, .ends, etc.                  */
/* 2.  Clean up parens around netnames                               */
/* 3.  Call doit, which does the actual translation.                 */
/* 4.  Check the results & return.                                   */
/* inp_subcktexpand takes as argument a pointer to deck, and         */
/* it returns a pointer to the same deck after the new subcircuits   */
/* are spliced in.                                                   */
/*-------------------------------------------------------------------*/
struct line *
inp_subcktexpand(struct line *deck)
{
    struct line *ll, *c;
    char *s;

    if(!cp_getvar("substart", VT_STRING, start))
        (void) strcpy(start, ".subckt");
    if(!cp_getvar("subend", VT_STRING, sbend))
        (void) strcpy(sbend, ".ends");
    if(!cp_getvar("subinvoke", VT_STRING, invoke))
        (void) strcpy(invoke, "X");
    if(!cp_getvar("modelcard", VT_STRING, model))
        (void) strcpy(model, ".model");
    if(!cp_getvar("modelline", VT_STRING, model))
        (void) strcpy(model, ".model");
    (void) cp_getvar("nobjthack", VT_BOOL, (char *) &nobjthack);

    /* Let's do a few cleanup things first... Get rid of ( ) around node
     * lists...
     */
    for (c = deck; c; c = c->li_next) {    /* iterate on lines in deck */
      if (prefix(start, c->li_line)) {   /* if we find .subckt . . . */
#ifdef TRACE
	/* SDB debug statement */
	printf("In inp_subcktexpand, found a .subckt: %s\n", c->li_line);
#endif
	for (s = c->li_line; *s && (*s != '('); s++)    /* Iterate charwise along line until ( is found */
	  ;
	if (*s) {
	  while (s[0] && (s[1] != ')')) {
	    s[0] = s[1];
	    s++;
	  }
	  while (s[1]) {
	    s[0] = s[2];
	    s++;
	  }
	}  /* if (*s)  . . . */
      } else {
	for (s = c->li_line; *s && !isspace(*s); s++)    /* Iterate charwise along line until space is found */
	  ;
	while (isspace(*s))
	  s++;
	if (*s == '(') {
	  while (s[0] && (s[1] != ')')) {
	    s[0] = s[1];
	    s++;
	  }
	  while (s[1]) {
	    s[0] = s[2];
	    s++;
	  } /* while */
	} /* if (*s == '(' . . . */
      }
    }   /*  for (c = deck . . . */
    

    /* doit does the actual splicing in of the .subckt . . .  */
#ifdef TRACE
	/* SDB debug statement */
	printf("In inp_subcktexpand, about to call doit.\n");
#endif
    ll = doit(deck);

   /* Now check to see if there are still subckt instances undefined... */
    if (ll!=NULL) for (c = ll; c; c = c->li_next)
	if (ciprefix(invoke, c->li_line)) {
	    fprintf(cp_err, "Error: unknown subckt: %s\n",
		    c->li_line);
	    return NULL;
	}

    return (ll);  /* return the spliced deck.  */
}


#define MAXNEST 21
/*-------------------------------------------------------------------*/
/*  doit does the actual substitution of .subckts.                   */
/*  It takes two passes:  the first extracts .subckts                */
/*  and sticks pointer to them into the linked list sss.  It does    */
/*  the extraction recursively.  Then, it look for subcircuit        */
/*  invocations and substitutes the stored .subckt into              */
/*  the main circuit file.                                           */
/*  It takes as argument a pointer to the deck, and returns a        */
/*  pointer to the deck after the subcircuit has been spliced in.    */
/*-------------------------------------------------------------------*/
static struct line *
doit(struct line *deck)
{
    struct line *c, *last, *lc, *lcc;
    struct subs *sss = (struct subs *) NULL, *ks;   /*  *sss and *ks temporarily hold decks to substitute  */
    char *s, *t, *scname, *subname;
    int nest, numpasses = MAXNEST, i;
    bool gotone;
    wordlist *wl;
    wordlist *tmodnames = modnames;
    wordlist *tsubmod = submod;
    struct subs *ts = subs;
    int error;

    /* Save all the old stuff... */
    modnames = NULL;
    subs = NULL;
    submod = NULL;

#ifdef TRACE
	/* SDB debug statement */
	printf("In doit, about to start first pass through deck.\n");
#endif

    /* First pass: xtract all the .subckts and stick pointers to them into sss.  */
    for (last = deck, lc = NULL;  last;  ) {
        if (prefix(sbend, last->li_line)) {         /* if line == .ends  */
            fprintf(cp_err, "Error: misplaced %s line: %s\n", sbend,
                    last->li_line);
            return (NULL);
        } 
	else if (prefix(start, last->li_line)) {    /* if line == .subckt  */
	    if (last->li_next == NULL) {            /* first check that next line is non null */
                fprintf(cp_err, "Error: no %s line.\n", sbend);
                return (NULL);
            }
            lcc = NULL;                           
            wl_free(submod);
            submod = NULL;
            gotone = FALSE;

	    /* Here we loop through the deck looking for .subckt and .ends cards.
	     * At the end of this section, last will point to the location of the
	     * .subckt card, and lcc will point to the location of the .ends card.
	     */
            for (nest = 0, c = last->li_next;  c;  c = c->li_next) {
	       if (prefix(sbend, c->li_line)) { /* found a .ends */ 
                    if (!nest)
		        break;   /* nest = 0 means we have balanced .subckt and .ends  */
                    else {
		        nest--;    /* decrement nesting, and assign lcc to the current card */
			lcc = c;   /* (lcc points to the position of the .ends)             */
                        continue;  /* then continue looping                                 */
                    }
	        } else if (prefix(start, c->li_line))  /* if .subckt, increment nesting */
                    nest++;
	        lcc = c;     /* lcc points to current pos of c  */
            } /* for (nest = 0 . . . */

	    /* Check to see if we have looped through remainder of deck without finding .ends */
            if (!c) {       
                fprintf(cp_err, "Error: no %s line.\n", sbend);
                return (NULL);  
            }


            sss = alloc(struct subs);
	    if (!lcc)               /* if lcc is null, then no .ends was found.  */
		lcc = last;
            lcc->li_next = NULL;    /* shouldn't we free some memory here????? */

	    /* At this point, last points to the .subckt card, and lcc points to the .ends card */

	    /*  what does this do!??!?!  */
            if (lc)
                lc->li_next = c->li_next;
            else
                deck = c->li_next;

	    /*  Now put the .subckt definition found into sss  */
            sss->su_def = last->li_next;  
            s = last->li_line;    
            txfree(gettok(&s));
            sss->su_name = gettok(&s);
            sss->su_args = copy(s);
	    /* count the number of args in the .subckt line */
            for (sss->su_numargs = 0, i = 0; s[i]; ) {
                while (isspace(s[i]))
                    i++;
                if (s[i]) {
                    sss->su_numargs++;
                    while (s[i] && !isspace(s[i]))
                        i++;
                }
            }
            sss->su_next = subs;   
            subs = sss;            /* Now that sss is built, assign it to subs */
            last = c->li_next;
            lcc = subs->su_def;

        } 
	else {  /*  line is neither .ends nor .subckt.  */
	  /* make lc point to this card, and advance last to next card. */
            lc = last;    
            last = last->li_next;
        }
    } /* for (last = deck . . . .  */


    /* At this point, sss holds the .subckt definition found, subs holds
     * all .subckt defs found, including this one,
     * last points to the NULL at the end of the deck,
     * lc points to the last non-.subckt or .ends card, 
     * and lcc points to the .ends card 
     */

    if (!sss)            /* if sss == FALSE, we have found no subckts.  Just return.  */
        return (deck);

    /* Otherwise, expand sub-subcircuits recursively. */
    for (ks = sss = subs; sss; sss = sss->su_next)  /* iterate through the list of subcircuits */
        if (!(sss->su_def = doit(sss->su_def)))
            return (NULL);
    subs = ks;  /* ks has held pointer to start of subcircuits list. */
    

    /* Get all the model names so we can deal with BJT's. 
    *  Stick all the model names into the doubly-linked wordlist wl.
    */
    for (c = deck; c; c = c->li_next)
        if (prefix(model, c->li_line)) {
            s = c->li_line;
            txfree(gettok(&s));
            wl = alloc(struct wordlist);
            wl->wl_next = modnames;
            if (modnames)
                modnames->wl_prev = wl;
            modnames = wl;
            wl->wl_word = gettok(&s);  /* wl->wl_word now holds name of model */
        }

#ifdef TRACE
	/* SDB debug statement */
	printf("In doit, about to start second pass through deck.\n");
#endif

    error = 0;
    /* Second pass: do the replacements. */
    do {                    /*  while (!error && numpasses-- && gotone)  */
        gotone = FALSE;
        for (c = deck, lc = NULL; c; ) {
	   if (ciprefix(invoke, c->li_line)) {  /* found reference to .subckt (i.e. component with refdes X)  */
		char *tofree, *tofree2;
	        gotone = TRUE;
                t = tofree = s = copy(c->li_line);       /*  s & t hold copy of component line  */

		/*  make scname point to first non-whitepace chars after refdes invocation
		 * e.g. if invocation is Xreference, *scname = reference
		 */
                tofree2 = scname = gettok(&s);            
                scname += strlen(invoke);   
                while ((*scname == ' ') || (*scname == '\t') ||
                        (*scname == ':'))
                    scname++;

		/*  Now set s to point to last non-space chars in line (i.e.
		 *   the name of the model invoked  
		 */
                while(*s)
                    s++;
                s--;
                while ((*s == ' ') || (*s == '\t'))
                    *s-- = '\0';
                while ((*s != ' ') && (*s != '\t'))
                    s--;
                s++;

		/* iterate through .subckt list and look for .subckt name invoked */
                for (sss = subs; sss; sss = sss->su_next)  
                    if (eq(sss->su_name, s))
                        break;


		/* At this point, sss points to the .subckt invoked, 
		 * and scname points to the netnames
		 * involved.
		 */


                /* If no .subckt is found, don't complain -- this might be an
                 * instance of a subckt that is defined above at higher level.
                 */
                if (!sss) {
                    lc = c;
                    c = c->li_next;
                    continue;
                }

                /* Now we have to replace this line with the
                 * macro definition.
                 */
                subname = copy(sss->su_name);  

		/*  make lcc point to a copy of the .subckt definition  */
                lcc = inp_deckcopy(sss->su_def);  

                /* Change the names of .models found in .subckts . . .  */
                if (modtranslate(lcc, scname))    /* this translates the model name in the .model line */
		   devmodtranslate(lcc, scname);  /* This translates the model name on all components in the deck */

                s = sss->su_args;
                txfree(gettok(&t));  /* Throw out the subcircuit refdes */

		/* now invoke translate, which handles the remainder of the
		 * translation.
		 */
                if (!translate(lcc, s, t, scname, subname))
		    error = 1;
		tfree(subname);

                /* Now splice the decks together. */
                if (lc)
                    lc->li_next = lcc;
                else
                    deck = lcc;
                while (lcc->li_next != NULL)
                    lcc = lcc->li_next;
                lcc->li_next = c->li_next;
                c = lcc->li_next;
                lc = lcc;
		tfree(tofree);
		tfree(tofree2);
            } 	  /* if (ciprefix(invoke, c->li_line)) . . . */
	   else {
                lc = c;
                c = c->li_next;
            }
        }
    } while (!error && numpasses-- && gotone);


    if (!numpasses) {
        fprintf(cp_err, "Error: infinite subckt recursion\n");
        return (NULL);
    }


    if (error)
	return NULL;	/* error message already reported; should free( ) */

    subs = ts;
    modnames = tmodnames;
    submod = tsubmod;

    return (deck);
}



/*-------------------------------------------------------------------*/
/* Copy a deck, including the actual lines.                          */
/*-------------------------------------------------------------------*/
struct line *
inp_deckcopy(struct line *deck)
{
    struct line *d = NULL, *nd = NULL;

    while (deck) {
        if (nd) {
            d->li_next = alloc(struct line);
            d = d->li_next;
        } else
            nd = d = alloc(struct line);
        d->li_linenum = deck->li_linenum;
        d->li_line = copy(deck->li_line);
        if (deck->li_error)
            d->li_error = copy(deck->li_error);
        d->li_actual = inp_deckcopy(deck->li_actual);
        deck = deck->li_next;
    }
    return (nd);
}

/*------------------------------------------------------------------------------------------*
 * Translate all of the device names and node names in the .subckt deck. They are
 * pre-pended with subname:, unless they are in the formal list, in which case
 * they are replaced with the corresponding entry in the actual list.
 * The one special case is node 0 -- this is always ground and we don't
 * touch it.
 *
 * Variable name meanings:
 * *deck = pointer to subcircuit definition (lcc) (struct line)
 * formal = copy of the .subckt definition line (e.g. ".subckt subcircuitname 1 2 3") (string)
 * actual = copy of the .subcircuit invocation line (e.g. "Xexample 4 5 6 subcircuitname") (string)
 * scname = refdes (- first letter) used at invocation (e.g. "example") (string)
 * subname = copy of the subcircuit name 
 *-------------------------------------------------------------------------------------------*/
static int
translate(struct line *deck, char *formal, char *actual, char *scname, char *subname)
{
    struct line *c;
    char *buffer, *next_name, dev_type, *name, *s, *t, ch, *nametofree;
    int nnodes, i, dim;
    int rtn=0;
    
    /* settrans builds the table holding the translated netnames.  */
    i = settrans(formal, actual, subname);
    if (i < 0) {
	fprintf(stderr,
	"Too few parameters for subcircuit type \"%s\" (instance: x%s)\n",
		subname, scname);
	goto quit;
    } else if (i > 0) {
	fprintf(stderr,
	"Too many parameters for subcircuit type \"%s\" (instance: x%s)\n",
		subname, scname);
	goto quit;
    }

    /* now iterate through the .subckt deck and translate the cards. */
    for (c = deck; c; c = c->li_next) {  

#ifdef TRACE
      /* SDB debug statement */
      printf("\nIn translate, examining line %s \n", c->li_line);
#endif

      
      dev_type = *(c->li_line);   

      /* Rename the device. */
        switch (dev_type) {
        case '\0':
        case '*':
        case '.':
            /* Just a pointer to the line into s and then break */
	  buffer = tmalloc(2000);    /* XXXXX */
	  s = c->li_line;
	  break;


#ifdef XSPICE
/*===================  case A  ====================*/
/* gtri - add - wbk - 10/23/90 - process A devices specially */
/* since they have a more involved and variable length node syntax */
	    
        case 'a':
        case 'A':
	  
            /* translate the instance name according to normal rules */

            buffer = tmalloc(2000);     /* XXXXX */

            s = c->li_line;
            name = MIFgettok(&s);

	    /* maschmann 
            sprintf(buffer, "%s:%s ", name, scname);   */
            sprintf(buffer, "a:%s:%s ", scname, name+1 );  
                   


            /* Now translate the nodes, looking ahead one token to recognize */
            /* when we reach the model name which should not be translated   */
            /* here.                                                         */

            next_name = MIFgettok(&s);

            while(1) {

                /* rotate the tokens and get the the next one */

                name = next_name;
                next_name = MIFgettok(&s);

                /* if next token is NULL, name holds the model name, so exit */

                if(next_name == NULL)
                    break;

                /* Process the token in name.  If it is special, then don't */
                /* translate it.                                            */

                switch(*name) {

                case '[':
                case ']':
                case '~':
                    sprintf(buffer + strlen(buffer), "%s ", name);
                    break;

                case '%':

                    sprintf(buffer + strlen(buffer), "%%");

                    /* don't translate the port type identifier */

                    name = next_name;
                    next_name = MIFgettok(&s);

                    sprintf(buffer + strlen(buffer), "%s ", name);
                    break;

                default:

                    /* must be a node name at this point, so translate it */

                    t = gettrans(name);
                    if (t)
                        sprintf(buffer + strlen(buffer), "%s ", t);
                    else
		      /* maschmann:  changed order 
                        sprintf(buffer + strlen(buffer), "%s:%s ", name, scname); */
		      if(name[0]=='v' || name[0]=='V')  sprintf(buffer + strlen(buffer), "v:%s:%s ", scname, name+1);
		      else sprintf(buffer + strlen(buffer), "%s:%s ", scname, name);

                    break;

                } /* switch */

            } /* while */


            /* copy in the last token, which is the model name */

            if(name)
                sprintf(buffer + strlen(buffer), "%s ", name);

            /* Set s to null string for compatibility with code */
            /* after switch statement                           */

            s = "";
            break;

/* gtri - end - wbk - 10/23/90 */
#endif

/*================   case E, F, G, H  ================*/
/* This section handles controlled sources and allows for SPICE2 POLY attributes.
 * This is a new section, added by SDB to handle POLYs in sources.  Significant
 * changes were made in here.
 * 4.21.2003 -- SDB.  mailto:sdb@cloud9.net
 */
	case 'E': case 'e':
	case 'F': case 'f':
	case 'G': case 'g':
	case 'H': case 'h':


	  s = c->li_line;       /* s now holds the SPICE line */
	  t = name = gettok(&s);    /* name points to the refdes  */
	  if (!name)
	    continue;
	  if (!*name) {
	    tfree(name);
	    continue;
	  }
	  
/* Here's where we translate the refdes to e.g. F:subcircuitname:57
 * and stick the translated name into buffer.
 */
	  ch = *name;           /* ch identifies the type of component */
	  buffer = tmalloc(2000);    /* XXXXX */
	  name++;
	  if (*name == ':')
	    name++;             /* now name point to the rest of the refdes */
	                           

	  if (*name)
	    (void) sprintf(buffer, "%c:%s:%s ", ch, scname,  /* F:subcircuitname:refdesname */
			   name);
	  else
	    (void) sprintf(buffer, "%c:%s ", ch, scname);    /* F:subcircuitname */
	  tfree(t);
	  

/* Next iterate over all nodes (netnames) found and translate them. */
	  nnodes = numnodes(c->li_line);
	  
	  while (nnodes-- > 0) {
	    name = gettok_node(&s);
	    if (name == NULL) {
	      fprintf(cp_err, "Error: too few nodes: %s\n",
		      c->li_line);
	      goto quit;
	    }
	      
	    /* call gettrans and see if netname was used in the invocation */
	    t = gettrans(name);     
	      
	    if (t) {   /* the netname was used during the invocation; print it into the buffer */
	      (void) sprintf(buffer + strlen(buffer), "%s ", t);
	    }
	    else {    /* net netname was not used during the invocation; place a 
		       * translated name into the buffer.
		       */
	      (void) sprintf(buffer + strlen(buffer),
			     "%s:%s ", scname, name);
	    }
	    tfree(name);
	  }  /* while (nnodes-- . . . . */
  

/*  Next we handle the POLY (if any) */
	  /* get next token */
	  t = s;
	  next_name = (char *)gettok_noparens(&t);
	  if ( (strcmp(next_name, "POLY") == 0) ||
	       (strcmp(next_name, "poly") == 0)) {         /* found POLY . . . . */

#ifdef TRACE
	      /* SDB debug statement */
	      printf("In translate, looking at e, f, g, h found poly\n");
#endif

	      /* move pointer ahead of (  */
	      if( get_l_paren(&s) == 1 ) {   
	      	fprintf(cp_err, "Error: no left paren after POLY %s\n",
	      		c->li_line);
	      	tfree(next_name);
	      	goto quit;
	      }

	      nametofree = gettok_noparens(&s);
	      dim = atoi(nametofree);  /* convert returned string to int */
	      tfree(nametofree);
	      
	      /* move pointer ahead of ) */
	      if( get_r_paren(&s) == 1 ) {   
	      	fprintf(cp_err, "Error: no right paren after POLY %s\n",
	      		c->li_line);
	      	tfree(next_name);
	      	goto quit;
	      }

	      /* Write POLY(dim) into buffer */
	      (void) sprintf(buffer + strlen(buffer),
			     "POLY( %d ) ", dim);
	      

	  } /* if ( (strcmp(next_name, "POLY") == 0) . . .  */
	  else
	    dim = 1;    /* only one controlling source . . . */
	  tfree(next_name);

/* Now translate the controlling source/nodes */
	  nnodes = dim * numdevs(c->li_line);     
	  while (nnodes-- > 0) {
	    nametofree = name = gettok_node(&s);   /* name points to the returned token  */
	    if (name == NULL) {
	      fprintf(cp_err, "Error: too few devs: %s\n",
		      c->li_line);
	      goto quit;
	    }

	    if ( (dev_type == 'f') ||
		 (dev_type == 'F') ||
		 (dev_type == 'h') ||
		 (dev_type == 'H') ) {   

	      /* Handle voltage source name */

#ifdef TRACE 
	      /* SDB debug statement */
	      printf("In translate, found type f or h\n");
#endif

	      ch = *name;         /*  ch is the first char of the token.  */
	      name++;
	      if (*name == ':')
		name++;           /* name now points to the remainder of the token */
	      (void) sprintf(buffer + strlen(buffer),
			     "%c:%s:%s ", ch, scname, name);  
	      /* From Vsense and Urefdes creates V:Urefdes:sense */
	    }
	    else {                              /* Handle netname */

#ifdef TRACE 
	      /* SDB debug statement */
	      printf("In translate, found type e or g\n");
#endif

	      /* call gettrans and see if netname was used in the invocation */
	      t = gettrans(name);
	      
	      if (t) {   /* the netname was used during the invocation; print it into the buffer */
		(void) sprintf(buffer + strlen(buffer), "%s ", t);
	      }
	      else {    /* net netname was not used during the invocation; place a 
			 * translated name into the buffer.
			 */
		(void) sprintf(buffer + strlen(buffer),
			       "%s:%s ", scname, name);
		/* From netname and Urefdes creates Urefdes:netname */
	      }
	    }
	    tfree(nametofree);
	  }      /* while (nnodes--. . . . */
	  
/* Now write out remainder of line (polynomial coeffs) */
	  finishLine(buffer + strlen(buffer), s, scname);
	  s = "";
	  break;


/*=================   Default case  ===================*/
        default:            /* this section handles ordinary components */
	  s = c->li_line;
	  nametofree = name = gettok_node(&s);  /* changed to gettok_node to handle netlists with ( , ) */
	  if (!name)
	    continue;
	  if (!*name) {
	    tfree(name);
	    continue;
	  }

/* Here's where we translate the refdes to e.g. R:subcircuitname:57
 * and stick the translated name into buffer.
 */
	  ch = *name;
	  buffer = tmalloc(2000);    /* XXXXX */
	  name++;
	  if (*name == ':')
	    name++;

	  if (*name)
	    (void) sprintf(buffer, "%c:%s:%s ", ch, scname,
			   name);
	  else
	    (void) sprintf(buffer, "%c:%s ", ch, scname);
	  tfree(nametofree);


/* Next iterate over all nodes (netnames) found and translate them. */
	  nnodes = numnodes(c->li_line);
	  
	  while (nnodes-- > 0) {
	    name = gettok_node(&s);
	    if (name == NULL) {
	      fprintf(cp_err, "Error: too few nodes: %s\n",
		      c->li_line);
	      goto quit;
	    }
	      
	    /* call gettrans and see if netname was used in the invocation */
	    t = gettrans(name);
	      
	    if (t) {   /* the netname was used during the invocation; print it into the buffer */
	      (void) sprintf(buffer + strlen(buffer), "%s ", t);
	    }
	    else {    /* net netname was not used during the invocation; place a 
		       * translated name into the buffer.
		       */
	      (void) sprintf(buffer + strlen(buffer),
			     "%s:%s ", scname, name);
	    }
	    free(name);
	  }  /* while (nnodes-- . . . . */
  
/* Now translate any devices (i.e. controlling sources).  
 * This may be supurfluous because we handle dependent
 * source devices above . . . .
 */
	  nnodes = numdevs(c->li_line);     
	  while (nnodes-- > 0) {
	    t = name = gettok_node(&s);
	    if (name == NULL) {
	      fprintf(cp_err, "Error: too few devs: %s\n",
		      c->li_line);
	      goto quit;
	    }
	    ch = *name;
	    name++;
	    if (*name == ':')
	      name++;
	    
	    if (*name)
	      (void) sprintf(buffer + strlen(buffer),
			     "%c:%s:%s ", ch, scname, name);
	    else
	      (void) sprintf(buffer + strlen(buffer),
			     "%c:%s ", ch, scname);
	    tfree(t);
	  } /* while (nnodes--. . . . */
	    
	    
/* Now we finish off the line.  For most components (R, C, etc),
 * this involves adding the component value to the buffer.
 * We also scan through the line for v(something) and
 * i(something)...
 */
	    finishLine(buffer + strlen(buffer), s, scname);
	    s = "";

	} /* switch(c->li_line . . . . */

	(void) strcat(buffer, s);
        tfree(c->li_line);
        c->li_line = copy(buffer);

#ifdef TRACE 
	/* SDB debug statement */
	printf("In translate, translated line = %s \n", c->li_line);
#endif

        tfree(buffer);
    }  /* for (c = deck . . . . */
    rtn = 1;
quit:
    for (i = 0; ; i++) {
	if(!table[i].t_old && !table[i].t_new)
	    break;
	FREE(table[i].t_old);
	FREE(table[i].t_new);
    }
    return rtn;
}



/*-------------------------------------------------------------------*
 * finishLine now doesn't handle current or voltage sources.
 * Therefore, it just writes out the final netnames, if required.
 * Changes made by SDB on 4.29.2003.
 *-------------------------------------------------------------------*/
static void
finishLine(char *dst, char *src, char *scname)
{
    char buf[4 * BSIZE_SP], which;
    char *s;
    int i;
    int lastwasalpha;

    lastwasalpha = 0;
    while (*src) {
        /* Find the next instance of "<non-alpha>[vi]<opt spaces>(" in
         * this string.
         */
        if (((*src != 'v') && (*src != 'V') &&
                (*src != 'i') && (*src != 'I')) ||
		lastwasalpha) {
	    lastwasalpha = isalpha(*src);
            *dst++ = *src++;
            continue;
        }
        for (s = src + 1; *s && isspace(*s); s++)
            ;
        if (!*s || (*s != '(')) {
	    lastwasalpha = isalpha(*src);
            *dst++ = *src++;
            continue;
        }
	lastwasalpha = 0;
        which = *dst++ = *src;
        src = s;
        *dst++ = *src++;
        while (isspace(*src))
            src++;
        for (i = 0; *src && !isspace(*src) && *src != ',' && (*src != ')');
	    i++)
	{
            buf[i] = *src++;
	}
        buf[i] = '\0';

        if ((which == 'v') || (which == 'V'))
            s = gettrans(buf);
        else
            s = NULL;

        if (s) {
            while (*s)
                *dst++ = *s++;
        } 
	else {    /* just a normal netname . . . . */
	    /*
	     * 
	     */
	    if (buf[0] == 'v' || buf[0] == 'V') {
		*dst++ = buf[0];
		*dst++ = ':';
		i = 1;
	    } else {
		i = 0;
	    }
            for (s = scname; *s; )
                *dst++ = *s++;
            *dst++ = ':';
            for (s = buf + i; *s; )
                *dst++ = *s++;
        }

	/* translate the reference node, as in the "2" in "v(4,2)" */

        if ((which == 'v') || (which == 'V')) {
	    while (*src && (isspace(*src) || *src == ',')) {
		src++;
	    }
	    if (*src && *src != ')') {
		for (i = 0; *src && !isspace(*src) && (*src != ')'); i++)
		    buf[i] = *src++;
		buf[i] = '\0';
		s = gettrans(buf);
		*dst++ = ',';
		if (s) {
		    while (*s)
			*dst++ = *s++;
		} else {
		    for (s = scname; *s; )
			*dst++ = *s++;
		    *dst++ = ':';
		    for (s = buf; *s; )
			*dst++ = *s++;
		}
	    }
	}
    }

    return;
}

/*------------------------------------------------------------------------------*
 * settrans builds the table which holds the old and new netnames.  
 * it also compares the number of nets present in the .subckt definition against
 * the number of nets present in the subcircuit invocation.  It returns 0 if they
 * match, otherwise, it returns an error.
 *
 * Variable definitions:
 * formal = copy of the .subckt definition line (e.g. ".subckt subcircuitname 1 2 3") (string)
 * actual = copy of the .subcircuit invocation line (e.g. "Xexample 4 5 6 subcircuitname") (string)
 * subname = copy of the subcircuit name 
 *------------------------------------------------------------------------------*/
static int
settrans(char *formal, char *actual, char *subname)
{
    int i;

    bzero(table,sizeof(*table));
    
    for (i = 0; ; i++) {
        table[i].t_old = gettok(&formal);
        table[i].t_new = gettok(&actual);

        if (table[i].t_new == NULL) {
	    return -1;		/* Too few actual / too many formal */
        } else if (table[i].t_old == NULL) {
	    if (eq(table[i].t_new, subname))
		break;
	    else
		return 1;	/* Too many actual / too few formal */
	}
    }
    return 0;
}


/*------------------------------------------------------------------------------*
 * gettrans returns the name of the top level net if it is in the list,
 * otherwise it returns NULL.
 *------------------------------------------------------------------------------*/
static char *
gettrans(char *name)
{
    int i;

#ifdef XSPICE
    /* gtri - wbk - 2/27/91 - don't translate the reserved word 'null' */
    if (eq(name, "null"))
      return (name);
    /* gtri - end */
#endif

    if (eq(name, "0"))
        return (name);
    for (i = 0; table[i].t_old; i++)
        if (eq(table[i].t_old, name))
            return (table[i].t_new);
    return (NULL);
}

/*-------------------------------------------------------------------*/
/*-------------------------------------------------------------------*/
static int
numnodes(char *name)
{
/* gtri - comment - wbk - 10/23/90 - Do not modify this routine for */
/* 'A' type devices since the callers will not know how to find the */
/* nodes even if they know how many there are.  Modify the callers  */
/* instead.                                                         */
/* gtri - end - wbk - 10/23/90 */
    char c;
    struct subs *sss;
    char *s, *t, buf[4 * BSIZE_SP];
    wordlist *wl;
    int n, i, gotit;

    while (*name && isspace(*name))
	name++;

    c = (isupper(*name) ? tolower(*name) : *name);

    (void) strncpy(buf, name, sizeof(buf));
    s = buf;
    if (c == 'x') {     /* Handle this ourselves. */
        while(*s)
            s++;
        s--;
        while ((*s == ' ') || (*s == '\t'))
            *s-- = '\0';
        while ((*s != ' ') && (*s != '\t'))
            s--;
        s++;
        for (sss = subs; sss; sss = sss->su_next)
            if (eq(sss->su_name, s))
                break;
        if (!sss) {
            fprintf(cp_err, "Error: no such subcircuit: %s\n", s);
            return (0);
        }
        return (sss->su_numargs);
    }

    n = inp_numnodes(c);
    
    /* Added this code for variable number of nodes on BSIM3SOI devices  */
    /* The consequence of this code is that the value returned by the    */
    /* inp_numnodes(c) call must be regarded as "maximun number of nodes */
    /* for a given device type.                                          */
    /* Paolo Nenzi Jan-2001                                              */
    
    /* I hope that works, this code is very very untested */
    
	if (c=='m') {		     /* IF this is a mos */
		
	        i = 0;
		s = buf;
		gotit = 0;
		txfree(gettok(&s));          /* Skip component name */
		while ((i < n) && (*s) && !gotit) {
		  t = gettok_node(&s);       /* get nodenames . . .  */
		  for (wl = modnames; wl; wl = wl->wl_next)
		    if (eq(t, wl->wl_word)) 
		      gotit = 1;
		  i++;
		  tfree(t);
		} /* while . . . . */
		
		/* Note: node checks must be done on #_of_node-1 because the */
		/* "while" cicle increments the counter even when a model is */
		/* recognized. This code may be better!                      */
	 		
		if (i < 5) {
		  fprintf(cp_err, "Error: too few nodes for MOS: %s\n", name);
		  return(0);
		}
		return(i-1); /* compesate the unnecessary inrement in the while cicle */
    	} /* if (c=='m' . . .  */
    
    
    if (nobjthack || (c != 'q'))
        return (n);

    for (s = buf, i = 0; *s && (i < 4); i++)
        txfree(gettok(&s));

    if (i == 3)
        return (3);

    else if (i < 4) {
        fprintf(cp_err, "Error: too few nodes for BJT: %s\n", name);
        return (0);
    }

    /* Now, is this a model? */
    t = gettok(&s);
    for (wl = modnames; wl; wl = wl->wl_next)
      if (eq(t, wl->wl_word)) {
	  tfree(t);
	  return (3);
      }
    tfree(t);
    return (4);
}


/*-------------------------------------------------------------------*
 *  This function returns the number of controlling voltage sources
 *  (for F, H) or controlling nodes (for G, E)  attached to a dependent 
 *  source.
 *-------------------------------------------------------------------*/
static int 
numdevs(char *s)
{

  while (*s && isspace(*s))
    s++;
  switch (*s) {
    case 'K':
    case 'k':
      return (2);
    
      /* two nodes per voltage controlled source */
    case 'G':
    case 'g':
    case 'E':
    case 'e':
      return(2);

      /* one source per current controlled source */
    case 'F':
    case 'f':
    case 'H':
    case 'h':
   /* 2 lines here added to fix w bug, NCF 1/31/95 */
    case 'W':
    case 'w':     
      return (1);
    
        default:
        return (0);
    }
}

/*----------------------------------------------------------------------*
 *  modtranslate --  translates .model liness found in subckt definitions.
 *  Calling arguments are:
 *  *deck = pointer to the .subckt definition (linked list)
 *  *subname = pointer to the subcircuit name used at the subcircuit invocation (string)
 *  Modtranslate returns TRUE if it translated a model name, FALSE
 *  otherwise.
 *----------------------------------------------------------------------*/
static bool
modtranslate(struct line *deck, char *subname)
{
    struct line *c;
    char *buffer, *name, *t, model[4 * BSIZE_SP];
    wordlist *wl, *wlsub;
    bool gotone;

    (void) strcpy(model, ".model");
    gotone = FALSE;
    for (c = deck; c; c = c->li_next) {       /* iterate through model def . . . */
        if (prefix(model, c->li_line)) {
            gotone = TRUE;
            t = c->li_line;
            name = gettok(&t);     /* at this point, name = .model */
            buffer = tmalloc(strlen(name) + strlen(t) +
                    strlen(subname) + 4);
            (void) sprintf(buffer, "%s ",name);    /* at this point, buffer = ".model " */
	    tfree(name);
            name = gettok(&t);                     /* name now holds model name */
            wlsub = alloc(struct wordlist);
            wlsub->wl_next = submod;
            if (submod)
                submod->wl_prev = wlsub;
            submod = wlsub;
            wlsub->wl_word = name;
            (void) sprintf(buffer + strlen(buffer), "%s:%s ", 
                    subname, name);                 /* buffer = "model subname:modelname " */
            (void) strcat(buffer, t);
            tfree(c->li_line);
            c->li_line = buffer;
            t = c->li_line;
            txfree(gettok(&t));
            wl = alloc(struct wordlist);
            wl->wl_next = modnames;
            if (modnames) 
                modnames->wl_prev = wl;
            modnames = wl;
            wl->wl_word = gettok(&t);
        }
    }
    return(gotone);
}


/*-------------------------------------------------------------------*
 *  Devmodtranslate scans through the deck, and translates the 
 *  name of the model in a line held in a .subckt.  For example:
 *  before:   .subckt U1 . . . .
 *            Q1 c b e 2N3904
 *  after:    Q1 c b e U1:2N3904
 *-------------------------------------------------------------------*/
static void
devmodtranslate(struct line *deck, char *subname)
{
    struct line *s;
    char *buffer, *name, *t, c;
    wordlist *wlsub;
    bool found;

    for (s = deck; s; s = s->li_next) {
        t = s->li_line;
#ifdef TRACE
	/* SDB debug stuff */
	printf("In devmodtranslate, examining line %s.\n", t);
#endif

	while (*t && isspace(*t))
	    t++;
        c = isupper(*t) ? tolower(*t) : *t;  /* set c to first char in line. . . . */
        found = FALSE;
        buffer = tmalloc(strlen(t) + strlen(subname) + 4);

	switch (c) {

        case 'r':
        case 'c':
	    name = gettok(&t);  /* get refdes */
            (void) sprintf(buffer,"%s ",name);
	    tfree(name);
            name = gettok_node(&t);  /* get first netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get second netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);

            if (*t) {    /* if there is a model, process it. . . . */
                name = gettok(&t);
		/* Now, is this a subcircuit model? */
		for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                    if (eq(name, wlsub->wl_word)) {
			(void) sprintf(buffer + strlen(buffer), "%s:%s ",
				subname, name);
			found = TRUE;
			break;
                    }
                }
                if (!found)
		    (void) sprintf(buffer + strlen(buffer), "%s ", name);
		tfree(name);
            }

            found = FALSE;
            if (*t) {
                name = gettok(&t);
                /* Now, is this a subcircuit model? */
                for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
		    if (eq(name, wlsub->wl_word)) {
			(void) sprintf(buffer + strlen(buffer), "%s:%s ",
				subname, name);
			found = TRUE;
			break;
		    }
                }
                if (!found)
		    (void) sprintf(buffer + strlen(buffer), "%s ", name);
		tfree(name);
            }

            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
            break;

	case 'd':
	    name = gettok(&t);  /* get refdes */
            (void) sprintf(buffer,"%s ",name);
	    tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok(&t);

            /* Now, is this a subcircuit model? */
            for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                if (eq(name, wlsub->wl_word)) {
		    (void) sprintf(buffer + strlen(buffer), "%s:%s ", 
			    subname, name);
                found = TRUE;
                break;
                }
            }

            if (!found)
                (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
            break;

	case 'w':
	case 'u':
	case 'j':
	case 'z':
	    /* What are these devices anyway?   J = JFET, W = trans line (?), 
	       and u = IC, but what is Z?.
	       Also, why is 'U' here?  A 'U' element can have an arbitrary
	       number of nodes attached. . . .   -- SDB. */
            name = gettok(&t);
            (void) sprintf(buffer,"%s ",name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
            name = gettok(&t);

            /* Now, is this a subcircuit model? */
            for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                if (eq(name, wlsub->wl_word)) {
		    (void) sprintf(buffer + strlen(buffer), "%s:%s ", 
			    subname, name);
		    found = TRUE;
		    break;
                }
            }

            if (!found)
                (void) sprintf(buffer + strlen(buffer), "%s ", name);
            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
            break;

	    /*  Changed gettok() to gettok_node() on 12.2.2003 by SDB 
		to enable parsing lines like "S1 10 11 (80,51) SLATCH1"
		which occurr in real Analog Devices SPICE models.
	    */
        case 'o':    /* what is this element?  -- SDB */
	case 's':
	case 'm':
	    name = gettok(&t);  /* get refdes */
            (void) sprintf(buffer,"%s ",name);
	    tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get third attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get fourth attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok(&t);

            /* Now, is this a subcircuit model? */
            for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                if (eq(name, wlsub->wl_word)) {
		    (void) sprintf(buffer + strlen(buffer), "%s:%s ", 
			    subname, name);
		    found = TRUE;
		    break;
                }
            }

            if (!found)
                (void) sprintf(buffer + strlen(buffer), "%s ", name);
            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
	    tfree(name);
            break;

	case 'q':
	    name = gettok(&t);  /* get refdes */
            (void) sprintf(buffer,"%s ",name);
	    tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* get third attached netname */
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);
            name = gettok_node(&t);  /* this can be either a model name or a node name. */

            /* Now, is this a subcircuit model? */
            for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                if (eq(name, wlsub->wl_word)) {
		    (void) sprintf(buffer + strlen(buffer), "%s:%s ", 
			    subname, name);
		    found = TRUE;
		    break;
                }
            }
            if (!found)
                (void) sprintf(buffer + strlen(buffer), "%s ", name);
	    tfree(name);

            found = FALSE;
            if (*t) {
                name = gettok(&t);
                /* Now, is this a subcircuit model? */
                for (wlsub = submod; wlsub; wlsub = wlsub->wl_next) {
                    if (eq(name, wlsub->wl_word)) {
			(void) sprintf(buffer + strlen(buffer),
				"%s:%s ", subname, name);
			found = TRUE;
			break;
		    }
                }
                if (!found)
		    (void) sprintf(buffer + strlen(buffer), "%s ", name);
		tfree(name);
            }

            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
            break;

	default:
            tfree(buffer);
            break;
        }
    }
    return;
}

/*----------------------------------------------------------------------*
 * inp_numnodes returns the number of nodes (netnames) attached to the
 * component.  
 * This is a spice-dependent thing.  It should probably go somewhere
 * else, but...  Note that we pretend that dependent sources and mutual
 * inductors have more nodes than they really do...
 *----------------------------------------------------------------------*/
int
inp_numnodes(char c)
{
    if (isupper(c))
        c = tolower(c);
    switch (c) {
        case ' ':
        case '\t':
        case '.':
        case 'x':
        case '*':
        return (0);

        case 'b': return (2);
        case 'c': return (2);
        case 'd': return (2);
        case 'e': return (2); /* changed from 4 to 2 by SDB on 4.22.2003 to enable POLY */
        case 'f': return (2);
        case 'g': return (2); /* changed from 4 to 2 by SDB on 4.22.2003 to enable POLY */
        case 'h': return (2);
        case 'i': return (2);
        case 'j': return (3);
        case 'k': return (0);
        case 'l': return (2);
        case 'm': return (7); /* This means that 7 is the maximun number of nodes */
        case 'o': return (4);
        case 'q': return (4);
        case 'r': return (2);
        case 's': return (4);
        case 't': return (4);
        case 'u': return (3);
        case 'v': return (2);
 /* change 3 to 2 here to fix w bug, NCF 1/31/95 */
        case 'w': return (2);
        case 'z': return (3);

        default:
        fprintf(cp_err, "Warning: unknown device type: %c\n", c);
            return (2);
    }
}








