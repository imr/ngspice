/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group 
Modified: 2000 AlansFixes
**********/

/*
 * Expand subcircuits. This is very spice-dependent. Bug fixes by Norbert 
 * Jeske on 10/5/85.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "fteinp.h"

#include "subckt.h"
#include "variable.h"

/* static declarations */
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



struct subs {
    char *su_name;      /* The name. */
    char *su_args;      /* The arguments, space seperated. */
    int su_numargs;
    struct line *su_def;    /* The deck that is to be substituted. */
    struct subs *su_next;
} ;

/* Expand all subcircuits in the deck. This handles imbedded .subckt
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
 */

static wordlist *modnames, *submod;
static struct subs *subs = NULL;
static bool nobjthack = FALSE;

static char start[32], sbend[32], invoke[32], model[32];

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
    for (c = deck; c; c = c->li_next) {
        if (prefix(start, c->li_line)) {
            for (s = c->li_line; *s && (*s != '('); s++)
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
            }
        } else {
            for (s = c->li_line; *s && !isspace(*s); s++)
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
                }
            }
        }
    }
    
    ll = doit(deck);

   /* Now check to see if there are still subckt instances undefined... */
    if (ll!=NULL) for (c = ll; c; c = c->li_next)
	if (ciprefix(invoke, c->li_line)) {
	    fprintf(cp_err, "Error: unknown subckt: %s\n",
		    c->li_line);
	    return NULL;
	}

    return (ll);
}

#define MAXNEST 21

static struct line *
doit(struct line *deck)
{
    struct line *c, *last, *lc, *lcc;
    struct subs *sss = (struct subs *) NULL, *ks;
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

    /* Extract all the .subckts */
    for (last = deck, lc = NULL; last; ) {
        if (prefix(sbend, last->li_line)) {
            fprintf(cp_err, "Error: misplaced %s line: %s\n", sbend,
                    last->li_line);
            return (NULL);
        } else if (prefix(start, last->li_line)) {
            if (last->li_next == NULL) {
                fprintf(cp_err, "Error: no %s line.\n", sbend);
                return (NULL);
            }
            lcc = NULL;
            wl_free(submod);
            submod = NULL;
            gotone = FALSE;
            for (nest = 0, c = last->li_next; c; c = c->li_next) {
                if (prefix(sbend, c->li_line)) {
                    if (!nest)
                        break;
                    else {
                        nest--;
			lcc = c;
                        continue;
                    }
                } else if (prefix(start, c->li_line))
                    nest++;
		lcc = c;
            }
            if (!c) {
                fprintf(cp_err, "Error: no %s line.\n", sbend);
                return (NULL);
            }
            sss = alloc(struct subs);
	    if (!lcc)
		lcc = last;
            lcc->li_next = NULL;
            if (lc)
                lc->li_next = c->li_next;
            else
                deck = c->li_next;
            sss->su_def = last->li_next;
            s = last->li_line;
            (void) gettok(&s);
            sss->su_name = gettok(&s);
            sss->su_args = copy(s);
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
            subs = sss;
            last = c->li_next;
            lcc = subs->su_def;
        } else {
            lc = last;
            last = last->li_next;
        }
    }

    if (!sss)
        return (deck);

    /* Expand sub-subcircuits. */
    for (ks = sss = subs; sss; sss = sss->su_next)
        if (!(sss->su_def = doit(sss->su_def)))
            return (NULL);
    subs = ks;
    
    /* Get all the model names so we can deal with BJT's. */
    for (c = deck; c; c = c->li_next)
        if (prefix(model, c->li_line)) {
            s = c->li_line;
            (void) gettok(&s);
            wl = alloc(struct wordlist);
            wl->wl_next = modnames;
            if (modnames)
                modnames->wl_prev = wl;
            modnames = wl;
            wl->wl_word = gettok(&s);
        }

    error = 0;
    /* Now do the replacements. */
    do {
        gotone = FALSE;
        for (c = deck, lc = NULL; c; ) {
            if (ciprefix(invoke, c->li_line)) {
                gotone = TRUE;
                t = s = copy(c->li_line);
                scname = gettok(&s);
                scname += strlen(invoke);
                while ((*scname == ' ') || (*scname == '\t') ||
                        (*scname == ':'))
                    scname++;
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
                /* Don't complain -- this might be an
                 * instance of a subckt that is defined above.
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
                lcc = inp_deckcopy(sss->su_def);

                /* Change the names of the models... */
                if (modtranslate(lcc, scname))
                    devmodtranslate(lcc, scname);

                s = sss->su_args;
                (void) gettok(&t); /* Throw out the name. */

                if (!translate(lcc, s, t, scname, subname))
		    error = 1;

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
            } else {
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

/* Copy a deck, including the actual lines. */

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

/* Translate all of the device names and node names in the deck. They are
 * pre-pended with subname:, unless they are in the formal list, in which case
 * they are replaced with the corresponding entry in the actual list.
 * The one special case is node 0 -- this is always ground and we don't
 * touch it.
 */

static int
translate(struct line *deck, char *formal, char *actual, char *scname, char *subname)
{
    struct line *c;
    char *buffer, *name, *s, *t, ch;
    int nnodes, i;

    i = settrans(formal, actual, subname);
    if (i < 0) {
	fprintf(stderr,
	"Too few parameters for subcircuit type \"%s\" (instance: x%s)\n",
		subname, scname);
	return 0;
    } else if (i > 0) {
	fprintf(stderr,
	"Too many parameters for subcircuit type \"%s\" (instance: x%s)\n",
		subname, scname);
	return 0;
    }

    for (c = deck; c; c = c->li_next) {
        /* Rename the device. */
        switch (*c->li_line) {
        case '\0':
        case '*':
        case '.':
            /* Nothing any good here. */
            continue;

        default:
                s = c->li_line;
            name = gettok(&s);
	    if (!name)
		continue;
	    if (!*name) {
		tfree(name);
		continue;
	    }
            ch = *name;
            buffer = tmalloc(10000);    /* XXXXX */
            name++;
            if (*name == ':')
            name++;
            if (*name)
                (void) sprintf(buffer, "%c:%s:%s ", ch, scname,
                    name);
            else
                (void) sprintf(buffer, "%c:%s ", ch, scname);

            nnodes = numnodes(c->li_line);
            while (nnodes-- > 0) {
            name = gettok(&s);
            if (name == NULL) {
                fprintf(cp_err, "Error: too few nodes: %s\n",
                        c->li_line);
                return 0;
            }
            t = gettrans(name);
            if (t)
                (void) sprintf(buffer + strlen(buffer), "%s ",
                        t);
            else
                (void) sprintf(buffer + strlen(buffer),
                        "%s:%s ", scname, name);
            }    
            nnodes = numdevs(c->li_line);
            while (nnodes-- > 0) {
            name = gettok(&s);
            if (name == NULL) {
                fprintf(cp_err, "Error: too few devs: %s\n",
                        c->li_line);
                return 0;
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
            }
            /* Now scan through the line for v(something) and
             * i(something)...
             */
            finishLine(buffer + strlen(buffer), s, scname);
            s = "";
        }
            (void) strcat(buffer, s);
        tfree(c->li_line);
        c->li_line = copy(buffer);
        tfree(buffer);
    }
    return 1;
}

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
        } else {
	    /*
	     * i(vname) -> i(v:subckt:name)
	     * i(v:other:name) -> i(v:subckt:other:name)
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

static struct tab {
    char *t_old;
    char *t_new;
} table[512];   /* That had better be enough. */

static int
settrans(char *formal, char *actual, char *subname)
{
    int i;

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

static char *
gettrans(char *name)
{
    int i;

    if (eq(name, "0"))
        return (name);
    for (i = 0; table[i].t_old; i++)
        if (eq(table[i].t_old, name))
            return (table[i].t_new);
    return (NULL);
}

static int
numnodes(char *name)
{
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
    
	if (c=='m') {		/* IF this is a mos */
		
		i = 0;
		s = buf;
		gotit = 0;
		t = gettok(&s);	/* Skip component name */
		while ((i < n) && (*s) && !gotit) {
			t = gettok(&s);
    			for (wl = modnames; wl; wl = wl->wl_next)
     		   if (eq(t, wl->wl_word)) 
     		   	gotit = 1;
     		i++;
		}
		
	/* Note: node checks must be done on #_of_node-1 because the */
	/* "while" cicle increments the counter even when a model is */
	/* recognized. This code may be better!                      */
	 		
     if (i < 5) {
     	fprintf(cp_err, "Error: too few nodes for MOS: %s\n", name);
     	return(0);
    		}
    	return(i-1); /* compesate the unnecessary inrement in the while cicle */
    	}
    
    
    if (nobjthack || (c != 'q'))
        return (n);
    for (s = buf, i = 0; *s && (i < 4); i++)
        (void) gettok(&s);
    if (i == 3)
        return (3);
    else if (i < 4) {
        fprintf(cp_err, "Error: too few nodes for BJT: %s\n", name);
        return (0);
    }
    /* Now, is this a model? */
    t = gettok(&s);
    for (wl = modnames; wl; wl = wl->wl_next)
        if (eq(t, wl->wl_word))
            return (3);
    return (4);
}

static int 
numdevs(char *s)
{

    while (*s && isspace(*s))
	s++;
    switch (*s) {
        case 'K':
        case 'k':
        return (2);
    
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

static bool
modtranslate(struct line *deck, char *subname)
{
    struct line *c;
    char *buffer, *name, *t, model[4 * BSIZE_SP];
    wordlist *wl, *wlsub;
    bool gotone;

    (void) strcpy(model, ".model");
    gotone = FALSE;
    for (c = deck; c; c = c->li_next) {
        if (prefix(model, c->li_line)) {
            gotone = TRUE;
            t = c->li_line;
            name = gettok(&t);
            buffer = tmalloc(strlen(name) + strlen(t) +
                    strlen(subname) + 4);
            (void) sprintf(buffer, "%s ",name);
            name = gettok(&t);
            wlsub = alloc(struct wordlist);
            wlsub->wl_next = submod;
            if (submod)
                submod->wl_prev = wlsub;
            submod = wlsub;
            wlsub->wl_word = name;
            (void) sprintf(buffer + strlen(buffer), "%s:%s ",
                    subname, name);
            (void) strcat(buffer, t);
            tfree(c->li_line);
            c->li_line = buffer;
            t = c->li_line;
            (void) gettok(&t);
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

static void
devmodtranslate(struct line *deck, char *subname)
{
    struct line *s;
    char *buffer, *name, *t, c;
    wordlist *wlsub;
    bool found;

    for (s = deck; s; s = s->li_next) {
        t = s->li_line;
	while (*t && isspace(*t))
	    t++;
        c = isupper(*t) ? tolower(*t) : *t;
        found = FALSE;
        buffer = tmalloc(strlen(t) + strlen(subname) + 4);

	switch (c) {

        case 'r':
        case 'c':
            name = gettok(&t);
            (void) sprintf(buffer,"%s ",name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);

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
            }

            (void) strcat(buffer, t);
            tfree(s->li_line);
            s->li_line = buffer;
            break;

	case 'd':
            name = gettok(&t);
            (void) sprintf(buffer,"%s ",name);
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

	case 'w':
	case 'u':
	case 'j':
	case 'z':
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

        case 'o':
	case 's':
	case 'm':
            name = gettok(&t);
            (void) sprintf(buffer,"%s ",name);
            name = gettok(&t);
            (void) sprintf(buffer + strlen(buffer), "%s ", name);
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

	case 'q':
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

/* This is a spice-dependent thing.  It should probably go somewhere
 * else, but...  Note that we pretend that dependent sources and mutual
 * inductors have more nodes than they really do...
 */

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
        case 'e': return (4);
        case 'f': return (2);
        case 'g': return (4);
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

