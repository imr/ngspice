/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * For dealing with spice input decks and command scripts
 */

/*
 * SJB 21 April 2005
 * Added support for end-of-line comments that begin with any of the following:
 *   ';'  (for PSpice compatability)
 *   '$ ' (for HSpice compatability)
 *   '//' (like in c++ and as per the numparam code)
 *   '--' (as per the numparam code)
 * Any following text to the end of the line is ignored.
 * Note requirement for $ to be followed by a space. This is to avoid conflict
 * with use in front of a variable.
 * Comments on a contunuation line (i.e. line begining with '+') are allowed
 * and are removed before lines are stitched.
 * Lines that contain only an end-of-line comment with or withou leading white
 * space are also allowed.
 */

/*
 * SJB 22 May 2001
 * Fixed memory leaks in inp_readall() when first(?) line of input begins with a '@'.
 * Fixed memory leaks in inp_readall() when .include lines have errors
 * Fixed crash where a NULL pointer gets freed in inp_readall()
 */

#include <config.h>
#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "fteinp.h"


#include "inpcom.h"
#include "variable.h"

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include new stuff */
#include "ipctiein.h"
#include "enh.h"
/* gtri - end - 12/12/90 */
#endif

/* SJB - Uncomment this line for debug tracing */
/*#define TRACE */

/* static declarations */
static char * readline(FILE *fd);
static void inp_stripcomments_deck(struct line *deck);
static void inp_stripcomments_line(char * s);

/*-------------------------------------------------------------------------*
 *  This routine reads a line (of arbitrary length), up to a '\n' or 'EOF' *
 *  and returns a pointer to the resulting null terminated string.         *
 *  The '\n' if found, is included in the returned string.                 *
 *  From: jason@ucbopal.BERKELEY.EDU (Jason Venner)                        *
 *  Newsgroups: net.sources                                                *
 *-------------------------------------------------------------------------*/
#define STRGROW 256

static char *
readline(FILE *fd)
{
    int c;
    int memlen;
    char *strptr;
    int strlen;
    
    strptr = NULL;
    strlen = 0;
    memlen = STRGROW; 
    strptr = tmalloc(memlen);
    memlen -= 1;          /* Save constant -1's in while loop */
    while((c = getc(fd)) != EOF) {
	if (strlen == 0 && (c == '\n' || c == ' ')) /* Leading spaces away */
	    continue;
        strptr[strlen] = c;
        strlen++;
        if( strlen >= memlen ) {
            memlen += STRGROW;
            if( !(strptr = trealloc(strptr, memlen + 1))) {
                return (NULL);
            }
        }
        if (c == '\n') {
            break;
        }
    }
    if (!strlen) {
        tfree(strptr);
        return (NULL);
    }
    strptr[strlen] = '\0'; 
    /* Trim the string */
    strptr = trealloc(strptr, strlen + 1);
    return (strptr);
}


/*-------------------------------------------------------------------------*
 * Look up the variable sourcepath and try everything in the list in order *
 * if the file isn't in . and it isn't an abs path name.                   *
 *-------------------------------------------------------------------------*/
FILE *
inp_pathopen(char *name, char *mode)
{
    FILE *fp;
    char buf[BSIZE_SP];
    struct variable *v;

    /* If this is an abs pathname, or there is no sourcepath var, just
     * do an fopen.
     */
    if (index(name, DIR_TERM)
	    || !cp_getvar("sourcepath", VT_LIST, (char *) &v))
        return (fopen(name, mode));

    while (v) {
        switch (v->va_type) {
            case VT_STRING:
		cp_wstrip(v->va_string);
		(void) sprintf(buf, "%s%s%s", v->va_string, DIR_PATHSEP, name);
		break;
            case VT_NUM:
		(void) sprintf(buf, "%d%s%s", v->va_num, DIR_PATHSEP, name);
		break;
            case VT_REAL:   /* This is foolish */
		(void) sprintf(buf, "%g%s%s", v->va_real, DIR_PATHSEP, name);
		break;
        }
        if ((fp = fopen(buf, mode)))
            return (fp);
        v = v->va_next;
    }
    return (NULL);
}


/*-------------------------------------------------------------------------
 * Read the entire input file and return  a pointer to the first line of   
 * the linked list of 'card' records in data.  The pointer is stored in
 * *data.
 *-------------------------------------------------------------------------*/
void
inp_readall(FILE *fp, struct line **data)
{
    struct line *end = NULL, *cc = NULL, *prev = NULL, *working, *newcard;
    char *buffer, *s, *t, c;
    /* segfault fix */
    char *copys=NULL;
    int line_number = 1; /* sjb - renamed to avoid confusion with struct line */ 
    FILE *newfp;

    /*   Must set this to NULL or non-tilde includes segfault. -- Tim Molteno   */
    /* copys = NULL; */   /*  This caused a parse error with gcc 2.96.  Why???  */

/*   gtri - modify - 12/12/90 - wbk - read from mailbox if ipc enabled   */
#ifdef XSPICE
    Ipc_Status_t    ipc_status;
    char            ipc_buffer[1025];  /* Had better be big enough */
    int             ipc_len;

  /* First read in all lines & put them in the struct cc */
    while (1) {

        /* If IPC is not enabled, do equivalent of what SPICE did before */
        if(! g_ipc.enabled) {
            buffer = readline(fp);
            if(! buffer)
                break;
        }
        else {
        /* else, get the line from the ipc channel. */
        /* We assume that newlines are not sent by the client */
        /* so we add them here */
            ipc_status = ipc_get_line(ipc_buffer, &ipc_len, IPC_WAIT);
            if(ipc_status == IPC_STATUS_END_OF_DECK) {
                buffer = NULL;
                break;
            }
            else if(ipc_status == IPC_STATUS_OK) {
                buffer = (void *) MALLOC(strlen(ipc_buffer) + 3);
                strcpy(buffer, ipc_buffer);
                strcat(buffer, "\n");
            }
            else {  /* No good way to report this so just die */
                exit(1);
            }
        }

/* gtri - end - 12/12/90 */
#else
    while ((buffer = readline(fp))) {
#endif

#ifdef TRACE
      /* SDB debug statement */
      printf ("in inp_readall, just read '%s' . . .\n", buffer); 
#endif

	/* OK -- now we have loaded the next line into 'buffer'.  Process it. */
        /* If input line is blank, ignore it & continue looping.  */
	if ( (strcmp(buffer,"\n") == 0)
	      || (strcmp(buffer,"\r\n") == 0) ) {
	    continue;
        }


        if (*buffer == '@') {
	    tfree(buffer);		/* was allocated by readline() */
            break;
	}


	/* loop through 'buffer' until end is reached.  Then test for
	   premature end.  If premature end is reached, spew
	   error and zap the line. */
        for (s = buffer; *s && (*s != '\n'); s++);
        if (!*s) {
            fprintf(cp_err, "Warning: premature EOF\n");
        }
        *s = '\0';      /* Zap the newline. */
	
	if(*(s-1) == '\r') /* Zop the carriage return under windows */
	  *(s-1) = '\0';

	/* now handle .include statements */
        if (ciprefix(".include", buffer)) {
	    for (s = buffer; *s && !isspace(*s); s++) /* advance past non-space chars */
                ;
            while (isspace(*s))                       /* now advance past space chars */
                s++;
            if (!*s) {                                /* if at end of line, error */
                fprintf(cp_err,  "Error: .include filename missing\n");
		tfree(buffer);		/* was allocated by readline() */
                continue;
            }                           /* Now s points to first char after .include */
            for (t = s; *t && !isspace(*t); t++)     /* now advance past non-space chars */
                ;
            *t = '\0';                         /* place \0 and end of file name in buffer */
		
	    if (*s == '~') {
		copys = cp_tildexpand(s); /* allocates memory, but can also return NULL */
		if(copys != NULL) {
		    s = copys;		/* reuse s, but remember, buffer still points to allocated memory */
		}
	    }
	    
	    /* open file specified by  .include statement */
            if (!(newfp = inp_pathopen(s, "r"))) { 
                perror(s);
		if(copys) {
			tfree(copys);	/* allocated by the cp_tildexpand() above */
		}
		tfree(buffer);		/* allocated by readline() above */
                continue;
            }
	    
	    if(copys) {
		tfree(copys);		/* allocated by the cp_tildexpand() above */
	    }  
	    
            inp_readall(newfp, &newcard);  /* read stuff in include file into netlist */
            (void) fclose(newfp);

            /* Make the .include a comment */
            *buffer = '*';

	    /* now check if this is the first pass (i.e. end points to null) */
            if (end) {                            /* end already exists */
	      end->li_next = alloc(struct line);  /* create next card */
	      end = end->li_next;                 /* make end point to next card */
            } else {
	      end = cc = alloc(struct line);   /* create the deck & end.  cc will
						point to beginning of deck, end to 
						the end */
            }

	    /* now fill out rest of struct end. */
	    end->li_next = NULL;
	    end->li_error = NULL;
	    end->li_actual = NULL;
            end->li_line = copy(buffer);
            end->li_linenum = line_number++;
            if (newcard) {
                end->li_next = newcard;
            /* Renumber the lines */
	    for (end = newcard; end && end->li_next; end = end->li_next)
                end->li_linenum = line_number++;
	        end->li_linenum = line_number++;	/* SJB - renumber the last line */
              }

            /* Fix the buffer up a bit. */
            (void) strncpy(buffer + 1, "end of:", 7);
        }   /*  end of .include handling  */

	/* now check if this is the first pass (i.e. end points to null) */
        if (end) {                              /* end already exists */
	  end->li_next = alloc(struct line);    /* create next card */
	  end = end->li_next;                   /* point to next card */
        } else {                              /* End doesn't exist.  Create it. */
            end = cc = alloc(struct line);   /* note that cc points to beginning
						of deck, end to the end */
        }

	/* now put buffer into li */
        end->li_next = NULL;
        end->li_error = NULL;
        end->li_actual = NULL;
        end->li_line = buffer;
        end->li_linenum = line_number++;
    }

    if (!end) { /* No stuff here */
        *data = NULL;
        return;
    }             /* end while ((buffer = readline(fp))) */

    /* This should be freed because we are done with it. */
    /* tfree(buffer);  */


    /* Now clean up li: remove comments & stitch together continuation lines. */
    working = cc->li_next;      /* cc points to head of deck.  Start with the
				   next card. */

    /* sjb - strip or convert end-of-line comments.
       This must be cone before stitching continuation lines.
       If the line only contains an end-of-line comment then it is converted
       into a normal comment with a '*' at the start.  This will then get
       stripped in the following code. */			   		   
    inp_stripcomments_deck(working);	

    while (working) {
	for (s = working->li_line; (c = *s) && c <= ' '; s++)
		;

#ifdef TRACE
	/* SDB debug statement */
	printf("In inp_readall, processing linked list element line = %d, s = %s . . . \n", working->li_linenum,s); 
#endif

        switch (c) {
            case '#':
            case '$':
            case '*':
            case '\0':
	      /* this used to be commented out.  Why? */
              /*  prev = NULL; */
                working = working->li_next;  /* for these chars, go to next card */
                break;

	    case '+':   /* handle continuation */
                if (!prev) {
                    working->li_error = copy(
			    "Illegal continuation line: ignored.");
                    working = working->li_next;
                    break;
                }

		/* create buffer and write last and current line into it. */
                buffer = tmalloc(strlen(prev->li_line) + strlen(s) + 2);
                (void) sprintf(buffer, "%s %s", prev->li_line, s + 1); 

                s = prev->li_line;
                prev->li_line = buffer;
                prev->li_next = working->li_next;
                working->li_next = NULL;
                if (prev->li_actual) {
                    for (end = prev->li_actual;
                        end->li_next; end = end->li_next)
                        ;
                    end->li_next = working;
                    tfree(s);
                } else {
                    newcard = alloc(struct line);
                    newcard->li_linenum = prev->li_linenum;
                    newcard->li_line = s;
                    newcard->li_next = working;
		    newcard->li_error = NULL;
		    newcard->li_actual = NULL;
                    prev->li_actual = newcard;
                }
                working = prev->li_next;
                break;

	    default:  /* regular one-line card */
                prev = working;
                working = working->li_next;
                break;
        }
    }

    *data = cc;
    return;
}

/*-------------------------------------------------------------------------*
 *                                                                         *
 *-------------------------------------------------------------------------*/
void
inp_casefix(char *string)
{
#ifdef HAVE_CTYPE_H
    if (string)
	while (*string) {
	    /* Let's make this really idiot-proof. */
#ifdef HAS_ASCII
	    *string = strip(*string);
#endif
	    if (*string == '"') {
		*string++ = ' ';
		while (*string && *string != '"')
		    string++;
		if (*string == '"')
		    *string = ' ';
	    }
	    if (!isspace(*string) && !isprint(*string))
		*string = '_';
	    if (isupper(*string))
		*string = tolower(*string);
	    string++;
	}
    return;
#endif
}


/* Strip all end-of-line comments from a deck */
static void
inp_stripcomments_deck(struct line *deck)
{
    struct line *c=deck;
    while( c!=NULL) {
	inp_stripcomments_line(c->li_line);
	c= c->li_next;
    }
}

/* Strip end of line comment from a string and remove trailing white space
   supports comments that begin with single characters ';'
   or double characters '$ ' or '//' or '--'
   If there is only white space before the end-of-line comment the
   the whole line is converted to a normal comment line (i.e. one that
   begins with a '*').
   BUG: comment characters in side of string literals are not ignored. */  
static void
inp_stripcomments_line(char * s)
{
    char c = ' '; /* anything other than a comment character */
    char * d = s;
    if(*s=='\0') return;	/* empty line */
    
    /* look for comments */
    while((c=*d)!='\0') {
	d++;
	if (*d==';') {	
	    break;
	} else if ((c=='$') && (*d==' ')) {
	    *d--; /* move d back to first comment character */
	    break;
	} else if( (*d==c) && ((c=='/') || (c=='-'))) {
	    *d--; /* move d back to first comment character */
	    break;
	}
    }
    /* d now points to the first comment character of the null at the string end */
    
    /* check for special case of comment at start of line */
    if(d==s) {
	*s = '*'; /* turn into normal comment */
	return;
    }
    
    if(d>s) {
	d--;
	/* d now points to character just before comment */
	
	/* eat white space at end of line */
	while(d>=s) {
	    if( (*d!=' ') && (*d!='\t' ) )
		break;
	    d--;
	}
	d++;
	/* d now points to the first white space character before the
	   end-of-line or end-of-line comment, or it points to the first
	   end-of-line comment character, or to the begining of the line */
    }
       
    /* Check for special case of comment at start of line 
       with or without preceeding white space */
    if(d<=s) {
	*s = '*'; /* turn the whole line into normal comment */
	return;
    }
    
    *d='\0'; /* terminate line in new location */    
}
