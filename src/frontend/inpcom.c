/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * For dealing with spice input decks and command scripts
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

/*  This routine reads a line (of arbitrary length), up to a '\n' or 'EOF'
 *  and returns a pointer to the resulting null terminated string.
 *  The '\n' if found, is included in the returned string.
 *  From: jason@ucbopal.BERKELEY.EDU (Jason Venner)
 *  Newsgroups: net.sources
 */

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
	if (strlen == 0 && c == '\n')
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

/* Look up the variable sourcepath and try everything in the list in order
 * if the file isn't in . and it isn't an abs path name.
 */

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

/* Read the entire input file and return  a pointer to the first line of 
 * the linked list of 'card' records in data.
 */

void
inp_readall(FILE *fp, struct line **data)
{
    struct line *end = NULL, *cc = NULL, *prev = NULL, *working, *newcard;
    char *buffer, *s, *t, c;
    /* segfault fix */
    char *copys=NULL;
    int line = 1;
    FILE *newfp;

/* gtri - modify - 12/12/90 - wbk - read from mailbox if ipc enabled */
#ifdef XSPICE
    Ipc_Status_t    ipc_status;
    char            ipc_buffer[1025];  /* Had better be big enough */
    int             ipc_len;

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
        if (*buffer == '@') {
	    tfree(buffer);		/* was allocated by readline() */
            break;
	}
        for (s = buffer; *s && (*s != '\n'); s++)
            ;
        if (!*s) {
            fprintf(cp_err, "Warning: premature EOF\n");
        }
        *s = '\0';      /* Zap the newline. */

        if (ciprefix(".include", buffer)) {
            for (s = buffer; *s && !isspace(*s); s++)
                ;
            while (isspace(*s))
                s++;
            if (!*s) {
                fprintf(cp_err,  "Error: .include filename missing\n");
		tfree(buffer);		/* was allocated by readline() */
                continue;
            }
            for (t = s; *t && !isspace(*t); t++)
                ;
            *t = '\0';
		
	    if (*s == '~') {
		copys = cp_tildexpand(s); /* allocates memory, but can also return NULL */
		if(copys != NULL) {
		    s = copys;		/* reuse s, but remember, buffer still points to allocated memory */
		}
	    }
				
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
	    
            inp_readall(newfp, &newcard);
            (void) fclose(newfp);

            /* Make the .include a comment */
            *buffer = '*';
            if (end) {
                end->li_next = alloc(struct line);
                end = end->li_next;
            } else {
                end = cc = alloc(struct line);
            }
	    end->li_next = NULL;
	    end->li_error = NULL;
	    end->li_actual = NULL;
            end->li_line = copy(buffer);
            end->li_linenum = line++;
            end->li_next = newcard;

            /* Renumber the lines */
            for (end = newcard; end && end->li_next; end = end->li_next)
                end->li_linenum = line++;

            /* Fix the buffer up a bit. */
            (void) strncpy(buffer + 1, "end of:", 7);
        }

        if (end) {
            end->li_next = alloc(struct line);
            end = end->li_next;
        } else {
            end = cc = alloc(struct line);
        }
        end->li_next = NULL;
        end->li_error = NULL;
        end->li_actual = NULL;
        end->li_line = buffer;
        end->li_linenum = line++;
    }
    if (!end) { /* No stuff here */
        *data = NULL;
        return;
    }

    /* Now make logical lines. */
    working = cc->li_next;      /* Skip title. */

    while (working) {
	for (s = working->li_line; (c = *s) && c <= ' '; s++)
		;
        switch (c) {
            case '#':
            case '$':
            case '*':
            case '\0':
		/*
                prev = NULL;
		*/
                working = working->li_next;
                break;
            case '+':
                if (!prev) {
                    working->li_error = copy(
			    "Illegal continuation line: ignored.");
                    working = working->li_next;
                    break;
                }
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
            default:
                prev = working;
                working = working->li_next;
                break;
        }
    }

    *data = cc;
    return;
}


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
