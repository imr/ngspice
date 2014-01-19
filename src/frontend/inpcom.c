/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
  For dealing with spice input decks and command scripts

  Central function is inp_readall()
*/

#include "ngspice/ngspice.h"

#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "ngspice/compatmode.h"

#include <limits.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>

#if !defined(__MINGW32__) && !defined(_MSC_VER)
#include <unistd.h>
#endif

#include "inpcom.h"
#include "variable.h"
#include "subckt.h"
#include "../misc/util.h" /* ngdirname() */
#include "ngspice/stringutil.h"
#include "ngspice/wordlist.h"

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include new stuff */
#include "ngspice/ipctiein.h"
#include "ngspice/enh.h"
/* gtri - end - 12/12/90 */
#endif

/* SJB - Uncomment this line for debug tracing */
/*#define TRACE*/

/* globals -- wanted to avoid complicating inp_readall interface */
#define N_LIBRARIES       1000
#define N_SECTIONS        1000
#define N_PARAMS          1000
#define N_SUBCKT_W_PARAMS 4000

#define VALIDCHARS "!$%_#?@.[]&"

static struct library {
    char *realpath;
    char *habitat;
    struct line *deck;
} libraries[N_LIBRARIES];

static int  num_libraries;

struct names {
    char *names[N_SUBCKT_W_PARAMS];
    int  num_names;
};

struct function_env
{
    struct function_env *up;

    struct function {
        struct function *next;
        char *name;
        char *macro;
        char *params[N_PARAMS];
        int   num_parameters;
    } *functions;
};

struct func_temper
{
    char* funcname;
    int subckt_depth;
    int subckt_count;
    struct func_temper *next;
};

extern void line_free_x(struct line *deck, bool recurse);

static COMPATMODE_T inp_compat_mode;

/* Collect information for dynamic allocation of numparam arrays */
/* number of lines in input deck */
int dynmaxline;  /* inpcom.c 1529 */
/* number of lines in deck after expansion */
int dynMaxckt = 0; /* subckt.c 307 */
/* number of parameter substitutions */
long dynsubst; /* spicenum.c 221 */

/* Expression handling with 'temper' parameter required */
bool expr_w_temper = FALSE;


static char *readline(FILE *fd);
static int  get_number_terminals(char *c);
static void inp_stripcomments_deck(struct line *deck, bool cs);
static void inp_stripcomments_line(char *s, bool cs);
static void inp_fix_for_numparam(struct names *subckt_w_params, struct line *deck);
static void inp_remove_excess_ws(struct line *deck);
static void expand_section_references(struct line *deck, char *dir_name);
static void inp_grab_func(struct function_env *, struct line *deck);
static void inp_fix_inst_calls_for_numparam(struct names *subckt_w_params, struct line *deck);
static void inp_expand_macros_in_func(struct function_env *);
static struct line *inp_expand_macros_in_deck(struct function_env *, struct line *deck);
static void inp_fix_param_values(struct line *deck);
static void inp_reorder_params(struct names *subckt_w_params, struct line *deck, struct line *list_head, struct line *end);
static int  inp_split_multi_param_lines(struct line *deck, int line_number);
static void inp_sort_params(struct line *start_card, struct line *end_card, struct line *card_bf_start, struct line *s_c, struct line *e_c);
static char *inp_remove_ws(char *s);
static void inp_compat(struct line *deck);
static void inp_bsource_compat(struct line *deck);
static void inp_temper_compat(struct line *card);
static void inp_dot_if(struct line *deck);
static char *inp_modify_exp(char* expression);
static void inp_new_func(char *funcname, char *funcbody, struct line *card,
                         struct func_temper **new_func, int *sub_count, int subckt_depth);
static void inp_rem_func(struct func_temper **new_func);

static bool chk_for_line_continuation(char *line);
static void comment_out_unused_subckt_models(struct line *start_card, int no_of_lines);
static void inp_fix_macro_param_func_paren_io(struct line *begin_card);
static void inp_fix_gnd_name(struct line *deck);
static void inp_chk_for_multi_in_vcvs(struct line *deck, int *line_number);
static void inp_add_control_section(struct line *deck, int *line_number);
static char *get_quoted_token(char *string, char **token);
static void replace_token(char *string, char *token, int where, int total);
static void inp_add_series_resistor(struct line *deck);
static void subckt_params_to_param(struct line *deck);
static void inp_fix_temper_in_param(struct line *deck);

static char *skip_back_non_ws(char *d) { while (d[-1] && !isspace(d[-1])) d--; return d; }
static char *skip_back_ws(char *d)     { while (isspace(d[-1]))           d--; return d; }
static char *skip_non_ws(char *d)      { while (*d && !isspace(*d)) d++; return d; }
static char *skip_ws(char *d)          { while (isspace(*d))        d++; return d; }

static char *skip_back_non_ws_(char *d, char *start) { while (d > start && !isspace(d[-1])) d--; return d; }
static char *skip_back_ws_(char *d, char *start)     { while (d > start && isspace(d[-1])) d--; return d; }

static char *inp_pathresolve(const char *name);
static char *inp_pathresolve_at(char *name, char *dir);
void tprint(struct line *deck);

struct inp_read_t
{ struct line *cc;
    int line_number;
};

static struct inp_read_t inp_read(FILE *fp, int call_depth, char *dir_name, bool comfile, bool intfile);


#ifndef XSPICE
static void inp_poly_err(struct line *deck);
#endif


static struct line *
xx_new_line(struct line *next, char *line, int linenum, int linenum_orig)
{
    struct line *x = TMALLOC(struct line, 1);

    x->li_next = next;
    x->li_error = NULL;
    x->li_actual = NULL;
    x->li_line = line;
    x->li_linenum = linenum;
    x->li_linenum_orig = linenum_orig;

    return x;
}


static struct library *
new_lib(void)
{
    if (num_libraries >= N_LIBRARIES) {
        fprintf(stderr, "ERROR, N_LIBRARIES overflow\n");
        controlled_exit(EXIT_FAILURE);
    }

    return & libraries[num_libraries++];
}


static void
delete_libs(void)
{
    int i;

    for (i = 0; i < num_libraries; i++) {
        tfree(libraries[i].realpath);
        tfree(libraries[i].habitat);
        line_free_x(libraries[i].deck, TRUE);
    }
}


static struct library *
find_lib(char *name)
{
    int i;

    for (i = 0; i < num_libraries; i++)
        if (cieq(libraries[i].realpath, name))
            return & libraries[i];

    return NULL;
}


static struct line *
find_section_definition(struct line *c, char *name)
{
    for (; c; c = c->li_next) {

        char *line = c->li_line;

        if (ciprefix(".lib", line)) {

            char *s, *t, *y;

            s = skip_non_ws(line);
            while (isspace(*s) || isquote(*s))
                s++;
            for (t = s; *t && !isspace(*t) && !isquote(*t); t++)
                ;
            y = t;
            while (isspace(*y) || isquote(*y))
                y++;

            if (!*y) {
                /* library section definition: `.lib <section-name>' .. `.endl' */

                char keep_char = *t;
                *t = '\0';

                if (strcasecmp(name, s) == 0) {
                    *t = keep_char;
                    return c;
                }

                *t = keep_char;
            }
        }
    }

    return NULL;
}


static struct library *
read_a_lib(char *y, char *dir_name)
{
    char *yy, *y_resolved;

    struct library *lib;

    y_resolved = inp_pathresolve_at(y, dir_name);

    if (!y_resolved) {
        fprintf(cp_err, "Error: Could not find library file %s\n", y);
        return NULL;
    }

#if defined(__MINGW32__) || defined(_MSC_VER)
    yy = _fullpath(NULL, y_resolved, 0);
#else
    yy = realpath(y_resolved, NULL);
#endif

    if (!yy) {
        fprintf(cp_err, "Error: Could not `realpath' library file %s\n", y);
        controlled_exit(EXIT_FAILURE);
    }

    lib = find_lib(yy);

    if (!lib) {

        FILE *newfp = fopen(y_resolved, "r");

        if (!newfp) {
            fprintf(cp_err, "Error: Could not open library file %s\n", y);
            return NULL;
        }

        /* lib points to a new entry in global lib array libraries[N_LIBRARIES] */
        lib = new_lib();

        lib->realpath = strdup(yy);
        lib->habitat = ngdirname(yy);

        lib->deck = inp_read(newfp, 1 /*dummy*/, lib->habitat, FALSE, FALSE) . cc;

        fclose(newfp);
    }

    free(yy);
    free(y_resolved);

    return lib;
}


static struct names *
new_names(void)
{
    struct names *p = TMALLOC(struct names, 1);
    p -> num_names = 0;

    return p;
}


static void
delete_names(struct names *p)
{
    int i;
    for (i = 0; i < p->num_names; i++)
        tfree(p->names[i]);
    tfree(p);
}



/* line1
   + line2
   ---->
   line1 line 2
   Proccedure: store regular card in prev, skip comment lines (*..) and some others
   */

static void
inp_stitch_continuation_lines(struct line *working)
{
    struct line *prev = NULL;

    while (working) {
        char *s, c, *buffer;

        for (s = working->li_line; (c = *s) != '\0' && c <= ' '; s++)
            ;

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_read, processing linked list element line = %d, s = %s . . . \n", working->li_linenum, s);
#endif

        switch (c) {
        case '#':
        case '$':
        case '*':
        case '\0':
            /* skip these cards, and keep prev as the last regular card */
            working = working->li_next;  /* for these chars, go to next card */
            break;

        case '+':   /* handle continuation */
            if (!prev) {
                working->li_error = copy("Illegal continuation line: ignored.");
                working = working->li_next;
                break;
            }

            /* We now may have lept over some comment lines, which are located among
            the continuation lines. We have to delete them here to prevent a memory leak */
            while (prev->li_next != working) {
                struct line *tmpl = prev->li_next->li_next;
                line_free_x(prev->li_next, FALSE);
                prev->li_next = tmpl;
            }

            /* create buffer and write last and current line into it. */
            buffer = TMALLOC(char, strlen(prev->li_line) + strlen(s) + 2);
            (void) sprintf(buffer, "%s %s", prev->li_line, s + 1);

            /* replace prev->li_line by buffer */
            s = prev->li_line;
            prev->li_line = buffer;
            prev->li_next = working->li_next;
            working->li_next = NULL;
            /* add original line to prev->li_actual */
            if (prev->li_actual) {
                struct line *end;
                for (end = prev->li_actual; end->li_next; end = end->li_next)
                    ;
                end->li_next = working;
                tfree(s);
            } else {
                prev->li_actual = xx_new_line(working, s, prev->li_linenum, 0);
            }
            working = prev->li_next;
            break;

        default:  /* regular one-line card */
            prev = working;
            working = working->li_next;
            break;
        }
    }
}


/*
 * search for `=' assignment operator
 *   take care of `!=' `<=' `==' and `>='
 */

static char *
find_assignment(char *str)
{
    char *p = str;

    while ((p = strchr(p, '=')) != NULL) {

        // check for equality '=='
        if (p[1] == '=') {
            p += 2;
            continue;
        }

        // check for '!=', '<=', '>='
        if (p > str)
            if (p[-1] == '!' || p[-1] == '<' || p[-1] == '>') {
                p += 1;
                continue;
            }

        return p;
    }

    return NULL;
}


/*-------------------------------------------------------------------------
  Read the entire input file and return  a pointer to the first line of
  the linked list of 'card' records in data.  The pointer is stored in
  *data.
  Called from fcn inp_spsource() in inp.c to load circuit or command files.
  Called from fcn com_alter_mod() in device.c to load model files.
  Called from here to load .library or .include files.

  Procedure:
  read in all lines & put them in the struct cc
  read next line
  process .TITLE line
  store contents in string new_title
  process .lib lines
  read file and library name, open file using fcn inp_pathopen()
  read file contents and put into struct libraries[].deck, one entry per .lib line
  process .inc lines
  read file and library name, open file using fcn inp_pathopen()
  read file contents and add lines to cc
  make line entry lower case
  allow for shell end of line continuation (\\)
  add '+' to beginning of next line
  add line entry to list cc
  add '.global gnd'
  add libraries
  find library section
  add lines
  add .end card
  strip end-of-line comments
  make continuation lines a single line
  *** end of processing for command files ***
  start preparation of input deck for numparam
  ...
  debug printout to debug-out.txt
  *-------------------------------------------------------------------------*/

struct line *
inp_readall(FILE *fp, char *dir_name, bool comfile, bool intfile)
{
    struct line *cc;
    struct inp_read_t rv;

    num_libraries = 0;
    inp_compat_mode = ngspice_compat_mode();

    rv = inp_read(fp, 0, dir_name, comfile, intfile);
    cc = rv . cc;

    /* The following processing of an input file is not required for command files
       like spinit or .spiceinit, so return command files here. */

    if (!comfile && cc) {

        unsigned int no_braces; /* number of '{' */
        size_t max_line_length; /* max. line length in input deck */
        struct line *tmp_ptr1, *end;
        struct names *subckt_w_params = new_names();

        struct line *working = cc->li_next;

        delete_libs();
        inp_fix_for_numparam(subckt_w_params, working);


        inp_remove_excess_ws(working);

        comment_out_unused_subckt_models(working, rv . line_number);

        subckt_params_to_param(working);

        rv . line_number = inp_split_multi_param_lines(working, rv . line_number);

        inp_fix_macro_param_func_paren_io(working);
        inp_fix_temper_in_param(working);

        inp_expand_macros_in_deck(NULL, working);
        inp_fix_param_values(working);

        for (end = cc; end->li_next; end = end->li_next)
            ;

        inp_reorder_params(subckt_w_params, working, cc, end);
        inp_fix_inst_calls_for_numparam(subckt_w_params, working);

        delete_names(subckt_w_params);
        subckt_w_params = NULL;

        inp_fix_gnd_name(working);
        inp_chk_for_multi_in_vcvs(working, &rv. line_number);

        if (cp_getvar("addcontrol", CP_BOOL, NULL))
            inp_add_control_section(working, &rv . line_number);
#ifndef XSPICE
        inp_poly_err(working);
#endif
        if (inp_compat_mode != COMPATMODE_SPICE3) {
            /* Do all the compatibility stuff here */
            working = cc->li_next;
            /* E, G, L, R, C compatibility transformations */
            inp_compat(working);
            working = cc->li_next;
            /* B source numparam compatibility transformation */
            inp_bsource_compat(working);
            inp_dot_if(working);
            inp_temper_compat(working);
        }

        inp_add_series_resistor(working);

        /* get max. line length and number of lines in input deck,
           and renumber the lines,
           count the number of '{' per line as an upper estimate of the number
           of parameter substitutions in a line*/
        dynmaxline = 0;
        max_line_length = 0;
        no_braces = 0;
        for (tmp_ptr1 = cc; tmp_ptr1; tmp_ptr1 = tmp_ptr1->li_next) {
            char *s;
            unsigned int braces_per_line = 0;
            /* count number of lines */
            dynmaxline++;
            /* renumber the lines of the processed input deck */
            tmp_ptr1->li_linenum = dynmaxline;
            if (max_line_length < strlen(tmp_ptr1->li_line))
                max_line_length = strlen(tmp_ptr1->li_line);
            /* count '{' */
            for (s = tmp_ptr1->li_line; *s; s++)
                if (*s == '{')
                    braces_per_line++;
            if (no_braces <  braces_per_line)
                no_braces = braces_per_line;
        }

        if (ft_ngdebug) {
            /*debug: print into file*/
            FILE *fd = fopen("debug-out.txt", "w");
            struct line *t;
            fprintf(fd, "**************** uncommented deck **************\n\n");
            /* always print first line */
            fprintf(fd, "%6d  %6d  %s\n", cc->li_linenum_orig, cc->li_linenum, cc->li_line);
            /* here without out-commented lines */
            for (t = cc->li_next; t; t = t->li_next) {
                if (*(t->li_line) == '*')
                    continue;
                fprintf(fd, "%6d  %6d  %s\n", t->li_linenum_orig, t->li_linenum, t->li_line);
            }
            fprintf(fd, "\n****************** complete deck ***************\n\n");
            /* now completely */
            for (t = cc; t; t = t->li_next)
                fprintf(fd, "%6d  %6d  %s\n", t->li_linenum_orig, t->li_linenum, t->li_line);
            fclose(fd);

            fprintf(stdout, "max line length %d, max subst. per line %d, number of lines %d\n",
                    (int) max_line_length, no_braces, dynmaxline);
        }
    }

    return cc;
}


struct inp_read_t
inp_read(FILE *fp, int call_depth, char *dir_name, bool comfile, bool intfile)
/* fp: in, pointer to file to be read,
   call_depth: in, nested call to fcn
   dir_name: in, name of directory of file to be read
   comfile: in, TRUE if command file (e.g. spinit, .spiceinit)
   intfile: in, TRUE if deck is generated from internal circarray
*/
{
    struct inp_read_t rv;
    struct line *end = NULL, *cc = NULL;
    char *buffer = NULL;
    /* segfault fix */
#ifdef XSPICE
    char big_buff[5000];
    int line_count = 0;
#endif
    char *new_title = NULL;
    int line_number = 1; /* sjb - renamed to avoid confusion with struct line */
    int line_number_orig = 1;
    int cirlinecount = 0; /* length of circarray */
    static int is_control = 0; /* We are reading from a .control section */

    bool found_end = FALSE, shell_eol_continuation = FALSE;

    /* First read in all lines & put them in the struct cc */
    for (;;) {
        /* derive lines from circarray */
        if (intfile) {
            buffer = circarray[cirlinecount++];
            if (!buffer) {
                tfree(circarray);
                break;
            }
        }
        /* read lines from file fp */
        else {

#ifdef XSPICE
            /* gtri - modify - 12/12/90 - wbk - read from mailbox if ipc enabled */

            /* If IPC is not enabled, do equivalent of what SPICE did before */
            if (! g_ipc.enabled) {
                if (call_depth == 0 && line_count == 0) {
                    line_count++;
                    if (fgets(big_buff, 5000, fp))
                        buffer = copy(big_buff);
                } else {
                    buffer = readline(fp);
                    if (!buffer)
                        break;
                }
            } else {
                /* else, get the line from the ipc channel. */
                /* We assume that newlines are not sent by the client */
                /* so we add them here */
                char         ipc_buffer[1025];  /* Had better be big enough */
                int          ipc_len;
                Ipc_Status_t ipc_status =
                    ipc_get_line(ipc_buffer, &ipc_len, IPC_WAIT);
                if (ipc_status == IPC_STATUS_END_OF_DECK) {
                    buffer = NULL;
                    break;
                } else if (ipc_status == IPC_STATUS_OK) {
                    buffer = TMALLOC(char, strlen(ipc_buffer) + 3);
                    strcpy(buffer, ipc_buffer);
                    strcat(buffer, "\n");
                } else {            /* No good way to report this so just die */
                    controlled_exit(EXIT_FAILURE);
                }
            }

            /* gtri - end - 12/12/90 */
#else

            buffer = readline(fp);
            if(!buffer)
                break;

#endif
        }

#ifdef TRACE
        /* SDB debug statement */
        printf("in inp_read, just read   %s", buffer);
#endif

        if (!buffer)
            continue;

        /* OK -- now we have loaded the next line into 'buffer'.  Process it. */
        /* If input line is blank, ignore it & continue looping.  */
        if ((strcmp(buffer, "\n") == 0) || (strcmp(buffer, "\r\n") == 0))
            if (call_depth != 0 || (call_depth == 0 && cc != NULL)) {
                line_number_orig++;
                tfree(buffer);  /* was allocated by readline() */
                continue;
            }

        if (*buffer == '@') {
            tfree(buffer);      /* was allocated by readline() */
            break;
        }

        /* now check if we are in a .control section */
        if (ciprefix(".control", buffer))
            is_control++;
        else if (ciprefix(".endc", buffer))
            is_control--;

        /* now handle .title statement */
        if (ciprefix(".title", buffer)) {
            char *s;
            s = skip_non_ws(buffer);   /* skip over .title */
            s = skip_ws(s);            /* advance past space chars */

            /* only the last title line remains valid */
            tfree(new_title);
            new_title = copy(s);
            if ((s = strchr(new_title, '\n')) != NULL)
                *s = ' ';
            *buffer = '*';      /* change .TITLE line to comment line */
        }

        /* now handle old style .lib entries */
        /* new style .lib entries handling is in expand_section_references() */
        if (ciprefix(".lib", buffer))
            if (inp_compat_mode == COMPATMODE_PS) {
                /* compatibility mode,
                 *   this is neither a libray section definition nor a reference
                 * interpret as old style
                 *   .lib <file name> (no lib name given)
                 */
                char *s = skip_non_ws(buffer); /* skip over .lib */
                fprintf(cp_err, "  File included as:   .inc %s\n", s);
                memcpy(buffer, ".inc", 4);
            }

        /* now handle .include statements */
        if (ciprefix(".include", buffer) || ciprefix(".inc", buffer)) {

            char *y = NULL;
            char *s, *t;

            struct line *newcard;

            inp_stripcomments_line(buffer, FALSE);

            s = skip_non_ws(buffer);               /* advance past non-space chars */

            t = get_quoted_token(s, &y);

            if (!y) {
                fprintf(cp_err, "Error: .include filename missing\n");
                tfree(buffer);  /* was allocated by readline() */
                controlled_exit(EXIT_FAILURE);
            }

            {
                char *y_resolved = inp_pathresolve_at(y, dir_name);
                char *y_dir_name;
                FILE *newfp;

                if (!y_resolved) {
                    fprintf(cp_err, "Error: Could not find include file %s\n", y);
                    rv . line_number = line_number;
                    rv . cc = NULL;
                    return rv;
                }

                newfp = fopen(y_resolved, "r");

                if (!newfp) {
                    fprintf(cp_err, "Error: .include statement failed.\n");
                    tfree(buffer);          /* allocated by readline() above */
                    controlled_exit(EXIT_FAILURE);
                }

                y_dir_name = ngdirname(y_resolved);

                newcard = inp_read(newfp, call_depth+1, y_dir_name, FALSE, FALSE) . cc;  /* read stuff in include file into netlist */

                tfree(y_dir_name);
                tfree(y_resolved);

                (void) fclose(newfp);
            }

            /* Make the .include a comment */
            *buffer = '*';

            /* append `buffer' to the (cc, end) chain of decks */
            {
                struct line *x = xx_new_line(NULL, copy(buffer), line_number, line_number);

                if (end)
                    end->li_next = x;
                else
                    cc = x;

                end = x;

                line_number++;
            }

            if (newcard) {
                int line_number_inc = 1;
                end->li_next = newcard;
                /* Renumber the lines */
                for (end = newcard; end && end->li_next; end = end->li_next) {
                    end->li_linenum = line_number++;
                    end->li_linenum_orig = line_number_inc++;
                }
                end->li_linenum = line_number++;        /* SJB - renumber the last line */
                end->li_linenum_orig = line_number_inc++;       /* SJB - renumber the last line */
            }

            /* Fix the buffer up a bit. */
            (void) strncpy(buffer + 1, "end of: ", 8);
        }   /*  end of .include handling  */

        /* loop through 'buffer' until end is reached.  Then test for
           premature end.  If premature end is reached, spew
           error and zap the line. */
        {
            char *s;
            /* no lower case letters for lines beginning with: */
            if ( !ciprefix("write", buffer) &&
                 !ciprefix("wrdata", buffer) &&
                 !ciprefix(".lib", buffer) &&
                 !ciprefix(".inc", buffer) &&
                 !ciprefix("codemodel", buffer) &&
                 !ciprefix("echo", buffer) &&
                 !ciprefix("shell", buffer) &&
                 !ciprefix("source", buffer) &&
                 !ciprefix("load", buffer)
                )
            {
                /* lower case for all lines (exceptions see above!) */
                for (s = buffer; *s && (*s != '\n'); s++)
                    *s = (char) tolower(*s);
            } else {
                /* exclude some commands to preserve filename case */
                for (s = buffer; *s && (*s != '\n'); s++)
                    ;
            }

            if (!*s) {
                // fprintf(cp_err, "Warning: premature EOF\n");
            }
            *s = '\0';      /* Zap the newline. */

            if ((s-1) >= buffer && *(s-1) == '\r') /* Zop the carriage return under windows */
                *(s-1) = '\0';
        }

        /* find the true .end command out of .endc, .ends, .endl, .end (comments may follow) */
        if (ciprefix(".end", buffer))
            if ((buffer[4] == '\0') || isspace(buffer[4])) {
                found_end = TRUE;
                *buffer   = '*';
            }

        if (shell_eol_continuation) {
            char *new_buffer = TMALLOC(char, strlen(buffer) + 2);
            sprintf(new_buffer, "+%s", buffer);

            tfree(buffer);
            buffer = new_buffer;
        }

        /* If \\ at end of line is found, next line in loop will get + (see code above) */
        shell_eol_continuation = chk_for_line_continuation(buffer);

        {
            struct line *x = xx_new_line(NULL, copy(buffer), line_number++, line_number_orig++);

            if (end)
                end->li_next = x;
            else
                cc = x;

            end = x;
        }

        tfree(buffer);
    }  /* end while ((buffer = readline(fp)) != NULL) */

    if (!end) /* No stuff here */
    {
        rv . line_number = line_number;
        rv . cc = NULL;
        return rv;
    }

    if (call_depth == 0 && !comfile) {
        cc->li_next = xx_new_line(cc->li_next, copy(".global gnd"), 1, 0);

        if (inp_compat_mode == COMPATMODE_ALL ||
            inp_compat_mode == COMPATMODE_HS  ||
            inp_compat_mode == COMPATMODE_NATIVE)
        {
            /* process all library section references */
            expand_section_references(cc, dir_name);
        }
    }

    /*
      add a terminal ".end" card
    */

    if (call_depth == 0 && !comfile) {
        if (found_end == TRUE) {
            struct line *x = xx_new_line(NULL, copy(".end"), line_number++, line_number_orig++);
            end->li_next = x;
            end = x;
        }
    }

    /* Replace first line with the new title, if available */
    if (call_depth == 0 && !comfile && new_title) {
        tfree(cc->li_line);
        cc->li_line = new_title;
    }

    /* Strip or convert end-of-line comments.
       Afterwards stitch the continuation lines.
       If the line only contains an end-of-line comment then it is converted
       into a normal comment with a '*' at the start.  Some special handling
       if this is a command file or called from within a .control section. */
    inp_stripcomments_deck(cc->li_next, comfile || is_control);

    inp_stitch_continuation_lines(cc->li_next);

    rv . line_number = line_number;
    rv . cc = cc;
    return rv;
}


static bool
is_absolute_pathname(const char *p)
{
#if defined(__MINGW32__) || defined(_MSC_VER)
    /* /... or \... or D:\... or D:/... */
    return
        p[0] == DIR_TERM  ||
        p[0] == DIR_TERM_LINUX  ||
        (isalpha(p[0]) && p[1] == ':' &&
         (p[2] == DIR_TERM_LINUX || p[2] == DIR_TERM));
#else
    return
        p[0] == DIR_TERM;
#endif
}


#if 0

static bool
is_plain_filename(const char *p)
{
#if defined(__MINGW32__) || defined(_MSC_VER)
    return
        !strchr(p, DIR_TERM) &&
        !strchr(p, DIR_TERM_LINUX);
#else
    return
        !strchr(p, DIR_TERM);
#endif
}

#endif


FILE *
inp_pathopen(char *name, char *mode)
{
    char *path = inp_pathresolve(name);

    if (path) {
        FILE *fp = fopen(path, mode);
        tfree(path);
        return fp;
    }

    return NULL;
}


/*-------------------------------------------------------------------------*
  Look up the variable sourcepath and try everything in the list in order
  if the file isn't in . and it isn't an abs path name.
  *-------------------------------------------------------------------------*/

static char *
inp_pathresolve(const char *name)
{
    char buf[BSIZE_SP];
    struct variable *v;
    struct stat st;

#if defined(__MINGW32__) || defined(_MSC_VER)

    /* If variable 'mingwpath' is set: convert mingw /d/... to d:/... */
    if (cp_getvar("mingwpath", CP_BOOL, NULL) && name[0] == DIR_TERM_LINUX && isalpha(name[1]) && name[2] == DIR_TERM_LINUX) {
        strcpy(buf, name);
        buf[0] = buf[1];
        buf[1] = ':';
        return inp_pathresolve(buf);
    }

#endif

    /* just try it */
    if (stat(name, &st) == 0)
        return copy(name);

    /* fail if this was an absolute filename or if there is no sourcepath var */
    if (is_absolute_pathname(name) || !cp_getvar("sourcepath", CP_LIST, &v))
        return NULL;

    for (; v; v = v->va_next) {

        switch (v->va_type) {
        case CP_STRING:
            cp_wstrip(v->va_string);
            (void) sprintf(buf, "%s%s%s", v->va_string, DIR_PATHSEP, name);
            break;
        case CP_NUM:
            (void) sprintf(buf, "%d%s%s", v->va_num, DIR_PATHSEP, name);
            break;
        case CP_REAL:           /* This is foolish */
            (void) sprintf(buf, "%g%s%s", v->va_real, DIR_PATHSEP, name);
            break;
        default:
            fprintf(stderr, "ERROR: enumeration value `CP_BOOL' or `CP_LIST' not handled in inp_pathresolve\nAborting...\n");
            controlled_exit(EXIT_FAILURE);
            break;
        }

        if (stat(buf, &st) == 0)
            return copy(buf);
    }

    return (NULL);
}


static char *
inp_pathresolve_at(char *name, char *dir)
{
    char buf[BSIZE_SP], *end;

    /* if name is an absolute path name,
     *   or if we haven't anything to prepend anyway
     */

    if (is_absolute_pathname(name) || !dir || !dir[0])
        return inp_pathresolve(name);

    if (name[0] == '~' && name[1] == '/') {
        char *y = cp_tildexpand(name);
        if (y) {
            char *r = inp_pathresolve(y);
            tfree(y);
            return r;
        }
    }

    /* concatenate them */

    strcpy(buf, dir);

    end = strchr(buf, '\0');
    if (end[-1] != DIR_TERM)
        *end++ = DIR_TERM;

    strcpy(end, name);

    return inp_pathresolve(buf);
}


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

    strlen = 0;
    memlen = STRGROW;
    strptr = TMALLOC(char, memlen);
    memlen -= 1;                /* Save constant -1's in while loop */

    while ((c = getc(fd)) != EOF) {

        if (strlen == 0 && (c == '\t' || c == ' ')) /* Leading spaces away */
            continue;

        strptr[strlen++] = (char) c;

        if (strlen >= memlen) {
            memlen += STRGROW;
            if ((strptr = TREALLOC(char, strptr, memlen + 1)) == NULL)
                return (NULL);
        }

        if (c == '\n')
            break;
    }

    if (!strlen) {
        tfree(strptr);
        return (NULL);
    }

    // strptr[strlen] = '\0';
    /* Trim the string */
    strptr = TREALLOC(char, strptr, strlen + 1);
    strptr[strlen] = '\0';

    return (strptr);
}


/* replace "gnd" by " 0 "
   Delimiters of gnd may be ' ' or ',' or '(' or ')' */

static void
inp_fix_gnd_name(struct line *c)
{
    for (; c; c = c->li_next) {

        char *gnd = c->li_line;

        // if there is a comment or no gnd, go to next line
        if ((*gnd == '*') || !strstr(gnd, "gnd"))
            continue;

        // replace "?gnd?" by "? 0 ?", ? being a ' '  ','  '('  ')'.
        while ((gnd = strstr(gnd, "gnd")) != NULL) {
            if ((isspace(gnd[-1]) || gnd[-1] == '(' || gnd[-1] == ',') &&
                (isspace(gnd[3]) || gnd[3] == ')' || gnd[3] == ',')) {
                memcpy(gnd, " 0 ", 3);
            }
            gnd += 3;
        }

        // now remove the extra white spaces around 0
        c->li_line = inp_remove_ws(c->li_line);
    }
}


static void
inp_chk_for_multi_in_vcvs(struct line *c, int *line_number)
{
    int skip_control = 0;

    for (; c; c = c->li_next) {

        char *line = c->li_line;

        /* there is no e source inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (*line == 'e') {

            char *bool_ptr;

            if ((bool_ptr = strstr(line, "nand(")) != NULL ||
                (bool_ptr = strstr(line, "and(")) != NULL ||
                (bool_ptr = strstr(line, "nor(")) != NULL ||
                (bool_ptr = strstr(line, "or(")) != NULL)
            {
                struct line *a_card, *model_card, *next_card;
                char *str_ptr1, *str_ptr2, keep, *comma_ptr, *xy_values1[5], *xy_values2[5];
                char *node_str, *ctrl_node_str, *xy_str1, *model_name, *fcn_name;
                char big_buf[1000];
                int  xy_count1, xy_count2;

                str_ptr1 = skip_non_ws(line);
                model_name = copy_substring(line, str_ptr1);

                str_ptr1 = skip_ws(str_ptr1);
                str_ptr2 = skip_back_ws(bool_ptr);
                keep = *str_ptr2;
                *str_ptr2 = '\0';
                node_str  = strdup(str_ptr1);
                *str_ptr2 = keep;

                str_ptr1 = bool_ptr;
                while (*++str_ptr1 != '(')
                    ;
                fcn_name = copy_substring(bool_ptr, str_ptr1);
                str_ptr1  = strchr(str_ptr1, ')');
                comma_ptr = strchr(line, ',');
                if (!str_ptr1 || !comma_ptr) {
                    fprintf(stderr, "ERROR: mal formed line: %s\n", line);
                    controlled_exit(EXIT_FAILURE);
                }
                str_ptr1 = skip_ws(str_ptr1 + 1);
                xy_str1 = skip_back_ws(comma_ptr);
                if (xy_str1[-1] == '}') {
                    while (*--xy_str1 != '{')
                        ;
                } else {
                    xy_str1 = skip_back_non_ws(xy_str1);
                }
                str_ptr2 = skip_back_ws(xy_str1);
                keep = *str_ptr2;
                *str_ptr2 = '\0';
                ctrl_node_str = strdup(str_ptr1);
                *str_ptr2 = keep;

                str_ptr1 = skip_ws(comma_ptr + 1);
                if (*str_ptr1 == '{') {
                    while (*str_ptr1++ != '}')
                        ;
                } else {
                    str_ptr1 = skip_non_ws(str_ptr1);
                }
                keep = *str_ptr1;
                *str_ptr1 = '\0';
                xy_count1 = get_comma_separated_values(xy_values1, xy_str1);
                *str_ptr1 = keep;

                str_ptr1 = skip_ws(str_ptr1);
                xy_count2 = get_comma_separated_values(xy_values2, str_ptr1);

                // place restrictions on only having 2 point values; this can change later
                if (xy_count1 != 2 && xy_count2 != 2)
                    fprintf(stderr, "ERROR: only expecting 2 pair values for multi-input vcvs!\n");

                sprintf(big_buf, "%s %%vd[ %s ] %%vd( %s ) %s",
                        model_name, ctrl_node_str, node_str, model_name);
                a_card = xx_new_line(NULL, copy(big_buf), *(line_number)++, 0);
                *a_card->li_line = 'a';

                sprintf(big_buf, ".model %s multi_input_pwl ( x = [%s %s] y = [%s %s] model = \"%s\" )",
                        model_name, xy_values1[0], xy_values2[0],
                        xy_values1[1], xy_values2[1], fcn_name);
                model_card = xx_new_line(NULL, copy(big_buf), (*line_number)++, 0);

                tfree(model_name);
                tfree(node_str);
                tfree(fcn_name);
                tfree(ctrl_node_str);
                tfree(xy_values1[0]);
                tfree(xy_values1[1]);
                tfree(xy_values2[0]);
                tfree(xy_values2[1]);

                *c->li_line = '*';
                next_card   = c->li_next;
                c->li_next  = a_card;
                a_card->li_next     = model_card;
                model_card->li_next = next_card;
            }
        }
    }
}


static void
inp_add_control_section(struct line *deck, int *line_number)
{
    struct line *c, *prev_card = NULL;
    bool        found_control = FALSE, found_run = FALSE;
    bool        found_end = FALSE;
    char        *op_line  = NULL, rawfile[1000], *line;

    for (c = deck; c; c = c->li_next) {

        if (*c->li_line == '*')
            continue;

        if (ciprefix(".op ", c->li_line)) {
            *c->li_line = '*';
            op_line = c->li_line + 1;
        }

        if (ciprefix(".end", c->li_line))
            found_end = TRUE;

        if (found_control && ciprefix("run", c->li_line))
            found_run = TRUE;

        if (ciprefix(".control", c->li_line))
            found_control = TRUE;

        if (ciprefix(".endc", c->li_line)) {
            found_control = FALSE;

            if (!found_run) {
                prev_card->li_next = xx_new_line(c, copy("run"), (*line_number)++, 0);
                prev_card = prev_card->li_next;
                found_run = TRUE;
            }

            if (cp_getvar("rawfile", CP_STRING, rawfile)) {
                line = TMALLOC(char, strlen("write") + strlen(rawfile) + 2);
                sprintf(line, "write %s", rawfile);
                prev_card->li_next = xx_new_line(c, line, (*line_number)++, 0);
                prev_card = prev_card->li_next;
            }
        }

        prev_card = c;
    }

    // check if need to add control section
    if (!found_run && found_end) {

        deck->li_next = xx_new_line(deck->li_next, copy(".endc"), (*line_number)++, 0);

        if (cp_getvar("rawfile", CP_STRING, rawfile)) {
            line = TMALLOC(char, strlen("write") + strlen(rawfile) + 2);
            sprintf(line, "write %s", rawfile);
            deck->li_next = xx_new_line(deck->li_next, line, (*line_number)++, 0);
        }

        if (op_line)
            deck->li_next = xx_new_line(deck->li_next, copy(op_line), (*line_number)++, 0);

        deck->li_next = xx_new_line(deck->li_next, copy("run"), (*line_number)++, 0);

        deck->li_next = xx_new_line(deck->li_next, copy(".control"), (*line_number)++, 0);
    }
}


// look for shell-style end-of-line continuation '\\'

static bool
chk_for_line_continuation(char *line)
{
    if (*line != '*' && *line != '$') {

        char *ptr = skip_back_ws_(strchr(line, '\0'), line);

        if ((ptr - 2 >= line) && (ptr[-1] == '\\') && (ptr[-2] == '\\')) {
            ptr[-1] = ' ';
            ptr[-2] = ' ';
            return TRUE;
        }
    }

    return FALSE;
}


//
// change .macro --> .subckt
//        .eom   --> .ends
//        .subckt name 1 2 3 params: w=9u l=180n --> .subckt name 1 2 3 w=9u l=180n
//        .subckt name (1 2 3) --> .subckt name 1 2 3
//        x1 (1 2 3)      --> x1 1 2 3
//        .param func1(x,y) = {x*y} --> .func func1(x,y) {x*y}

static void
inp_fix_macro_param_func_paren_io(struct line *card)
{
    char        *str_ptr, *new_str;

    for (; card; card = card->li_next) {

        if (*card->li_line == '*')
            continue;

        if (ciprefix(".macro", card->li_line) || ciprefix(".eom", card->li_line)) {
            str_ptr = skip_non_ws(card->li_line);

            if (ciprefix(".macro", card->li_line)) {
                new_str = TMALLOC(char, strlen(".subckt") + strlen(str_ptr) + 1);
                sprintf(new_str, ".subckt%s", str_ptr);
            } else {
                new_str = TMALLOC(char, strlen(".ends") + strlen(str_ptr) + 1);
                sprintf(new_str, ".ends%s", str_ptr);
            }

            tfree(card->li_line);
            card->li_line = new_str;
        }

        if (ciprefix(".subckt", card->li_line) || ciprefix("x", card->li_line)) {
            /* remove () */
            str_ptr = skip_non_ws(card->li_line);  // skip over .subckt, instance name
            str_ptr = skip_ws(str_ptr);
            if (ciprefix(".subckt", card->li_line)) {
                str_ptr = skip_non_ws(str_ptr);  // skip over subckt name
                str_ptr = skip_ws(str_ptr);
            }
            if (*str_ptr == '(') {
                *str_ptr = ' ';
                while (*str_ptr != '\0') {
                    if (*str_ptr == ')') {
                        *str_ptr = ' ';
                        break;
                    }
                    str_ptr++;
                }
                card->li_line = inp_remove_ws(card->li_line); /* remove the extra white spaces just introduced */
            }
        }

        if (ciprefix(".param", card->li_line)) {
            bool is_func = FALSE;
            str_ptr = skip_non_ws(card->li_line);  // skip over .param
            str_ptr = skip_ws(str_ptr);
            while (!isspace(*str_ptr) && *str_ptr != '=') {
                if (*str_ptr == '(')
                    is_func = TRUE;
                str_ptr++;
            }

            if (is_func) {
                str_ptr = strchr(card->li_line, '=');
                if (str_ptr)
                    *str_ptr = ' ';
                str_ptr = card->li_line + 1;
                str_ptr[0] = 'f';
                str_ptr[1] = 'u';
                str_ptr[2] = 'n';
                str_ptr[3] = 'c';
                str_ptr[4] = ' ';
            }
        }
    }
}


static char *
get_instance_subckt(char *line)
{
    char *end_ptr, *inst_name_ptr;
    char *equal_ptr = strchr(line, '=');

    // see if instance has parameters
    if (equal_ptr) {
        end_ptr = skip_back_ws_(equal_ptr, line);
        end_ptr = skip_back_non_ws_(end_ptr, line);
    } else {
        end_ptr = strchr(line, '\0');
    }

    end_ptr = skip_back_ws_(end_ptr, line);

    inst_name_ptr = skip_back_non_ws_(end_ptr, line);

    return copy_substring(inst_name_ptr, end_ptr);
}


static char*
get_subckt_model_name(char *line)
{
    char *name, *end_ptr;

    name = skip_non_ws(line);   // eat .subckt|.model
    name = skip_ws(name);

    end_ptr = skip_non_ws(name);

    return copy_substring(name, end_ptr);
}


static char*
get_model_name(char *line, int num_terminals)
{
    char *beg_ptr, *end_ptr;
    int  i = 0;

    beg_ptr = skip_non_ws(line); /* eat device name */
    beg_ptr = skip_ws(beg_ptr);

    for (i = 0; i < num_terminals; i++) { /* skip the terminals */
        beg_ptr = skip_non_ws(beg_ptr);
        beg_ptr = skip_ws(beg_ptr);
    }

    if (*line == 'r')           /* special dealing for r models */
        if ((*beg_ptr == '+') || (*beg_ptr == '-') || isdigit(*beg_ptr)) { /* looking for a value before model */
            beg_ptr = skip_non_ws(beg_ptr); /* skip the value */
            beg_ptr = skip_ws(beg_ptr);
        }

    end_ptr = skip_non_ws(beg_ptr);

    return copy_substring(beg_ptr, end_ptr);
}


static char*
get_model_type(char *line)
{
    char *beg_ptr;

    if (!ciprefix(".model", line))
        return NULL;

    beg_ptr = skip_non_ws(line); /* eat .model */
    beg_ptr = skip_ws(beg_ptr);

    beg_ptr = skip_non_ws(beg_ptr); /* eat model name */
    beg_ptr = skip_ws(beg_ptr);

    return gettok(&beg_ptr);
}


static char *
get_adevice_model_name(char *line)
{
    char *ptr_end, *ptr_beg;

    ptr_end = skip_back_ws_(strchr(line, '\0'), line);
    ptr_beg = skip_back_non_ws_(ptr_end, line);

    return copy_substring(ptr_beg, ptr_end);
}


static void
get_subckts_for_subckt(struct line *start_card, char *subckt_name,
                       char *used_subckt_names[], int *num_used_subckt_names,
                       char *used_model_names[], int *num_used_model_names,
                       bool has_models)
{
    struct line *card;
    char *curr_subckt_name, *inst_subckt_name, *model_name, *new_names[100];
    bool found_subckt = FALSE, have_subckt = FALSE, found_model = FALSE;
    int  i, num_terminals = 0, tmp_cnt = 0;

    for (card = start_card; card; card = card->li_next) {

        char *line = card->li_line;

        if (*line == '*')
            continue;

        if ((ciprefix(".ends", line) || ciprefix(".eom", line)) && found_subckt)
            break;

        if (ciprefix(".subckt", line) || ciprefix(".macro", line)) {
            curr_subckt_name = get_subckt_model_name(line);

            if (strcmp(curr_subckt_name, subckt_name) == 0)
                found_subckt = TRUE;

            tfree(curr_subckt_name);
        }

        if (found_subckt) {
            if (*line == 'x') {
                inst_subckt_name = get_instance_subckt(line);
                have_subckt = FALSE;
                for (i = 0; i < *num_used_subckt_names; i++)
                    if (strcmp(used_subckt_names[i], inst_subckt_name) == 0)
                        have_subckt = TRUE;
                if (!have_subckt) {
                    new_names[tmp_cnt++] = used_subckt_names[*num_used_subckt_names] = inst_subckt_name;
                    *num_used_subckt_names += 1;
                } else {
                    tfree(inst_subckt_name);
                }
            } else if (*line == 'a') {
                model_name = get_adevice_model_name(line);
                found_model = FALSE;
                for (i = 0; i < *num_used_model_names; i++)
                    if (strcmp(used_model_names[i], model_name) == 0)
                        found_model = TRUE;
                if (!found_model) {
                    used_model_names[*num_used_model_names] = model_name;
                    *num_used_model_names += 1;
                } else {
                    tfree(model_name);
                }
            } else if (has_models) {
                num_terminals = get_number_terminals(line);

                if (num_terminals != 0) {
                    char *tmp_name, *tmp_name1;
                    tmp_name1 = tmp_name = model_name = get_model_name(line, num_terminals);

                    if (isalpha(*model_name) ||
                        /* first character is digit, second is alpha, third is digit,
                           e.g. 1N4002 */
                        ((strlen(model_name) > 2) && isdigit(*tmp_name) &&
                         isalpha(*(++tmp_name)) && isdigit(*(++tmp_name))) ||
                        /* first character is is digit, second is alpha, third is alpha, fourth is digit
                           e.g. 2SK456 */
                        ((strlen(model_name) > 3) && isdigit(*tmp_name1) && isalpha(*(++tmp_name1)) &&
                         isalpha(*(++tmp_name1)) && isdigit(*(++tmp_name1)))) {
                        found_model = FALSE;
                        for (i = 0; i < *num_used_model_names; i++)
                            if (strcmp(used_model_names[i], model_name) == 0) found_model = TRUE;
                        if (!found_model) {
                            used_model_names[*num_used_model_names] = model_name;
                            *num_used_model_names += 1;
                        } else {
                            tfree(model_name);
                        }
                    } else {
                        tfree(model_name);
                    }
                }
            }
        }
    }

    // now make recursive call on instances just found above
    for (i = 0; i < tmp_cnt; i++)
        get_subckts_for_subckt(start_card, new_names[i], used_subckt_names, num_used_subckt_names,
                               used_model_names, num_used_model_names, has_models);
}


/*
  check if current token matches model bin name -- <token>.[0-9]+
*/

static bool
model_bin_match(char *token, char *model_name)
{
    char *dot_char;
    bool  flag = FALSE;

    if (strncmp(model_name, token, strlen(token)) == 0)
        if ((dot_char = strchr(model_name, '.')) != NULL) {
            flag = TRUE;
            dot_char++;
            while (*dot_char != '\0') {
                if (!isdigit(*dot_char)) {
                    flag = FALSE;
                    break;
                }
                dot_char++;
            }
        }

    return flag;
}


/*
  iterate through the deck and comment out unused subckts, models
  (don't want to waste time processing everything)
  also comment out .param lines with no parameters defined
*/

static void
comment_out_unused_subckt_models(struct line *start_card, int no_of_lines)
{
    struct line *card;
    char **used_subckt_names, **used_model_names, *subckt_name, *model_name;
    int  num_used_subckt_names = 0, num_used_model_names = 0, i = 0, num_terminals = 0, tmp_cnt = 0;
    bool processing_subckt = FALSE, found_subckt = FALSE, remove_subckt = FALSE, found_model = FALSE, has_models = FALSE;
    int skip_control = 0, nested_subckt = 0;

    /* generate arrays of *char for subckt or model names. Start
       with 1000, but increase, if number of lines in deck is larger */
    if (no_of_lines < 1000)
        no_of_lines = 1000;

    used_subckt_names = TMALLOC(char*, no_of_lines);
    used_model_names = TMALLOC(char*, no_of_lines);

    for (card = start_card; card; card = card->li_next) {
        if (ciprefix(".model", card->li_line))
            has_models = TRUE;
        if (ciprefix(".cmodel", card->li_line))
            has_models = TRUE;
        if (ciprefix(".param", card->li_line) && !strchr(card->li_line, '='))
            *card->li_line = '*';
    }

    for (card = start_card; card; card = card->li_next) {

        char *line = card->li_line;

        if (*line == '*')
            continue;

        /* there is no .subckt, .model or .param inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (ciprefix(".subckt", line) || ciprefix(".macro", line))
            processing_subckt = TRUE;
        if (ciprefix(".ends", line) || ciprefix(".eom", line))
            processing_subckt = FALSE;
        if (!processing_subckt) {
            if (*line == 'x') {
                subckt_name = get_instance_subckt(line);
                found_subckt = FALSE;
                for (i = 0; i < num_used_subckt_names; i++)
                    if (strcmp(used_subckt_names[i], subckt_name) == 0) found_subckt = TRUE;
                if (!found_subckt) {
                    used_subckt_names[num_used_subckt_names++] = subckt_name;
                    tmp_cnt++;
                } else {
                    tfree(subckt_name);
                }
            } else if (*line == 'a') {
                model_name = get_adevice_model_name(line);
                found_model = FALSE;
                for (i = 0; i < num_used_model_names; i++)
                    if (strcmp(used_model_names[i], model_name) == 0)
                        found_model = TRUE;
                if (!found_model)
                    used_model_names[num_used_model_names++] = model_name;
                else
                    tfree(model_name);
            } else if (has_models) {
                /* This is a preliminary version, until we have found a reliable
                   method to detect the model name out of the input line (Many
                   options have to be taken into account.). */
                num_terminals = get_number_terminals(line);
                if (num_terminals != 0) {
                    bool model_ok = FALSE;
                    char *tmp_name, *tmp_name1;
                    tmp_name = tmp_name1 = model_name = get_model_name(line, num_terminals);
                    /* first character of model name is character from alphabet */
                    if (isalpha(*model_name))
                        model_ok = TRUE;
                    /* first character is digit, second is alpha, third is digit,
                       e.g. 1N4002 */
                    else if ((strlen(model_name) > 2) && isdigit(*tmp_name) &&
                             isalpha(*(++tmp_name)) && isdigit(*(++tmp_name)))
                        model_ok = TRUE;
                    /* first character is is digit, second is alpha, third is alpha, fourth is digit
                       e.g. 2SK456 */
                    else if ((strlen(model_name) > 3) && isdigit(*tmp_name1) &&
                             isalpha(*(++tmp_name1)) && isalpha(*(++tmp_name1)) &&
                             isdigit(*(++tmp_name1)))
                        model_ok = TRUE;
                    /* Check if model has already been recognized, if not, add its name to
                       list used_model_names[i] */
                    if (model_ok) {
                        found_model = FALSE;
                        for (i = 0; i < num_used_model_names; i++)
                            if (strcmp(used_model_names[i], model_name) == 0)
                                found_model = TRUE;
                        if (!found_model)
                            used_model_names[num_used_model_names++] = model_name;
                        else
                            tfree(model_name);
                    } else {
                        tfree(model_name);
                    }
                }
            } /* if (has_models)  */
        } /* if (!processing_subckt) */
    } /* for loop through all cards */

    for (i = 0; i < tmp_cnt; i++)
        get_subckts_for_subckt
            (start_card, used_subckt_names[i],
             used_subckt_names, &num_used_subckt_names,
             used_model_names, &num_used_model_names, has_models);

    /* comment out any unused subckts, currently only at top level */
    for (card = start_card; card; card = card->li_next) {

        char *line = card->li_line;

        if (*line == '*')
            continue;

        if (ciprefix(".subckt", line) || ciprefix(".macro", line)) {
            nested_subckt++;
            subckt_name = get_subckt_model_name(line);
            if (nested_subckt == 1) {
                /* check if unused, only at top level */
                remove_subckt = TRUE;
                for (i = 0; i < num_used_subckt_names; i++)
                    if (strcmp(used_subckt_names[i], subckt_name) == 0)
                        remove_subckt = FALSE;
            }
            tfree(subckt_name);
        }

        if (ciprefix(".ends", line) || ciprefix(".eom", line)) {
            nested_subckt--;
            if (remove_subckt)
                *line = '*';
            if (nested_subckt == 0)
                remove_subckt = FALSE;
        }

        if (remove_subckt)
            *line = '*';
        else if (has_models &&
                 (ciprefix(".model", line) || ciprefix(".cmodel", line)))
        {
            char *model_type = get_model_type(line);
            model_name = get_subckt_model_name(line);
            /* keep R, L, C models because in addition to no. of terminals the value may be given,
               as in RE1 1 2 800 newres dtemp=5, so model name may be token no. 4 or 5,
               and, if 5, will not be detected by get_subckt_model_name()*/
            if (cieq(model_type, "c") ||
                cieq(model_type, "l") ||
                cieq(model_type, "r"))
            {
                found_model = TRUE;
            } else {
                found_model = FALSE;
                for (i = 0; i < num_used_model_names; i++)
                    if (strcmp(used_model_names[i], model_name) == 0 ||
                        model_bin_match(used_model_names[i], model_name))
                        found_model = TRUE;
            }
            tfree(model_type);
            if (!found_model)
                *line = '*';
            tfree(model_name);
        }
    }

    for (i = 0; i < num_used_subckt_names; i++)
        tfree(used_subckt_names[i]);
    for (i = 0; i < num_used_model_names;  i++)
        tfree(used_model_names[i]);
    tfree(used_subckt_names);
    tfree(used_model_names);
}


#if 0
// find closing paren
static char *
inp_search_closing_paren(char *s)
{
    int count = 0;
    // assert(*s == '(')
    while (*s) {
        if (*s == '(')
            count++;
        if (*s == ')')
            count--;
        if (count == 0)
            return s + 1;
        s++;
    }

    return NULL;
}
#endif


#if 0
/* search backwards for opening paren */
static char *
inp_search_opening_paren(char *s, char *start)
{
    int count = 0;
    // assert(*s == ')')
    while (s >= start) {
        if (*s == '(')
            count--;
        if (*s == ')')
            count++;
        if (count == 0)
            return s;
        s--;
    }

    return NULL;
}
#endif


/*-------------------------------------------------------------------------*
  removes  " " quotes, returns lower case letters,
  replaces non-printable characterss with '_'                                                                       *
  *-------------------------------------------------------------------------*/

void
inp_casefix(char *string)
{
#ifdef HAVE_CTYPE_H
    if (string)
        while (*string) {
#ifdef HAS_ASCII
            /* ((*string) & 0177): mask off all but the first seven bits, 0177: octal */
            *string = (char) strip(*string);
#endif
            if (*string == '"') {
                *string++ = ' ';
                while (*string && *string != '"')
                    string++;
                if (*string == '\0')
                    continue; /* needed if string is "something ! */
                if (*string == '"')
                    *string = ' ';
            }
            if (!isspace(*string) && !isprint(*string))
                *string = '_';
            if (isupper(*string))
                *string = (char) tolower(*string);
            string++;
        }
#endif
}


/* Strip all end-of-line comments from a deck
   For cf == TRUE (script files, command files like spinit, .spiceinit)
   and for .control sections only '$ ' is accepted as end-of-line comment,
   to avoid conflict with $variable definition, otherwise we accept '$'. */
static void
inp_stripcomments_deck(struct line *c, bool cf)
{
    bool found_control = FALSE;
    for (; c; c = c->li_next) {
        /* exclude lines between .control and .endc from removing white spaces */
        if (ciprefix(".control", c->li_line))
            found_control = TRUE;
        if (ciprefix(".endc", c->li_line))
            found_control = FALSE;
        inp_stripcomments_line(c->li_line, found_control|cf);
    }
}


/*
 * Support for end-of-line comments that begin with any of the following:
 *   ';'
 *   '$' (only outside of a .control section)
 *   '$ '
 *   '//' (like in c++ and as per the numparam code)
 * Any following text to the end of the line is ignored.
 * Note requirement for $ to be followed by a space, if we are inside of a
 * .control section or in a command file. This is to avoid conflict
 * with use of $ in front of a variable.
 * Comments on a continuation line (i.e. line begining with '+') are allowed
 * and are removed before lines are stitched.
 * Lines that contain only an end-of-line comment with or without leading white
 * space are also allowed.

 If there is only white space before the end-of-line comment the
 the whole line is converted to a normal comment line (i.e. one that
 begins with a '*').
 BUG: comment characters in side of string literals are not ignored
 ('$' outside of .control section is o.k. however). */

static void
inp_stripcomments_line(char *s, bool cs)
{
    char c = ' ';               /* anything other than a comment character */
    char *d = s;
    if (*s == '\0')
        return;                 /* empty line */
    if (*s == '*')
        return;                 /* line is already a comment */
    /* look for comments */
    while ((c = *d) != '\0') {
        d++;
        if (*d == ';') {
            break;
        } else if (!cs && (c == '$')) { /* outside of .control section */
            /* The character before '&' has to be ',' or ' ' or tab.
               A valid numerical expression directly before '$' is not yet supported. */
            if ((d - 2 >= s) && ((d[-2] == ' ') || (d[-2] == ',') || (d[-2] == '\t'))) {
                d--;
                break;
            }
        } else if (cs && (c == '$') && (*d == ' ')) {/* inside of .control section or command file */
            d--;                /* move d back to first comment character */
            break;
        } else if ((c == '/') && (*d == '/')) {
            d--;                /* move d back to first comment character */
            break;
        }
    }
    /* d now points to the first comment character or the null at the string end */

    /* check for special case of comment at start of line */
    if (d == s) {
        *s = '*';               /* turn into normal comment */
        return;
    }

    if (d > s) {
        d--;
        /* d now points to character just before comment */

        /* eat white space at new end of line */
        while (d >= s) {
            if ((*d != ' ') && (*d != '\t'))
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
    if (d <= s) {
        *s = '*'; /* turn the whole line into normal comment */
        return;
    }

    *d = '\0'; /* terminate line in new location */
}


static void
inp_change_quotes(char *s)
{
    bool first_quote = FALSE;

    for (; *s; s++)
        if (*s == '\'') {
            if (first_quote == FALSE) {
                *s = '{';
                first_quote = TRUE;
            } else {
                *s = '}';
                first_quote = FALSE;
            }
        }
}


static void
add_name(struct names *p, char *name)
{
    if (p->num_names >= N_SUBCKT_W_PARAMS) {
        fprintf(stderr, "ERROR, N_SUBCKT_W_PARMS overflow\n");
        controlled_exit(EXIT_FAILURE);
    }

    p->names[p->num_names++] = name;
}


static char **
find_name(struct names *p, char *name)
{
    int i;

    for (i = 0; i < p->num_names; i++)
        if (strcmp(p->names[i], name) == 0)
            return & p->names[i];

    return NULL;
}


static char*
inp_fix_subckt(struct names *subckt_w_params, char *s)
{
    struct line *head = NULL, *start_card = NULL, *end_card = NULL, *prev_card = NULL, *c = NULL;
    char *equal, *beg, *buffer, *ptr1, *ptr2, *new_str = NULL;
    char keep;
    int  num_params = 0, i = 0, bracedepth = 0;
    /* find first '=' */
    equal = strchr(s, '=');
    if (!strstr(s, "params:") && equal != NULL) {
        /* get subckt name (ptr1 will point to name) */
        ptr1 = skip_non_ws(s);
        ptr1 = skip_ws(ptr1);
        for (ptr2 = ptr1; *ptr2 && !isspace(*ptr2) && !isquote(*ptr2); ptr2++)
            ;

        add_name(subckt_w_params, copy_substring(ptr1, ptr2));

        /* go to beginning of first parameter word  */
        /* s    will contain only subckt definition */
        /* beg  will point to start of param list   */
        beg = skip_back_ws_(equal, s);
        beg = skip_back_non_ws_(beg, s);
        beg[-1] = '\0';         /* fixme can be < s */

        head = xx_new_line(NULL, NULL, 0, 0);
        /* create list of parameters that need to get sorted */
        while (*beg && (ptr1 = strchr(beg, '=')) != NULL) {
#ifndef NOBRACE
            /* alternative patch to cope with spaces:
               get expression between braces {...} */
            ptr2 = skip_ws(ptr1 + 1);
            ptr1 = skip_back_ws_(ptr1, beg);
            ptr1 = skip_back_non_ws_(ptr1, beg);
            /* ptr1 points to beginning of parameter */

            /* if parameter is an expression and starts with '{', find closing '}'
               Braces maybe nested (will they ever be ?). */
            if (*ptr2 == '{') {
                bracedepth = 1;
                while (bracedepth > 0) {
                    ptr2++;
                    if (*ptr2 == '{')
                        bracedepth++;
                    else if (*ptr2 == '}')
                        bracedepth--;
                    else if (*ptr2 == '\0') {
                        fprintf(stderr, "Error: Missing } in line %s\n", s);
                        controlled_exit(EXIT_FAILURE);
                    }
                }
                ptr2++;/* ptr2 points past end of parameter {...} */
            }
            else
                /* take only the next token (separated by space) as the parameter */
                ptr2 = skip_non_ws(ptr2); /* ptr2 points past end of parameter       */

            keep  = *ptr2;
            if (keep == '\0') {
                /* End of string - don't go over end. */
                beg = ptr2;
            } else {
                *ptr2 = '\0';
                beg = ptr2 + 1;
            }

            end_card = xx_new_line(NULL, strdup(ptr1), 0, 0);

            if (start_card == NULL)
                head->li_next = start_card = end_card;
            else
                prev_card->li_next = end_card;

            prev_card = end_card;
            num_params++;
#else
            /* patch provided by Ivan Riis Nielsen */
            bool done = FALSE;
            int buf_len = 32, buf_idx = 0;
            char *buf = TMALLOC(char, buf_len), *p1 = beg, *p2 = beg;

            while (!done) {
                while (*p2 && !isspace(*p2)) {

                    if (buf_idx >= buf_len) {
                        buf_len *= 2;
                        buf = TREALLOC(char, buf, buf_len);
                    }
                    buf[buf_idx++] = *(p2++);
                }
                p1 = skip_ws(p2);
                if (*p1 == '\0' || !(strchr("+-*/<>=(!,{", p2[-1]) || strchr("+-*/<>=()!,}", *p1))) {
                    if (buf_idx >= buf_len) {
                        buf_len *= 2;
                        buf = TREALLOC(char, buf, buf_len);
                    }
                    buf[buf_idx++] = '\0';
                    beg = p1;

                    end_card = xx_new_line(NULL, buf, 0, 0);

                    if (start_card == NULL)
                        head->li_next = start_card = end_card;
                    else
                        prev_card->li_next = end_card;

                    prev_card = end_card;
                    num_params++;

                    done = TRUE;
                }
                else
                    p2 = p1;
            }
#endif
        }
        /* now sort parameters in order of dependencies */
        inp_sort_params(start_card, end_card, head, start_card, end_card);

        /* create new ordered parameter string for subckt call */
        c = head->li_next;
        tfree(head);
        for (i = 0; i < num_params && c != NULL; i++) {
            if (new_str == NULL) {
                new_str = strdup(c->li_line);
            } else {
                char *x = TMALLOC(char, strlen(new_str) + strlen(c->li_line) + 2);
                sprintf(x, "%s %s", new_str, c->li_line);
                tfree(new_str);
                new_str = x;
            }
            tfree(c->li_line);
            head = c;
            c = c->li_next;
            tfree(head);
        }

        /* create buffer and insert params: */
        buffer = TMALLOC(char, strlen(s) + 9 + strlen(new_str) + 1);
        sprintf(buffer, "%s params: %s", s, new_str);

        tfree(s);
        tfree(new_str);

        s = buffer;
    }

    return s;
}


static char*
inp_remove_ws(char *s)
{
    char *big_buff;
    int  big_buff_index = 0;
    char *buffer, *curr;
    bool is_expression = FALSE;

    big_buff = TMALLOC(char, strlen(s) + 2);
    curr = s;

    while (*curr != '\0') {
        if (*curr == '{')
            is_expression = TRUE;
        if (*curr == '}')
            is_expression = FALSE;

        big_buff[big_buff_index++] = *curr;
        if (*curr == '=' || (is_expression && (is_arith_char(*curr) || *curr == ','))) {
            curr = skip_ws(curr + 1);

            if (*curr == '{')
                is_expression = TRUE;
            if (*curr == '}')
                is_expression = FALSE;

            big_buff[big_buff_index++] = *curr;
        }
        if (*curr != '\0')
            curr++;
        if (isspace(*curr)) {
            curr = skip_ws(curr);
            if (is_expression) {
                if (*curr != '=' && !is_arith_char(*curr) && *curr != ',')
                    big_buff[big_buff_index++] = ' ';
            } else {
                if (*curr != '=')
                    big_buff[big_buff_index++] = ' ';
            }
        }
    }

    big_buff[big_buff_index] = '\0';

    buffer = copy(big_buff);

    tfree(s);
    tfree(big_buff);

    return buffer;
}


/*
  change quotes from '' to {}
  .subckt name 1 2 3 params: l=1 w=2 --> .subckt name 1 2 3 l=1 w=2
  x1 1 2 3 params: l=1 w=2 --> x1 1 2 3 l=1 w=2
  modify .subckt lines by calling inp_fix_subckt()
  No changes to lines in .control section !
*/

static void
inp_fix_for_numparam(struct names *subckt_w_params, struct line *c)
{
    bool found_control = FALSE;

    for (; c; c = c->li_next) {

        if (*(c->li_line) == '*' || ciprefix(".lib", c->li_line))
            continue;

        /* exclude lines between .control and .endc from getting quotes changed */
        if (ciprefix(".control", c->li_line))
            found_control = TRUE;
        if (ciprefix(".endc", c->li_line))
            found_control = FALSE;

        if (found_control)
            continue;

        inp_change_quotes(c->li_line);

        if ((inp_compat_mode == COMPATMODE_ALL) || (inp_compat_mode == COMPATMODE_PS))
            if (ciprefix(".subckt", c->li_line) || ciprefix("x", c->li_line)) {
                /* remove params: */
                char *str_ptr = strstr(c->li_line, "params:");
                if (str_ptr)
                    memcpy(str_ptr, "       ", 7);
            }

        if (ciprefix(".subckt", c->li_line))
            c->li_line = inp_fix_subckt(subckt_w_params, c->li_line);
    }
}


static void
inp_remove_excess_ws(struct line *c)
{
    bool found_control = FALSE;

    for (; c; c = c->li_next) {

        if (*c->li_line == '*')
            continue;

        /* exclude echo lines between .control and .endc from removing white spaces */
        if (ciprefix(".control", c->li_line))
            found_control = TRUE;
        if (ciprefix(".endc", c->li_line))
            found_control = FALSE;

        if (found_control && ciprefix("echo", c->li_line))
            continue;

        c->li_line = inp_remove_ws(c->li_line); /* freed in fcn */
    }
}


static struct line *
expand_section_ref(struct line *c, char *dir_name)
{
    char *line = c->li_line;

    char *s, *t, *y;

    s = skip_non_ws(line);
    while (isspace(*s) || isquote(*s))
        s++;
    for (t = s; *t && !isspace(*t) && !isquote(*t); t++)
        ;
    y = t;
    while (isspace(*y) || isquote(*y))
        y++;

    if (*y) {
        /* library section reference: `.lib <library-file> <section-name>' */

        struct line *section_def;
        char keep_char1, keep_char2;
        char *z;
        struct library *lib;

        for (z = y; *z && !isspace(*z) && !isquote(*z); z++)
            ;
        keep_char1 = *t;
        keep_char2 = *z;
        *t = '\0';
        *z = '\0';

        lib = read_a_lib(s, dir_name);

        if (!lib) {
            fprintf(stderr, "ERROR, library file %s not found\n", s);
            controlled_exit(EXIT_FAILURE);
        }

        section_def = find_section_definition(lib->deck, y);

        if (!section_def) {
            fprintf(stderr, "ERROR, library file %s, section definition %s not found\n", s, y);
            controlled_exit(EXIT_FAILURE);
        }

        /* recursively expand the refered section itself */
        {
            struct line *t = section_def;
            for (; t; t = t->li_next) {
                if (ciprefix(".endl", t->li_line))
                    break;
                if (ciprefix(".lib", t->li_line))
                    t = expand_section_ref(t, lib->habitat);
            }
            if (!t) {
                fprintf(stderr, "ERROR, .endl not found\n");
                controlled_exit(EXIT_FAILURE);
            }
        }

        /* insert the library section definition into `c' */
        {
            struct line *cend = NULL, *newl;
            struct line *rest = c->li_next;
            struct line *t = section_def;
            for (; t; t=t->li_next) {
                newl = xx_new_line(NULL, copy(t->li_line), t->li_linenum, t->li_linenum_orig);
                if (cend)
                    cend->li_next = newl;
                else {
                    c->li_next = newl;
                    newl->li_line[0] = '*';
                    newl->li_line[1] = '<';
                }
                cend = newl;
                if(ciprefix(".endl", t->li_line))
                    break;
            }
            if (!t) {
                fprintf(stderr, "ERROR, .endl not found\n");
                controlled_exit(EXIT_FAILURE);
            }
            cend->li_line[0] = '*';
            cend->li_line[1] = '>';
            cend->li_next = rest;

            c = cend;
        }

        *line = '*';  /* comment out .lib line */
        *t = keep_char1;
        *z = keep_char2;
    }

    return c;
}


/*
 * recursively expand library section references,
 * either
 *    every library section reference (when the given section_name_ === NULL)
 * or
 *    just those references occuring in the given library section definition
 */

static void
expand_section_references(struct line *c, char *dir_name)
{
    for (; c; c = c->li_next)
        if (ciprefix(".lib", c->li_line))
            c = expand_section_ref(c, dir_name);
}


static char*
inp_get_subckt_name(char *s)
{
    char *subckt_name, *end_ptr = strchr(s, '=');

    if (end_ptr) {
        end_ptr = skip_back_ws_(end_ptr, s);
        end_ptr = skip_back_non_ws_(end_ptr, s);
    } else {
        end_ptr = strchr(s, '\0');
    }

    end_ptr = skip_back_ws_(end_ptr, s);
    subckt_name = skip_back_non_ws_(end_ptr, s);

    return copy_substring(subckt_name, end_ptr);
}


static int
inp_get_params(char *line, char *param_names[], char *param_values[])
{
    char *equal_ptr = strchr(line, '=');
    char *end, *name, *value;
    int  num_params = 0;
    char tmp_str[1000];
    char keep;
    bool is_expression = FALSE;

    while ((equal_ptr = find_assignment(line)) != NULL) {

        is_expression = FALSE;

        /* get parameter name */
        end = skip_back_ws_(equal_ptr, line);
        name = skip_back_non_ws_(end, line);

        param_names[num_params++] = copy_substring(name, end);

        /* get parameter value */
        value = skip_ws(equal_ptr + 1);

        if (*value == '{')
            is_expression = TRUE;
        end = value;
        if (is_expression)
            while (*end && *end != '}')
                end++;
        else
            end = skip_non_ws(end);

        if (is_expression)
            end++;
        keep = *end;
        *end = '\0';

        if (*value != '{' &&
            !(isdigit(*value) || (*value == '.' && isdigit(value[1])))) {
            sprintf(tmp_str, "{%s}", value);
            value = tmp_str;
        }
        param_values[num_params-1] = strdup(value);
        *end = keep;

        line = end;
    }

    return num_params;
}


static char*
inp_fix_inst_line(char *inst_line,
                  int num_subckt_params, char *subckt_param_names[], char *subckt_param_values[],
                  int num_inst_params, char *inst_param_names[], char *inst_param_values[])
{
    char *end, *inst_name, *inst_name_end;
    char *curr_line = inst_line, *new_line = NULL;
    int i, j;

    inst_name_end = skip_non_ws(inst_line);
    inst_name = copy_substring(inst_line, inst_name_end);

    end = strchr(inst_line, '=');
    if (end) {
        end = skip_back_ws_(end, inst_line);
        end = skip_back_non_ws_(end, inst_line);
        end[-1] = '\0';         /* fixme can be < inst_line */
    }

    for (i = 0; i < num_subckt_params; i++)
        for (j = 0; j < num_inst_params; j++)
            if (strcmp(subckt_param_names[i], inst_param_names[j]) == 0) {
                tfree(subckt_param_values[i]);
                subckt_param_values[i] = strdup(inst_param_values[j]);
            }

    for (i = 0; i < num_subckt_params; i++) {
        new_line = TMALLOC(char, strlen(curr_line) + strlen(subckt_param_values[i]) + 2);
        sprintf(new_line, "%s %s", curr_line, subckt_param_values[i]);

        tfree(curr_line);
        tfree(subckt_param_names[i]);
        tfree(subckt_param_values[i]);

        curr_line = new_line;
    }

    for (i = 0; i < num_inst_params; i++) {
        tfree(inst_param_names[i]);
        tfree(inst_param_values[i]);
    }

    tfree(inst_name);

    return curr_line;
}


/* If multiplier parameter 'm' is found on a X line, flag is set
   to TRUE.
   Function is called from inp_fix_inst_calls_for_numparam() */

static bool
found_mult_param(int num_params, char *param_names[])
{
    int i;

    for (i = 0; i < num_params; i++)
        if (strcmp(param_names[i], "m") == 0)
            return TRUE;

    return FALSE;
}


/* If a subcircuit invocation (X-line) is found, which contains the
   multiplier parameter 'm', m is added to all lines inside
   the corresponding subcircuit except of some excluded in the code below
   (FIXME: It may be necessary to exclude more of them, at least
   for all devices that are not supporting the 'm' parameter).

   Function is called from inp_fix_inst_calls_for_numparam() */

static int
inp_fix_subckt_multiplier(struct names *subckt_w_params, struct line *subckt_card,
                          int num_subckt_params, char *subckt_param_names[], char *subckt_param_values[])
{
    struct line *card;
    char *new_str;

    subckt_param_names[num_subckt_params]  = strdup("m");
    subckt_param_values[num_subckt_params] = strdup("1");
    num_subckt_params ++;

    if (!strstr(subckt_card->li_line, "params:")) {
        new_str = TMALLOC(char, strlen(subckt_card->li_line) + 13);
        sprintf(new_str, "%s params: m=1", subckt_card->li_line);
        add_name(subckt_w_params, get_subckt_model_name(subckt_card->li_line));
    } else {
        new_str = TMALLOC(char, strlen(subckt_card->li_line) + 5);
        sprintf(new_str, "%s m=1", subckt_card->li_line);
    }

    tfree(subckt_card->li_line);
    subckt_card->li_line = new_str;

    for (card = subckt_card->li_next;
         card && !ciprefix(".ends", card->li_line);
         card = card->li_next) {
        /* no 'm' for B, V, E, H or comment line */
        if ((*(card->li_line) == '*') || (*(card->li_line) == 'b') || (*(card->li_line) == 'v') ||
            (*(card->li_line) == 'e') || (*(card->li_line) == 'h'))
            continue;
        /* no 'm' for model cards */
        if (ciprefix(".model", card->li_line))
            continue;
        new_str = TMALLOC(char, strlen(card->li_line) + 7);
        sprintf(new_str, "%s m={m}", card->li_line);

        tfree(card->li_line);
        card->li_line = new_str;
    }

    return num_subckt_params;
}


static void
inp_fix_inst_calls_for_numparam(struct names *subckt_w_params, struct line *deck)
{
    struct line *c;
    struct line *d, *p = NULL;
    char *subckt_name;
    char *subckt_param_names[1000];
    char *subckt_param_values[1000];
    char *inst_param_names[1000];
    char *inst_param_values[1000];
    char name_w_space[1000];
    int  num_subckt_params = 0;
    int  num_inst_params   = 0;
    int  i, j, k;
    bool flag = FALSE;
    bool found_subckt = FALSE;
    bool found_param_match = FALSE;

    // first iterate through instances and find occurences where 'm' multiplier needs to be
    // added to the subcircuit -- subsequent instances will then need this parameter as well
    for (c = deck; c; c = c->li_next) {
        char *inst_line = c->li_line;

        if (*inst_line == '*')
            continue;

        if (ciprefix("x", inst_line)) {
            num_inst_params = inp_get_params(inst_line, inst_param_names, inst_param_values);
            subckt_name     = inp_get_subckt_name(inst_line);

            if (found_mult_param(num_inst_params, inst_param_names)) {
                flag = FALSE;
                // iterate through the deck to find the subckt (last one defined wins)

                for (d = deck; d; d = d->li_next) {
                    char *subckt_line = d->li_line;
                    if (ciprefix(".subckt", subckt_line)) {
                        subckt_line = skip_non_ws(subckt_line);
                        subckt_line = skip_ws(subckt_line);

                        sprintf(name_w_space, "%s ", subckt_name);
                        if (strncmp(subckt_line, name_w_space, strlen(name_w_space)) == 0) {
                            p = d;
                            flag = TRUE;
                        }
                    }
                }
                if (flag) {
                    num_subckt_params = inp_get_params(p->li_line, subckt_param_names, subckt_param_values);

                    if (num_subckt_params == 0 || !found_mult_param(num_subckt_params, subckt_param_names))
                        inp_fix_subckt_multiplier(subckt_w_params, p, num_subckt_params, subckt_param_names, subckt_param_values);
                }
            }
            tfree(subckt_name);
            if (flag)
                for (i = 0; i < num_subckt_params; i++) {
                    tfree(subckt_param_names[i]);
                    tfree(subckt_param_values[i]);
                }
            for (i = 0; i < num_inst_params; i++) {
                tfree(inst_param_names[i]);
                tfree(inst_param_values[i]);
            }
        }
    }

    for (c = deck; c; c = c->li_next) {
        char *inst_line = c->li_line;

        if (*inst_line == '*')
            continue;

        if (ciprefix("x", inst_line)) {
            subckt_name = inp_get_subckt_name(inst_line);

            if (find_name(subckt_w_params, subckt_name)) {
                sprintf(name_w_space, "%s ", subckt_name);

                /* find .subckt line */
                found_subckt = FALSE;

                d = deck;
                while (d != NULL) {
                    char *subckt_line = d->li_line;
                    if (ciprefix(".subckt", subckt_line)) {
                        subckt_line = skip_non_ws(subckt_line);
                        subckt_line = skip_ws(subckt_line);

                        if (strncmp(subckt_line, name_w_space, strlen(name_w_space)) == 0) {
                            num_subckt_params = inp_get_params(subckt_line, subckt_param_names, subckt_param_values);
                            num_inst_params   = inp_get_params(inst_line, inst_param_names, inst_param_values);

                            // make sure that if have inst params that one matches subckt
                            found_param_match = FALSE;
                            if (num_inst_params == 0) {
                                found_param_match = TRUE;
                            } else {
                                for (j = 0; j < num_inst_params; j++) {
                                    for (k = 0; k < num_subckt_params; k++)
                                        if (strcmp(subckt_param_names[k], inst_param_names[j]) == 0) {
                                            found_param_match = TRUE;
                                            break;
                                        }
                                    if (found_param_match)
                                        break;
                                }
                            }

                            if (!found_param_match) {
                                // comment out .subckt and continue
                                while (d != NULL && !ciprefix(".ends", d->li_line)) {
                                    *(d->li_line) = '*';
                                    d = d->li_next;
                                }
                                *(d->li_line) = '*';
                                d = d->li_next;
                                continue;
                            }

                            c->li_line = inp_fix_inst_line(inst_line, num_subckt_params, subckt_param_names, subckt_param_values, num_inst_params, inst_param_names, inst_param_values);
                            found_subckt = TRUE;
                            for (i = 0; i < num_subckt_params; i++) {
                                tfree(subckt_param_names[i]);
                                tfree(subckt_param_values[i]);
                            }
                            for (i = 0; i < num_inst_params; i++) {
                                tfree(inst_param_names[i]);
                                tfree(inst_param_values[i]);
                            }
                        }
                    }
                    if (found_subckt)
                        break;
                    d = d->li_next;
                }
            }
            tfree(subckt_name);
        }
    }
}


static struct function *
new_function(struct function_env *env, char *name)
{
    struct function *f = TMALLOC(struct function, 1);

    f -> name = name;
    f -> num_parameters = 0;

    f -> next = env->functions;
    env -> functions  = f;

    return f;
}


static struct function *
find_function(struct function_env *env, char *name)
{
    struct function *f;

    for (; env; env = env->up)
        for (f = env->functions; f; f = f->next)
            if (strcmp(f->name, name) == 0)
                return f;

    return NULL;
}


static void
free_function(struct function *fcn)
{
    int i;

    tfree(fcn->name);
    tfree(fcn->macro);

    for (i = 0; i < fcn->num_parameters; i++)
        tfree(fcn->params[i]);
}


static void
new_function_parameter(struct function *fcn, char *parameter)
{
    if (fcn->num_parameters >= N_PARAMS) {
        fprintf(stderr, "ERROR, N_PARAMS overflow\n");
        controlled_exit(EXIT_FAILURE);
    }

    fcn->params[fcn->num_parameters++] = parameter;
}


static void
inp_get_func_from_line(struct function_env *env, char *line)
{
    char *end;
    char temp_buf[5000];
    int  str_len = 0;
    struct function *function;

    /* skip `.func' */
    line = skip_non_ws(line);
    line = skip_ws(line);

    /* get function name */
    end = line;
    while (*end && !isspace(*end) && *end != '(')
        end++;

    function = new_function(env, copy_substring(line, end));

    while (*end && *end != '(')
        end++;

    /* get function parameters */
    while (*end && *end != ')') {
        char *beg = skip_ws(end + 1);
        end = beg;
        while (*end && !isspace(*end) && *end != ',' && *end != ')')
            end++;
        if (end > beg)
            new_function_parameter(function, copy_substring(beg, end));
    }


    /* skip to the beinning of the function body */
    while (*end && *end++ != '{')
        ;

    /* get function body */
    str_len = 0;
    while (*end  && *end != '}') {
        if (!isspace(*end))
            temp_buf[str_len++] = *end;
        end++;
    }
    temp_buf[str_len++] = '\0';

    function->macro = strdup(temp_buf);
}


/*
 * grab functions at the current .subckt nesting level
 */

static void
inp_grab_func(struct function_env *env, struct line *c)
{
    int nesting = 0;

    for (; c; c = c->li_next) {

        if (*c->li_line == '*')
            continue;

        if (ciprefix(".subckt", c->li_line))
            nesting++;
        if (ciprefix(".ends", c->li_line))
            nesting--;

        if (nesting < 0)
            break;

        if (nesting > 0)
            continue;

        if (ciprefix(".func", c->li_line)) {
            inp_get_func_from_line(env, c->li_line);
            *c->li_line = '*';
        }
    }
}


static char*
inp_do_macro_param_replace(struct function *fcn, char *params[])
{
    char *param_ptr, *curr_ptr, *new_str, *curr_str = NULL, *search_ptr;
    char keep, before, after;
    int  i;

    if (fcn->num_parameters == 0)
        return strdup(fcn->macro);

    for (i = 0; i < fcn->num_parameters; i++) {

        if (curr_str == NULL) {
            search_ptr = curr_ptr = strdup(fcn->macro);
        } else {
            search_ptr = curr_ptr = curr_str;
            curr_str = NULL;
        }

        while ((param_ptr = strstr(search_ptr, fcn->params[i])) != NULL) {
            char *op_ptr = NULL, *cp_ptr = NULL;
            int is_vi = 0;

            /* make sure actually have the parameter name */
            if (param_ptr == search_ptr) /* no valid 'before' */
                before = '\0';
            else
                before = *(param_ptr-1);
            after  = param_ptr [ strlen(fcn->params[i]) ];
            if (!(is_arith_char(before) || isspace(before) ||
                  before == ',' || before == '=' || (param_ptr-1) < curr_ptr) ||
                !(is_arith_char(after) || isspace(after) ||
                  after == ',' || after  == '=' || after  == '\0'))
            {
                search_ptr = param_ptr + 1;
                continue;
            }

            /* exclude v(nn, parameter), v(parameter, nn), v(parameter),
            and i(parameter) if here 'parameter' is also a node name */
            if (before != '\0') {
                /* go backwards from 'parameter' and find '(' */
                for (op_ptr = param_ptr-1; op_ptr > curr_ptr; op_ptr--) {
                    if (*op_ptr == ')') {
                        is_vi = 0;
                        break;
                    }
                    if ((*op_ptr == '(') && (op_ptr - 2 > curr_ptr) &&
                       ((*(op_ptr - 1) == 'v') || (*(op_ptr - 1) == 'i')) &&
                       (is_arith_char(*(op_ptr - 2)) || isspace(*(op_ptr - 2)) ||
                       *(op_ptr - 2) == ',' || *(op_ptr - 2) == '=' )) {
                        is_vi = 1;
                        break;
                    }
                }
                /* We have a true v( or i( */
                if (is_vi) {
                    cp_ptr = param_ptr;
                    /* go forward and find closing ')' */
                    while (*cp_ptr) {
                       cp_ptr++;
                       if (*cp_ptr == '(') {
                           is_vi = 0;
                           break;
                       }
                       if (*cp_ptr == ')')
                           break;
                    }
                    if (*cp_ptr == '\0')
                        is_vi = 0;
                }
                /* We have a true v(...) or i(...),
                   so skip it, and continue searching for new 'parameter' */
                if (is_vi) {
                    search_ptr = cp_ptr;
                    continue;
                }
            }

            keep       = *param_ptr;
            *param_ptr = '\0';

            {
                size_t curr_str_len = curr_str ? strlen(curr_str) : 0;
                size_t len = strlen(curr_ptr) + strlen(params[i]) + 1;
                if (str_has_arith_char(params[i])) {
                    curr_str = TREALLOC(char, curr_str, curr_str_len + len + 2);
                    sprintf(curr_str + curr_str_len, "%s(%s)", curr_ptr, params[i]);
                } else {
                    curr_str = TREALLOC(char, curr_str, curr_str_len + len);
                    sprintf(curr_str + curr_str_len, "%s%s", curr_ptr, params[i]);
                }
            }

            *param_ptr = keep;
            search_ptr = curr_ptr = param_ptr + strlen(fcn->params[i]);
        }

        if (param_ptr == NULL) {
            if (curr_str == NULL) {
                curr_str = curr_ptr;
            } else {
                new_str = TMALLOC(char, strlen(curr_str) + strlen(curr_ptr) + 1);
                sprintf(new_str, "%s%s", curr_str, curr_ptr);
                tfree(curr_str);
                curr_str = new_str;
            }
        }
    }

    return curr_str;
}


static char*
inp_expand_macro_in_str(struct function_env *env, char *str)
{
    struct function *function;
    char *c;
    char *open_paren_ptr, *close_paren_ptr, *fcn_name, *params[1000];
    char *curr_ptr, *macro_str, *curr_str = NULL;
    int  num_parens, num_params;
    char *orig_ptr = str, *search_ptr = str, *orig_str = strdup(str);
    char keep;

    // printf("%s: enter(\"%s\")\n", __FUNCTION__, str);
    while ((open_paren_ptr = strchr(search_ptr, '(')) != NULL) {

        fcn_name = open_paren_ptr;
        while (--fcn_name >= search_ptr)
        /* function name consists of numbers, letters and special characters (VALIDCHARS) */
            if (!isalnum(*fcn_name) && !strchr(VALIDCHARS, *fcn_name))
                break;
        fcn_name++;

        search_ptr = open_paren_ptr + 1;
        if (open_paren_ptr == fcn_name)
            continue;

        *open_paren_ptr = '\0';

        function = find_function(env, fcn_name);

        *open_paren_ptr = '(';

        if (!function)
            continue;

        /* find the closing paren */
        num_parens = 1;
        for (c = open_paren_ptr + 1; *c; c++) {
            if (*c == '(')
                num_parens++;
            if (*c == ')' && --num_parens == 0)
                break;
        }

        if (num_parens) {
            fprintf(stderr, "ERROR: did not find closing parenthesis for function call in str: %s\n", orig_str);
            controlled_exit(EXIT_FAILURE);
        }

        close_paren_ptr = c;

        /*
         * if (ciprefix("v(", curr_ptr)) {
         *     // look for any commas and change to ' '
         *     char *str_ptr = curr_ptr;
         *     while (*str_ptr != '\0' && *str_ptr != ')') {
         *         if (*str_ptr == ',' || *str_ptr == '(')
         *             *str_ptr = ' '; str_ptr++; }
         *     if (*str_ptr == ')')
         *         *str_ptr = ' ';
         * }
         */

        /* get the parameters */
        curr_ptr = open_paren_ptr + 1;

        for (num_params = 0; curr_ptr < close_paren_ptr; curr_ptr++) {
            char *beg_parameter;
            int num_parens;
            if (isspace(*curr_ptr))
                continue;
            beg_parameter = curr_ptr;
            num_parens = 0;
            for (; curr_ptr < close_paren_ptr; curr_ptr++) {
                if (*curr_ptr == '(')
                    num_parens++;
                if (*curr_ptr == ')')
                    num_parens--;
                if (*curr_ptr == ',' && num_parens == 0)
                    break;
            }
            params[num_params++] =
                inp_expand_macro_in_str(env, copy_substring(beg_parameter, curr_ptr));
        }

        if (function->num_parameters != num_params) {
            fprintf(stderr, "ERROR: parameter mismatch for function call in str: %s\n", orig_str);
            controlled_exit(EXIT_FAILURE);
        }

        macro_str = inp_do_macro_param_replace(function, params);
        macro_str = inp_expand_macro_in_str(env, macro_str);
        keep  = *fcn_name;
        *fcn_name = '\0';
        {
            size_t curr_str_len = curr_str ? strlen(curr_str) : 0;
            size_t len = strlen(str) + strlen(macro_str) + 3;
            curr_str = TREALLOC(char, curr_str, curr_str_len + len);
            sprintf(curr_str + curr_str_len, "%s(%s)", str, macro_str);
        }
        *fcn_name = keep;
        tfree(macro_str);

        search_ptr = str = close_paren_ptr + 1;
    }

    if (curr_str == NULL) {
        curr_str = orig_ptr;
    } else {
        if (str != NULL) {
            size_t curr_str_len = strlen(curr_str);
            size_t len = strlen(str) + 1;
            curr_str = TREALLOC(char, curr_str, curr_str_len + len);
            sprintf(curr_str + curr_str_len, "%s", str);
        }
        tfree(orig_ptr);
    }

    tfree(orig_str);
    // printf("%s: --> \"%s\"\n", __FUNCTION__, curr_str);

    return curr_str;
}


static void
inp_expand_macros_in_func(struct function_env *env)
{
    struct function *f;

    for (f = env->functions; f ; f = f->next)
        f->macro = inp_expand_macro_in_str(env, f->macro);
}


static struct function_env *
new_function_env(struct function_env *up)
{
    struct function_env *env = TMALLOC(struct function_env, 1);

    env -> up = up;
    env -> functions = NULL;

    return env;
}


static struct function_env *
delete_function_env(struct function_env *env)
{
    struct function_env *up = env -> up;
    struct function *f;

    for (f = env -> functions; f; ) {
        struct function *here = f;
        f = f -> next;
        free_function(here);
        tfree(here);
    }

    tfree(env);

    return up;
}


static struct line *
inp_expand_macros_in_deck(struct function_env *env, struct line *c)
{
    env = new_function_env(env);

    inp_grab_func(env, c);

    inp_expand_macros_in_func(env);

    for (; c; c = c->li_next) {

        if (*c->li_line == '*')
            continue;

        if (ciprefix(".subckt", c->li_line)) {
            c = inp_expand_macros_in_deck(env, c->li_next);
            continue;
        }

        if (ciprefix(".ends", c->li_line))
            break;

        c->li_line = inp_expand_macro_in_str(env, c->li_line);
    }

    env = delete_function_env(env);

    return c;
}


/* Put {} around tokens for handling in numparam.
   Searches for the next '=' in the line to become active.
   Several exceptions (eg. no 'set' or 'b' lines, no .cmodel lines,
   no lines between .control and .endc, no .option lines).
   Special handling of vectors with [] and complex values with < >

   h_vogt 20 April 2008
   * For xspice and num_pram compatibility .cmodel added
   * .cmodel will be replaced by .model in inp_fix_param_values()
   * and then the entire line is skipped (will not be changed by this function).
   * Usage of numparam requires {} around the parameters in the .cmodel line.
   * May be obsolete?
   */

static void
inp_fix_param_values(struct line *c)
{
    char *beg_of_str, *end_of_str, *old_str, *equal_ptr, *new_str;
    char *vec_str, *tmp_str, *natok, *buffer, *newvec, *whereisgt;
    bool control_section = FALSE;
    wordlist *nwl;
    int parens;

    for (; c; c = c->li_next) {
        char *line = c->li_line;

        if (*line == '*' || (ciprefix(".param", line) && strchr(line, '{')))
            continue;

        if (ciprefix(".control", line)) {
            control_section = TRUE;
            continue;
        }

        if (ciprefix(".endc", line)) {
            control_section = FALSE;
            continue;
        }

        /* no handling of params in "option" lines */
        if (control_section || ciprefix(".option", line))
            continue;

        /* no handling of params in "set" lines */
        if (ciprefix("set", line))
            continue;

        /* no handling of params in B source lines */
        if (*line == 'b')
            continue;

        /* for xspice .cmodel: replace .cmodel with .model and skip entire line) */
        if (ciprefix(".cmodel", line)) {
            *(++line) = 'm';
            *(++line) = 'o';
            *(++line) = 'd';
            *(++line) = 'e';
            *(++line) = 'l';
            *(++line) = ' ';
            continue;
        }

        /* exclude CIDER models */
        if (ciprefix(".model", line) && (strstr(line, "numos") || strstr(line, "numd") ||
                                         strstr(line, "nbjt") || strstr(line, "nbjt2") ||
                                         strstr(line, "numd2")))
        {
            continue;
        }

        /* exclude CIDER devices with ic.file parameter */
        if (strstr(line, "ic.file"))
            continue;

        while ((equal_ptr = find_assignment(line)) != NULL) {

            // special case: .MEASURE {DC|AC|TRAN} result FIND out_variable WHEN out_variable2=out_variable3
            // no braces around out_variable3. out_variable3 may be v(...) or i(...)
            if (ciprefix(".meas", line))
                if (((equal_ptr[1] == 'v') || (equal_ptr[1] == 'i')) &&
                    (equal_ptr[2] == '('))
                {
                    // find closing ')' and skip token v(...) or i(...)
                    while (*equal_ptr != ')' && equal_ptr[1] != '\0')
                        equal_ptr++;
                    line = equal_ptr + 1;
                    continue;
                }

            beg_of_str = skip_ws(equal_ptr + 1);
            /* all cases where no {} have to be put around selected token */
            if (isdigit(*beg_of_str) ||
                *beg_of_str == '{' ||
                *beg_of_str == '.' ||
                *beg_of_str == '"' ||
                (*beg_of_str == '-' && isdigit(beg_of_str[1])) ||
                (*beg_of_str == '-' && beg_of_str[1] == '.' && isdigit(beg_of_str[2])) ||
                ciprefix("true", beg_of_str) ||
                ciprefix("false", beg_of_str))
            {
                line = equal_ptr + 1;
            } else if (*beg_of_str == '[') {
                /* A vector following the '=' token: code to put curly brackets around all params
                   inside a pair of square brackets */
                end_of_str = beg_of_str;
                while (*end_of_str != ']')
                    end_of_str++;
                /* string xx yyy from vector [xx yyy] */
                tmp_str = vec_str = copy_substring(beg_of_str + 1, end_of_str);

                /* work on vector elements inside [] */
                nwl = NULL;
                for (;;) {
                    natok = gettok(&vec_str);
                    if (!natok)
                        break;

                    buffer = TMALLOC(char, strlen(natok) + 4);
                    if (isdigit(*natok) || *natok == '{' || *natok == '.' ||
                        *natok == '"' || (*natok == '-' && isdigit(natok[1])) ||
                        ciprefix("true", natok) || ciprefix("false", natok) ||
                        eq(natok, "<") || eq(natok, ">"))
                    {
                        (void) sprintf(buffer, "%s", natok);
                        /* A complex value found inside a vector [< x1 y1> <x2 y2>] */
                        /* < xx and yy > have been dealt with before */
                        /* <xx */
                    } else if (*natok == '<') {
                        if (isdigit(natok[1]) ||
                            (natok[1] == '-' && isdigit(natok[2])))
                        {
                            (void) sprintf(buffer, "%s", natok);
                        } else {
                            *natok = '{';
                            (void) sprintf(buffer, "<%s}", natok);
                        }
                        /* yy> */
                    } else if (strchr(natok, '>')) {
                        if (isdigit(*natok) || (*natok == '-' && isdigit(natok[1]))) {
                            (void) sprintf(buffer, "%s", natok);
                        } else {
                            whereisgt = strchr(natok, '>');
                            *whereisgt = '}';
                            (void) sprintf(buffer, "{%s>", natok);
                        }
                        /* all other tokens */
                    } else {
                        (void) sprintf(buffer, "{%s}", natok);
                    }
                    tfree(natok);
                    nwl = wl_cons(copy(buffer), nwl);
                    tfree(buffer);
                }
                tfree(tmp_str);
                nwl = wl_reverse(nwl);
                /* new vector elements */
                newvec = wl_flatten(nwl);
                wl_free(nwl);
                /* insert new vector into actual line */
                *equal_ptr  = '\0';
                new_str = TMALLOC(char, strlen(c->li_line) + strlen(newvec) + strlen(end_of_str + 1) + 5);
                sprintf(new_str, "%s=[%s] %s", c->li_line, newvec, end_of_str+1);
                tfree(newvec);

                old_str    = c->li_line;
                c->li_line = new_str;
                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            } else if (*beg_of_str == '<') {
                /* A complex value following the '=' token: code to put curly brackets around all params
                   inside a pair < > */
                end_of_str = beg_of_str;
                while (*end_of_str != '>')
                    end_of_str++;
                /* string xx yyy from vector [xx yyy] */
                vec_str = copy_substring(beg_of_str + 1, end_of_str);

                /* work on tokens inside <> */
                nwl = NULL;
                for (;;) {
                    natok = gettok(&vec_str);
                    if (!natok)
                        break;

                    buffer = TMALLOC(char, strlen(natok) + 4);
                    if (isdigit(*natok) || *natok == '{' || *natok == '.' ||
                        *natok == '"' || (*natok == '-' && isdigit(natok[1])) ||
                        ciprefix("true", natok) || ciprefix("false", natok))
                    {
                        (void) sprintf(buffer, "%s", natok);
                    } else {
                        (void) sprintf(buffer, "{%s}", natok);
                    }
                    tfree(natok);
                    nwl = wl_cons(copy(buffer), nwl);
                    tfree(buffer);
                }
                nwl = wl_reverse(nwl);
                /* new elements of complex variable */
                newvec = wl_flatten(nwl);
                wl_free(nwl);
                /* insert new complex value into actual line */
                *equal_ptr  = '\0';
                new_str = TMALLOC(char, strlen(c->li_line) + strlen(newvec) + strlen(end_of_str + 1) + 5);
                sprintf(new_str, "%s=<%s> %s", c->li_line, newvec, end_of_str+1);
                tfree(newvec);

                old_str    = c->li_line;
                c->li_line = new_str;
                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            } else {
                /* put {} around token to be accepted as numparam */
                end_of_str = beg_of_str;
                parens = 0;
                while (*end_of_str != '\0' &&
                       (!isspace(*end_of_str) || (parens > 0)))
                {
                    if (*end_of_str == '(')
                        parens++;
                    if (*end_of_str == ')')
                        parens--;
                    end_of_str++;
                }

                *equal_ptr  = '\0';

                if (*end_of_str == '\0') {
                    new_str = TMALLOC(char, strlen(c->li_line) + strlen(beg_of_str) + 4);
                    sprintf(new_str, "%s={%s}", c->li_line, beg_of_str);

                } else {
                    *end_of_str = '\0';
                    new_str = TMALLOC(char, strlen(c->li_line) + strlen(beg_of_str) + strlen(end_of_str + 1) + 5);
                    sprintf(new_str, "%s={%s} %s", c->li_line, beg_of_str, end_of_str+1);
                }
                old_str    = c->li_line;
                c->li_line = new_str;

                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            }
        }
    }
}


static char*
get_param_name(char *line)
{
    char *beg;
    char *equal_ptr = strchr(line, '=');

    if (!equal_ptr) {
        fprintf(stderr, "ERROR: could not find '=' on parameter line '%s'!\n", line);
        controlled_exit(EXIT_FAILURE);
        return NULL;
    }

    equal_ptr = skip_back_ws_(equal_ptr, line);

    beg = skip_back_non_ws_(equal_ptr, line);

    return copy_substring(beg, equal_ptr);
}


static char*
get_param_str(char *line)
{
    char *equal_ptr = strchr(line, '=');

    if (equal_ptr)
        return skip_ws(equal_ptr + 1);
    else
        return line;
}


static int
inp_get_param_level(int param_num, char ***depends_on, char **param_names, char **param_strs, int total_params, int *level)
{
    int index1 = 0, comp_level = 0, temp_level = 0;
    int index2 = 0;

    if (level[param_num] != -1)
        return level[param_num];

    while (depends_on[param_num][index1] != NULL) {
        index2 = 0;
        while (index2 <= total_params &&
               param_names[index2] != depends_on[param_num][index1])
            index2++;

        if (index2 > total_params) {
            fprintf(stderr, "ERROR: unable to find dependency parameter for %s!\n", param_names[param_num]);
            controlled_exit(EXIT_FAILURE);
        }
        temp_level = inp_get_param_level(index2, depends_on, param_names, param_strs, total_params, level);
        temp_level++;

        if (comp_level < temp_level)
            comp_level = temp_level;
        index1++;
    }

    level[param_num] = comp_level;

    return comp_level;
}


static int
get_number_terminals(char *c)
{
    int i, j, k;
    char *name[12];
    char nam_buf[128];
    bool area_found = FALSE;

    switch (*c) {
    case 'r':
    case 'c':
    case 'l':
    case 'k':
    case 'f':
    case 'h':
    case 'b':
    case 'v':
    case 'i':
    case 'd':
        return 2;
        break;
    case 'u':
    case 'j':
    case 'w':
    case 'z':
        return 3;
        break;
    case 't':
    case 'o':
    case 'g':
    case 'e':
    case 's':
    case 'y':
        return 4;
        break;
    case 'm': /* recognition of 4, 5, 6, or 7 nodes for SOI devices needed */
        i = 0;
        /* find the first token with "off" or "=" in the line*/
        while ((i < 20) && (*c != '\0')) {
            char *inst = gettok_instance(&c);
            strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
            txfree(inst);
            if (strstr(nam_buf, "off") || strchr(nam_buf, '='))
                break;
            i++;
        }
        return i-2;
        break;
    case 'p': /* recognition of up to 100 cpl nodes */
        i = j = 0;
        /* find the last token in the line*/
        while ((i < 100) && (*c != '\0')) {
            char *tmp_inst = gettok_instance(&c);
            strncpy(nam_buf, tmp_inst, 32);
            tfree(tmp_inst);
            if (strchr(nam_buf, '='))
                j++;
            i++;
        }
        if (i == 100)
            return 0;
        return i - j - 2;
        break;
    case 'q': /* recognition of 3/4 terminal bjt's needed */
        /* QXXXXXXX NC NB NE <NS> MNAME <AREA> <OFF> <IC=VBE, VCE> <TEMP=T> */
        /* 12 tokens maximum */
        i = j = 0;
        while ((i < 12) && (*c != '\0')) {
            char *comma;
            name[i] = gettok_instance(&c);
            if (strstr(name[i], "off") || strchr(name[i], '='))
                j++;
            /* If we have IC=VBE, VCE instead of IC=VBE,VCE we need to inc j */
            if ((comma = strchr(name[i], ',')) != NULL && (*(++comma) == '\0'))
                j++;
            /* If we have IC=VBE , VCE ("," is a token) we need to inc j */
            if (eq(name[i], ","))
                j++;
            i++;
        }
        i--;
        area_found = FALSE;
        for (k = i; k > i-j-1; k--) {
            bool only_digits = TRUE;
            char *nametmp = name[k];
            /* MNAME has to contain at least one alpha character. AREA may be assumed
               if we have a token with only digits, and where the previous token does not
               end with a ',' */
            while (*nametmp) {
                if (isalpha(*nametmp) || (*nametmp == ','))
                    only_digits = FALSE;
                nametmp++;
            }
            if (only_digits && (strchr(name[k-1], ',') == NULL))
                area_found = TRUE;
        }
        for (k = i; k >= 0; k--)
            tfree(name[k]);
        if (area_found) {
            return i-j-2;
        } else {
            return i-j-1;
        }
        break;
    default:
        return 0;
        break;
    }
}


/* sort parameters based on parameter dependencies */

static void
inp_sort_params(struct line *start_card, struct line *end_card, struct line *card_bf_start, struct line *s_c, struct line *e_c)
{
    char *param_name = NULL, *param_str = NULL, *param_ptr = NULL;
    int  i, j, num_params = 0, ind = 0, max_level = 0, num_terminals = 0, ioff = 1;
    bool in_control = FALSE;

    bool found_in_list = FALSE;

    struct line *ptr;
    char *str_ptr, *beg, *end, *new_str;
    int  skipped = 0;
    int arr_size = 12000;

    int *level;
    int *param_skip;
    char **param_names;
    char **param_strs;
    char ***depends_on;
    struct line **ptr_array;
    struct line **ptr_array_ordered;

    NG_IGNORE(end_card);

    if (start_card == NULL)
        return;

    /* determine the number of lines with .param */

    for (ptr = start_card; ptr; ptr = ptr->li_next)
        if (strchr(ptr->li_line, '='))
            num_params++;

    arr_size = num_params;
    num_params = 0; /* This is just to keep the code in row 2907ff. */

    // dynamic memory allocation
    level = TMALLOC(int, arr_size);
    param_skip = TMALLOC(int, arr_size);
    param_names = TMALLOC(char *, arr_size);
    param_strs = TMALLOC(char *, arr_size);

    /* array[row][column] -> depends_on[array_size][100] */
    /* rows */
    depends_on = TMALLOC(char **, arr_size);
    /* columns */
    for (i = 0; i < arr_size; i++)
        depends_on[i] = TMALLOC(char *, 100);

    ptr_array = TMALLOC(struct line *, arr_size);
    ptr_array_ordered = TMALLOC(struct line *, arr_size);

    ptr = start_card;
    for (ptr = start_card; ptr; ptr = ptr->li_next)
        // ignore .param lines without '='
        if (strchr(ptr->li_line, '=')) {
            depends_on[num_params][0] = NULL;
            level[num_params]         = -1;
            param_names[num_params]   = get_param_name(ptr->li_line); /* strdup in fcn */
            param_strs[num_params]    = strdup(get_param_str(ptr->li_line));

            ptr_array[num_params++]   = ptr;
        }

    // look for duplicately defined parameters and mark earlier one to skip
    // param list is ordered as defined in netlist
    for (i = 0; i < num_params; i++)
        param_skip[i] = 0;

    for (i = 0; i < num_params; i++)
        for (j = num_params-1; j >= 0 && !param_skip[i]; j--)
            if (i != j && i < j && strcmp(param_names[i], param_names[j]) == 0) {
                // skip earlier one in list
                param_skip[i] = 1;
                skipped++;
            }

    for (i = 0; i < num_params; i++) {
        if (param_skip[i] == 1)
            continue;

        param_name = param_names[i];
        for (j = 0; j < num_params; j++) {
//        for (j = i + 1; j < num_params; j++) {  /* FIXME: to be tested */
            if (j == i)
                continue;

            param_str = param_strs[j];

            while ((param_ptr = strstr(param_str, param_name)) != NULL) {
                ioff = (strchr(param_ptr, '}') != NULL ? 1 : 0);  /* want prevent wrong memory access below */
                /* looking for curly braces or other string limiter */
                if ((!isalnum(param_ptr[-ioff]) && param_ptr[-ioff] != '_' &&
                     !isalnum(param_ptr[strlen(param_name)]) &&
                     param_ptr[strlen(param_name)] != '_') ||
                    strcmp(param_ptr, param_name) == 0)
                { /* this are cases without curly braces */
                    ind = 0;
                    found_in_list = FALSE;
                    while (depends_on[j][ind] != NULL) {
                        if (strcmp(param_name, depends_on[j][ind]) == 0) {
                            found_in_list = TRUE;
                            break;
                        }
                        ind++;
                    }
                    if (!found_in_list) {
                        depends_on[j][ind++] = param_name;
                        depends_on[j][ind]   = NULL;
                    }
                    break;
                }
                param_str = param_ptr + strlen(param_name);
            }
        }
    }

    for (i = 0; i < num_params; i++) {
        level[i] = inp_get_param_level(i, depends_on, param_names, param_strs, num_params, level);
        if (max_level < level[i])
            max_level = level[i];
    }

    /* look for unquoted parameters and quote them */

    in_control = FALSE;
    for (ptr = s_c; ptr && ptr != e_c; ptr = ptr->li_next) {

        char *curr_line = ptr->li_line;

        if (ciprefix(".control", curr_line)) {
            in_control = TRUE;
            continue;
        }

        if (ciprefix(".endc", curr_line)) {
            in_control = FALSE;
            continue;
        }

        if (in_control || curr_line[0] == '.' || curr_line[0] == '*')
            continue;

/* FIXME: useless and potentially buggy code, when called from line 2225:
   we check parameters like l={length}, but not complete lines: We just
   live from the fact, that there are device names for all characters
   of the alphabet */
        num_terminals = get_number_terminals(curr_line);

        if (num_terminals <= 0)
            continue;

        for (i = 0; i < num_params; i++) {
            str_ptr = curr_line;

/* FIXME: useless and potentially buggy code, when called from line 2225:
   we check parameters like
   l={length}, but not complete lines: this will always lead to str_ptr = "" */
            for (j = 0; j < num_terminals+1; j++) {
                str_ptr = skip_non_ws(str_ptr);
                str_ptr = skip_ws(str_ptr);
            }

/* FIXME: useless and potentially buggy code: we check parameters like
   l={length}, but the following will not work for such a parameter string.
   We just live from the fact that str_ptr = "". */
            while ((str_ptr = strstr(str_ptr, param_names[i])) != NULL) {
                /* make sure actually have the parameter name */
                char before = *(str_ptr-1);
                char after  = *(str_ptr+strlen(param_names[i]));
                if (!(is_arith_char(before) || isspace(before) || (str_ptr-1) < curr_line) ||
                    !(is_arith_char(after)  || isspace(after)  || after == '\0')) {
                    str_ptr ++;
                    continue;
                }
                beg = str_ptr - 1;
                end = str_ptr + strlen(param_names[i]);
                if ((isspace(*beg) || *beg == '=') &&
                    (isspace(*end) || *end == '\0' || *end == ')')) {
                    if (isspace(*beg)) {
                        while (isspace(*beg))
                            beg--;
                        if (*beg != '{')
                            beg++;
                        str_ptr = beg;
                    }
                    if (isspace(*end)) {
                        /* possible case: "{  length }" -> {length} */
                        while (*end && isspace(*end))
                            end++;
                        if (*end == '}')
                            end++;
                        else
                            end--;
                    }
                    *str_ptr = '\0';
                    if (*end != '\0') {
                        new_str = TMALLOC(char, strlen(curr_line) + strlen(param_names[i]) + strlen(end) + 3);
                        sprintf(new_str, "%s{%s}%s", curr_line, param_names[i], end);
                    } else {
                        new_str = TMALLOC(char, strlen(curr_line) + strlen(param_names[i]) + 3);
                        sprintf(new_str, "%s{%s}", curr_line, param_names[i]);
                    }
                    str_ptr = new_str + strlen(curr_line) + strlen(param_names[i]);

                    tfree(ptr->li_line);
                    curr_line = ptr->li_line = new_str;
                }
                str_ptr++;
            }
        }
    }

    ind = 0;
    for (i = 0; i <= max_level; i++)
        for (j = num_params-1; j >= 0; j--)
            if (level[j] == i)
                if (param_skip[j] == 0)
                    ptr_array_ordered[ind++] = ptr_array[j];

    num_params -= skipped;
    if (ind != num_params) {
        fprintf(stderr, "ERROR: found wrong number of parameters during levelization ( %d instead of %d parameter s)!\n", ind, num_params);
        controlled_exit(EXIT_FAILURE);
    }

    /* fix next ptrs */
    ptr                                      = card_bf_start->li_next;
    card_bf_start->li_next                   = ptr_array_ordered[0];
    ptr_array_ordered[num_params-1]->li_next = ptr;
    for (i = 0; i < num_params-1; i++)
        ptr_array_ordered[i]->li_next = ptr_array_ordered[i+1];

    // clean up memory
    for (i = 0; i < arr_size; i++) {
        tfree(param_names[i]);
        tfree(param_strs[i]);
    }

    tfree(level);
    tfree(param_skip);
    tfree(param_names);
    tfree(param_strs);

    for (i = 0; i< arr_size; i++)
        tfree(depends_on[i]);
    tfree(depends_on);

    tfree(ptr_array);
    tfree(ptr_array_ordered);

}


static void
inp_add_params_to_subckt(struct names *subckt_w_params, struct line *subckt_card)
{
    struct line *card        = subckt_card->li_next;
    char        *subckt_line = subckt_card->li_line;
    char        *new_line, *param_ptr, *subckt_name, *end_ptr;

    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        if (!ciprefix(".param", curr_line))
            break;

        param_ptr = strchr(curr_line, ' ');
        param_ptr = skip_ws(param_ptr);

        if (!strstr(subckt_line, "params:")) {
            new_line = TMALLOC(char, strlen(subckt_line) + strlen("params: ") + strlen(param_ptr) + 2);
            sprintf(new_line, "%s params: %s", subckt_line, param_ptr);

            subckt_name = skip_non_ws(subckt_line);
            subckt_name = skip_ws(subckt_name);
            end_ptr = skip_non_ws(subckt_name);
            add_name(subckt_w_params, copy_substring(subckt_name, end_ptr));
        } else {
            new_line = TMALLOC(char, strlen(subckt_line) + strlen(param_ptr) + 2);
            sprintf(new_line, "%s %s", subckt_line, param_ptr);
        }

        tfree(subckt_line);
        subckt_line = new_line;

        *curr_line = '*';
    }

    subckt_card->li_line = subckt_line;
}


/*
 * process a sequence of decks
 *   starting from a         `.suckt' deck
 *   upto the corresponding  `.ends'  deck
 * return a pointer to the terminating `.ends' deck
 *
 * recursivly descend
 *   when another `.subckt' is found
 *
 * parameters are removed from the main list
 *   and collected into a local list `first_param_card'
 * then processed and reinserted into the main list
 *
 */

static struct line *
inp_reorder_params_subckt(struct names *subckt_w_params, struct line *subckt_card)
{
    struct line *first_param_card = NULL;
    struct line *last_param_card = NULL;

    struct line *prev_card = subckt_card;
    struct line *c         = subckt_card->li_next;

    /* move .param lines to beginning of deck */
    while (c != NULL) {

        char *curr_line = c->li_line;

        if (*curr_line == '*') {
            prev_card = c;
            c = c->li_next;
            continue;
        }

        if (ciprefix(".subckt", curr_line)) {
            prev_card = inp_reorder_params_subckt(subckt_w_params, c);
            c         = prev_card->li_next;
            continue;
        }

        if (ciprefix(".ends", curr_line)) {
            if (first_param_card) {
                inp_sort_params(first_param_card, last_param_card, subckt_card, subckt_card, c);
                inp_add_params_to_subckt(subckt_w_params, subckt_card);
            }
            return c;
        }

        if (ciprefix(".param", curr_line)) {
            if (first_param_card)
                last_param_card->li_next = c;
            else
                first_param_card = c;

            last_param_card    = c;
            prev_card->li_next = c->li_next;
            c                  = c->li_next;

            last_param_card->li_next = NULL;
            continue;
        }

        prev_card = c;
        c         = c->li_next;
    }

    /* the terminating `.ends' deck wasn't found */
    controlled_exit(EXIT_FAILURE);
    return NULL;
}


static void
inp_reorder_params(struct names *subckt_w_params, struct line *deck, struct line *list_head, struct line *end)
{
    struct line *first_param_card = NULL;
    struct line *last_param_card = NULL;

    struct line *prev_card = list_head;
    struct line *c = deck;

    /* move .param lines to beginning of deck */
    while (c != NULL) {

        char *curr_line = c->li_line;

        if (*curr_line == '*') {
            prev_card = c;
            c = c->li_next;
            continue;
        }

        if (ciprefix(".subckt", curr_line)) {
            prev_card = inp_reorder_params_subckt(subckt_w_params, c);
            c         = prev_card->li_next;
            continue;
        }

        /* check for an unexpected extra `.ends' deck */
        if (ciprefix(".ends", curr_line)) {
            fprintf(stderr, "Error: Unexpected extra .ends in line:\n  %s.\n", curr_line);
            controlled_exit(EXIT_FAILURE);
        }

        if (ciprefix(".param", curr_line)) {
            if (first_param_card)
                last_param_card->li_next = c;
            else
                first_param_card = c;

            last_param_card    = c;
            prev_card->li_next = c->li_next;
            c                  = c->li_next;

            last_param_card->li_next = NULL;
            continue;
        }

        prev_card = c;
        c = c->li_next;
    }

    inp_sort_params(first_param_card, last_param_card, list_head, deck, end);
}


// iterate through deck and find lines with multiply defined parameters
//
// split line up into multiple lines and place those new lines immediately
// afetr the current multi-param line in the deck

static int
inp_split_multi_param_lines(struct line *card, int line_num)
{
    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        if (*curr_line == '*')
            continue;

        if (ciprefix(".param", curr_line)) {

            struct line *param_end, *param_beg;
            char *equal_ptr, **array;
            int i, counter = 0;

            while ((equal_ptr = find_assignment(curr_line)) != NULL) {
                counter++;
                curr_line = equal_ptr + 1;
            }

            if (counter <= 1)
                continue;

            array = TMALLOC(char *, counter);

            // need to split multi param line
            curr_line = card->li_line;
            counter   = 0;
            while ((equal_ptr = find_assignment(curr_line)) != NULL) {

                char keep, *beg_param, *end_param, *new_line;

                bool get_expression = FALSE;
                bool get_paren_expression = FALSE;

                beg_param = skip_back_ws_(equal_ptr, curr_line);
                beg_param = skip_back_non_ws_(beg_param, curr_line);
                end_param = skip_ws(equal_ptr + 1);
                while (*end_param != '\0' && (!isspace(*end_param) || get_expression || get_paren_expression)) {
                    if (*end_param == '{')
                        get_expression = TRUE;
                    if (*end_param == '(')
                        get_paren_expression = TRUE;
                    if (*end_param == '}')
                        get_expression = FALSE;
                    if (*end_param == ')')
                        get_paren_expression = FALSE;
                    end_param++;
                }
                keep       = *end_param;
                *end_param = '\0';
                new_line   = TMALLOC(char, strlen(".param ") + strlen(beg_param) + 1);
                sprintf(new_line, ".param %s", beg_param);
                array[counter++] = new_line;
                *end_param = keep;
                curr_line = end_param;
            }

            param_beg = param_end = NULL;

            for (i = 0; i < counter; i++) {
                struct line *x = xx_new_line(NULL, array[i], line_num++, 0);

                if (param_end)
                    param_end->li_next = x;
                else
                    param_beg = x;

                param_end = x;
            }

            tfree(array);

            // comment out current multi-param line
            *(card->li_line)   = '*';
            // insert new param lines immediately after current line
            param_end->li_next = card->li_next;
            card->li_next      = param_beg;
            // point 'card' pointer to last in scalar list
            card               = param_end;
        }
    }

    return line_num;
}


/* ps compatibility:
   ECOMP 3 0 TABLE {V(1,2)} = (-1 0V) (1, 10V)
   -->
   ECOMP 3 0 int3 int0 1
   BECOMP int3 int0 V = pwl(V(1,2), -2, 0, -1, 0 , 1, 10V, 2, 10V)

   GD16 16 1 TABLE {V(16,1)} ((-100V,-1pV)(0,0)(1m,1u)(2m,1m))
   -->
   GD16 16 1 int_16 int_1 1
   BGD16 int_16 int_1 V = pwl (v(16,1) , -100V , -1pV , 0 , 0 , 1m , 1u , 2m , 1m)
*/

/* hs compatibility:
   Exxx n1 n2 VCVS n3 n4 gain --> Exxx n1 n2 n3 n4 gain
   Gxxx n1 n2 VCCS n3 n4 tr --> Gxxx n1 n2 n3 n4 tr

   Two step approach to keep the original names for reuse,
   i.e. for current measurements like i(Exxx):
   Exxx n1 n2 VOL = {equation}
   -->
   Exxx n1 n2 int1 0 1
   BExxx int1 0 V = {equation}

   Gxxx n1 n2 CUR = {equation}
   -->
   Gxxx n1 n2 int1 0 1
   BGxxx int1 0 V = {equation}

   Do the following transformations only if {equation} contains
   simulation output like v(node), v(node1, node2), i(branch).
   Otherwise let do numparam the substitutions (R=const is handled
   in inp2r.c).

   Rxxx n1 n2 R = {equation} or Rxxx n1 n2 {equation}
   -->
   BRxxx n1 n2 I = V(n1,n2)/{equation}

   Unfortunately the capability for ac noise calculation of
   resistance may be lost.

   Cxxx n1 n2 C = {equation} or Cxxx n1 n2 {equation}
   -->
   Exxx  n-aux 0  n1 n2  1
   Cxxx  n-aux 0         1
   Bxxx  n2 n1  I = i(Exxx) * equation

   Lxxx n1 n2 L = {equation} or Lxxx n1 n2 {equation}
   -->
   Fxxx n-aux 0  Bxxx -1
   Lxxx n-aux 0      1
   Bxxx n1 n2 V = v(n-aux) * 1e-16

*/

static void
inp_compat(struct line *card)
{
    char *str_ptr, *cut_line, *title_tok, *node1, *node2;
    char *out_ptr, *exp_ptr, *beg_ptr, *end_ptr, *copy_ptr, *del_ptr;
    char *xline;
    size_t xlen, i, pai = 0, paui = 0, ii;
    char *ckt_array[100];
    struct line *new_line;

    struct line  *param_end = NULL, *param_beg = NULL;
    int skip_control = 0;

    char *equation, *tc1_ptr = NULL, *tc2_ptr = NULL;
    double tc1 = 0.0, tc2 = 0.0;

    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (*curr_line == '*')
            continue;

        if (*curr_line == 'e') {
            /*    Exxx n1 n2 VCVS n3 n4 gain --> Exxx n1 n2 n3 n4 gain
                  remove vcvs */
            replace_token(curr_line, "vcvs", 4, 7);

            /* Exxx n1 n2 value={equation}
               -->
               Exxx n1 n2   vol={equation} */
            if ((str_ptr = strstr(curr_line, "value=")) != NULL) {
                str_ptr[0] = ' ';
                str_ptr[1] = ' ';
                str_ptr[2] = 'v';
                str_ptr[3] = 'o';
                str_ptr[4] = 'l';
            }
            /* Exxx n1 n2 TABLE {expression} = (x0, y0) (x1, y1) (x2, y2)
               -->
               Exxx n1 n2 int1 0 1
               BExxx int1 0 V = pwl (expression, x0-(x2-x0)/2, y0, x0, y0, x1, y1, x2, y2, x2+(x2-x0)/2, y2)
            */
            if ((str_ptr = strstr(curr_line, "table")) != NULL) {
                char *expression, *firstno, *ffirstno, *secondno, *midline, *lastno, *lastlastno;
                double fnumber, lnumber, delta;
                int nerror;
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                // Exxx  n1 n2 int1 0 1
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2)
                    + 20 - 4*2 + 1;
                ckt_array[0] = TMALLOC(char, xlen);
                sprintf(ckt_array[0], "%s %s %s %s_int1 0 1",
                        title_tok, node1, node2, title_tok);
                // get the expression
                str_ptr = gettok(&cut_line); /* ignore 'table' */
                if (!cieq(str_ptr, "table")) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                tfree(str_ptr);
                str_ptr =  gettok_char(&cut_line, '{', FALSE, FALSE);
                expression = gettok_char(&cut_line, '}', TRUE, TRUE); /* expression */
                if (!expression || !str_ptr) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                tfree(str_ptr);
                /* remove '{' and '}' from expression */
                if ((str_ptr = strchr(expression, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(expression, '}')) != NULL)
                    *str_ptr = ' ';
                /* cut_line may now have a '=', if yes, it will have '{' and '}'
                   (braces around token after '=') */
                if ((str_ptr = strchr(cut_line, '=')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(cut_line, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(cut_line, '}')) != NULL)
                    *str_ptr = ' ';
                /* get first two numbers to establish extrapolation */
                str_ptr = cut_line;
                ffirstno = gettok_node(&cut_line);
                if (!ffirstno) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                firstno = copy(ffirstno);
                fnumber = INPevaluate(&ffirstno, &nerror, TRUE);
                secondno = gettok_node(&cut_line);
                midline = cut_line;
                cut_line = strrchr(str_ptr, '(');
                if (!cut_line) {
                    fprintf(stderr, "Error: bad syntax in line %d (missing parentheses)\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                /* replace '(' with ',' and ')' with ' ' */
                for (; *str_ptr; str_ptr++)
                    if (*str_ptr == '(')
                        *str_ptr = ',';
                    else if (*str_ptr == ')')
                        *str_ptr = ' ';
                /* scan for last two numbers */
                lastno = gettok_node(&cut_line);
                lnumber = INPevaluate(&lastno, &nerror, FALSE);
                /* check for max-min and take half the difference for delta */
                delta = (lnumber-fnumber)/2.;
                lastlastno = gettok_node(&cut_line);
                if (!secondno || (*midline == 0) || (delta <= 0.) || !lastlastno) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                xlen = 2*strlen(title_tok) + strlen(expression) + 14 + strlen(firstno) +
                    2*strlen(secondno) + strlen(midline) + 14 +
                    strlen(lastlastno) + 50;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 v = pwl(%s, %e, %s, %s, %s, %s, %e, %s)",
                        title_tok, title_tok, expression, fnumber-delta, secondno, firstno, secondno,
                        midline, lnumber + delta, lastlastno);

                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable e line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(firstno);
                tfree(lastlastno);
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
            }
            /* Exxx n1 n2 VOL = {equation}
               -->
               Exxx n1 n2 int1 0 1
               BExxx int1 0 V = {equation}
            */
            /* search for ' vol=' or ' vol =' */
            if (((str_ptr = strchr(curr_line, '=')) != NULL) && prefix("vol", skip_back_non_ws(skip_back_ws(str_ptr)))) {
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                /* Find equation, starts with '{', till end of line */
                str_ptr = strchr(cut_line, '{');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed E line: %s\n", curr_line);
                    controlled_exit(EXIT_FAILURE);
                }

                // Exxx  n1 n2 int1 0 1
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2)
                    + 20 - 4*2 + 1;
                ckt_array[0] = TMALLOC(char, xlen);
                sprintf(ckt_array[0], "%s %s %s %s_int1 0 1",
                        title_tok, node1, node2, title_tok);
                // BExxx int1 0 V = {equation}
                xlen = 2*strlen(title_tok) + strlen(str_ptr)
                    + 20 - 3*2 + 1;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 v = %s",
                        title_tok, title_tok, str_ptr);

                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable e line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
            }
        } else if (*curr_line == 'g') {
            /* Gxxx n1 n2 VCCS n3 n4 tr --> Gxxx n1 n2 n3 n4 tr
               remove vccs */
            replace_token(curr_line, "vccs", 4, 7);

            /* Gxxx n1 n2 value={equation}
               -->
               Gxxx n1 n2   cur={equation} */
            if ((str_ptr = strstr(curr_line, "value=")) != NULL) {
                str_ptr[0] = ' ';
                str_ptr[1] = ' ';
                str_ptr[2] = 'c';
                str_ptr[3] = 'u';
                str_ptr[4] = 'r';
            }

            /* Gxxx n1 n2 TABLE {expression} = (x0, y0) (x1, y1) (x2, y2)
               -->
               Gxxx n1 n2 int1 0 1
               BGxxx int1 0 V = pwl (expression, x0-(x2-x0)/2, y0, x0, y0, x1, y1, x2, y2, x2+(x2-x0)/2, y2)
            */
            if ((str_ptr = strstr(curr_line, "table")) != NULL) {
                char *expression, *firstno, *ffirstno, *secondno, *midline, *lastno, *lastlastno;
                char *m_ptr, *m_token;
                double fnumber, lnumber, delta;
                int nerror;
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                // Gxxx  n1 n2 int1 0 1
                // or
                // Gxxx  n1 n2 int1 0 m='expr'
                /* find multiplier m at end of line */
                m_ptr = strstr(cut_line, "m=");
                if (m_ptr) {
                    m_token = copy(m_ptr + 2);  // get only the expression
                    *m_ptr = '\0';
                }
                else
                    m_token = copy("1");
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2)
                    + 20 - 4*2 + strlen(m_token);
                ckt_array[0] = TMALLOC(char, xlen);
                sprintf(ckt_array[0], "%s %s %s %s_int1 0 %s",
                        title_tok, node1, node2, title_tok, m_token);
                // get the expression
                str_ptr = gettok(&cut_line); /* ignore 'table' */
                if (!cieq(str_ptr, "table")) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                tfree(str_ptr);
                str_ptr =  gettok_char(&cut_line, '{', FALSE, FALSE);
                expression = gettok_char(&cut_line, '}', TRUE, TRUE); /* expression */
                if (!expression || !str_ptr) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                tfree(str_ptr);
                /* remove '{' and '}' from expression */
                if ((str_ptr = strchr(expression, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(expression, '}')) != NULL)
                    *str_ptr = ' ';
                /* cut_line may now have a '=', if yes, it will have '{' and '}'
                   (braces around token after '=') */
                if ((str_ptr = strchr(cut_line, '=')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(cut_line, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(cut_line, '}')) != NULL)
                    *str_ptr = ' ';
                /* get first two numbers to establish extrapolation */
                str_ptr = cut_line;
                ffirstno = gettok_node(&cut_line);
                if (!ffirstno) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                firstno = copy(ffirstno);
                fnumber = INPevaluate(&ffirstno, &nerror, TRUE);
                secondno = gettok_node(&cut_line);
                midline = cut_line;
                cut_line = strrchr(str_ptr, '(');
                /* replace '(' with ',' and ')' with ' ' */
                for (; *str_ptr; str_ptr++)
                    if (*str_ptr == '(')
                        *str_ptr = ',';
                    else if (*str_ptr == ')')
                        *str_ptr = ' ';
                /* scan for last two numbers */
                lastno = gettok_node(&cut_line);
                lnumber = INPevaluate(&lastno, &nerror, FALSE);
                /* check for max-min and take half the difference for delta */
                delta = (lnumber-fnumber)/2.;
                lastlastno = gettok_node(&cut_line);
                if (!secondno || (*midline == 0) || (delta <= 0.) || !lastlastno) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->li_linenum_orig, card->li_line);
                    controlled_exit(EXIT_BAD);
                }
                /* BGxxx int1 0 V = pwl (expression, x0-(x2-x0)/2, y0, x0, y0, x1, y1, x2, y2, x2+(x2-x0)/2, y2) */
                xlen = 2*strlen(title_tok) + strlen(expression) + 14 + strlen(firstno) +
                    2*strlen(secondno) + strlen(midline) + 14 +
                    strlen(lastlastno) + 50;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 v = pwl(%s, %e, %s, %s, %s, %s, %e, %s)",
                        title_tok, title_tok, expression, fnumber-delta, secondno, firstno, secondno,
                        midline, lnumber + delta, lastlastno);

                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable e line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(firstno);
                tfree(lastlastno);
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                tfree(m_token);
            }
            /*
              Gxxx n1 n2 CUR = {equation}
              -->
              Gxxx n1 n2 int1 0 1
              BGxxx int1 0 V = {equation}
            */
            /* search for ' cur=' or ' cur =' */
            if (((str_ptr = strchr(curr_line, '=')) != NULL) && prefix("cur", skip_back_non_ws(skip_back_ws(str_ptr)))) {
                char *m_ptr, *m_token;
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                /* Find equation, starts with '{', till end of line */
                str_ptr = strchr(cut_line, '{');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed G line: %s\n", curr_line);
                    controlled_exit(EXIT_FAILURE);
                }
                /* find multiplier m at end of line */
                m_ptr = strstr(cut_line, "m=");
                if (m_ptr) {
                    m_token = copy(m_ptr + 2); //get only the expression
                    *m_ptr = '\0';
                }
                else
                    m_token = copy("1");
                // Gxxx  n1 n2 int1 0 1
                // or
                // Gxxx  n1 n2 int1 0 m='expr'
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2)
                    + 20 - 4*2 + strlen(m_token);
                ckt_array[0] = TMALLOC(char, xlen);
                sprintf(ckt_array[0], "%s %s %s %s_int1 0 %s",
                        title_tok, node1, node2, title_tok, m_token);
                // BGxxx int1 0 V = {equation}
                xlen = 2*strlen(title_tok) + strlen(str_ptr)
                    + 20 - 3*2 + 1;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 v = %s",
                        title_tok, title_tok, str_ptr);

                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable g line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(title_tok);
                tfree(m_token);
                tfree(node1);
                tfree(node2);
            }
        }

        /* F element compatibility */
        else if (*curr_line == 'f') {
            char actchar, *beg_tstr, *equastr, *vnamstr;
            /* Fxxx n1 n2 CCCS vnam gain --> Fxxx n1 n2 vnam gain
               remove cccs */
            replace_token(curr_line, "cccs", 4, 6);
            /* Deal with
               Fxxx n1 n2 vnam {equation}
               if equation contains the 'temper' token */
            beg_tstr = curr_line;
            while ((beg_tstr = strstr(beg_tstr, "temper")) != NULL) {
                actchar = *(beg_tstr - 1);
                if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '{')) {
                    beg_tstr++;
                    continue;
                }
                actchar = *(beg_tstr + 6);
                if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '}')) {
                    beg_tstr++;
                    continue;
                }
                /* we have found a true 'temper' */
                cut_line = curr_line;
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                vnamstr = gettok(&cut_line);
                equastr = gettok(&cut_line);
                /*
                Fxxx n1 n2 vnam {equation}
                -->
                Fxxx n1 n2 vbFxxx -1
                bFxxx int1 0 i = i(vnam)*{equation}
                vbFxxx int1 0 0
                */
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2) + 9;
                ckt_array[0] = TMALLOC(char, xlen);
                //Fxxx n1 n2 VBFxxx -1
                sprintf(ckt_array[0], "%s %s %s vb%s -1",
                        title_tok, node1, node2, title_tok);
                //BFxxx BFxxx_int1 0 I = I(vnam)*{equation}
                xlen = 2*strlen(title_tok) + strlen(vnamstr) + strlen(equastr)
                    + 23;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 i = i(%s) * (%s)",
                        title_tok, title_tok, vnamstr, equastr);
                //VBFxxx int1 0 0
                xlen = 2*strlen(title_tok)
                    + 16;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "vb%s %s_int1 0 dc 0",
                        title_tok, title_tok);
                // insert new three lines immediately after current line
                for (i = 0; i < 3; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable f line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(title_tok);
                tfree(vnamstr);
                tfree(equastr);
                tfree(node1);
                tfree(node2);
                break;
            }
        }
        /* H element compatibility */
        else if (*curr_line == 'h') {
            char actchar, *beg_tstr, *equastr, *vnamstr;
            /* Hxxx n1 n2 CCVS vnam transres --> Hxxx n1 n2 vnam transres
               remove cccs */
            replace_token(curr_line, "ccvs", 4, 6);
            /* Deal with
               Hxxx n1 n2 vnam {equation}
               if equation contains the 'temper' token */
            beg_tstr = curr_line;
            while ((beg_tstr = strstr(beg_tstr, "temper")) != NULL) {
                actchar = *(beg_tstr - 1);
                if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '{')) {
                    beg_tstr++;
                    continue;
                }
                actchar = *(beg_tstr + 6);
                if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '}')) {
                    beg_tstr++;
                    continue;
                }
                /* we have found a true 'temper' */
                cut_line = curr_line;
                title_tok = gettok(&cut_line);
                node1 =  gettok(&cut_line);
                node2 =  gettok(&cut_line);
                vnamstr = gettok(&cut_line);
                equastr = gettok(&cut_line);
                /*
                Hxxx n1 n2 vnam {equation}
                -->
                Hxxx n1 n2 vbHxxx -1
                bHxxx int1 0 i = i(vnam)*{equation}
                vbHxxx int1 0 0
                */
                xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2) + 9;
                ckt_array[0] = TMALLOC(char, xlen);
                //Hxxx n1 n2 VBHxxx -1
                sprintf(ckt_array[0], "%s %s %s vb%s -1",
                        title_tok, node1, node2, title_tok);
                //BHxxx BHxxx_int1 0 I = I(vnam)*{equation}
                xlen = 2*strlen(title_tok) + strlen(vnamstr) + strlen(equastr)
                    + 23;
                ckt_array[1] = TMALLOC(char, xlen);
                sprintf(ckt_array[1], "b%s %s_int1 0 i = i(%s) * (%s)",
                        title_tok, title_tok, vnamstr, equastr);
                //VBHxxx int1 0 0
                xlen = 2*strlen(title_tok)
                    + 16;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "vb%s %s_int1 0 dc 0",
                        title_tok, title_tok);
                // insert new three lines immediately after current line
                for (i = 0; i < 3; i++) {
                    struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable h line
                *(card->li_line)   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                tfree(title_tok);
                tfree(vnamstr);
                tfree(equastr);
                tfree(node1);
                tfree(node2);
                break;
            }
        }

        /* Rxxx n1 n2 R = {equation} or Rxxx n1 n2 {equation}
           -->
           BRxxx pos neg I = V(pos, neg)/{equation}
        */
        else if (*curr_line == 'r') {
            cut_line = curr_line;
            /* make BRxxx pos neg I = V(pos, neg)/{equation}*/
            title_tok = gettok(&cut_line);
            node1 =  gettok(&cut_line);
            node2 =  gettok(&cut_line);
            /* check only after skipping Rname and nodes, either may contain time (e.g. Rtime)*/
            if ((!strstr(cut_line, "v(")) &&  (!strstr(cut_line, "i(")) &&
                (!strstr(cut_line, "temper")) &&  (!strstr(cut_line, "hertz")) &&
                (!strstr(cut_line, "time"))) {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }

            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                /* if not, equation may start with a '(' */
                str_ptr = strchr(cut_line, '(');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed R line: %s\n", curr_line);
                    controlled_exit(EXIT_FAILURE);
                }
                equation = gettok_char(&str_ptr, ')', TRUE, TRUE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);
            str_ptr = strstr(cut_line, "tc1");
            if (str_ptr) {
                /* We need to have 'tc1=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc1_ptr = strchr(str_ptr, '=');
                    if (tc1_ptr)
                        tc1 = atof(tc1_ptr+1);
                }
            }
            str_ptr = strstr(cut_line, "tc2");
            if (str_ptr) {
                /* We need to have 'tc2=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc2_ptr = strchr(str_ptr, '=');
                    if (tc2_ptr)
                        tc2 = atof(tc2_ptr+1);
                }
            }
            if ((tc1_ptr == NULL) && (tc2_ptr == NULL)) {
                xlen = strlen(title_tok) + strlen(node1) + strlen(node2) +
                    strlen(node1) + strlen(node2) + strlen(equation)  +
                    28 - 6*2 + 1;
                xline = TMALLOC(char, xlen);
                sprintf(xline, "b%s %s %s i = v(%s, %s)/(%s)", title_tok, node1, node2,
                        node1, node2, equation);
            } else if (tc2_ptr == NULL) {
                xlen = strlen(title_tok) + strlen(node1) + strlen(node2) +
                    strlen(node1) + strlen(node2) + strlen(equation)  +
                    28 - 6*2 + 1 + 21 + 13;
                xline = TMALLOC(char, xlen);
                sprintf(xline, "b%s %s %s i = v(%s, %s)/(%s) tc1=%15.8e reciproctc=1", title_tok, node1, node2,
                        node1, node2, equation, tc1);
            } else {
                xlen = strlen(title_tok) + strlen(node1) + strlen(node2) +
                    strlen(node1) + strlen(node2) + strlen(equation)  +
                    28 - 6*2 + 1 + 21 + 21 + 13;
                xline = TMALLOC(char, xlen);
                sprintf(xline, "b%s %s %s i = v(%s, %s)/(%s) tc1=%15.8e tc2=%15.8e reciproctc=1", title_tok, node1, node2,
                        node1, node2, equation, tc1, tc2);
            }
            tc1_ptr = NULL;
            tc2_ptr = NULL;
            new_line = xx_new_line(card->li_next, xline, 0, 0);

            // comment out current old R line
            *(card->li_line)   = '*';
            // insert new B source line immediately after current line
            card->li_next     = new_line;
            // point 'card' pointer to the new line
            card              = new_line;
            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }
        /* Cxxx n1 n2 C = {equation} or Cxxx n1 n2 {equation}
           -->
           Exxx  n-aux 0  n1 n2  1
           Cxxx  n-aux 0         1
           Bxxx  n2 n1  I = i(Exxx) * equation
        */
        else if (*curr_line == 'c') {
            cut_line = curr_line;
            title_tok = gettok(&cut_line);
            node1 =  gettok(&cut_line);
            node2 =  gettok(&cut_line);
            /* check only after skipping Cname and nodes, either may contain time (e.g. Ctime)*/
            if ((!strstr(cut_line, "v(")) &&  (!strstr(cut_line, "i(")) &&
                (!strstr(cut_line, "temper")) &&  (!strstr(cut_line, "hertz")) &&
                (!strstr(cut_line, "time")))
            {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }

            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                /* if not, equation may start with a '(' */
                str_ptr = strchr(cut_line, '(');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed C line: %s\n", curr_line);
                    controlled_exit(EXIT_FAILURE);
                }
                equation = gettok_char(&str_ptr, ')', TRUE, TRUE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);
            str_ptr = strstr(cut_line, "tc1");
            if (str_ptr) {
                /* We need to have 'tc1=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc1_ptr = strchr(str_ptr, '=');
                    if (tc1_ptr)
                        tc1 = atof(tc1_ptr+1);
                }
            }
            str_ptr = strstr(cut_line, "tc2");
            if (str_ptr) {
                /* We need to have 'tc2=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc2_ptr = strchr(str_ptr, '=');
                    if (tc2_ptr)
                        tc2 = atof(tc2_ptr+1);
                }
            }
            // Exxx  n-aux 0  n1 n2  1
            xlen = 2*strlen(title_tok) + strlen(node1) + strlen(node2)
                + 21 - 4*2 + 1;
            ckt_array[0] = TMALLOC(char, xlen);
            sprintf(ckt_array[0], "e%s %s_int2 0 %s %s 1",
                    title_tok, title_tok, node1, node2);
            // Cxxx  n-aux 0  1
            xlen = 2*strlen(title_tok)
                + 15 - 2*2 + 1;
            ckt_array[1] = TMALLOC(char, xlen);
            sprintf(ckt_array[1], "c%s %s_int2 0 1", title_tok, title_tok);
            // Bxxx  n2 n1  I = i(Exxx) * equation
            if ((tc1_ptr == NULL) && (tc2_ptr == NULL)) {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 27 - 2*5 + 1;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s i = i(e%s) * (%s)",
                        title_tok, node2, node1, title_tok, equation);
            } else if (tc2_ptr == NULL) {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 27 - 2*5 + 1 + 21 + 13;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s i = i(e%s) * (%s) tc1=%15.8e reciproctc=1",
                        title_tok, node2, node1, title_tok, equation, tc1);
            } else {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 27 - 2*5 + 1 + 21 + 21 + 13;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s i = i(e%s) * (%s) tc1=%15.8e tc2=%15.8e reciproctc=1",
                        title_tok, node2, node1, title_tok, equation, tc1, tc2);
            }
            tc1_ptr = NULL;
            tc2_ptr = NULL;
            // insert new B source line immediately after current line
            for (i = 0; i < 3; i++) {
                struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                if (param_end)
                    param_end->li_next = x;
                else
                    param_beg = x;

                param_end = x;
            }
            // comment out current variable capacitor line
            *(card->li_line)   = '*';
            // insert new param lines immediately after current line
            param_end->li_next = card->li_next;
            card->li_next      = param_beg;
            // point 'card' pointer to last in scalar list
            card               = param_end;

            param_beg = param_end = NULL;
            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }

        /* Lxxx n1 n2 L = {equation} or Lxxx n1 n2 {equation}
           -->
           Fxxx n-aux 0  Bxxx -1
           Lxxx n-aux 0      1
           Bxxx n1 n2 V = v(n-aux) * equation
        */
        else if (*curr_line == 'l') {
            cut_line = curr_line;
            /* title and nodes */
            title_tok = gettok(&cut_line);
            node1 =  gettok(&cut_line);
            node2 =  gettok(&cut_line);
            if ((!strstr(cut_line, "v(")) &&  (!strstr(cut_line, "i(")) &&
                (!strstr(cut_line, "temper")) &&  (!strstr(cut_line, "hertz")) &&
                (!strstr(cut_line, "time")))
            {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }

            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                /* if not, equation may start with a '(' */
                str_ptr = strchr(cut_line, '(');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed L line: %s\n", curr_line);
                    controlled_exit(EXIT_FAILURE);
                }
                equation = gettok_char(&str_ptr, ')', TRUE, TRUE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);
            str_ptr = strstr(cut_line, "tc1");
            if (str_ptr) {
                /* We need to have 'tc1=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc1_ptr = strchr(str_ptr, '=');
                    if (tc1_ptr)
                        tc1 = atof(tc1_ptr+1);
                }
            }
            str_ptr = strstr(cut_line, "tc2");
            if (str_ptr) {
                /* We need to have 'tc2=something */
                if (str_ptr[3] && (isspace(str_ptr[3]) || (str_ptr[3] == '='))) {
                    tc2_ptr = strchr(str_ptr, '=');
                    if (tc2_ptr)
                        tc2 = atof(tc2_ptr+1);
                }
            }
            // Fxxx  n-aux 0  Bxxx  1
            xlen = 3*strlen(title_tok)
                + 20 - 3*2 + 1;
            ckt_array[0] = TMALLOC(char, xlen);
            sprintf(ckt_array[0], "f%s %s_int2 0 b%s -1",
                    title_tok, title_tok, title_tok);
            // Lxxx  n-aux 0  1
            xlen = 2*strlen(title_tok)
                + 15 - 2*2 + 1;
            ckt_array[1] = TMALLOC(char, xlen);
            sprintf(ckt_array[1], "l%s %s_int2 0 1", title_tok, title_tok);
            // Bxxx  n1 n2  V = v(n-aux) * equation
            if ((tc1_ptr == NULL) && (tc2_ptr == NULL)) {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 31 - 2*5 + 1;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s v = v(%s_int2) * (%s)",
                        title_tok, node1, node2, title_tok, equation);
            } else if (tc2_ptr == NULL) {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 31 - 2*5 + 1 + 21 + 13;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s v = v(%s_int2) * (%s) tc1=%15.8e reciproctc=0",
                        title_tok, node2, node1, title_tok, equation, tc1);
            } else {
                xlen = 2*strlen(title_tok) + strlen(node2) + strlen(node1)
                    + strlen(equation) + 31 - 2*5 + 1 + 21 + 21 + 13;
                ckt_array[2] = TMALLOC(char, xlen);
                sprintf(ckt_array[2], "b%s %s %s v = v(%s_int2) * (%s) tc1=%15.8e tc2=%15.8e reciproctc=0",
                        title_tok, node2, node1, title_tok, equation, tc1, tc2);
            }
            tc1_ptr = NULL;
            tc2_ptr = NULL;
            // insert new B source line immediately after current line
            for (i = 0; i < 3; i++) {
                struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                if (param_end)
                    param_end->li_next = x;
                else
                    param_beg = x;

                param_end = x;
            }
            // comment out current variable inductor line
            *(card->li_line)   = '*';
            // insert new param lines immediately after current line
            param_end->li_next = card->li_next;
            card->li_next      = param_beg;
            // point 'card' pointer to last in scalar list
            card               = param_end;

            param_beg = param_end = NULL;
            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }
        /* .probe -> .save
           .print, .plot, .save, .four,
           An ouput vector may be replaced by the following:
           myoutput=par('expression')
           .meas
           A vector out_variable may be replaced by
           par('expression')
        */
        else if (*curr_line == '.') {
            // replace .probe by .save
            if ((str_ptr = strstr(curr_line, ".probe")) != NULL)
                memcpy(str_ptr, ".save ", 6);

            /* Various formats for measure statement:
             * .MEASURE {DC|AC|TRAN} result WHEN out_variable=val
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result WHEN out_variable=out_variable2
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable WHEN out_variable2=val
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable WHEN out_variable2=out_variable3
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable AT=val
             * + <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result {AVG|MIN|MAX|MIN_AT|MAX_AT|PP|RMS} out_variable
             * + <TD=td> <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result INTEG<RAL> out_variable
             * + <TD=td> <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable AT=val
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable WHEN out_variable2=val
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable WHEN out_variable2=out_variable3
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>

             The user may set any out_variable to par(' expr ').
             We have to replace this by v(pa_xx) and generate a B source line.

             * ----------------------------------------------------------------- */
            if (ciprefix(".meas", curr_line)) {
                if (strstr(curr_line, "par(") == NULL)
                    continue;
                cut_line = curr_line;
                // search for 'par('
                while ((str_ptr = strstr(cut_line, "par(")) != NULL) {
                    if (pai > 99) {
                        fprintf(stderr, "ERROR: More than 99 function calls to par()\n");
                        fprintf(stderr, "  Limited to 99 per input file\n");
                        controlled_exit(EXIT_FAILURE);
                    }

                    // we have ' par({ ... })', the right delimeter is a ' ' or '='
                    if (ciprefix(" par({", (str_ptr-1))) {
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '=') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr-2);
                        cut_line = str_ptr;
                        // generate node
                        out_ptr = TMALLOC(char, 6);
                        sprintf(out_ptr, "pa_%02d", (int)pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        xlen = 2*strlen(out_ptr) + strlen(exp_ptr)+ 15 - 2*3 + 1;
                        ckt_array[pai] = TMALLOC(char, xlen);
                        sprintf(ckt_array[pai], "b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        xlen = strlen(out_ptr) + 4;
                        del_ptr = copy_ptr = TMALLOC(char, xlen);
                        sprintf(copy_ptr, "v(%s)", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 7;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // or we have '={par({ ... })}', the right delimeter is a ' '
                    else if (ciprefix("={par({", (str_ptr-2))) {
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr-3);
                        // generate node
                        out_ptr = TMALLOC(char, 6);
                        sprintf(out_ptr, "pa_%02d", (int)pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        xlen = 2*strlen(out_ptr) + strlen(exp_ptr)+ 15 - 2*3 + 1;
                        ckt_array[pai] = TMALLOC(char, xlen);
                        sprintf(ckt_array[pai], "b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        xlen = strlen(out_ptr) + 4;
                        del_ptr = copy_ptr = TMALLOC(char, xlen);
                        sprintf(copy_ptr, "v(%s)", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 9;
                        // skip '='
                        cut_line++;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    } else {
                        // nothing to replace
                        cut_line = str_ptr + 1;
                        continue;
                    }

                } // while 'par'
                // no replacement done, go to next line
                if (pai == paui)
                    continue;
                // remove white spaces
                card->li_line = inp_remove_ws(curr_line);
                // insert new B source line immediately after current line
                for (ii = paui; ii < pai; ii++) {
                    struct line *x = xx_new_line(NULL, ckt_array[ii], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }

                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                paui = pai;
            } else if ((ciprefix(".save", curr_line)) ||
                       (ciprefix(".four", curr_line)) ||
                       (ciprefix(".print", curr_line)) ||
                       (ciprefix(".plot", curr_line)))
            {
                if (strstr(curr_line, "par(") == NULL)
                    continue;
                cut_line = curr_line;
                // search for 'par('
                while ((str_ptr = strstr(cut_line, "par(")) != NULL) {
                    if (pai > 99) {
                        fprintf(stderr, "ERROR: More than 99 function calls to par()\n");
                        fprintf(stderr, "  Limited to 99 per input file\n");
                        controlled_exit(EXIT_FAILURE);
                    }

                    // we have ' par({ ... })'
                    if (ciprefix(" par({", (str_ptr-1))) {

                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr-2);
                        cut_line = str_ptr;
                        // generate node
                        out_ptr = TMALLOC(char, 6);
                        sprintf(out_ptr, "pa_%02d", (int)pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        xlen = 2*strlen(out_ptr) + strlen(exp_ptr)+ 15 - 2*3 + 1;
                        ckt_array[pai] = TMALLOC(char, xlen);
                        sprintf(ckt_array[pai], "b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        xlen = strlen(out_ptr) + 1;
                        del_ptr = copy_ptr = TMALLOC(char, xlen);
                        sprintf(copy_ptr, "%s", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 7;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // or we have '={par({ ... })}'
                    else if (ciprefix("={par({", (str_ptr-2))) {

                        // find myoutput
                        beg_ptr = end_ptr = str_ptr - 2;
                        while (*beg_ptr != ' ')
                            beg_ptr--;
                        out_ptr = copy_substring(beg_ptr + 1, end_ptr);
                        cut_line = beg_ptr + 1;
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr-3);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        xlen = 2*strlen(out_ptr) + strlen(exp_ptr)+ 15 - 2*3 + 1;
                        ckt_array[pai] = TMALLOC(char, xlen);
                        sprintf(ckt_array[pai], "b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        xlen = strlen(out_ptr) + 1;
                        del_ptr = copy_ptr = TMALLOC(char, xlen);
                        sprintf(copy_ptr, "%s", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(out_ptr) + strlen(exp_ptr) + 10;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // nothing to replace
                    else
                        cut_line = str_ptr + 1;
                } // while 'par('
                // no replacement done, go to next line
                if (pai == paui)
                    continue;
                // remove white spaces
                card->li_line = inp_remove_ws(curr_line);
                // insert new B source line immediately after current line
                for (ii = paui; ii < pai; ii++) {
                    struct line *x = xx_new_line(NULL, ckt_array[ii], 0, 0);

                    if (param_end)
                        param_end->li_next = x;
                    else
                        param_beg = x;

                    param_end = x;
                }
                // comment out current variable capacitor line
                // *(ckt_array[0])   = '*';
                // insert new param lines immediately after current line
                param_end->li_next = card->li_next;
                card->li_next      = param_beg;
                // point 'card' pointer to last in scalar list
                card               = param_end;

                param_beg = param_end = NULL;
                paui = pai;
                // continue;
            } // if .print etc.
        } // if ('.')
    }
}


/* replace a token (length 4 char) in string by spaces, if it is found
   at the correct position and the total number of tokens is o.k. */

static void
replace_token(char *string, char *token, int wherereplace, int total)
{
    int count = 0, i;
    char *actstring = string;

    /* token to be replaced not in string */
    if (strstr(string, token) == NULL)
        return;

    /* get total number of tokens */
    while (*actstring) {
        txfree(gettok(&actstring));
        count++;
    }
    /* If total number of tokens correct */
    if (count == total) {
        actstring = string;
        for (i = 1; i < wherereplace; i++)
            txfree(gettok(&actstring));
        /* If token to be replaced at right position */
        if (ciprefix(token, actstring)) {
            actstring[0] = ' ';
            actstring[1] = ' ';
            actstring[2] = ' ';
            actstring[3] = ' ';
        }
    }
}


/* lines for B sources: no parsing in numparam code, just replacement of parameters.
   Parsing done in B source parser.
   To achive this, do the following:
   Remove all '{' and '}' --> no parsing of equations in numparam
   Place '{' and '}' directly around all potential parameters,
   but skip function names like exp (search for 'exp(' to detect fcn name),
   functions containing nodes like v(node), v(node1, node2), i(branch)
   and other keywords like TEMPER. --> Only parameter replacement in numparam
*/

static void
inp_bsource_compat(struct line *card)
{
    char *equal_ptr, *str_ptr, *tmp_char, *new_str, *final_str;
    char actchar;
    struct line *new_line;
    wordlist *wl = NULL, *wlist = NULL;
    char buf[512];
    size_t i, xlen, ustate = 0;
    int skip_control = 0;
    int error1;

    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (*curr_line == 'b') {
            /* remove white spaces of everything inside {}*/
            card->li_line = inp_remove_ws(card->li_line);
            curr_line = card->li_line;
            /* store starting point for later parsing, beginning of {expression} */
            equal_ptr = strchr(curr_line, '=');
            /* check for errors */
            if (equal_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed B line: %s\n", curr_line);
                controlled_exit(EXIT_FAILURE);
            }
            /* find the m={m} token and remove it */
            if ((str_ptr = strstr(curr_line, "m={m}")) != NULL)
                memcpy(str_ptr, "     ", 5);
            /* scan the line and remove all '{' and '}' */
            str_ptr = curr_line;
            while (*str_ptr) {
                if ((*str_ptr == '{') || (*str_ptr == '}'))
                    *str_ptr = ' ';
                str_ptr++;
            }
            /* scan the expression */
            str_ptr = equal_ptr + 1;
            while (*str_ptr != '\0') {
                str_ptr = skip_ws(str_ptr);
                if (*str_ptr == '\0')
                    break;
                actchar = *str_ptr;
                wl_append_word(&wlist, &wl, NULL);
                if ((actchar == ',') || (actchar == '(') || (actchar == ')') ||
                    (actchar == '*') || (actchar == '/') || (actchar == '^') ||
                    (actchar == '+') || (actchar == '?') || (actchar == ':'))
                {
                    if ((actchar == '*') && (str_ptr[1] == '*')) {
                        actchar = '^';
                        str_ptr++;
                    }
                    buf[0] = actchar;
                    buf[1] = '\0';
                    wl->wl_word = copy(buf);
                    str_ptr++;
                    if (actchar == ')')
                        ustate = 0;
                    else
                        ustate = 1; /* we have an operator */
                } else if ((actchar == '>') || (actchar == '<') ||
                           (actchar == '!') || (actchar == '='))
                {
                    /* >=, <=, !=, ==, <>, ... */
                    char *beg = str_ptr++;
                    if ((*str_ptr == '=') || (*str_ptr == '<') || (*str_ptr == '>'))
                        str_ptr++;
                    wl->wl_word = copy_substring(beg, str_ptr);
                    ustate = 1; /* we have an operator */
                } else if ((actchar == '|') || (actchar == '&')) {
                    char *beg = str_ptr++;
                    if ((*str_ptr == '|') || (*str_ptr == '&'))
                        str_ptr++;
                    wl->wl_word = copy_substring(beg, str_ptr);
                    ustate = 1; /* we have an operator */
                } else if ((actchar == '-') && (ustate == 0)) {
                    buf[0] = actchar;
                    buf[1] = '\0';
                    wl->wl_word = copy(buf);
                    str_ptr++;
                    ustate = 1; /* we have an operator */
                } else if ((actchar == '-') && (ustate == 1)) {
                    wl->wl_word = copy("");
                    str_ptr++;
                    ustate = 2; /* place a '-' in front of token */
                } else if (isalpha(actchar)) {
                    /* unary -, change sign */
                    if (ustate == 2) {
                        i = 1;
                        buf[0] = '-';
                    } else {
                        i = 0;
                    }

                    if (((actchar == 'v') || (actchar == 'i')) && (str_ptr[1] == '(')) {
                        while (*str_ptr != ')') {
                            buf[i] = *str_ptr;
                            i++;
                            str_ptr++;
                        }
                        buf[i] = *str_ptr;
                        buf[i+1] = '\0';
                        wl->wl_word = copy(buf);
                        str_ptr++;
                    } else {
                        while (isalnum(*str_ptr) ||
                               (*str_ptr == '!') || (*str_ptr == '#') ||
                               (*str_ptr == '$') || (*str_ptr == '%') ||
                               (*str_ptr == '_') || (*str_ptr == '[') ||
                               (*str_ptr == ']'))
                        {
                            buf[i] = *str_ptr;
                            i++;
                            str_ptr++;
                        }
                        buf[i] = '\0';
                        /* no parens {} around time, hertz, temper, the constants
                           pi and e which are defined in inpptree.c, around pwl and temp. coeffs */
                        if ((*str_ptr == '(') ||
                            cieq(buf, "hertz") || cieq(buf, "temper") ||
                            cieq(buf, "time") || cieq(buf, "pi") ||
                            cieq(buf, "e") || cieq(buf, "pwl"))
                        {
                            /* special handling of pwl lines:
                               Put braces around tokens and around expressions, use ','
                               as separator like:
                               pwl(i(Vin), {x0-1},{y0},
                               {x0},{y0},{x1},{y1}, {x2},{y2},{x3},{y3},
                               {x3+1},{y3})
                            */
                            /*
                             * if (cieq(buf, "pwl")) {
                             *     // go past i(Vin)
                             *     i = 3;
                             *     while (*str_ptr != ')') {
                             *         buf[i] = *str_ptr;
                             *         i++;
                             *         str_ptr++;
                             *     }
                             *     buf[i] = *str_ptr;
                             *     i++;
                             *     str_ptr++;
                             *     // find first ','
                             *     while (*str_ptr != ',') {
                             *         buf[i] = *str_ptr;
                             *         i++;
                             *         str_ptr++;
                             *     }
                             *     buf[i] = *str_ptr;
                             *     i++;
                             *     buf[i] = '{';
                             *     i++;
                             *     str_ptr++;
                             *     while (*str_ptr != ')') {
                             *         if (*str_ptr == ',') {
                             *             buf[i] = '}';
                             *             i++;
                             *             buf[i] = ',';
                             *             i++;
                             *             buf[i] = '{';
                             *             i++;
                             *             str_ptr++;
                             *         }
                             *         else {
                             *             buf[i] = *str_ptr;
                             *             i++;
                             *             str_ptr++;
                             *         }
                             *     }
                             *     buf[i] = '}';
                             *     i++;
                             *     buf[i] = *str_ptr;
                             *     i++;
                             *     buf[i] = '\0';
                             *     str_ptr++;
                             * }
                             */
                            wl->wl_word = copy(buf);

                        } else if (cieq(buf, "tc1") || cieq(buf, "tc2") ||
                                   cieq(buf, "reciproctc"))
                        {

                            str_ptr = skip_ws(str_ptr);
                            /* no {} around tc1 = or tc2 = , these are temp coeffs. */
                            if (str_ptr[0] == '='  &&  str_ptr[1] != '=') {
                                buf[i++] = '=';
                                buf[i] = '\0';
                                str_ptr++;
                                wl->wl_word = copy(buf);
                            } else {
                                xlen = strlen(buf);
                                tmp_char = TMALLOC(char, xlen + 3);
                                sprintf(tmp_char, "{%s}", buf);
                                wl->wl_word = tmp_char;
                            }

                        } else {
                            /* {} around all other tokens */
                            xlen = strlen(buf);
                            tmp_char = TMALLOC(char, xlen + 3);
                            sprintf(tmp_char, "{%s}", buf);
                            wl->wl_word = tmp_char;
                        }
                    }
                    ustate = 0; /* we have a number */
                } else if (isdigit(actchar) || (actchar == '.')) { /* allow .5 format too */
                    /* allow 100p, 5MEG etc. */
                    double dvalue = INPevaluate(&str_ptr, &error1, 0);
                    char   cvalue[19];
                    /* unary -, change sign */
                    if (ustate == 2)
                        dvalue *= -1;
                    sprintf(cvalue, "%18.10e", dvalue);
                    wl->wl_word = copy(cvalue);
                    ustate = 0; /* we have a number */
                    /* skip the `unit', FIXME INPevaluate() should do this */
                    while (isalpha(*str_ptr))
                        str_ptr++;
                } else { /* strange char */
                    printf("Preparing B line for numparam\nWhat is this?\n%s\n", str_ptr);
                    buf[0] = *str_ptr;
                    buf[1] = '\0';
                    wl->wl_word = copy(buf);
                    str_ptr++;
                }
            }

            new_str = wl_flatten(wlist);
            wl_free(wlist);
            wlist = NULL;
            wl = NULL;

            tmp_char = copy(curr_line);
            equal_ptr = strchr(tmp_char, '=');
            if (str_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed B line:\n  %s\n", curr_line);
                controlled_exit(EXIT_FAILURE);
            }
            /* cut the tmp_char after the equal sign */
            equal_ptr[1] = '\0';
            xlen = strlen(tmp_char) + strlen(new_str) + 2;
            final_str = TMALLOC(char, xlen);
            sprintf(final_str, "%s %s", tmp_char, new_str);

            /* Copy old line numbers into new B source line */
            new_line = xx_new_line(card->li_next, final_str, card->li_linenum, card->li_linenum_orig);

            // comment out current line (old B source line)
            *(card->li_line)   = '*';
            // insert new B source line immediately after current line
            card->li_next     = new_line;
            // point 'card' pointer to the new line
            card              = new_line;

            tfree(new_str);
            tfree(tmp_char);
        } /* end of if 'b' */
    } /* end of for loop */
}


/* Find all expressions containing the keyword 'temper',
 * except for B lines and some other exclusions. Prepare
 * these expressions by calling inp_modify_exp() and return
 * a modified card->li_line
 */

static void
inp_temper_compat(struct line *card)
{
    int skip_control = 0;
    char *beg_str, *end_str, *beg_tstr, *end_tstr, *exp_str;
    char actchar;

    for (; card; card = card->li_next) {

        char *new_str = NULL;
        char *curr_line = card->li_line;

        if (curr_line == NULL)
            continue;
        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }
        /* exclude some elements */
        if ((*curr_line == '*') || (*curr_line == 'v') || (*curr_line == 'b') || (*curr_line == 'i') ||
            (*curr_line == 'e') || (*curr_line == 'g') || (*curr_line == 'f') || (*curr_line == 'h'))
            continue;
        /* exclude all dot commands except .model */
        if ((*curr_line == '.') && (!prefix(".model", curr_line)))
            continue;
        /* exclude lines not containing 'temper' */
        if (strstr(curr_line, "temper") == NULL)
            continue;
        /* now start processing of the remaining lines containing 'temper' */
        /* remove white spaces of everything inside {}*/
        card->li_line = inp_remove_ws(card->li_line);
        curr_line = card->li_line;
        /* now check if 'temper' is a token or just a substring of another string, e.g. mytempers */
        /* we may have multiple temper and mytempers in multiple expressions in a line */
        beg_str = beg_tstr = curr_line;
        while ((beg_tstr = strstr(beg_tstr, "temper")) != NULL) {
            actchar = *(beg_tstr - 1);
            if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '{')) {
                beg_tstr++;
                continue;
            }
            actchar = *(beg_tstr + 6);
            if (!isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',') && !(actchar == '}')) {
                beg_tstr++;
                continue;
            }
            /* we have found a true 'temper' */
            /* set the global variable */
            expr_w_temper = TRUE;
            /* find the expression: first go back to the opening '{',
               then find the closing '}' */
            while ((*beg_tstr) != '{')
                beg_tstr--;
            end_str = end_tstr = beg_tstr;
            exp_str = gettok_char(&end_tstr, '}', TRUE, TRUE);
            /* modify the expression string */
            exp_str = inp_modify_exp(exp_str);
            /* add the intermediate string between previous and next expression to the new line */
            new_str = INPstrCat(new_str, copy_substring(beg_str, end_str), " ");
            /* add the modified expression string to the new line */
            new_str = INPstrCat(new_str, exp_str, " ");
            new_str = INPstrCat(new_str, copy(" "), " ");
            /* move on to the next intermediate string */
            beg_str = beg_tstr = end_tstr;
        }
        if (*beg_str)
            new_str = INPstrCat(new_str, copy(beg_str), " ");
        tfree(card->li_line);
        card->li_line = inp_remove_ws(new_str);
    }
}


/* lines containing expressions with keyword 'temper':
 * no parsing in numparam code, just replacement of parameters.
 * Parsing done with B source parser in function inp_parse_temper
 * in inp.c. Evaluation is the done with fcn inp_evaluate_temper
 * from inp.c, taking the actual temperature into account.
 * To achive this, do the following here:
 * Remove all '{' and '}' --> no parsing of equations in numparam
 * Place '{' and '}' directly around all potential parameters,
 * but skip function names like exp (search for 'exp(' to detect fcn name),
 * functions containing nodes like v(node), v(node1, node2), i(branch)
 * and other keywords like TEMPER. --> Only parameter replacement in numparam
 */

static char *
inp_modify_exp(char* expr)
{
    char * str_ptr, *tmp_char, *new_str;
    char actchar;
    wordlist *wl = NULL, *wlist = NULL;
    char buf[512];
    size_t i, xlen, ustate = 0;
    int error1;

    /* scan the expression and remove all '{' and '}' */
    str_ptr = expr;
    while (*str_ptr) {
        if ((*str_ptr == '{') || (*str_ptr == '}'))
            *str_ptr = ' ';
        str_ptr++;
    }
    /* scan the expression */
    str_ptr = expr;
    while (*str_ptr != '\0') {
        str_ptr = skip_ws(str_ptr);
        if (*str_ptr == '\0')
            break;
        actchar = *str_ptr;
        wl_append_word(&wlist, &wl, NULL);
        if ((actchar == ',') || (actchar == '(') || (actchar == ')') ||
            (actchar == '*') || (actchar == '/') || (actchar == '^') ||
            (actchar == '+') || (actchar == '?') || (actchar == ':'))
        {
            if ((actchar == '*') && (str_ptr[1] == '*')) {
                actchar = '^';
                str_ptr++;
            }
            buf[0] = actchar;
            buf[1] = '\0';
            wl->wl_word = copy(buf);
            str_ptr++;
            if (actchar == ')')
                ustate = 0;
            else
                ustate = 1; /* we have an operator */
        } else if ((actchar == '>') || (actchar == '<') ||
                   (actchar == '!') || (actchar == '='))
        {
            /* >=, <=, !=, ==, <>, ... */
            char *beg = str_ptr++;
            if ((*str_ptr == '=') || (*str_ptr == '<') || (*str_ptr == '>'))
                str_ptr++;
            wl->wl_word = copy_substring(beg, str_ptr);
            ustate = 1; /* we have an operator */
        } else if ((actchar == '|') || (actchar == '&')) {
            char *beg = str_ptr++;
            if ((*str_ptr == '|') || (*str_ptr == '&'))
                str_ptr++;
            wl->wl_word = copy_substring(beg, str_ptr);
            ustate = 1; /* we have an operator */
        } else if ((actchar == '-') && (ustate == 0)) {
            buf[0] = actchar;
            buf[1] = '\0';
            wl->wl_word = copy(buf);
            str_ptr++;
            ustate = 1; /* we have an operator */
        } else if ((actchar == '-') && (ustate == 1)) {
            wl->wl_word = copy("");
            str_ptr++;
            ustate = 2; /* place a '-' in front of token */
        } else if (isalpha(actchar)) {
            /* unary -, change sign */
            if (ustate == 2) {
                i = 1;
                buf[0] = '-';
            } else {
                i = 0;
            }

            if (((actchar == 'v') || (actchar == 'i')) && (str_ptr[1] == '(')) {
                while (*str_ptr != ')') {
                    buf[i] = *str_ptr;
                    i++;
                    str_ptr++;
                }
                buf[i] = *str_ptr;
                buf[i+1] = '\0';
                wl->wl_word = copy(buf);
                str_ptr++;
            } else {
                while (isalnum(*str_ptr) ||
                       (*str_ptr == '!') || (*str_ptr == '#') ||
                       (*str_ptr == '$') || (*str_ptr == '%') ||
                       (*str_ptr == '_') || (*str_ptr == '[') ||
                       (*str_ptr == ']'))
                {
                    buf[i] = *str_ptr;
                    i++;
                    str_ptr++;
                }
                buf[i] = '\0';
                /* no parens {} around time, hertz, temper, the constants
                   pi and e which are defined in inpptree.c, around pwl and temp. coeffs */
                if ((*str_ptr == '(') ||
                    cieq(buf, "hertz") || cieq(buf, "temper") ||
                    cieq(buf, "time") || cieq(buf, "pi") ||
                    cieq(buf, "e") || cieq(buf, "pwl"))
                {
                    wl->wl_word = copy(buf);

                } else if (cieq(buf, "tc1") || cieq(buf, "tc2") ||
                           cieq(buf, "reciproctc"))
                {
                    str_ptr = skip_ws(str_ptr);
                    /* no {} around tc1 = or tc2 = , these are temp coeffs. */
                    if (str_ptr[0] == '='  &&  str_ptr[1] != '=') {
                        buf[i++] = '=';
                        buf[i] = '\0';
                        str_ptr++;
                        wl->wl_word = copy(buf);
                    } else {
                        xlen = strlen(buf);
                        tmp_char = TMALLOC(char, xlen + 3);
                        sprintf(tmp_char, "{%s}", buf);
                        wl->wl_word = tmp_char;
                    }

                } else {
                    /* {} around all other tokens */
                    xlen = strlen(buf);
                    tmp_char = TMALLOC(char, xlen + 3);
                    sprintf(tmp_char, "{%s}", buf);
                    wl->wl_word = tmp_char;
                }
            }
            ustate = 0; /* we have a number */
        } else if (isdigit(actchar) || (actchar == '.')) { /* allow .5 format too */
            /* allow 100p, 5MEG etc. */
            double dvalue = INPevaluate(&str_ptr, &error1, 0);
            char   cvalue[19];
            /* unary -, change sign */
            if (ustate == 2)
                dvalue *= -1;
            sprintf(cvalue, "%18.10e", dvalue);
            wl->wl_word = copy(cvalue);
            ustate = 0; /* we have a number */
            /* skip the `unit', FIXME INPevaluate() should do this */
            while (isalpha(*str_ptr))
                str_ptr++;
        } else { /* strange char */
            printf("Preparing expression for numparam\nWhat is this?\n%s\n", str_ptr);
            buf[0] = *str_ptr;
            buf[1] = '\0';
            wl->wl_word = copy(buf);
            str_ptr++;
        }
    }

    new_str = wl_flatten(wlist);
    wl_free(wlist);
    wlist = NULL;
    wl = NULL;
    tfree(expr);
    return(new_str);
}


/*
 * destructively fetch a token from the input string
 *   token is either quoted, or a plain nonwhitespace sequence
 * function will return the place from where to continue
 */

static char *
get_quoted_token(char *string, char **token)
{
    char *s = skip_ws(string);

    if (!*s)            /* nothing found */
        return string;

    if (isquote(*s)) {

        char *t = ++s;

        while (*t && !isquote(*t))
            t++;

        if (!*t) {        /* teriminator quote not found */
            *token = NULL;
            return string;
        }

        *t++ = '\0';

        *token = s;
        return t;

    } else {

        char *t = skip_non_ws(s);

        if (t == s) {     /* nothing found */
            *token = NULL;
            return string;
        }

        if (*t)
            *t++ = '\0';

        *token = s;
        return t;
    }
}


/* Option RSERIES=rval
 * Lxxx n1 n2 Lval
 * -->
 * Lxxx n1 n2_intern__ Lval
 * RLxxx_n2_intern__ n2_intern__ n2 rval
 */

static void
inp_add_series_resistor(struct line *deck)
{
    size_t skip_control = 0, xlen, i;
    bool has_rseries = FALSE;
    struct line *card;
    char *tmp_p, *title_tok, *node1, *node2, *rval = NULL;
    char *ckt_array[10];
    struct line  *param_end = NULL, *param_beg = NULL;

    for (card = deck; card; card = card->li_next) {
        char *curr_line = card->li_line;
        if (*curr_line == '*')
            continue;
        if (strstr(curr_line, "option") && strstr(curr_line, "rseries"))
            has_rseries = TRUE;
        else
            continue;
        tmp_p = strstr(curr_line, "rseries");
        tmp_p += 7;
        /* default to "1e-3" if no value given */
        if (ciprefix("=", tmp_p)) {
            tmp_p = strchr(tmp_p, '=') + 1;
            rval = gettok(&tmp_p);
        }
        else
            rval = copy("1e-3");
    }

    if (!has_rseries || !rval)
        return;

    fprintf(stdout,
            "\nOption rseries given: \n"
            "resistor %s Ohms added in series to each inductor L\n\n", rval);

    for (card = deck; card; card = card->li_next) {
        char *cut_line;
        char *curr_line = cut_line = card->li_line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (ciprefix("l", curr_line)) {
            title_tok = gettok(&cut_line);
            node1 =  gettok(&cut_line);
            node2 =  gettok(&cut_line);
            /* new L line */
            xlen = strlen(curr_line) + 10;
            ckt_array[0] = TMALLOC(char, xlen);
            sprintf(ckt_array[0], "%s %s %s_intern__ %s",
                    title_tok, node1, node2, cut_line);
            /* new R line */
            xlen = strlen(curr_line) + 19;
            ckt_array[1] = TMALLOC(char, xlen);
            sprintf(ckt_array[1], "R%s_intern__ %s_intern__ %s %s",
                    title_tok, node2, node2, rval);
            /* assemble new L and R lines */
            for (i = 0; i < 2; i++) {
                struct line *x = xx_new_line(NULL, ckt_array[i], 0, 0);

                if (param_end)
                    param_end->li_next = x;
                else
                    param_beg = x;

                param_end = x;
            }
            // comment out current L line
            *(card->li_line)   = '*';
            // insert new new L and R lines immediately after current line
            param_end->li_next = card->li_next;
            card->li_next      = param_beg;
            // point 'card' pointer to last in scalar list
            card               = param_end;
            param_beg = param_end = NULL;
            tfree(title_tok);
            tfree(node1);
            tfree(node2);
        }
    }

    tfree(rval);
}


/*
 * rewrite
 *   .subckt node1 node2 node3 name params: l={x} w={y}
 * to
 *   .subckt node1 node2 node3 name
 *   .param l={x} w={y}
 */

static void
subckt_params_to_param(struct line *card)
{
    for (; card; card = card->li_next) {
        char *curr_line = card->li_line;
        if (ciprefix(".subckt", curr_line)) {
            char *cut_line, *new_line;
            cut_line = strstr(curr_line, "params:");
            if (!cut_line)
                continue;
            /* new_line starts with "params: " */
            new_line = copy(cut_line);
            /* replace "params:" by ".param " */
            memcpy(new_line, ".param ", 7);
            /* card->li_line ends with subcircuit name */
            cut_line[-1] = '\0';
            /* insert new_line after card->li_line */
            card->li_next = xx_new_line(card->li_next, new_line,
                                        card->li_linenum + 1, 0);
        }
    }
}


/* If XSPICE option is not selected, run this function to alert and exit
   if the 'poly' option is found in e, g, f, or h controlled sources. */

#ifndef XSPICE

static void
inp_poly_err(struct line *card)
{
    size_t skip_control = 0;

    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        if (*curr_line == '*')
            continue;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        /* get the fourth token in a controlled source line and exit,
           if it is 'poly' */
        if ((ciprefix("e", curr_line)) || (ciprefix("g", curr_line)) ||
            (ciprefix("f", curr_line)) || (ciprefix("h", curr_line)))
        {
            txfree(gettok(&curr_line));
            txfree(gettok(&curr_line));
            txfree(gettok(&curr_line));
            if (ciprefix("poly", curr_line)) {
                fprintf(stderr,
                        "\nError: XSPICE is required to run the 'poly' option in line %d\n",
                        card->li_linenum_orig);
                fprintf(stderr, "  %s\n", card->li_line);
                fprintf(stderr, "\nSee manual chapt. 31 for installation instructions\n");
                controlled_exit(EXIT_BAD);
            }
        }
    }
}

#endif


void
tprint(struct line *t)
{
    struct line *tmp;

    /*debug: print into file*/
    FILE *fd = fopen("tprint-out.txt", "w");
    for (tmp = t; tmp; tmp = tmp->li_next)
        if (*(tmp->li_line) != '*')
            fprintf(fd, "%6d  %6d  %s\n", tmp->li_linenum_orig, tmp->li_linenum, tmp->li_line);
    fprintf(fd, "\n*********************************************************************************\n");
    fprintf(fd, "*********************************************************************************\n");
    fprintf(fd, "*********************************************************************************\n\n");
    for (tmp = t; tmp; tmp = tmp->li_next)
        fprintf(fd, "%6d  %6d  %s\n", tmp->li_linenum_orig, tmp->li_linenum, tmp->li_line);
    fprintf(fd, "\n*********************************************************************************\n");
    fprintf(fd, "*********************************************************************************\n");
    fprintf(fd, "*********************************************************************************\n\n");
    for (tmp = t; tmp; tmp = tmp->li_next)
        if (*(tmp->li_line) != '*')
            fprintf(fd, "%s\n",tmp->li_line);
    fclose(fd);
}


/* prepare .if and .elseif for numparam
   .if(expression) --> .if{expression} */

static void
inp_dot_if(struct line *card)
{
    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        if (*curr_line == '*')
            continue;

        if (ciprefix(".if", curr_line) || ciprefix(".elseif", curr_line)) {
            char *firstbr = strchr(curr_line, '(');
            char *lastbr = strrchr(curr_line, ')');
            if ((!firstbr) || (!lastbr)) {
                fprintf(cp_err, "Error in netlist line %d\n", card->li_linenum_orig);
                fprintf(cp_err, "   Bad syntax: %s\n\n", curr_line);
                controlled_exit(EXIT_BAD);
            }
            *firstbr = '{';
            *lastbr = '}';
        }
    }
}


/* Convert .param lines containing keyword 'temper' into .func lines:
 * .param xxx1 = 'temper + 25'  --->  .func xxx1() 'temper + 25'
 * Add info about the functions (name, subcircuit depth, number of
 * subckt) to linked list new_func.
 * Then scan new_func, for each xxx1 scan all lines of deck,
 * find all xxx1 and convert them to a function:
 * xxx1   --->  xxx1()
 * If this happens to be in another .param line, convert it to .func,
 * add info to end of new_func and continue scanning.
 */

static void
inp_fix_temper_in_param(struct line *deck)
{
    int skip_control = 0, subckt_depth = 0, j, *sub_count;
    char *beg_pstr, *beg_tstr, *end_tstr, *funcbody, *funcname;
    char actchar;
    struct func_temper *new_func = NULL, *beg_func;
    struct line *card;

    sub_count = TMALLOC(int, 16);
    for(j = 0; j < 16; j++)
        sub_count[j] = 0;

    /* first pass: determine all .param with temper inside and replace by .func
       .param xxx1 = 'temper + 25'
       will become
       .func xxx1() 'temper + 25'
    */
    card = deck;
    for (; card; card = card->li_next) {

        char *curr_line = card->li_line;

        if (*curr_line == '*')
            continue;

        /* determine nested depths of subcircuits */
        if (ciprefix(".subckt", curr_line)) {
            subckt_depth ++;
            sub_count[subckt_depth]++;
            continue;
        } else if (ciprefix(".ends", curr_line)) {
            subckt_depth --;
            continue;
        }

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control ++;
            continue;
        } else if (ciprefix(".endc", curr_line)) {
            skip_control --;
            continue;
        } else if (skip_control > 0) {
            continue;
        }

        if (ciprefix(".param", curr_line)) {
            /* check if we have a true 'temper' */
            beg_tstr = curr_line;
            while ((end_tstr = beg_tstr = strstr(beg_tstr, "temper")) != NULL) {
                actchar = *(beg_tstr - 1);
                if (!(actchar == '{') && !isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',')) {
                    beg_tstr++;
                    continue;
                }
                actchar = *(beg_tstr + 6);
                if (actchar == '=') {
                    fprintf(stderr, "Error: you cannot assign a value to TEMPER\n");
                    fprintf(stderr, "  Line no. %d, %s\n", card->li_linenum, curr_line);
                    controlled_exit(EXIT_BAD);
                }

                if (!(actchar == '}') && !isspace(actchar) && !is_arith_char(actchar) && !(actchar == ',')) {
                    beg_tstr++;
                    continue;
                }
                /* we have found a true 'temper', so start conversion */
                /* find function name and function body: We may have multiple
                   params in a linie!
                */
                while ((*beg_tstr) != '=')
                    beg_tstr--;
                beg_pstr = beg_tstr;
                /* go back over param name */
                while(isspace(*beg_pstr))
                    beg_pstr--;
                while(!isspace(*beg_pstr))
                    beg_pstr--;
                /* get function name from parameter name */
                funcname = copy_substring(beg_pstr + 1, beg_tstr);
                /* find end of function body */
                while (((*end_tstr) != '\0') && ((*end_tstr) != '='))
                    end_tstr++;
                /* go back over next param name */
                if (*end_tstr == '=') {
                    end_tstr--;
                    while(isspace(*end_tstr))
                        end_tstr--;
                    while(!isspace(*end_tstr))
                        end_tstr--;
                }

                funcbody = copy_substring(beg_tstr + 1, end_tstr);
                inp_new_func(funcname, funcbody, card, &new_func, sub_count, subckt_depth);
                tfree(funcbody);

                beg_tstr = end_tstr;
            }
        }
    }

    /* second pass */
    /* for each .func entry in new_func start the insertion operation:
       search each line from the deck, which has the suitable subcircuit nesting data,
       for tokens xxx equalling the funcname, replace xxx by xxx(). After insertion,
       remove the respective entry in new_fuc. If the replacement is done in a
       .param line, convert it to a .func line and add an entry to new_func.
       Continue until new_func is empty.
     */

    beg_func = new_func;
    for (; new_func; new_func = new_func->next) {

        for(j = 0; j < 16; j++)
            sub_count[j] = 0;

        card = deck;
        for (; card; card = card->li_next) {

            char *new_str = NULL; /* string we assemble here */
            char *curr_line = card->li_line;
            char * new_tmp_str, *tmp_str, *firsttok_str;
            /* Some new variables... */
            char *chp;
            char *chp_start;
            char *var_name;
            char ch;
            int state;

            if (*curr_line == '*')
                continue;

            /* determine nested depths of subcircuits */
            if (ciprefix(".subckt", curr_line)) {
                subckt_depth ++;
                sub_count[subckt_depth]++;
                continue;
            } else if (ciprefix(".ends", curr_line)) {
                subckt_depth --;
                continue;
            }

            /* exclude any command inside .control ... .endc */
            if (ciprefix(".control", curr_line)) {
                skip_control ++;
                continue;
            } else if (ciprefix(".endc", curr_line)) {
                skip_control --;
                continue;
            } else if (skip_control > 0) {
                continue;
            }

            /* exclude lines which do not have the same subcircuit
               nesting depth and number as found in new_func */
            if (subckt_depth != new_func->subckt_depth)
                continue;
            if (sub_count[subckt_depth] != new_func->subckt_count)
                continue;

            /* remove first token, ignore it here, restore it later */
            firsttok_str = gettok(&curr_line);
            if (*curr_line == '\0') {
                tfree(firsttok_str);
                continue;
            }

            /* This is the new code - it finds each variable name and checks it against new_func->funcname */
            for (state = 0, var_name = chp_start = chp = curr_line; ; chp++) {
                switch(state)
                {
                case 0:
                    /* in state 0 we are looking for the first character of a variable name,
                       which has to be an alphabetic character. */
                    if (isalpha(*chp))
                    {
                        state = 1;
                        var_name = chp;
                    }
                    break;
                case 1:
                    /* In state 1 we are looking for the last character of a variable name.
                       The variable name consists of alphanumeric characters and special characters,
                       which are defined above as VALIDCHARS. */
                    state = (*chp) && (isalphanum(*chp) || strchr(VALIDCHARS, *chp));
                    if (!state) {
                        ch = *chp;
                        *chp = 0;
                        if (strcmp(var_name, new_func->funcname) == 0 && ch != '(') {
                            new_str = INPstrCat(new_str, copy(chp_start), "");
                            new_str = INPstrCat(new_str, copy("()"), "");
                            chp_start=chp;
                        }
                        *chp = ch;
                    }
                    break;
                }
                if (!(*chp))
                    break;
            }
            if (new_str) {
                /* add final part of line */
                new_str = INPstrCat(new_str, copy(chp_start), "");
                /* restore first part of the line */
                new_str = INPstrCat(firsttok_str, new_str, " ");
                new_str = inp_remove_ws(new_str);
            }
            else
                continue;

            /* if we have inserted into a .param line, convert to .func */
            new_tmp_str = new_str;
            if (prefix(".param", new_tmp_str)) {
                tmp_str = gettok(&new_tmp_str);
                tfree(tmp_str);
                funcname = gettok_char(&new_tmp_str, '=', FALSE, FALSE);
                funcbody = copy(new_tmp_str + 1);
                inp_new_func(funcname, funcbody, card, &new_func, sub_count, subckt_depth);
                tfree(new_str);
                tfree(funcbody);
            } else {
                /* Or just enter new line into deck */
                card->li_next = xx_new_line(card->li_next, new_str, 0, card->li_linenum);
                *card->li_line = '*';
            }
        }
    }

    /* final memory clearance */
    tfree(sub_count);
    /* remove new_func */
    inp_rem_func(&beg_func);
}


/* enter function name, nested .subckt depths, and
 * number of .subckt at given level into struct new_func
 * and add line to deck
 */

static void
inp_new_func(char *funcname, char *funcbody, struct line *card, struct func_temper **new_func,
             int *sub_count, int subckt_depth)
{
    struct func_temper *new_func_tmp;
    static struct func_temper *new_func_end;
    char *new_str;

    new_func_tmp = TMALLOC(struct func_temper, 1);
    new_func_tmp->funcname = funcname;
    new_func_tmp->next = NULL;
    new_func_tmp->subckt_depth = subckt_depth;
    new_func_tmp->subckt_count = sub_count[subckt_depth];

    /* Insert at the back */
    if (*new_func == NULL) {
        *new_func = new_func_end = new_func_tmp;
    } else {
        new_func_end->next = new_func_tmp;
        new_func_end = new_func_tmp;
    }

    /* replace line in deck */
    new_str = TMALLOC(char, strlen(funcname) + strlen(funcbody) + 10);
    sprintf(new_str, ".func %s() %s", funcname, funcbody);
    card->li_next = xx_new_line(card->li_next, new_str, 0, card->li_linenum);
    *card->li_line = '*';
}


static void
inp_rem_func(struct func_temper **beg_func)
{
    struct func_temper *next_func;

    for(; *beg_func; *beg_func = next_func) {
        next_func = (*beg_func)->next;
        tfree((*beg_func)->funcname);
        tfree((*beg_func));
    }
}
