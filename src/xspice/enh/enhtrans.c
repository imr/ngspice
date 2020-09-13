/* ===========================================================================
FILE    ENHtranslate_poly.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    7/24/2012 Holger Vogt Update to error messages

SUMMARY

    This file contains functions used by the simulator in
    calling the internal "poly" code model to substitute for
    SPICE 2G6 style poly sources found in the input deck.

INTERFACES

    ENHtranslate_poly()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

/*=== FUNCTION PROTOTYPES ===*/

/*=== INCLUDE FILES ===*/


/* #include "prefix.h"   */

#include "ngspice/ngspice.h"
//#include "misc.h"

#include "ngspice/fteinp.h"
#include "ngspice/enh.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/mifproto.h"

/* #include "suffix.h" */


/*=== FUNCTION PROTOTYPES ===*/
static int needs_translating(char *card);
static int count_tokens(char *card);
static char *two2three_translate(char  *orig_card, char  **inst_card,
                                 char  **mod_card);
static int get_poly_dimension(char *card);

/*
ENHtranslate_poly()

Translate all 2G6 style polynomial controlled sources in the deck
to new polynomial controlled source code model syntax.
*/

/*---------------------------------------------------------------------*/
/*  ENHtranslate_poly takes (a pointer to) the SPICE deck as argument. */
/*  It loops through the deck, and translates all POLY statements      */
/*  in dependent sources into a .model conformant with the needs of    */
/*  XSPICE.  It splices the new statements in the deck, and comments   */
/*  out the old dependent  source.                                     */
/*  It returns (a pointer to) the processed deck.                      */
/*---------------------------------------------------------------------*/
struct card *
ENHtranslate_poly(
    struct card  *deck) {        /* Linked list of lines in input deck */
    struct card  *d;
    struct card  *l1;
    struct card  *l2;

    char  *card;

    /* Iterate through each card in the deck and translate as needed */
    for(d = deck; d; d = d->nextcard) {

#ifdef TRACE
        /* SDB debug statement */
        printf("In ENHtranslate_poly, now examining card %s . . . \n", d->line);
#endif

        /* If doesn't need to be translated, continue to next card */
        if(! needs_translating(d->line)) {

#ifdef TRACE
            /* SDB debug statement */
            /* printf("Card doesn't need translating.  Continuing . . . .\n"); */
#endif
            continue;
        }

#ifdef TRACE
        /* SDB debug statement */
        printf("Found a card to translate . . . .\n");
#endif

        /* Create two new line structs and splice into deck */
        l1 = TMALLOC(struct card, 1);
        l2 = TMALLOC(struct card, 1);
        l2->nextcard = d->nextcard;
        l1->nextcard = l2;
        d->nextcard  = l1;

        /* PN 2004: Add original linenumber to ease the debug process
         * for malfromned netlist
         */

        l1->linenum = d->linenum;
        l2->linenum = d->linenum;

        /* Create the translated cards */
        d->error = two2three_translate(d->line, &(l1->line), &(l2->line));

        /* Comment out the original line */
        card = TMALLOC(char, strlen(d->line) + 2);
        strcpy(card,"*");
        strcat(card, d->line);
        tfree(d->line);
        d->line = card;

#ifdef TRACE
        /* SDB debug statement */
        printf("In ENHtranslate_poly, translated card = %s . . . \n", card);
#endif

        /* Advance deck pointer to last line added */
        d = l2;
    }

    /* Return head of deck */
    return(deck);

} /* ENHtranslate_poly */


/*---------------------------------------------------------------------*/
/*
needs_translating()

Test to see if card needs translating.  Return true if card defines
an e,f,g, or h controlled source, has too many tokens to be
a simple linear dependent source and contains the 'poly' token.
Otherwise return false.
*/


static int needs_translating(
    char *card)           /* the card text to check */
{

#ifdef TRACE
    /* SDB debug statement */
    /* printf("In needs_translating, examining card %s . . . \n", card); */
#endif

    switch(*card) {

    case 'e':
    case 'E':
    case 'g':
    case 'G':
        if(count_tokens(card) > 6)
            return(1);
        else
            return(0);

    case 'f':
    case 'F':
    case 'h':
    case 'H':
        if(count_tokens(card) > 5)
            return(1);
        else
            return(0);

    default:
        return(0);
    }

} /* needs_translating */



/*---------------------------------------------------------------------*/
/*
count_tokens()

Count and return the number of tokens on the card.
*/
static int count_tokens(
    char *card)           /* the card text on which to count tokens */
{
    int   i;
    bool has_poly = FALSE;

    /* Get and count tokens until end of line reached and find the "poly" token */
    for(i = 0; *card != '\0'; i++) {
        char *newtoken;
        newtoken = MIFgettok(&card);
        if ((i == 3) && ciprefix(newtoken, "poly"))
            has_poly = TRUE;
        txfree(newtoken);
    }
    /* no translation, if 'poly' not in the line */
    if (!has_poly) i=0;
    return(i);

} /* count_tokens */



/********************************************************************/
/*====================================================================

two2three_translate()

Do the syntax translation of the 2G6 source to the new code model
syntax.  The translation proceeds according to the template below.

--------------------------------------------
VCVS:
ename N+ N- POLY(dim) NC+ NC- P0 P1 P2 . . .
N+ N- = outputs
NC+ NC- = inputs

aname %vd[NC+ NC-] %vd[N+ N-] pname
.model pname spice2poly(coef=P0 P1 P2 . . . )
%vd[NC+ NC-] = inputs
%vd[N+ N-] = outputs
--------------------------------------------
CCCS
fname N+ N- POLY(dim) Vname P0 P1 P2 . . .
N+ N- = outputs
Vname = input voltage source (measures current)

aname %vnam[Vname] %id[N+ N-] pname
.model pname spice2poly(coef=P0 P1 P2 . . . )
%vnam[Vname] = input
%id[N+ N-] = output
--------------------------------------------
VCCS
gname N+ N- POLY(dim) NC+ NC- P0 P1 P2 , , ,
N+ N- = outputs
NC+ NC- = inputs

aname %vd[NC+ NC-] %id[N+ N-] pname
.model pname spice2poly(coef=P0 P1 P2 . . . )
%vd[NC+ NC-] = inputs
%id[N+ N-] = outputs
--------------------------------------------
CCVS
hname N+ N- POLY(dim) Vname P0 P1 P2 . . .
N+ N- = outputs
Vname = input voltage source (measures current)

aname %vnam[Vname] %vd[N+ N-] pname
.model pname spice2poly(coef=P0 P1 P2 . . . )
%vnam[Vname] = input
%vd[N+ N-] = output

====================================================================*/
/********************************************************************/

static char *two2three_translate(
    char  *orig_card,    /* the original untranslated card */
    char  **inst_card,   /* the instance card created by the translation */
    char  **mod_card)    /* the model card created by the translation */
{
    int   dim;
    int   num_tokens;

    int   num_conns;
    int   num_coefs;
    size_t inst_card_len;
    size_t mod_card_len;

    int   i;

    char  type;

    char  *name;
    char  **out_conn;
    char  **in_conn;
    char  **coef;

    char  *card;


#ifdef TRACE
    /* SDB debug statement */
    printf("In two2three_translate, card to translate = %s . . .\n", orig_card);
#endif

    /* Put the first character into local storage for checking type */
    type = *orig_card;

    /* Count the number of tokens for use in parsing */
    num_tokens = count_tokens(orig_card);

    /* Determine the dimension of the poly source */
    /* Note that get_poly_dimension returns 0 for "no poly", -1 for
       invalid dimensiion, otherwise returns numeric value of POLY */
    dim = get_poly_dimension(orig_card);
    if(dim == -1) {
        char *errmsg;
        printf("ERROR in two2three_translate -- Argument to poly() is not an integer\n");
        printf("ERROR  while parsing: %s\n", orig_card);
        errmsg = copy("ERROR in two2three_translate -- Argument to poly() is not an integer\n");
        *inst_card = copy(" * ERROR Argument to poly() is not an integer");
        *mod_card  = copy(" * ERROR Argument to poly() is not an integer");
        return errmsg;
    }

    /* Compute number of output connections based on type and dimension */
    switch(type) {
    case 'E':
    case 'e':
    case 'G':
    case 'g':
        num_conns = 2 * dim;
        break;

    default:
        num_conns = dim;
    }

    /* Compute number of coefficients.  Return error if less than one. */
    if(dim == 0)
        num_coefs = num_tokens - num_conns - 3;  /* no POLY token */
    else
        num_coefs = num_tokens - num_conns - 5;  /* POLY token present */

#ifdef TRACE
    /* SDB debug statement */
    printf("In two2three_translate, num_tokens=%d, num_conns=%d, num_coefs=%d . . .\n", num_tokens, num_conns, num_coefs);
#endif

    if(num_coefs < 1) {
        char *errmsg;
        printf("ERROR - Number of connections differs from poly dimension\n");
        printf("ERROR  while parsing: %s\n", orig_card);
        errmsg = copy("ERROR in two2three_translate -- Argument to poly() is not an integer\n");
        *inst_card = copy("* ERROR - Number of connections differs from poly dimension\n");
        *mod_card  = copy(" * ERROR - Number of connections differs from poly dimension\n");
        return(errmsg);
    }
    /* Split card into name, output connections, input connections, */
    /* and coefficients */

    card = orig_card;
    name = MIFgettok(&card);

    /* Get output connections (2 netnames) */
    out_conn = TMALLOC(char *, 2);
    for(i = 0; i < 2; i++)
        out_conn[i] = MIFgettok(&card);

    /* check for POLY, and ignore it if present */
    if (dim >  0) {

#ifdef TRACE
        /* SDB debug statement */
        printf("In two2three_translate, found poly!!!  dim = %d \n", dim);
#endif

        txfree(MIFgettok(&card)); /* read and discard POLY */
        txfree(MIFgettok(&card)); /* read and discard dimension */
    }


    /* Get input connections (2 netnames per dimension) */
    in_conn = TMALLOC(char *, num_conns);
    for(i = 0; i < num_conns; i++)
        in_conn[i] = MIFgettok(&card);

    /* The remainder of the line are the poly coeffs. */
    coef = TMALLOC(char *, num_coefs);
    for(i = 0; i < num_coefs; i++)
        coef[i] = MIFgettok(&card);

    /* Compute the size needed for the new cards to be created */
    /* Allow a fair amount of extra space for connection types, etc. */
    /* to be safe... */

    inst_card_len = 70;
    inst_card_len += 2 * (strlen(name) + 1);
    for(i = 0; i < 2; i++)

        inst_card_len += strlen(out_conn[i]) + 1;
    for(i = 0; i < num_conns; i++)
        inst_card_len += strlen(in_conn[i]) + 1;

    mod_card_len = 70;
    mod_card_len += strlen(name) + 1;
    for(i = 0; i < num_coefs; i++)
        mod_card_len += strlen(coef[i]) + 1;

    /* Allocate space for the cards and write them into the strings */

    *inst_card = TMALLOC(char, inst_card_len);
    *mod_card = TMALLOC(char, mod_card_len);

    strcpy(*inst_card, "a$poly$");
    sprintf(*inst_card + strlen(*inst_card), "%s ", name);

    /* Write input nets/sources */
    if((type == 'e') || (type == 'g') ||
            (type == 'E') || (type == 'G')) /* These input port types are vector & need a [. */
        sprintf(*inst_card + strlen(*inst_card), "%%vd [ ");
    else                                    /* This input port type is scalar */
        sprintf(*inst_card + strlen(*inst_card), "%%vnam [ ");


    for(i = 0; i < num_conns; i++)
        sprintf(*inst_card + strlen(*inst_card), "%s ", in_conn[i]);


    sprintf(*inst_card + strlen(*inst_card), "] ");


    /* Write output nets */
    if((type == 'e') || (type == 'h') ||
            (type == 'E') || (type == 'H'))
        sprintf(*inst_card + strlen(*inst_card), "%%vd ( ");
    else
        sprintf(*inst_card + strlen(*inst_card), "%%id ( ");

    for(i = 0; i < 2; i++)
        sprintf(*inst_card + strlen(*inst_card), "%s ", out_conn[i]);

    sprintf(*inst_card + strlen(*inst_card), ") ");


    /* Write model name */
    sprintf(*inst_card + strlen(*inst_card), "a$poly$%s", name);


    /* Now create model card */
    sprintf(*mod_card, ".model a$poly$%s spice2poly coef = [ ", name);
    for(i = 0; i < num_coefs; i++)
        sprintf(*mod_card + strlen(*mod_card), "%s ", coef[i]);
    sprintf(*mod_card + strlen(*mod_card), "]");

#ifdef TRACE
    /* SDB debug statement */
    printf("In two2three_translate, translated statements:\n%s \n%s \n", *inst_card, *mod_card);
#endif

    /* Free the temporary space */
    FREE(name);
    name = NULL;

    for(i = 0; i < 2; i++) {
        FREE(out_conn[i]);
        out_conn[i] = NULL;
    }

    FREE(out_conn);
    out_conn = NULL;

    for(i = 0; i < num_conns; i++) {
        FREE(in_conn[i]);
        in_conn[i] = NULL;
    }

    FREE(in_conn);
    in_conn = NULL;

    for(i = 0; i < num_coefs; i++) {
        FREE(coef[i]);
        coef[i] = NULL;
    }

    FREE(coef);

    coef = NULL;

    /* Return NULL to indicate no error */
    return(NULL);

} /* two2three_translate */



/*--------------------------------------------------------------------*/
/*
get_poly_dimension()

Get the poly source dimension from the token immediately following
the 'poly' if any.  Return values changed by SDB on 5.23.2003 to be:
If "poly" is not present, return 0.
If the dimension token following "poly" is invalid, return -1.
Otherwise, return the integer dimension.
*/


static int get_poly_dimension(
    char *card)                 /* the card text */
{

    int   i;
    int   dim;
    char  *local_tok;

    /* Skip over name and output connections */
    for(i = 0; i < 3; i++)
        txfree(MIFgettok(&card));

    /* Check the next token to see if it is "poly" */
    /* If not, return  0                           */
    local_tok = MIFgettok(&card);
    if( strcmp(local_tok, "poly") &&
            strcmp(local_tok, "POLY") ) { /* check that local_tok is *not* poly */
        FREE(local_tok);
        local_tok = NULL;
        return(0);
    }

    FREE(local_tok);

    /* Must have been "poly", so next line must be a number */
    /* Try to convert it.  If successful, return the number */
    /* else, return -1 to indicate an error...               */
    local_tok = MIFgettok(&card);
    dim = atoi(local_tok);
    FREE(local_tok);

    if (dim > 0) {
        return(dim);
    } else {
        return(-1);
    }

} /* get_poly_dimension */

