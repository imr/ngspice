/* ===========================================================================
FILE    ENHtranslate_poly.c

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

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

//void free(void *); //ka removed compiler error
/* int  atoi(char *);  */


/*=== INCLUDE FILES ===*/


/* #include "prefix.h"   */

#include "ngspice.h"
//#include "misc.h"

#include "fteinp.h"
#include "enh.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "mifproto.h"

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
struct line * ENHtranslate_poly(
   struct   line  *deck)          /* Linked list of lines in input deck */
{
   struct   line  *d;
   struct   line  *l1;
   struct   line  *l2;

   char  *card;
   int poly_dimension;
   char *buff;

   /* Iterate through each card in the deck and translate as needed */
   for(d = deck; d; d = d->li_next) 
   {

#ifdef TRACE
     /* SDB debug statement */
     printf("In ENHtranslate_poly, now examining card %s . . . \n", d->li_line);
#endif

     /* If doesn't need to be translated, continue to next card */
     if(! needs_translating(d->li_line)) {

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
/*      l1 = alloc(line);  */  /* jgroves */
/*      l2 = alloc(line);  */  /* jgroves */
     l1 = alloc(struct line);
     l2 = alloc(struct line);
     l2->li_next = d->li_next;
     l1->li_next = l2;
     d->li_next  = l1;
     
     /* Create the translated cards */
     d->li_error = two2three_translate(d->li_line, &(l1->li_line), &(l2->li_line));
     
     /* Comment out the original line */
     card = (void *) MALLOC(strlen(d->li_line) + 2);
     strcpy(card,"*");
     strcat(card, d->li_line);
     d->li_line = card;

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
an e,f,g, or h controlled source and has too many tokens to be
a simple linear dependent source.  Otherwise return false.
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
   case 'g':
      if(count_tokens(card) <=6)
         return(0);
      else
         return(1);

   case 'f':
   case 'h':
      if(count_tokens(card) <= 5)
         return(0);
      else
         return(1);

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

   /* Get and count tokens until end of line reached */
   for(i = 0; *card != '\0'; i++)
      txfree(MIFgettok(&card));

   return(i);

} /* count_tokens */




/*--------------------------------------------------------------------*/
/*
two2three_translate()

Do the syntax translation of the 2G6 source to the new code model syntax.

Renamed by SDB to eliminate clash with translate fcn defined in subckt.c
4.17.2003 -- SDB

*/

static char *two2three_translate(
   char  *orig_card,    /* the original untranslated card */
   char  **inst_card,   /* the instance card created by the translation */
   char  **mod_card)    /* the model card created by the translation */
{
   int   dim;
   int   num_tokens;

   int   num_conns;
   int   num_coefs;
   int   inst_card_len;
   int   mod_card_len;

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

   /* Get the first character into local storage for checking type */
   type = *orig_card;

   /* Count the number of tokens for use in parsing */
   num_tokens = count_tokens(orig_card);

   /* Determine the dimension of the poly source */
   dim = get_poly_dimension(orig_card);
   if(dim <= 0) {
      printf("ERROR in two2three_translate -- Argument to poly() is not an integer\n");
      return("ERROR - Argument to poly() is not an integer\n");
   }

   /* Compute number of input connections based on type and dimension */
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
   if(dim == 1)
      num_coefs = num_tokens - num_conns - 3;
   else
      num_coefs = num_tokens - num_conns - 5;

   if(num_coefs < 1)
      return("ERROR - Number of connections differs from poly dimension\n");

   /* Split card into name, output connections, input connections, */
   /* and coefficients */

   card = orig_card;
   name = MIFgettok(&card);

   out_conn = (void *) MALLOC(2 * sizeof(char *));
   for(i = 0; i < 2; i++)
      out_conn[i] = MIFgettok(&card);

   if(dim > 1)
      for(i = 0; i < 2; i++)
         txfree(MIFgettok(&card));

   in_conn = (void *) MALLOC(num_conns * sizeof(char *));
   for(i = 0; i < num_conns; i++)
      in_conn[i] = MIFgettok(&card);

   coef = (void *) MALLOC(num_coefs * sizeof(char *));
   for(i = 0; i < num_coefs; i++)
      coef[i] = MIFgettok(&card);

   /* Compute the size needed for the new cards to be created */
   /* Allow a fair amount of extra space for connection types, etc. */
   /* to be safe... */

   inst_card_len = 50;
   inst_card_len += 2 * (strlen(name) + 1);
   for(i = 0; i < 2; i++)
      inst_card_len += strlen(out_conn[i]) + 1;
   for(i = 0; i < num_conns; i++)
      inst_card_len += strlen(in_conn[i]) + 1;

   mod_card_len = 50;
   mod_card_len += strlen(name) + 1;
   for(i = 0; i < num_coefs; i++)
      mod_card_len += strlen(coef[i]) + 1;

   /* Allocate space for the cards and write them into the strings */

   *inst_card = (void *) MALLOC(inst_card_len);
   *mod_card = (void *) MALLOC(mod_card_len);

   strcpy(*inst_card, "a$poly$");
   sprintf(*inst_card + strlen(*inst_card), "%s ", name);

   if((type == 'e') || (type == 'g') ||
      (type == 'E') || (type == 'G'))
      sprintf(*inst_card + strlen(*inst_card), "%%vd [ ");
   else
      sprintf(*inst_card + strlen(*inst_card), "%%vnam [ ");

   for(i = 0; i < num_conns; i++)
      sprintf(*inst_card + strlen(*inst_card), "%s ", in_conn[i]);

   sprintf(*inst_card + strlen(*inst_card), "] ");

   if((type == 'e') || (type == 'h') ||
      (type == 'E') || (type == 'H'))
      sprintf(*inst_card + strlen(*inst_card), "%%vd ");
   else
      sprintf(*inst_card + strlen(*inst_card), "%%id ");

   for(i = 0; i < 2; i++)
      sprintf(*inst_card + strlen(*inst_card), "%s ", out_conn[i]);

   sprintf(*inst_card + strlen(*inst_card), "a$poly$%s", name);


   sprintf(*mod_card, ".model a$poly$%s poly coef = [ ", name);
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

   for(i = 0; i < 2; i++)
   {
      FREE(out_conn[i]);
      out_conn[i] = NULL;
   }

   FREE(out_conn);
   out_conn = NULL;

   for(i = 0; i < num_conns; i++)
   {
      FREE(in_conn[i]);
      in_conn[i] = NULL;
   }

   FREE(in_conn);
   in_conn = NULL;

   for(i = 0; i < num_coefs; i++)
   {
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
the 'poly' if any.  If 'poly' is not present, return 1.  If poly is
present and token following is a valid integer, return it.  Else
return 0.
*/


static int get_poly_dimension(
   char *card)                 /* the card text */
{

   int   i;
   int   dim;
   char  *tok;


   /* Skip over name and output connections */
   for(i = 0; i < 3; i++)
      txfree(MIFgettok(&card));

   /* Check the next token to see if it is "poly" */
   /* If not, return a dimension of 1             */
   tok = MIFgettok(&card);
   if(strcmp(tok, "poly")) 
   {
      FREE(tok);

	  tok = NULL;

      return(1);
   }

   FREE(tok);

   /* Must have been "poly", so next line must be a number */
   /* Try to convert it.  If successful, return the number */
   /* else, return 0 to indicate an error...               */
   tok = MIFgettok(&card);
   dim = atoi(tok);
   FREE(tok);

   return(dim);

} /* get_poly_dimension */

