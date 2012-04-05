%{
    /*
     * (compile (concat "bison " (file-relative-name buffer-file-name)))
     */

  #include <stdio.h>
  #include <stdlib.h>

  struct PPltype {
    const char *start, *stop;
  };

  # define YYLTYPE struct PPltype

  # define YYLLOC_DEFAULT(Current, Rhs, N)                               \
     do                                                                  \
       if (N) {                                                          \
           (Current).start = YYRHSLOC(Rhs, 1).start;                     \
           (Current).stop  = YYRHSLOC(Rhs, N).stop;                      \
       } else {                                                          \
           (Current).start = (Current).stop = YYRHSLOC(Rhs, 0).stop;     \
       }                                                                 \
     while (0)

  #include "parse-bison.h"

  extern int PPlex (YYSTYPE *lvalp, struct PPltype *llocp, char **line);
  extern int PPdebug;

  static void PPerror (YYLTYPE *locp, char **line, struct pnode **retval, char const *);

  #if defined (_MSC_VER)
  # define __func__ __FUNCTION__ /* __func__ is C99, but MSC can't */
  #endif


%}

%name-prefix="PP"
%output="parse-bison.c"

%defines
%locations
%debug

%pure-parser

%parse-param {char **line}
%lex-param   {char **line}

%parse-param {struct pnode **retval}

%union {
  double num;
  const char  *str;
  struct pnode *pnode;
}

/*
 * This gramar has two expected shift/reduce conflicts
 *   exp1 '-' exp2  can
 *     o  yield an exp, interpreting '-' as a binary operator
 *     o  yield a list of two expressions
 *          exp1  and
 *          unary '-' exp2
 *     the first interpretation is favoured. (bison defaults to 'shift')
 *   TOK_STR '(' exp1 ')'  can
 *     o  yield an exp, per function application
 *     o  yield a list of two expressions
 *          TOK_STR  and
 *          '(' exp1 ')'  which will be reduced to exp1
 *     the first interpretation is favoured. (bison defaults to 'shift')
 *
 * to verify:
 *   execute bison --report=state
 *
 * the %expect 2
 *   manifests my expectation, and will issue a `warning' when not met
 */

%expect 2

%token <num>  TOK_NUM
%token <str>  TOK_STR
%token        TOK_LE TOK_GE TOK_NE TOK_LRANGE TOK_RRANGE

%type  <pnode>   exp exp_list one_exp

/* Operator Precedence */

%right  '?' ':'
%left   '|'
%left   '&'
%left   '=' TOK_NE TOK_LE '<' TOK_GE '>'
%left   '~'
%right  ','
%left   '-' '+'
%left   '*' '/' '%'
%right  '^'      /* exponentiation */
%left   NEG      /* negation--unary minus */
%left   '[' ']'
%left   TOK_LRANGE TOK_RRANGE

%initial-action      /* initialize yylval */
{
    $$.num = 0.0;
    yylloc.start = yylloc.stop = NULL;
};

%%

/*
 */


expression:
             { *retval = NULL; }
  | exp_list { *retval = $1; }
;

exp_list:
    one_exp
  | one_exp exp_list                  { $1->pn_next = $2; $2->pn_use ++; $$ = $1; }
;

one_exp:
    exp  {
        $1->pn_name = copy_substring(@1.start, @1.stop);
        $$ = $1;
    }
;

exp:
    TOK_NUM                           { $$ = mknnode($1); }
  | TOK_STR                           { $$ = mksnode($1); txfree((void*)$1); }

  | exp ',' exp                       { $$ = mkbnode(PT_OP_COMMA,  $1, $3); }
  | exp '+' exp                       { $$ = mkbnode(PT_OP_PLUS,   $1, $3); }
  | exp '-' exp                       { $$ = mkbnode(PT_OP_MINUS,  $1, $3); }
  | exp '*' exp                       { $$ = mkbnode(PT_OP_TIMES,  $1, $3); }
  | exp '%' exp                       { $$ = mkbnode(PT_OP_MOD,    $1, $3); }
  | exp '/' exp                       { $$ = mkbnode(PT_OP_DIVIDE, $1, $3); }
  | exp '^' exp                       { $$ = mkbnode(PT_OP_POWER,  $1, $3); }

  | '(' exp ')'                       { $$ = $2; }

  | '-' exp  %prec NEG                { $$ = mkunode(PT_OP_UMINUS, $2); }
  | '~' exp                           { $$ = mkunode(PT_OP_NOT, $2); }

  | TOK_STR '(' exp ')'               { $$ = mkfnode($1, $3);
                                        txfree((void*)$1);
                                        if(!$$)
                                            YYABORT;
                                      }

  | exp '=' exp                       { $$ = mkbnode(PT_OP_EQ, $1, $3); }
  | exp TOK_NE exp                    { $$ = mkbnode(PT_OP_NE, $1, $3); }
  | exp '>' exp                       { $$ = mkbnode(PT_OP_GT, $1, $3); }
  | exp '<' exp                       { $$ = mkbnode(PT_OP_LT, $1, $3); }
  | exp TOK_GE exp                    { $$ = mkbnode(PT_OP_GE, $1, $3); }
  | exp TOK_LE exp                    { $$ = mkbnode(PT_OP_LE, $1, $3); }

  | exp '&' exp                       { $$ = mkbnode(PT_OP_AND, $1, $3); }
  | exp '|' exp                       { $$ = mkbnode(PT_OP_OR,  $1, $3); }

  | exp '[' exp ']'                   { $$ = mkbnode(PT_OP_INDX,  $1, $3); }
  | exp TOK_LRANGE exp TOK_RRANGE     { $$ = mkbnode(PT_OP_RANGE, $1, $3); }
  | exp '?' exp ':' exp               { $$ = mkbnode(PT_OP_TERNARY,$1,
                                                     mkbnode(PT_OP_COMMA,$3,$5)); }
;

%%


/* Called by yyparse on error.  */
static void
PPerror (YYLTYPE *locp, char **line, struct pnode **retval, char const *s)
{
  NG_IGNORE(locp);
  NG_IGNORE(line);
  NG_IGNORE(retval);

  fprintf (stderr, "%s: %s\n", __func__, s);
}
