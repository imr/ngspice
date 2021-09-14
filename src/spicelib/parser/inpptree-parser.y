%{
    /*
     * (compile (concat "bison -ydo inpptree-parser.c " (file-relative-name buffer-file-name)))
     */

  #include "ngspice/ngspice.h"
  #include "ngspice/inpptree.h"

  #include <stdio.h>
  #include <stdlib.h>

  # define YYLTYPE struct PTltype

  #include "inpptree-parser.h"
  #include "inpptree-parser-y.h"


  # define YYLLOC_DEFAULT(Current, Rhs, N)                               \
     do                                                                  \
       if (N) {                                                          \
           (Current).start = YYRHSLOC(Rhs, 1).start;                     \
           (Current).stop  = YYRHSLOC(Rhs, N).stop;                      \
       } else {                                                          \
           (Current).start = (Current).stop = YYRHSLOC(Rhs, 0).stop;     \
       }                                                                 \
     while (0)

  static void PTerror (YYLTYPE *locp, char **line, struct INPparseNode **retval, void *ckt, char const *);
%}

%name-prefix "PT"

%defines

%pure-parser

%parse-param {char **line}
%lex-param   {char **line}

%parse-param {struct INPparseNode **retval}
%parse-param {CKTcircuit *ckt}

%union {
  double num;
  const char  *str;
  struct INPparseNode *pnode;
}

%token <num>  TOK_NUM
%token <str>  TOK_STR
%token <pnode> TOK_pnode
%token        TOK_LE TOK_LT TOK_GE TOK_GT TOK_EQ TOK_NE

%type  <pnode>   exp nonempty_arglist

// Operator Precedence

%left   ','
%right  '?' ':'
%left   TOK_OR
%left   TOK_AND
%left   TOK_EQ TOK_NE
%left   TOK_LE TOK_LT TOK_GE TOK_GT
%left   '-' '+'
%left   '*' '/'
%left   '^'      /* exponentiation */
%left   NEG '!'  /* negation--unary minus, and boolean not */

%initial-action      /* initialize yylval */
{
    $$.num = 0.0;
    yylloc.start = yylloc.stop = NULL;
};

%%

expression: exp
  {
      *retval = $1;
      *line = @1.stop;
      YYACCEPT;
 }


exp:
    TOK_NUM                           { $$ = PT_mknnode($1); }
  | TOK_STR                           { $$ = PT_mksnode($1, ckt); txfree($1); }

  | exp '+' exp                       { $$ = PT_mkbnode("+", $1, $3); }
  | exp '-' exp                       { $$ = PT_mkbnode("-", $1, $3); }
  | exp '*' exp                       { $$ = PT_mkbnode("*", $1, $3); }
  | exp '/' exp                       { $$ = PT_mkbnode("/", $1, $3); }
  | exp '^' exp                       { $$ = PT_mkbnode("^", $1, $3); }

  | '(' exp ')'                       { $$ = $2; }

  | '-' exp  %prec NEG                { $$ = PT_mkfnode("-",$2); }
  | '+' exp  %prec NEG                { $$ = $2; }

  | TOK_STR '(' nonempty_arglist ')'  { $$ = PT_mkfnode($1, $3);
                                        if (!$$)
                                            YYERROR;
                                        txfree($1); }

  | TOK_pnode

  | exp '?' exp ':' exp               { $$ = PT_mkfnode("ternary_fcn",
                                               PT_mkbnode(",",
                                                 PT_mkbnode(",", $1, $3),
                                                 $5)); }

  | exp TOK_EQ exp                    { $$ = PT_mkfnode("eq0", PT_mkbnode("-",$1,$3)); }
  | exp TOK_NE exp                    { $$ = PT_mkfnode("ne0", PT_mkbnode("-",$1,$3)); }
  | exp TOK_GT exp                    { $$ = PT_mkfnode("gt0", PT_mkbnode("-",$1,$3)); }
  | exp TOK_LT exp                    { $$ = PT_mkfnode("lt0", PT_mkbnode("-",$1,$3)); }
  | exp TOK_GE exp                    { $$ = PT_mkfnode("ge0", PT_mkbnode("-",$1,$3)); }
  | exp TOK_LE exp                    { $$ = PT_mkfnode("le0", PT_mkbnode("-",$1,$3)); }

  | exp TOK_OR exp                    { $$ = PT_mkfnode("ne0",
                                               PT_mkbnode("+",
                                                 PT_mkfnode("ne0", $1),
                                                   PT_mkfnode("ne0", $3))); }
  | exp TOK_AND exp                   { $$ = PT_mkfnode("eq0",
                                               PT_mkbnode("+",
                                                 PT_mkfnode("eq0", $1),
                                                   PT_mkfnode("eq0", $3))); }
  | '!' exp                           { $$ = PT_mkfnode("eq0", $2); }

  ;

nonempty_arglist:
    exp
  | nonempty_arglist ',' exp     { $$ = PT_mkbnode(",", $1, $3); }

%%


/* Called by yyparse on error.  */
static void
PTerror (YYLTYPE *locp, char **line, struct INPparseNode **retval, void *ckt, char const *s)
{
  NG_IGNORE(line);
  NG_IGNORE(retval);
  NG_IGNORE(ckt);

  fprintf (stderr, "\n%s: %s, parsing stopped at\n    %s\n\n", __func__, s, locp->start);
}
