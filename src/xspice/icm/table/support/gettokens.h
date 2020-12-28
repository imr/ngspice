#ifndef gettokens_h_included
#define gettokens_h_included

/* Type definition for each possible token returned. */
typedef enum token_type_s { CNV_NO_TOK, CNV_STRING_TOK } Cnv_Token_Type_t;

char * CNVget_token(char **s, Cnv_Token_Type_t *type);
char *CNVgettok(char **s);
int cnv_get_spice_value(char *str, double *p_value);
#endif /* gettokens_h_included */
