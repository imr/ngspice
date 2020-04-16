struct PTltype {
    char *start, *stop;
};

extern int PTlex(YYSTYPE *lvalp, struct PTltype *llocp, char **line);

extern INPparseNode *PT_mkbnode(const char *opstr, INPparseNode *arg1, INPparseNode *arg2);
extern INPparseNode *PT_mkfnode(const char *fname, INPparseNode *arg);
extern INPparseNode *PT_mknnode(double number);
extern INPparseNode *PT_mksnode(const char *string, void *ckt);
