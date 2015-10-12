struct PPltype {
    const char *start, *stop;
};

extern int PPlex(YYSTYPE *lvalp, struct PPltype *llocp, char **line);

extern struct pnode *PP_mkunode(int op, struct pnode *arg);
extern struct pnode *PP_mkfnode(const char *func, struct pnode *arg);
extern struct pnode *PP_mknnode(double number);
extern struct pnode *PP_mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2);
extern struct pnode *PP_mksnode(const char *string);


#if defined (_MSC_VER)
# define __func__ __FUNCTION__ /* __func__ is C99, but MSC can't */
#endif
