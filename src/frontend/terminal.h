#ifndef _TERMINAL_H
#define _TERMINAL_H

extern bool out_isatty;

#ifndef TCL_MODULE

void out_init(void);
void outbufputc(void);
void promptreturn(void);
void out_send(char *string);
void out_printf(char *fmt, char *s1, char *s2, char *s3,
		char *s4, char *s5, char *s6, 
		char *s7, char *s8, char *s9, char *s10);
void  term_clear(void);
void  term_home(void);
void  term_cleol(void);
void tcap_init(void);

#else

extern int tcl_printf(const char *format, ...);

inline extern void out_init(void) {}
inline extern void outbufputc(void) {}
inline extern void promptreturn(void) {}
inline extern void term_clear(void) {}
inline extern void term_home(void) {}
inline extern void term_cleol(void) {}
inline extern void tcap_init(void) {}

inline extern void out_send(char *string) {tcl_printf(string);}

inline extern void
out_printf(char *fmt, char *s1, char *s2, char *s3, char *s4, char *s5, char *s6, char *s7, char *s8, char *s9, char *s10) { 	
	tcl_printf(fmt, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10);
}


#endif



#endif
