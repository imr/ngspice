/* Forwards and external function declarations
   for winmain.c
*/

/* Forward definition of main() */
int xmain( int argc, char * argv[]);
/* forward of Update function */
#ifdef __CYGWIN__
static char* rlead(char*);
#endif
void winmessage(char*);

static void HistoryInit(void);
static void HistoryScroll(void);
static void HistoryEnter( char * newLine);
static char * HistoryGetPrev(void);
static char * HistoryGetNext(void);
void WaitForIdle(void);
static void WaitForMessage(void);
static void ClearInput(void);
void SetSource( char * Name);
void SetAnalyse(char *, int );
static void AdjustScroller(void);
static void _DeleteFirstLine(void);
static void AppendChar( char c);
static void AppendString( const char * Line);
static void DisplayText( void);

static int w_getch(void);
static int w_putch( int c);

static void Main_OnSize(HWND hwnd, UINT state, int cx, int cy);
static void PostSpiceCommand( const char * const cmd);
static LRESULT CALLBACK MainWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK StringWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK TextWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static void Element_OnPaint(HWND hwnd);
static LRESULT CALLBACK ElementWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static int MakeArgcArgv(char *cmdline,int *argc,char ***argv);

int	f_c_l_o_s_e( FILE * __stream);
int	f_f_l_u_s_h( FILE * __stream);
int	fg_e_t_c( FILE * __stream);
int	f_g_e_t_p_o_s( FILE * __stream, fpos_t * __pos);
char * fg_e_t_s(char * __s, int __n, FILE * __stream);
int    fp_r_i_n_t_f(FILE * __stream, const char * __format, ...);
int    fp_u_t_c(int __c, FILE * __stream);
int    fp_u_t_s(const char * __s, FILE * __stream);
size_t f_r_e_a_d(void * __ptr, size_t __size, size_t __n, FILE * __stream);
FILE * f_r_e_o_p_e_n(const char * __path, const char * __mode, FILE * __stream);
int    fs_c_a_n_f(FILE * __stream, const char * __format, ...);
int    f_s_e_e_k(FILE * __stream, long __offset, int __whence);
int    f_s_e_t_p_o_s(FILE * __stream, const fpos_t*__pos);
long   f_t_e_l_l(FILE * __stream);
size_t f_w_r_i_t_e(const void * __ptr, size_t __size, size_t __n, FILE * __stream);
char * g_e_t_s(char * __s);
void   p_e_r_r_o_r(const char * __s);
int    p_r_i_n_t_f(const char * __format, ...);
int    p_u_t_s(const char * __s);
int    s_c_a_n_f(const char * __format, ...);
int    ung_e_t_c(int __c, FILE * __stream);
int    vfp_r_i_n_t_f(FILE * __stream, const char * __format, void * __arglist);
/*int   vfs_c_a_n_f(FILE * __stream, const char * __format, void * __arglist);*/
int    vp_r_i_n_t_f(const char * __format, void * __arglist);
/*int   vs_c_a_n_f(const char * __format, void * __arglist); */
int r_e_a_d(int fd, char * __buf, int __n);
int g_e_t_c(FILE * __fp);
int g_e_t_char(void);
int p_u_t_char(const int __c);
int p_u_t_c(const int __c, FILE * __fp);
int f_e_o_f(FILE * __fp);
int f_e_r_r_o_r(FILE * __fp);
int fg_e_t_char(void);
int fp_u_t_char(int __c);
