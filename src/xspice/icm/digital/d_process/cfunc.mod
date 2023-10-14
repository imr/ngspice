/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_process/cfunc.mod

Copyright 2017-2018 Isotel d.o.o. http://www.isotel.eu
PROJECT http://isotel.eu/mixedsim

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


AUTHORS

    2017-2018 Uros Platse <uros@isotel.eu>


SUMMARY

    This module provides an interface to an external process via
    standard unix stdin/stdout pipe to extend ngspice functionality
    to the level of embedded systems.

    If a process ends with | character then rather than invoking
    a process it opens named pipe, process_in which is input to the
    process and pipe process_out for reading back from process.

    Communication between this code model and a process is in 8-bit
    binary format. On start-up

        0x01: version
        0x00-0xFF: number of inputs, max 255, 0 means none
        0x00-0xFF: number of outputs, max 255, 0 means none

    On start:

        outputs are set to uknown state and high impedance

    On each rising edge of a clock and reset being low

        double (8-byte): positive value of TIME if reset is low otherwise -TIME
        [inputs array ]: input bytes, each byte packs up to 8 inputs
        outputs are defined by returning process

    and process must return:

        [output array]: output bytes, each byte packs up to 8 outputs

    For example project please see: http://isotel.eu/mixedsim


MODIFICATIONS

    9 November 2017 Uros Platse <uros@isotel.eu>
        - Initial design, ready for use with projects

    4 April 2018 Uros Platse <uros@isotel.eu>
        - Tested and polished ready to be published

    7 April 2018 Uros Platse <uros@isotel.eu>
        Removed async reset and converted it to synchronous reset only.
        Code cleanup.

    30 September 2023 Brian Taylor
        Modify the code for Windows VisualC and Mingw.
        In VisualC, #pragma pack(...) compiler directives are needed for
        the struct declarations using __attribute__(packed).
        Use Microsoft CRT library functions _pipe, _read, _write, _spawn.
        For Windows VisualC and Mingw the pipes use binary mode, and
        named pipes or fifos are not supported.


REFERENCED FILES

    Inputs from and outputs to ARGS structure.

==============================================================================*/

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#if !defined(_MSC_VER)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#endif
#include <fcntl.h>

#define D_PROCESS_FORMAT_VERSION    0x01
#define DLEN(x)                     (uint8_t)( ((x)==0) ? 0 : (((x)-1)/8 + 1) )
#define DIN_SIZE_MAX                256     // also represents a theoretical maximum for 8-bit input length specifier

typedef unsigned char uint8_t;

typedef struct {
    int pipe_to_child;
    int pipe_from_child;
    int start_failed;
    int header_failed;
    int exchange_failed;
    uint8_t N_din, N_dout;          // number of inputs/outputs bytes
    Digital_State_t dout_old[256];  // max possible storage to track output changes
} Process_t;

#if defined(_MSC_VER) || defined(__MINGW64__)
#include <io.h>
static unsigned int w_start(char *system_command, const char *const *argv, Process_t * process);
#endif

static void report_sendheader_message(unsigned int msg_num)
{
    static char *messages[] = {
        "",
        "ERROR: d_process supports max 255 input and max 255 output signals",
        "ERROR: d_process when sending header",
        "ERROR: d_process didn't respond to the header",
        "ERROR: d_process returned invalid header version",
        "ERROR: d_process header I/O mismatch N_din and N_dout counts"
    };
    cm_message_send(messages[msg_num]);
}

static void report_exchangedata_message(unsigned int msg_num)
{
    static char *messages[] = {
        "",
        "ERROR: d_process when writing exchange data",
        "ERROR: d_process received invalid dout count when reading exchange data"
    };
    cm_message_send(messages[msg_num]);
}

static void report_start_message(unsigned int msg_num)
{
    static char *messages[] = {
        "",
        "ERROR: d_process process_file argument is invalid or not given",
        "ERROR: d_process failed to open pipe",
        "ERROR: d_process failed to fork or start process"
    };
    cm_message_send(messages[msg_num]);
}

static unsigned int sendheader(Process_t * process, int N_din, int N_dout)
{
#if defined(_MSC_VER)
#pragma pack(push, 1)
    struct header_s {
        uint8_t version, N_din, N_dout;
    } header = {D_PROCESS_FORMAT_VERSION, (uint8_t)N_din, (uint8_t)N_dout};
#pragma pack(pop)
#else
    struct header_s {
        uint8_t version, N_din, N_dout;
    } __attribute__((packed)) header = {D_PROCESS_FORMAT_VERSION, (uint8_t)N_din, (uint8_t)N_dout};
#endif

    if (N_din > 255 || N_dout > 255) {
        return 1;
    }

#if defined(_MSC_VER) || defined(__MINGW64__)
    if (_write(process->pipe_to_child, &header, sizeof(header)) == -1) {
#else
    if (write(process->pipe_to_child, &header, sizeof(header)) == -1) {
#endif
        return 2;
    }

    // Wait for echo which must return the same header to ack transfer
#if defined(_MSC_VER) || defined(__MINGW64__)
    if (_read(process->pipe_from_child, &header, sizeof(header)) != sizeof(header)) {
#else
    if (read(process->pipe_from_child, &header, sizeof(header)) != sizeof(header)) {
#endif
        return 3;
    }
    if (header.version != D_PROCESS_FORMAT_VERSION) {
        return 4;
    }
    if (header.N_din != N_din || header.N_dout != N_dout) {
        return 5;
    }

    process->N_din  = (uint8_t)DLEN(N_din);
    process->N_dout = (uint8_t)DLEN(N_dout);
    return 0;
}


static unsigned int dprocess_exchangedata(Process_t * process, double time, uint8_t din[], uint8_t dout[])
{
#if defined(_MSC_VER)
#pragma pack(push, 1)
    typedef struct {
        double time;
        uint8_t din[DIN_SIZE_MAX];
    } packet_t;
#pragma pack(pop)
#else
    typedef struct {
        double time;
        uint8_t din[DIN_SIZE_MAX];
    } __attribute__((packed)) packet_t;
#endif

    int wlen = 0;
    packet_t packet;
    packet.time = time;
    if (process->N_din > 0) {
        memcpy(packet.din, din, process->N_din);
    } else {
        packet.din[0] = 0;
    }

#if defined(_MSC_VER) || defined(__MINGW64__)
    wlen = _write(process->pipe_to_child, &packet, sizeof(double) + process->N_din);
#else
    wlen = write(process->pipe_to_child, &packet, sizeof(double) + process->N_din);
#endif
    if (wlen == -1) {
        return 1;
    }

#if defined(_MSC_VER) || defined(__MINGW64__)
    if (_read(process->pipe_from_child, dout, process->N_dout) != process->N_dout) {
#else
    if (read(process->pipe_from_child, dout, process->N_dout) != process->N_dout) {
#endif
        return 2;
    }
    return 0;
}


#if !defined(_MSC_VER) && !defined(__MINGW64__)
static unsigned int start(char *system_command, char * c_argv[], Process_t * process)
{
    int pipe_to_child[2];
    int pipe_from_child[2];
    int pid;
    size_t syscmd_len = strlen(system_command);

    if (syscmd_len == 0) {
        return 1;
    }
    if (system_command[syscmd_len-1] == '|') {
        char *filename_in = NULL, *filename_out = NULL;
        filename_in = (char *) malloc(syscmd_len + 5);
        filename_out = (char *) malloc(syscmd_len + 5);
        filename_in[0] = '\0';
        filename_out[0] = '\0';
        strncpy(filename_in, system_command, syscmd_len-1);
        strcpy(&filename_in[syscmd_len-1], "_in");
        strncpy(filename_out, system_command, syscmd_len-1);
        strcpy(&filename_out[syscmd_len-1], "_out");
        if ((process->pipe_to_child = open(filename_in, O_WRONLY)) == -1) {
            return 2;
        }
        if ((process->pipe_from_child = open(filename_out, O_RDONLY)) == -1) {
            return 2;
        }
        if (filename_in) free(filename_in);
        if (filename_out) free(filename_out);
    }
    else {
        if (pipe(pipe_to_child) || pipe(pipe_from_child)) {
            return 2;
        }
        if ((pid=fork()) ==-1) {
            return 3;
        }
        if (pid == 0) {
            dup2(pipe_to_child[0],0);
            dup2(pipe_from_child[1],1);
            close(pipe_to_child[1]);
            close(pipe_from_child[0]);

            if (execv(system_command, c_argv) == -1) {
                fprintf(stderr,
                    "ERROR: d_process failed to fork or start process %s\n",
                    system_command);
                return 3;
            }
        } else {
            process->pipe_to_child = pipe_to_child[1];
            process->pipe_from_child = pipe_from_child[0];
            close(pipe_to_child[0]);
            close(pipe_from_child[1]);
        }
    }
    return 0;
}
#endif


static void cm_d_process_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Process_t *proc = STATIC_VAR(process);
            if (proc) {
                free(proc);
                STATIC_VAR(process) = NULL;
            }
            break;
        }
    }
}

static int any_failed(Process_t *p)
{
    if (p->start_failed || p->header_failed || p->exchange_failed) {
        return 1;
    } else {
        return 0;
    }
}

void cm_d_process(ARGS)
{
    int                        i;   /* generic loop counter index */
    unsigned int          errnum;

    Digital_State_t       *reset,   /* storage for reset value  */
                      *reset_old;   /* previous reset value     */

    Digital_State_t         *clk,   /* storage for clock value  */
                        *clk_old;   /* previous clock value     */

    Process_t     *local_process;   /* Structure containing process vars */


    if (INIT) {
#define C_ARGV_SIZE 1024
        char * c_argv[C_ARGV_SIZE];
        int c_argc = 1;
        cm_event_alloc(0,sizeof(Digital_State_t));
        cm_event_alloc(1,sizeof(Digital_State_t));

        clk   = clk_old   = (Digital_State_t *) cm_event_get_ptr(0,0);
        reset = reset_old = (Digital_State_t *) cm_event_get_ptr(1,0);

        STATIC_VAR(process) = calloc(1, sizeof(Process_t));
        local_process       = STATIC_VAR(process);
        CALLBACK = cm_d_process_callback;

        if (!PARAM_NULL(process_params)) {
            int upper_limit;
            if (PARAM_SIZE(process_params) > (C_ARGV_SIZE - 2)) {
                upper_limit = C_ARGV_SIZE - 2;
            } else {
                upper_limit = PARAM_SIZE(process_params);
            }
            for (i=0; i<upper_limit; i++) {
                c_argv[c_argc++] = PARAM(process_params[i]);
            }
        }
        c_argv[0]      = PARAM(process_file);
        c_argv[c_argc] = NULL;
#undef C_ARGV_SIZE

#if defined(_MSC_VER) || defined(__MINGW64__)
        errnum = w_start(c_argv[0], (const char *const *)c_argv, local_process);
#else
        errnum = start(c_argv[0], c_argv, local_process);
#endif
        if (errnum) {
            report_start_message(errnum);
            local_process->start_failed = 1;
            return;
        }
        if (any_failed(local_process)) return;
        errnum = sendheader(local_process, PORT_SIZE(in), PORT_SIZE(out));
        if (errnum) {
            report_sendheader_message(errnum);
            local_process->header_failed = 1;
            return;
        }
        if (any_failed(local_process)) return;

        for (i=0; i<PORT_SIZE(in); i++) {
            LOAD(in[i]) = PARAM(input_load);
        }

        LOAD(clk) = PARAM(clk_load);

        if ( !PORT_NULL(reset) ) {
            LOAD(reset) = PARAM(reset_load);
        }
    }
    else {
        local_process = STATIC_VAR(process);
        if (any_failed(local_process)) {
            raise(SIGINT);
        }
        clk = (Digital_State_t *) cm_event_get_ptr(0,0);
        clk_old = (Digital_State_t *) cm_event_get_ptr(0,1);

        reset = (Digital_State_t *) cm_event_get_ptr(1,0);
        reset_old = (Digital_State_t *) cm_event_get_ptr(1,1);
    }


    if ( 0.0 == TIME ) {    /****** DC analysis...output w/o delays ******/
        for (i=0; i<PORT_SIZE(out); i++) {
            local_process->dout_old[i] = UNKNOWN;
            OUTPUT_STATE(out[i])       = UNKNOWN;
            OUTPUT_STRENGTH(out[i])    = HI_IMPEDANCE;
        }
    }
    else {                  /****** Transient Analysis ******/
        *clk = INPUT_STATE(clk);

        if ( PORT_NULL(reset) ) {
            *reset = *reset_old = ZERO;
        }
        else {
            *reset = INPUT_STATE(reset);
        }

        if (*clk != *clk_old && ONE == *clk) {
            uint8_t *dout = NULL, *din = NULL;
            uint8_t b;
            dout = (uint8_t *)malloc(local_process->N_dout * sizeof(uint8_t));
            if (local_process->N_din > 0) {
                din = (uint8_t *)malloc(local_process->N_din * sizeof(uint8_t));
                memset(din, 0, local_process->N_din);
            }

            for (i=0; i<PORT_SIZE(in); i++) {
                switch(INPUT_STATE(in[i])) {
                    case ZERO: b = 0; break;
                    case ONE:  b = 1; break;
#if defined(_MSC_VER) || defined(__MINGW64__)
                    default:   b = rand() & 1; break;
#else
                    default:   b = random() & 1; break;
#endif
                }
                din[i >> 3] |= (uint8_t)(b << (i & 7));
            }

            errnum = dprocess_exchangedata(local_process, (ONE != *reset) ? TIME : -TIME, din, dout);
            if (errnum) {
                report_exchangedata_message(errnum);
                local_process->exchange_failed = 1;
                return;
            }

            for (i=0; i<PORT_SIZE(out); i++) {
                Digital_State_t new_state = ((dout[i >> 3] >> (i & 7)) & 0x01) ? ONE : ZERO;

                if (new_state != local_process->dout_old[i]) {
                    OUTPUT_STATE(out[i])    = new_state;
                    OUTPUT_STRENGTH(out[i]) = STRONG;
                    OUTPUT_DELAY(out[i])    = PARAM(clk_delay);
                    local_process->dout_old[i] = new_state;
                }
                else {
                    OUTPUT_CHANGED(out[i]) = FALSE;
                }
            }
            if (din) free(din);
            if (dout) free(dout);
        }
        else {
            for (i=0; i<PORT_SIZE(out); i++) {
                OUTPUT_CHANGED(out[i]) = FALSE;
            }
        }
    }
}

#if defined(_MSC_VER) || defined(__MINGW64__)

#include <process.h>
#include <io.h>

static unsigned int w_start(char *system_command, const char *const *argv, Process_t * process)
{
    int pipe_to_child[2];
    int pipe_from_child[2];
    intptr_t pid = 0;
    int mode = _O_BINARY;
    size_t syscmd_len = strlen(system_command);

    if (syscmd_len == 0 || system_command[syscmd_len-1] == '|') {
        return 1;
    }
    if (_pipe(pipe_to_child, 1024, mode) == -1) {
        return 2;
    }
    if (_pipe(pipe_from_child, 1024, mode) == -1) {
        return 2;
    }

    _dup2(pipe_to_child[0],0);
    _dup2(pipe_from_child[1],1);
    _close(pipe_to_child[0]);
    _close(pipe_from_child[1]);

    _flushall();
    pid = _spawnvp(_P_NOWAIT, system_command, argv);
    if (pid == -1) {
        return 3;
    }

    process->pipe_to_child = pipe_to_child[1];
    process->pipe_from_child = pipe_from_child[0];
    return 0;
}
#endif
