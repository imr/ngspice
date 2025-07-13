/*
 * msline_common.h - common definitions for microstrip devices
 *
 * Copyright (C) 2025 Vadim Kuznetsov <ra3xdh@gmail.com>
 * Based on works by Stefan Jahn, Qucsator project
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this package; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 * $Id$
 *
 */


#ifndef MSLINE_COMMON_H
#define MSLINE_COMMON_H

//TRAN model
#define TRAN_DC 0
#define TRAN_FULL 1

// MS line model
#define HAMMERSTAD 0
#define KIRSCHING 1
#define WHEELER 2
#define SCHNEIDER 3


// Dispersion model
#define DISP_KIRSCHING 0
#define KOBAYASHI 1
#define YAMASHITA 2
#define DISP_HAMMERSTAD 3
#define GETSINGER 4
#define DISP_SCHNEIDER 5
#define PRAMANICK 6

void Hammerstad_ab (double, double,
                    double*, double*);
void Hammerstad_er (double, double, double,
                    double, double*);
void Hammerstad_zl (double, double*);
void Getsinger_disp (double, double, double,
                     double, double,
                     double*, double*);
void Kirschning_er (double, double, double,
                    double, double*);
void Kirschning_zl (double, double, double,
                    double, double, double,
                    double*, double*);

void mslineAnalyseQuasiStatic (double W, double h, double t,
		double er, int Model,
		double *ZlEff, double *ErEff,
		double *WEff);

void mslineAnalyseDispersion (double W, double h, double er,
		double ZlEff, double ErEff,
		double frequency, int Model,
		double* ZlEffFreq,
		double* ErEffFreq);

void analyseLoss (double, double, double, double,
                         double, double, double, double,
                         double, double, int,
                         double*, double*);

#endif
