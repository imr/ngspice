/* NG-SPICE -- An electrical circuit simulator
 *
 * Copyright (c) 2000 Arno W. Peters
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation without fee, and without a written agreement is
 * hereby granted, provided that the above copyright notice, this
 * paragraph and the following three paragraphs appear in all copies.
 *
 * This software program and documentation are copyrighted by their
 * authors. The software program and documentation are supplied "as
 * is", without any accompanying services from the authors. The
 * authors do not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the
 * program was developed for research purposes and is advised not to
 * rely exclusively on the program for any reason.
 * 
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. THE AUTHORS SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
 * SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE AUTHORS
 * HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 * ENHANCEMENTS, OR MODIFICATIONS. */

#ifndef _DEVLIST_H
#define _DEVLIST_H

SPICEdev **first_device(void);
SPICEdev **next_device(SPICEdev **current);

#endif
