#!/bin/sh
# WishFix \
	exec wish -f "$0" ${1+"$@"}
###

package require spice

spice::source example.cir
spice::step 100

spice::plot a0 vs b0
spice::bltplot a0
