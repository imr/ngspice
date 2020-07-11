
#
#   ngspice DC, AC and noise test routines
#

#
#  Rel  Date            Who             Comments
# ====  ==========      =============   ========
#  1.1  07/05/17        Dietmar Warning Version detection included
#  1.0  05/13/11        Dietmar Warning Initial version
#

package simulate;
if (defined($main::simulatorCommand)) {
    $simulatorCommand=$main::simulatorCommand;
} else {
    $simulatorCommand="ngspice";
}
$netlistFile="ngspiceCkt";
use strict;

sub version {
    my($version,$vaVersion);
    $version="unknown";
    $vaVersion="unknown";
    if (!open(OF,">$simulate::netlistFile")) {
        die("ERROR: cannot open file $simulate::netlistFile, stopped");
    }
    print OF "version test";
    print OF "r1 1 0 1";
    print OF "v1 1 0 1";
    print OF ".control";
    print OF "version";
    print OF ".endc";
    print OF ".end";
    close(OF);
    if (!open(SIMULATE,"$simulate::simulatorCommand < $simulate::netlistFile 2>/dev/null|")) {
        die("ERROR: cannot run $simulate::simulatorCommand, stopped");
    }
    while (<SIMULATE>) {
        chomp;
        # Check whether this line is the one we are looking for.
        # Also the term in parenthesis "()" stores the number to $1 which we can reuse later.
        if (m/.+ngspice-([0-9]+)/) {
            # Simple read the stored group from the matching in the if clause
            $version=$1;
            last;
        }
    }
    close(SIMULATE);
    return($version,$vaVersion);
}

sub runNoiseTest {
    my($variant,$outputFile)=@_;
    my($arg,$name,$value,$type,$pin,$noisePin);
    my(@BiasList,$i,@Field);
    my(@X,@Noise,$temperature,$biasVoltage,$sweepVoltage,$inData);

#
#   Make up the netlist, using a subckt to encapsulate the
#   instance. This simplifies handling of the variants as
#   the actual instance is driven by voltage-controlled
#   voltage sources from the subckt pins, and the currents
#   are fed back to the subckt pins using current-controlled
#   current sources. Pin swapping, polarity reversal, and
#   m-factor scaling can all be handled by simple modifications
#   of this subckt.
#

    @X=();@Noise=();
    $noisePin=$main::Outputs[0];
    if ($main::fMin == $main::fMax) {
        $main::frequencySpec="lin 0 $main::fMin ".(10*$main::fMin); # spice3f5 bug workaround
    }
    foreach $temperature (@main::Temperature) {
        foreach $biasVoltage (split(/\s+/,$main::biasListSpec)) {
            if ($main::fMin == $main::fMax) {
                push(@X,@main::BiasSweepList);
            }
            foreach $sweepVoltage (@main::BiasSweepList) {
                if (!open(OF,">$simulate::netlistFile")) {
                    die("ERROR: cannot open file $simulate::netlistFile, stopped");
                }
                print OF "* Noise simulation for $main::simulatorName";
                &generateCommonNetlistInfo($variant,$temperature);
                print OF "vin dummy 0 0 ac 1";
                print OF "rin dummy 0 1.0 noise=0";
                foreach $pin (@main::Pin) {
                    if ($main::isFloatingPin{$pin}) {
                        print OF "i_$pin $pin 0 0";
                    } elsif ($pin eq $main::biasListPin) {
                        print OF "v_$pin $pin 0 $biasVoltage";
                    } elsif ($pin eq $main::biasSweepPin) {
                        print OF "v_$pin $pin 0 $sweepVoltage";
                    } else {
                        print OF "v_$pin $pin 0 $main::BiasFor{$pin}";
                    }
                }
                print OF "x1 ".join(" ",@main::Pin)." mysub";
                print OF "hn 0 n_$noisePin v_$noisePin 1";
                print OF ".control";
                print OF "set sqrnoise";
                print OF ".endc";
                print OF ".noise v(n_$noisePin) vin $main::frequencySpec";
                print OF ".print noise all";
                print OF ".end";
                close(OF);
        
#
#   Run simulations and get the results
#

                if (!open(SIMULATE,"$simulate::simulatorCommand < $simulate::netlistFile 2>/dev/null|")) {
                    die("ERROR: cannot run $simulate::simulatorCommand, stopped");
                }
                $inData=0;
                while (<SIMULATE>) {
                    chomp;s/^\s+//;s/\s+$//;s/,/ /g;
                    if (/Index\s+frequency\s+inoise_spectrum\s+onoise_spectrum/i) {
                        $inData=1;<SIMULATE>;next;
                    }
                    @Field=split;
                    if (/\*/ || ($#Field != 3)) {$inData=0}
                    next if (!$inData);
                    if ($main::fMin == $main::fMax) {
                        push(@Noise,1*$Field[3]);$inData=0;next; # spice3f5 bug workaround
                    }
                    push(@X,1*$Field[1]);
                    push(@Noise,1*$Field[3]);
                }
                close(SIMULATE);
            }
        }
    }

#
#   Write the results to a file
#

    if (!open(OF,">$outputFile")) {
        die("ERROR: cannot open file $outputFile, stopped");
    }
    if ($main::fMin == $main::fMax) {
        printf OF ("V($main::biasSweepPin)");
    } else {
        printf OF ("Freq");
    }
    foreach (@main::Outputs) {
        printf OF (" N($_)");
    }
    printf OF ("\n");
    for ($i=0;$i<=$#X;++$i) {
        if (defined($Noise[$i])) {printf OF ("$X[$i] $Noise[$i]\n")}
    }
    close(OF);

#
#   Clean up, unless the debug flag was specified
#

    if (! $main::debug) {
        unlink($simulate::netlistFile);
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
    }
}

sub runAcTest {
    my($variant,$outputFile)=@_;
    my($arg,$name,$value,$type,$pin,$mPin,$fPin,%NextPin);
    my(@BiasList,$acStim,$i,@Field);
    my(@X,$omega,$twoPi,%g,%c,$temperature,$biasVoltage,$sweepVoltage,$inData,$outputLine);
    $twoPi=8.0*atan2(1.0,1.0);

#
#   Make up the netlist, using a subckt to encapsulate the
#   instance. This simplifies handling of the variants as
#   the actual instance is driven by voltage-controlled
#   voltage sources from the subckt pins, and the currents
#   are fed back to the subckt pins using current-controlled
#   current sources. Pin swapping, polarity reversal, and
#   m-factor scaling can all be handled by simple modifications
#   of this subckt.
#

    foreach $mPin (@main::Pin) {
        foreach $fPin (@main::Pin) {
            @{$g{$mPin,$fPin}}=();
            @{$c{$mPin,$fPin}}=();
        }
    }
    @X=();
    foreach $temperature (@main::Temperature) {
        foreach $biasVoltage (split(/\s+/,$main::biasListSpec)) {
            if ($main::fMin == $main::fMax) {
                push(@X,@main::BiasSweepList);
            }
            foreach $sweepVoltage (@main::BiasSweepList) {
                if (!open(OF,">$simulate::netlistFile")) {
                    die("ERROR: cannot open file $simulate::netlistFile, stopped");
                }
                print OF "* AC simulation for $main::simulatorName";
                &generateCommonNetlistInfo($variant,$temperature);
                foreach $fPin (@main::Pin) {
                    foreach $mPin (@main::Pin) {
                        if ($mPin eq $fPin) {
                            $acStim=" ac 1";
                        } else {
                            $acStim="";
                        }
                        if ($main::isFloatingPin{$mPin}) {
                            print OF "i_${mPin}_$fPin ${mPin}_$fPin 0 0";
                        } elsif ($mPin eq $main::biasListPin) {
                            print OF "v_${mPin}_$fPin ${mPin}_$fPin 0 $biasVoltage$acStim";
                        } elsif ($mPin eq $main::biasSweepPin) {
                            print OF "v_${mPin}_$fPin ${mPin}_$fPin 0 $sweepVoltage$acStim";
                        } else {
                            print OF "v_${mPin}_$fPin ${mPin}_$fPin 0 $main::BiasFor{$mPin}$acStim";
                        }
                    }
                    print OF "x_$fPin ".join("_$fPin ",@main::Pin)."_$fPin mysub";
                }
                print OF ".ac $main::frequencySpec";
                foreach $mPin (@main::Pin) {
                    foreach $fPin (@main::Pin) {
                        print OF ".print ac i(v_${mPin}_$fPin)";
                    }
                }
                print OF ".end";
                close(OF);
        
#
#   Run simulations and get the results
#

                if (!open(SIMULATE,"$simulate::simulatorCommand < $simulate::netlistFile 2>/dev/null|")) {
                    die("ERROR: cannot run $simulate::simulatorCommand, stopped");
                }
                $inData=0;
                while (<SIMULATE>) {
                    chomp;s/^\s+//;s/\s+$//;s/,/ /g;
                    if (/^Index\s+frequency\s+v_([a-zA-Z][a-zA-Z0-9]*)_([a-zA-Z][a-zA-Z0-9]*)#branch/i) {
                        $mPin=$1;$fPin=$2;<SIMULATE>;$inData=1;next;
                    }
                    @Field=split;
                    if (/^\*/ || ($#Field != 3)) {$inData=0;}
                    next if (!$inData);
                    if (($main::fMin != $main::fMax) && ($mPin eq $fPin) && ($mPin eq $main::Pin[0])) {
                        push(@X,1*$Field[1]);
                    }
                    push(@{$g{$mPin,$fPin}},$Field[2]);
                    $omega=$twoPi*$Field[1];
                    if ($mPin eq $fPin) {
                        push(@{$c{$mPin,$fPin}},$Field[3]/$omega);
                    } else {
                        push(@{$c{$mPin,$fPin}},-1*$Field[3]/$omega);
                    }
                }
                close(SIMULATE);
            }
        }
    }

#
#   Write the results to a file
#

    if (!open(OF,">$outputFile")) {
        die("ERROR: cannot open file $outputFile, stopped");
    }
    if ($main::fMin == $main::fMax) {
        printf OF ("V($main::biasSweepPin)");
    } else {
        printf OF ("Freq");
    }
    foreach (@main::Outputs) {
        ($type,$mPin,$fPin)=split(/\s+/,$_);
        printf OF (" $type($mPin,$fPin)");
    }
    printf OF ("\n");
    for ($i=0;$i<=$#X;++$i) {
        $outputLine="$X[$i]";
        foreach (@main::Outputs) {
            ($type,$mPin,$fPin)=split(/\s+/,$_);
            if ($type eq "g") {
                if (defined(${$g{$mPin,$fPin}}[$i])) {
                    $outputLine.=" ${$g{$mPin,$fPin}}[$i]";
                } else {
                    undef($outputLine);last;
                }
            } else {
                if (defined(${$c{$mPin,$fPin}}[$i])) {
                    $outputLine.=" ${$c{$mPin,$fPin}}[$i]";
                } else {
                    undef($outputLine);last;
                }
            }
        }
        if (defined($outputLine)) {printf OF ("$outputLine\n")}
    }
    close(OF);

#
#   Clean up, unless the debug flag was specified
#

    if (! $main::debug) {
        unlink($simulate::netlistFile);
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
    }
}

sub runDcTest {
    my($variant,$outputFile)=@_;
    my($arg,$name,$value,$i,@Field,$pin);
    my($start,$stop,$step);
    my(@V,%DC,$temperature,$biasVoltage);
    my($inData,$inResults);

#
#   Make up the netlist, using a subckt to encapsulate the
#   instance. This simplifies handling of the variants as
#   the actual instance is driven by voltage-controlled
#   voltage sources from the subckt pins, and the currents
#   are fed back to the subckt pins using current-controlled
#   current sources. Pin swapping, polarity reversal, and
#   m-factor scaling can all be handled by simple modifications
#   of this subckt.
#

    @V=();
    foreach $pin (@main::Outputs) {@{$DC{$pin}}=()}
    ($start,$stop,$step)=split(/\s+/,$main::biasSweepSpec);
    $start-=$step;
    foreach $temperature (@main::Temperature) {
        foreach $biasVoltage (split(/\s+/,$main::biasListSpec)) {
            if (!open(OF,">$simulate::netlistFile")) {
                die("ERROR: cannot open file $simulate::netlistFile, stopped");
            }
            print OF "* DC simulation for $main::simulatorName";
            &generateCommonNetlistInfo($variant,$temperature);
            foreach $pin (@main::Pin) {
                if ($main::isFloatingPin{$pin}) {
                    print OF "i_$pin $pin 0 0";
                } elsif ($pin eq $main::biasListPin) {
                    print OF "v_$pin $pin 0 $biasVoltage";
                } elsif ($pin eq $main::biasSweepPin) {
                    print OF "v_$pin $pin 0 $start";
                } else {
                    print OF "v_$pin $pin 0 $main::BiasFor{$pin}";
                }
            }
            print OF "x1 ".join(" ",@main::Pin)." mysub";
            print OF ".dc v_$main::biasSweepPin $main::biasSweepSpec";
            foreach $pin (@main::Outputs) {
                if ($main::isFloatingPin{$pin}) {
                    print OF ".print dc v($pin)";
                } else {
                    print OF ".print dc i(v_$pin)";
                }
            }
            print OF ".end";
            close(OF);
        
#
#   Run simulations and get the results
#

            if (!open(SIMULATE,"$simulate::simulatorCommand < $simulate::netlistFile 2>/dev/null|")) {
                die("ERROR: cannot run $simulate::simulatorCommand, stopped");
            }
            $inResults=0;
            while (<SIMULATE>) {
                chomp;s/^\s+//;s/\s+$//;s/#branch//;s/\(/_/;s/\)//;
                if (/^Index\s+v-sweep\s+v_/i) {$inResults=1;($pin=$');<SIMULATE>;next}
                @Field=split;
                if ($#Field != 2) {$inResults=0}
                next if (!$inResults);
                if ($pin eq $main::Outputs[0]) {
                    push(@V,$Field[1]);
                }
                push(@{$DC{$pin}},$Field[2]);
            }
            close(SIMULATE);
        }
    }

#
#   Write the results to a file
#

    if (!open(OF,">$outputFile")) {
        die("ERROR: cannot open file $outputFile, stopped");
    }
    printf OF ("V($main::biasSweepPin)");
    foreach $pin (@main::Outputs) {
        if ($main::isFloatingPin{$pin}) {
            printf OF (" V($pin)");
        } else {
            printf OF (" I($pin)");
        }
    }
    printf OF ("\n");
    for ($i=0;$i<=$#V;++$i) {
        next if (abs($V[$i]-$start) < abs(0.1*$step)); # this is dummy first bias point
        printf OF ("$V[$i]");
        foreach $pin (@main::Outputs) {printf OF (" ${$DC{$pin}}[$i]")}
        printf OF ("\n");
    }
    close(OF);

#
#   Clean up, unless the debug flag was specified
#

    if (! $main::debug) {
        unlink($simulate::netlistFile);
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
    }
}

sub generateCommonNetlistInfo {
    my($variant,$temperature)=@_;
    my(@Pin_x,$arg,$name,$value,$eFactor,$fFactor,$pin);
    foreach $pin (@main::Pin) {push(@Pin_x,"${pin}_x")}
    print OF ".options temp=$temperature gmin=1e-13 abstol=1e-13 reltol=1e-3";
    if ($variant=~/^scale$/) {
        die("ERROR: there is no scale or shrink option for ngspice, stopped");
    }
    if ($variant=~/^shrink$/) {
        die("ERROR: there is no scale or shrink option for ngspice, stopped");
    }
    if ($variant=~/_P/) {
        $eFactor=-1;$fFactor=1;
    } else {
        $eFactor=1;$fFactor=-1;
    }
    if ($variant=~/^m$/) {
        if ($main::outputNoise) {
            $fFactor/=sqrt($main::mFactor);
        } else {
            $fFactor/=$main::mFactor;
        }
    }
    if (defined($main::verilogaFile)) {
        die("ERROR: Verilog-A model support is not implemented for ngspice, stopped");
    }
    print OF ".subckt mysub ".join(" ",@Pin_x);
    foreach $pin (@main::Pin) {
        if ($main::isFloatingPin{$pin}) { # assumed "dt" thermal pin, no scaling sign change
            print OF "v_$pin ${pin} ${pin}_x 0";
        } elsif ($variant=~/^Flip/ && defined($main::flipPin{$pin})) {
            print OF "e_$pin ${pin}_v 0 $main::flipPin{$pin}_x 0 $eFactor";
            print OF "v_$pin ${pin}_v ${pin} 0";
            print OF "f_$pin $main::flipPin{$pin}_x 0 v_$pin   $fFactor";
        } else {
            print OF "e_$pin ${pin}_v 0 ${pin}_x 0 $eFactor";
            print OF "v_$pin ${pin}_v ${pin} 0";
            print OF "f_$pin ${pin}_x 0 v_$pin   $fFactor";
        }
    }
    print OF "${main::keyLetter}1 ".join(" ",@main::Pin)." mymodel";
    foreach $arg (@main::InstanceParameters) {
        ($name,$value)=split(/=/,$arg);
        if ($variant=~/^scale$/) {
            if ($main::isLinearScale{$name}) {
                $value/=$main::scaleFactor;
            } elsif ($main::isAreaScale{$name}) {
                $value/=$main::scaleFactor**2;
            }
        }
        if ($variant=~/^shrink$/) {
            if ($main::isLinearScale{$name}) {
                $value/=(1.0-$main::shrinkPercent*0.01);
            } elsif ($main::isAreaScale{$name}) {
                $value/=(1.0-$main::shrinkPercent*0.01)**2;
            }
        }
        print OF "+ $name=$value";
    }
    if ($variant eq "m") {
        print OF "+ m=$main::mFactor";
    }
    if ($variant=~/_P/) {
        print OF ".model mymodel $main::pTypeSelectionArguments";
    } else {
        print OF ".model mymodel $main::nTypeSelectionArguments";
    }
    foreach $arg (@main::ModelParameters) {
        print OF "+ $arg";
    }
    print OF ".ends";
}

1;
