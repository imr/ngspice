
#
#   hspice DC, AC and noise test routines
#

#
#  Rel  Date            Who             Comments
# ====  ==========      =============   ========
#  1.4  04/06/11        Geoffrey Coram  Fixed version detection; fixed ac-freq result printing
#  1.3  06/21/07        Rob Jones       Verilog-A model support added
#                                       HSPICE version detection updated
#  1.2  06/30/06        Colin McAndrew/ Floating node support added
#                       Rob Jones       Noise simulation added
#                                       HSPICE version detection improved
#  1.0  04/13/06        Colin McAndrew  Initial version
#

package simulate;
if (defined($main::simulatorCommand)) {
    $simulatorCommand=$main::simulatorCommand;
} else {
    $simulatorCommand="hspice";
}
$netlistFile="hspiceCkt";
use strict;

sub version {
    my($version,$vaVersion);
    $version="unknown";
    $vaVersion="unknown";
    if (!open(OF,">$simulate::netlistFile")) {
        die("ERROR: cannot open file $simulate::netlistFile, stopped");
    }
    print OF "";
    print OF "r1 1 0 1";
    print OF "v1 1 0 1";
    print OF ".op";
    print OF ".end";
    close(OF);
    if (!open(SIMULATE,"$simulate::simulatorCommand $simulate::netlistFile 2>/dev/null|")) {
        die("ERROR: cannot run $main::simulatorName, stopped");
    }
    while (<SIMULATE>) {
        chomp;
        if (s/.+HSPICE\s+-*\s*//) {
            ($version=$_)=~s/\s+.*//;
            last;
        }
    }
    close(SIMULATE);
    if (! $main::debug) {
        unlink($simulate::netlistFile);
        unlink("$simulate::netlistFile.st0");
        if (defined($main::verilogaFile)) {
            unlink("$simulate::netlistFile.val");
        }
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
        unlink("hspice.errors");
        unlink("simout.tmp");
    }
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
                print OF "rin dummy 0 rmod";
                print OF ".model rmod r res=1 noise=0";
                foreach $pin (@main::Pin) {
                    if ($main::isFloatingPin{$pin}) {
                        print OF "i_$pin $pin 0 0";
                    } elsif ($pin eq $main::biasListPin) {
                        if (defined($main::referencePinFor{$pin})) {
                            print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $biasVoltage";
                            print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
                        } else {
                            print OF "v_$pin $pin 0 $biasVoltage";
                        }
                    } elsif ($pin eq $main::biasSweepPin) {
                        if (defined($main::referencePinFor{$pin})) {
                            print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $sweepVoltage";
                            print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
                        } else {
                            print OF "v_$pin $pin 0 $sweepVoltage";
                        }
                    } else {
                        if (defined($main::referencePinFor{$pin})) {
                            print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $main::BiasFor{$pin}";
                            print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
                        } else {
                            print OF "v_${pin} ${pin} 0 $main::BiasFor{$pin}";
                        }
                    }
                }
                print OF "x1 ".join(" ",@main::Pin)." mysub";
                if ($main::outputNoise == 2) {
                    print OF ".ac $main::frequencySpec";
                    print OF ".noise v($noisePin,$main::Outputs[1]) vin";
                } else {
                    if (! $main::isFloatingPin{$noisePin}) {
                        print OF "fn 0 n_$noisePin v_$noisePin 1";
                        print OF "rn 0 n_$noisePin rmod";
                    }
                    print OF ".ac $main::frequencySpec";
                    if ($main::isFloatingPin{$noisePin}) {
                        print OF ".noise v($noisePin) vin";
                    } else {
                        print OF ".noise v(n_$noisePin) vin";
                    }
                }
                print OF ".print noise onoise";
                print OF ".end";
                close(OF);
        
#
#   Run simulations and get the results
#

                if (!open(SIMULATE,"$simulate::simulatorCommand $simulate::netlistFile 2>/dev/null|")) {
                    die("ERROR: cannot run $main::simulatorName, stopped");
                }
                $inData=0;
                while (<SIMULATE>) {
                    chomp;s/^\s+//;s/\s+$//;@Field=split;
                    if (/freq\s+onoise/i) {$inData=1;<SIMULATE>;next}
                    if ($#Field != 1) {$inData=0;}
                    next if (!$inData);
                    if ($main::fMin != $main::fMax) {
                        push(@X,&modelQa::unScale($Field[0]));
                    }
                    push(@Noise,(&modelQa::unScale($Field[1]))**2);
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
    if ($main::outputNoise == 2) {
        print OF (" N($noisePin,$main::Outputs[1])");
    } else {
        print OF (" N($noisePin)");
    }
    for ($i=0;$i<=$#X;++$i) {
        if (defined($Noise[$i])) {printf OF ("$X[$i] $Noise[$i]\n")}
    }
    close(OF);

#
#   Clean up, unless the debug flag was specified
#

    if (! $main::debug) {
        unlink($simulate::netlistFile);
        unlink("$simulate::netlistFile.st0");
        if (defined($main::verilogaFile)) {
            unlink("$simulate::netlistFile.val");
        }
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
    }
}

sub runAcTest {
    my($variant,$outputFile)=@_;
    my($arg,$name,$value,$type,$pin,$mPin,$fPin,%NextPin,%PrevPin,$first_fPin);
    my(@BiasList,$i,$j,@Field);
    my(@X,$omega,%g,%c,%q,$twoPi,$temperature,$biasVoltage,$sweepVoltage);
    my($inData,$inResults,$outputLine);
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
        if ($main::needAcStimulusFor{$mPin}) {
            $first_fPin=$mPin;
            last;
        }
    }
    if (!open(OF,">$simulate::netlistFile")) {
        die("ERROR: cannot open file $simulate::netlistFile, stopped");
    }
    print OF "* AC simulation for $main::simulatorName";
    &generateCommonNetlistInfo($variant,$main::Temperature[0]);
    @BiasList=split(/\s+/,$main::biasListSpec);
    print OF ".param vbias=$BiasList[0]";
    print OF ".param vsweep=$main::BiasSweepList[0]";
    foreach $pin (@main::Pin) {
        if ($pin eq $first_fPin) {
            print OF ".param ac_$pin=1";
        } else {
            print OF ".param ac_$pin=0";
        }
        if ($main::isFloatingPin{$pin}) {
            print OF "i_$pin $pin 0 0";
        } elsif ($pin eq $main::biasListPin) {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} vbias ac ac_$pin";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 vbias ac ac_$pin";
            }
        } elsif ($pin eq $main::biasSweepPin) {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} vsweep ac ac_$pin";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 vsweep ac ac_$pin";
            }
        } else {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $main::BiasFor{$pin} ac ac_$pin";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 $main::BiasFor{$pin} ac ac_$pin";
            }
        }
    }
    print OF "x1 ".join(" ",@main::Pin)." mysub";
    print OF ".ac $main::frequencySpec";
    foreach $pin (@main::Pin) {print OF ".print ac ir(v_$pin) ii(v_$pin)"}
    for ($i=0;$i<=$#main::Pin;++$i) {
        next if (!$main::needAcStimulusFor{$main::Pin[$i]});
        $j=$i;
        while (1) {
            --$j;
            $j=$#main::Pin if ($j < 0);
            if ($main::needAcStimulusFor{$main::Pin[$j]}) {
                $PrevPin{$main::Pin[$i]}=$main::Pin[$j];
                last;
            }
        }
    }
    foreach $temperature (@main::Temperature) {
        foreach $biasVoltage (@BiasList) {
            foreach $sweepVoltage (@main::BiasSweepList) {
                foreach $pin (@main::Pin) {
                    next if (!$main::needAcStimulusFor{$pin});
                    next if ($temperature == $main::Temperature[0] && $biasVoltage == $BiasList[0]
                             && $sweepVoltage == $main::BiasSweepList[0] && $pin eq $first_fPin);
                    print OF ".alter";
                    if ($biasVoltage == $BiasList[0] && $sweepVoltage == $main::BiasSweepList[0] && $pin eq $first_fPin) {
                        print OF ".temp $temperature";
                    }
                    if ($sweepVoltage == $main::BiasSweepList[0] && $pin eq $main::Pin[0]) {
                        print OF ".param vbias=$biasVoltage";
                    }
                    if ($pin eq $first_fPin) {
                        print OF ".param vsweep=$sweepVoltage";
                    }
                    print OF ".param ac_$PrevPin{$pin}=0";
                    print OF ".param ac_$pin=1";
                }
            }
        }
    }
    print OF ".end";
    close(OF);

#
#   Run simulations and get the results
#

    foreach $mPin (@main::Pin) {
        foreach $fPin (@main::Pin) {
            @{$g{$mPin,$fPin}}=();
            @{$c{$mPin,$fPin}}=();
            @{$q{$mPin,$fPin}}=();
        }
    }
    for ($i=0;$i<=$#main::Pin;++$i) {
        next if (!$main::needAcStimulusFor{$main::Pin[$i]});
        $j=$i;
        while (1) {
            ++$j;
            $j=0 if ($j > $#main::Pin);
            if ($main::needAcStimulusFor{$main::Pin[$j]}) {
                $NextPin{$main::Pin[$i]}=$main::Pin[$j];
                last;
            }
        }
    }
    if (!open(SIMULATE,"$simulate::simulatorCommand $simulate::netlistFile 2>/dev/null|")) {
        die("ERROR: cannot run $main::simulatorName, stopped");
    }
    $inData=0;$inResults=0;
    if ($main::fMin == $main::fMax) {
        @X=();
        foreach $temperature (@main::Temperature) {
            foreach $biasVoltage (split(/\s+/,$main::biasListSpec)) {
                push(@X,@main::BiasSweepList);
            }
        }
    }
    $fPin=$first_fPin;
    while (<SIMULATE>) {
        chomp;
        if (/ac\s+analysis/i && /temp=/i) {$inResults=1;$inData=0;next}
        if (/info/i && /job\s+concluded/) {$inResults=0;$fPin=$NextPin{$fPin}}
        next if (!$inResults);
        s/^\s+//;s/\s+$//;
        if (/^v_([a-zA-z][a-zA-Z0-9]*)/) {$mPin=$1;$inData=1;next;}
        @Field=split;
        if ($#Field != 2
            || &modelQa::unScale($Field[0]) !~ /^($main::number)$/
            || &modelQa::unScale($Field[1]) !~ /^($main::number)$/
            || &modelQa::unScale($Field[2]) !~ /^($main::number)$/) {
            $inData=0;
            next;
        }
        next if (! $inData);
        if (($main::fMin != $main::fMax) && ($fPin eq $first_fPin)) {
            push(@X,&modelQa::unScale($Field[0]));
        }
        $omega=$twoPi*&modelQa::unScale($Field[0]);
        push(@{$g{$mPin,$fPin}},&modelQa::unScale($Field[1]));
        if ($mPin eq $fPin) {
            push(@{$c{$mPin,$fPin}},&modelQa::unScale($Field[2])/$omega);
        } else {
            push(@{$c{$mPin,$fPin}},-1*&modelQa::unScale($Field[2])/$omega);
        }
        if (abs(&modelQa::unScale($Field[1])) > 1.0e-99) {
            push(@{$q{$mPin,$fPin}},&modelQa::unScale($Field[2])/&modelQa::unScale($Field[1]));
        } else {
            push(@{$q{$mPin,$fPin}},1.0e99);
        }
    }
    close(SIMULATE);

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
            } elsif ($type eq "c") {
                if (defined(${$c{$mPin,$fPin}}[$i])) {
                    $outputLine.=" ${$c{$mPin,$fPin}}[$i]";
                } else {
                    undef($outputLine);last;
                }
            } else {
                if (defined(${$q{$mPin,$fPin}}[$i])) {
                    $outputLine.=" ${$q{$mPin,$fPin}}[$i]";
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
        unlink("$simulate::netlistFile.st0");
        if (defined($main::verilogaFile)) {
            unlink("$simulate::netlistFile.val");
        }
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
        unlink("hspice.errors");
        unlink("simout.tmp");
    }
}

sub runDcTest {
    my($variant,$outputFile)=@_;
    my($arg,$name,$value,$i,$pin,@Field);
    my(@BiasList,$start,$stop,$step);
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

    if (!open(OF,">$simulate::netlistFile")) {
        die("ERROR: cannot open file $simulate::netlistFile, stopped");
    }
    print OF "* DC simulation for $main::simulatorName";
    &generateCommonNetlistInfo($variant,$main::Temperature[0]);
    @BiasList=split(/\s+/,$main::biasListSpec);
    ($start,$stop,$step)=split(/\s+/,$main::biasSweepSpec);
    $start-=$step;
    print OF ".param vbias=$BiasList[0]";
    foreach $pin (@main::Pin) {
        if ($main::isFloatingPin{$pin}) {
            print OF "i_$pin $pin 0 0";
        } elsif ($pin eq $main::biasListPin) {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} vbias";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 vbias";
            }
        } elsif ($pin eq $main::biasSweepPin) {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $start";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 $start";
            }
        } else {
            if (defined($main::referencePinFor{$pin})) {
                print OF "v_${pin} ${pin} ${pin}_$main::referencePinFor{$pin} $main::BiasFor{$pin}";
                print OF "e_${pin} ${pin}_$main::referencePinFor{$pin} 0 $main::referencePinFor{$pin} 0 1";
            } else {
                print OF "v_${pin} ${pin} 0 $main::BiasFor{$pin}";
            }
        }
    }
    print OF "x1 ".join(" ",@main::Pin)." mysub";
    print OF ".dc v_$main::biasSweepPin $main::biasSweepSpec";
    foreach $pin (@main::Outputs) {
        if ($main::isFloatingPin{$pin}) {
            print OF ".print v($pin)"
        } else {
            print OF ".print i(v_$pin)"
        }
    }
    foreach $temperature (@main::Temperature) {
        foreach $biasVoltage (@BiasList) {
            next if ($temperature == $main::Temperature[0] && $biasVoltage == $BiasList[0]);
            print OF ".alter";
            if ($biasVoltage == $BiasList[0]) {
                print OF ".temp $temperature";
            }
            print OF ".param vbias=$biasVoltage";
        }
    }
    print OF ".end";
    close(OF);

#
#   Run simulations and get the results
#

    if (!open(SIMULATE,"$simulate::simulatorCommand $simulate::netlistFile 2>/dev/null|")) {
        die("ERROR: cannot run $main::simulatorName, stopped");
    }
    $inData=0;$inResults=0;@V=();
    foreach $pin (@main::Outputs) {@{$DC{$pin}}=()}
    while (<SIMULATE>) {
        chomp;
        if (/dc\s+transfer\s+curves/i) {$inResults=1;$inData=0;next}
        if (/job\s+concluded/) {$inResults=0}
        next if (!$inResults);
        s/^\s+//;s/\s+$//;
        if (/^volt\s+(current|voltage)/) {
            chomp($_=<SIMULATE>);s/^\s*(v_)?//;s/\s+$//;$pin=$_;$inData=1;next;
        }
        @Field=split;
        if ($#Field != 1
            || &modelQa::unScale($Field[0]) !~ /^($main::number)$/
            || &modelQa::unScale($Field[1]) !~ /^($main::number)$/) {
            $inData=0;next;
        }
        if ($pin eq $main::Outputs[0]) {
            push(@V,&modelQa::unScale($Field[0]));
        }
        push(@{$DC{$pin}},&modelQa::unScale($Field[1]));
    }
    close(SIMULATE);

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
        foreach $pin (@main::Outputs) {
            printf OF (" ${$DC{$pin}}[$i]")
        }
        printf OF ("\n");
    }
    close(OF);

#
#   Clean up, unless the debug flag was specified
#

    if (! $main::debug) {
        unlink($simulate::netlistFile);
        unlink("$simulate::netlistFile.st0");
        if (defined($main::verilogaFile)) {
            unlink("$simulate::netlistFile.val");
        }
        if (!opendir(DIRQA,".")) {
            die("ERROR: cannot open directory ., stopped");
        }
        foreach (grep(/^$simulate::netlistFile\.ic/,readdir(DIRQA))) {unlink($_)}
        closedir(DIRQA);
        unlink("hspice.errors");
        unlink("simout.tmp");
    }
}

sub generateCommonNetlistInfo {
    my($variant,$temperature) = @_;
    my(@Pin_x,$arg,$name,$value,$eFactor,$fFactor,$pin);
    print OF ".option tnom=27 reltol=1u vntol=1n abstol=1f"; # default for HSPICE is 25
    print OF ".temp $temperature";
    if ($variant=~/^scale$/) {
        print OF ".option scale=$main::scaleFactor";
    }
    if ($variant=~/^shrink$/) {
        print OF ".option scale=".(1.0-$main::shrinkPercent*0.01);
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
        print OF ".hdl \"$main::verilogaFile\"";
    }
    foreach $pin (@main::Pin) {push(@Pin_x,"${pin}_x")}
    print OF ".subckt mysub ".join(" ",@Pin_x);
    foreach $pin (@main::Pin) {
        if ($main::isFloatingPin{$pin}) {
            if ($main::outputNoise && $pin eq $main::Outputs[0]) {
                if ($variant =~ /^m$/) {
                    $eFactor = sqrt($main::mFactor);
                } else {
                    $eFactor = 1;
                }
                print OF "e_$pin ${pin}_x 0 ${pin} 0 $eFactor";
            } else { # assumed "dt" thermal pin, no scaling sign change
                print OF "v_$pin ${pin} ${pin}_x 0";
            }
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
    if (defined($main::verilogaFile)) {
        if ($variant=~/_P/) {
            print OF "${main::keyLetter}1 ".join(" ",@main::Pin)." $main::pTypeSelectionArguments";
        } else {
            print OF "${main::keyLetter}1 ".join(" ",@main::Pin)." $main::nTypeSelectionArguments";
        }
        if ($variant=~/^scale$/) {
            print OF "+ scale=$main::scaleFactor";
        }
        if ($variant=~/^shrink$/) {
            print OF "+ shrink=$main::shrinkPercent";
        }
    } else {
        print OF "${main::keyLetter}1 ".join(" ",@main::Pin)." mymodel";
    }
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
    if (!defined($main::verilogaFile)) {
        if ($variant=~/_P/) {
            print OF ".model mymodel $main::pTypeSelectionArguments";
        } else {
            print OF ".model mymodel $main::nTypeSelectionArguments";
        }
    }
    foreach $arg (@main::ModelParameters) {
        print OF "+ $arg";
    }
    print OF ".ends";
}

1;
