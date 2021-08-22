import matplotlib.pyplot as pt
import numpy as np
import math
import os
import sys


def get_units_of_type(type, dtypes_d):
    units = ''
    if len(type) >= 1:
        dunit = dtypes_d.get(type)
        if dunit is not None:
            units = dunit
        elif type == 'voltage':
            units = 'V'
        elif type == 'current':
            units = 'A'
        else:
            units = ''
    return units


def get_name_type_units(index, vdefs_l, dtypes_d):
    name = ''
    type = ''
    units = ''
    dunit = ''
    for v in vdefs_l:
        if v[0] == index:
            name = v[1]
            type = v[2]
            break
    if len(name) >= 1 and len(type) >= 1:
        units = get_units_of_type(type, dtypes_d)
    return name, type, units


def get_name_units(index, vdefs_l, dtypes_d):
    name, type, units = get_name_type_units(index, vdefs_l, dtypes_d)
    value = ''
    if len(name) >= 1:
        value = name
    if len(units) >= 1:
        value = value + ' ' + units
    return value


def get_index_of_name(nm, vdefs_l):
    for v in vdefs_l:
        if nm == v[1]:
            return v[0]
    return -1


def get_zvar_wanted(vdefs_l, dtypes_d, twod=True):
    count = 0
    for v in vdefs_l:
        unit = get_units_of_type(v[2], dtypes_d)
        count = count + 1
        print('  %3d  %6s   %s  %s' % (int(v[0]), v[1], v[2], unit))
    if twod:
        lownum = 2
    else:
        lownum = 1
    hinum = count - 1
    msg = 'Enter variable number in the range >=' + str(lownum) \
        + ' and <=' + str(hinum) + ': '
    try:
        vnum = int(input(msg))
        if vnum < lownum or vnum > hinum:
            print('Variable number', vnum, 'out of range')
            return -1
        else:
            return vnum
    except Exception:
        print('What\'s that?')
        return -1


def show_line(lnum, line):
    print('line#', lnum, ' -> ', line)


def yn_input(question):
    q = question + '(y/n) '
    reply = str(input(q))
    if reply == 'y' or reply == 'Y':
        return True
    else:
        return False


def do_plots_1d(fig, title, xinfo, yinfo, xlist, ylist, yabs, ylog):
    if yabs:
        yinfo = 'ABS ' + yinfo
    if ylog:
        yinfo = 'LOG ' + yinfo
    pt.figure(fig)
    pt.plot(xlist, ylist, marker='.')
    pt.title(title)
    pt.xlabel(xinfo)
    pt.ylabel(yinfo)
    pt.grid()
    pt.show(block=False)
    return 0


def do_plots_2d(fnum, fill, xinfo, yinfo, zinfo, xl, yl, zarr):
    header = zinfo.split('\n')
    if len(header) == 2:
        print('Plotting', '(%s)' % header[1],
              'Axes', '(%s) vs (%s)' % (xinfo, yinfo))
    if fill:
        fig, ax = pt.subplots()
        c1 = pt.contourf(xl, yl, zarr)
        fig.colorbar(c1)
    else:
        pt.figure(fnum)
        c1 = pt.contour(xl, yl, zarr)
        pt.clabel(c1, fmt='%3.2e', fontsize=8.0, inline=False)

    pt.title(zinfo)
    pt.xlabel(xinfo)
    pt.ylabel(yinfo)
    pt.grid()
    pt.show(block=False)
    return 0


def list_of_magnitudes(vdefs_l):
    exindex = get_index_of_name('ex', vdefs_l)
    eyindex = get_index_of_name('ey', vdefs_l)
    jdxindex = get_index_of_name('jdx', vdefs_l)
    jdyindex = get_index_of_name('jdy', vdefs_l)
    jnxindex = get_index_of_name('jnx', vdefs_l)
    jnyindex = get_index_of_name('jny', vdefs_l)
    jpxindex = get_index_of_name('jpx', vdefs_l)
    jpyindex = get_index_of_name('jpy', vdefs_l)
    mag_l = []
    if exindex != -1 and eyindex != -1:
        mag = [['e', int(exindex), int(eyindex)]]
        mag_l += mag
    if jdxindex != -1 and jdyindex != -1:
        mag = [['jd', int(jdxindex), int(jdyindex)]]
        mag_l += mag
    if jnxindex != -1 and jnyindex != -1:
        mag = [['jn', int(jnxindex), int(jnyindex)]]
        mag_l += mag
    if jpxindex != -1 and jpyindex != -1:
        mag = [['jp', int(jpxindex), int(jpyindex)]]
        mag_l += mag
    return mag_l


def choose_magnitude(mag_l):
    choices = ''
    xindex = -1
    yindex = -1
    the_chosen_one = ''
    if len(mag_l) < 1:
        return -1, -1, ''
    for item in mag_l:
        choices = choices + '  ' + item[0]
    print(choices)
    chosen = str(input('Which one? : '))
    for item in mag_l:
        if chosen == item[0]:
            the_chosen_one = item[0]
            xindex = item[1]
            yindex = item[2]
            break
    if xindex == -1 or yindex == -1:
        print('You chose badly!')
        return -1, -1, ''
    else:
        print('You chose well')
    return xindex, yindex, the_chosen_one


def interactive_2d(xl, yl, xdim, ydim, zarr, xinfo, yinfo, title,
                   vdefs_l, dtypes_d):
    one_more = True
    fig = 1
    mags = list_of_magnitudes(vdefs_l)
    # Start plotting loop
    while one_more:
        plot_mag = False
        xindex = -1
        yindex = -1
        plot_mag = yn_input('Do you want to plot a magnitude? ')
        if plot_mag:
            xindex, yindex, choice = choose_magnitude(mags)
            if xindex == -1 or yindex == -1:
                continue
            mname, mtype, munits \
                = get_name_type_units(xindex, vdefs_l, dtypes_d)
            zinfo = title + '\nMagnitude of vector ' + choice + ' ' + munits
            fillit = yn_input('Fill the contours? ')
            zz = np.ndarray((ydim, xdim))
            for i in range(len(xl)):
                for j in range(len(yl)):
                    xval = zarr[j][i][xindex - 2]
                    yval = zarr[j][i][yindex - 2]
                    zz[j][i] = math.sqrt((xval * xval) + (yval * yval))
            do_plots_2d(fig, fillit, xinfo, yinfo, zinfo, xl, yl, zz)
            fig = fig + 1
        else:
            which_var = get_zvar_wanted(vdefs_l, dtypes_d)
            if which_var != -1:
                zinfo = get_name_units(which_var, vdefs_l, dtypes_d)
                zinfo = title + '\n' + zinfo
                fillit = yn_input('Fill the contours? ')
                zz = np.ndarray((ydim, xdim))
                for i in range(len(xl)):
                    for j in range(len(yl)):
                        zz[j][i] = zarr[j][i][which_var - 2]
                do_plots_2d(fig, fillit, xinfo, yinfo, zinfo, xl, yl, zz)
            fig = fig + 1

        one_more = yn_input('Do another plot? ')
    # End of plotting loop
    return 0


def interactive_1d(xl, zarr, xinfo, title, vdefs_l, dtypes_d):
    one_more = True
    fig = 1
    # Start plotting loop
    while one_more:
        yabs = False
        ylog = False
        which_var = get_zvar_wanted(vdefs_l, dtypes_d, twod=False)
        if which_var != -1:
            yinfo = get_name_units(which_var, vdefs_l, dtypes_d)
            yabs = yn_input('Abs y values ')
            ylog = yn_input('Log y values ')
            ylist = []
            for i in range(len(xl)):
                val = float(zarr[i][which_var - 1])
                if yabs:
                    val = math.fabs(val)
                if ylog and not val > 0.0:
                    ylog = False
                ylist = ylist + [val]
            if len(ylist) != len(xl):
                return 1
            if ylog:
                for i in range(len(ylist)):
                    ylist[i] = math.log(ylist[i], 10)
            do_plots_1d(fig, title, xinfo, yinfo, xl, ylist, yabs, ylog)
        fig = fig + 1
        one_more = yn_input('Do another plot? ')
    # End of plotting loop
    return 0


def inpdata(infile, title_wanted, plots=False, microns=True):
    if title_wanted < 1:
        print('title_wanted too small', title_wanted)
        return 1
    line_num = 0
    found_values = False
    numvars = 0
    numpoints = 0
    xdim = -1
    ydim = -1
    curr_var = 0
    xlist = []
    ylist = []
    title_num = 0
    title_matched = False
    title_str = ''
    found_variables = False
    vardefs = []
    dtypes = {}
    xvar_info = ''
    yvar_info = ''
    xvalue_next = False
    zarray = np.ndarray((1, 1, 1))
    z1darray = np.ndarray((1, 1))
    convert_meters_to_microns = False
    is_2d = False

    # Start processing plot data
    for line in infile:
        line_num = line_num + 1
        a = line.split()
        if len(a) < 1:
            continue
        if a[0] == 'Title:':
            title_num = title_num + 1
            found_values = False
            numvars = 0
            numpoints = 0
            curr_var = 0
            vardefs = []
            dtypes = {}
            xdim = -1
            ydim = -1
            curr_x = -1
            curr_y = -1
            prev_x = -1
            xvalue_next = False
            xlist = []
            ylist = []
            zarray = np.ndarray((1, 1, 1))
            z1darray = np.ndarray((1, 1))
            convert_meters_to_microns = False
            is_2d = False

            title_matched = (title_num == title_wanted)
            if title_matched:
                title_str = ' '.join(a)
        elif title_matched:
            if found_values:
                if is_2d:
                    # Store the 2D values
                    if len(a) == 3:
                        curr_y = int(a[0])
                        curr_x = int(a[1])
                        if len(ylist) < ydim:
                            # y value
                            if microns and convert_meters_to_microns:
                                yvalue = [1.0e6 * float(a[2])]
                            else:
                                yvalue = [float(a[2])]
                            ylist += yvalue
                        curr_var = 0
                        if curr_y == 1 and curr_x == 1:
                            yvar_info = get_name_units(0, vardefs, dtypes)
                        xvalue_next = True
                    elif len(a) == 1:
                        curr_var = curr_var + 1
                        if xvalue_next:
                            if curr_var != 1:
                                show_line(line_num, a)
                                print('x values out of order')
                                return 1
                            xvalue_next = False
                            if len(xlist) < xdim and curr_x != prev_x:
                                if microns and convert_meters_to_microns:
                                    xvalue = [1.0e6 * float(a[0])]
                                else:
                                    xvalue = [float(a[0])]
                                xlist += xvalue
                            prev_x = curr_x
                            if curr_y == 1 and curr_x == 1:
                                xvar_info = get_name_units(1, vardefs, dtypes)
                        else:
                            zarray[curr_y-1][curr_x-1][curr_var-2] \
                                = float(a[0])
                    # End of current 2D value
                else:
                    # Store the 1D values
                    if len(a) == 2:
                        curr_x = int(a[0])
                        if len(xlist) < numpoints:
                            # x value
                            if microns and convert_meters_to_microns:
                                xvalue = [1.0e6 * float(a[1])]
                            else:
                                xvalue = [float(a[1])]
                            xlist += xvalue
                        curr_var = 0
                        if curr_x == 0:
                            xvar_info = get_name_units(0, vardefs, dtypes)
                    elif len(a) == 1:
                        z1darray[curr_x][curr_var] = float(a[0])
                        curr_var = curr_var + 1
                    # End of current 1D value
            elif len(a) == 2 and a[0] == 'Dimensions:':
                # This precedes 2D data
                dims = a[1].split(',')
                if len(dims) != 2:
                    show_line(line_num, a)
                    print('Must be 2D')
                    return 1
                xdim = int(dims[0])
                ydim = int(dims[1])
                if xdim < 1 or ydim < 1:
                    show_line(line_num, a)
                    return 1
                if numvars <= 0:
                    print('No. Variables: has not been found yet')
                    return 1
                zarray = np.ndarray((ydim, xdim, numvars - 2))
                is_2d = True
            elif len(a) == 1 and a[0] == 'Variables:':
                found_variables = True
            elif len(a) == 1 and a[0] == 'Values:':
                found_values = True
                found_variables = False
                if not is_2d:
                    if numvars <= 0:
                        print('No. Variables: has not been found yet')
                        return 1
                    if numpoints <= 0:
                        print('No. Points: has not been found yet')
                        return 1
                    z1darray = np.ndarray((numpoints, numvars - 1))
            elif found_variables and len(a) == 3:
                vdef = [[int(a[0]), a[1], a[2]]]
                if is_2d:
                    if int(a[0]) == 0:
                        if a[1] != 'y' or a[2] != 'distance':
                            show_line(line_num, a)
                            print('var 0 is not y distance')
                            print(vdef)
                            return 1
                    elif int(a[0]) == 1:
                        if a[1] != 'x' or a[2] != 'distance':
                            show_line(line_num, a)
                            print('var 1 is not x distance')
                            print(vdef)
                            return 1
                else:
                    if int(a[0]) == 0:
                        if a[1] != 'x' or a[2] != 'distance':
                            show_line(line_num, a)
                            print('var 0 is not x distance')
                            print(vdef)
                            return 1
                vardefs += vdef
            elif len(a) == 3 and a[0] == 'No.' and a[1] == 'Variables:':
                numvars = int(a[2])
            elif len(a) == 3 and a[0] == 'No.' and a[1] == 'Points:':
                numpoints = int(a[2])
            elif len(a) == 5:
                if a[0] == 'Command:' and a[1] == 'deftype' and a[2] == 'v':
                    if microns and a[3] == 'distance' and a[4] == 'm':
                        dt = {a[3]: 'microns'}
                        convert_meters_to_microns = True
                    else:
                        dt = {a[3]: a[4]}
                    dtypes.update(dt)
    # End of processing plot data

    if plots and title_matched and is_2d:
        interactive_2d(xlist, ylist, xdim, ydim, zarray, xvar_info,
                       yvar_info, title_str, vardefs, dtypes)
    elif plots and title_matched and not is_2d:
        print('NOTE this Cider save file is for a 1D model.')
        print('Instead you could use the ngspice commands:')
        print('  load <file_name>    followed by    plot <variable>\n')
        interactive_1d(xlist, z1darray, xvar_info, title_str, vardefs, dtypes)

    return 0


def main():
    titlen = 2
    plotit = True
    run_convert = True
    if run_convert:
        prog = os.getenv('CIDER_CONVERTER')
        if prog is None:
            prog = 'ciderconvert'
        arguments = sys.argv[1:]
        if len(arguments) == 1:
            cmd = prog + ' ' + arguments[0]
        else:
            fname = str(input('Cider save file name? '))
            if len(fname) > 0:
                cmd = prog + ' ' + fname
            else:
                return 1
        infile = os.popen(cmd, mode='r')
        inpdata(infile, titlen, plots=plotit)
        infile.close()
    else:
        fname = str(input('Cider save file name? '))
        if len(fname) > 0:
            infile = open(fname, 'r')
            inpdata(infile, titlen, plots=plotit)
            infile.close()
    return 0


if __name__ == '__main__':
    main()
