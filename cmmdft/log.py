#!/usr/bin/env python

from __future__ import print_function, absolute_import, unicode_literals
from io import IOBase
import os, sys, datetime, getpass, atexit
from pathlib import Path
import numpy, scipy, matplotlib


__all__ = ['version', 'log']

version = '0.1'

header = """
********************************************************************************

                            Welcome to CmmDFT
    a Python package for computing density profiles using classical DFT

                               Written by
            Louis Vanduyfhuys(1) and Vic De Ridder(1)* and Steven Vandenbrande(1)

         (1) Center for Molecular Modeling, Ghent University Belgium.
                   * mailto: Vic.DeRidder@UGent.be

********************************************************************************
"""

footer = "********************************************************************************"

def splitstring(string, length, separators=[' ','/','_']):
    result = []
    remainder = string
    while len(remainder)>length:
        i=length-1
        while remainder[i] not in separators and i>0:
            i -= 1
        result.append(remainder[:i+1])
        remainder = remainder[i+1:]
    if len(remainder)>0:
        result.append(remainder)
    return result


class Section(object):
    def __init__(self, logger, label, level, timer_description):
        self.logger = logger
        self.old_label = logger.label
        self.new_label = label
        self.old_level = self.logger.section_level
        self.new_level = level
        self.timer_description = timer_description

    def __enter__(self):
        if self.new_label!=self.old_label and self.logger.log_level>0:
            self.logger.add_blank_line = True
        self.logger.label = self.new_label
        self.logger.section_level = self.new_level
        if self.timer_description is not None:
            self.begin = datetime.datetime.now()
            self.end = None

    def __exit__(self, type, value, traceback):
        if self.new_label!=self.old_label and self.logger.log_level>0:
            self.logger.add_blank_line = True
        self.logger.label = self.old_label
        self.logger.section_level = self.old_level
        if self.timer_description is not None:
            self.end = datetime.datetime.now()
            for i, (description, time) in enumerate(self.logger.timetable):
                if description==self.timer_description:
                    self.logger.timetable[i][1] = time + self.end-self.begin
                    return
            self.logger.timetable.append([self.timer_description, self.end-self.begin])


class Logger(object):
    def __init__(self, level, _f=sys.stdout, max_label_length=9, line_length=100):
        self.set_level(level)
        self._f = _f
        self.mll = max_label_length
        self.ll = line_length
        self._active = False
        self.label = 'QFF'
        self.add_blank_line = False
        self.timetable = []
        self.warnings = []
        self.second_log = False
        self.dr = ''
        self.f2 = None

    def set_level(self, level):
        if isinstance(level, int):
            if level>=0 and level<=4:
                self.log_level = level
            else:
                raise ValueError('Integer level should be between 0 and 4 (boundaries included).')
        elif isinstance(level, str):
            allowed = ['silent', 'low', 'medium', 'high', 'highest']
            if level.lower() in allowed:
                self.log_level = allowed.index(level.lower())
            else:
                raise ValueError('String level should be silent, low, medium, high or highest.')
        self.section_level = None

    def write_to_file(self, dr=None, f=None, second_log=False, overwrite=True):
        """
        dr: the directory in which the logfile is stored
        f: the filename of the logfile
        """
        assert dr is not None, 'Must provide a directory in which to store the logfile, can be an empty string'
        self.dr = dr
        if not isinstance(dr, os.PathLike):
            dr = Path(dr)

        if f is None:
            f =  sys.stdout
        elif f is not None:
            f = dr / f
            self.f2 = f
        self.second_log = second_log

        if overwrite:
            mode = 'w'
        else:
            mode = 'a'
        
        if second_log:
            if isinstance(f, str) or isinstance(f, os.PathLike):
                self._f2 = open(f, mode)
            elif isinstance(f, IOBase):
                self._f2 = f  
            else:      
                raise ValueError('File argument f should be a string representing a filename or a File instance')
        else:
            if isinstance(f, str) or isinstance(f, os.PathLike):
                self._f = open(f, mode)
            elif isinstance(f, IOBase):
                self._f = f        
            else:
                raise ValueError('File argument f should be a string representing a filename or a File instance')            

    def section(self, label, level, timer=None):
        '''
            Construct a section instance for use in with statements to control
            section printing and timing.
        '''
        return Section(self, label, level, timer)

    def dump(self, message, new_line=True, write=True):
        if self.section_level<=self.log_level:
            if not self._active:
                self._active = True
                self.print_header()
                self.start_time = datetime.datetime.now()
            assert self.label is not None
            if new_line and self.add_blank_line:
                print('', file=self._f)
                if self.second_log:
                    print('', file=self._f2)
                self.add_blank_line = False
            line = ''
            for piece in splitstring(message, self.ll-self.mll):
                line += ' ' + self.label[:self.mll-2] + ' '
                line += ' '*(self.mll-2 - len(self.label[:self.mll-2]))
                line += piece
                line += '\n'
            line = line.rstrip('\n')
            # self._f.write(line)
            print(line, file=self._f)
            if self.second_log:
                # self._f2.write(line)
                print(line, file=self._f2)
            if write:
                if self.second_log:
                    self._f2.close()
                self.write_to_file(self.dr, self.f2, self.second_log, overwrite=False)



    def warning(self, message, new_line=True, label_section=None):
        '''
            Warnings are printed whenever log_level is higher than 0, i.e. not
            in silent mode.
        '''
        if self.log_level>0:
            if not self._active:
                self._active = True
                self.print_header()
            assert self.label is not None
            if new_line and self.add_blank_line:
                print('', file=self._f)
                self.add_blank_line = False
            line = ''
            for piece in splitstring('WARNING: '+message, self.ll-self.mll):
                line += ' ' + self.label[:self.mll-2] + ' '
                line += ' '*(self.mll-2 - len(self.label[:self.mll-2]))
                line += piece #.encode('utf-8')
                line += '\n'
            line = line.rstrip('\n')
            if label_section is None:
                self.warnings.append((self.label, message))
            else:
                self.warnings.append((label_section, message))
            print(line, file=self._f)
            if self.second_log:
                print(line, file=self._f2)


    def print_header(self):
        if self.log_level>0:
            print(header, file=self._f)
            print('', file=self._f)
            if self.second_log:
                print(header, file=self._f2)
                print('', file=self._f2)

        mll = self.mll
        self.mll = 20
        with self.section('USER', 1): self.dump(getpass.getuser(), new_line=False)
        with self.section('MACHINE', 1): self.dump(' '.join(os.uname()), new_line=False)
        with self.section('TIME', 1): self.dump(datetime.datetime.now().isoformat().replace('T', ' '), new_line=False)
        with self.section('CMMDFT VERSION', 1): self.dump(version.replace('\n', ''), new_line=False)
        with self.section('PYTHON VERSION', 1): self.dump(sys.version.replace('\n', ''), new_line=False)
        with self.section('NUMPY VERSION', 1): self.dump(numpy.__version__, new_line=False)
        with self.section('SCIPY VERSION', 1): self.dump(scipy.__version__, new_line=False)
        with self.section('MATPLOTLIB VERSION', 1): self.dump(matplotlib.__version__, new_line=False)
        with self.section('CURRENT DIR', 1): self.dump(os.getcwd(), new_line=False)
        with self.section('COMMAND LINE', 1): self.dump(' '.join(sys.argv), new_line=False)
        self.mll = mll
        if self.log_level>0:
            print('', file=self._f)
            print('~'*80, file=self._f)
            if self.second_log:
                print('', file=self._f2)
                print('~'*80, file=self._f2)


    def exit(self):
        if self._active:
            self.end_time = datetime.datetime.now()
            self.print_timetable()
            self.print_warnings()
            self.print_footer()
        self.close()

    def print_footer(self):
        if self.log_level>0:
            print(footer, file=self._f)
            if self.second_log:
                print(footer, file=self._f2)


    def print_timetable(self):
        if self.log_level>0:
            print('', file=self._f)
            print('~'*80, file=self._f)
            if self.second_log:
                print('', file=self._f2)
                print('~'*80, file=self._f2)

        with self.section('TIMING', 1):
            for label, time in self.timetable:
                line = '%30s  ' %(label+' '*(30-len(label)))
                line += str(time)
                self.dump(line)
            line = '%30s  ' %('TOTAL TIME' +' '*(30-len('TOTAL TIME')))
            line += str(self.end_time-self.start_time)
            self.dump(line)

    def print_warnings(self):
        if len(self.warnings):
            print('', file=self._f)
            print('~'*80, file=self._f)
            with self.section('WARNINGS', 1):
                self.dump('%i NUMBER OF WARNINGS'%len(self.warnings))
                i = 0
                for label, message in self.warnings:
                    i += 1
                    line = 'WARNING NR %i             encountered in %s  ' %(i, label)
                    self.dump(line)
                    self.dump(message)

    def close(self):
        if isinstance(self._f, IOBase):
            self._f.close()
            if self.second_log:
                self._f2.close()

log = Logger('medium')
atexit.register(log.exit)
