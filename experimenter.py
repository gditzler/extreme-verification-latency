#!/usr/bin/env python 

# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os 
import pandas as pd
from utils import config_parser
from models import APT, ComposeV1, ComposeV2, FastCompose, LevelIW, MClassification, Scargc

# run the main program 
if __name__ == '__main__': 
    parser = config_parser()
    args = parser.parse_args()

    # make the result directory if it does not exist 
    if not os.path.isdir('results/'): 
        os.mkdir('results/')

    # load the config file and make sure it exists. 
    try: 
        df = pd.read_csv(args.config)
    except: 
        raise(FileExistsError('Config file %s does not exist' % args.config))
