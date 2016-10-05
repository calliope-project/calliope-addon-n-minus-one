"""
Copyright (C) 2016 Christoph Thormeyer.
Licensed under the Apache 2.0 License.

run.py
~~~~~~~

Execute overall calculation with respective parameters.

"""
from functions import *

t_start = time.time()
clean_temp(os.path.abspath(''))
full_error = 0
if full_error == 0:
    try:
        a = complete_analysis(os.path.abspath('') + '/example_model/',
                              new_run=True, new_copies='parallel',
                              new_plots=True)  # , mode='check')
        print(get_time(t_start))
        print(a)
    except Exception as e:
        print(str(e))
else:
    complete_analysis(os.path.abspath('') + '/example_model/', new_run=True,
                      new_copies='parallel', new_plots=True, mode='check')
    print(get_time(t_start))
