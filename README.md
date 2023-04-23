# Image-Classification-with-OpenCV
Ex1 from Set: https://neuroinflab.wordpress.com/tasks/

I attached two versions of program solving Task1:
- ex1.py is a purely sequential program, which is slower but easier to run
You run it: "python ex1.py <first_border> <second_border> <cut_off> <activation> <path_to_the_folder_with_images>"
(on some OSes "python3" instead of "python").
All arguments are required.
- ex1mpi.py is a partly paralelized version of the program with usage of MPI
To run it, install MPI first (the method depends on operating system).
Then You run program by command: "mpiexec -n <number_of_threads> python ex1mpi.py <first_border> <second_border> <cut_off> <activation> <path_to_the_folder_with_images>"
(I tested Linux and Windows, idk about Mac)
argument number_of_threads should be small natural number (I tested n = 1-8)
other arguments as earlier
During running, ex1mpi.py will display some spam in the console, but soon will successfully finish with proper results.
On my PC ex1mpi.py works ~15% faster than ex1.py
Suggested values of parameters:
first_border: ~280
second_border: ~460
cut_off: 48
activation: ~1000
n: 8
So, suggested way to run ex1.py is "python ex1.py 280 460 48 1000 <path_to_the_folder_with_images>"
and for ex1mpi.py "mpiexec -n 8 python ex1mpi.py 280 460 48 1000 <path_to_the_folder_with_images>"
