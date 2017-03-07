# Advanced Tpoics in ML
# Benoit Gaujac
# UCL

# Set up
Three folders part1, part2 and part3 corresponding to different task of assignment.

#### part1 ####:
Need to create manually part1/Perf folder in part1 repository.
# Usage:
part1 [-m <flag>] [-s <flag>]
Options:
-m <flag> --model <flag>:  a flag for choosing the model in [“lstm1l32u”, “lstm1l64u”, “lstm1l128u”, “lstm3l32u”, “gru1l32u”, “gru1l64u”, “gru1l128u”, “gru3l32u”]
-s <flag> --mode <flag>:  a flag for running mode in ["train", "test”]

#### part2 ####:
Need to create manually part2/Perf folder in part2 repository.
# Usage:
part2 [-m <flag>] [-s <flag>]
Options:
-m <flag> --model <flag>:  a flag for choosing the model in [“gru1l32u”, “gru1l64u”, “gru1l128u”, “gru3l32u”]
-s <flag> --mode <flag>:  a flag for running mode in ["train", "test", “inpainting”]

#### part3 ####:
Need to create manually part3/Perf folder in part2 repository and to create and save datasets in part3/missing_pixels repo. 
# Usage:
part3 [-d <flag>]
Options:
-d <flag> —data <flag>:  a flag for the dataset to use in [“1x1”, “2x2”]
