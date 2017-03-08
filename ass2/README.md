# Advanced Tpoics in ML
# Benoit Gaujac
# UCL

# Set up
Three folders codes, data and trained_models.
- codes folder contains the scripts for each parts, in subfolders part1, part2 and part3.
- data contains the data.
- trained_models contains the weights for the trained models. 3 subfolders part1, part2 and part3 corresponding to each parts of the assignment. In each of these subfolders, trained weights are ordered by models in subsubfolders named after the models. Moreover, for parts 2 and 3, the subsubfolders contain also the resulting in-painting images. In the subfolders, we also find a Perf folder containing the training and testing performances of the models.
 

#### part1 ####:
# Usage:
part1 [-m <flag>] [-s <flag>]
Options:
-m <flag> --model <flag>:  a flag for choosing the model in [“lstm1l32u”, “lstm1l64u”, “lstm1l128u”, “lstm3l32u”, “gru1l32u”, “gru1l64u”, “gru1l128u”, “gru3l32u”]
-s <flag> --mode <flag>:  a flag for running mode in ["train", "test”]

#### part2 ####:
# Usage:
part2 [-m <flag>] [-s <flag>]
Options:
-m <flag> --model <flag>:  a flag for choosing the model in [“gru1l32u”, “gru1l64u”, “gru1l128u”, “gru3l32u”]
-s <flag> --mode <flag>:  a flag for running mode in ["train", "test", “inpainting”]

#### part3 ####:
# Usage:
part3 [-d <flag>]
Options:
-d <flag> —data <flag>:  a flag for the dataset to use in [“1x1”, “2x2”]
