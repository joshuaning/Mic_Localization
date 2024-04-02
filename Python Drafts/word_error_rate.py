import whisper
import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between a reference string and a hypothesis string.
    
    :param reference: The reference sentence as a list of words.
    :param hypothesis: The hypothesis sentence as a list of words.
    :return: The Word Error Rate (WER) score.
    """
    # Split sentences into words
    reference = reference.split()
    hypothesis = hypothesis.split()

    # Create a matrix for the dynamic programming algorithm
    len_ref = len(reference)
    len_hyp = len(hypothesis)
    dp_matrix = [[0 for j in range(len_hyp + 1)] for i in range(len_ref + 1)]

    # Initialize the matrix, this covers the cases of all insertions and all deletions
    for i in range(1, len_ref + 1):
        dp_matrix[i][0] = i
    for j in range(1, len_hyp + 1):
        dp_matrix[0][j] = j

    # Iterate over the matrix
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            # If the words are equal, take the diagonal value
            if reference[i - 1] == hypothesis[j - 1]:
                dp_matrix[i][j] = dp_matrix[i - 1][j - 1]
            else:
                # Take the minimum of substitution, insertion, deletion
                substitution = dp_matrix[i - 1][j - 1] + 1
                insertion    = dp_matrix[i][j - 1] + 1
                deletion     = dp_matrix[i - 1][j] + 1
                dp_matrix[i][j] = min(substitution, insertion, deletion)

    # The bottom-right cell contains the edit distance which is the minimum number of operations 
    # (substitutions, insertions, deletions) required to change one string into the other
    edit_distance = dp_matrix[len_ref][len_hyp]

    # WER is edit distance normalized by the length of the reference
    wer_score = float(edit_distance) / len_ref
    return wer_score

# 1 The birch canoe slid on the smooth planks.
# 2 Glue the sheet to the dark blue background.
# 3 It's easy to tell the depth of a well.
# 4 These days a chicken leg is a rare dish.
# 5 Rice is often served in round bowls.

model = whisper.load_model("base.en")

reference_sentence = ["", "The birch canoe slid on the smooth planks.", "Glue the sheet to the dark blue background.", 
    "It's easy to tell the depth of a well.", "These days a chicken leg is a rare dish.", "Rice is often served in round bowls."]

path = "/Users/zijunning/Desktop/Mic_Localization/Sounds/Test_Output"
dir_list = os.listdir(path)
dir_list = sorted(dir_list)

#create an array of empty array with length of reference_sentence - 1  to store the performance
performance = [[] for _ in range(len(reference_sentence) - 1)]

# Iterate over the files in the directory
for file in tqdm(range(len(dir_list))):
    cur_file = dir_list[file]
    if cur_file.endswith(".wav"):
        sentence_num = int(cur_file[9])
        audio = whisper.load_audio(f"{path}/{cur_file}")
        audio = whisper.pad_or_trim(audio)
        result = whisper.transcribe(model, audio)
        hypothesis = result["text"]
        error_rate = wer(reference_sentence[sentence_num], hypothesis)
        performance[sentence_num-1] = performance[sentence_num-1] + [error_rate]

# Print the performance
for(i, perf) in enumerate(performance):
    print(f"Performance for sentence {i + 1}: {np.mean(perf):.2f} (mean), {np.std(perf):.2f} (std)")

flattened_list = [item for sublist in performance for item in sublist]
print(f"Overall performance: {np.mean(flattened_list):.2f} (mean), {np.std(flattened_list):.2f} (std)")

print(performance)
