import subprocess
from bleu_rouge_meteor import test_result

#cad_file:model generated summary file path  eg:data/candidates
#ref_file:reference summary file path    eg:data/references
#output_filePath:the output score files directory path   eg:data
#eg: data/bleurt_scores.txt  data/bleu_scores.txt   data/rouge_scores.txt   data/meteor_scores.txt
def eval(cad_file,ref_file,output_filePath):
    #bleurt evaluation
    candidate_file = cad_file
    reference_file = ref_file
    bleurt_checkpoint = "../../models/bleurt/my_new_bleurt_checkpoint/new_humanScore_base/export/bleurt_best/1714400660"
    scores_file = output_filePath + "/bleurt_scores.txt"

    command = [
        "python","../../models/bleurt/bleurt/score_files.py",
        f"-candidate_file={candidate_file}",
        f"-reference_file={reference_file}",
        f"-bleurt_checkpoint={bleurt_checkpoint}",
        f"-scores_file={scores_file}"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

    #bleu,rouge,meteor evaluation
    test_result(cad_file,ref_file,output_filePath)



eval('data/input/candidates','data/input/references','data')
# eval('data/candidates','data/references','data')