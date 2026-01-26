
# AlphaFold3 on ALICE
*By Belmin Bajramovic [@B-Bajramovic](https://github.com/B-Bajramovic)*

Running AlphaFold on HPC ALICE requires preparation and some assistance if you are new to HPCs. We cannot directly run the prediction like you would on your own computer. On ALICE we use SLURM as a queuing system to organise faculty-wide usage. This guide prepares your job for submission using a script.

For more on structural bioinformatics tools visit our other pages or contact us directly.

---
## 1. input
---

AF3 accepts input in JSON format. For proteins, we have a script that converts fasta to json. It can take multiple fasta files at once if you refer to the directory containing the files. There are some default settings to the script that you can adjust once you understand what you are doing.

Requirements for the script are
- Python3.10+
- Bio (python package, can be installed in conda using: conda install Bio -c bioconda)

```
python AF3_job_prepare.py --prey /path/to/directory/with/fasta/files --project name_of_project --make-sbatch --job-dirs

```
The full options of the script are a bit intimidating, but to begin you only need --prey, which refers to your fasta file, and --project, which is the directory into which your json file will go and your AlphaFold3 output later on. The --make-sbatch and --job-dirs options let the python script prepare the slurm job for you.

The project folder will automatically write to /data1/$USER. If you want this changed to a custom location, please use the option --output-root /path/that/you/prefer

---
## 2. Submitting your AlphaFold job to the queue system SLURM
---

If you used the --make--sbatch option, your project directory should now contain a submit_all.sh. This is meant for submitting your AlphaFold job to SLURM

```
cd /path/to/your/project/directory
bash submit_all.sh
```

Your job may run instantly, but it could also take hours before it gets accepted. The more you run, the lower your priority in the queue.

---
## 3. Understanding output
---

If you ran your job using the defaults in AF3_job_prepare.py, your project folder should now contain structure and confidence output data in the subfolders. The best predicted structure is always named proteinname_model.cif, with the replicates under separate seed folders. Confidences are stored in JSON files and can be used for further analysis and visualisation.

---
## 4. Visualising output
---

For this we use PyMOL to visualise the structure, and some of our self-made scripts for confidence assessment and contact analysis.

---
## 5. Final remarks
---

A single protein prediction is easy to do, but for larger jobs and complicated proteins (or if you just dont know how to use any of this) please reach out to belmin bajramovic on slack for more help.

For publication of data please consult [the AlphaFold3 publication](https://www.nature.com/articles/s41586-024-07487-w) and HPC alice wiki.
