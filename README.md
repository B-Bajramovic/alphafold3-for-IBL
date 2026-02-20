
# AlphaFold3 on ALICE
*By Belmin Bajramovic [@B-Bajramovic](https://github.com/B-Bajramovic)*

This page prepares your AlphaFold3 job for submission to SLURM on HPC ALICE. 

For more on structural bioinformatics tools visit our other pages or contact us directly.

---
## 1. input
---

AF3 accepts input in JSON format. You can input single proteins, PPI-complexes, as well as ligands and ions. To make this more convenient I have prepared a script to automate FASTA to JSON conversion, prepare SBATCH subsmission script for SLURM, and a few more advanced, optional features. 

Requirements for the script are
- Python3.10+
- Bio (python package, can be installed in conda using: conda install Bio -c bioconda)

```
git clone https://github.com/B-Bajramovic/alphafold3-for-IBL
cd alphafold3-for-IBL
```

Here you will find the AF3_prepare.py. Please install the python package Bio if you have not yet

```
python AF3_prepare.py  /path/to/fasta
```

This should produce a directory following the name of your fasta entry header. Within that directory, you will find a job.sbatch script that you can use to submit the SLURM job.

```
sbatch job.sbatch
```
There are many more options to explore. For the full list of options run the script with ``` -h ``` or without any argument. Feel free to explore and ask question in Slack

---
## 3. Understanding output
---

If you ran your job using the defaults in my script, you will have run your prediction using 5 replicates. Each replicate run produces 5 models with confidence scores, and the best is automatically selected for you. The remaining models are in the subfolders with seed number in the folder name. 


---
## 4. Visualising output
---

For this we use PyMOL to visualise the structure, and some of our self-made scripts for confidence assessment and contact analysis. I will add this at a later time. If you already want to do do this yourself, feel free to contact me for the scripts. 

---
## 5. Final remarks
---

A single protein prediction is easy to do, but for larger jobs and complicated proteins (or if you just dont know how to use any of this) please reach out to belmin bajramovic on slack for more help.

For publication of data please consult [the AlphaFold3 publication](https://www.nature.com/articles/s41586-024-07487-w) and HPC alice wiki.
