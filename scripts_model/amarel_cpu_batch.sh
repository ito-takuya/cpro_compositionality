#!/bin/bash

##**** Modify these variables: ****
scriptDir="/projects/f_mc1689_1/CPROCompositionality/docs/scripts_model/"
subjBatchScriptDir="${scriptDir}/batch/"
mkdir -p $subjBatchScriptDir
jobNamePrefix="ann_cpro_"

##Make and execute a batch script for each subject
for i in {0..19..1}
do
#Only create and run script if the output file doesn't already exist
    j=$(($i+1))
    cd ${subjBatchScriptDir}

    batchFilename=${i}_simulation.sh

    echo "#!/bin/bash" > $batchFilename
    echo "#SBATCH --partition=main" >> $batchFilename
    echo "#SBATCH --requeue" >> $batchFilename
    echo "#SBATCH --time=05:00:00" >> $batchFilename
    echo "#SBATCH --nodes=1" >> $batchFilename
    echo "#SBATCH --ntasks=1" >> $batchFilename
    echo "#SBATCH --job-name=${jobNamePrefix}${subjNum}" >> $batchFilename
    echo "#SBATCH --output=slurm.${jobNamePrefix}${i}.out" >> $batchFilename
    echo "#SBATCH --error=slurm.${jobNamePrefix}${i}.err" >> $batchFilename
    echo "#SBATCH --cpus-per-task=1" >> $batchFilename
    echo "#SBATCH --mem=10000" >> $batchFilename
    echo "#SBATCH --export=ALL" >>$batchFilename

    echo "cd $scriptDir" >> $batchFilename
    echo "python experiment8_accuracyBenchmarkComparison.py --simstart $i --nsimulations $j --acc_cutoff 99 --practice --verbose" >> $batchFilename

    # Submit the job
    sbatch $batchFilename


done





