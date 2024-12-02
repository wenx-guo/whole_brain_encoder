#!/bin/bash

# Define flags for each job
subj_flag="--subj 1"
lr_flag="--lr 0.0005"
batch_size_flag="--batch_size 16"
enc_output_layer_flag="--enc_output_layer 1"
run_flag="--run $SLURM_JOB_ID"

common_flags="$lr_flag $subj_flag $enc_output_layer_flag $run_flag $batch_size_flag"

flags_job1="$common_flags --hemi lh --axis posterior"
flags_job2="$common_flags --hemi rh --axis posterior"
flags_job3="$common_flags --hemi lh --axis anterior"
flags_job4="$common_flags --hemi rh --axis anterior"

# Submit the 4 independent jobs with their flags
job1_id=$(sbatch --parsable train_passflags.sh $flags_job1)
job2_id=$(sbatch --parsable train_passflags.sh $flags_job2)
job3_id=$(sbatch --parsable train_passflags.sh $flags_job3)
job4_id=$(sbatch --parsable train_passflags.sh $flags_job4)

# Submit the dependent job
final_job_id=$(sbatch --parsable --dependency=afterok:$job1_id:$job2_id:$job3_id:$job4_id plot_results.sh $subj_flag $enc_output_layer_flag $run_flag)

# Output submitted jobs
echo "Submitted jobs:"
echo "Job 1 ID: $job1_id (Flags: $flags_job1)"
echo "Job 2 ID: $job2_id (Flags: $flags_job2)"
echo "Job 3 ID: $job3_id (Flags: $flags_job3)"
echo "Job 4 ID: $job4_id (Flags: $flags_job4)"
echo "Final job ID (dependent on all): $final_job_id"
