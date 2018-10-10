#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --job-name=caiman_pipeline
#SBATCH --mem=200GB
#SBATCH --time=0-02:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH -o caiman_%A_%j_%a.log
#SBATCH -e caiman_%A_%j_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stetlb01@nyulangone.org


# Make sure there are enough numbers in the "array" range above to hold all of the jobs.


## List all video files here. 
## If the video is a .tif (single file), point directly to the .tif file. 
## If the video is a .tiff stack (many .tiff files), point to the containing folder and the algorithm will run on evert .tiff in the folder (so don't group multiple .tiff stacks in the same folder).

vid_files=(/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.00.00/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.00.01/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.00.02/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.00.03/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.01.00/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.02.00/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.03.00/images
/gpfs/data/shohamlab/ben/segmentation_project/neurofinder/data/neurofinder.04.00/images
/gpfs/data/shohamlab/ben/segmentation_project/ob_data/JG10982_171121_field3_stim_00002_00001_aligned_clean.tif
/gpfs/data/shohamlab/ben/segmentation_project/ACx_data/kkjg7stim003_aligned.tif
)


if [ ${#vid_files[@]} -lt 1 ]
then
    echo "ERROR: no video files provided in list." >&2
    exit
fi


if [ $SLURM_ARRAY_TASK_ID -ge  ${#vid_files[@]}  ]
then
    echo "Array task ID greater than needed. Exiting."
    exit
fi

vid=${vid_files[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "********************"
echo "Running CaImAn pipeline on video $vid"
echo "********************"
echo ""

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1




python caiman_pipeline.py $vid TEMP


exit


