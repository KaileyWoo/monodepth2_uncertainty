current=`pwd`
position='../../Datasets/kitti_raw_eigen/'

# unzip and move gt to proper folders
pwd
cd  $position
seqs=`cat $current/splits/eigen_benchmark/test_files.txt | cut -d' ' -f1 | cut -d'/' -f2 | uniq`    
pwd
for s in $seqs; do
    date=`echo $s | cut -d'_' -f1-3`
    
    if [ -d train/$s ];
    then
        mv ./train/$s/* $position/$date/$s/
    else
        mv ./val/$s/* $position/$date/$s/
    fi
done
#rm -r train
#rm -r val

#python export_gt_depth.py --data_path $1 --split eigen_benchmark

# ready to go!
