#! /bin/sh

# Script to convert .nii files to .jpg using
# med2image package for BraTS 2018 data training. 
#
# Place script within the 'MICCAI_BraTS_2018_...'
# folder before executing. 
#
# Execute script using: bash convert_to_jpg.sh

convert_img() {
	#med2image -i $1 -d ../jpg	# cannot use, each time it is run overwrites the previous files.
	med2image -i $1 -d jpg		# save jpg files inside the current dir. 
}

access_dir() {
	cd $1
	for img in *; 
		do convert_img $img
	done;
	cd ..
}

# measure start time
start=$(date +%s)

# Iterate HGG files
cd HGG
for subdir in *; 
	do access_dir $subdir
done;
cd ..

# Iterate LGG files
cd LGG 
for subdir in *;
	do access_dir $subdir
done;
cd ..

# measure end time and display difference
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
