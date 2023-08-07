#! /bin/sh

# Select jpg images to use with the experiement. 
# 
# This script randomly selects 10 jpg images from each subfolder in the
# dataset. The script uses a number seeder to ensure the same images 
# are selected everytime the script is executed.
#
# Script is executed using the command: bash select_jpg_images.sh

DATASET_PATH=../dataset/MICCAI_BraTS_2018_Data_Training
SUBDIRS=(HGG LGG)

MIN_IMAGE_RANGE=70
MAX_IMAGE_RANGE=110


cd $DATASET_PATH

for dir in "${SUBDIRS[@]}": 
do
	cd "${dir:0:3}"

	for image_dir in *
	do
		cd "$image_dir/jpg"
	
		# DO NOT CHANGE	
		current_path="MICCAI_BraTS_2018_Data_Training/${dir:0:3}/$image_dir/jpg"
		IMAGES_PATH=../../../../images_used
		
		for jpg in *
		do 
			hundred_num="${jpg:12:1}"
			tens_num="${jpg:13:2}"

			# generate uid
			case "$hundred_num" in
				0) 
					ID="$image_dir-$tens_num"
					if [[ ${tens_num:0:1} -ge ${MIN_IMAGE_RANGE:0:1} ]]
					then
						mod=$(expr $tens_num % 4)
						if [[ $mod -eq 0 ]]
						then
							cp "$jpg" "$IMAGES_PATH/$ID"
						fi
					fi
					;;
				*) 
					ID="$image_dir-$hundred_num$tens_num"
					if [[ ${tens_num:0:1} -lt ${MAX_IMAGE_RANGE:1:1} ||
						${tens_num} -eq ${MAX_IMAGE_RANGE:1:2}
						]]
					then
						if [[ ${tens_num:1:1} -lt ${MAX_IMAGE_RANGE:1:2} ]] 
						then
							mod=$(expr $tens_num % 4)
							if [[ $mod -eq 0 ]]
							then 
								cp "$jpg" "$IMAGES_PATH/$ID"
							fi
						fi
					fi
					;;
			esac
		done
		cd ../..	# back to 'dir' folder
	done
	cd ..		# back to original dataset folder.
done


