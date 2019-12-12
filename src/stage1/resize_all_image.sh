#!/usr/bin/env bash
set -e

export NUMJOBS=$1
export INPUTDIR=$2
export OUTPUTDIR=$3

echo "Flattening Gigaword with ${NUMJOBS} processes..."
#find ${OUTPUTDIR}/* | parallel --gnu --progress -j ${NUMJOBS} python3.6 fix_image_rotation.py ${OUTPUTDIR}/{/}
ls ${INPUTDIR} | grep \. | parallel --gnu --progress -j ${NUMJOBS} python3.6 resize.py ${INPUTDIR}/{/} ${OUTPUTDIR}/{/}
#find ${OUTPUTDIR} | parallel --gnu --progress -j ${NUMJOBS} echo  ${OUTPUTDIR}/{/}
