SCRIPTDIR=$(dirname "$0")

${SCRIPTDIR}/Dataset_blender/generate_deformed_shapes.sh
echo ""
${SCRIPTDIR}/Simulator/generate_med_files.sh
echo ""
${SCRIPTDIR}/Simulator/simulate.sh
echo ""
