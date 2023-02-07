SCRIPTDIR=$(dirname "$0")
BLENFILE=${SCRIPTDIR}"/scene_dataset.blend"
PYFILE_GENERATE_OUTPUTS=${SCRIPTDIR}"/generate_surfaces_out.py"

blender --background ./${BLENFILE} --python ${PYFILE_GENERATE_OUTPUTS}
