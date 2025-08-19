import trimesh
import os


dir = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexGraspBench/output/debug_dummy_arm_shadow/vis_obj/core_bottle_b13f6dc78d904e5c30612f5c0ef21eb8/tabletop_ur10e/scale008_pose007_0"

# Load the two OBJ files
mesh1 = trimesh.load(os.path.join(dir, "10_grasp_grasp_0.obj"))
mesh2 = trimesh.load(os.path.join(dir, "10_grasp_obj.obj"))

# # Translate mesh2 to avoid overlap with mesh1
# mesh2.apply_translation([2.0, 0, 0])  # Adjust as needed

# Create a scene and add both meshes
scene = trimesh.Scene()
scene.add_geometry(mesh1)
scene.add_geometry(mesh2)

# Show the scene in an interactive window
scene.show()
