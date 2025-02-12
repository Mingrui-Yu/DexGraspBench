import os
import trimesh
import numpy as np
import mujoco
import transforms3d.quaternions as tq


class HOMjSpec:

    hand_prefix: str = "child-"
    hf_name: str = "hand_freejoint"
    of_name: str = "obj_freejoint"

    def __init__(
        self,
        obj_path,
        obj_pose,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_tendon_cfg,
        friction_coef,
        has_floor_z0,
        pregrasp_pose,
        pregrasp_qpos,
        grasp_pose,
        grasp_qpos,
    ):
        self.hand_tendon_cfg = {}
        if hand_tendon_cfg is not None:
            for k, v in hand_tendon_cfg.items():
                self.hand_tendon_cfg[self.hand_prefix + k] = [
                    self.hand_prefix + v[0],
                    self.hand_prefix + v[1],
                ]

        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = 0.004
        self.spec.option.impratio = 10
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.spec.option.enableflags = mujoco.mjtEnableBit.mjENBL_NATIVECCD
        self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.3, 0.5, 0.7],
            width=512,
            height=512,
        )

        self.joint_names, self.actuator_targets = self._add_hand(hand_xml_path)
        self._add_object(
            obj_path,
            obj_pose,
            obj_scale,
            has_floor_z0,
            obj_density=obj_density,
        )
        self._set_friction(friction_coef)
        self._add_key(pregrasp_pose, pregrasp_qpos, obj_pose)
        self._add_key(grasp_pose, grasp_qpos, obj_pose)

        return

    def _add_hand(self, xml_path):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        # Add freejoint of hand root
        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.hand_prefix, ""
        )
        child_world.add_freejoint(name=self.hf_name)
        self.spec.worldbody.add_body(name="mocap_body", mocap=True)
        mocap_solimp = [0.9, 0.95, 0.001, 0.5, 2]  # default parameters
        mocap_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name1="mocap_body",
            name2=f"{self.hand_prefix}world",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            solimp=mocap_solimp,
            data=mocap_data,
        )

        joint_names = [joint.name for joint in self.spec.joints]
        hf_id = joint_names.index(self.hf_name)
        joint_names.pop(hf_id)
        assert hf_id == 0
        actuator_targets = [actuator.target for actuator in self.spec.actuators]
        return joint_names, actuator_targets

    def _add_object(
        self,
        obj_path,
        obj_pose,
        obj_scale,
        has_floor_z0,
        obj_density=1000,
    ):
        # TODO: be careful about the floor for testing!
        if has_floor_z0:
            floor_geom = self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
                margin=0.02,
            )

        obj_body = self.spec.worldbody.add_body(
            name="object", pos=obj_pose[:3], quat=obj_pose[3:]
        )
        obj_body.add_freejoint(name=self.of_name)
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
                margin=0,
                gap=0,
            )

        return

    def _add_key(self, hand_pose, hand_qpos, obj_pose):

        key_qpos = self.merge_pose_qpos(hand_pose, hand_qpos, obj_pose)
        key_ctrl = self.qpos_to_ctrl(hand_qpos)
        self.spec.add_key(
            qpos=key_qpos, ctrl=key_ctrl, mpos=hand_pose[:3], mquat=hand_pose[3:]
        )

        return

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def merge_pose_qpos(self, hand_pose, hand_qpos, obj_pose):
        merged_qpos = np.concatenate([hand_pose, hand_qpos, obj_pose], axis=0)
        return merged_qpos

    def split_qpos_pose(self, merged_qpos):
        hand_pose = merged_qpos[:7]
        hand_qpos = merged_qpos[7:-7]
        obj_pose = merged_qpos[-7:]
        return hand_pose, hand_qpos, obj_pose

    def qpos_to_ctrl(self, qpos):
        key_ctrl = []
        for at in self.actuator_targets:
            if at in self.joint_names:
                qid = self.joint_names.index(at)
                key_ctrl.append(qpos[qid])
            elif at in self.hand_tendon_cfg.keys():
                qid0 = self.joint_names.index(self.hand_tendon_cfg[at][0])
                qid1 = self.joint_names.index(self.hand_tendon_cfg[at][1])
                key_ctrl.append(qpos[qid0] + qpos[qid1])
        return key_ctrl


class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.mj_model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self):
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = v["vert"] @ geom_rot.T + geom_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


def get_pregrasp_grasp_squeeze_poses(grasp_data, pose_config):
    # grasp
    grasp_pose = grasp_data["hand_pose"]
    grasp_qpos = grasp_data["hand_qpos"]

    # pregrasp
    if pose_config.pregrasp_type is None:
        pregrasp_pose = grasp_data["pregrasp_pose"]
        pregrasp_qpos = grasp_data["pregrasp_qpos"]
    elif pose_config.pregrasp_type == "minus":
        pregrasp_pose = grasp_pose
        pregrasp_qpos = grasp_qpos - pose_config.pregrasp_coef_minus
    elif pose_config.pregrasp_type == "multiply":
        pregrasp_pose = grasp_pose
        pregrasp_qpos = grasp_qpos * pose_config.pregrasp_coef_multiply
    else:
        raise NotImplementedError

    # squeeze pose
    if pose_config.squeeze_type is None:
        squeeze_pose = grasp_data["hand_pose"]
        squeeze_qpos = grasp_data["squeeze_qpos"]
    elif pose_config.squeeze_type == "multiply":
        squeeze_pose = grasp_pose
        squeeze_qpos = (
            grasp_qpos - pregrasp_qpos
        ) * pose_config.squeeze_coef + grasp_qpos
    else:
        raise NotImplementedError

    return {
        "pregrasp": [pregrasp_pose, pregrasp_qpos],
        "grasp": [grasp_pose, grasp_qpos],
        "squeeze": [squeeze_pose, squeeze_qpos],
    }


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../../assets/hand/shadow/customized.xml"
    )
    kinematic = RobotKinematics(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export(f"debug_hand.obj")
