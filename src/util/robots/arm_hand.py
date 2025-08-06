from .base import Robot, ArmHand, Arm, Hand, register_robot, RobotFactory
from abc import ABC, abstractmethod


@register_robot("dummy_arm_shadow")
class DummyArmShadow(ArmHand):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.name = "dummy_arm_shadow"

        arm_type = "dummy_arm"
        hand_type = "shadow"
        self.arm: Arm = RobotFactory.create_robot(robot_type=arm_type, prefix=prefix)
        self.hand: Hand = RobotFactory.create_robot(robot_type=hand_type, prefix=prefix)

        self.side = self.prefix  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            # self._urdf_path = "assets/robots/dummy_arm_shadow/dummy_arm_shadow.urdf"
            self._mjcf_path = "assets/hand/dummy_arm_shadow/right_no_tendon.xml"
            # self._cfg_path = "dexgrasp_rl/cfg/robots/dummy_arm_shadow.yml"
        else:
            raise NotImplementedError()

        self._base_pose = [0.0, 0.0, 0.0, 0, 0.0, 0, 1.0]  # (xyz, xyzw), base pose in the world frame
        assert len(self._base_pose) == 7


@register_robot("no_arm_shadow")
class NoArmShadow(ArmHand):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.name = "no_arm_shadow"

        self.arm: Arm = None
        self.hand: Hand = RobotFactory.create_robot(robot_type="shadow", prefix=prefix)

        self.side = self.prefix  # TODO
        assert self.side == "rh" or self.side == "lh"

        if self.side == "rh":
            self._urdf_path = "assets/robots/shadow_hand/shadow_hand_body.urdf"
        else:
            raise NotImplementedError()

        self._base_pose = [0, 0, 0, 0, 0, 0, 1]  # (xyz, xyzw)
