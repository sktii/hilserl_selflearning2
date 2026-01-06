from flask import Flask, request, jsonify
import logging as pylog
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from absl import app, logging, flags

import ur5e_driver
import ur5e_robotiq_gripper_driver as robotiq_gripper_driver

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.0.10", "IP address of the UR5e robot")
flags.DEFINE_string("gripper_ip", "192.168.0.10", "IP address of the Robotiq gripper")
flags.DEFINE_integer("port", 5000, "Port to run the Flask server on")

def main(_):
    robot_ip = FLAGS.robot_ip
    gripper_ip = FLAGS.gripper_ip

    ur5e = ur5e_driver.UR_controller(ip=robot_ip)

    # Initialize gripper safely - don't crash if it fails
    gripper = robotiq_gripper_driver.RobotiqGripper()
    try:
        gripper.connect(gripper_ip, 63352)
        print("Gripper connected")
        time.sleep(3)
        gripper.reset()
    except Exception as e:
        print(f"Gripper connection failed: {e}. Proceeding without gripper.")
        gripper = None

    webapp = Flask(__name__)

    # Route for Setting Load
    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        return "Set Load"

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        return "Started impedance"

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        return "Stopped impedance"

    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        pose = np.array(ur5e.get_TCP_pose())
        xyz = pose[:3]
        r = R.from_rotvec(pose[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        pose = np.array(ur5e.get_TCP_pose())
        xyz = pose[:3]
        r = R.from_rotvec(pose[3:]).as_quat()
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": ur5e.get_TCP_vel()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": ur5e.get_force()[:3]})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": ur5e.get_force()[3:]})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": ur5e.get_joint_q()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": ur5e.get_joint_dq()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": ur5e.get_jacobian()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        if gripper:
            try:
                pos = 1.0 - gripper.get_current_position() / 256.0
            except:
                pos = 0.0
        else:
            pos = 0.0
        return jsonify({"gripper": pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        ur5e.joint_reset()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        if gripper: gripper.activate()
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        if gripper: gripper.reset()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        if gripper: gripper.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        if gripper: gripper.close()
        return "Closed"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        if gripper: gripper.close_slow()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        if gripper:
            gripper_pos = request.json
            pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
            print(f"move gripper to {pos}")
            gripper.move(position=pos, speed=1, force=1)
        return "Moved Gripper"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        return "Clear"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        if pos[3] < 0:
            pos *= np.array([1, 1, 1, -1, -1, -1, -1])
        xyz = pos[:3]
        r = R.from_quat(pos[3:])
        ur5e.pose(xyz, r)
        return "Moved"

    @webapp.route("/movel", methods=["POST"])
    def movel():
        pos = np.array(request.json["arr"])
        if pos[3] < 0:
            pos *= np.array([1, 1, 1, -1, -1, -1, -1])
        xyz = pos[:3]
        r = R.from_quat(pos[3:])
        ur5e.movel(xyz, r)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        pose = np.array(ur5e.get_TCP_pose())
        xyz = pose[:3]
        r = R.from_rotvec(pose[3:]).as_quat()
        torque = ur5e.get_torque()

        gripper_pos = 0.0
        if gripper:
            try:
                gripper_pos = 1.0 - gripper.get_current_position() / 256.0
            except:
                pass

        return jsonify(
            {
                "pose": np.concatenate([xyz, r]).tolist(),
                "vel": ur5e.get_TCP_vel(),
                "force": torque[:3],
                "torque": torque[3:],
                "q": ur5e.get_joint_q(),
                "dq": ur5e.get_joint_dq(),
                "jacobian": ur5e.get_jacobian(),
                "gripper_pos": gripper_pos,
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        return "Updated compliance parameters"

    print(f"Starting UR5e Server on 0.0.0.0:{FLAGS.port}")
    webapp.run(host="0.0.0.0", port=FLAGS.port)


if __name__ == "__main__":
    pylog.getLogger('werkzeug').setLevel(pylog.ERROR)
    logging.set_verbosity(logging.FATAL)
    app.run(main)
