from pathlib import Path
import math

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "lecarm" / "lecarm.xml"
_HAND_XML = _HERE / "leap_hand" / "left_hand.xml"

fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]

# fmt: off
HOME_QPOS = [
    # lecarm.
    0, 0, 0, 0, 0, 0,
    # leap.
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on


def _ensure_leap_left_tip_sites(hand: mujoco.MjSpec) -> None:
    tip_sites = {
        "tip_1": ("fingertip", (0, -0.04, 0.015)),
        "tip_2": ("fingertip_2", (0, -0.04, 0.015)),
        "tip_3": ("fingertip_3", (0, -0.04, 0.015)),
        "th_tip": ("thumb_fingertip", (0, -0.045, -0.015)),
    }
    for site_name, (body_name, pos) in tip_sites.items():
        body = hand.body(body_name)
        body.add_site(name=site_name, pos=pos)


def construct_model() -> mujoco.MjModel:
    arm = mujoco.MjSpec.from_file(_ARM_XML.as_posix())
    hand = mujoco.MjSpec.from_file(_HAND_XML.as_posix())

    _ensure_leap_left_tip_sites(hand)

    ee_body = arm.body("link_6")
    site = ee_body.add_site(name="attachment_site", pos=(0, 0, 0.06), group=5)

    palm = hand.body("palm_lower")
    palm.quat[:] = (0, 1, 0, 0)
    palm.pos[:] = (0.065, 0.04, 0)
    arm.attach(hand, prefix="leap_left/", site=site)

    try:
        home_key = arm.key("home")
    except KeyError:
        home_key = None
    if home_key is not None:
        arm.delete(home_key)
    arm.add_key(name="home", qpos=HOME_QPOS)

    return arm.compile()


if __name__ == "__main__":
    model = construct_model()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        home_qpos = data.qpos.copy()

        rate = RateLimiter(frequency=120.0, warn=False)
        t = 0.0
        while viewer.is_running():
            t += rate.dt
            qpos = home_qpos.copy()

            # Fake joint actions: smooth sinusoidal motion for arm + hand.
            for i in range(6):
                qpos[i] += 0.35 * (1.0 - 0.1 * i) * math.sin(t + i * 0.7)
            for i in range(6, model.nq):
                qpos[i] += 0.6 * math.sin(t * 1.5 + i * 0.4)

            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
