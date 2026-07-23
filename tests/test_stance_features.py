"""站立几何特征单测。"""

from __future__ import annotations


def test_stance_feature_names_registered():
    from analysis.features.registry import default_registry

    names = default_registry().frame_feature_names()
    assert "stance.shoulder_hip_knee_angle_min" in names


def test_shoulder_hip_knee_angle_min_standing_like():
    from analysis.features.stance import shoulder_hip_knee_angle_min

    person = {
        "keypoints": [[0, 0, 0.0]] * 17,
    }
    kpts = person["keypoints"]
    kpts[5] = [100.0, 100.0, 0.9]
    kpts[6] = [120.0, 100.0, 0.9]
    kpts[11] = [100.0, 200.0, 0.9]
    kpts[12] = [120.0, 200.0, 0.9]
    kpts[13] = [100.0, 300.0, 0.9]
    kpts[14] = [120.0, 300.0, 0.9]

    angle = shoulder_hip_knee_angle_min(person)
    assert angle > 150.0
