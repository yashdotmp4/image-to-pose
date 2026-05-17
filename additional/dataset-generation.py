"""
generate_lighting_dataset.py

Generates synthetic renders of wooden.glb in realistic 3D scenes
with random poses, ground planes, walls, and random objects.

Output:
    data/lighting/
        images/  000000.jpg ...
        labels.csv  (filename, light_x, light_y, light_z)

Usage:
    python3 generate_lighting_dataset.py --n 50000 --glb frontend/wooden.glb --out data/lighting
"""

import csv
import math
import random
import argparse
import numpy as np
if not hasattr(np, 'infty'):
    np.infty = np.inf
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import trimesh
import trimesh.transformations as tf
import pyrender

# ── Fixed mannequin colour ─────────────────────────────────────────────────────
def mannequin_colour():
    base = np.array([0.32, 0.16, 0.07])
    return np.clip(base + np.random.uniform(-0.05, 0.05, 3), 0.04, 1.0)

# ── Background colours ─────────────────────────────────────────────────────────
BG_COLOURS = [
    [0.12, 0.12, 0.18, 1.0],
    [0.08, 0.08, 0.08, 1.0],
    [0.18, 0.10, 0.10, 1.0],
    [0.10, 0.16, 0.10, 1.0],
    [0.14, 0.12, 0.20, 1.0],
    [0.35, 0.28, 0.20, 1.0],
    [0.20, 0.28, 0.38, 1.0],
    [0.38, 0.22, 0.15, 1.0],
    [0.20, 0.35, 0.25, 1.0],
    [0.30, 0.25, 0.18, 1.0],
]

# ── Vivid object colours ───────────────────────────────────────────────────────
VIVID_COLOURS = [
    [0.92, 0.12, 0.10], [0.95, 0.52, 0.04], [0.92, 0.88, 0.05],
    [0.08, 0.80, 0.18], [0.05, 0.42, 0.92], [0.55, 0.08, 0.90],
    [0.92, 0.08, 0.58], [0.05, 0.85, 0.82], [0.88, 0.88, 0.88],
    [0.80, 0.62, 0.04], [0.28, 0.62, 0.92], [0.90, 0.38, 0.38],
    [0.38, 0.90, 0.50], [0.90, 0.65, 0.20], [0.60, 0.20, 0.20],
    [0.20, 0.55, 0.35],
]

# ── Floor tile palette pairs ───────────────────────────────────────────────────
FLOOR_PAIRS = [
    ([0.72, 0.62, 0.45], [0.42, 0.32, 0.18]),
    ([0.25, 0.25, 0.28], [0.40, 0.40, 0.45]),
    ([0.55, 0.40, 0.30], [0.30, 0.20, 0.12]),
    ([0.68, 0.65, 0.55], [0.38, 0.35, 0.28]),
    ([0.35, 0.48, 0.35], [0.18, 0.28, 0.18]),
    ([0.65, 0.55, 0.72], [0.35, 0.25, 0.42]),
    ([0.72, 0.42, 0.30], [0.42, 0.20, 0.12]),
    ([0.42, 0.58, 0.68], [0.22, 0.35, 0.45]),
]

WALL_COLOURS = [
    [0.55, 0.48, 0.38], [0.32, 0.32, 0.38], [0.48, 0.55, 0.48],
    [0.60, 0.50, 0.40], [0.38, 0.42, 0.55], [0.55, 0.38, 0.35],
    [0.42, 0.50, 0.42], [0.28, 0.22, 0.32],
]

# ── Bone hierarchy ─────────────────────────────────────────────────────────────
BONE_HIERARCHY = [
    ('Thigh.L_01',    'Pelvis_00',    [0, -1, 0], 60),
    ('Leg.L_02',      'Thigh.L_01',   [0, -1, 0], 80),
    ('Thigh.R_014',   'Pelvis_00',    [0, -1, 0], 60),
    ('Leg.R_015',     'Thigh.R_014',  [0, -1, 0], 80),
    ('Arm.L_011',     'Chest_05',     [0,  0, -1], 90),
    ('Forearm.L_012', 'Arm.L_011',    [0,  0, -1], 80),
    ('Arm.R_08',      'Chest_05',     [0,  0,  1], 90),
    ('Forearm.R_09',  'Arm.R_08',     [0,  0,  1], 80),
]


# ── Pose randomisation ─────────────────────────────────────────────────────────

def apply_random_pose(scene):
    POSES = [
        'tpose', 'arms_down', 'arms_raised', 'walk', 'squat',
        'reach', 'lunge', 'warrior', 'kick', 'boxing_guard',
        'celebrate', 'lean_side', 'point', 'crouch', 'stumble', 'random',
    ]
    pose_type = random.choice(POSES)

    def update_bone_and_children(bone_name, rot_matrix):
        try:
            current_t, _ = scene.graph.get(bone_name)
            new_t = rot_matrix @ current_t
            scene.graph.update(bone_name, matrix=new_t)
            for child in scene.graph.children_nodes(bone_name):
                update_bone_and_children(child, rot_matrix)
        except Exception:
            pass

    def set_bone_rotation(bone_name, axis, angle_deg):
        try:
            current_t, _ = scene.graph.get(bone_name)
            pivot = current_t[:3, 3]
            rot = tf.rotation_matrix(math.radians(angle_deg), axis, pivot)
            update_bone_and_children(bone_name, rot)
        except Exception:
            pass

    if pose_type == 'tpose':
        pass
    elif pose_type == 'arms_down':
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(55, 85))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-85, -55))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(10, 35))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-35, -10))
    elif pose_type == 'arms_raised':
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-110, -60))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(60, 110))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-30, 30))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-30, 30))
    elif pose_type == 'walk':
        leg_swing = random.uniform(25, 50)
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  leg_swing)
        set_bone_rotation('Thigh.R_014',   [1, 0, 0], -leg_swing)
        set_bone_rotation('Leg.L_02',      [1, 0, 0],  random.uniform(15, 40))
        set_bone_rotation('Leg.R_015',     [1, 0, 0],  random.uniform(0, 20))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(35, 65))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-65, -35))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(10, 35))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-35, -10))
    elif pose_type == 'squat':
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(40, 70))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0],  random.uniform(40, 70))
        set_bone_rotation('Leg.L_02',      [1, 0, 0], -random.uniform(60, 100))
        set_bone_rotation('Leg.R_015',     [1, 0, 0], -random.uniform(60, 100))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(25, 55))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-55, -25))
    elif pose_type == 'reach':
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-80, 80))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-50, 50))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(40, 80))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-30, 30))
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(-20, 20))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0],  random.uniform(-20, 20))
    elif pose_type == 'lunge':
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(40, 65))
        set_bone_rotation('Leg.L_02',      [1, 0, 0], -random.uniform(50, 80))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0], -random.uniform(20, 40))
        set_bone_rotation('Leg.R_015',     [1, 0, 0],  random.uniform(10, 25))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(40, 70))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-70, -40))
    elif pose_type == 'warrior':
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(30, 55))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0], -random.uniform(30, 55))
        set_bone_rotation('Leg.L_02',      [1, 0, 0], -random.uniform(20, 45))
        set_bone_rotation('Leg.R_015',     [1, 0, 0],  random.uniform(20, 45))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-20, 20))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-40, 40))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-20, 20))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-40, 40))
    elif pose_type == 'kick':
        side = random.choice(['L', 'R'])
        if side == 'L':
            set_bone_rotation('Thigh.L_01',  [1, 0, 0],  random.uniform(60, 90))
            set_bone_rotation('Leg.L_02',    [1, 0, 0], -random.uniform(30, 60))
            set_bone_rotation('Thigh.R_014', [1, 0, 0], -random.uniform(10, 25))
        else:
            set_bone_rotation('Thigh.R_014', [1, 0, 0],  random.uniform(60, 90))
            set_bone_rotation('Leg.R_015',   [1, 0, 0], -random.uniform(30, 60))
            set_bone_rotation('Thigh.L_01',  [1, 0, 0], -random.uniform(10, 25))
        set_bone_rotation('Arm.L_011', [0, 0, 1], random.uniform(20, 50))
        set_bone_rotation('Arm.R_08',  [0, 0, 1], random.uniform(-50, -20))
    elif pose_type == 'boxing_guard':
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-20, 20))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-60, -30))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-20, 20))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(30, 60))
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(10, 30))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0], -random.uniform(5, 20))
    elif pose_type == 'celebrate':
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-130, -80))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-30, 10))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(80, 130))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-10, 30))
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(5, 20))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0], -random.uniform(5, 20))
    elif pose_type == 'lean_side':
        side = random.choice([1, -1])
        set_bone_rotation('Arm.L_011',   [0, 0, 1],  random.uniform(40, 70) * side)
        set_bone_rotation('Arm.R_08',    [0, 0, 1], -random.uniform(20, 50) * side)
        set_bone_rotation('Thigh.L_01',  [0, 0, 1],  random.uniform(10, 25) * side)
        set_bone_rotation('Thigh.R_014', [0, 0, 1],  random.uniform(10, 25) * side)
    elif pose_type == 'point':
        side = random.choice(['L', 'R'])
        if side == 'L':
            set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-40, 40))
            set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-20, 20))
            set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-65, -35))
            set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-25, -5))
        else:
            set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-40, 40))
            set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-20, 20))
            set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(35, 65))
            set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(5, 25))
    elif pose_type == 'crouch':
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(55, 80))
        set_bone_rotation('Leg.L_02',      [1, 0, 0], -random.uniform(70, 110))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0],  random.uniform(55, 80))
        set_bone_rotation('Leg.R_015',     [1, 0, 0], -random.uniform(70, 110))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(10, 45))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-50, -20))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-45, -10))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(20, 50))
    elif pose_type == 'stumble':
        set_bone_rotation('Thigh.L_01',    [1, 0, 0],  random.uniform(-30, 50))
        set_bone_rotation('Thigh.R_014',   [1, 0, 0],  random.uniform(-30, 50))
        set_bone_rotation('Leg.L_02',      [1, 0, 0],  random.uniform(-60, 20))
        set_bone_rotation('Leg.R_015',     [1, 0, 0],  random.uniform(-60, 20))
        set_bone_rotation('Arm.L_011',     [0, 0, 1],  random.uniform(-90, 90))
        set_bone_rotation('Forearm.L_012', [0, 0, 1],  random.uniform(-60, 60))
        set_bone_rotation('Arm.R_08',      [0, 0, 1],  random.uniform(-90, 90))
        set_bone_rotation('Forearm.R_09',  [0, 0, 1],  random.uniform(-60, 60))
    else:  # random
        for bone, _, axis, max_deg in BONE_HIERARCHY:
            set_bone_rotation(bone, axis, random.uniform(-max_deg * 0.8, max_deg * 0.8))


# ── Scene builder ──────────────────────────────────────────────────────────────

def _mat(r, g, b, roughness=None, metallic=0.0):
    if roughness is None:
        roughness = random.uniform(0.4, 1.0)
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[float(r), float(g), float(b), 1.0],
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
    )


def add_mesh(py_scene, tm, colour, pose=None, roughness=None, metallic=0.0):
    if pose is None:
        pose = np.eye(4)
    mat = _mat(*colour, roughness=roughness, metallic=metallic)
    mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)
    py_scene.add(mesh, pose=pose)


def make_checker_floor(tile_size, n_tiles, col_a, col_b, centre_x, centre_z):
    """Checkerboard floor centred on the figure, not the world origin."""
    half = n_tiles // 2
    thickness = 0.05  # thin slab — closed mesh so shadow-map pass never skips it
    for ix in range(n_tiles):
        for iz in range(n_tiles):
            col = col_a if (ix + iz) % 2 == 0 else col_b
            cx = centre_x + (ix - half) * tile_size + tile_size * 0.5
            cz = centre_z + (iz - half) * tile_size + tile_size * 0.5
            box = trimesh.creation.box(extents=[tile_size, thickness, tile_size])
            # Shift so the TOP face sits at Y=0; floor_pose raises it to floor_y
            box.apply_translation([cx, -thickness * 0.5, cz])
            yield box, col


def build_scene(py_scene, floor_y=0.0, cam_theta=0.0, centre_x=0.0, centre_z=0.0):
    col_a, col_b = random.choice(FLOOR_PAIRS)
    col_a = np.clip(np.array(col_a) + np.random.uniform(-0.06, 0.06, 3), 0, 1).tolist()
    col_b = np.clip(np.array(col_b) + np.random.uniform(-0.06, 0.06, 3), 0, 1).tolist()

    # Floor — always present
    tile_size     = random.uniform(0.5, 0.9)
    n_tiles       = 30
    shiny_floor   = random.random() < 0.80
    floor_roughness = random.uniform(0.04, 0.20) if shiny_floor else random.uniform(0.40, 0.65)
    floor_metallic  = random.uniform(0.50, 0.90) if shiny_floor else random.uniform(0.00, 0.15)

    floor_pose = tf.translation_matrix([0, floor_y, 0])
    for tm, col in make_checker_floor(tile_size, n_tiles, col_a, col_b, centre_x, centre_z):
        add_mesh(py_scene, tm, col, floor_pose,
                 roughness=floor_roughness, metallic=floor_metallic)

    # Single wall — always present, always face-on to the camera.
    # cam_theta is the camera's azimuth around the mannequin, so the wall
    # sits directly behind the mannequin (opposite side) and is rotated so
    # its normal points straight toward the camera — no skewed angles.
    wall_faces = np.array([[0,1,2],[0,2,3]])
    wall_col = np.clip(np.array(random.choice(WALL_COLOURS)) + np.random.uniform(-0.08, 0.08, 3), 0, 1).tolist()
    wall_dist = random.uniform(2.0, 3.5)
    # Position: behind the mannequin from the camera's perspective
    wx = centre_x - wall_dist * math.cos(cam_theta)
    wz = centre_z - wall_dist * math.sin(cam_theta)
    # Rotation: local +Z normal rotated to face toward camera.
    # Y-rotation by (pi/2 - cam_theta) maps (0,0,1) -> (cos(cam_theta), 0, sin(cam_theta)).
    wall_rot = math.pi / 2 - cam_theta
    wall_verts = np.array([
        [-8, 0,  0], [8, 0,  0],
        [ 8, 10, 0], [-8, 10, 0],
    ], dtype=float)
    wall_tm = trimesh.Trimesh(vertices=wall_verts, faces=wall_faces)
    wall_pose = tf.concatenate_matrices(
        tf.translation_matrix([wx, floor_y, wz]),
        tf.rotation_matrix(wall_rot, [0, 1, 0]),
    )
    add_mesh(py_scene, wall_tm, wall_col, wall_pose,
             roughness=random.uniform(0.50, 0.85), metallic=random.uniform(0.00, 0.15))

    # Objects — darkened to stay in the same luminance ballpark as the mannequin
    n_objects = random.randint(3, 6)
    for _ in range(n_objects):
        angle = cam_theta + random.uniform(-math.pi * 0.55, math.pi * 0.55)
        dist  = random.uniform(1.0, 2.5)
        ox = centre_x + math.cos(angle) * dist
        oz = centre_z + math.sin(angle) * dist
        # Scale vivid colours down so objects don't blow out relative to mannequin
        scale = random.uniform(0.25, 0.50)
        col = np.clip(np.array(random.choice(VIVID_COLOURS)) * scale
                      + np.random.uniform(-0.04, 0.04, 3), 0, 1).tolist()
        obj_type = random.choice(['box', 'tall_box', 'flat_box', 'cylinder', 'sphere'])

        if obj_type == 'box':
            w, h, d = random.uniform(0.3, 0.8), random.uniform(0.3, 0.8), random.uniform(0.3, 0.8)
            obj = trimesh.creation.box(extents=[w, h, d])
            pose = tf.translation_matrix([ox, floor_y + h/2, oz])
        elif obj_type == 'tall_box':
            w, h, d = random.uniform(0.15, 0.45), random.uniform(0.8, 1.8), random.uniform(0.15, 0.45)
            obj = trimesh.creation.box(extents=[w, h, d])
            pose = tf.translation_matrix([ox, floor_y + h/2, oz])
        elif obj_type == 'flat_box':
            w, h, d = random.uniform(0.5, 1.2), random.uniform(0.06, 0.25), random.uniform(0.5, 1.2)
            obj = trimesh.creation.box(extents=[w, h, d])
            pose = tf.translation_matrix([ox, floor_y + h/2, oz])
        elif obj_type == 'cylinder':
            r2 = random.uniform(0.12, 0.40)
            h  = random.uniform(0.3, 1.2)
            obj = trimesh.creation.cylinder(radius=r2, height=h, sections=20)
            pose = tf.translation_matrix([ox, floor_y + h/2, oz])
        else:
            r2 = random.uniform(0.12, 0.50)
            obj = trimesh.creation.icosphere(subdivisions=2, radius=r2)
            pose = tf.translation_matrix([ox, floor_y + r2, oz])

        pose = tf.concatenate_matrices(
            pose,
            tf.rotation_matrix(random.uniform(0, math.pi * 2), [0, 1, 0]),
        )
        add_mesh(py_scene, obj, col, pose, metallic=random.uniform(0.0, 0.4))


# ── Camera ─────────────────────────────────────────────────────────────────────

def random_camera_pose(centre, radius_range=(4, 7)):
    radius = random.uniform(*radius_range)
    theta  = random.uniform(0, 2 * math.pi)
    phi    = random.uniform(math.pi / 5, math.pi / 3)

    cam = np.array([
        centre[0] + radius * math.sin(phi) * math.cos(theta),
        centre[1] + radius * math.cos(phi),
        centre[2] + radius * math.sin(phi) * math.sin(theta),
    ])

    fwd = np.array(centre) - cam
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1, 0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 0, 1])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    pose = np.eye(4)
    pose[:3, 0] =  right
    pose[:3, 1] =  up
    pose[:3, 2] = -fwd
    pose[:3, 3] =  cam
    return pose, theta


# ── Lighting ───────────────────────────────────────────────────────────────────

def random_light_direction():
    theta = random.uniform(0, 2 * math.pi)
    # phi from vertical: 0.6 (~34°) to 1.05 (~60°) — oblique enough for clear
    # shadows without grazing angles that push shadows off screen.
    phi   = random.uniform(0.60, 1.05)
    return np.array([
        math.sin(phi) * math.cos(theta),
        math.cos(phi),
        math.sin(phi) * math.sin(theta),
    ])


def light_pose_from_direction(light_dir):
    z = np.array([0, 0, -1.0])
    d = light_dir / np.linalg.norm(light_dir)
    axis = np.cross(z, d)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np.eye(4)
    angle = math.acos(np.clip(np.dot(z, d), -1, 1))
    return tf.rotation_matrix(angle, axis / axis_norm)


# ── Augmentations ──────────────────────────────────────────────────────────────

def augment(img):
    img = img.astype(np.float32)

    # Contrast boost — pull midtones toward extremes
    if random.random() < 0.7:
        factor = random.uniform(1.2, 1.8)
        img = np.clip((img - 128) * factor + 128, 0, 255)

    # Brightness
    if random.random() < 0.5:
        img = np.clip(img * random.uniform(0.85, 1.15), 0, 255)

    # Noise
    if random.random() < 0.5:
        img = np.clip(img + np.random.normal(0, random.uniform(1, 5), img.shape), 0, 255)

    # Occlusion patch
    if random.random() < 0.25:
        h, w = img.shape[:2]
        px, py = random.randint(0, w-1), random.randint(0, h-1)
        pw, ph = random.randint(5, w//5), random.randint(5, h//5)
        img[py:py+ph, px:px+pw] *= random.uniform(0.0, 0.3)

    return img.astype(np.uint8)


# ── Main render function ───────────────────────────────────────────────────────

def render_sample(glb_path, img_size=(256, 256)):
    scene_trimesh = trimesh.load(glb_path)
    apply_random_pose(scene_trimesh)

    tint = mannequin_colour()

    try:
        bounds = scene_trimesh.bounds
        if bounds is None or not np.all(np.isfinite(bounds)):
            raise ValueError("bad bounds")
        floor_y  = float(bounds[0, 1])
        top_y    = float(bounds[1, 1])
        centre_x = float((bounds[0, 0] + bounds[1, 0]) / 2)
        centre_z = float((bounds[0, 2] + bounds[1, 2]) / 2)
        centre_y = floor_y + (top_y - floor_y) * 0.55
    except Exception:
        floor_y  = 0.0
        top_y    = 2.0
        centre_y = 1.1
        centre_x = 0.0
        centre_z = 0.0

    bg = random.choice(BG_COLOURS)
    # Moderate ambient so mannequin shadow-side stays visible and
    # matches the illumination level of the background objects.
    ambient = random.uniform(0.17, 0.30)
    py_scene = pyrender.Scene(
        ambient_light=np.array([ambient, ambient, ambient]),
        bg_color=np.array(bg),
    )

    # Mannequin
    for name, geom in scene_trimesh.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            mat = _mat(tint[0], tint[1], tint[2], roughness=0.75)
            mesh = pyrender.Mesh.from_trimesh(geom, material=mat, smooth=True)
            try:
                transform = scene_trimesh.graph.get(name)[0]
                if not np.all(np.isfinite(transform)):
                    transform = np.eye(4)
            except Exception:
                transform = np.eye(4)
            py_scene.add(mesh, pose=transform)

    # Camera
    centre = [float(centre_x), float(centre_y), float(centre_z)]
    cam_pose, cam_theta = random_camera_pose(centre)
    aspect = img_size[0] / img_size[1]
    camera = pyrender.PerspectiveCamera(
        yfov=math.radians(random.uniform(50, 62)), aspectRatio=aspect)
    py_scene.add(camera, pose=cam_pose)

    # 3D scene — pass figure centre so floor tiles are centred there
    build_scene(py_scene, floor_y=floor_y, cam_theta=cam_theta,
                centre_x=centre_x, centre_z=centre_z)

    # Main light — stronger so directional contrast is clearly visible.
    light_dir = random_light_direction()
    light = pyrender.DirectionalLight(
        color=[1.0, 1.0, 1.0], intensity=random.uniform(5.0, 9.0))
    py_scene.add(light, pose=light_pose_from_direction(light_dir))

    # Fill light (25%) — infrequent and weak so it never flattens the scene.
    # The main cause of "flat" images was fill firing too often at near-equal
    # intensity to the key light, cancelling its directionality.
    if random.random() < 0.25:
        fill_dir = -light_dir + np.random.uniform(-0.4, 0.4, 3)
        fill = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=random.uniform(0.1, 0.4))
        py_scene.add(fill, pose=light_pose_from_direction(fill_dir))

    # Rim light (25%)
    if random.random() < 0.25:
        rim_dir = np.array([-light_dir[0], light_dir[1] * 0.5, -light_dir[2]])
        rim = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=random.uniform(0.5, 2.0))
        py_scene.add(rim, pose=light_pose_from_direction(rim_dir))

    r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
    colour, _ = r.render(py_scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    r.delete()

    return augment(colour), light_dir


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int, default=50000,                 help='Number of images')
    parser.add_argument('--glb',  type=str, default='frontend/wooden.glb', help='Path to GLB')
    parser.add_argument('--out',  type=str, default='data/lighting',       help='Output directory')
    parser.add_argument('--size', type=int, default=256,                   help='Image size (square)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    img_size = (args.size, args.size)
    rows = []

    print(f"Generating {args.n} images at {img_size[0]}x{img_size[1]}...")

    for i in tqdm(range(args.n)):
        try:
            img, light = render_sample(args.glb, img_size)
            fname = f"{i:06d}.jpg"
            Image.fromarray(img).save(out_dir / 'images' / fname, quality=90)
            rows.append((fname,
                         round(float(light[0]), 6),
                         round(float(light[1]), 6),
                         round(float(light[2]), 6)))
        except Exception as e:
            print(f"[warn] sample {i} failed: {e}")

    with open(out_dir / 'labels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'light_x', 'light_y', 'light_z'])
        w.writerows(rows)

    print(f"Done. {len(rows)}/{args.n} saved to {out_dir}")


if __name__ == '__main__':
    main()