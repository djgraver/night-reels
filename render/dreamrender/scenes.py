"""Procedural dream scenes rendered with Blender Cycles (bpy).

This is the stand-in for a generative base-image model. Every scene is built
from primitives and procedural materials (no downloaded assets), and is
rendered with a depth pass so the fog stage can be depth-aware, which is what
a production version would get from a depth estimator (Depth Anything / MiDaS)
run on generated frames.

Run as:  python3 -m dreamrender.scenes spec.json outdir --frames 16 --res 960x540
"""
from __future__ import annotations
import json, math, os, random, sys, time
import bpy
from mathutils import Vector, Euler

# ------------------------------------------------------------------ helpers
R = random.Random(0)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def nodes_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes, nt.links, nt.nodes["Principled BSDF"]


def mat(name, color=(0.8, 0.8, 0.8), rough=0.6, metal=0.0, emit=None, strength=0.0,
        transmission=0.0, ior=1.45, spec=0.5, noise_var=0.0, noise_scale=8.0, bump=0.0, bump_scale=30.0,
        subsurface=0.0):
    m, nt, N, L, b = nodes_mat(name)
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["IOR"].default_value = ior
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Subsurface Weight"].default_value = subsurface
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    if noise_var > 0:  # break up flat colour with a noise-driven value shift
        tc = N.new("ShaderNodeTexCoord")
        nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = noise_scale
        nz.inputs["Detail"].default_value = 8
        L.new(tc.outputs["Object"], nz.inputs["Vector"])
        mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "OVERLAY"
        mix.inputs[0].default_value = noise_var
        mix.inputs[6].default_value = (*color, 1)
        L.new(nz.outputs["Color"], mix.inputs[7])
        L.new(mix.outputs[2], b.inputs["Base Color"])
    if bump > 0:
        tc = N.new("ShaderNodeTexCoord")
        nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = bump_scale
        nz.inputs["Detail"].default_value = 12
        L.new(tc.outputs["Object"], nz.inputs["Vector"])
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump
        L.new(nz.outputs["Fac"], bp.inputs["Height"]); L.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


def emissive(name, color, strength):
    m, nt, N, L, b = nodes_mat(name)
    N.remove(b)
    e = N.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1); e.inputs["Strength"].default_value = strength
    L.new(e.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def _finish(obj, material, name=None):
    if name:
        obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    return obj


def box(loc, size, material=None, rot=(0, 0, 0), bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = o.modifiers.new("bev", "BEVEL"); mod.width = bevel; mod.segments = 2
    return _finish(o, material)


def cyl(loc, r, h, material=None, rot=(0, 0, 0), verts=24, r2=None):
    if r2 is None:
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, rotation=rot, vertices=verts)
    else:
        bpy.ops.mesh.primitive_cone_add(radius1=r, radius2=r2, depth=h, location=loc, rotation=rot, vertices=verts)
    o = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    return _finish(o, material)


def sphere(loc, r, material=None, scale=(1, 1, 1), seg=32):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=seg, ring_count=seg // 2)
    o = bpy.context.active_object
    o.scale = scale
    bpy.ops.object.shade_smooth()
    return _finish(o, material)


def plane(loc, size, material=None, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.scale = (size[0], size[1], 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, material)


def grid(loc, size, subdiv, material=None):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv, y_subdivisions=subdiv, size=size, location=loc)
    o = bpy.context.active_object
    return _finish(o, material)


def displace(obj, strength, scale=1.0, kind="CLOUDS", depth=6, mid=0.5, basis="BLENDER_ORIGINAL", seed_offset=(0, 0)):
    tex = bpy.data.textures.new(f"tx{len(bpy.data.textures)}", kind)
    if kind == "CLOUDS":
        tex.noise_scale = scale; tex.noise_depth = depth; tex.noise_basis = basis
    elif kind == "MUSGRAVE":
        tex.musgrave_type = "RIDGED_MULTIFRACTAL"; tex.noise_scale = scale; tex.noise_basis = basis
        tex.octaves = depth; tex.lacunarity = 2.0; tex.dimension_max = 1.25; tex.offset = 1.0; tex.gain = 2.0
    mod = obj.modifiers.new("disp", "DISPLACE")
    mod.texture = tex; mod.strength = strength; mod.mid_level = mid
    mod.texture_coords = "GLOBAL"
    return mod


def point_light(loc, power, color=(1, 0.8, 0.6), radius=0.1):
    d = bpy.data.lights.new(f"pl{len(bpy.data.lights)}", "POINT")
    d.energy = power; d.color = color; d.shadow_soft_size = radius
    o = bpy.data.objects.new(d.name, d); o.location = loc
    bpy.context.collection.objects.link(o)
    return o


def area_light(loc, power, size, color=(1, 0.8, 0.6), rot=(0, 0, 0)):
    d = bpy.data.lights.new(f"al{len(bpy.data.lights)}", "AREA")
    d.energy = power; d.color = color; d.size = size
    o = bpy.data.objects.new(d.name, d); o.location = loc; o.rotation_euler = rot
    bpy.context.collection.objects.link(o)
    return o


def sun_light(direction_elev, direction_az, strength, color=(1, 0.95, 0.9), angle=0.02):
    d = bpy.data.lights.new("sun", "SUN")
    d.energy = strength; d.color = color; d.angle = angle
    o = bpy.data.objects.new("sun", d)
    # light shines along -Z of the object; aim it from (elev, az)
    o.rotation_euler = Euler((math.radians(90 - direction_elev), 0, math.radians(direction_az)), "XYZ")
    bpy.context.collection.objects.link(o)
    return o


def world_sky(elev, az, strength=1.0, air=1.0, aerosol=1.0, altitude=0.0):
    w = bpy.data.worlds.new("sky"); w.use_nodes = True
    N, L = w.node_tree.nodes, w.node_tree.links
    sky = N.new("ShaderNodeTexSky"); sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(elev); sky.sun_rotation = math.radians(az)
    sky.air_density = air; sky.aerosol_density = aerosol; sky.altitude = altitude
    bg = N["Background"]; bg.inputs["Strength"].default_value = strength
    L.new(sky.outputs[0], bg.inputs[0])
    bpy.context.scene.world = w
    return w


def world_flat(color, strength=1.0, top=None):
    w = bpy.data.worlds.new("flat"); w.use_nodes = True
    N, L = w.node_tree.nodes, w.node_tree.links
    bg = N["Background"]; bg.inputs["Strength"].default_value = strength
    if top is None:
        bg.inputs[0].default_value = (*color, 1)
    else:  # vertical gradient horizon->zenith
        tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ")
        ramp = N.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*color, 1); ramp.color_ramp.elements[1].color = (*top, 1)
        ramp.color_ramp.elements[1].position = 0.5
        L.new(tc.outputs["Generated"], sep.inputs[0]); L.new(sep.outputs["Z"], ramp.inputs[0])
        L.new(ramp.outputs[0], bg.inputs[0])
    bpy.context.scene.world = w
    return w


def figure(loc, height=1.75, material=None, facing=0.0, pose=0.0, wave=False):
    """Featureless human figure: dreams rarely give strangers faces."""
    x, y, z = loc
    s = height / 1.75
    parts = []
    m = material or mat("figure", (0.035, 0.032, 0.03), 0.8)
    parts.append(cyl((x, y, z + 1.18 * s), 0.19 * s, 0.62 * s, m, r2=0.15 * s))        # torso / coat
    parts.append(sphere((x, y, z + 1.62 * s), 0.115 * s, m, scale=(0.9, 1, 1.1)))     # head
    parts.append(cyl((x, y, z + 1.43 * s), 0.05 * s, 0.12 * s, m))                     # neck
    for side in (-1, 1):
        parts.append(cyl((x + side * 0.09 * s, y + pose * side * 0.05, z + 0.44 * s), 0.07 * s, 0.9 * s, m, r2=0.05 * s))
        if wave and side == 1:   # one arm raised, waving
            parts.append(cyl((x + 0.33 * s, y, z + 1.62 * s), 0.05 * s, 0.6 * s, m, rot=(0, 0.45, 0), r2=0.04 * s))
        else:
            parts.append(cyl((x + side * 0.235 * s, y, z + 1.12 * s), 0.05 * s, 0.6 * s, m,
                             rot=(0, side * 0.08, 0), r2=0.04 * s))
    parts.append(cyl((x, y, z + 0.78 * s), 0.2 * s, 0.35 * s, m, r2=0.19 * s))          # long coat hem
    for p in parts:
        p.rotation_euler.z += facing
    return parts


def bake(obj):
    """Apply modifiers so the mesh can be shared by many instances; hide the template."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    for m_ in obj.data.materials:
        if m_ and m_.name not in [x.name for x in me.materials if x]:
            me.materials.append(m_)
    bpy.data.objects.remove(obj)
    return me


def inst(mesh, loc, rot=(0, 0, 0), scale=(1, 1, 1)):
    o = bpy.data.objects.new("inst", mesh)
    o.location = loc; o.rotation_euler = rot; o.scale = scale
    bpy.context.collection.objects.link(o)
    return o


def camera(lens=32):
    c = bpy.data.cameras.new("cam"); c.lens = lens; c.clip_end = 3000; c.clip_start = 0.05
    o = bpy.data.objects.new("cam", c); bpy.context.collection.objects.link(o)
    bpy.context.scene.camera = o
    return o


def look_at(cam, loc, target, roll=0.0):
    cam.location = loc
    d = Vector(target) - Vector(loc)
    q = d.to_track_quat("-Z", "Y")
    cam.rotation_euler = q.to_euler()
    cam.rotation_euler.rotate_axis("Z", roll)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def colour_of(spec, key, default):
    from dreamrender.parse import COLOURS
    name = spec.get("object_colours", {}).get(key)
    return COLOURS.get(name, default) if name else default


# ------------------------------------------------------------------ scenes
# Every builder returns a camera path: dict(start=(loc, target), end=(loc, target), lens=, shake=)

def sky_for_time(tod, az=200.0, altitude=0.0, day_elev=32):
    if tod == "day":
        world_sky(day_elev, az, 0.9, altitude=altitude); sun_light(day_elev, az + 180, 4.0, (1, 0.93, 0.85))
    elif tod == "dusk":
        world_sky(2.5, az, 1.0, aerosol=2.5); sun_light(2.5, az + 180, 2.5, (1, 0.62, 0.35), angle=0.05)
    elif tod == "dawn":
        world_sky(4, az, 1.0, aerosol=1.8); sun_light(4, az + 180, 2.0, (1, 0.7, 0.6), angle=0.05)
    else:
        world_flat((0.006, 0.009, 0.02), 1.0, top=(0.001, 0.002, 0.006))
        sun_light(35, az + 180, 0.25, (0.6, 0.7, 1.0), angle=0.01)


def scene_hallway(spec):
    L_ = 34.0; W = 2.4; H = 2.9
    wall = mat("wallpaper", (0.42, 0.36, 0.28), 0.85, noise_var=0.35, noise_scale=40, bump=0.05, bump_scale=200)
    wains = mat("wainscot", (0.22, 0.13, 0.08), 0.45, noise_var=0.2, noise_scale=3)
    floor = mat("floor", (0.18, 0.1, 0.055), 0.32, noise_var=0.6, noise_scale=2.5, bump=0.05, bump_scale=60)
    ceil = mat("ceiling", (0.55, 0.52, 0.47), 0.9, noise_var=0.15)
    dcol = colour_of(spec, "door", (0.17, 0.1, 0.06))
    doorm = mat("door", dcol, 0.4, noise_var=0.25, noise_scale=4)
    frame = mat("frame", (0.6, 0.57, 0.5), 0.5)
    brass = mat("brass", (0.8, 0.6, 0.3), 0.3, metal=1)
    runner = mat("runner", (0.32, 0.05, 0.05), 0.95, noise_var=0.5, noise_scale=12, bump=0.1, bump_scale=300)
    plane((0, L_ / 2, 0), (W, L_ + 4), floor)
    plane((0, L_ / 2, 0.003), (1.0, L_ + 4), runner)
    plane((0, L_ / 2, H), (W, L_ + 4), ceil, rot=(math.pi, 0, 0))
    for side in (-1, 1):
        box((side * (W / 2 + 0.05), L_ / 2, H / 2), (0.1, L_ + 4, H), wall)
        box((side * (W / 2 - 0.01), L_ / 2, 0.5), (0.04, L_ + 4, 1.0), wains)
        box((side * (W / 2 - 0.02), L_ / 2, 1.01), (0.06, L_ + 4, 0.04), frame)
    box((0, L_ + 1.0, H / 2), (W, 0.1, H), wall)  # end wall
    box((0, -2.0, H / 2), (W, 0.1, H), wall)
    # doors down both sides, slightly irregular spacing (dream hallways don't measure right)
    y = 2.5
    while y < L_ - 1.5:
        for side in (-1, 1):
            yy = y + R.uniform(-0.4, 0.4)
            x = side * (W / 2 - 0.02)
            box((x, yy, 1.05), (0.06, 0.92, 2.1), doorm, bevel=0.01)
            for dz in (0.55, 1.5):
                box((x - side * 0.02, yy, dz), (0.02, 0.62, 0.7), doorm, bevel=0.02)
            box((x - side * 0.03, yy, 2.17), (0.05, 1.12, 0.1), frame)
            for dy in (-0.51, 0.51):
                box((x - side * 0.03, yy + dy, 1.08), (0.05, 0.1, 2.2), frame)
            sphere((x - side * 0.06, yy + 0.36, 1.0), 0.035, brass)
            if R.random() < 0.35:  # picture frames between doors
                py = yy + 1.5
                box((x, py, 1.6), (0.04, 0.55, 0.42), mat(f"pic{py}", (0.15, 0.1, 0.06), 0.4))
                box((x - side * 0.01, py, 1.6), (0.03, 0.45, 0.32),
                    mat(f"canvas{py}", tuple(R.uniform(0.15, 0.45) for _ in range(3)), 0.8, noise_var=0.8, noise_scale=6))
        y += R.uniform(3.0, 4.2)
    # end door with light leaking from beneath and through the gap
    ed = box((0, L_ + 0.93, 1.05), (0.95, 0.06, 2.1), doorm, bevel=0.01)
    ed.rotation_euler.z = 0.05
    box((0, L_ + 0.9, 0.02), (0.95, 0.02, 0.025), emissive("leak", (1.0, 0.75, 0.4), 60))
    box((0.49, L_ + 0.9, 1.05), (0.015, 0.02, 2.1), emissive("leak2", (1.0, 0.75, 0.4), 25))
    area_light((0, L_ + 0.7, 0.05), 25, 0.9, (1, 0.72, 0.4), rot=(math.radians(-80), 0, 0))
    # sparse dim ceiling lamps, one of them flickering out
    for i, ly in enumerate([4, 11, 18, 25]):
        sphere((0, ly, H - 0.12), 0.11, emissive(f"bulb{i}", (1, 0.78, 0.5), 18 if i != 2 else 1.5), scale=(1, 1, 0.6))
        if i != 2:
            point_light((0, ly, H - 0.3), 70, (1, 0.72, 0.45), 0.15)
    world_flat((0, 0, 0), 0)
    if spec.get("figures"):
        figure((0.25, L_ - 7, 0), 1.72, facing=math.pi)
    return dict(start=((0.05, 0.2, 1.62), (0, L_, 1.2)), end=((-0.1, 4.8, 1.6), (0.05, L_, 1.15)), lens=28, shake=0.6,
                subject=(0.25, L_ - 7, 1.2) if spec.get("figures") else (0, L_ + 0.9, 0.8))


def scene_beach(spec):
    tod = spec["time_of_day"]
    sand = mat("sand", (0.55, 0.48, 0.38), 0.95, noise_var=0.4, noise_scale=30, bump=0.3, bump_scale=80)
    s_ = grid((-110, 100, 2.6), 220, 400, sand)
    s_.rotation_euler.y = 0.025
    displace(s_, 0.3, 6.0)
    bpy.ops.object.shade_smooth()
    calm = any(w in spec["text"].lower() for w in ("still", "glass", "calm", "flat"))
    wm, nt, N, L, b = nodes_mat("sea")
    b.inputs["Base Color"].default_value = (0.006, 0.012, 0.02, 1); b.inputs["Roughness"].default_value = 0.04 if calm else 0.12
    b.inputs["IOR"].default_value = 1.33; b.inputs["Specular IOR Level"].default_value = 1.0
    tc = N.new("ShaderNodeTexCoord"); mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.08, 0.5, 1)
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.2; nz.inputs["Detail"].default_value = 8
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.05 if calm else 0.25
    L.new(tc.outputs["Object"], mp.inputs[0]); L.new(mp.outputs[0], nz.inputs["Vector"])
    L.new(nz.outputs["Fac"], bp.inputs["Height"]); L.new(bp.outputs["Normal"], b.inputs["Normal"])
    plane((1400, 100, 0.12), (2800, 4000), wm)
    moon_r = 45 if "huge" in spec["text"].lower() or "giant" in spec["text"].lower() else 18
    if tod == "night":
        world_flat((0.004, 0.007, 0.016), 1.0, top=(0.0005, 0.001, 0.004))
        moon = mat("moonsurf", (0.9, 0.88, 0.82), 1.0, emit=(1.0, 0.95, 0.85), strength=1.1, noise_var=0.8, noise_scale=2)
        mm = moon.node_tree.nodes; ml = moon.node_tree.links
        ml.new(mm["Principled BSDF"].inputs["Base Color"].links[0].from_socket, mm["Principled BSDF"].inputs["Emission Color"])
        sphere((520, 700, 75), moon_r * 2.2, moon, seg=48)
        sun_light(6, 215, 0.3, (0.7, 0.78, 1.0), angle=math.radians(3))
        # stars: tiny emissive specks
        star = emissive("star", (1, 1, 1), 40)
        sm = bake(sphere((0, 0, 0), 1.0, star, seg=8))
        for i in range(160):
            a, e = R.uniform(-1.4, 1.4), R.uniform(0.06, 0.9)
            d = 1500
            sz = R.uniform(0.7, 1.8)
            inst(sm, (d * math.sin(a) * math.cos(e), d * math.cos(a) * math.cos(e), d * math.sin(e)), scale=(sz, sz, sz))
    else:
        sky_for_time(tod, 180)
    if spec.get("figures"):
        figure((-8, 30, 0.45), 1.7, facing=0.4)
    rock = mat("rock", (0.08, 0.075, 0.07), 0.7, noise_var=0.5, bump=0.6, bump_scale=8)
    for i in range(7):
        r_ = sphere((R.uniform(-60, -12), R.uniform(10, 120), 0.3), R.uniform(0.6, 2.6), rock,
                    scale=(1, R.uniform(0.6, 1.2), R.uniform(0.35, 0.7)), seg=24)
        displace(r_, 0.4, 0.4)
    return dict(start=((-14, 0, 1.9), (330, 300, 24)), end=((-14.5, 6, 1.9), (320, 310, 24)), lens=26, shake=0.8,
                subject=(520, 700, 40))


def scene_mountain(spec):
    terrain_m, nt, N, L, b = nodes_mat("terrain")
    geo = N.new("ShaderNodeNewGeometry"); sep = N.new("ShaderNodeSeparateXYZ")
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.72; ramp.color_ramp.elements[0].color = (0.11, 0.1, 0.095, 1)
    ramp.color_ramp.elements[1].position = 0.86; ramp.color_ramp.elements[1].color = (0.05, 0.11, 0.025, 1)
    L.new(geo.outputs["Normal"], sep.inputs[0]); L.new(sep.outputs["Z"], ramp.inputs[0])
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.08; nz.inputs["Detail"].default_value = 10
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "OVERLAY"; mix.inputs["Factor"].default_value = 0.5
    L.new(ramp.outputs[0], mix.inputs[6]); L.new(nz.outputs["Color"], mix.inputs[7])
    # snow above a height line
    pos = N.new("ShaderNodeSeparateXYZ"); L.new(geo.outputs["Position"], pos.inputs[0])
    snow = N.new("ShaderNodeMapRange"); snow.inputs["From Min"].default_value = 95; snow.inputs["From Max"].default_value = 110
    L.new(pos.outputs["Z"], snow.inputs["Value"])
    mix2 = N.new("ShaderNodeMix"); mix2.data_type = "RGBA"
    L.new(snow.outputs[0], mix2.inputs[0]); L.new(mix.outputs[2], mix2.inputs[6])
    mix2.inputs[7].default_value = (0.85, 0.87, 0.9, 1)
    L.new(mix2.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.4
    nz2 = N.new("ShaderNodeTexNoise"); nz2.inputs["Scale"].default_value = 1.5; nz2.inputs["Detail"].default_value = 14
    L.new(nz2.outputs["Fac"], bp.inputs["Height"]); L.new(bp.outputs["Normal"], b.inputs["Normal"])

    t = grid((0, 0, 0), 1400, 420, terrain_m)
    displace(t, 70, 300, depth=5, kind="MUSGRAVE", mid=0.0, basis="IMPROVED_PERLIN")
    displace(t, 220, 450, depth=3, basis="BLENDER_ORIGINAL")
    displace(t, 10, 22, depth=4)
    bpy.ops.object.shade_smooth()
    dg = bpy.context.evaluated_depsgraph_get()
    me = t.evaluated_get(dg).to_mesh()
    vs = [t.matrix_world @ v.co for v in me.vertices]
    zs = sorted(v.z for v in vs)
    lake_z = zs[int(len(zs) * 0.18)]
    snow.inputs["From Min"].default_value = zs[int(len(zs) * 0.88)]
    snow.inputs["From Max"].default_value = zs[int(len(zs) * 0.93)]
    # camera on the highest point in a strip near the south edge, looking north over the valley
    strip = [v for v in vs if -520 < v.y < -380 and -150 < v.x < 150]
    peak = max(strip, key=lambda v: v.z)
    t.evaluated_get(dg).to_mesh_clear()
    water = mat("lake", (0.02, 0.05, 0.07), 0.02, spec=1.0, ior=1.33)
    plane((0, 0, lake_z), (1400, 1400), water)
    sky_for_time(spec["time_of_day"] if spec["time_of_day"] != "night" else "dusk", 75, altitude=1200, day_elev=14)
    cam = (peak.x, peak.y + 4, peak.z + 1.8)
    if spec.get("figures"):
        figure((peak.x + 3, peak.y + 14, peak.z - 0.6), 1.75)
    return dict(start=(cam, (peak.x - 60, peak.y + 500, lake_z + 10)),
                end=((cam[0] + 2, cam[1] + 3, cam[2]), (peak.x + 80, peak.y + 500, lake_z + 15)), lens=26, shake=0.3)


def scene_station(spec):
    concrete = mat("concrete", (0.38, 0.37, 0.35), 0.85, noise_var=0.45, noise_scale=3, bump=0.2, bump_scale=40)
    steel = mat("steel", (0.25, 0.25, 0.27), 0.35, metal=1)
    rail = mat("rail", (0.5, 0.48, 0.45), 0.25, metal=1)
    wood = mat("sleeper", (0.12, 0.09, 0.07), 0.9, noise_var=0.5, noise_scale=10)
    ballast = mat("ballast", (0.2, 0.19, 0.18), 1.0, noise_var=0.7, noise_scale=60, bump=0.8, bump_scale=120)
    roofm = mat("roof", (0.15, 0.16, 0.17), 0.6)
    paint = mat("paint", (0.75, 0.62, 0.2), 0.6)
    Lp = 90
    box((0, Lp / 2 - 10, 0.5), (6, Lp, 1), concrete)
    box((2.85, Lp / 2 - 10, 1.003), (0.3, Lp, 0.01), paint)
    box((15, Lp / 2 - 10, 0.5), (6, Lp, 1), concrete)
    plane((7.5, Lp / 2 - 10, -0.1), (9, Lp + 200), ballast)
    for tx in (5.2, 9.8):
        for dx in (-0.72, 0.72):
            box((tx + dx, Lp / 2 + 40, 0.05), (0.08, Lp + 200, 0.14), rail)
        sl = bake(box((0, 0, 0), (2.2, 0.24, 0.1), wood))
        for i in range(-20, 200):
            inst(sl, (tx, i * 0.75, -0.04))
    for px, w in ((0, 6.5), (15, 6.5)):
        box((px, 22, 5.2), (w, 50, 0.25), roofm)
        for i in range(9):
            cyl((px - 1.8 if px == 0 else px + 1.8, -2 + i * 6, 3.1), 0.14, 4.2, steel)
    lamp = emissive("lamp", (1, 0.85, 0.6), 30)
    for i in range(8):
        sphere((0.6, -1 + i * 6, 4.6), 0.16, lamp)
        point_light((0.6, -1 + i * 6, 4.3), 120, (1, 0.8, 0.55), 0.2)
    # the clock
    cy = 14
    cyl((0, cy, 4.85), 0.03, 0.6, steel)
    for i in range(40):  # far side: warehouses and poplars
        bx = R.uniform(25, 120); by = R.uniform(-40, 160); bh = R.uniform(4, 16)
        box((bx, by, bh / 2), (R.uniform(6, 20), R.uniform(8, 24), bh), concrete)
    plane((60, 60, -0.15), (300, 400), mat("field", (0.12, 0.12, 0.1), 1, noise_var=0.5, noise_scale=0.2))
    face_col = colour_of(spec, "clock", (0.9, 0.88, 0.8))
    cyl((0, cy, 4.0), 0.85, 0.14, mat("clockrim", (0.05, 0.05, 0.05), 0.3, metal=1), rot=(math.pi / 2, 0, 0), verts=48)
    for s_ in (-1, 1):
        cyl((0, cy + s_ * 0.075, 4.0), 0.78, 0.01, mat("clockface", face_col, 0.4, emit=face_col, strength=1.2),
            rot=(math.pi / 2, 0, 0), verts=48)
        hand = mat("hand", (0.02, 0.02, 0.02), 0.5)
        box((0.18, cy + s_ * 0.085, 4.12), (0.45, 0.01, 0.04), hand, rot=(0, -0.6, 0))
        box((0.0, cy + s_ * 0.085, 4.28), (0.03, 0.01, 0.6), hand, rot=(0, 0.02, 0))
        for k in range(12):
            a = k * math.pi / 6
            box((0.68 * math.sin(a), cy + s_ * 0.085, 4.0 + 0.68 * math.cos(a)), (0.04, 0.01, 0.1), hand, rot=(0, a, 0))
    bench = mat("bench", (0.15, 0.1, 0.07), 0.6)
    for by in (4, 22, 33):
        box((-1.9, by, 1.45), (0.5, 2.0, 0.06), bench); box((-2.15, by, 1.75), (0.06, 2.0, 0.5), bench)
    if spec.get("figures") or "figure" in spec.get("objects", []):
        figure((0.3, cy + 0.6, 1.0), 1.78, facing=math.pi + 0.3)
    sky_for_time(spec["time_of_day"], 175)
    return dict(start=((-1.0, 0, 2.6), (0.2, 30, 3.0)), end=((-0.8, 3.5, 2.6), (0.1, 30, 3.0)), lens=30, shake=0.5,
                subject=(0.2, cy + 0.3, 2.8))


def window_facade_mat(name, base, lit_frac=0.35, warm=(1, 0.7, 0.4)):
    m, nt, N, L, b = nodes_mat(name)
    tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ"); comb = N.new("ShaderNodeCombineXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    add = N.new("ShaderNodeMath"); add.operation = "ADD"
    L.new(sep.outputs["X"], add.inputs[0]); L.new(sep.outputs["Y"], add.inputs[1])
    L.new(add.outputs[0], comb.inputs["X"]); L.new(sep.outputs["Z"], comb.inputs["Y"])
    br = N.new("ShaderNodeTexBrick")
    br.inputs["Scale"].default_value = 0.3; br.inputs["Mortar Size"].default_value = 0.4
    br.inputs["Brick Width"].default_value = 1.0; br.inputs["Row Height"].default_value = 1.05
    br.offset = 0.0; br.squash = 1.0
    br.inputs["Color1"].default_value = (0, 0, 0, 1); br.inputs["Color2"].default_value = (1, 1, 1, 1)
    br.inputs["Mortar"].default_value = (0, 0, 0, 1)
    L.new(comb.outputs[0], br.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB"); ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[1].position = 1 - lit_frac
    L.new(br.outputs["Color"], ramp.inputs[0])
    b.inputs["Base Color"].default_value = (*base, 1); b.inputs["Roughness"].default_value = 0.8
    b.inputs["Emission Color"].default_value = (*warm, 1)
    mul = N.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 6
    sepc = N.new("ShaderNodeSeparateColor"); L.new(ramp.outputs[0], sepc.inputs[0])
    L.new(sepc.outputs[0], mul.inputs[0]); L.new(mul.outputs[0], b.inputs["Emission Strength"])
    return m


def scene_flooded_city(spec):
    facades = [window_facade_mat(f"fac{i}", c, R.uniform(0.15, 0.4), w) for i, (c, w) in enumerate([
        ((0.25, 0.22, 0.2), (1, 0.7, 0.4)), ((0.18, 0.18, 0.2), (1, 0.8, 0.55)),
        ((0.3, 0.26, 0.22), (0.9, 0.75, 1.0)), ((0.15, 0.14, 0.15), (1, 0.6, 0.35))])]
    for side in (-1, 1):
        y = -10
        while y < 260:
            w = R.uniform(7, 14); h = R.uniform(14, 48); d = 12
            box((side * (9 + d / 2), y + w / 2, h / 2 - 6), (d, w - 0.4, h), R.choice(facades))
            y += w
    water_level = 3.0
    wm, nt, N, L, b = nodes_mat("floodwater")
    b.inputs["Base Color"].default_value = (0.05, 0.08, 0.1, 1); b.inputs["Roughness"].default_value = 0.06
    b.inputs["Transmission Weight"].default_value = 1.0; b.inputs["IOR"].default_value = 1.33
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 2.5; nz.inputs["Detail"].default_value = 6
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.08
    L.new(nz.outputs["Fac"], bp.inputs["Height"]); L.new(bp.outputs["Normal"], b.inputs["Normal"])
    vol = N.new("ShaderNodeVolumeAbsorption"); vol.inputs["Density"].default_value = 0.12
    vol.inputs["Color"].default_value = (0.25, 0.55, 0.5, 1)
    L.new(vol.outputs[0], N["Material Output"].inputs["Volume"])
    box((0, 120, water_level - 6), (18, 300, 12), wm)
    plane((0, 120, -8.5), (40, 300), mat("streetbed", (0.1, 0.1, 0.1), 0.9))
    pole = mat("pole", (0.06, 0.06, 0.07), 0.4, metal=1)
    lamp = emissive("streetlamp", (1, 0.75, 0.45), 40)
    for i in range(14):
        for side in (-1, 1):
            y = 6 + i * 13 + (6 if side > 0 else 0)
            under = (i + (side > 0)) % 3 == 0 or i < 2
            top = 2.0 if under else 6.2
            x = side * 7.5
            cyl((x, y, (top - 8) / 2), 0.08, top + 8, pole)
            sphere((x - side * 0.6, y, top), 0.22, lamp)
            box((x - side * 0.3, y, top + 0.15), (0.65, 0.06, 0.06), pole)
            point_light((x - side * 0.6, y, top - 0.2), 260 if not under else 400, (1, 0.72, 0.42), 0.25)
    # a drowned car and a bus stop sign for scale
    box((-3, 30, -0.3), (2, 4.4, 1.4), mat("car", (0.35, 0.08, 0.08), 0.3, metal=0.6), bevel=0.3)
    tod = spec["time_of_day"]
    if tod == "night":
        sky_for_time("night")
    else:
        world_sky(1.0, 180, 1.2, aerosol=3.0)
        sun_light(1.0, 0, 1.5, (1, 0.55, 0.5), angle=0.08)
    return dict(start=((1.0, -6, water_level + 0.35), (0, 120, 10)), end=((-0.6, 4, water_level + 0.25), (-2, 120, 9)),
                lens=24, shake=1.2, subject=(0, 60, water_level + 1))


def scene_forest(spec):
    ground = mat("litter", (0.09, 0.07, 0.045), 0.95, noise_var=0.7, noise_scale=6, bump=0.5, bump_scale=25)
    g = grid((0, 40, 0), 260, 200, ground)
    displace(g, 1.6, 8)
    bark = mat("bark", (0.06, 0.055, 0.05), 0.9, noise_var=0.5, noise_scale=6, bump=1.0, bump_scale=5)
    needles = mat("needles", (0.03, 0.06, 0.035), 0.9, noise_var=0.5, noise_scale=3, bump=0.6, bump_scale=10)
    trunk_meshes = [bake(cyl((0, 0, 0.5), 1.0, 1.0, bark, r2=0.35, verts=12)) for _ in range(1)]
    cone_meshes = []
    for v in range(5):
        c = cyl((0, 0, 0), 1.0, 1.0, needles, r2=0.04, verts=20)
        c.modifiers.new("sub", "SUBSURF").levels = 1
        displace(c, 0.18, 0.25 + 0.05 * v)
        cone_meshes.append(bake(c))
    for i in range(340):
        x, y = R.uniform(-70, 70), R.uniform(-15, 130)
        if abs(x) < 2.6 + 0.04 * max(y, 0) and y < 30:
            continue  # leave a path towards the door
        r_ = R.uniform(0.16, 0.5); h = R.uniform(18, 32)
        inst(trunk_meshes[0], (x, y, h / 2 - 0.6), (R.uniform(-0.03, 0.03), R.uniform(-0.03, 0.03), R.uniform(0, 6.3)), (r_, r_, h))
        if R.random() < 0.6:
            base = R.uniform(5, 10)
            for k in range(5):
                z = base + k * (h - base) / 5.2
                rad = (h - z) * 0.26 + 0.6
                inst(R.choice(cone_meshes), (x + R.uniform(-0.2, 0.2), y + R.uniform(-0.2, 0.2), z + 2.2),
                     (R.uniform(-0.08, 0.08), R.uniform(-0.08, 0.08), R.uniform(0, 6.3)), (rad, rad, 4.6))
    dcol = colour_of(spec, "door", (0.2, 0.12, 0.08))
    frame = mat("doorframe", (0.3, 0.22, 0.16), 0.5, noise_var=0.2)
    dm = mat("forestdoor", dcol, 0.45, noise_var=0.2, noise_scale=3)
    dy = 16
    box((-0.55, dy, 1.1), (0.1, 0.12, 2.3), frame); box((0.55, dy, 1.1), (0.1, 0.12, 2.3), frame)
    box((0, dy, 2.28), (1.2, 0.12, 0.12), frame)
    # door slab + raised panels joined into one object, hinged on the left and open a crack
    parts = [box((0.0, dy, 1.05), (0.92, 0.05, 2.1), dm)]
    for dz in (0.55, 1.5):
        parts.append(box((0.0, dy - 0.03, dz), (0.62, 0.02, 0.72), dm))
    parts.append(sphere((0.36, dy - 0.07, 1.0), 0.035, mat("knob", (0.8, 0.6, 0.3), 0.3, metal=1)))
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    door = bpy.context.active_object
    bpy.context.scene.cursor.location = (-0.46, dy, 0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    door.rotation_euler.z = -0.3          # free edge swings towards the camera, light spills through the gap
    bpy.ops.object.select_all(action="DESELECT")
    plane((0, dy + 0.08, 1.05), (0.95, 2.1), emissive("beyond", (1.0, 0.82, 0.55), 6), rot=(math.pi / 2, 0, 0))
    area_light((0.3, dy + 0.5, 1.1), 40, 0.6, (1, 0.8, 0.55), rot=(math.radians(90), 0, math.pi))
    if spec.get("figures"):
        figure((5, dy + 9, 0), 1.75, facing=math.pi + 0.6)
    tod = spec["time_of_day"]
    sky_for_time(tod if tod != "day" else "dusk", 20)
    return dict(start=((0.3, -2, 1.7), (0, dy, 1.4)), end=((0.0, 6.0, 1.66), (0, dy, 1.3)), lens=30, shake=0.7,
                subject=(0, dy, 1.1))


def scene_stairs(spec):
    marble = mat("marble", (0.85, 0.83, 0.8), 0.25, noise_var=0.2, noise_scale=4, subsurface=0.1)
    n = 46
    for i in range(n):
        a = i * 0.11
        r = 9
        x, y, z = r * math.sin(a) - 0, r * (1 - math.cos(a)) + i * 0.35, i * 0.32
        box((x, y + 5, z), (2.4, 0.5, 0.2), marble, rot=(0, 0, a), bevel=0.02)
    # cloud sea + puffs as metaball clusters with subsurface-white material
    cloud = mat("cloud", (0.98, 0.9, 0.9), 0.9, subsurface=1.0)
    cloud.node_tree.nodes["Principled BSDF"].inputs["Subsurface Radius"].default_value = (2.0, 2.0, 2.2)
    cloud.node_tree.nodes["Principled BSDF"].inputs["Subsurface Scale"].default_value = 3.0
    sea = grid((0, 60, -6), 600, 260, cloud)
    displace(sea, 7, 9, depth=5)
    displace(sea, 2, 2.5, depth=3)
    bpy.ops.object.shade_smooth()
    mb = bpy.data.metaballs.new("puffs"); mb.resolution = 0.6; mb.render_resolution = 0.5
    mo = bpy.data.objects.new("puffs", mb); bpy.context.collection.objects.link(mo)
    mb.materials.append(cloud)
    for c in range(14):
        cx, cy, cz = R.uniform(-60, 60), R.uniform(25, 160), R.uniform(-3, 22)
        for k in range(10):
            e = mb.elements.new(); e.co = (cx + R.uniform(-6, 6), cy + R.uniform(-4, 4), cz + R.uniform(-1.5, 2.5))
            e.radius = R.uniform(3, 6.5)
    tod = spec["time_of_day"]
    if tod == "day":   # golden light makes the cloud sea read as colour, not white
        world_sky(4, 250, 1.0, aerosol=1.5, altitude=3000); sun_light(4, 250 + 180, 5.0, (1, 0.68, 0.45), angle=0.03)
    else:
        sky_for_time(tod if tod != "night" else "dusk", 250, altitude=3000)
    if spec.get("figures"):
        figure((3.2, 14, 4.9), 1.7, facing=math.pi)
    return dict(start=((-2.2, 0.5, 0.9), (1.5, 16, 5.5)), end=((-1.6, 2.5, 1.6), (2.5, 16, 6.5)), lens=22, shake=0.9,
                subject=(1.5, 14, 4.5))


def scene_street(spec):
    """Night city street, wet asphalt, rain. With motion 'falling' the camera drops between the buildings."""
    facades = [window_facade_mat(f"sfac{i}", c, R.uniform(0.08, 0.22), w) for i, (c, w) in enumerate([
        ((0.2, 0.19, 0.18), (1, 0.75, 0.45)), ((0.14, 0.14, 0.16), (1, 0.85, 0.6)), ((0.24, 0.22, 0.2), (0.85, 0.8, 1.0))])]
    for side in (-1, 1):
        y = -10
        while y < 260:
            w = R.uniform(7, 14); h = R.uniform(16, 55); d = 12
            box((side * (9 + d / 2), y + w / 2, h / 2), (d, w - 0.4, h), R.choice(facades))
            y += w
    am, nt, N, L, b = nodes_mat("wet_asphalt")
    b.inputs["Base Color"].default_value = (0.03, 0.03, 0.032, 1)
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.35; nz.inputs["Detail"].default_value = 4
    ramp = N.new("ShaderNodeValToRGB")          # puddles: mirror-smooth patches in rough asphalt
    ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[0].color = (0.06, 0.06, 0.06, 1)
    ramp.color_ramp.elements[1].position = 0.42; ramp.color_ramp.elements[1].color = (0.5, 0.5, 0.5, 1)
    L.new(nz.outputs["Fac"], ramp.inputs[0]); L.new(ramp.outputs[0], b.inputs["Roughness"])
    bn = N.new("ShaderNodeTexNoise"); bn.inputs["Scale"].default_value = 40; bn.inputs["Detail"].default_value = 10
    bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.15
    L.new(bn.outputs["Fac"], bp.inputs["Height"]); L.new(bp.outputs["Normal"], b.inputs["Normal"])
    plane((0, 120, 0), (18, 320), am)
    curb = mat("curb", (0.2, 0.2, 0.2), 0.6, noise_var=0.4)
    for side in (-1, 1):
        box((side * 7.0, 120, 0.08), (4, 320, 0.16), curb)
    pole = mat("spole", (0.05, 0.05, 0.06), 0.4, metal=1)
    lamp = emissive("slamp", (1, 0.8, 0.55), 35)
    for i in range(14):
        for side in (-1, 1):
            y = 6 + i * 14 + (7 if side > 0 else 0); x = side * 7.6
            cyl((x, y, 3.2), 0.08, 6.4, pole)
            sphere((x - side * 0.7, y, 6.2), 0.2, lamp)
            point_light((x - side * 0.7, y, 6.0), 300, (1, 0.75, 0.48), 0.25)
    rain = emissive("rain", (0.8, 0.85, 0.9), 0.12)
    rm = bake(cyl((0, 0, 0), 0.003, 0.45, rain, verts=4))
    falling = spec.get("motion") == "falling"
    for i in range(900 if not falling else 1500):
        if falling:
            loc = (R.uniform(-9, 9), R.uniform(20, 70), R.uniform(2, 60))
        else:
            loc = (R.uniform(-8, 8), R.uniform(2, 45), R.uniform(0, 9))
        inst(rm, loc, (0.05, 0.02, 0))
    sky_for_time("night")
    if spec.get("figures"):
        figure((1.2, 30, 0.0), 1.8, facing=math.pi)
    if falling:
        return dict(start=((0.5, 30, 62), (0, 46, 0)), end=((0.2, 33, 40), (0, 48, 0)), lens=22, shake=1.4,
                    subject=(0, 46, 0))
    return dict(start=((0.8, -4, 1.7), (0, 60, 2.2)), end=((0.2, 3, 1.75), (-0.2, 60, 2.0)), lens=28, shake=1.3,
                subject=(1.2, 30, 1.2))


def scene_field(spec):
    """Tall grass at sunset (Cycles hair), birds, a far figure."""
    soil = mat("soil", (0.12, 0.1, 0.05), 0.95, noise_var=0.5, noise_scale=0.3)
    g = grid((0, 120, 0), 700, 220, soil)
    displace(g, 9, 120, depth=3)
    bpy.ops.object.shade_smooth()
    near = grid((0, 20, 0), 90, 140, soil)
    displace(near, 9, 120, depth=3)
    near.location.z = 0.02
    gm, nt, N, L, b = nodes_mat("grass")
    info = N.new("ShaderNodeHairInfo"); ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.08, 0.09, 0.02, 1); ramp.color_ramp.elements[1].color = (0.6, 0.5, 0.2, 1)
    L.new(info.outputs["Intercept"], ramp.inputs[0]); L.new(ramp.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Transmission Weight"].default_value = 0.0
    b.inputs["Subsurface Weight"].default_value = 0.3
    near.data.materials.append(gm)
    ps = near.modifiers.new("grass", "PARTICLE_SYSTEM").particle_system
    st = ps.settings
    st.type = "HAIR"; st.count = 30000; st.hair_length = 0.9; st.use_advanced_hair = False
    st.material = 2
    st.factor_random = 0.4; st.brownian_factor = 0.05
    st.child_type = "INTERPOLATED"; st.rendered_child_count = 10; st.child_length = 1.0; st.child_length_threshold = 0.3
    st.child_radius = 0.4; st.roughness_1 = 0.1; st.roughness_2 = 0.15
    st.kink = "CURL" if False else "NO"
    st.root_radius = 0.6; st.tip_radius = 0.0; st.radius_scale = 0.01
    st.display_step = 2; st.render_step = 3
    st.normal_factor = 1.0; st.object_align_factor[2] = 0.0
    st.use_rotations = True
    st.hair_length = 1.25     # must be set last: toggling hair options resets it
    bird = mat("bird", (0.02, 0.02, 0.02), 0.8)
    bm = bake(box((0, 0, 0), (0.6, 0.08, 0.03), bird))
    for i in range(16):
        cx, cy, cz = R.uniform(-22, 18), R.uniform(22, 55), R.uniform(5, 13)
        for side in (-1, 1):
            inst(bm, (cx + side * 0.27, cy, cz + 0.08), (0, side * R.uniform(0.25, 0.6), 0))
    world_sky(2.0, 0, 1.0, aerosol=2.2)
    sun_light(2.0, 180, 3.0, (1, 0.6, 0.35), angle=0.06)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    def ground(x, y):
        hit, loc, *_ = bpy.context.scene.ray_cast(dg, Vector((x, y, 200)), Vector((0, 0, -1)))
        return loc.z if hit else 0.0
    if spec.get("figures"):
        figure((-4, 38, ground(-4, 38) - 0.1), 1.8, wave=True)
    z0, z1, zt = ground(0.5, -8), ground(0.2, -4), ground(-4, 38)
    return dict(start=((0.5, -8, z0 + 1.6), (-4, 60, zt + 1.4)), end=((0.2, -4, z1 + 1.65), (-5, 60, zt + 1.3)),
                lens=32, shake=0.6, subject=(-4, 38, zt + 1.4))


BUILDERS = {"hallway": scene_hallway, "beach": scene_beach, "mountain": scene_mountain, "station": scene_station,
            "flooded_city": scene_flooded_city, "forest": scene_forest, "stairs": scene_stairs,
            "street": scene_street, "field": scene_field}


# ------------------------------------------------------------------ render
def _exr(s):
    s.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
    s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    s.render.image_settings.color_depth = "32"
    s.render.image_settings.exr_codec = "ZIP"


def depth_override():
    """Emission = camera distance, used for a fast second 'depth' render (EXR passes aren't saved by bpy 5.3)."""
    m, nt, N, L, b = nodes_mat("depth_override")
    N.remove(b)
    cd = N.new("ShaderNodeCameraData"); e = N.new("ShaderNodeEmission")
    L.new(cd.outputs["View Distance"], e.inputs["Strength"])
    L.new(e.outputs[0], N["Material Output"].inputs["Surface"])
    return m


def render_depth(s, path, samples):
    vl = s.view_layers[0]
    keep = dict(samples=s.cycles.samples, den=s.cycles.use_denoising, fw=s.cycles.pixel_filter_type,
                transp=s.render.film_transparent, exp=s.view_settings.exposure, vt=s.view_settings.view_transform,
                adapt=s.cycles.use_adaptive_sampling)
    vl.material_override = DEPTH_MAT
    s.cycles.samples = 1; s.cycles.use_denoising = False; s.cycles.pixel_filter_type = "BOX"
    s.cycles.use_adaptive_sampling = False
    s.render.film_transparent = True; s.view_settings.exposure = 0
    s.render.image_settings.media_type = "IMAGE"; s.render.image_settings.file_format = "OPEN_EXR"
    s.render.image_settings.color_depth = "32"
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    vl.material_override = None
    s.cycles.samples = keep["samples"]; s.cycles.use_denoising = keep["den"]; s.cycles.pixel_filter_type = keep["fw"]
    s.render.film_transparent = keep["transp"]; s.view_settings.exposure = keep["exp"]
    s.cycles.use_adaptive_sampling = keep["adapt"]


def setup_render(res, samples):
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"
    s.cycles.samples = samples; s.cycles.use_denoising = True; s.cycles.denoiser = "OPENIMAGEDENOISE"
    s.cycles.use_adaptive_sampling = True; s.cycles.adaptive_threshold = 0.03
    s.cycles.max_bounces = 6; s.cycles.transmission_bounces = 6; s.cycles.volume_bounces = 1
    s.cycles.caustics_reflective = False; s.cycles.caustics_refractive = False
    s.cycles.blur_glossy = 1.0
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.view_settings.view_transform = "AgX"
    s.view_settings.look = "AgX - Medium High Contrast"
    vl = s.view_layers[0]
    vl.use_pass_z = True
    _exr(s)


def path_at(path, t, frame):
    """Eased camera path plus a slow, breath-like sway."""
    e = 0.5 - 0.5 * math.cos(math.pi * t)
    loc = list(lerp(path["start"][0], path["end"][0], e))
    tgt = list(lerp(path["start"][1], path["end"][1], e))
    k = path.get("shake", 0.5)
    loc[0] += k * 0.03 * math.sin(frame * 0.7 + 1.3)
    loc[2] += k * 0.025 * math.sin(frame * 1.1)
    roll = k * 0.012 * math.sin(frame * 0.45 + 0.4)
    return loc, tgt, roll


def subject_on_screen(cam, subject):
    """Normalised (x, y-down) screen position of the dream's subject, for the stylizer's attention island."""
    try:
        from bpy_extras.object_utils import world_to_camera_view
        bpy.context.view_layer.update()
        v = world_to_camera_view(bpy.context.scene, cam, Vector(subject))
        if v.z <= 0 or not (-0.2 < v.x < 1.2 and -0.2 < v.y < 1.2):
            return None
        return [round(v.x, 4), round(1 - v.y, 4)]
    except Exception:
        return None


def render(spec: dict, outdir: str, frames: int = 16, res=(960, 540), samples=48, exposure=0.0, hero_res=None,
           meta_only=False):
    global R
    R = random.Random(spec.get("seed", 0))
    reset()
    path = BUILDERS[spec["setting"]](spec)
    cam = camera(path.get("lens", 30))
    setup_render(res, samples)
    global DEPTH_MAT
    DEPTH_MAT = depth_override()
    s = bpy.context.scene
    s.view_settings.exposure = exposure + {"night": 0.6, "dusk": -0.3, "dawn": -0.3, "day": -1.3}[spec["time_of_day"]] \
        + {"hallway": 0.6, "flooded_city": 0.3, "beach": 0.8, "stairs": -0.6, "field": -0.5, "street": 0.3}.get(spec["setting"], 0.0)
    os.makedirs(outdir, exist_ok=True)
    meta = dict(camera=path, frames=[])
    for i in range(frames):
        t = i / max(frames - 1, 1)
        loc, tgt, roll = path_at(path, t, i)
        look_at(cam, loc, tgt, roll)
        if hero_res and i == frames // 2:
            pass
        base = os.path.join(outdir, f"f{i:03d}")
        focus = subject_on_screen(cam, path["subject"]) if path.get("subject") else None
        if meta_only:
            meta["frames"].append(dict(i=i, focus=focus))
            continue
        t0 = time.time()
        bpy.ops.render.render(write_still=False)
        # also save the display-referred (AgX) beauty as PNG
        img = bpy.data.images["Render Result"]
        s.render.image_settings.media_type = "IMAGE"
        s.render.image_settings.file_format = "PNG"; s.render.image_settings.color_depth = "16"
        img.save_render(base + ".png", scene=s)
        render_depth(s, base + "_depth.exr", samples)
        _exr(s)
        meta["frames"].append(dict(i=i, seconds=round(time.time() - t0, 1), focus=focus))
        print(f"[{spec['setting']}] frame {i + 1}/{frames} {time.time() - t0:.1f}s", flush=True)
    if meta_only and os.path.exists(os.path.join(outdir, "render_meta.json")):
        old = json.load(open(os.path.join(outdir, "render_meta.json")))
        for a, b in zip(old.get("frames", []), meta["frames"]):
            b["seconds"] = a.get("seconds")
    with open(os.path.join(outdir, "render_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=list)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("outdir")
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--res", default="960x540")
    ap.add_argument("--samples", type=int, default=48); ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--meta-only", action="store_true", help="only write camera/subject metadata")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    render(spec, a.outdir, a.frames, tuple(int(v) for v in a.res.split("x")), a.samples, a.exposure, meta_only=a.meta_only)
