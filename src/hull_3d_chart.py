"""
Interactive 3D archetype scene (WebGL / three.js), for The 8 Player Types.

AI-ASSISTED (Claude Code, chat) - Prompt: "我想在3D图像上显示这个球员头像 ... 当我
用波轮扩大这个图像内部的时候 那些黑点应该也显示球员 这样会不会更好呢".
Plotly's scatter3d can't put an image on a 3D marker (no image markers, and an
HTML overlay can't track 3D rotation), so this replaces it with three.js, where
every player is a billboard SPRITE - a small circular headshot that always faces
the camera and so stays legible while the scene rotates and zooms.

Design:
  - The 8 archetype CORNERS are always shown as headshot sprites with a colored
    ring and the player's name - each corner is a real player, which is the whole
    point of ADA.
  - The ~430 other players are a colored point cloud when zoomed out; scroll in
    (OrbitControls distance below a threshold) and the dots become their own
    headshot sprites, so exploring the interior reveals who each point is.
  - Faint axes + a bounding box give the rotation a spatial reference; a Reset
    button returns the camera to its default framing.

All faces are embedded as same-origin data URIs (WebGL rejects cross-origin
textures without CORS, and the NBA CDN sends none) - see
src/pipeline/precompute_player_thumbnails.py. three.js + OrbitControls load from
CDN, the same way the old Plotly chart did inside its st.iframe.
Not AI: the idea (faces in 3D; dots reveal players on zoom) - the owner's own.
"""

from __future__ import annotations

import json

_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  html,body{margin:0;padding:0;background:__BG__;overflow:hidden;
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;}
  #cvs{display:block;width:100%;height:100%;}
  #tip{position:fixed;display:none;z-index:20;pointer-events:none;background:__CARD_BG__;
    border:1px solid __LINE__;border-radius:8px;padding:8px 10px;box-shadow:0 6px 20px rgba(0,0,0,.18);
    display:none;align-items:center;gap:9px;}
  #tip img{width:40px;height:40px;border-radius:50%;object-fit:cover;}
  #tip .nm{font-size:13px;font-weight:700;color:__INK__;line-height:1.2;}
  #tip .sub{font-size:11px;color:__MUTED__;margin-top:2px;}
  #reset{position:absolute;left:12px;bottom:12px;z-index:20;cursor:pointer;
    background:__CARD_BG__;border:1px solid __LINE__;border-radius:7px;padding:6px 12px;
    font:600 12px system-ui,sans-serif;color:__INK__;}
  #reset:hover{background:#efeae0;}
  #hint{position:absolute;right:12px;bottom:12px;z-index:20;font:11px system-ui,sans-serif;color:__MUTED__;}
</style></head>
<body>
<div id="wrap">
  <canvas id="cvs"></canvas>
  <button id="reset">Reset view</button>
  <div id="hint">drag to rotate · scroll to zoom in for faces</div>
  <div id="tip"></div>
</div>
<script>
const PLAYERS = __PLAYERS__;   // {x,y,z,name,arch,pct,thumb}
const CORNERS = __CORNERS__;   // {x,y,z,name,label,color,thumb}
const COLORS  = __COLORS__;    // per-archetype hex
const EDGES   = __EDGES__;     // [[i,j],...] into CORNERS
const BG = "__BG__";

const wrap = document.getElementById("wrap");
const canvas = document.getElementById("cvs");
const tip = document.getElementById("tip");

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth/wrap.clientHeight, 0.01, 100);
const CAM0 = new THREE.Vector3(2.1, 1.7, 2.1);
camera.position.copy(CAM0);
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(wrap.clientWidth, wrap.clientHeight);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.target.set(0,0,0);

// faint axes + bounding box for spatial reference
const ax = new THREE.AxesHelper(1.15); ax.material.transparent = true; ax.material.opacity = 0.28; scene.add(ax);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(2,2,2)),
  new THREE.LineBasicMaterial({color:0xb9b2a5, transparent:true, opacity:0.25}));
scene.add(box);

// hull edges among the 8 corners
if (EDGES.length){
  const g = new THREE.BufferGeometry(); const pos=[];
  EDGES.forEach(function(e){ const a=CORNERS[e[0]], b=CORNERS[e[1]];
    pos.push(a.x,a.y,a.z, b.x,b.y,b.z); });
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos,3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial(
    {color:0x8fa89a, transparent:true, opacity:0.45})));
}

const loader = new THREE.TextureLoader();
function faceSprite(thumb, scale){
  const m = new THREE.SpriteMaterial({map: loader.load(thumb), transparent:true, depthWrite:false});
  const s = new THREE.Sprite(m); s.scale.set(scale, scale, 1); return s;
}
function ringTexture(hex){
  const c=document.createElement("canvas"); c.width=c.height=64; const g=c.getContext("2d");
  g.beginPath(); g.arc(32,32,29,0,7); g.lineWidth=6; g.strokeStyle=hex; g.stroke();
  const t=new THREE.Texture(c); t.needsUpdate=true; return t;
}
function textSprite(txt, hex){
  const c=document.createElement("canvas"); const ctx=c.getContext("2d");
  const f=44; ctx.font="700 "+f+"px system-ui,sans-serif";
  const w=Math.ceil(ctx.measureText(txt).width)+20; c.width=w; c.height=f+18;
  ctx.font="700 "+f+"px system-ui,sans-serif"; ctx.textBaseline="top";
  ctx.fillStyle=hex; ctx.fillText(txt, 10, 6);
  const t=new THREE.Texture(c); t.needsUpdate=true;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:t, transparent:true, depthWrite:false}));
  sp.scale.set(w/c.height*0.16, 0.16, 1); return sp;
}

// cloud: colored points (zoomed out) + face sprites (zoomed in)
const pts=[], cols=[];
PLAYERS.forEach(function(p){ pts.push(p.x,p.y,p.z);
  const c=new THREE.Color(COLORS[p.arch]); cols.push(c.r,c.g,c.b); });
const pg=new THREE.BufferGeometry();
pg.setAttribute("position", new THREE.Float32BufferAttribute(pts,3));
pg.setAttribute("color", new THREE.Float32BufferAttribute(cols,3));
const cloudPoints=new THREE.Points(pg, new THREE.PointsMaterial(
  {size:0.055, vertexColors:true, transparent:true, opacity:0.75, sizeAttenuation:true}));
scene.add(cloudPoints);

const cloudFaces=new THREE.Group(); cloudFaces.visible=false; scene.add(cloudFaces);
PLAYERS.forEach(function(p){ const s=faceSprite(p.thumb, 0.10);
  s.position.set(p.x,p.y,p.z); s.userData=p; cloudFaces.add(s); });

// corners: ring + face + name, always visible
const cornerFaces=[];
CORNERS.forEach(function(c){
  const ring=new THREE.Sprite(new THREE.SpriteMaterial({map:ringTexture(c.color), transparent:true, depthWrite:false}));
  ring.position.set(c.x,c.y,c.z); ring.scale.set(0.235,0.235,1); scene.add(ring);
  const f=faceSprite(c.thumb, 0.20); f.position.set(c.x,c.y,c.z); f.userData=c; scene.add(f); cornerFaces.push(f);
  const lab=textSprite(c.name, c.color); lab.position.set(c.x, c.y+0.17, c.z); scene.add(lab);
});

const ZOOM_FACES = 2.7;  // camera-to-target distance below which cloud dots become faces
function updateLOD(){
  const d = camera.position.distanceTo(controls.target);
  const faces = d < ZOOM_FACES;
  cloudFaces.visible = faces; cloudPoints.visible = !faces;
}
controls.addEventListener("change", updateLOD);

// hover tooltip (raycast against whatever faces are currently shown + corners)
const ray=new THREE.Raycaster(); const mouse=new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", function(ev){
  const r=renderer.domElement.getBoundingClientRect();
  mouse.x=((ev.clientX-r.left)/r.width)*2-1; mouse.y=-((ev.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(mouse, camera);
  const targets = cornerFaces.concat(cloudFaces.visible ? cloudFaces.children : []);
  const hit = ray.intersectObjects(targets, false)[0];
  if (hit && hit.object.userData){
    const p=hit.object.userData;
    const sub = p.label ? p.label : (COLORS[p.arch] ? p.arch_label : "");
    tip.innerHTML='<img src="'+p.thumb+'"><div><div class="nm">'+p.name+'</div>'+
      '<div class="sub">'+(p.label || (p.arch_label+" "+p.pct+"%"))+'</div></div>';
    tip.style.display="flex";
    let left=ev.clientX+14, top=ev.clientY+14;
    if(left+180>window.innerWidth) left=ev.clientX-194;
    tip.style.left=left+"px"; tip.style.top=top+"px";
  } else { tip.style.display="none"; }
});
renderer.domElement.addEventListener("pointerleave", function(){ tip.style.display="none"; });

document.getElementById("reset").addEventListener("click", function(){
  controls.reset(); camera.position.copy(CAM0); controls.target.set(0,0,0); updateLOD();
});

window.addEventListener("resize", function(){
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  camera.aspect=wrap.clientWidth/wrap.clientHeight; camera.updateProjectionMatrix();
});

updateLOD();
(function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();
</script>
</body></html>
"""


def build_hull_3d_html(players, corners, colors, edges, height, bg, ink, muted, line, card_bg):
    """players: [{x,y,z,name,arch,arch_label,pct,thumb}]; corners: [{x,y,z,name,
    label,color,thumb}]; colors: 8 hex; edges: [[i,j],...]. Coords must already
    be normalized to roughly [-1,1]. Returns a self-contained HTML page."""
    return (_TEMPLATE
            .replace("__PLAYERS__", json.dumps(players))
            .replace("__CORNERS__", json.dumps(corners))
            .replace("__COLORS__", json.dumps(colors))
            .replace("__EDGES__", json.dumps(edges))
            .replace("__HEIGHT__", str(int(height)))
            .replace("__BG__", bg)
            .replace("__CARD_BG__", card_bg)
            .replace("__INK__", ink)
            .replace("__MUTED__", muted)
            .replace("__LINE__", line))
