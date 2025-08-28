// static/viewer.js
import * as THREE from '/static/three.module.js';
import { PLYLoader } from '/static/PLYLoader.js';
import { OrbitControls } from '/static/OrbitControls.js';

const container = document.getElementById('viewer');
const statusEl = document.getElementById('status');              // optional
const garmentsStatus = document.getElementById('garmentsStatus'); // optional
const loadBtn = document.getElementById('loadGarmentsBtn');

let scene, camera, renderer, controls;
const loader = new PLYLoader();
const meshes = { body: null, garments: [] };

init();
loadBody('/body');                 // keep your fixed body endpoint
if (loadBtn) loadBtn.addEventListener('click', () => loadAllGarments());

window.refreshGarments = async () => { await loadAllGarments(); };

function init() {
  scene = new THREE.Scene();

  const aspect = container.clientWidth / container.clientHeight;
  camera = new THREE.PerspectiveCamera(60, aspect, 0.001, 1000);
  camera.position.set(0.8, 0.6, 1.2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.screenSpacePanning = true; // shift+drag pans in screen space

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  const grid = new THREE.GridHelper(2.0, 20);
  grid.position.y = 0;
  scene.add(grid);

  window.addEventListener('resize', onResize);
  animate();
}

function onResize() {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ---------- Body ----------
function loadBody(url) {
  loader.load(
    url,
    (geom) => {
      geom.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        metalness: 0.0, roughness: 1.0, side: THREE.FrontSide
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.name = 'Body';
      scene.add(mesh);
      meshes.body = mesh;

      if (statusEl) statusEl.textContent = 'Body loaded';
      fitSceneToCamera();
    },
    undefined,
    (err) => {
      console.error('Failed to load body mesh', err);
      if (statusEl) statusEl.textContent = 'Failed to load body (see console).';
    }
  );
}

// ---------- Garments ----------
async function loadAllGarments() {
  setGarmentsStatus('Scanning garment patches…');
  try {
    const res = await fetch('/garments');
    if (!res.ok) {
      setGarmentsStatus('Failed to list garments');
      console.error(await res.text());
      return;
    }
    const data = await res.json();
    clearGarments();

    const upper = data.upper || [];
    const lower = data.lower || [];
    const total = upper.length + lower.length;

    if (!total) {
      setGarmentsStatus('No patches found.');
      return;
    }

    await Promise.all([
      loadGroup('upper', upper),
      loadGroup('lower', lower),
    ]);

    setGarmentsStatus(`Loaded ${meshes.garments.length} patches.`);
    fitSceneToCamera(); // frame body + garments together
  } catch (e) {
    setGarmentsStatus('Error (see console)');
    console.error(e);
  }
}

function setGarmentsStatus(text) {
  if (garmentsStatus) garmentsStatus.textContent = text;
  else if (statusEl) statusEl.textContent = text;
}

function clearGarments() {
  for (const m of meshes.garments) {
    scene.remove(m);
    m.geometry?.dispose();
    if (m.material?.isMaterial) m.material.dispose();
  }
  meshes.garments.length = 0;
}

function loadGroup(groupName, paths) {
  const total = paths.length || 1;
  return Promise.all(paths.map((p, i) => loadPatch(groupName, p, i, total)));
}

function loadPatch(group, path, idx, totalInGroup) {
  return new Promise((resolve) => {
    loader.load(
      `/file?path=${encodeURIComponent(path)}`,
      (geom) => {
        geom.computeVertexNormals();

        // inside loadPatch(...) just before creating the mesh:
const mat = new THREE.MeshStandardMaterial({
  flatShading: false,
  metalness: 0.0,
  roughness: 1.0,
  transparent: true,
  opacity: 0.9,
  polygonOffset: true,
  // pull toward camera slightly to avoid z-fight with body
  polygonOffsetFactor: -2,   // try -1 to -4
  polygonOffsetUnits:  -2,
  side: THREE.DoubleSide,
});



        mat.color = colorForPatch(idx, totalInGroup, group);

        const mesh = new THREE.Mesh(geom, mat);
        // draw patches after body so their (slightly nearer) fragments win
mesh.renderOrder = 2;   // body can stay at default 0/1
        mesh.name = `${group}:${path}`;
        mesh.userData = { type: 'garment', group, path, idx };
        scene.add(mesh);
        meshes.garments.push(mesh);

        resolve(mesh);
      },
      undefined,
      (err) => {
        console.error(`Failed to load ${group} patch: ${path}`, err);
        resolve(null);
      }
    );
  });
}

// ---------- Framing / Camera ----------
function fitSceneToCamera(fitOffset = 1.2) {
  // Build a bbox over all visible meshes
  const bbox = new THREE.Box3();
  let hasAny = false;
  scene.traverse((obj) => {
    if (obj.isMesh && obj.visible) {
      obj.geometry.computeBoundingBox?.();
      const box = new THREE.Box3().setFromObject(obj);
      if (!box.isEmpty()) {
        if (!hasAny) { bbox.copy(box); hasAny = true; }
        else { bbox.union(box); }
      }
    }
  });
  if (!hasAny) return;

  const size = new THREE.Vector3();
  bbox.getSize(size);
  const center = new THREE.Vector3();
  bbox.getCenter(center);

  // Position camera at a distance that fits the box
  const maxSize = Math.max(size.x, size.y, size.z);
  const fitHeightDistance = maxSize / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)));
  const fitWidthDistance  = fitHeightDistance / camera.aspect;
  const distance = fitOffset * Math.max(fitHeightDistance, fitWidthDistance);

  // Move camera along its current view direction to the new distance
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  camera.position.copy(dir.multiplyScalar(distance).add(center));

  camera.near = distance / 100;
  camera.far  = distance * 100;
  camera.updateProjectionMatrix();

  controls.maxDistance = distance * 10;
  controls.target.copy(center);
  controls.update();
}

// ---------- Coloring ----------
function colorForPatch(i, total, group) {
  // Separate groups in hue space
  const base = group === 'upper' ? 0.02 : 0.58;
  const hue = (base + (i / Math.max(1, total))) % 1.0;
  const c = new THREE.Color();
  c.setHSL(hue, 0.65, 0.55);
  return c;
}
