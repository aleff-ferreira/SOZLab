"""Integrated NGL-based 3D density/structure viewer."""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    if str(os.environ.get("SOZLAB_DISABLE_WEBENGINE", "")).strip().lower() in {"1", "true", "yes"}:
        raise ImportError("WebEngine disabled by SOZLAB_DISABLE_WEBENGINE")
    from PyQt6 import QtWebEngineCore, QtWebEngineWidgets
except Exception:  # pragma: no cover - optional runtime dependency
    QtWebEngineCore = None
    QtWebEngineWidgets = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityProfile:
    ngl_quality: str


_QUALITY_PROFILES: dict[str, QualityProfile] = {
    "Draft": QualityProfile(ngl_quality="low"),
    "Balanced": QualityProfile(ngl_quality="medium"),
    "High": QualityProfile(ngl_quality="high"),
    "Ultra": QualityProfile(ngl_quality="high"),
}


def _resolve_ngl_js_path() -> Path | None:
    env_path = os.environ.get("SOZLAB_NGL_JS", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) > 2 else Path.cwd()
    candidates.extend(
        [
            repo_root / "ngl" / "dist" / "ngl.js",
            repo_root / "ngl" / "dist" / "ngl.umd.js",
            repo_root / "src" / "app" / "assets" / "ngl.js",
            Path.cwd() / "ngl" / "dist" / "ngl.js",
        ]
    )

    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except Exception:
            continue
    return None


if QtWebEngineWidgets is not None and QtWebEngineCore is not None:

    class _NGLPage(QtWebEngineCore.QWebEnginePage):
        console_event = QtCore.pyqtSignal(object)

        def javaScriptConsoleMessage(self, level, message, line_number, source_id):  # noqa: N802
            prefix = "SOZLAB_EVENT:"
            if message.startswith(prefix):
                payload = message[len(prefix) :].strip()
                try:
                    self.console_event.emit(json.loads(payload))
                    return
                except Exception:
                    pass
            lowered = message.lower()
            if "uselegacylights has been deprecated" in lowered:
                # Benign three.js warning from upstream NGL integration.
                return
            if (
                "webgl context could not be created" in lowered
                or "could not initialize renderer" in lowered
                or "error creating webgl context" in lowered
            ):
                self.console_event.emit(
                    {
                        "type": "error",
                        "message": "NGL could not create a WebGL context on this machine/session.",
                    }
                )
                return
            super().javaScriptConsoleMessage(level, message, line_number, source_id)

else:

    class _NGLPage(QtCore.QObject):
        console_event = QtCore.pyqtSignal(object)


def _build_ngl_html() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body, #viewport {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #05080d;
    }
    #badge {
      position: absolute;
      top: 10px;
      left: 10px;
      max-width: calc(100% - 24px);
      padding: 5px 9px;
      border-radius: 8px;
      background: rgba(8, 16, 24, 0.72);
      color: #d7e9f6;
      font-family: sans-serif;
      font-size: 11px;
      border: 1px solid rgba(120, 144, 166, 0.30);
      pointer-events: none;
      z-index: 21;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
  </style>
  <script src="ngl.js"></script>
</head>
<body>
  <div id="viewport"></div>
  <div id="badge">NGL Viewer</div>
  <script>
    (function () {
      var stage = null;
      var structureComponent = null;
      var densityComponent = null;
      var structureReps = [];
      var densityReps = [];
      var densityBox = null;
      var highlightRep = null;
      var labelComponents = [];
      var labelComponentsByKey = Object.create(null);
      var pendingStructureLayers = [];
      var pendingDensity = {
        visible: true,
        isolevel: 0.5,
        color: "#55aaff",
        opacity: 0.35,
        style: "Translucent",
        quality: "medium",
        dualIso: false,
        isolevel2: 0.8,
        color2: "#f97316",
        opacity2: 0.25
      };
      var desiredQuality = "medium";
      var lodEnabled = true;
      var lodActive = false;
      var lodRestoreTimer = null;
      var measurementMode = "none";
      var measurementQueue = [];
      var autoLabel = false;
      var cameraType = "perspective";

      function emit(type, payload) {
        var data = payload || {};
        data.type = type;
        console.log("SOZLAB_EVENT:" + JSON.stringify(data));
      }

      function clamp(v, lo, hi) {
        var n = Number(v);
        if (!isFinite(n)) return lo;
        return Math.max(lo, Math.min(hi, n));
      }

      function finiteOr(value, fallback) {
        var n = Number(value);
        return isFinite(n) ? n : fallback;
      }

      function isFiniteVec3(vec) {
        return !!vec && isFinite(Number(vec.x)) && isFinite(Number(vec.y)) && isFinite(Number(vec.z));
      }

      function applyPickTransforms(vec, pickingProxy) {
        if (!isFiniteVec3(vec)) {
          return null;
        }
        var out = new NGL.Vector3(finiteOr(vec.x, NaN), finiteOr(vec.y, NaN), finiteOr(vec.z, NaN));
        if (!isFiniteVec3(out)) {
          return null;
        }
        try {
          if (pickingProxy && pickingProxy.component && pickingProxy.component.matrix) {
            out.applyMatrix4(pickingProxy.component.matrix);
          }
        } catch (_e0) {}
        try {
          if (pickingProxy && pickingProxy.instance && pickingProxy.instance.matrix) {
            out.applyMatrix4(pickingProxy.instance.matrix);
          }
        } catch (_e1) {}
        return isFiniteVec3(out) ? out : null;
      }

      function vectorFromFlatArray(values, index) {
        if (!values) {
          return null;
        }
        var idx = Number(index);
        if (!isFinite(idx)) {
          return null;
        }
        idx = Math.max(0, Math.floor(idx));
        var base = idx * 3;
        if (base + 2 >= values.length) {
          return null;
        }
        var x = finiteOr(values[base], NaN);
        var y = finiteOr(values[base + 1], NaN);
        var z = finiteOr(values[base + 2], NaN);
        if (!isFinite(x) || !isFinite(y) || !isFinite(z)) {
          return null;
        }
        return new NGL.Vector3(x, y, z);
      }

      function pickDensityPosition(pickingProxy) {
        try {
          var surfacePick = pickingProxy ? pickingProxy.surface : null;
          if (surfacePick && surfacePick.surface && surfacePick.surface.position) {
            var surfVec = vectorFromFlatArray(surfacePick.surface.position, surfacePick.index);
            var surfPos = applyPickTransforms(surfVec, pickingProxy);
            if (surfPos) {
              return surfPos;
            }
          }
        } catch (_e0) {}
        try {
          var volumePick = pickingProxy ? pickingProxy.volume : null;
          if (volumePick && volumePick.volume && volumePick.volume.position) {
            var volVec = vectorFromFlatArray(volumePick.volume.position, volumePick.index);
            var volPos = applyPickTransforms(volVec, pickingProxy);
            if (volPos) {
              return volPos;
            }
          }
        } catch (_e1) {}
        try {
          if (pickingProxy && pickingProxy.position) {
            var pos = pickingProxy.position;
            var fallback = new NGL.Vector3(
              finiteOr(pos.x, NaN),
              finiteOr(pos.y, NaN),
              finiteOr(pos.z, NaN)
            );
            if (isFiniteVec3(fallback)) {
              return fallback;
            }
          }
        } catch (_e2) {}
        return null;
      }

      function labelPositionKey(x, y, z) {
        var precision = 1000.0;
        var px = Math.round(finiteOr(x, 0) * precision) / precision;
        var py = Math.round(finiteOr(y, 0) * precision) / precision;
        var pz = Math.round(finiteOr(z, 0) * precision) / precision;
        return String(px) + "|" + String(py) + "|" + String(pz);
      }

      function removeLabelAtKey(key) {
        if (!key || !stage) {
          return;
        }
        var comp = labelComponentsByKey[key];
        if (!comp) {
          return;
        }
        try {
          stage.removeComponent(comp);
        } catch (_e0) {}
        delete labelComponentsByKey[key];
        for (var i = labelComponents.length - 1; i >= 0; i -= 1) {
          if (labelComponents[i] === comp) {
            labelComponents.splice(i, 1);
          }
        }
      }

      function serialiseBox(box) {
        if (!box || !box.min || !box.max) {
          return null;
        }
        return {
          min: {
            x: finiteOr(box.min.x, 0),
            y: finiteOr(box.min.y, 0),
            z: finiteOr(box.min.z, 0)
          },
          max: {
            x: finiteOr(box.max.x, 0),
            y: finiteOr(box.max.y, 0),
            z: finiteOr(box.max.z, 0)
          }
        };
      }

      function serialiseMatrix4(matrixLike) {
        if (!matrixLike) {
          return null;
        }
        try {
          var elems = matrixLike.elements || matrixLike;
          if (!elems || elems.length !== 16) {
            return null;
          }
          var out = [];
          for (var i = 0; i < 16; i += 1) {
            out.push(finiteOr(elems[i], 0));
          }
          return out;
        } catch (_e0) {
          return null;
        }
      }

      function boxCenterAndRadius(box) {
        if (!box || !box.min || !box.max) {
          return null;
        }
        var cx = (finiteOr(box.min.x, 0) + finiteOr(box.max.x, 0)) * 0.5;
        var cy = (finiteOr(box.min.y, 0) + finiteOr(box.max.y, 0)) * 0.5;
        var cz = (finiteOr(box.min.z, 0) + finiteOr(box.max.z, 0)) * 0.5;
        var dx = finiteOr(box.max.x, 0) - finiteOr(box.min.x, 0);
        var dy = finiteOr(box.max.y, 0) - finiteOr(box.min.y, 0);
        var dz = finiteOr(box.max.z, 0) - finiteOr(box.min.z, 0);
        var radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;
        return {
          center: { x: cx, y: cy, z: cz },
          radius: finiteOr(radius, 0)
        };
      }

      function getComponentTransform(component) {
        if (!component) {
          return null;
        }
        try {
          return serialiseMatrix4(component.matrix);
        } catch (_e0) {
          return null;
        }
      }

      function tryGetStructureBox() {
        if (!structureComponent) {
          return null;
        }
        var box = null;
        try {
          if (structureComponent.getBoxUntransformed) {
            box = structureComponent.getBoxUntransformed();
          }
        } catch (_e0) {}
        if (!box) {
          try {
            if (structureComponent.getBox) {
              box = structureComponent.getBox();
            }
          } catch (_e1) {}
        }
        if (!box) {
          try {
            if (structureComponent.boundingBox) {
              box = structureComponent.boundingBox;
            }
          } catch (_e2) {}
        }
        return box || null;
      }

      function tryGetDensityBox() {
        if (!densityComponent) {
          return null;
        }
        var box = null;
        try {
          if (densityComponent.getBoxUntransformed) {
            box = densityComponent.getBoxUntransformed();
          }
        } catch (_e0) {}
        if (!box) {
          try {
            if (densityComponent.getBox) {
              box = densityComponent.getBox();
            }
          } catch (_e1) {}
        }
        if (!box) {
          try {
            if (densityComponent.boundingBox) {
              box = densityComponent.boundingBox;
            }
          } catch (_e2) {}
        }
        return box || null;
      }

      function emitSceneDiagnostics(reason) {
        var structureBox = tryGetStructureBox();
        var densityBoxNow = tryGetDensityBox();
        var structureMeta = boxCenterAndRadius(structureBox);
        var densityMeta = boxCenterAndRadius(densityBoxNow);
        emit("scene_diag", {
          ok: true,
          reason: String(reason || ""),
          hasStructure: !!structureComponent,
          hasDensity: !!densityComponent,
          structureBBox: serialiseBox(structureBox),
          structureCenter: structureMeta ? structureMeta.center : null,
          structureRadius: structureMeta ? structureMeta.radius : null,
          structureTransform: getComponentTransform(structureComponent),
          densityBBox: serialiseBox(densityBoxNow),
          densityCenter: densityMeta ? densityMeta.center : null,
          densityRadius: densityMeta ? densityMeta.radius : null,
          densityTransform: getComponentTransform(densityComponent)
        });
      }

      function normalizeIsoLevel(rawLevel, fallback, dataMin, dataMax) {
        var level = finiteOr(rawLevel, fallback);
        if (!isFinite(level)) {
          level = fallback;
        }
        if (!(dataMax > dataMin)) {
          return level;
        }
        if (level <= dataMin || level >= dataMax) {
          level = fallback;
        }
        var span = Math.max(dataMax - dataMin, 1e-6);
        return clamp(level, dataMin + span * 1e-4, dataMax - span * 1e-4);
      }

      function v3(x, y, z) {
        return { x: Number(x) || 0, y: Number(y) || 0, z: Number(z) || 0 };
      }

      function vecSub(a, b) {
        return v3(a.x - b.x, a.y - b.y, a.z - b.z);
      }

      function vecDot(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
      }

      function vecCross(a, b) {
        return v3(
          a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x
        );
      }

      function vecLen(a) {
        return Math.sqrt(Math.max(0.0, vecDot(a, a)));
      }

      function vecNorm(a) {
        var n = vecLen(a);
        if (n <= 1e-12) return v3(0, 0, 0);
        return v3(a.x / n, a.y / n, a.z / n);
      }

      function repTypeForStyle(style) {
        var key = String(style || "").toLowerCase();
        if (key === "cartoon") return "cartoon";
        if (key === "surface") return "surface";
        if (key === "ball+stick" || key === "ball and stick") return "ball+stick";
        if (key === "licorice") return "licorice";
        if (key === "ribbon") return "ribbon";
        if (key === "spacefill") return "spacefill";
        if (key === "hyperballs") return "hyperball";
        if (key === "trace") return "backbone";
        if (key === "tube") return "tube";
        return null;
      }

      function colorModeToNgl(mode) {
        var key = String(mode || "").toLowerCase();
        if (key === "element") return { colorScheme: "element" };
        if (key === "chainid" || key === "chain") return { colorScheme: "chainid" };
        if (key === "residueindex" || key === "residue") return { colorScheme: "residueindex" };
        if (key === "bfactor") return { colorScheme: "bfactor" };
        if (key === "sstruc" || key === "secondary") return { colorScheme: "sstruc" };
        if (key === "custom") return { color: null };
        return { colorScheme: "element" };
      }

      function removeStructureRepresentations() {
        if (!structureComponent) return;
        for (var i = 0; i < structureReps.length; i += 1) {
          try {
            structureComponent.removeRepresentation(structureReps[i].rep);
          } catch (_e) {}
        }
        structureReps = [];
        if (highlightRep) {
          try {
            structureComponent.removeRepresentation(highlightRep);
          } catch (_e) {}
          highlightRep = null;
        }
        for (var j = 0; j < labelComponents.length; j += 1) {
          try {
            if (stage) {
              stage.removeComponent(labelComponents[j]);
            }
          } catch (_e2) {}
        }
        labelComponents = [];
        labelComponentsByKey = Object.create(null);
      }

      function removeDensityRepresentations() {
        if (!densityComponent) return;
        for (var i = 0; i < densityReps.length; i += 1) {
          try {
            densityComponent.removeRepresentation(densityReps[i]);
          } catch (_e) {}
        }
        densityReps = [];
      }

      function removeStructure() {
        removeStructureRepresentations();
        if (structureComponent && stage) {
          try { stage.removeComponent(structureComponent); } catch (_e) {}
          structureComponent = null;
        }
      }

      function removeDensity() {
        removeDensityRepresentations();
        if (densityComponent && stage) {
          try { stage.removeComponent(densityComponent); } catch (_e) {}
          densityComponent = null;
        }
        densityBox = null;
      }

      function setRepQuality(quality) {
        for (var i = 0; i < structureReps.length; i += 1) {
          try {
            structureReps[i].rep.setParameters({ quality: quality });
          } catch (_e) {}
        }
        for (var j = 0; j < densityReps.length; j += 1) {
          try {
            densityReps[j].setParameters({ quality: quality });
          } catch (_e2) {}
        }
      }

      function setLodActive(active) {
        if (!lodEnabled) {
          return;
        }
        if (lodActive === active) {
          return;
        }
        lodActive = active;
        setRepQuality(lodActive ? "low" : desiredQuality);
      }

      function installLodHandlers() {
        if (!stage || !stage.viewer || !stage.viewer.container) {
          return;
        }
        var root = stage.viewer.container;
        if (root.__sozlabLodInstalled) {
          return;
        }
        root.__sozlabLodInstalled = true;

        function startLod() {
          setLodActive(true);
          if (lodRestoreTimer) {
            clearTimeout(lodRestoreTimer);
            lodRestoreTimer = null;
          }
        }

        function stopLodSoon() {
          if (lodRestoreTimer) {
            clearTimeout(lodRestoreTimer);
          }
          lodRestoreTimer = setTimeout(function () {
            setLodActive(false);
          }, 140);
        }

        root.addEventListener("pointerdown", startLod, { passive: true });
        root.addEventListener("pointerup", stopLodSoon, { passive: true });
        root.addEventListener("pointercancel", stopLodSoon, { passive: true });
        root.addEventListener("wheel", function () {
          startLod();
          stopLodSoon();
        }, { passive: true });
      }

      function ensureStage() {
        if (stage) {
          return stage;
        }
        if (typeof NGL === "undefined") {
          emit("error", { message: "NGL is not available in web runtime." });
          return null;
        }

        try {
          stage = new NGL.Stage("viewport", {
            backgroundColor: "#05080d",
            sampleLevel: 1,
            tooltip: false,
            cameraType: cameraType
          });
        } catch (err) {
          stage = null;
          emit("error", { message: "WebGL initialization failed: " + err });
          return null;
        }

        if (!stage || !stage.viewer) {
          stage = null;
          emit("error", { message: "WebGL initialization failed: viewer backend unavailable." });
          return null;
        }

        window.addEventListener("resize", function () {
          if (stage) {
            stage.handleResize();
          }
        }, { passive: true });

        stage.signals.clicked.add(function (pickingProxy) {
          if (!pickingProxy) {
            return;
          }
          try {
            if (pickingProxy.atom) {
              var atom = pickingProxy.atom;
              var pos = pickingProxy.position || atom.positionToVector3();
              var payload = {
                kind: "atom",
                label: "",
                atomIndex: Number(atom.index || 0),
                atomName: String(atom.atomname || ""),
                resname: String(atom.resname || ""),
                resno: Number(atom.resno || 0),
                chain: String(atom.chainname || ""),
                bfactor: Number(atom.bfactor || 0),
                x: finiteOr(pos ? pos.x : NaN, 0),
                y: finiteOr(pos ? pos.y : NaN, 0),
                z: finiteOr(pos ? pos.z : NaN, 0)
              };
              payload.label = payload.atomName + " " + payload.resname + payload.resno;
              emit("pick", payload);
              handleMeasurement(payload);
              if (autoLabel) {
                addCustomLabel({
                  x: payload.x,
                  y: payload.y,
                  z: payload.z,
                  text: payload.resname + payload.resno,
                  color: "#f8fafc"
                });
              }
              return;
            }
            var densityPos = pickDensityPosition(pickingProxy);
            if (densityPos) {
              emit("pick", {
                kind: "density",
                label: "",
                x: finiteOr(densityPos.x, 0),
                y: finiteOr(densityPos.y, 0),
                z: finiteOr(densityPos.z, 0)
              });
              return;
            }
            emit("error", { message: "Density pick did not provide a valid position." });
          } catch (err) {
            emit("error", { message: "Pick handler failed: " + err });
          }
        });

        installLodHandlers();

        var last = performance.now();
        var acc = 0.0;
        var frames = 0;

        function tick(now) {
          var dt = now - last;
          last = now;
          if (dt > 0.01) {
            acc += dt;
            frames += 1;
          }
          if (acc >= 700.0 && frames > 0) {
            emit("perf", { fps: (frames * 1000.0) / acc, frameMs: acc / frames });
            acc = 0.0;
            frames = 0;
          }
          window.requestAnimationFrame(tick);
        }

        window.requestAnimationFrame(tick);
        emit("ready", { ok: true });
        return stage;
      }

      function applyStructureLayers(layers, opts) {
        if (Array.isArray(layers)) {
          pendingStructureLayers = layers.slice();
        }
        opts = opts || {};
        desiredQuality = String(opts.quality || desiredQuality || "medium");

        var st = ensureStage();
        if (!st || !structureComponent) {
          return;
        }

        removeStructureRepresentations();

        var radiusScale = clamp(opts.radiusScale || 0.55, 0.08, 4.0);
        var layersToApply = pendingStructureLayers.slice();
        var layerDiagnostics = [];
        var totalSelectedAtoms = 0;
        var structureAtomCount = 0;
        try {
          if (structureComponent && structureComponent.structure) {
            structureAtomCount = Number(structureComponent.structure.atomCount || 0) || 0;
          }
        } catch (_eCount) {}
        if (!layersToApply.length) {
          layersToApply.push({
            id: "default",
            type: "Ball+Stick",
            selection: "all",
            colorScheme: "element",
            opacity: 1.0,
            visible: true
          });
        }

        for (var i = 0; i < layersToApply.length; i += 1) {
          var layer = layersToApply[i] || {};
          if (layer.visible === false) {
            continue;
          }
          var repType = repTypeForStyle(layer.type);
          if (!repType) {
            continue;
          }

          var params = {
            sele: String(layer.selection || "all"),
            opacity: clamp(layer.opacity != null ? layer.opacity : 1.0, 0.03, 1.0),
            quality: lodActive ? "low" : desiredQuality,
            visible: true
          };
          var selectedAtoms = -1;
          try {
            if (structureComponent && structureComponent.structure && structureComponent.structure.getView && typeof NGL.Selection === "function") {
              var selectionObj = new NGL.Selection(String(params.sele || "all"));
              var view = structureComponent.structure.getView(selectionObj);
              if (view && view.atomCount != null) {
                selectedAtoms = Number(view.atomCount) || 0;
                totalSelectedAtoms += Math.max(0, selectedAtoms);
              }
            }
          } catch (_eSel) {}

          var colorData = colorModeToNgl(layer.colorScheme);
          if (layer.colorScheme === "custom") {
            params.color = String(layer.colorValue || "#55aaff");
          } else if (colorData.colorScheme) {
            params.colorScheme = colorData.colorScheme;
          }

          if (repType === "spacefill") {
            params.radiusScale = radiusScale;
          } else if (repType === "ball+stick") {
            params.radiusScale = radiusScale * 0.70;
            params.multipleBond = true;
            params.aspectRatio = 1.7;
          } else if (repType === "licorice") {
            params.radiusScale = radiusScale * 0.58;
            params.multipleBond = true;
          } else if (repType === "cartoon") {
            params.radiusScale = Math.max(0.15, radiusScale * 0.34);
            params.aspectRatio = 4.0;
            params.smoothSheet = true;
          } else if (repType === "surface") {
            params.surfaceType = "av";
            params.probeRadius = 1.4;
            params.useWorker = true;
          } else if (repType === "hyperball") {
            params.shrink = 0.15;
            params.scale = Math.max(0.3, radiusScale);
          }

          try {
            var rep = structureComponent.addRepresentation(repType, params);
            structureReps.push({ id: String(layer.id || ("layer-" + i)), rep: rep, cfg: layer });
            layerDiagnostics.push({
              id: String(layer.id || ("layer-" + i)),
              type: String(repType),
              selection: String(params.sele || "all"),
              selectedAtoms: selectedAtoms,
              visible: true,
              ok: true
            });
          } catch (err) {
            layerDiagnostics.push({
              id: String(layer.id || ("layer-" + i)),
              type: String(repType),
              selection: String(params.sele || "all"),
              selectedAtoms: selectedAtoms,
              visible: true,
              ok: false,
              error: String(err)
            });
            emit("error", { message: "Failed to apply structure layer: " + err });
          }
        }

        if (!structureReps.length) {
          try {
            var fallback = structureComponent.addRepresentation("ball+stick", {
              sele: "all",
              opacity: 1.0,
              quality: lodActive ? "low" : desiredQuality,
              radiusScale: Math.max(0.22, radiusScale * 0.70),
              multipleBond: true,
              aspectRatio: 1.7,
              visible: true
            });
            structureReps.push({ id: "fallback-all", rep: fallback, cfg: { type: "Ball+Stick", selection: "all" } });
            layerDiagnostics.push({
              id: "fallback-all",
              type: "ball+stick",
              selection: "all",
              selectedAtoms: structureAtomCount,
              visible: true,
              ok: true
            });
          } catch (errFallback) {
            emit("error", { message: "Failed to apply fallback structure layer: " + errFallback });
          }
        }

        if (typeof st.requestRender === "function") {
          st.requestRender();
        }
        emit("rep", {
          ok: true,
          count: structureReps.length,
          structureAtoms: structureAtomCount,
          selectedAtomsTotal: totalSelectedAtoms,
          layers: layerDiagnostics
        });
        emitSceneDiagnostics("apply-structure-layers");
      }

      function updateHighlightSelection(sele) {
        if (!structureComponent) {
          return;
        }
        if (highlightRep) {
          try {
            structureComponent.removeRepresentation(highlightRep);
          } catch (_e) {}
          highlightRep = null;
        }
        if (!sele) {
          return;
        }
        try {
          highlightRep = structureComponent.addRepresentation("ball+stick", {
            sele: String(sele),
            color: "#22d3ee",
            radiusScale: 0.38,
            aspectRatio: 1.6,
            opacity: 1.0,
            quality: "high"
          });
        } catch (_e2) {}
      }

      async function loadStructure(url) {
        var st = ensureStage();
        if (!st) return;
        try {
          removeStructure();
          structureComponent = await st.loadFile(url, { defaultRepresentation: false });
          var atomCount = 0;
          var allSelectionAtoms = 0;
          try {
            atomCount = structureComponent.structure.atomCount || 0;
          } catch (_e) {}
          try {
            if (structureComponent && structureComponent.structure && structureComponent.structure.getView && typeof NGL.Selection === "function") {
              var allView = structureComponent.structure.getView(new NGL.Selection("all"));
              allSelectionAtoms = Number(allView ? allView.atomCount : atomCount) || 0;
            }
          } catch (_eAll) {
            allSelectionAtoms = atomCount;
          }
          var sBox = tryGetStructureBox();
          var sMeta = boxCenterAndRadius(sBox);
          applyStructureLayers(pendingStructureLayers, { quality: desiredQuality, radiusScale: 0.55 });
          autoViewCombined(180, { expandClip: true });
          emit("structure", {
            ok: true,
            atoms: atomCount,
            allSelectionAtoms: allSelectionAtoms,
            bbox: serialiseBox(sBox),
            center: sMeta ? sMeta.center : null,
            radius: sMeta ? sMeta.radius : null,
            transform: getComponentTransform(structureComponent)
          });
          emitSceneDiagnostics("load-structure");
        } catch (err) {
          emit("error", { message: "Failed to load structure: " + err });
        }
      }

      function applyDensityReps() {
        if (!densityComponent || !stage) {
          return;
        }

        removeDensityRepresentations();

        if (!pendingDensity.visible) {
          if (typeof stage.requestRender === "function") stage.requestRender();
          return;
        }

        var style = String(pendingDensity.style || "Translucent");
        var wireframe = style === "Wireframe";
        var primaryOpacity = clamp(Number(pendingDensity.opacity || 0.35), 0.03, 1.0);
        var quality = lodActive ? "low" : String(pendingDensity.quality || desiredQuality || "medium");
        var dataMin = finiteOr(pendingDensity.dataMin, 0.0);
        var dataMax = finiteOr(pendingDensity.dataMax, 1.0);
        if (!(dataMax > dataMin)) {
          dataMax = dataMin + 1e-6;
        }
        var span = Math.max(dataMax - dataMin, 1e-6);
        var fallbackIso = normalizeIsoLevel(
          finiteOr(pendingDensity.suggestedIsolevel, dataMin + span * 0.2),
          dataMin + span * 0.2,
          dataMin,
          dataMax
        );
        var primaryIso = normalizeIsoLevel(pendingDensity.isolevel, fallbackIso, dataMin, dataMax);
        var secondaryIso = normalizeIsoLevel(
          pendingDensity.isolevel2,
          clamp(primaryIso + span * 0.15, dataMin + span * 1e-4, dataMax - span * 1e-4),
          dataMin,
          dataMax
        );

        try {
          var rep1 = densityComponent.addRepresentation("surface", {
            isolevel: primaryIso,
            isolevelType: "value",
            color: String(pendingDensity.color || "#55aaff"),
            opacity: wireframe ? 1.0 : primaryOpacity,
            wireframe: wireframe,
            quality: quality,
            side: "double",
            useWorker: true
          });
          densityReps.push(rep1);
        } catch (err0) {
          emit("error", { message: "Failed to apply primary density: " + err0 });
        }

        if (pendingDensity.dualIso) {
          try {
            var rep2 = densityComponent.addRepresentation("surface", {
              isolevel: secondaryIso,
              isolevelType: "value",
              color: String(pendingDensity.color2 || "#f97316"),
              opacity: wireframe ? 1.0 : clamp(Number(pendingDensity.opacity2 || 0.24), 0.03, 1.0),
              wireframe: wireframe,
              quality: quality,
              side: "double",
              useWorker: true
            });
            densityReps.push(rep2);
          } catch (err1) {
            emit("error", { message: "Failed to apply secondary density: " + err1 });
          }
        }

        densityBox = tryGetDensityBox();

        if (pendingDensity.autoView) {
          try {
            autoViewCombined(180, { expandClip: true });
          } catch (_eAuto) {}
          pendingDensity.autoView = false;
        }

        if (typeof stage.requestRender === "function") {
          stage.requestRender();
        }
        emit("density", {
          ok: true,
          reps: densityReps.length,
          visible: !!pendingDensity.visible,
          isolevel: primaryIso,
          isolevel2: secondaryIso,
          dataMin: dataMin,
          dataMax: dataMax,
          bbox: serialiseBox(densityBox),
          center: (boxCenterAndRadius(densityBox) || {}).center || null,
          radius: (boxCenterAndRadius(densityBox) || {}).radius || null,
          transform: getComponentTransform(densityComponent)
        });
        emitSceneDiagnostics("apply-density-reps");
      }

      async function loadDensity(url, opts) {
        var st = ensureStage();
        if (!st) return;
        var source = String(url || "");
        if (!source) {
          emit("error", { message: "Failed to load density: empty URL." });
          return;
        }
        try {
          pendingDensity = Object.assign({}, pendingDensity, opts || {});
          removeDensity();
          densityComponent = await st.loadFile(source, { defaultRepresentation: false });
          var hasVolume = false;
          var valueCount = 0;
          try {
            var vol = densityComponent ? (densityComponent.volume || densityComponent.obj || densityComponent.data) : null;
            hasVolume = !!vol;
            if (vol && vol.data && vol.data.length != null) {
              valueCount = Number(vol.data.length) || 0;
            }
          } catch (_eVol) {}
          densityBox = tryGetDensityBox();
          var densityMeta = boxCenterAndRadius(densityBox);
          emit("density_load", {
            ok: true,
            url: source,
            hasVolume: hasVolume,
            values: valueCount,
            bbox: serialiseBox(densityBox),
            center: densityMeta ? densityMeta.center : null,
            radius: densityMeta ? densityMeta.radius : null,
            transform: getComponentTransform(densityComponent)
          });
          applyDensityReps();
          emitSceneDiagnostics("load-density");
        } catch (err) {
          emit("error", { message: "Failed to load density: " + err + " [url=" + source + "]" });
          emit("density_load", { ok: false, url: source, error: String(err) });
        }
      }

      function updateDensity(opts) {
        pendingDensity = Object.assign({}, pendingDensity, opts || {});
        if (!densityComponent || !stage) {
          return;
        }
        applyDensityReps();
      }

      function setMousePreset(preset) {
        if (!stage || !stage.mouseControls || !stage.mouseControls.preset) {
          return;
        }
        var key = String(preset || "Default").toLowerCase();
        var mapped = "default";
        if (key.indexOf("coot") >= 0) {
          mapped = "coot";
        } else if (key.indexOf("pymol") >= 0) {
          mapped = "pymol";
        }
        try {
          stage.mouseControls.preset(mapped);
        } catch (_e) {}
      }

      function setStageOptions(opts) {
        var st = ensureStage();
        if (!st || !opts) return;

        var params = {};
        if (opts.backgroundColor != null) params.backgroundColor = String(opts.backgroundColor);
        if (opts.clipNear != null) params.clipNear = clamp(opts.clipNear, 0, 100);
        if (opts.clipFar != null) params.clipFar = clamp(opts.clipFar, 0, 100);
        if (opts.clipDist != null) params.clipDist = Number(opts.clipDist) || 0;
        if (opts.fogNear != null) params.fogNear = clamp(opts.fogNear, 0, 100);
        if (opts.fogFar != null) params.fogFar = clamp(opts.fogFar, 0, 100);
        if (opts.lightIntensity != null) params.lightIntensity = clamp(opts.lightIntensity, 0, 4);
        if (opts.ambientIntensity != null) params.ambientIntensity = clamp(opts.ambientIntensity, 0, 4);
        if (opts.cameraType != null) {
          cameraType = String(opts.cameraType || "perspective");
          params.cameraType = cameraType;
        }

        try {
          st.setParameters(params);
        } catch (_e0) {}

        if (opts.fov != null && st.viewer && st.viewer.camera) {
          try {
            st.viewer.camera.fov = clamp(opts.fov, 10, 140);
            if (st.viewer.camera.updateProjectionMatrix) {
              st.viewer.camera.updateProjectionMatrix();
            }
          } catch (_e1) {}
        }

        if (opts.spinEnabled != null) {
          try {
            if (opts.spinEnabled) {
              st.setSpin(Number(opts.spinSpeed || 0.01));
            } else {
              st.setSpin(false);
            }
          } catch (_e2) {}
        }

        if (opts.rockEnabled != null) {
          try {
            if (st.setRock) {
              if (opts.rockEnabled) {
                st.setRock(Number(opts.rockSpeed || 0.02));
              } else {
                st.setRock(false);
              }
            }
          } catch (_e3) {}
        }

        if (opts.mousePreset != null) {
          setMousePreset(opts.mousePreset);
        }

        if (opts.lodEnabled != null) {
          lodEnabled = !!opts.lodEnabled;
        }

        if (typeof st.requestRender === "function") {
          st.requestRender();
        }
      }

      function autoViewCombined(duration, opts) {
        if (!stage) return;
        opts = opts || {};
        var dur = Number(duration);
        if (!isFinite(dur)) {
          dur = 180;
        }
        if (opts.expandClip !== false) {
          try {
            stage.setParameters({ clipNear: 0, clipFar: 100 });
          } catch (_eClip) {}
        }
        try {
          stage.autoView(dur);
        } catch (_eAuto) {
          try {
            stage.autoView();
          } catch (_eAuto2) {}
        }
        if (typeof stage.requestRender === "function") {
          stage.requestRender();
        }
      }

      function autoView() {
        if (!stage) return;
        try {
          stage.autoView();
        } catch (_e) {}
      }

      function focusSelection(sele) {
        if (!stage || !structureComponent || !sele) {
          return;
        }
        try {
          structureComponent.autoView(String(sele));
          updateHighlightSelection(String(sele));
        } catch (_e) {
          autoView();
        }
      }

      function focusPosition(pos) {
        if (!stage || !pos) {
          return;
        }
        try {
          var center = new NGL.Vector3(Number(pos.x || 0), Number(pos.y || 0), Number(pos.z || 0));
          if (stage.viewerControls && stage.viewerControls.move) {
            stage.viewerControls.move(center, 180);
          }
          if (stage.viewerControls && stage.viewerControls.zoom) {
            stage.viewerControls.zoom(-0.18);
          }
        } catch (_e) {
          autoView();
        }
      }

      function focusDensityVolume(opts) {
        if (!stage || !densityComponent) {
          return;
        }
        if (opts && opts.point) {
          focusPosition(opts.point);
          return;
        }
        try {
          autoViewCombined(180, { expandClip: true });
        } catch (_e0) {}
      }

      function setMeasurementMode(mode) {
        measurementMode = String(mode || "none").toLowerCase();
        measurementQueue = [];
      }

      function measurementNeeded(mode) {
        if (mode === "distance") return 2;
        if (mode === "angle") return 3;
        if (mode === "dihedral") return 4;
        return 0;
      }

      function computeDistance(a, b) {
        return vecLen(vecSub(a, b));
      }

      function computeAngle(a, b, c) {
        var ba = vecSub(a, b);
        var bc = vecSub(c, b);
        var nba = vecNorm(ba);
        var nbc = vecNorm(bc);
        var d = clamp(vecDot(nba, nbc), -1.0, 1.0);
        return Math.acos(d) * (180.0 / Math.PI);
      }

      function computeDihedral(a, b, c, d) {
        var b1 = vecSub(b, a);
        var b2 = vecSub(c, b);
        var b3 = vecSub(d, c);
        var n1 = vecNorm(vecCross(b1, b2));
        var n2 = vecNorm(vecCross(b2, b3));
        var m1 = vecCross(n1, vecNorm(b2));
        var x = vecDot(n1, n2);
        var y = vecDot(m1, n2);
        return Math.atan2(y, x) * (180.0 / Math.PI);
      }

      function handleMeasurement(atomPayload) {
        var needed = measurementNeeded(measurementMode);
        if (needed <= 0) {
          measurementQueue = [];
          return;
        }
        measurementQueue.push(atomPayload);
        if (measurementQueue.length < needed) {
          return;
        }

        var points = measurementQueue.slice(-needed);
        var value = 0.0;
        var unit = "";
        if (measurementMode === "distance") {
          value = computeDistance(points[0], points[1]);
          unit = "A";
        } else if (measurementMode === "angle") {
          value = computeAngle(points[0], points[1], points[2]);
          unit = "deg";
        } else if (measurementMode === "dihedral") {
          value = computeDihedral(points[0], points[1], points[2], points[3]);
          unit = "deg";
        }

        emit("measurement", {
          mode: measurementMode,
          value: Number(value),
          unit: unit,
          points: points
        });

        measurementQueue = [];
      }

      function clearMeasurements() {
        measurementQueue = [];
        emit("measurement_clear", {});
      }

      function clearLabels() {
        if (!stage) {
          return;
        }
        for (var i = 0; i < labelComponents.length; i += 1) {
          try {
            stage.removeComponent(labelComponents[i]);
          } catch (_e) {}
        }
        labelComponents = [];
        labelComponentsByKey = Object.create(null);
        if (highlightRep && structureComponent) {
          try {
            structureComponent.removeRepresentation(highlightRep);
          } catch (_e2) {}
          highlightRep = null;
        }
        if (typeof stage.requestRender === "function") {
          stage.requestRender();
        }
        emit("label_clear", {});
      }

      function addCustomLabel(payload) {
        if (!stage || !payload) {
          return;
        }
        var text = String(payload.text || "Label");
        var x = finiteOr(payload.x, NaN);
        var y = finiteOr(payload.y, NaN);
        var z = finiteOr(payload.z, NaN);
        if (!isFinite(x) || !isFinite(y) || !isFinite(z)) {
          emit("error", { message: "Failed to add custom label: invalid coordinates." });
          return;
        }
        var dedupKey = labelPositionKey(x, y, z);
        var colorHex = String(payload.color || "#f8fafc");

        function hexToRgb01(hex) {
          var clean = String(hex).replace("#", "");
          if (clean.length !== 6) {
            return [1.0, 1.0, 1.0];
          }
          var r = parseInt(clean.slice(0, 2), 16) / 255.0;
          var g = parseInt(clean.slice(2, 4), 16) / 255.0;
          var b = parseInt(clean.slice(4, 6), 16) / 255.0;
          return [r, g, b];
        }

        try {
          removeLabelAtKey(dedupKey);
          var shape = new NGL.Shape("label-" + Date.now());
          shape.addText([x, y, z], hexToRgb01(colorHex), 1.15, text);
          var comp = stage.addComponentFromObject(shape);
          comp.addRepresentation("buffer");
          labelComponents.push(comp);
          labelComponentsByKey[dedupKey] = comp;
          if (typeof stage.requestRender === "function") {
            stage.requestRender();
          }
        } catch (err) {
          emit("error", { message: "Failed to add custom label: " + err });
        }
      }

      function setAutoLabel(enabled) {
        autoLabel = !!enabled;
      }

      function captureImage(opts) {
        var st = ensureStage();
        if (!st) return;
        opts = opts || {};
        var factor = clamp(opts.factor || 2, 1, 8);
        var transparent = !!opts.transparent;
        var reqFormat = String(opts.format || "png");
        try {
          st.makeImage({
            factor: factor,
            antialias: true,
            trim: false,
            transparent: transparent
          }).then(function (blob) {
            var reader = new FileReader();
            reader.onloadend = function () {
              emit("image", {
                ok: true,
                format: reqFormat,
                factor: factor,
                transparent: transparent,
                dataUrl: reader.result || ""
              });
            };
            reader.readAsDataURL(blob);
          }).catch(function (err) {
            emit("error", { message: "Screenshot capture failed: " + err });
          });
        } catch (err) {
          emit("error", { message: "Screenshot capture failed: " + err });
        }
      }

      function getViewState() {
        var state = {
          cameraType: cameraType,
          layers: pendingStructureLayers,
          density: pendingDensity,
          quality: desiredQuality,
          lodEnabled: lodEnabled
        };
        try {
          if (stage && stage.viewerControls && stage.viewerControls.getOrientation) {
            state.orientation = stage.viewerControls.getOrientation();
          }
        } catch (_e) {}
        emit("state", { ok: true, state: state });
      }

      function setBadge(text) {
        var badge = document.getElementById("badge");
        if (!badge) {
          return;
        }
        badge.textContent = String(text || "NGL Viewer");
      }

      window.sozlabApi = {
        ensureStage: ensureStage,
        loadStructure: loadStructure,
        applyStructureLayers: applyStructureLayers,
        applyStructureRep: function (opts) {
          opts = opts || {};
          applyStructureLayers([
            {
              id: "compat",
              type: String(opts.style || "Cartoon"),
              selection: "all",
              colorScheme: String(opts.colorScheme || "element"),
              opacity: 1.0,
              visible: true
            }
          ], {
            quality: String(opts.quality || "medium"),
            radiusScale: Number(opts.radiusScale || 0.55)
          });
        },
        highlightSelection: updateHighlightSelection,
        loadDensity: loadDensity,
        updateDensity: updateDensity,
        setStageOptions: setStageOptions,
        setBackground: function (color) {
          setStageOptions({ backgroundColor: color });
        },
        setCameraType: function (type) {
          setStageOptions({ cameraType: type || "perspective" });
        },
        autoView: autoView,
        autoViewCombined: autoViewCombined,
        focusSelection: focusSelection,
        focusPosition: focusPosition,
        focusDensityVolume: focusDensityVolume,
        setMeasurementMode: setMeasurementMode,
        clearMeasurements: clearMeasurements,
        clearLabels: clearLabels,
        setAutoLabel: setAutoLabel,
        addCustomLabel: addCustomLabel,
        captureImage: captureImage,
        getViewState: getViewState,
        setBadge: setBadge,
        clearAll: function () {
          removeDensity();
          removeStructure();
          clearMeasurements();
          clearLabels();
          if (stage && typeof stage.requestRender === "function") {
            stage.requestRender();
          }
        }
      };

      ensureStage();
    })();
  </script>
</body>
</html>
"""


class Density3DWidget(QtWidgets.QWidget):
    """NGL-backed 3D viewer for density and structure overlays."""

    sig_update_structure = QtCore.pyqtSignal(object)
    sig_update_data = QtCore.pyqtSignal(object, float, object, str)
    sig_pick_event = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.grid_data: np.ndarray | None = None
        self.grid_spacing: float = 1.0
        self.grid_origin = np.zeros(3, dtype=float)
        self.current_iso_level: float = 0.5
        self.secondary_iso_level: float = 0.7
        self.data_range = (0.0, 1.0)
        self.view_mode = "physical"

        self.structure_atoms = None
        self.colormap = None

        self._ngl_js_path = _resolve_ngl_js_path()
        self._webengine_available = QtWebEngineWidgets is not None and QtWebEngineCore is not None
        self._ngl_ready = False
        self._ngl_runtime_ready = False
        self._ngl_had_error = False
        self._last_ngl_error = ""
        self._ngl_pending_calls: list[str] = []
        self._ngl_boot_timer = QtCore.QTimer(self)
        self._ngl_boot_timer.setSingleShot(True)
        self._ngl_boot_timer.setInterval(4500)
        self._ngl_boot_timer.timeout.connect(self._on_ngl_boot_timeout)

        self._iso_update_timer = QtCore.QTimer(self)
        self._iso_update_timer.setSingleShot(True)
        self._iso_update_timer.setInterval(100)
        self._iso_update_timer.timeout.connect(self.update_isosurface)

        self._stage_update_timer = QtCore.QTimer(self)
        self._stage_update_timer.setSingleShot(True)
        self._stage_update_timer.setInterval(100)
        self._stage_update_timer.timeout.connect(self._apply_stage_options)

        self._rep_update_timer = QtCore.QTimer(self)
        self._rep_update_timer.setSingleShot(True)
        self._rep_update_timer.setInterval(120)
        self._rep_update_timer.timeout.connect(self._apply_structure_representation)

        self._focus_label: str | None = None
        self._focus_selection: str | None = None
        self._max_density_point: np.ndarray | None = None
        self._max_density_value: float | None = None
        self._latest_fps: float = 0.0

        self._pending_capture: dict | None = None
        self._pending_state_action: dict | None = None
        self._last_pick_event: dict | None = None

        self._rep_rows: list[dict] = []
        self._rep_layer_counter = 0
        self._custom_bg_color = "#05080d"
        self._density_color_1 = "#55aaff"
        self._density_color_2 = "#f97316"
        self._context_panel_visible = False
        self._context_panel_min_width = 460
        self._context_panel_max_width = 620
        self._context_panel_open_width = 520
        self._context_sections: dict[str, dict] = {}
        self._active_context_section: str | None = None
        self._density_error_retries = 0
        self._structure_export_mode = "manual"
        self._structure_parse_retry = False
        self._density_file_version = 0
        self._density_auto_view_pending = False
        self._density_diag: dict[str, object] = {}

        self._work_dir = Path(tempfile.mkdtemp(prefix="sozlab_ngl_"))
        self._density_file: Path | None = None
        self._structure_file: Path | None = None
        self._grid_cache_key = None
        self._structure_cache_key = None

        self._build_ui()
        self._wire_signals()

        self._add_representation_row(
            {
                "visible": True,
                "type": "Ball+Stick",
                "selection": "all",
                "colorScheme": "element",
                "opacity": 1.0,
                "colorValue": "#55aaff",
            }
        )

        if self._is_ngl_runtime_ready():
            self._load_ngl_page()
        else:
            self._set_mode_badge("NGL unavailable")
            reason = ""
            if not self._webengine_available:
                reason = "PyQt6-WebEngine is not installed in the active environment."
            elif self._ngl_js_path is None:
                reason = "Could not locate local NGL build (expected ngl/dist/ngl.js)."
            self._set_notice(reason + " Install requirements and click Reload Viewer.", warning=True)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("3D Molecular Viewer")
        title.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.mode_badge = QtWidgets.QLabel("")
        self.mode_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setMinimumWidth(120)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight
        )

        title_row.addWidget(title)
        title_row.addWidget(self.mode_badge)
        title_row.addStretch(1)
        title_row.addWidget(self.stats_label)
        root.addLayout(title_row)

        self.viewer_stack = QtWidgets.QStackedWidget()
        self.viewer_stack.setMinimumHeight(360)
        self.web_panel = self._build_web_panel()
        self.fallback_panel = self._build_fallback_panel()
        self.viewer_stack.addWidget(self.web_panel)
        self.viewer_stack.addWidget(self.fallback_panel)
        if self._is_ngl_runtime_ready():
            self.viewer_stack.setCurrentWidget(self.web_panel)
            self._set_mode_badge("NGL loading")
        else:
            self.viewer_stack.setCurrentWidget(self.fallback_panel)

        theme_mode = "light"
        parent = self.parentWidget()
        while parent is not None:
            mode = str(getattr(parent, "_theme_mode", "")).strip().lower()
            if mode in {"light", "dark"}:
                theme_mode = mode
                break
            parent = parent.parentWidget()
        arrow_file = "chevron-down-dark.png" if theme_mode == "dark" else "chevron-down-light.png"
        arrow_up_file = "chevron-up-dark.png" if theme_mode == "dark" else "chevron-up-light.png"
        combo_arrow_path = (Path(__file__).resolve().parent / "assets" / arrow_file).as_posix()
        chevron_up_path = (Path(__file__).resolve().parent / "assets" / arrow_up_file).as_posix()
        self._chevron_down_icon = QtGui.QIcon(combo_arrow_path)
        self._chevron_up_icon = QtGui.QIcon(chevron_up_path)
        right_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
        down_pix = QtGui.QPixmap(combo_arrow_path)
        if not down_pix.isNull():
            right_pix = down_pix.transformed(
                QtGui.QTransform().rotate(-90),
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            if not right_pix.isNull():
                right_icon = QtGui.QIcon(right_pix)
        self._chevron_right_icon = right_icon
        app = QtWidgets.QApplication.instance()
        style_blob = app.styleSheet() if app is not None else ""
        self.floating_toolbar = QtWidgets.QFrame()
        self.floating_toolbar.setStyleSheet(
            "QFrame { background: rgba(248,250,252,0.92); border: 1px solid rgba(15,23,42,0.12);"
            " border-radius: 8px; }"
            "QToolButton, QComboBox { min-height: 32px; padding: 4px 12px;"
            " border: 1px solid rgba(148,163,184,0.38); border-radius: 12px;"
            " background: #ffffff; color: #1f2937; margin: 0; }"
            "QComboBox { padding: 4px 30px 4px 12px; }"
            "QToolButton[ctxMenuButton=\"true\"] { padding: 4px 30px 4px 12px; }"
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 22px;"
            " border: none; background: transparent; }"
            f"QComboBox::down-arrow {{ image: url(\"{combo_arrow_path}\"); width: 10px; height: 10px; }}"
            "QToolButton[ctxMenuButton=\"true\"]::menu-indicator {"
            " subcontrol-origin: padding; subcontrol-position: center right;"
            f" image: url(\"{combo_arrow_path}\"); width: 10px; height: 10px; right: 8px; }}"
            "QToolButton:hover, QComboBox:hover { border-color: rgba(59,130,246,0.55);"
            " background: rgba(239,246,255,0.95); }"
            "QToolButton:focus, QComboBox:focus { border-color: rgba(59,130,246,0.62); }"
            "QToolButton:checked { background: rgba(59,130,246,0.18); color: #1d4ed8;"
            " border-color: rgba(59,130,246,0.50); }"
        )
        popup_menu_style = (
            "QMenu { background: #f8fafc; color: #0f172a; border: 1px solid rgba(148,163,184,0.45); padding: 4px; }"
            "QMenu::item { background: transparent; color: #0f172a; padding: 6px 10px; border-radius: 6px; }"
            "QMenu::item:selected { background: rgba(219,234,254,0.95); color: #0f172a; }"
            "QMenu::item:disabled { color: #94a3b8; }"
            "QMenu::separator { height: 1px; background: rgba(148,163,184,0.35); margin: 4px 8px; }"
        )
        floating_row = QtWidgets.QHBoxLayout(self.floating_toolbar)
        floating_row.setContentsMargins(6, 4, 6, 4)
        floating_row.setSpacing(4)

        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(
            ["Cartoon", "Surface", "Ball+Stick", "Licorice", "Ribbon", "Spacefill", "HyperBalls"]
        )
        self.style_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.style_combo.setMinimumContentsLength(11)
        self.style_combo.setCurrentText("Ball+Stick")
        self.style_combo.setToolTip("Quick style preset")
        floating_row.addWidget(self.style_combo)

        self.screenshot_btn = QtWidgets.QToolButton()
        self.screenshot_btn.setText("Shot")
        self.screenshot_btn.setToolTip("Screenshot")
        self.screenshot_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.screenshot_btn.setProperty("ctxMenuButton", True)
        shot_menu = QtWidgets.QMenu(self.screenshot_btn)
        act_png_1x = shot_menu.addAction("PNG 1x")
        act_png_2x = shot_menu.addAction("PNG 2x")
        act_png_4x = shot_menu.addAction("PNG 4x")
        shot_menu.addSeparator()
        act_svg = shot_menu.addAction("SVG")
        shot_menu.setStyleSheet(popup_menu_style)
        self.screenshot_btn.setMenu(shot_menu)
        act_png_1x.triggered.connect(lambda: self._request_capture("png", 1))
        act_png_2x.triggered.connect(lambda: self._request_capture("png", 2))
        act_png_4x.triggered.connect(lambda: self._request_capture("png", 4))
        act_svg.triggered.connect(lambda: self._request_capture("svg", 2))
        floating_row.addWidget(self.screenshot_btn)

        self.reset_cam_btn = QtWidgets.QToolButton()
        self.reset_cam_btn.setText("Reset")
        self.reset_cam_btn.setToolTip("Reset view")
        floating_row.addWidget(self.reset_cam_btn)

        self.settings_btn = QtWidgets.QToolButton()
        self.settings_btn.setText("Opts")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip("Show viewer controls")
        floating_row.addWidget(self.settings_btn)

        self.layers_btn = QtWidgets.QToolButton()
        self.layers_btn.setText("Layers")
        self.layers_btn.setToolTip("Open representation layers")
        floating_row.addWidget(self.layers_btn)

        self.perf_label = QtWidgets.QLabel("FPS: --")
        self.perf_label.setStyleSheet(
            "color: rgba(148,163,184,0.8); font-size: 10px; font-family: 'JetBrains Mono', monospace;"
            "background: rgba(2,6,23,0.34); border-radius: 4px; padding: 2px 5px;"
        )

        viewer_shell = QtWidgets.QWidget()
        viewer_grid = QtWidgets.QGridLayout(viewer_shell)
        viewer_grid.setContentsMargins(0, 0, 0, 0)
        viewer_grid.addWidget(self.viewer_stack, 0, 0)
        viewer_grid.addWidget(
            self.floating_toolbar,
            0,
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        viewer_grid.addWidget(
            self.perf_label,
            0,
            0,
            alignment=QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignRight,
        )

        self.insight_badge = QtWidgets.QLabel("Max density: -")
        self.insight_badge.setStyleSheet(
            "background: rgba(15,23,42,0.08); border: 1px solid rgba(148,163,184,0.25);"
            "border-radius: 8px; padding: 5px 8px; color: #334155;"
        )

        left_host = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_host)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(viewer_shell, 1)
        left_layout.addWidget(self.insight_badge, 0)

        self.context_panel = QtWidgets.QFrame()
        self.context_panel.setObjectName("ViewerContextPanel")
        self.context_panel.setMinimumWidth(self._context_panel_min_width)
        self.context_panel.setMaximumWidth(self._context_panel_max_width)
        self.context_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        checkbox_border = "rgba(148,163,184,0.55)"
        checkbox_background = "#ffffff"
        checkbox_accent = "#3B82F6"
        checkbox_disabled = "rgba(148,163,184,0.35)"
        if app is not None:
            try:
                palette = app.palette()
                checkbox_background = palette.color(QtGui.QPalette.ColorRole.Base).name()
                checkbox_disabled = palette.color(QtGui.QPalette.ColorRole.Mid).name()
            except Exception:
                pass
            if style_blob:
                try:
                    unchecked_block = re.search(
                        r"QCheckBox::indicator\s*\{(?P<body>.*?)\}",
                        style_blob,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if unchecked_block:
                        unchecked_body = unchecked_block.group("body")
                        border_match = re.search(
                            r"border\s*:\s*1px\s+solid\s+([^;]+);",
                            unchecked_body,
                            flags=re.IGNORECASE,
                        )
                        background_match = re.search(
                            r"background\s*:\s*([^;]+);",
                            unchecked_body,
                            flags=re.IGNORECASE,
                        )
                        if border_match:
                            checkbox_border = border_match.group(1).strip()
                        if background_match:
                            checkbox_background = background_match.group(1).strip()
                    checked_block = re.search(
                        r"QCheckBox::indicator:checked\s*\{(?P<body>.*?)\}",
                        style_blob,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if checked_block:
                        checked_body = checked_block.group("body")
                        checked_background_match = re.search(
                            r"background\s*:\s*([^;]+);",
                            checked_body,
                            flags=re.IGNORECASE,
                        )
                        checked_border_match = re.search(
                            r"border\s*:\s*1px\s+solid\s+([^;]+);",
                            checked_body,
                            flags=re.IGNORECASE,
                        )
                        if checked_background_match:
                            checkbox_accent = checked_background_match.group(1).strip()
                        elif checked_border_match:
                            checkbox_accent = checked_border_match.group(1).strip()
                except Exception:
                    pass
        self.context_panel.setStyleSheet(
            "QFrame#ViewerContextPanel { background: rgba(248,250,252,0.98); border: 1px solid rgba(148,163,184,0.30);"
            " border-radius: 10px; }"
            "QFrame[ctxTabPanel=\"true\"] { border: 1px solid rgba(148,163,184,0.28); border-radius: 12px;"
            " background: rgba(255,255,255,0.90); }"
            "QFrame[ctxSegmented=\"true\"] { border: 1px solid rgba(148,163,184,0.35); border-radius: 16px;"
            " background: rgba(248,250,252,0.94); }"
            "QLabel { color: #334155; }"
            "QLineEdit, QDoubleSpinBox, QSpinBox { min-height: 32px; padding: 4px 10px;"
            " border: 1px solid rgba(148,163,184,0.45); border-radius: 10px; background: #ffffff; }"
            "QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: rgba(59,130,246,0.60); }"
            "QPushButton, QToolButton, QComboBox { min-height: 32px; padding: 4px 12px;"
            " border: 1px solid rgba(148,163,184,0.38); border-radius: 12px; background: #ffffff; color: #1f2937; }"
            "QComboBox { padding: 4px 30px 4px 12px; }"
            "QPushButton:hover, QToolButton:hover, QComboBox:hover { border-color: rgba(59,130,246,0.55);"
            " background: rgba(239,246,255,0.95); }"
            "QPushButton:focus, QToolButton:focus, QComboBox:focus { border-color: rgba(59,130,246,0.62); }"
            "QPushButton:checked, QToolButton:checked { background: rgba(59,130,246,0.18); color: #1d4ed8;"
            " border-color: rgba(59,130,246,0.50); }"
            "QToolButton[ctxMenuButton=\"true\"] { padding: 4px 30px 4px 12px; }"
            "QToolButton[ctxMenuButton=\"true\"]::menu-indicator {"
            " subcontrol-origin: padding; subcontrol-position: center right;"
            f" image: url(\"{combo_arrow_path}\"); width: 10px; height: 10px; right: 8px; }}"
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; border: none;"
            " width: 22px; background: transparent; }"
            f"QComboBox::down-arrow {{ image: url(\"{combo_arrow_path}\"); width: 10px; height: 10px; }}"
            "QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QLineEdit:disabled,"
            " QDoubleSpinBox:disabled, QSpinBox:disabled { color: #94a3b8; background: rgba(248,250,252,0.86);"
            " border-color: rgba(148,163,184,0.25); }"
            "QToolButton[ctxQuick=\"true\"] { color: #475569; background: rgba(241,245,249,0.82); font-weight: 500; }"
            "QToolButton[ctxQuick=\"true\"]:checked { background: rgba(59,130,246,0.18); color: #1d4ed8;"
            " border-color: rgba(59,130,246,0.50); }"
            "QToolButton[ctxSegmentBtn=\"true\"] { border: none; border-radius: 12px; min-height: 28px; padding: 4px 10px;"
            " background: transparent; }"
            "QToolButton[ctxSegmentBtn=\"true\"]:hover { background: rgba(219,234,254,0.85); }"
            "QToolButton[ctxSegmentBtn=\"true\"]:checked { background: rgba(59,130,246,0.18); color: #1d4ed8;"
            " border: 1px solid rgba(59,130,246,0.45); }"
            "QCheckBox { spacing: 6px; }"
            f"QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 6px;"
            f" border: 1px solid {checkbox_border}; background: {checkbox_background}; }}"
            f"QCheckBox::indicator:hover {{ border-color: {checkbox_accent}; }}"
            f"QCheckBox::indicator:focus {{ border-color: {checkbox_accent}; }}"
            f"QCheckBox::indicator:checked {{ border-color: {checkbox_accent}; background: {checkbox_accent}; }}"
            f"QCheckBox::indicator:disabled {{ border-color: {checkbox_disabled}; background: {checkbox_background}; }}"
            f"QCheckBox::indicator:checked:disabled {{ border-color: {checkbox_disabled}; background: {checkbox_disabled}; }}"
            "QCheckBox:disabled { color: #94a3b8; }"
            "QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: rgba(148,163,184,0.35); }"
            "QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px;"
            " border: 1px solid rgba(59,130,246,0.7); background: #ffffff; }"
        )
        context_layout = QtWidgets.QVBoxLayout(self.context_panel)
        context_layout.setContentsMargins(10, 8, 10, 8)
        context_layout.setSpacing(6)

        context_header = QtWidgets.QHBoxLayout()
        context_title = QtWidgets.QLabel("Viewer Controls")
        context_title.setStyleSheet("font-size: 12px; letter-spacing: 0.5px; color: #64748b; font-weight: 600;")
        self.context_close_btn = QtWidgets.QToolButton()
        self.context_close_btn.setText("X")
        self.context_close_btn.setToolTip("Hide controls")
        context_header.addWidget(context_title)
        context_header.addStretch(1)
        context_header.addWidget(self.context_close_btn)
        context_layout.addLayout(context_header)

        section_nav = QtWidgets.QHBoxLayout()
        section_nav.setContentsMargins(0, 0, 0, 0)
        section_nav.setSpacing(4)
        self.context_section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.context_section_group = QtWidgets.QButtonGroup(self.context_panel)
        self.context_section_group.setExclusive(True)
        for sec_key, sec_label in (
            ("density", "Density"),
            ("layers", "Layers"),
            ("render", "Render"),
            ("pick", "Pick"),
            ("insights", "Insights"),
        ):
            btn = QtWidgets.QToolButton()
            btn.setText(sec_label)
            btn.setToolTip(f"Open {sec_label} controls")
            btn.setAutoRaise(False)
            btn.setCheckable(True)
            btn.setProperty("ctxQuick", "true")
            btn.setMinimumWidth(0)
            btn.setMinimumHeight(28)
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            btn.clicked.connect(lambda _=False, k=sec_key: self._open_context_section(k))
            self.context_section_group.addButton(btn)
            section_nav.addWidget(btn)
            self.context_section_buttons[sec_key] = btn
        context_layout.addLayout(section_nav)

        self.context_scroll = QtWidgets.QScrollArea()
        self.context_scroll.setWidgetResizable(True)
        self.context_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.context_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.context_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.context_body = QtWidgets.QWidget()
        self.context_body_layout = QtWidgets.QVBoxLayout(self.context_body)
        self.context_body_layout.setContentsMargins(0, 0, 0, 0)
        self.context_body_layout.setSpacing(0)
        self.context_scroll.setWidget(self.context_body)
        context_layout.addWidget(self.context_scroll, 1)

        tab_section_spacing = 6
        form_label_min_width = 108

        def _normalize_form_layout(form: QtWidgets.QFormLayout) -> None:
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(tab_section_spacing)
            form.setLabelAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
                label_widget = label_item.widget() if label_item is not None else None
                if isinstance(label_widget, QtWidgets.QLabel):
                    label_widget.setMinimumWidth(form_label_min_width)

        density_body = QtWidgets.QWidget()
        density_body_layout = QtWidgets.QVBoxLayout(density_body)
        density_body_layout.setContentsMargins(0, 0, 0, 0)
        density_body_layout.setSpacing(tab_section_spacing)

        density_essentials = QtWidgets.QFormLayout()
        density_essentials.setHorizontalSpacing(8)
        density_essentials.setVerticalSpacing(tab_section_spacing)
        density_essentials.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        density_essentials.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.show_density_check = QtWidgets.QCheckBox("Show density")
        self.show_density_check.setChecked(True)
        density_essentials.addRow("Density", self.show_density_check)
        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.iso_slider.setRange(0, 100)
        self.iso_slider.setValue(50)
        self.iso_slider.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.iso_label = QtWidgets.QLabel("0.50")
        self.iso_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.iso_label.setMinimumWidth(72)
        iso_row = QtWidgets.QWidget()
        iso_row_layout = QtWidgets.QHBoxLayout(iso_row)
        iso_row_layout.setContentsMargins(0, 0, 0, 0)
        iso_row_layout.setSpacing(6)
        iso_row_layout.addWidget(self.iso_slider, 1)
        iso_row_layout.addWidget(self.iso_label, 0)
        density_essentials.addRow("Iso level", iso_row)
        self.jump_max_btn = QtWidgets.QPushButton("Jump to max density")
        jump_row = QtWidgets.QWidget()
        jump_row_layout = QtWidgets.QHBoxLayout(jump_row)
        jump_row_layout.setContentsMargins(0, 0, 0, 0)
        jump_row_layout.setSpacing(6)
        jump_row_layout.addWidget(self.jump_max_btn, 0)
        jump_row_layout.addStretch(1)
        density_essentials.addRow("Focus", jump_row)
        _normalize_form_layout(density_essentials)
        density_body_layout.addLayout(density_essentials)

        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Advanced density")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setProperty("ctxExpander", True)
        self.advanced_toggle.setIconSize(QtCore.QSize(10, 10))
        self._set_expander_icon(self.advanced_toggle, False)
        density_body_layout.addWidget(self.advanced_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.density_advanced_panel = QtWidgets.QFrame()
        self.density_advanced_panel.setVisible(False)
        density_adv = QtWidgets.QFormLayout(self.density_advanced_panel)
        density_adv.setHorizontalSpacing(8)
        density_adv.setVerticalSpacing(tab_section_spacing)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["Custom", "P90", "P95", "P99", "Rel 1.0", "Rel 2.0"])
        density_adv.addRow("Preset", self.preset_combo)

        self.density_style_combo = QtWidgets.QComboBox()
        self.density_style_combo.addItems(["Translucent", "Solid", "Wireframe"])
        density_adv.addRow("Surface", self.density_style_combo)

        self.dual_iso_check = QtWidgets.QCheckBox("Dual iso")
        self.dual_iso_check.setChecked(False)
        density_adv.addRow("", self.dual_iso_check)

        self.secondary_iso_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.secondary_iso_slider.setRange(0, 100)
        self.secondary_iso_slider.setValue(70)

        self.secondary_iso_label = QtWidgets.QLabel("0.70")
        self.secondary_iso_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.secondary_iso_label.setMinimumWidth(88)
        secondary_iso_row = QtWidgets.QWidget()
        secondary_iso_row_layout = QtWidgets.QHBoxLayout(secondary_iso_row)
        secondary_iso_row_layout.setContentsMargins(0, 0, 0, 0)
        secondary_iso_row_layout.setSpacing(6)
        secondary_iso_row_layout.addWidget(self.secondary_iso_slider, 1)
        secondary_iso_row_layout.addWidget(self.secondary_iso_label, 0)
        density_adv.addRow("Secondary iso", secondary_iso_row)

        self.primary_color_btn = QtWidgets.QPushButton("Primary color")
        self.secondary_color_btn = QtWidgets.QPushButton("Secondary color")
        colors_row = QtWidgets.QWidget()
        colors_row_layout = QtWidgets.QHBoxLayout(colors_row)
        colors_row_layout.setContentsMargins(0, 0, 0, 0)
        colors_row_layout.setSpacing(6)
        colors_row_layout.addWidget(self.primary_color_btn, 1)
        colors_row_layout.addWidget(self.secondary_color_btn, 1)
        density_adv.addRow("Colors", colors_row)
        _normalize_form_layout(density_adv)
        density_body_layout.addWidget(self.density_advanced_panel)

        rep_body = QtWidgets.QWidget()
        rep_layout = QtWidgets.QVBoxLayout(rep_body)
        rep_layout.setContentsMargins(0, 0, 0, 0)
        rep_layout.setSpacing(tab_section_spacing)
        rep_host = QtWidgets.QWidget()
        self.rep_rows_layout = QtWidgets.QVBoxLayout(rep_host)
        self.rep_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rep_rows_layout.setSpacing(tab_section_spacing)
        self.rep_rows_layout.addStretch(1)
        rep_layout.addWidget(rep_host, 1)
        self.add_rep_btn = QtWidgets.QPushButton("Add representation")
        rep_layout.addWidget(self.add_rep_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        render_body = QtWidgets.QWidget()
        render_layout = QtWidgets.QVBoxLayout(render_body)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.setSpacing(tab_section_spacing)

        render_top = QtWidgets.QFormLayout()
        render_top.setHorizontalSpacing(8)
        render_top.setVerticalSpacing(tab_section_spacing)
        self.camera_toggle_btn = QtWidgets.QToolButton()
        self.camera_toggle_btn.setCheckable(True)
        self.camera_toggle_btn.setChecked(False)
        self.camera_toggle_btn.setText("Perspective")
        self.camera_toggle_btn.setToolTip("Toggle perspective/orthographic camera")
        self.center_btn = QtWidgets.QToolButton()
        self.center_btn.setText("Center")
        self.center_btn.setToolTip("Center on selection")
        self.background_combo = QtWidgets.QComboBox()
        self.background_combo.addItems(["Dark", "Light", "Black", "White", "Custom"])
        self.background_combo.setCurrentText("Dark")
        self.bg_custom_btn = QtWidgets.QToolButton()
        self.bg_custom_btn.setText("Color")
        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(["Draft", "Balanced", "High", "Ultra"])
        self.quality_combo.setCurrentText("Balanced")
        self.transparent_bg_check = QtWidgets.QCheckBox("Transparent capture")

        camera_row = QtWidgets.QWidget()
        camera_row.setProperty("ctxSegmented", True)
        camera_row_layout = QtWidgets.QHBoxLayout(camera_row)
        camera_row_layout.setContentsMargins(3, 3, 3, 3)
        camera_row_layout.setSpacing(2)
        self.camera_toggle_btn.setProperty("ctxSegmentBtn", True)
        self.center_btn.setProperty("ctxSegmentBtn", True)
        camera_row_layout.addWidget(self.camera_toggle_btn)
        camera_row_layout.addWidget(self.center_btn)
        camera_row_layout.addStretch(1)
        render_top.addRow("Camera", camera_row)

        background_row = QtWidgets.QWidget()
        background_row_layout = QtWidgets.QHBoxLayout(background_row)
        background_row_layout.setContentsMargins(0, 0, 0, 0)
        background_row_layout.setSpacing(6)
        background_row_layout.addWidget(self.background_combo, 1)
        background_row_layout.addWidget(self.bg_custom_btn, 0)
        render_top.addRow("Background", background_row)

        render_top.addRow("Quality", self.quality_combo)
        render_top.addRow("Transparent", self.transparent_bg_check)
        _normalize_form_layout(render_top)
        render_layout.addLayout(render_top)

        render_state_row = QtWidgets.QHBoxLayout()
        render_state_row.setContentsMargins(0, 0, 0, 0)
        render_state_row.setSpacing(6)
        self.state_btn = QtWidgets.QToolButton()
        self.state_btn.setText("View state")
        self.state_btn.setProperty("ctxMenuButton", True)
        self.state_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        state_menu = QtWidgets.QMenu(self.state_btn)
        act_copy = state_menu.addAction("Copy JSON")
        act_save = state_menu.addAction("Save JSON")
        state_menu.setStyleSheet(popup_menu_style)
        self.state_btn.setMenu(state_menu)
        act_copy.triggered.connect(self._request_state_copy)
        act_save.triggered.connect(self._request_state_save)
        self.retry_gl_btn = QtWidgets.QToolButton()
        self.retry_gl_btn.setText("Reload")
        self.retry_gl_btn.setToolTip("Reload NGL runtime")
        render_state_row.addWidget(self.state_btn)
        render_state_row.addWidget(self.retry_gl_btn)
        render_state_row.addStretch(1)
        render_layout.addLayout(render_state_row)

        self.render_advanced_toggle = QtWidgets.QToolButton()
        self.render_advanced_toggle.setText("Advanced rendering")
        self.render_advanced_toggle.setCheckable(True)
        self.render_advanced_toggle.setChecked(False)
        self.render_advanced_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.render_advanced_toggle.setProperty("ctxExpander", True)
        self.render_advanced_toggle.setIconSize(QtCore.QSize(10, 10))
        self._set_expander_icon(self.render_advanced_toggle, False)
        render_layout.addWidget(self.render_advanced_toggle, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.advanced_panel = QtWidgets.QFrame()
        self.advanced_panel.setVisible(False)
        adv = QtWidgets.QFormLayout(self.advanced_panel)
        adv.setHorizontalSpacing(8)
        adv.setVerticalSpacing(tab_section_spacing)

        self.clip_near_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.clip_near_slider.setRange(0, 100)
        self.clip_near_slider.setValue(0)
        self.clip_far_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.clip_far_slider.setRange(1, 100)
        self.clip_far_slider.setValue(100)
        self.clip_dist_spin = QtWidgets.QDoubleSpinBox()
        self.clip_dist_spin.setRange(0.0, 500.0)
        self.clip_dist_spin.setDecimals(2)
        self.clip_dist_spin.setSingleStep(1.0)

        self.fog_near_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fog_near_slider.setRange(0, 100)
        self.fog_near_slider.setValue(50)
        self.fog_far_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fog_far_slider.setRange(0, 100)
        self.fog_far_slider.setValue(100)

        self.ambient_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ambient_slider.setRange(0, 200)
        self.ambient_slider.setValue(100)
        self.light_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.light_slider.setRange(0, 200)
        self.light_slider.setValue(100)
        self.shininess_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.shininess_slider.setRange(0, 100)
        self.shininess_slider.setValue(40)
        self.metalness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.metalness_slider.setRange(0, 100)
        self.metalness_slider.setValue(20)

        self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fov_slider.setRange(15, 120)
        self.fov_slider.setValue(40)

        self.spin_check = QtWidgets.QCheckBox("Spin")
        self.spin_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.spin_speed_slider.setRange(1, 100)
        self.spin_speed_slider.setValue(8)

        self.rock_check = QtWidgets.QCheckBox("Rock")
        self.rock_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rock_speed_slider.setRange(1, 100)
        self.rock_speed_slider.setValue(10)

        self.mouse_preset_combo = QtWidgets.QComboBox()
        self.mouse_preset_combo.addItems(["Default", "Coot", "PyMOL-style"])

        adv.addRow("Clip near %", self.clip_near_slider)
        adv.addRow("Clip far %", self.clip_far_slider)
        adv.addRow("Clip dist (A)", self.clip_dist_spin)
        adv.addRow("FOV", self.fov_slider)
        adv.addRow("Fog near %", self.fog_near_slider)
        adv.addRow("Fog far %", self.fog_far_slider)
        adv.addRow("Ambient", self.ambient_slider)
        adv.addRow("Light", self.light_slider)
        adv.addRow("Shininess", self.shininess_slider)
        adv.addRow("Metalness", self.metalness_slider)

        spin_row = QtWidgets.QWidget()
        spin_row_layout = QtWidgets.QHBoxLayout(spin_row)
        spin_row_layout.setContentsMargins(0, 0, 0, 0)
        spin_row_layout.setSpacing(6)
        spin_row_layout.addWidget(self.spin_check, 0)
        spin_row_layout.addWidget(self.spin_speed_slider, 1)
        adv.addRow("Spin", spin_row)

        rock_row = QtWidgets.QWidget()
        rock_row_layout = QtWidgets.QHBoxLayout(rock_row)
        rock_row_layout.setContentsMargins(0, 0, 0, 0)
        rock_row_layout.setSpacing(6)
        rock_row_layout.addWidget(self.rock_check, 0)
        rock_row_layout.addWidget(self.rock_speed_slider, 1)
        adv.addRow("Rock", rock_row)

        adv.addRow("Mouse preset", self.mouse_preset_combo)
        _normalize_form_layout(adv)
        render_layout.addWidget(self.advanced_panel)

        measure_body = QtWidgets.QWidget()
        measure_body.setObjectName("PickControlsBody")
        measure_body.setStyleSheet(
            "QFrame[pickCard=\"true\"] {"
            " background: rgba(241,245,249,0.72);"
            " border: 1px solid rgba(148,163,184,0.32);"
            " border-radius: 10px;"
            "}"
            "QFrame[pickDivider=\"true\"] {"
            " background: rgba(148,163,184,0.32);"
            " min-height: 1px;"
            " max-height: 1px;"
            " border: none;"
            " margin: 2px 0;"
            "}"
            "QLabel[pickTitle=\"true\"] {"
            " color: #0f172a;"
            " font-weight: 600;"
            "}"
            "QLabel[pickHint=\"true\"] {"
            " color: #64748b;"
            "}"
            "QLabel[pickBodyText=\"true\"] {"
            " margin: 0px;"
            " padding-top: 0px;"
            " padding-bottom: 0px;"
            "}"
            "QLabel[pickEmpty=\"true\"] {"
            " color: #64748b;"
            " border: 1px dashed rgba(148,163,184,0.5);"
            " border-radius: 8px;"
            " background: rgba(248,250,252,0.92);"
            " padding: 10px;"
            "}"
            "QListWidget[pickLog=\"true\"] {"
            " background: rgba(255,255,255,0.95);"
            " border: 1px solid rgba(148,163,184,0.45);"
            " border-radius: 8px;"
            " padding: 4px;"
            "}"
            "QListWidget[pickLog=\"true\"]::item {"
            " border: none;"
            " margin: 1px;"
            "}"
            "QListWidget[pickLog=\"true\"]::item:selected {"
            " background: rgba(219,234,254,0.85);"
            " border-radius: 8px;"
            "}"
            "QFrame[measureRow=\"true\"] {"
            " background: rgba(248,250,252,0.96);"
            " border: 1px solid rgba(148,163,184,0.35);"
            " border-radius: 8px;"
            "}"
            "QLabel[measureValue=\"true\"] {"
            " color: #0f172a;"
            " font-weight: 600;"
            "}"
            "QLabel[measurePath=\"true\"] {"
            " color: #475569;"
            "}"
            "QPushButton[pickAction=\"true\"] {"
            " border: 1px solid rgba(148,163,184,0.42);"
            " background: #ffffff;"
            " color: #1f2937;"
            "}"
            "QPushButton[pickAction=\"true\"]:hover {"
            " border-color: rgba(59,130,246,0.55);"
            " background: rgba(239,246,255,0.95);"
            "}"
            "QPushButton[pickAction=\"true\"][pickActive=\"true\"] {"
            " border-color: rgba(37,99,235,0.62);"
            " background: rgba(219,234,254,0.98);"
            " color: #1e3a8a;"
            "}"
            "QPushButton[pickAction=\"true\"]:disabled {"
            " border-color: rgba(148,163,184,0.25);"
            " color: #94a3b8;"
            " background: rgba(248,250,252,0.86);"
            "}"
        )
        meas_layout = QtWidgets.QVBoxLayout(measure_body)
        meas_layout.setContentsMargins(0, 0, 0, 0)
        meas_layout.setSpacing(8)
        pick_intro = QtWidgets.QLabel(
            "Pick atoms in the viewer to measure distances and add custom labels."
        )
        pick_intro.setWordWrap(True)
        pick_intro.setProperty("pickHint", True)
        pick_intro.setProperty("pickBodyText", True)
        meas_layout.addWidget(pick_intro)

        intro_divider = QtWidgets.QFrame()
        intro_divider.setProperty("pickDivider", True)
        meas_layout.addWidget(intro_divider)

        measure_card = QtWidgets.QFrame()
        measure_card.setProperty("pickCard", True)
        measure_card_layout = QtWidgets.QVBoxLayout(measure_card)
        measure_card_layout.setContentsMargins(8, 8, 8, 8)
        measure_card_layout.setSpacing(6)
        measure_title = QtWidgets.QLabel("Measurement")
        measure_title.setProperty("pickTitle", True)
        measure_card_layout.addWidget(measure_title, 0)
        measure_controls_row = QtWidgets.QHBoxLayout()
        measure_controls_row.setContentsMargins(0, 0, 0, 0)
        measure_controls_row.setSpacing(6)
        self.measure_mode_label = QtWidgets.QLabel("Distance")
        self.measure_mode_label.setProperty("pickHint", True)
        self.clear_measure_btn = QtWidgets.QPushButton("Clear")
        self.clear_measure_btn.setProperty("pickAction", True)
        measure_controls_row.addWidget(self.measure_mode_label, 1)
        measure_controls_row.addWidget(self.clear_measure_btn, 0)
        measure_card_layout.addLayout(measure_controls_row)
        self.measure_mode_hint_label = QtWidgets.QLabel("Pick 2 atoms in order to calculate Distance.")
        self.measure_mode_hint_label.setWordWrap(True)
        self.measure_mode_hint_label.setProperty("pickHint", True)
        self.measure_mode_hint_label.setProperty("pickBodyText", True)
        measure_card_layout.addWidget(self.measure_mode_hint_label)
        meas_layout.addWidget(measure_card, 0)

        mid_divider = QtWidgets.QFrame()
        mid_divider.setProperty("pickDivider", True)
        meas_layout.addWidget(mid_divider)

        label_card = QtWidgets.QFrame()
        label_card.setProperty("pickCard", True)
        label_card_layout = QtWidgets.QVBoxLayout(label_card)
        label_card_layout.setContentsMargins(8, 8, 8, 8)
        label_card_layout.setSpacing(6)
        label_title = QtWidgets.QLabel("Labels")
        label_title.setProperty("pickTitle", True)
        label_card_layout.addWidget(label_title, 0)
        self.auto_label_check = QtWidgets.QCheckBox("Auto-label picks")
        label_card_layout.addWidget(self.auto_label_check, 0)
        label_row_widget = QtWidgets.QWidget()
        label_row = QtWidgets.QHBoxLayout(label_row_widget)
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(6)
        label_row.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.custom_label_edit = QtWidgets.QLineEdit()
        self.custom_label_edit.setPlaceholderText("Label text for the last pick")
        label_row.addWidget(self.custom_label_edit, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.add_label_btn = QtWidgets.QPushButton("Add")
        self.add_label_btn.setProperty("pickAction", True)
        self.add_label_btn.setProperty("pickActive", False)
        label_row.addWidget(self.add_label_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        label_card_layout.addWidget(label_row_widget, 0)
        label_hint = QtWidgets.QLabel("Tip: press Enter to add a label quickly.")
        label_hint.setWordWrap(True)
        label_hint.setProperty("pickHint", True)
        label_hint.setProperty("pickBodyText", True)
        label_card_layout.addWidget(label_hint, 0)
        meas_layout.addWidget(label_card, 0)

        log_divider = QtWidgets.QFrame()
        log_divider.setProperty("pickDivider", True)
        meas_layout.addWidget(log_divider)

        btn_width = max(
            self.clear_measure_btn.sizeHint().width(),
            self.add_label_btn.sizeHint().width(),
        )
        measure_field_height = max(
            self.clear_measure_btn.sizeHint().height(),
            self.measure_mode_label.sizeHint().height(),
        )
        label_field_height = max(
            self.add_label_btn.sizeHint().height(),
            self.custom_label_edit.sizeHint().height(),
            measure_field_height,
        )
        self.clear_measure_btn.setMinimumWidth(btn_width)
        self.add_label_btn.setMinimumWidth(btn_width)
        self.clear_measure_btn.setFixedHeight(measure_field_height)
        self.add_label_btn.setFixedHeight(label_field_height)
        self.custom_label_edit.setFixedHeight(label_field_height)

        log_card = QtWidgets.QFrame()
        log_card.setProperty("pickCard", True)
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)
        log_title = QtWidgets.QLabel("Measurements Log")
        log_title.setProperty("pickTitle", True)
        log_layout.addWidget(log_title, 0)

        self.measure_log = QtWidgets.QListWidget()
        self.measure_log.setProperty("pickLog", True)
        self.measure_log.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.measure_log.setAlternatingRowColors(False)
        self.measure_log.setSpacing(4)
        self.measure_log.setMinimumHeight(110)
        self.measure_log.setMaximumHeight(190)
        self.measure_log_empty_label = QtWidgets.QLabel(
            "No measurements yet. Pick 2 atoms to measure distance."
        )
        self.measure_log_empty_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.measure_log_empty_label.setWordWrap(True)
        self.measure_log_empty_label.setProperty("pickEmpty", True)
        self.measure_log_empty_label.setProperty("pickBodyText", True)
        self.measure_log_stack = QtWidgets.QStackedWidget()
        self.measure_log_stack.addWidget(self.measure_log_empty_label)
        self.measure_log_stack.addWidget(self.measure_log)
        log_layout.addWidget(self.measure_log_stack, 1)
        meas_layout.addWidget(log_card, 1)
        self.pick_status_label = QtWidgets.QLabel("")
        self.pick_status_label.setWordWrap(True)
        self.pick_status_label.setVisible(False)
        self.pick_status_label.setProperty("pickHint", True)
        self.pick_status_label.setProperty("pickBodyText", True)
        meas_layout.addWidget(self.pick_status_label)
        self._set_measure_log_empty(True)

        insight_body = QtWidgets.QWidget()
        insight_layout = QtWidgets.QVBoxLayout(insight_body)
        insight_layout.setContentsMargins(0, 0, 0, 0)
        insight_layout.setSpacing(tab_section_spacing)
        self.notice_label = QtWidgets.QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setVisible(False)
        insight_layout.addWidget(self.notice_label)
        self.structure_info_label = QtWidgets.QLabel("Structure: none")
        self.structure_info_label.setStyleSheet("color: #94a3b8;")
        insight_layout.addWidget(self.structure_info_label)
        self.density_insights_text = QtWidgets.QTextEdit()
        self.density_insights_text.setReadOnly(True)
        self.density_insights_text.setPlaceholderText("Density insights will appear here.")
        self.density_insights_text.setMinimumHeight(96)
        self.density_insights_text.setMaximumHeight(240)
        self.density_insights_text.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.density_insights_text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        insight_layout.addWidget(self.density_insights_text)

        self._add_context_section("density", "Density Controls", density_body)
        self._add_context_section("layers", "Representation Layers", rep_body)
        self._add_context_section("render", "Rendering Settings", render_body)
        self._add_context_section("pick", "Picking Labels", measure_body)
        self._add_context_section("insights", "Density Insights", insight_body)
        self.context_body_layout.addStretch(1)

        self._set_active_context_section("density")
        self._set_context_panel_visible(True)

        content = QtWidgets.QWidget()
        content_row = QtWidgets.QHBoxLayout(content)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)
        content_row.addWidget(left_host, 1)
        content_row.addWidget(self.context_panel, 0)
        root.addWidget(content, 1)

    def _add_context_section(self, key: str, title: str, body_widget: QtWidgets.QWidget) -> None:
        _ = title  # Top-level labeling is handled by the tab row buttons.
        section_panel = QtWidgets.QFrame()
        section_panel.setProperty("ctxTabPanel", True)
        section_panel_layout = QtWidgets.QVBoxLayout(section_panel)
        section_panel_layout.setContentsMargins(8, 8, 8, 8)
        section_panel_layout.setSpacing(0)

        body_widget.setVisible(True)
        body_widget.setContentsMargins(0, 0, 0, 0)
        section_panel_layout.addWidget(body_widget)

        self.context_body_layout.addWidget(section_panel)
        self._context_sections[key] = {
            "body": section_panel,
            "content": body_widget,
        }

    def _set_active_context_section(self, key: str | None) -> None:
        if key not in self._context_sections:
            key = next(iter(self._context_sections.keys()), None)
        self._active_context_section = key
        for sec_key, sec in self._context_sections.items():
            is_open = sec_key == key
            body = sec["body"]
            body.setVisible(is_open)
            quick_btn = self.context_section_buttons.get(sec_key)
            if quick_btn is not None:
                quick_btn.blockSignals(True)
                quick_btn.setChecked(is_open)
                quick_btn.blockSignals(False)

    def _set_context_panel_visible(self, visible: bool) -> None:
        self._context_panel_visible = bool(visible)
        if self._context_panel_visible:
            target = int(
                np.clip(
                    self._context_panel_open_width,
                    self._context_panel_min_width,
                    self._context_panel_max_width,
                )
            )
            self.context_panel.setMinimumWidth(target)
            self.context_panel.setMaximumWidth(target)
            self.context_panel.setFixedWidth(target)
            self.context_panel.setVisible(True)
        else:
            try:
                current_width = int(self.context_panel.width())
                if current_width > 0:
                    self._context_panel_open_width = current_width
            except Exception:
                pass
            self.context_panel.setMinimumWidth(0)
            self.context_panel.setMaximumWidth(0)
            self.context_panel.setFixedWidth(0)
            self.context_panel.setVisible(False)
        self.settings_btn.blockSignals(True)
        self.settings_btn.setChecked(self._context_panel_visible)
        self.settings_btn.blockSignals(False)

    def _open_context_section(self, key: str) -> None:
        if key not in self._context_sections:
            return
        self._set_context_panel_visible(True)
        self._set_active_context_section(key)

    def _on_settings_toggled(self, checked: bool) -> None:
        self._set_context_panel_visible(bool(checked))
        if checked:
            self._set_active_context_section(self._active_context_section or "density")

    def set_context_panel_visible(self, visible: bool) -> None:
        self._set_context_panel_visible(bool(visible))

    def _build_web_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = None
        self.web_page = None
        if QtWebEngineWidgets is not None and QtWebEngineCore is not None:
            self.web_view = QtWebEngineWidgets.QWebEngineView(panel)
            self.web_page = _NGLPage(self.web_view)
            self.web_view.setPage(self.web_page)
            self.web_page.console_event.connect(self._on_js_event)
            self.web_view.loadFinished.connect(self._on_web_load_finished)

            settings = self.web_view.settings()
            wa = QtWebEngineCore.QWebEngineSettings.WebAttribute
            settings.setAttribute(wa.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(wa.LocalContentCanAccessRemoteUrls, False)
            settings.setAttribute(wa.ErrorPageEnabled, True)

            layout.addWidget(self.web_view)
        else:
            placeholder = QtWidgets.QLabel("Qt WebEngine is unavailable.")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #94a3b8;")
            layout.addWidget(placeholder)
        return panel

    def _build_fallback_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        label = QtWidgets.QLabel(
            "NGL viewer could not be initialized.\n"
            "Install `PyQt6-WebEngine`, ensure `ngl/dist/ngl.js` is present, then click Reload."
        )
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "background: #0f172a; color: #d1d5db; border: 1px solid #334155;"
            "padding: 18px; border-radius: 8px;"
        )
        layout.addWidget(label, 1)
        return panel

    def _wire_signals(self) -> None:
        self.iso_slider.valueChanged.connect(self._on_slider_changed)
        self.secondary_iso_slider.valueChanged.connect(self._on_secondary_slider_changed)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.quality_combo.currentTextChanged.connect(self._on_quality_changed)
        self.style_combo.currentTextChanged.connect(self._on_style_preset_changed)
        self.background_combo.currentTextChanged.connect(self._on_background_mode_changed)
        self.bg_custom_btn.clicked.connect(self._pick_custom_background)
        self.show_density_check.toggled.connect(self._on_density_toggle)
        self.density_style_combo.currentTextChanged.connect(lambda _=None: self._iso_update_timer.start())
        self.dual_iso_check.toggled.connect(lambda _=None: self._iso_update_timer.start())
        self.primary_color_btn.clicked.connect(lambda: self._pick_density_color(primary=True))
        self.secondary_color_btn.clicked.connect(lambda: self._pick_density_color(primary=False))

        self.reset_cam_btn.clicked.connect(self._on_reset_camera)
        self.center_btn.clicked.connect(self._on_center_selection)
        self.camera_toggle_btn.toggled.connect(self._on_camera_toggled)
        self.retry_gl_btn.clicked.connect(self._on_reload_viewer)
        self.jump_max_btn.clicked.connect(self._on_jump_to_max_density)
        self.settings_btn.toggled.connect(self._on_settings_toggled)
        self.layers_btn.clicked.connect(lambda: self._open_context_section("layers"))
        self.context_close_btn.clicked.connect(lambda: self._set_context_panel_visible(False))

        self.advanced_toggle.toggled.connect(self._toggle_advanced_panel)
        self.render_advanced_toggle.toggled.connect(self._toggle_render_advanced_panel)
        for widget in (
            self.clip_near_slider,
            self.clip_far_slider,
            self.clip_dist_spin,
            self.fog_near_slider,
            self.fog_far_slider,
            self.ambient_slider,
            self.light_slider,
            self.shininess_slider,
            self.metalness_slider,
            self.fov_slider,
            self.spin_check,
            self.spin_speed_slider,
            self.rock_check,
            self.rock_speed_slider,
            self.mouse_preset_combo,
        ):
            if isinstance(widget, QtWidgets.QSlider):
                widget.valueChanged.connect(self._queue_stage_update)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.valueChanged.connect(self._queue_stage_update)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._queue_stage_update)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._queue_stage_update)

        self.clear_measure_btn.clicked.connect(self._on_clear_measurements)
        self.auto_label_check.toggled.connect(self._on_auto_label_toggled)
        self.add_label_btn.clicked.connect(self._on_add_custom_label)
        self.custom_label_edit.returnPressed.connect(self._on_add_custom_label)

        self.add_rep_btn.clicked.connect(lambda: self._add_representation_row())

        self.sig_update_data.connect(self._update_data_slot)
        self.sig_update_structure.connect(self._update_structure_slot)
        self._refresh_pick_controls_state()

    def _is_ngl_runtime_ready(self) -> bool:
        return self._webengine_available and self._ngl_js_path is not None and self.web_view is not None

    def _load_ngl_page(self) -> None:
        if not self._is_ngl_runtime_ready():
            return
        self._ngl_ready = False
        self._ngl_runtime_ready = False
        self._ngl_had_error = False
        self._ngl_pending_calls.clear()
        self._ngl_boot_timer.stop()
        base_url = QtCore.QUrl.fromLocalFile(str(self._ngl_js_path.parent) + "/")
        self.web_view.setHtml(_build_ngl_html(), base_url)

    def _on_web_load_finished(self, ok: bool) -> None:
        if not ok:
            self._set_mode_badge("NGL unavailable")
            self._set_notice("Failed to load embedded NGL page.", warning=True)
            self._refresh_pick_controls_state()
            self.viewer_stack.setCurrentWidget(self.fallback_panel)
            return

        self._ngl_ready = True
        self.viewer_stack.setCurrentWidget(self.web_panel)
        if self._ngl_had_error:
            self._set_mode_badge("NGL unavailable")
        elif self._ngl_runtime_ready:
            self._set_mode_badge("NGL WebGL")
        else:
            self._set_mode_badge("NGL loading")
            self._ngl_boot_timer.start()
        self._flush_pending_js()
        self._refresh_all_components()

    def _js_invoke(self, method: str, *args) -> None:
        if not self._is_ngl_runtime_ready():
            return
        args_code = ", ".join(json.dumps(arg, ensure_ascii=False) for arg in args)
        script = (
            "(function(){"
            f"if(window.sozlabApi && window.sozlabApi.{method}){{window.sozlabApi.{method}({args_code});}}"
            "})();"
        )
        if self._ngl_ready:
            self.web_view.page().runJavaScript(script)
        else:
            self._ngl_pending_calls.append(script)

    def _flush_pending_js(self) -> None:
        if not self._is_ngl_runtime_ready() or not self._ngl_ready:
            return
        while self._ngl_pending_calls:
            script = self._ngl_pending_calls.pop(0)
            self.web_view.page().runJavaScript(script)

    @QtCore.pyqtSlot(object)
    def _on_js_event(self, event: dict) -> None:
        if not isinstance(event, dict):
            return

        etype = event.get("type", "")
        if etype == "error":
            msg = str(event.get("message", "NGL runtime error."))
            if msg != self._last_ngl_error:
                logger.warning("NGL viewer error: %s", msg)
                self._last_ngl_error = msg
            lowered = msg.lower()
            fatal = (
                "webgl initialization failed" in lowered
                or "ngl is not available in web runtime" in lowered
                or "could not create a webgl context" in lowered
            )
            if not fatal:
                if "density" in lowered and self.grid_data is not None and self._density_error_retries < 2:
                    self._density_error_retries += 1
                    min_v, max_v = self.data_range
                    span = max(max_v - min_v, 1e-6)
                    self.current_iso_level = self._clamp_iso_level(min_v + span * 0.55)
                    self.secondary_iso_level = self._clamp_iso_level(min_v + span * 0.70)
                    self._sync_slider_from_level(False)
                    self._sync_slider_from_level(True)
                    self._update_iso_label(self.current_iso_level, secondary=False)
                    self._update_iso_label(self.secondary_iso_level, secondary=True)
                    self.show_density_check.blockSignals(True)
                    self.show_density_check.setChecked(True)
                    self.show_density_check.blockSignals(False)
                    self._set_notice(
                        f"{msg} Retrying density render (attempt {self._density_error_retries}).",
                        warning=True,
                    )
                    if "load density" in lowered:
                        self._write_density_file_if_needed(force=True)
                        self._load_density_component(force_reload=True)
                    else:
                        self.update_isosurface()
                    return
                self._set_notice(msg, warning=True)
                return
            self._ngl_had_error = True
            self._ngl_runtime_ready = False
            self._ngl_boot_timer.stop()
            self._set_mode_badge("NGL unavailable")
            self._set_notice(msg, warning=True)
            self._refresh_pick_controls_state()
            return

        if etype == "ready":
            self._ngl_runtime_ready = True
            self._density_error_retries = 0
            self._ngl_boot_timer.stop()
            self._set_mode_badge("NGL WebGL")
            self._set_notice("")
            self._refresh_pick_controls_state()
            self._enable_distance_measurement_mode()
            self._on_auto_label_toggled(self.auto_label_check.isChecked())
            self._set_pick_status(
                "Distance mode is active. Pick 2 atoms to measure; labels require an atom or residue pick.",
                warning=False,
            )
            self._apply_stage_options()
            self._apply_structure_representation()
            self.update_isosurface()
            return

        if etype == "perf":
            fps = float(event.get("fps", 0.0))
            frame_ms = float(event.get("frameMs", 0.0))
            self._latest_fps = fps
            self.perf_label.setText(f"FPS {fps:.1f}")
            self.perf_label.setToolTip(f"FPS: {fps:.1f} | frame: {frame_ms:.2f} ms")
            return

        if etype == "structure":
            atoms = int(event.get("atoms", 0))
            atoms_all = int(event.get("allSelectionAtoms", atoms) or atoms)
            bbox = event.get("bbox", None)
            center = event.get("center", None)
            radius = float(event.get("radius", 0.0) or 0.0)
            transform = event.get("transform", None)
            logger.warning(
                "Density3D NGL structure: atoms=%d selection_all=%d bbox=%s center=%s radius=%.6f transform=%s",
                atoms,
                atoms_all,
                bbox,
                center,
                radius,
                transform,
            )
            self._set_stats(f"{atoms:,} atoms")
            expected = int(getattr(self.structure_atoms, "n_atoms", 0) or 0)
            if atoms <= 1 and expected > 1 and not self._structure_parse_retry:
                self._structure_parse_retry = True
                self._structure_export_mode = (
                    "manual" if self._structure_export_mode != "manual" else "mda"
                )
                if self._reload_structure_component():
                    self._set_notice(
                        f"Structure parse fallback: retrying load via {self._structure_export_mode} export.",
                        warning=True,
                    )
                    return
            elif atoms <= 1 and expected > 1:
                self._set_notice(
                    "Structure loaded with unexpectedly low atom count in viewer.",
                    warning=True,
                )
            if atoms_all <= 0 and atoms > 0:
                self._set_notice(
                    "Structure loaded but NGL selection 'all' resolved to 0 atoms.",
                    warning=True,
                )
            self._apply_structure_representation()
            if self.grid_data is not None and self.show_density_check.isChecked():
                self._js_invoke("autoViewCombined", 220, {"expandClip": True})
            return

        if etype == "density_load":
            ok = bool(event.get("ok", False))
            url = str(event.get("url", ""))
            has_volume = bool(event.get("hasVolume", False))
            values = int(event.get("values", 0) or 0)
            bbox = event.get("bbox", None)
            center = event.get("center", None)
            radius = float(event.get("radius", 0.0) or 0.0)
            transform = event.get("transform", None)
            logger.warning(
                "Density3D NGL load: ok=%s has_volume=%s values=%d url=%s bbox=%s center=%s radius=%.6f transform=%s",
                ok,
                has_volume,
                values,
                url,
                bbox,
                center,
                radius,
                transform,
            )
            if not ok:
                if self.grid_data is not None and self._density_error_retries < 2:
                    self._density_error_retries += 1
                    self._set_notice(
                        f"NGL failed to load density volume. Retrying ({self._density_error_retries}).",
                        warning=True,
                    )
                    self._write_density_file_if_needed(force=True)
                    self._load_density_component(force_reload=True)
                    return
                self._set_notice("NGL failed to load density volume.", warning=True)
            return

        if etype == "density":
            reps = int(event.get("reps", -1))
            visible = bool(event.get("visible", True))
            iso = float(event.get("isolevel", 0.0) or 0.0)
            dmin = float(event.get("dataMin", 0.0) or 0.0)
            dmax = float(event.get("dataMax", 0.0) or 0.0)
            bbox = event.get("bbox", None)
            center = event.get("center", None)
            radius = float(event.get("radius", 0.0) or 0.0)
            transform = event.get("transform", None)
            logger.warning(
                "Density3D NGL reps: reps=%d visible=%s isolevel=%.6f range=(%.6f, %.6f) bbox=%s center=%s radius=%.6f transform=%s",
                reps,
                visible,
                iso,
                dmin,
                dmax,
                bbox,
                center,
                radius,
                transform,
            )
            if (
                visible
                and reps == 0
                and self.grid_data is not None
                and self._density_error_retries < 2
            ):
                self._density_error_retries += 1
                min_v, max_v = self.data_range
                span = max(max_v - min_v, 1e-6)
                self.current_iso_level = self._clamp_iso_level(min_v + span * 0.35)
                self.secondary_iso_level = self._clamp_iso_level(min_v + span * 0.55)
                self._sync_slider_from_level(False)
                self._sync_slider_from_level(True)
                self._update_iso_label(self.current_iso_level, secondary=False)
                self._update_iso_label(self.secondary_iso_level, secondary=True)
                self._set_notice(
                    "Density surface was empty at current threshold; lowering iso level automatically.",
                    warning=True,
                )
                self.update_isosurface()
                return
            self._density_error_retries = 0
            if reps > 0:
                self._set_notice("")
            return

        if etype == "rep":
            rep_count = int(event.get("count", 0) or 0)
            structure_atoms = int(event.get("structureAtoms", 0) or 0)
            selected_atoms_total = int(event.get("selectedAtomsTotal", 0) or 0)
            layers = event.get("layers", None)
            logger.warning(
                "Density3D NGL structure reps: count=%d structure_atoms=%d selected_atoms_total=%d layers=%s",
                rep_count,
                structure_atoms,
                selected_atoms_total,
                layers,
            )
            if structure_atoms > 0 and selected_atoms_total <= 0 and rep_count > 0:
                self._set_notice(
                    "Structure reps were created but selections resolved to zero atoms.",
                    warning=True,
                )
            return

        if etype == "scene_diag":
            logger.warning(
                "Density3D NGL scene diag: reason=%s has_structure=%s has_density=%s "
                "structure_bbox=%s structure_center=%s structure_radius=%s structure_transform=%s "
                "density_bbox=%s density_center=%s density_radius=%s density_transform=%s",
                event.get("reason", ""),
                bool(event.get("hasStructure", False)),
                bool(event.get("hasDensity", False)),
                event.get("structureBBox", None),
                event.get("structureCenter", None),
                event.get("structureRadius", None),
                event.get("structureTransform", None),
                event.get("densityBBox", None),
                event.get("densityCenter", None),
                event.get("densityRadius", None),
                event.get("densityTransform", None),
            )
            return

        if etype == "pick":
            self._last_pick_event = dict(event)
            self._refresh_pick_controls_state()
            self._handle_pick_event(event)
            self.sig_pick_event.emit(dict(event))
            return

        if etype == "measurement":
            self._append_measurement(event)
            return

        if etype == "measurement_clear":
            self._clear_measure_log()
            return

        if etype == "label_clear":
            self._clear_pick_label_state(clear_input=True)
            self._refresh_pick_controls_state()
            return

        if etype == "image":
            self._handle_capture_event(event)
            return

        if etype == "state":
            self._handle_state_event(event)
            return

    def _on_ngl_boot_timeout(self) -> None:
        if self._ngl_runtime_ready:
            return
        self._set_mode_badge("NGL unavailable")
        self._set_notice(
            "NGL did not finish WebGL initialization. Check graphics driver/WebGL support.",
            warning=True,
        )
        self._refresh_pick_controls_state()

    def closeEvent(self, event) -> None:
        try:
            if self._is_ngl_runtime_ready():
                self._js_invoke("clearAll")
        except Exception:
            pass
        try:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(event)

    def _set_mode_badge(self, text: str) -> None:
        self.mode_badge.setText(text)
        if "NGL" in text and "unavailable" not in text.lower():
            self.mode_badge.setStyleSheet(
                "padding: 3px 10px; border-radius: 9px; background: #0f3f2f; color: #c6f6d5;"
            )
        else:
            self.mode_badge.setStyleSheet(
                "padding: 3px 10px; border-radius: 9px; background: #4a3410; color: #fde68a;"
            )

    def _set_notice(self, text: str, warning: bool = False) -> None:
        if not text:
            self.notice_label.clear()
            self.notice_label.setVisible(False)
            return
        self.notice_label.setText(text)
        self.notice_label.setStyleSheet("color: #f59e0b;" if warning else "color: #94a3b8;")
        self.notice_label.setVisible(True)

    def _set_measure_log_empty(self, empty: bool) -> None:
        stack = getattr(self, "measure_log_stack", None)
        if stack is None:
            return
        stack.setCurrentIndex(0 if empty else 1)

    def _clear_measure_log(self) -> None:
        if hasattr(self, "measure_log"):
            try:
                self.measure_log.clear()
            except Exception:
                pass
        self._set_measure_log_empty(True)

    def _build_measurement_entry_widget(
        self,
        mode: str,
        value: float,
        unit: str,
        labels: list[str],
    ) -> QtWidgets.QWidget:
        mode_key = (mode or "measurement").strip().lower()
        mode_name = (mode or "measurement").strip().title() or "Measurement"
        value_text = f"{value:.3f} {unit}".strip()
        atoms_text = " -> ".join(labels) if labels else "No atom labels"
        chip_colors = {
            "distance": ("#1d4ed8", "#dbeafe"),
            "angle": ("#6d28d9", "#ede9fe"),
            "dihedral": ("#c2410c", "#ffedd5"),
        }
        chip_fg, chip_bg = chip_colors.get(mode_key, ("#334155", "#e2e8f0"))

        row = QtWidgets.QFrame()
        row.setProperty("measureRow", True)
        row_layout = QtWidgets.QVBoxLayout(row)
        row_layout.setContentsMargins(8, 7, 8, 7)
        row_layout.setSpacing(4)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)

        chip = QtWidgets.QLabel(mode_name)
        chip.setStyleSheet(
            f"color: {chip_fg}; background: {chip_bg}; border-radius: 7px; padding: 2px 8px; font-weight: 600;"
        )
        head.addWidget(chip, 0)
        head.addStretch(1)

        value_label = QtWidgets.QLabel(value_text)
        value_label.setProperty("measureValue", True)
        head.addWidget(value_label, 0)
        row_layout.addLayout(head)

        path_label = QtWidgets.QLabel(atoms_text)
        path_label.setWordWrap(True)
        path_label.setProperty("measurePath", True)
        row_layout.addWidget(path_label)
        return row

    def _set_pick_status(self, text: str, warning: bool = False) -> None:
        if not hasattr(self, "pick_status_label"):
            return
        if not text:
            self.pick_status_label.clear()
            self.pick_status_label.setVisible(False)
            return
        self.pick_status_label.setText(text)
        lowered = text.strip().lower()
        if warning:
            style = (
                "color: #b45309; background: rgba(254,243,199,0.75);"
                " border: 1px solid rgba(245,158,11,0.45); border-radius: 8px; padding: 6px 8px;"
            )
        elif lowered.startswith("added label"):
            style = (
                "color: #166534; background: rgba(220,252,231,0.75);"
                " border: 1px solid rgba(34,197,94,0.42); border-radius: 8px; padding: 6px 8px;"
            )
        elif lowered.startswith("picked "):
            style = (
                "color: #1e40af; background: rgba(219,234,254,0.78);"
                " border: 1px solid rgba(59,130,246,0.42); border-radius: 8px; padding: 6px 8px;"
            )
        else:
            style = (
                "color: #475569; background: rgba(241,245,249,0.78);"
                " border: 1px solid rgba(148,163,184,0.35); border-radius: 8px; padding: 6px 8px;"
            )
        self.pick_status_label.setStyleSheet(style)
        self.pick_status_label.setVisible(True)

    def _has_valid_label_pick_target(self) -> bool:
        pick = self._last_pick_event if isinstance(self._last_pick_event, dict) else None
        if not pick:
            return False
        kind = str(pick.get("kind", "")).strip().lower()
        if kind not in {"atom", "residue"}:
            return False
        try:
            x = float(pick.get("x", np.nan))
            y = float(pick.get("y", np.nan))
            z = float(pick.get("z", np.nan))
        except (TypeError, ValueError):
            return False
        return bool(np.all(np.isfinite([x, y, z])))

    def _clear_pick_label_state(self, clear_input: bool = False) -> None:
        self._last_pick_event = None
        if clear_input and hasattr(self, "custom_label_edit"):
            self.custom_label_edit.clear()

    def _refresh_pick_controls_state(self) -> None:
        ready = bool(self._is_ngl_runtime_ready() and self._ngl_runtime_ready and not self._ngl_had_error)
        controls = (
            "measure_mode_label",
            "clear_measure_btn",
            "auto_label_check",
            "custom_label_edit",
        )
        for name in controls:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(ready)
        has_pick = self._has_valid_label_pick_target()
        add_btn = getattr(self, "add_label_btn", None)
        if add_btn is not None:
            add_enabled = ready and has_pick
            add_btn.setEnabled(add_enabled)
            try:
                add_btn.setProperty("pickActive", bool(add_enabled))
                style = add_btn.style()
                if style is not None:
                    style.unpolish(add_btn)
                    style.polish(add_btn)
            except Exception:
                pass
        if not ready:
            self._set_pick_status("Picking is unavailable while the 3D viewer initializes.", warning=True)

    def _set_stats(self, text: str) -> None:
        self.stats_label.setText(text)

    def _quality_profile(self) -> QualityProfile:
        key = self.quality_combo.currentText().strip() or "Balanced"
        return _QUALITY_PROFILES.get(key, _QUALITY_PROFILES["Balanced"])

    def _clamp_iso_level(self, value: float) -> float:
        min_v, max_v = self.data_range
        if not np.isfinite(min_v):
            min_v = 0.0
        if not np.isfinite(max_v) or max_v <= min_v:
            return float(min_v)
        span = max_v - min_v
        eps = max(1e-8, span * 1e-4)
        lo = min_v + eps
        hi = max_v - eps
        if hi <= lo:
            return float(np.clip(value, min_v, max_v))
        return float(np.clip(value, lo, hi))

    def _default_iso_level(self) -> float:
        if self.grid_data is None:
            return 0.5
        min_v, max_v = self.data_range
        values = np.asarray(self.grid_data, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return self._clamp_iso_level(min_v + (max_v - min_v) * 0.5)
        positive = values[values > (min_v + max((max_v - min_v) * 1e-6, 1e-12))]
        if self.view_mode == "physical":
            source = positive if positive.size else values
            mean_v = float(np.mean(source))
            std_v = float(np.std(source))
            level = mean_v + 2.0 * std_v
            if not np.isfinite(level) or level <= min_v or level >= max_v:
                level = float(max_v * 0.2)
        elif self.view_mode == "relative":
            level = 2.0
        elif self.view_mode == "score":
            source = positive if positive.size else values
            level = float(np.percentile(source, 30))
        else:
            level = min_v + (max_v - min_v) * 0.6
        return self._clamp_iso_level(level)

    def _sync_slider_from_level(self, secondary: bool = False) -> None:
        min_v, max_v = self.data_range
        value = self.secondary_iso_level if secondary else self.current_iso_level
        if max_v <= min_v:
            slider_value = 50
        else:
            slider_value = int(round((value - min_v) / (max_v - min_v) * 100.0))
        slider_value = int(np.clip(slider_value, 0, 100))
        target = self.secondary_iso_slider if secondary else self.iso_slider
        target.blockSignals(True)
        target.setValue(slider_value)
        target.blockSignals(False)

    def _update_iso_label(self, value: float, secondary: bool = False) -> None:
        label = self.secondary_iso_label if secondary else self.iso_label
        if self.view_mode == "physical":
            label.setText(f"{value:.4f} A^-3")
        else:
            label.setText(f"{value:.3f}")

    def _on_slider_changed(self, value: int) -> None:
        if self.grid_data is None:
            return
        min_v, max_v = self.data_range
        self.current_iso_level = self._clamp_iso_level(min_v + (max_v - min_v) * (value / 100.0))
        self._update_iso_label(self.current_iso_level)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self._iso_update_timer.start()

    def _on_secondary_slider_changed(self, value: int) -> None:
        if self.grid_data is None:
            return
        min_v, max_v = self.data_range
        self.secondary_iso_level = self._clamp_iso_level(min_v + (max_v - min_v) * (value / 100.0))
        self._update_iso_label(self.secondary_iso_level, secondary=True)
        self._iso_update_timer.start()

    def _on_preset_changed(self, text: str) -> None:
        if self.grid_data is None or text == "Custom":
            return
        min_v, max_v = self.data_range
        level = self.current_iso_level
        if text.startswith("P"):
            try:
                percentile = float(text[1:])
                level = float(np.percentile(self.grid_data, percentile))
            except Exception:
                return
        elif text.startswith("Rel"):
            try:
                level = float(text.split()[-1])
            except Exception:
                return
        self.current_iso_level = self._clamp_iso_level(level)
        self._sync_slider_from_level()
        self._update_iso_label(self.current_iso_level)
        self.update_isosurface()

    def _on_quality_changed(self, _text: str) -> None:
        # Quality should only affect renderer tessellation, not density data.
        self.update_isosurface()
        self._queue_stage_update()
        self._apply_structure_representation()

    def _on_style_preset_changed(self, text: str) -> None:
        if not self._rep_rows:
            return
        row = self._rep_rows[0]
        combo = row.get("type_combo")
        if combo is None:
            return
        combo.blockSignals(True)
        combo.setCurrentText(text)
        combo.blockSignals(False)
        self._rep_update_timer.start()

    def _on_density_toggle(self, _checked: bool) -> None:
        self.update_isosurface()

    def set_focus_label(self, label: str | None) -> None:
        self._focus_label = label.strip() if label else None
        if self._focus_label:
            self._set_notice(f"Linked selection: {self._focus_label}", warning=False)
        elif self.notice_label.isVisible() and self.notice_label.text().startswith("Linked selection:"):
            self._set_notice("")

    def _on_reset_camera(self) -> None:
        self._js_invoke("autoViewCombined", 200, {"expandClip": True})

    def _on_center_selection(self) -> None:
        selection = self._focus_selection
        if not selection and self._last_pick_event and self._last_pick_event.get("kind") == "atom":
            idx = int(self._last_pick_event.get("atomIndex", 0))
            selection = f"@{idx}"
        if selection:
            self._js_invoke("focusSelection", selection)
        else:
            self._js_invoke("autoView")

    def _on_camera_toggled(self, checked: bool) -> None:
        self.camera_toggle_btn.setText("Orthographic" if checked else "Perspective")
        self._queue_stage_update()

    def _on_background_mode_changed(self, mode: str) -> None:
        if mode == "Custom":
            self._pick_custom_background()
        self._queue_stage_update()

    def _pick_custom_background(self) -> None:
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._custom_bg_color), self, "Background")
        if not color.isValid():
            return
        self._custom_bg_color = color.name()
        self.background_combo.blockSignals(True)
        self.background_combo.setCurrentText("Custom")
        self.background_combo.blockSignals(False)
        self._queue_stage_update()

    def _pick_density_color(self, primary: bool) -> None:
        current = self._density_color_1 if primary else self._density_color_2
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, "Density color")
        if not color.isValid():
            return
        if primary:
            self._density_color_1 = color.name()
        else:
            self._density_color_2 = color.name()
        self._iso_update_timer.start()

    def _on_reload_viewer(self) -> None:
        self._ngl_js_path = _resolve_ngl_js_path()
        self._ngl_runtime_ready = False
        self._ngl_had_error = False
        self._clear_pick_label_state(clear_input=True)
        self._refresh_pick_controls_state()
        if self._is_ngl_runtime_ready():
            self.viewer_stack.setCurrentWidget(self.web_panel)
            self._set_mode_badge("NGL loading")
            self._load_ngl_page()
        else:
            self.viewer_stack.setCurrentWidget(self.fallback_panel)
            self._set_mode_badge("NGL unavailable")
            self._set_notice("NGL runtime is not available in this environment.", warning=True)

    def _queue_stage_update(self, *_args) -> None:
        self._stage_update_timer.start()

    def _set_expander_icon(self, button: QtWidgets.QToolButton, expanded: bool) -> None:
        if button is None:
            return
        icon = self._chevron_down_icon if expanded else self._chevron_right_icon
        button.setIcon(icon)

    def _toggle_advanced_panel(self, expanded: bool) -> None:
        self.density_advanced_panel.setVisible(expanded)
        self._set_expander_icon(self.advanced_toggle, expanded)

    def _toggle_render_advanced_panel(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self._set_expander_icon(self.render_advanced_toggle, expanded)

    def _background_color(self) -> str:
        mode = self.background_combo.currentText().strip().lower()
        if mode == "dark":
            return "#05080d"
        if mode == "light":
            return "#e8edf3"
        if mode == "black":
            return "#000000"
        if mode == "white":
            return "#ffffff"
        return self._custom_bg_color

    def _stage_payload(self) -> dict:
        return {
            "backgroundColor": self._background_color(),
            "cameraType": "orthographic" if self.camera_toggle_btn.isChecked() else "perspective",
            "clipNear": int(self.clip_near_slider.value()),
            "clipFar": int(self.clip_far_slider.value()),
            "clipDist": float(self.clip_dist_spin.value()),
            "fogNear": int(self.fog_near_slider.value()),
            "fogFar": int(self.fog_far_slider.value()),
            "ambientIntensity": float(self.ambient_slider.value() / 100.0),
            "lightIntensity": float(self.light_slider.value() / 100.0),
            "shininess": float(self.shininess_slider.value() / 100.0),
            "metalness": float(self.metalness_slider.value() / 100.0),
            "fov": int(self.fov_slider.value()),
            "spinEnabled": bool(self.spin_check.isChecked()),
            "spinSpeed": float(self.spin_speed_slider.value() / 100.0),
            "rockEnabled": bool(self.rock_check.isChecked()),
            "rockSpeed": float(self.rock_speed_slider.value() / 100.0),
            "mousePreset": self.mouse_preset_combo.currentText(),
            "lodEnabled": True,
        }

    def _apply_stage_options(self) -> None:
        self._js_invoke("setStageOptions", self._stage_payload())

    def _enable_distance_measurement_mode(self) -> None:
        self._js_invoke("setMeasurementMode", "distance")
        if hasattr(self, "measure_mode_hint_label"):
            self.measure_mode_hint_label.setText("Pick 2 atoms in order to calculate Distance.")

    def _on_measure_mode_changed(self, _text: str) -> None:
        # Backward-compatible shim in case external callers still invoke this hook.
        self._enable_distance_measurement_mode()

    def _on_clear_measurements(self) -> None:
        self._clear_measure_log()
        self._clear_pick_label_state(clear_input=True)
        self._focus_selection = None
        self._js_invoke("highlightSelection", "")
        self._js_invoke("clearMeasurements")
        self._js_invoke("clearLabels")
        self._enable_distance_measurement_mode()
        self._refresh_pick_controls_state()
        self._set_pick_status("Cleared picks and labels.", warning=False)
        self._set_notice("Cleared picks and labels.", warning=False)

    def _on_auto_label_toggled(self, enabled: bool) -> None:
        self._js_invoke("setAutoLabel", bool(enabled))
        state = "enabled" if enabled else "disabled"
        self._set_pick_status(
            f"Auto-label is {state}. Manual labels replace auto labels at the same point.",
            warning=False,
        )

    def _on_add_custom_label(self) -> None:
        text = self.custom_label_edit.text().strip()
        pick = self._last_pick_event or {}
        kind = str(pick.get("kind", "")).strip().lower()
        if not self._has_valid_label_pick_target():
            self._set_pick_status("Pick an atom or residue first, then add a custom label.", warning=True)
            self._set_notice("Pick an atom or residue first, then add a custom label.", warning=True)
            return
        if not text:
            if kind == "atom":
                fallback = str(pick.get("label", "")).strip()
                if not fallback:
                    atom = str(pick.get("atomName", "")).strip()
                    resname = str(pick.get("resname", "")).strip()
                    resno = int(pick.get("resno", 0) or 0)
                    fallback = f"{atom} {resname}{resno}".strip()
                text = fallback or "Atom"
            else:
                fallback = str(pick.get("label", "")).strip()
                if not fallback:
                    resname = str(pick.get("resname", "")).strip()
                    resno = int(pick.get("resno", 0) or 0)
                    fallback = f"{resname}{resno}".strip()
                text = fallback or "Residue"
        try:
            x = float(pick.get("x", np.nan))
            y = float(pick.get("y", np.nan))
            z = float(pick.get("z", np.nan))
        except (TypeError, ValueError):
            self._set_pick_status("Picked point has invalid coordinates. Pick again.", warning=True)
            self._set_notice("Picked point has invalid coordinates. Pick again.", warning=True)
            return
        if not bool(np.all(np.isfinite([x, y, z]))):
            self._set_pick_status("Picked point has invalid coordinates. Pick again.", warning=True)
            self._set_notice("Picked point has invalid coordinates. Pick again.", warning=True)
            return
        payload = {
            "x": x,
            "y": y,
            "z": z,
            "text": text,
            "color": "#f8fafc",
        }
        self._js_invoke("addCustomLabel", payload)
        self._set_pick_status(f"Added label '{text}'.", warning=False)
        self._set_notice(f"Added label '{text}'.", warning=False)

    def _append_measurement(self, event: dict) -> None:
        mode = str(event.get("mode", "")).strip()
        value = float(event.get("value", 0.0))
        unit = str(event.get("unit", "")).strip()
        points = event.get("points", []) or []
        labels = []
        for p in points:
            resname = str(p.get("resname", "")).strip()
            resno = int(p.get("resno", 0))
            atom = str(p.get("atomName", "")).strip()
            labels.append(f"{resname}{resno}:{atom}")
        if isinstance(getattr(self, "measure_log", None), QtWidgets.QListWidget):
            item = QtWidgets.QListWidgetItem()
            row_widget = self._build_measurement_entry_widget(mode, value, unit, labels)
            item.setSizeHint(row_widget.sizeHint())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mode.strip().lower())
            self.measure_log.addItem(item)
            self.measure_log.setItemWidget(item, row_widget)
            self.measure_log.setCurrentItem(item)
            self.measure_log.scrollToItem(item)
            self._set_measure_log_empty(False)
            return
        msg = f"{mode.title()}: {value:.3f} {unit} | {' -> '.join(labels)}"
        if hasattr(self.measure_log, "append"):
            self.measure_log.append(msg)

    def _handle_pick_event(self, event: dict) -> None:
        kind = str(event.get("kind", "")).strip().lower()
        if kind == "atom":
            atom = str(event.get("atomName", "")).strip()
            resname = str(event.get("resname", "")).strip()
            resno = int(event.get("resno", 0) or 0)
            chain = str(event.get("chain", "")).strip()
            x = float(event.get("x", 0.0))
            y = float(event.get("y", 0.0))
            z = float(event.get("z", 0.0))
            self._set_notice(
                f"Picked {atom} in {resname}{resno} chain {chain} at ({x:.2f}, {y:.2f}, {z:.2f})",
                warning=False,
            )
            idx = int(event.get("atomIndex", 0) or 0)
            self._focus_selection = f"@{idx}"
            self._js_invoke("highlightSelection", self._focus_selection)
            self._set_context_panel_visible(True)
            self._set_active_context_section("pick")
            self._set_pick_status(
                f"Picked {atom} in {resname}{resno}. Use Add to place a label.",
                warning=False,
            )
            return

        if kind == "density":
            point = np.array(
                [float(event.get("x", 0.0)), float(event.get("y", 0.0)), float(event.get("z", 0.0))],
                dtype=float,
            )
            nearest = self._nearest_structure_info(point)
            if nearest:
                label = nearest["label"]
                self._focus_selection = nearest["selection"]
                self._js_invoke("highlightSelection", self._focus_selection)
                self._set_notice(
                    f"Picked point ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) -> nearest {label}",
                    warning=False,
                )
            else:
                self._set_notice(
                    f"Picked point ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})",
                    warning=False,
                )
            self._set_context_panel_visible(True)
            self._set_active_context_section("pick")
            self._set_pick_status(
                "Picked point. Manual labels require an atom or residue pick.",
                warning=False,
            )

    def _request_capture(self, fmt: str, factor: int) -> None:
        fmt = (fmt or "png").strip().lower()
        if fmt == "svg":
            filt = "SVG (*.svg)"
            suffix = "svg"
            default = "ngl_view.svg"
        else:
            filt = "PNG (*.png)"
            suffix = "png"
            default = "ngl_view.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export View", default, filt)
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != f".{suffix}":
            p = p.with_suffix(f".{suffix}")

        self._pending_capture = {
            "path": p,
            "format": fmt,
            "factor": int(max(1, factor)),
            "transparent": bool(self.transparent_bg_check.isChecked()),
        }
        self._js_invoke(
            "captureImage",
            {
                "format": "png",
                "factor": int(max(1, factor)),
                "transparent": bool(self.transparent_bg_check.isChecked()),
            },
        )

    def _handle_capture_event(self, event: dict) -> None:
        pending = self._pending_capture
        self._pending_capture = None
        if not pending:
            return

        data_url = str(event.get("dataUrl", ""))
        if not data_url.startswith("data:image/"):
            self._set_notice("Capture failed: invalid image payload.", warning=True)
            return

        try:
            encoded = data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded)
        except Exception:
            self._set_notice("Capture failed: could not decode payload.", warning=True)
            return

        path: Path = pending["path"]
        fmt = str(pending.get("format", "png")).lower()

        try:
            if fmt == "svg":
                b64 = base64.b64encode(image_bytes).decode("ascii")
                svg = (
                    "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
                    "viewBox=\"0 0 100 100\" preserveAspectRatio=\"none\">"
                    f"<image href=\"data:image/png;base64,{b64}\" x=\"0\" y=\"0\" "
                    "width=\"100\" height=\"100\" preserveAspectRatio=\"none\"/>"
                    "</svg>"
                )
                path.write_text(svg, encoding="utf-8")
            else:
                path.write_bytes(image_bytes)
        except Exception as exc:
            self._set_notice(f"Capture failed: {exc}", warning=True)
            return

        self._set_notice(f"Captured view to {path}", warning=False)

    def _request_state_copy(self) -> None:
        self._pending_state_action = {"mode": "clipboard"}
        self._js_invoke("getViewState")

    def _request_state_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save View State",
            "ngl_view_state.json",
            "JSON (*.json)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        self._pending_state_action = {"mode": "file", "path": p}
        self._js_invoke("getViewState")

    def _handle_state_event(self, event: dict) -> None:
        action = self._pending_state_action
        self._pending_state_action = None
        if not action:
            return

        state = event.get("state", {})
        payload = json.dumps(state, indent=2, ensure_ascii=False)
        mode = action.get("mode")
        if mode == "clipboard":
            QtWidgets.QApplication.clipboard().setText(payload)
            self._set_notice("Copied view state JSON to clipboard.", warning=False)
            return

        if mode == "file":
            path: Path = action["path"]
            try:
                path.write_text(payload, encoding="utf-8")
            except Exception as exc:
                self._set_notice(f"Failed to save view state: {exc}", warning=True)
                return
            self._set_notice(f"Saved view state to {path}", warning=False)

    def set_colormap(self, cmap) -> None:
        self.colormap = cmap
        if self.grid_data is not None:
            self.update_isosurface()

    def set_data(self, grid: np.ndarray, spacing: float, origin: np.ndarray, view_mode: str = "physical") -> None:
        self.sig_update_data.emit(grid, spacing, origin, view_mode)

    def _update_data_slot(self, grid, spacing, origin, view_mode) -> None:
        if grid is None:
            self.grid_data = None
            self._density_file = None
            self._max_density_point = None
            self._max_density_value = None
            self._density_diag = {}
            self._js_invoke(
                "updateDensity",
                {
                    "visible": False,
                    "isolevel": 0.0,
                    "isolevel2": 0.0,
                    "dualIso": False,
                    "color": self._density_color_1,
                    "color2": self._density_color_2,
                    "opacity": 0.0,
                    "opacity2": 0.0,
                    "quality": "medium",
                    "style": "Translucent",
                },
            )
            self._set_stats("")
            self._update_density_insight()
            return

        arr = np.asarray(grid, dtype=np.float32)
        if arr.ndim != 3 or arr.size == 0:
            self._set_notice("Density volume is empty.", warning=True)
            return

        self.grid_data = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._density_error_retries = 0
        self.grid_spacing = max(float(spacing), 1e-6)
        try:
            parsed_origin = np.asarray(origin, dtype=float).reshape(-1)
            self.grid_origin = parsed_origin[:3] if parsed_origin.size >= 3 else np.zeros(3, dtype=float)
        except Exception:
            self.grid_origin = np.zeros(3, dtype=float)
        self.view_mode = view_mode or "physical"

        min_val = float(np.min(self.grid_data))
        max_val = float(np.max(self.grid_data))
        if not np.isfinite(min_val):
            min_val = 0.0
        if not np.isfinite(max_val) or max_val <= min_val:
            max_val = min_val + 1e-6
        self.data_range = (min_val, max_val)
        self._density_diag = self._build_density_diag(self.grid_data)
        self._log_density_diag()
        self._log_density_structure_alignment()

        self.current_iso_level = self._default_iso_level()
        self.secondary_iso_level = self._clamp_iso_level(
            self.current_iso_level + (max_val - min_val) * 0.12
        )
        if not self.show_density_check.isChecked():
            self.show_density_check.blockSignals(True)
            self.show_density_check.setChecked(True)
            self.show_density_check.blockSignals(False)
        self._sync_slider_from_level(secondary=False)
        self._sync_slider_from_level(secondary=True)
        self._update_iso_label(self.current_iso_level, secondary=False)
        self._update_iso_label(self.secondary_iso_level, secondary=True)

        self._compute_max_density_hotspot()
        self._update_density_insight()
        self._density_auto_view_pending = True

        self._write_density_file_if_needed(force=True)
        self._load_density_component(force_reload=True)

    def _build_density_diag(self, grid: np.ndarray) -> dict[str, object]:
        grid_f = np.asarray(grid, dtype=np.float32)
        shape = tuple(int(v) for v in grid_f.shape)
        spacing = float(self.grid_spacing)
        origin = np.asarray(self.grid_origin, dtype=float).reshape(3)
        bbox_min = origin
        bbox_max = origin + (np.array(shape, dtype=float) - 1.0) * spacing
        finite = grid_f[np.isfinite(grid_f)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=np.float32)
        dmin = float(np.min(finite))
        dmax = float(np.max(finite))
        dmean = float(np.mean(finite))
        dstd = float(np.std(finite))
        nonzero = int(np.count_nonzero(finite))
        voxel_count = int(np.prod(shape))
        suggested_iso = dmean + 2.0 * dstd
        if not np.isfinite(suggested_iso) or suggested_iso <= dmin or suggested_iso >= dmax:
            suggested_iso = float(dmax * 0.2) if dmax > 0 else float(dmin + max((dmax - dmin) * 0.2, 1e-6))
        suggested_iso = float(np.clip(suggested_iso, dmin, dmax))
        return {
            "shape": shape,
            "spacing": spacing,
            "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
            "bbox_min": (float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
            "bbox_max": (float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
            "min": dmin,
            "max": dmax,
            "mean": dmean,
            "std": dstd,
            "nonzero": nonzero,
            "voxels": voxel_count,
            "dtype": str(grid_f.dtype),
            "bytes": int(grid_f.nbytes),
            "suggested_iso": suggested_iso,
        }

    def _log_density_diag(self) -> None:
        if not self._density_diag:
            return
        d = self._density_diag
        logger.warning(
            "Density3D volume: shape=%s spacing=%.6f origin=(%.3f, %.3f, %.3f) "
            "bbox=((%.3f, %.3f, %.3f)->(%.3f, %.3f, %.3f)) min=%.6f mean=%.6f max=%.6f std=%.6f "
            "nonzero=%d/%d dtype=%s bytes=%d suggested_iso=%.6f",
            d["shape"],
            float(d["spacing"]),
            *d["origin"],
            *d["bbox_min"],
            *d["bbox_max"],
            float(d["min"]),
            float(d["mean"]),
            float(d["max"]),
            float(d["std"]),
            int(d["nonzero"]),
            int(d["voxels"]),
            d["dtype"],
            int(d["bytes"]),
            float(d["suggested_iso"]),
        )

    def _log_density_structure_alignment(self) -> None:
        if self.structure_atoms is None or not self._density_diag:
            return
        try:
            coords = np.asarray(self.structure_atoms.positions, dtype=float)
            if coords.ndim != 2 or coords.shape[0] == 0:
                return
            s_min = coords.min(axis=0)
            s_max = coords.max(axis=0)
            d_min = np.asarray(self._density_diag["bbox_min"], dtype=float)
            d_max = np.asarray(self._density_diag["bbox_max"], dtype=float)
            overlap = np.all(np.minimum(s_max, d_max) >= np.maximum(s_min, d_min))
            if not overlap:
                logger.warning(
                    "Density3D alignment warning: structure bbox ((%.3f, %.3f, %.3f)->(%.3f, %.3f, %.3f)) "
                    "does not overlap density bbox ((%.3f, %.3f, %.3f)->(%.3f, %.3f, %.3f)).",
                    float(s_min[0]),
                    float(s_min[1]),
                    float(s_min[2]),
                    float(s_max[0]),
                    float(s_max[1]),
                    float(s_max[2]),
                    float(d_min[0]),
                    float(d_min[1]),
                    float(d_min[2]),
                    float(d_max[0]),
                    float(d_max[1]),
                    float(d_max[2]),
                )
        except Exception:
            return

    def _compute_max_density_hotspot(self) -> None:
        if self.grid_data is None or self.grid_data.size == 0:
            self._max_density_point = None
            self._max_density_value = None
            return
        flat_idx = int(np.nanargmax(self.grid_data))
        max_val = float(np.ravel(self.grid_data)[flat_idx])
        idx = np.array(np.unravel_index(flat_idx, self.grid_data.shape), dtype=float)
        point = self.grid_origin + idx * float(self.grid_spacing)
        self._max_density_point = point
        self._max_density_value = max_val

    def _update_density_insight(self) -> None:
        if self._max_density_point is None or self._max_density_value is None:
            self.insight_badge.setText("Max density: -")
            if hasattr(self, "density_insights_text"):
                self.density_insights_text.setPlainText("Density map is not loaded.")
            self._js_invoke("setBadge", "NGL Viewer")
            return

        point = self._max_density_point
        nearest = self._nearest_structure_info(point)
        if nearest:
            text = (
                f"Max density: {self._max_density_value:.4f} at "
                f"{nearest['label']} ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
            )
            self._focus_selection = nearest["selection"]
        else:
            text = (
                f"Max density: {self._max_density_value:.4f} at "
                f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
            )
        self.insight_badge.setText(text)
        if hasattr(self, "density_insights_text"):
            detail_lines = [
                text,
                f"Iso level: {self.current_iso_level:.4f}",
                f"Secondary iso: {self.secondary_iso_level:.4f}",
                f"View mode: {self.view_mode}",
            ]
            self.density_insights_text.setPlainText("\n".join(detail_lines))
        self._js_invoke("setBadge", text)

    def _on_jump_to_max_density(self) -> None:
        if self._max_density_point is None:
            self._set_notice("Density map is not loaded.", warning=True)
            return
        point = self._max_density_point
        payload = {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
        }
        self._js_invoke("focusDensityVolume", {"point": payload})
        self._js_invoke("focusPosition", payload)
        nearest = self._nearest_structure_info(point)
        if nearest:
            self._focus_selection = nearest["selection"]
            self._js_invoke("highlightSelection", self._focus_selection)

    def _nearest_structure_info(self, point: np.ndarray) -> dict | None:
        atoms = self.structure_atoms
        if atoms is None:
            return None
        try:
            n_atoms = int(getattr(atoms, "n_atoms", 0))
            if n_atoms <= 0:
                return None
            coords = np.asarray(atoms.positions, dtype=float)
            if coords.ndim != 2 or coords.shape[0] == 0:
                return None
            delta = coords - point.reshape(1, 3)
            idx = int(np.argmin(np.sum(delta * delta, axis=1)))

            atom_names = np.asarray(getattr(atoms, "names", np.array(["?"] * n_atoms)), dtype=object)
            resnames = np.asarray(getattr(atoms, "resnames", np.array(["MOL"] * n_atoms)), dtype=object)
            resids = np.asarray(getattr(atoms, "resids", np.arange(n_atoms)), dtype=int)
            chainids = np.asarray(getattr(atoms, "chainIDs", np.array(["A"] * n_atoms)), dtype=object)

            atom_name = str(atom_names[idx]) if idx < len(atom_names) else "?"
            resname = str(resnames[idx]) if idx < len(resnames) else "MOL"
            resid = int(resids[idx]) if idx < len(resids) else idx
            chain = str(chainids[idx]).strip() if idx < len(chainids) else "A"
            chain = chain if chain else "A"

            selection = f":{chain} and {resid}"
            label = f"{resname}{resid}:{atom_name}"
            return {"selection": selection, "label": label}
        except Exception:
            return None

    def _downsample_grid(self, grid: np.ndarray, max_voxels: int) -> tuple[np.ndarray, int]:
        voxels = int(np.prod(grid.shape))
        if voxels <= max_voxels:
            return grid, 1
        ratio = voxels / float(max_voxels)
        step = max(1, int(math.ceil(ratio ** (1.0 / 3.0))))
        return grid[::step, ::step, ::step], step

    def _density_cache_identity(self) -> tuple | None:
        if self.grid_data is None:
            return None
        shape = tuple(int(x) for x in self.grid_data.shape)
        origin_key = tuple(float(v) for v in np.asarray(self.grid_origin, dtype=float))
        return (
            id(self.grid_data),
            shape,
            float(self.grid_spacing),
            origin_key,
        )

    def _write_density_file_if_needed(self, force: bool = False) -> None:
        if self.grid_data is None:
            return
        key = self._density_cache_identity()
        if not force and key is not None and key == self._grid_cache_key and self._density_file is not None:
            return
        work_grid = self.grid_data
        step = 1
        spacing = float(self.grid_spacing)
        self._density_file_version += 1
        path = self._work_dir / f"volume_{self._density_file_version:04d}.dx"
        self._write_dx(path, work_grid, spacing, np.asarray(self.grid_origin, dtype=float))
        self._density_file = path
        self._grid_cache_key = key
        self._set_stats(
            f"grid {work_grid.shape[0]}x{work_grid.shape[1]}x{work_grid.shape[2]} | step {step}"
        )
        try:
            file_bytes = int(path.stat().st_size)
        except Exception:
            file_bytes = -1
        logger.warning(
            "Density3D file written: path=%s shape=%sx%sx%s spacing=%.6f step=%d dtype=float32 bytes=%d",
            path,
            int(work_grid.shape[0]),
            int(work_grid.shape[1]),
            int(work_grid.shape[2]),
            spacing,
            step,
            file_bytes,
        )

    def _write_dx(self, path: Path, grid: np.ndarray, spacing: float, origin: np.ndarray) -> None:
        nx, ny, nz = (int(x) for x in grid.shape)
        flat = np.asarray(grid, dtype=np.float32).ravel(order="C")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
            handle.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
            handle.write(f"delta {spacing:.6f} 0.0 0.0\n")
            handle.write(f"delta 0.0 {spacing:.6f} 0.0\n")
            handle.write(f"delta 0.0 0.0 {spacing:.6f}\n")
            handle.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
            handle.write(f"object 3 class array type float rank 0 items {len(flat)} data follows\n")
            for idx in range(0, len(flat), 3):
                chunk = flat[idx : idx + 3]
                handle.write(" ".join(f"{float(val):.7e}" for val in chunk) + "\n")
            handle.write('attribute "dep" string "positions"\n')
            handle.write('object "density" class field\n')
            handle.write('component "positions" value 1\n')
            handle.write('component "connections" value 2\n')
            handle.write('component "data" value 3\n')

    def _iso_color_hex(self, secondary: bool = False) -> str:
        if secondary:
            return self._density_color_2
        if self.colormap is None:
            return self._density_color_1
        min_v, max_v = self.data_range
        norm = 0.5 if max_v <= min_v else (self.current_iso_level - min_v) / (max_v - min_v)
        try:
            rgba = np.asarray(self.colormap.map(np.clip(norm, 0.0, 1.0), mode="float")).reshape(-1)
            if rgba.size >= 3:
                r = int(np.clip(rgba[0] * 255, 0, 255))
                g = int(np.clip(rgba[1] * 255, 0, 255))
                b = int(np.clip(rgba[2] * 255, 0, 255))
                return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            pass
        return self._density_color_1

    def _density_payload(self) -> dict:
        profile = self._quality_profile()
        style = self.density_style_combo.currentText().strip() or "Translucent"
        iso_level = self._clamp_iso_level(self.current_iso_level)
        iso_level_2 = self._clamp_iso_level(self.secondary_iso_level)
        diag = self._density_diag or {}

        primary_opacity = 0.35
        secondary_opacity = 0.22
        if style == "Solid":
            primary_opacity = 0.95
            secondary_opacity = 0.65
        elif style == "Wireframe":
            primary_opacity = 1.0
            secondary_opacity = 1.0

        return {
            "isolevel": float(iso_level),
            "visible": bool(self.show_density_check.isChecked()),
            "quality": profile.ngl_quality,
            "color": self._iso_color_hex(secondary=False),
            "opacity": float(primary_opacity),
            "style": style,
            "dualIso": bool(self.dual_iso_check.isChecked()),
            "isolevel2": float(iso_level_2),
            "color2": self._iso_color_hex(secondary=True),
            "opacity2": float(secondary_opacity),
            "dataMin": float(diag.get("min", self.data_range[0])),
            "dataMax": float(diag.get("max", self.data_range[1])),
            "dataMean": float(diag.get("mean", 0.0)),
            "dataStd": float(diag.get("std", 0.0)),
            "suggestedIsolevel": float(diag.get("suggested_iso", iso_level)),
            "bboxMin": [float(v) for v in diag.get("bbox_min", (0.0, 0.0, 0.0))],
            "bboxMax": [float(v) for v in diag.get("bbox_max", (0.0, 0.0, 0.0))],
        }

    def _load_density_component(self, force_reload: bool = False) -> None:
        if self._density_file is None:
            return
        if not self._density_file.exists():
            self._set_notice(f"Density file missing: {self._density_file}", warning=True)
            return
        payload = self._density_payload()
        payload["autoView"] = bool(self._density_auto_view_pending)
        if self._max_density_point is not None:
            payload["maxPoint"] = {
                "x": float(self._max_density_point[0]),
                "y": float(self._max_density_point[1]),
                "z": float(self._max_density_point[2]),
            }
        logger.warning(
            "Density3D load request: file=%s visible=%s isolevel=%.6f isolevel2=%.6f autoView=%s",
            self._density_file,
            bool(payload.get("visible", True)),
            float(payload.get("isolevel", 0.0)),
            float(payload.get("isolevel2", 0.0)),
            bool(payload.get("autoView", False)),
        )
        if force_reload:
            url = QtCore.QUrl.fromLocalFile(str(self._density_file)).toString()
            self._js_invoke("loadDensity", url, payload)
            self._density_auto_view_pending = False
        else:
            self._js_invoke("updateDensity", payload)

    def update_isosurface(self) -> None:
        if self.grid_data is None:
            return
        self._load_density_component(force_reload=False)

    def set_structure(self, atoms) -> None:
        self.sig_update_structure.emit(atoms)

    def _update_structure_slot(self, atoms) -> None:
        self.structure_atoms = atoms
        self.update_structure()
        self._update_density_insight()
        self._log_density_structure_alignment()

    def _structure_identity(self, atoms) -> tuple | None:
        if atoms is None:
            return None
        try:
            n = int(getattr(atoms, "n_atoms", 0))
            if n <= 0:
                return None
            return (id(atoms), n)
        except Exception:
            return None

    def _export_structure_pdb(self, atoms, path: Path, mode: str = "mda") -> bool:
        mode_key = str(mode or "mda").strip().lower()

        def _write_mda() -> bool:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Found no information for attr: '.*'",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Found chainIDs with invalid length.*",
                        category=UserWarning,
                    )
                    atoms.write(str(path))
                return True
            except Exception as exc:
                logger.warning("MDAnalysis PDB export failed: %s", exc)
                return False

        def _write_manual() -> bool:
            try:
                pos = np.asarray(atoms.positions, dtype=float)
                names = np.asarray(getattr(atoms, "names", np.array(["C"] * len(pos))), dtype=object)
                resids = np.asarray(getattr(atoms, "resids", np.arange(len(pos))), dtype=int)
                resnames = np.asarray(getattr(atoms, "resnames", np.array(["MOL"] * len(pos))), dtype=object)
                chainids = np.asarray(getattr(atoms, "chainIDs", np.array(["A"] * len(pos))), dtype=object)
                with path.open("w", encoding="utf-8") as handle:
                    for idx, xyz in enumerate(pos, start=1):
                        atom_name = str(names[idx - 1]).strip() or "C"
                        element = atom_name[0].upper()
                        name = atom_name[:4]
                        if len(name) <= 3:
                            name = f" {name:>3}"
                        else:
                            name = name[:4]
                        resn = (str(resnames[idx - 1]).strip() or "MOL")[:3].rjust(3)
                        chain_raw = str(chainids[idx - 1]).strip()
                        chain = chain_raw[0] if chain_raw else "A"
                        resid = int(resids[idx - 1]) if idx - 1 < len(resids) else idx
                        x, y, z = (float(v) for v in xyz)
                        handle.write(
                            f"ATOM  {idx:5d} {name} {resn} {chain}{resid:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}\n"
                        )
                    handle.write("END\n")
                return True
            except Exception as exc:
                logger.warning("Manual PDB export failed: %s", exc)
                return False

        if mode_key == "manual":
            return _write_manual() or _write_mda()
        return _write_mda() or _write_manual()

    def _reload_structure_component(self) -> bool:
        atoms = self.structure_atoms
        if atoms is None or getattr(atoms, "n_atoms", 0) == 0:
            return False
        path = self._work_dir / "model.pdb"
        ok = self._export_structure_pdb(atoms, path, mode=self._structure_export_mode)
        if not ok:
            return False
        self._structure_file = path
        url = QtCore.QUrl.fromLocalFile(str(path)).toString()
        self._js_invoke("loadStructure", url)
        return True

    def update_structure(self) -> None:
        atoms = self.structure_atoms
        if atoms is None or getattr(atoms, "n_atoms", 0) == 0:
            self.structure_info_label.setText("Structure: none")
            self._js_invoke("applyStructureLayers", [])
            return

        identity = self._structure_identity(atoms)
        if identity != self._structure_cache_key or self._structure_file is None:
            self._structure_export_mode = "manual"
            self._structure_parse_retry = False
            ok = self._reload_structure_component()
            if not ok:
                self.structure_info_label.setText("Structure: export failed")
                self._set_notice("Failed to export structure for NGL viewer.", warning=True)
                return
            self._structure_cache_key = identity

        shown = int(getattr(atoms, "n_atoms", 0))
        self.structure_info_label.setText(f"Structure: {shown:,} atoms loaded")
        self._apply_structure_representation()

    def _queue_representation_update(self) -> None:
        self._rep_update_timer.start()

    def _add_representation_row(self, layer: dict | None = None) -> None:
        layer = dict(layer or {})
        self._rep_layer_counter += 1
        row_id = f"layer-{self._rep_layer_counter}"

        row = QtWidgets.QFrame()
        row.setStyleSheet(
            "QFrame { border: 1px solid rgba(148,163,184,0.25); border-radius: 6px; "
            "background: rgba(248,250,252,0.65); }"
        )
        outer = QtWidgets.QVBoxLayout(row)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        vis = QtWidgets.QCheckBox("Visible")
        vis.setChecked(bool(layer.get("visible", True)))
        vis.setToolTip("Toggle layer visibility")
        top.addWidget(vis, 0)

        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(
            ["Cartoon", "Surface", "Ball+Stick", "Licorice", "Ribbon", "Spacefill", "HyperBalls"]
        )
        type_combo.setCurrentText(str(layer.get("type", "Ball+Stick")))
        type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        top.addWidget(type_combo, 1)

        delete_btn = QtWidgets.QToolButton()
        delete_btn.setText("Delete")
        delete_btn.setToolTip("Delete this representation layer")
        top.addWidget(delete_btn, 0)
        outer.addLayout(top)

        select_row = QtWidgets.QHBoxLayout()
        select_row.setSpacing(6)
        select_label = QtWidgets.QLabel("Selection")
        select_label.setMinimumWidth(64)
        select_row.addWidget(select_label, 0)
        sele = QtWidgets.QLineEdit(str(layer.get("selection", "all")))
        sele.setPlaceholderText("protein, ligand, water, chain A, etc.")
        select_row.addWidget(sele, 1)
        outer.addLayout(select_row)

        style_row = QtWidgets.QHBoxLayout()
        style_row.setSpacing(6)
        style_row.addWidget(QtWidgets.QLabel("Color"), 0)
        color_combo = QtWidgets.QComboBox()
        color_combo.addItems(["element", "chainid", "residueindex", "bfactor", "sstruc", "custom"])
        color_combo.setCurrentText(str(layer.get("colorScheme", "element")))
        color_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        style_row.addWidget(color_combo, 1)
        outer.addLayout(style_row)

        extra_row = QtWidgets.QHBoxLayout()
        extra_row.setSpacing(6)
        color_btn = QtWidgets.QPushButton("Pick color")
        color_btn.setProperty("colorValue", str(layer.get("colorValue", "#55aaff")))
        extra_row.addWidget(color_btn, 0)

        opacity_label = QtWidgets.QLabel("Opacity")
        extra_row.addWidget(opacity_label, 0)
        opacity = QtWidgets.QDoubleSpinBox()
        opacity.setRange(0.05, 1.0)
        opacity.setDecimals(2)
        opacity.setSingleStep(0.05)
        opacity.setValue(float(np.clip(float(layer.get("opacity", 1.0)), 0.05, 1.0)))
        opacity.setMinimumWidth(84)
        extra_row.addWidget(opacity, 0)
        extra_row.addStretch(1)
        outer.addLayout(extra_row)

        row_data = {
            "id": row_id,
            "widget": row,
            "visible": vis,
            "type_combo": type_combo,
            "selection": sele,
            "color_combo": color_combo,
            "color_btn": color_btn,
            "opacity": opacity,
        }
        self._rep_rows.append(row_data)

        vis.toggled.connect(self._queue_representation_update)
        type_combo.currentTextChanged.connect(self._queue_representation_update)
        sele.textChanged.connect(self._queue_representation_update)
        color_combo.currentTextChanged.connect(self._queue_representation_update)
        opacity.valueChanged.connect(self._queue_representation_update)
        delete_btn.clicked.connect(lambda: self._remove_representation_row(row_data))
        color_btn.clicked.connect(lambda: self._pick_layer_custom_color(row_data))

        stretch_item = self.rep_rows_layout.takeAt(self.rep_rows_layout.count() - 1)
        self.rep_rows_layout.addWidget(row)
        if stretch_item is not None:
            self.rep_rows_layout.addItem(stretch_item)
        else:
            self.rep_rows_layout.addStretch(1)

        self._queue_representation_update()

    def _remove_representation_row(self, row_data: dict) -> None:
        if row_data not in self._rep_rows:
            return
        self._rep_rows.remove(row_data)
        widget = row_data.get("widget")
        if widget is not None:
            widget.deleteLater()
        self._queue_representation_update()

    def _pick_layer_custom_color(self, row_data: dict) -> None:
        button = row_data.get("color_btn")
        if button is None:
            return
        current = str(button.property("colorValue") or "#55aaff")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, "Layer color")
        if not color.isValid():
            return
        button.setProperty("colorValue", color.name())
        self._queue_representation_update()

    def _collect_rep_layers(self) -> list[dict]:
        layers = []
        for row in self._rep_rows:
            color_btn = row.get("color_btn")
            color_value = "#55aaff"
            if color_btn is not None:
                color_value = str(color_btn.property("colorValue") or "#55aaff")
            layers.append(
                {
                    "id": row["id"],
                    "visible": bool(row["visible"].isChecked()),
                    "type": row["type_combo"].currentText(),
                    "selection": row["selection"].text().strip() or "all",
                    "colorScheme": row["color_combo"].currentText(),
                    "colorValue": color_value,
                    "opacity": float(row["opacity"].value()),
                }
            )
        return layers

    def _apply_structure_representation(self) -> None:
        layers = self._collect_rep_layers()
        radius_scale = 0.55
        payload_opts = {
            "quality": self._quality_profile().ngl_quality,
            "radiusScale": float(max(0.08, radius_scale)),
        }
        self._js_invoke("applyStructureLayers", layers, payload_opts)

    def _refresh_all_components(self) -> None:
        self._queue_stage_update()
        if self._structure_file is not None:
            url = QtCore.QUrl.fromLocalFile(str(self._structure_file)).toString()
            self._js_invoke("loadStructure", url)
            self._apply_structure_representation()
        if self._density_file is not None:
            self._load_density_component(force_reload=True)
        self._update_density_insight()
