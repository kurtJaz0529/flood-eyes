(function () {
  const root = element.querySelector("#heye-map") || element;
  if (root.getAttribute("data-heye-ready") === "1") return;
  root.setAttribute("data-heye-ready", "1");
  root.classList.add("heye-map");
  root.innerHTML =
    '<div class="heye-tiles"></div>' +
    '<div class="heye-hint">高德地图 · 拖动/滚轮缩放 · 点击选点</div>' +
    '<div class="heye-layers">' +
    '<button type="button" data-ly="amap" class="on">高德</button>' +
    '<button type="button" data-ly="sat">影像</button>' +
    "</div>" +
    '<div class="heye-pin"></div>';
  const tilesEl = root.querySelector(".heye-tiles");
  const hintEl = root.querySelector(".heye-hint");
  const pinEl = root.querySelector(".heye-pin");
  const PRESETS = window.__HEYE_PRESETS || [];

  function outChina(lat, lon) {
    return lon < 72.004 || lon > 137.8347 || lat < 0.8293 || lat > 55.8271;
  }
  function tfLat(x, y) {
    let r = -100 + 2 * x + 3 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x));
    r += ((20 * Math.sin(6 * x * Math.PI) + 20 * Math.sin(2 * x * Math.PI)) * 2) / 3;
    r += ((20 * Math.sin(y * Math.PI) + 40 * Math.sin((y / 3) * Math.PI)) * 2) / 3;
    r += ((160 * Math.sin((y / 12) * Math.PI) + 320 * Math.sin((y * Math.PI) / 30)) * 2) / 3;
    return r;
  }
  function tfLon(x, y) {
    let r = 300 + x + 2 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x));
    r += ((20 * Math.sin(6 * x * Math.PI) + 20 * Math.sin(2 * x * Math.PI)) * 2) / 3;
    r += ((20 * Math.sin(x * Math.PI) + 40 * Math.sin((x / 3) * Math.PI)) * 2) / 3;
    r += ((150 * Math.sin((x / 12) * Math.PI) + 300 * Math.sin((x * Math.PI) / 30)) * 2) / 3;
    return r;
  }
  function wgsToGcj(lat, lon) {
    if (outChina(lat, lon)) return [lat, lon];
    const a = 6378245.0, ee = 0.006693421622965943;
    let dLat = tfLat(lon - 105, lat - 35), dLon = tfLon(lon - 105, lat - 35);
    const rad = (lat / 180) * Math.PI;
    let magic = Math.sin(rad);
    magic = 1 - ee * magic * magic;
    const sqrtMagic = Math.sqrt(magic);
    dLat = (dLat * 180) / (((a * (1 - ee)) / (magic * sqrtMagic)) * Math.PI);
    dLon = (dLon * 180) / ((a / sqrtMagic) * Math.cos(rad) * Math.PI);
    return [lat + dLat, lon + dLon];
  }
  function gcjToWgs(lat, lon) {
    if (outChina(lat, lon)) return [lat, lon];
    let wLat = lat, wLon = lon;
    for (let i = 0; i < 6; i++) {
      const g = wgsToGcj(wLat, wLon);
      const dLat = lat - g[0], dLon = lon - g[1];
      wLat += dLat;
      wLon += dLon;
      if (Math.max(Math.abs(dLat), Math.abs(dLon)) < 1e-9) break;
    }
    return [wLat, wLon];
  }
  function lon2x(lon, z) {
    return ((lon + 180) / 360) * Math.pow(2, z);
  }
  function lat2y(lat, z) {
    const r = (lat * Math.PI) / 180;
    return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * Math.pow(2, z);
  }
  function x2lon(x, z) {
    return (x / Math.pow(2, z)) * 360 - 180;
  }
  function y2lat(y, z) {
    const n = Math.PI - (2 * Math.PI * y) / Math.pow(2, z);
    return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  }
  function tileUrl(ly, x, y, z) {
    const n = Math.pow(2, z);
    if (x < 0 || y < 0 || x >= n || y >= n) return "";
    const s = (x + y) % 4;
    if (ly === "sat") return "https://webst0" + (s + 1) + ".is.autonavi.com/appmaptile?style=6&x=" + x + "&y=" + y + "&z=" + z;
    return "https://webrd0" + (s + 1) + ".is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x=" + x + "&y=" + y + "&z=" + z;
  }

  const st = { lat: 29.15, lon: 116.3, z: 7, ly: "amap", slat: 29.15, slon: 116.3 };
  let dragging = false, lastX = 0, lastY = 0, moved = false;

  function emit(extra) {
    try {
      window._heyeLon = st.slon;
      window._heyeLat = st.slat;
      if (extra) {
        window._heyePreStart = extra.pre_start || window._heyePreStart;
        window._heyePreEnd = extra.pre_end || window._heyePreEnd;
        window._heyePostStart = extra.post_start || window._heyePostStart;
        window._heyePostEnd = extra.post_end || window._heyePostEnd;
      }
      hintEl.textContent = extra && extra.label
        ? ("已选 " + extra.label + "  " + st.slon.toFixed(4) + ", " + st.slat.toFixed(4) + "  时间范围已带出")
        : ("已选 WGS84 " + st.slon.toFixed(4) + ", " + st.slat.toFixed(4) + "  → 可直接开始分析");
    } catch (e) {}
  }

  function render() {
    const w = root.clientWidth || 640;
    const h = root.clientHeight || 420;
    const gcj = wgsToGcj(st.lat, st.lon);
    const cx = lon2x(gcj[1], st.z) * 256;
    const cy = lat2y(gcj[0], st.z) * 256;
    const left = cx - w / 2;
    const top = cy - h / 2;
    const x0 = Math.floor(left / 256);
    const y0 = Math.floor(top / 256);
    const x1 = Math.floor((left + w) / 256);
    const y1 = Math.floor((top + h) / 256);
    let html = "";
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const src = tileUrl(st.ly, x, y, st.z);
        if (!src) continue;
        html += '<img class="heye-tile" alt="" draggable="false" src="' + src + '" style="left:' + (x * 256 - left) + "px;top:" + (y * 256 - top) + 'px"/>';
      }
    }
    PRESETS.forEach(function (p) {
      const g = wgsToGcj(p.lat, p.lon);
      const px = lon2x(g[1], st.z) * 256 - left;
      const py = lat2y(g[0], st.z) * 256 - top;
      if (px < -8 || py < -8 || px > w + 8 || py > h + 8) return;
      html += '<div class="heye-dot" data-key="' + p.key + '" style="left:' + px + "px;top:" + py + 'px" title="' + String(p.label).replace(/"/g, "") + '"></div>';
    });
    tilesEl.innerHTML = html;
    const sg = wgsToGcj(st.slat, st.slon);
    pinEl.style.left = lon2x(sg[1], st.z) * 256 - left + "px";
    pinEl.style.top = lat2y(sg[0], st.z) * 256 - top + "px";
  }

  function pixToWgs(px, py) {
    const w = root.clientWidth || 640;
    const h = root.clientHeight || 420;
    const gcj = wgsToGcj(st.lat, st.lon);
    const cx = lon2x(gcj[1], st.z) * 256;
    const cy = lat2y(gcj[0], st.z) * 256;
    const wx = cx - w / 2 + px;
    const wy = cy - h / 2 + py;
    const glon = x2lon(wx / 256, st.z);
    const glat = y2lat(wy / 256, st.z);
    return gcjToWgs(glat, glon);
  }

  root.addEventListener("mousedown", function (e) {
    if (e.button !== 0) return;
    dragging = true;
    moved = false;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    lastX = e.clientX;
    lastY = e.clientY;
    const gcj = wgsToGcj(st.lat, st.lon);
    const x = lon2x(gcj[1], st.z) - dx / 256;
    const y = lat2y(gcj[0], st.z) - dy / 256;
    const w = gcjToWgs(y2lat(y, st.z), x2lon(x, st.z));
    st.lat = w[0];
    st.lon = w[1];
    render();
  });
  window.addEventListener("mouseup", function () {
    dragging = false;
  });
  root.addEventListener("click", function (e) {
    if (moved) return;
    const dot = e.target.closest(".heye-dot");
    if (dot) {
      const p = PRESETS.find(function (x) { return x.key === dot.getAttribute("data-key"); });
      if (p) {
        st.lat = p.lat; st.lon = p.lon; st.slat = p.lat; st.slon = p.lon;
        st.z = Math.max(st.z, 9);
        render();
        emit(p);
        return;
      }
    }
    const r = root.getBoundingClientRect();
    const wgs = pixToWgs(e.clientX - r.left, e.clientY - r.top);
    st.slat = wgs[0];
    st.slon = wgs[1];
    render();
    emit();
  });
  root.addEventListener("wheel", function (e) {
    e.preventDefault();
    const nz = Math.max(4, Math.min(18, st.z + (e.deltaY > 0 ? -1 : 1)));
    if (nz === st.z) return;
    const r = root.getBoundingClientRect();
    const before = pixToWgs(e.clientX - r.left, e.clientY - r.top);
    st.z = nz;
    const after = pixToWgs(e.clientX - r.left, e.clientY - r.top);
    st.lat += before[0] - after[0];
    st.lon += before[1] - after[1];
    render();
  }, { passive: false });
  root.querySelectorAll(".heye-layers button").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      st.ly = btn.getAttribute("data-ly");
      root.querySelectorAll(".heye-layers button").forEach(function (b) { b.classList.toggle("on", b === btn); });
      render();
    });
  });

  window._heyeFly = function (lat, lon) {
    if (!isFinite(lat) || !isFinite(lon)) return;
    st.lat = lat; st.lon = lon; st.slat = lat; st.slon = lon;
    render();
    emit();
  };
  emit();
  render();
  setTimeout(render, 200);
  setTimeout(render, 800);
  window.addEventListener("resize", render);
})();
