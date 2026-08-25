"""
Precompute small circular player thumbnails as data URIs for the 3D scene.

WHY: the 3D archetype view (step2_intro) renders each player as a WebGL sprite
whose texture must be same-origin - WebGL rejects cross-origin images without
CORS headers, and the NBA CDN sends none. So the faces have to be embedded, not
linked. Full-res headshots would be ~10MB; downscaled to 72px circular WebP they
are ~0.8KB each, ~440KB for the whole league - small enough to ship in the page.

Also solves the deploy gap: only a handful of full headshots are committed, but
this artifact carries every current player's face, so the 3D view has faces on
Streamlit Cloud regardless of what's in data/headshot_cache/.

Run: python src/pipeline/precompute_player_thumbnails.py
Output: data/player_thumbs_2025_26.json  ({player_id: data-uri})
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from PIL import Image, ImageDraw  # noqa: E402

SEASON = "2025-26"
SIZE = 72
OUT = REPO / "data" / f"player_thumbs_{SEASON.replace('-', '_')}.json"
CACHE = REPO / "data" / "headshot_cache"


def _circular_thumb(png_path: Path) -> str | None:
    try:
        im = Image.open(png_path).convert("RGBA")
    except Exception:
        return None
    w, h = im.size
    # square crop from the top-center (headshots frame the face there)
    side = min(w, h)
    left = (w - side) // 2
    im = im.crop((left, 0, left + side, side)).resize((SIZE, SIZE), Image.LANCZOS)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, SIZE - 1, SIZE - 1), fill=255)
    out = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    buf = io.BytesIO()
    out.save(buf, "WEBP", quality=82)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    import warnings; warnings.filterwarnings("ignore")
    from portal_shared import load_static
    recipes, _k, _labels, _onc = load_static()
    pids = [int(p) for p in recipes["PLAYER_ID"].astype(int).unique()]

    thumbs, missing = {}, []
    for pid in pids:
        p = CACHE / f"{pid}.png"
        uri = _circular_thumb(p) if p.exists() else None
        if uri:
            thumbs[str(pid)] = uri
        else:
            missing.append(pid)

    OUT.write_text(json.dumps(thumbs))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} - {len(thumbs)}/{len(pids)} players, {kb:.0f} KB")
    if missing:
        print(f"  {len(missing)} without a cached headshot (will fall back to a dot): {missing[:10]}")


if __name__ == "__main__":
    main()
