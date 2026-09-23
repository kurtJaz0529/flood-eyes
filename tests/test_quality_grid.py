"""Quality classes must remain aligned after reprojection, even at equal size."""
import numpy as np
from rasterio.transform import from_origin

from src.preprocess import Scene, reproject_scene_to


def test_scl_moves_with_shifted_spectral_grid():
    bands = {name: np.full((8, 8), 0.1, dtype=np.float32)
             for name in ("blue", "green", "red", "nir")}
    scl = np.full((8, 8), 4, dtype=np.uint8)
    scl[:, 2] = 9
    source = Scene(bands=bands, transform=from_origin(500010, 3000000, 10, 10),
                   crs="EPSG:32650", meta={"scl": scl})
    reference = Scene(bands=bands, transform=from_origin(500000, 3000000, 10, 10),
                      crs="EPSG:32650")
    aligned = reproject_scene_to(source, reference)
    assert np.all(aligned.meta["scl"][:, 3] == 9)
    assert np.all(aligned.meta["scl"][:, 2] == 4)
    assert np.all(aligned.meta["scl"][:, 0] == 0)
    assert np.all(aligned.nodata_mask[:, 0])


def test_dem_nodata_center_remains_unassessed(tmp_path):
    import rasterio
    from src.terrain import assess_dem

    transform = from_origin(500000, 3000000, 10, 10)
    dem = np.full((8, 8), 100.0, dtype=np.float32)
    dem[3, 3] = -9999
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=1,
                       dtype="float32", crs="EPSG:32650", transform=transform,
                       nodata=-9999) as dst:
        dst.write(dem, 1)
    scene = Scene(bands={"green": np.ones((8, 8), dtype=np.float32)},
                  transform=transform, crs="EPSG:32650")
    risk, summary = assess_dem(str(path), scene)
    assert risk[3, 3] == 255, "missing DEM must not be inferred flat from its neighbors"
    assert np.all(risk[risk != 255] == 0)
