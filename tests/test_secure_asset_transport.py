import socket
import urllib.request

import pytest
from scripts import fetch_real_samples as fetch

URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/example.tif"


def dns(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw:
                        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])


def test_fake_dns_requires_explicit_loopback_proxy(monkeypatch):
    dns(monkeypatch, "198.18.0.10")
    with pytest.raises(fetch.AssetRejected):
        fetch._assert_public_https_url(URL)
    assert fetch._assert_public_https_url(URL, local_proxy="http://127.0.0.1:61686") == URL
    with pytest.raises(fetch.AssetRejected):
        fetch._assert_public_https_url(URL, local_proxy="http://example.com:8080")


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254"])
def test_proxy_does_not_allow_internal_assets(monkeypatch, address):
    dns(monkeypatch, address)
    with pytest.raises(fetch.AssetRejected):
        fetch._assert_public_https_url(URL, local_proxy="http://127.0.0.1:61686")


def test_no_proxy_disables_fake_dns_exception(monkeypatch):
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: True)
    assert fetch._asset_local_proxy(URL, {"https": "http://127.0.0.1:61686"}) is None


@pytest.mark.skipif(fetch.os.name != "nt", reason="Windows proxy registry")
def test_gui_no_proxy_does_not_hide_system_proxy(monkeypatch):
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"no": "127.0.0.1,localhost"})
    monkeypatch.setattr(urllib.request, "getproxies_registry", lambda: {"https": "http://127.0.0.1:8686"})
    assert fetch._configured_proxies() == {"https": "http://127.0.0.1:8686", "no": "127.0.0.1,localhost"}


def test_verified_opener_rejects_redirect_to_untrusted_host(monkeypatch):
    dns(monkeypatch, "8.8.8.8")
    monkeypatch.setattr(fetch, "_DEADLINE", None)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    class Opener:
        def __init__(self, handlers):
            self.redirect = next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
        def open(self, request, timeout):
            return self.redirect.redirect_request(request, None, 302, "Found", {}, "https://untrusted.example/x")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener(handlers))
    with pytest.raises(fetch.AssetRejected):
        fetch._open_verified_request(urllib.request.Request(URL))


def test_python_transport_uses_virtual_filename(monkeypatch):
    import rasterio
    dns(monkeypatch, "8.8.8.8")
    monkeypatch.setenv("FLOOD_HTTP_TRANSPORT", "python")
    def open_mock(name, opener):
        assert name.startswith("asset_") and "https" not in name
        with pytest.raises(FileNotFoundError):
            opener(name + ".aux.xml")
        return "opened"
    monkeypatch.setattr(rasterio, "open", open_mock)
    assert fetch._open_vsicurl(URL) == "opened"


def test_clipped_read_preserves_pixels_without_boundless_vrt(monkeypatch, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "grid.tif"
    values = np.arange(400, dtype=np.float32).reshape(20, 20)
    transform = from_origin(0, 20, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1,
                       dtype="float32", crs="EPSG:32650", transform=transform) as ds:
        ds.write(values, 1)
    class CheckedDataset:
        def __enter__(self):
            self.ds = rasterio.open(path)
            return self
        def __exit__(self, *args):
            self.ds.close()
        def __getattr__(self, name):
            return getattr(self.ds, name)
        def read(self, *args, **kwargs):
            assert kwargs["boundless"] is False
            return self.ds.read(*args, **kwargs)
    monkeypatch.setattr(fetch, "_open_vsicurl", lambda href: CheckedDataset())
    monkeypatch.setattr(fetch, "_DEADLINE", None)
    out, geo = fetch.read_window(URL, "EPSG:32650", (10, 10, 14, 14), out_size=4)
    np.testing.assert_array_equal(out, values[6:10, 10:14])
    assert geo == from_origin(10, 14, 1, 1)
