import os

import pytest


@pytest.fixture(autouse=True)
def no_live_model_http(monkeypatch):
    import httpx

    def reject(self, request):
        pytest.fail("Live HTTP is disabled in tests; use httpx.MockTransport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    for name in os.environ:
        if name == "ATRAG_TEST_DATABASE_URL":
            continue
        if name.startswith(("ATRAG_", "OPENAI_")) or name == "DATABASE_URL":
            monkeypatch.delenv(name)


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / ".env").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def make_pdf(tmp_path):
    """Build tiny synthetic PDFs with text, vector tables and a raster image."""

    def make(contents, height=842):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Count {len(contents)} /Kids [".encode()
            + b" ".join(f"{5 + i * 2} 0 R".encode() for i in range(len(contents)))
            + b"] >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\nstream\n\x00\x00\x00\nendstream",
        ]
        for i, content in enumerate(contents):
            stream = content.encode("ascii")
            objects.extend(
                [
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 {height}] /Resources << /Font << /F1 3 0 R >> /XObject << /Im1 4 0 R >> >> /Contents {6 + i * 2} 0 R >>".encode(),
                    f"<< /Length {len(stream)} >>\nstream\n".encode()
                    + stream
                    + b"\nendstream",
                ]
            )
        document, offsets = bytearray(b"%PDF-1.4\n"), [0]
        for i, obj in enumerate(objects, 1):
            offsets.append(len(document))
            document.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
        xref = len(document)
        document.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]:
            document.extend(f"{offset:010} 00000 n \n".encode())
        document.extend(
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        path = tmp_path / "synthetic.pdf"
        path.write_bytes(document)
        return path

    return make


@pytest.fixture
def pdf_text():
    def text(top, value, x=85):
        return f"BT /F1 10 Tf {x} {842 - top - 10} Td <{value.encode('cp1252').hex()}> Tj ET\n"

    return text


@pytest.fixture
def prepared_document(tmp_path):
    from fixtures.prepared import prepare_document

    return prepare_document(tmp_path)
