"""Count unique source bytes while retaining every loaded prepared version."""

from io import BytesIO
from zipfile import ZipFile

import pytest
from fixtures.faq_pipeline import make_faq_package
from fixtures.prepared import prepare_revision

from atrag.admin.service import AdminService
from atrag.corpus import publish_corpus
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package

pytestmark = pytest.mark.integration
pytest_plugins = ["test_admin"]


def test_documents_include_unpublished_versions_and_deduplicate_source_sizes(
    administration, engine, prepared_document, faq_snapshot, tmp_path
):
    admin, _, actor, reader = administration
    storage = tmp_path / "server"
    admin = AdminService(
        engine, admin.settings.model_copy(update={"data_dir": storage})
    )
    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "pdf-package")
    first = index_package(package, engine, storage)
    revision = prepare_revision(prepared_document, tmp_path / "revision")
    second = index_package(revision, engine, storage)
    html_package = make_faq_package(faq_snapshot, tmp_path)
    faq = index_package(html_package, engine, storage)
    publish_corpus(engine, {"pt-at-cirs": first.snapshot_id})
    documents = {row.source_id: row for row in admin.list_documents(actor)}
    assert set(documents) == {"pt-at-cirs", "pt-at-faq"}
    pdf = documents["pt-at-cirs"]
    assert len(pdf.versions) == 2
    assert pdf.size_bytes == pdf.versions[0].size_bytes
    assert pdf.published
    assert {v.snapshot_id for v in pdf.versions if v.published} == {first.snapshot_id}
    assert pdf.indexed_at == max(v.indexed_at for v in pdf.versions)
    assert all(v.available for v in pdf.versions)
    assert all(v.indexed_at > v.ingested_at for v in pdf.versions)
    raw_pdf = admin.download_source(actor, second.snapshot_id)
    assert raw_pdf.mime == "application/pdf"
    assert raw_pdf.data.startswith(b"%PDF-")
    assert len(raw_pdf.data) == pdf.size_bytes
    html = documents["pt-at-faq"]
    assert not html.published
    assert html.source_format == "html_collection"
    assert html.size_bytes == sum(
        page.size_bytes for page in faq_snapshot.collection.pages
    )
    download = admin.download_source(actor, faq.snapshot_id)
    assert download.mime == "application/zip"
    with ZipFile(BytesIO(download.data)) as archive:
        assert set(archive.namelist()) == {
            "collection.json",
            *(str(page.local_path) for page in faq_snapshot.collection.pages),
        }
        assert (
            sum(
                len(archive.read(str(page.local_path)))
                for page in faq_snapshot.collection.pages
            )
            == html.size_bytes
        )
    with pytest.raises(PermissionError):
        admin.download_source(reader, first.snapshot_id)
    with pytest.raises(PermissionError):
        admin.download_source(actor, "unknown")
    (storage / pdf.versions[0].local_path).unlink()
    refreshed = {row.source_id: row for row in admin.list_documents(actor)}
    assert not any(version.available for version in refreshed["pt-at-cirs"].versions)
    with pytest.raises(FileNotFoundError):
        admin.download_source(actor, first.snapshot_id)
