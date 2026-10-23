# Parallelitäts- und Integrationsvertrag für die Berichtsanonymisierung

Diese Leitlinie definiert die Bibliotheksgrenze von `lx-anonymizer` für stabile
parallele PDF-Verarbeitung. Sie ergänzt
`endoreg-db/docs/report_import_concurrency_implementation.md`.

Fortschritt und Produktionsfreigabe des integrierten Berichtsworkflows werden
im Repository `endoreg-db` ausschließlich über
`feature-tracking/Reporting.yml` bewertet. Dieses Dokument ist keine
eigenständige Roadmap und enthält keinen Fertigstellungsstatus.

## Verantwortungsgrenze

`endoreg-db` besitzt:

- Annahme und Validierung der Importquelle;
- verschlüsselten sensiblen Snapshot;
- Pfad- und Inhaltssperren, Lease und Fencing;
- Datenbanktransaktionen und Deduplizierung;
- kanonische Speicherpfade, atomare Publikation und Cleanup;
- Zugriffskontrolle, Audit und betriebliche Readiness.

`lx-anonymizer` besitzt:

- PDF-Textgewinnung und optische Zeichenerkennung;
- Erkennung sensitiver Metadaten;
- Anonymisierungs- und Redaktionslogik;
- Erzeugung eines noch nicht kanonisch veröffentlichten Ergebnis-PDF;
- fachliche Provenienz, Warnungen und Qualitätsinformationen.

`lx-anonymizer` darf weder Importquellen beanspruchen noch
`RawPdfFile`-Datensätze anlegen, globale kanonische Ausgabepfade wählen oder
eine verlorene Lease eigenständig zurückgewinnen.

## Eingangsvertrag

Die langfristige API ersetzt lose optionale Parameter durch ein strikt
validiertes, versioniertes Modell. Gemeinsame Modelle sollen in `lx_dtypes`
liegen.

```python
class ReportAnonymizationRequest(BaseModel):
    contract_version: Literal["report_anonymization_v2"]
    attempt_id: UUID
    source_path: Path
    source_sha256: str
    source_size_bytes: PositiveInt
    output_directory: Path
    create_anonymized_pdf: Literal[True]
    deadline_monotonic_ns: int | None
    options: ReportAnonymizationOptions
```

Beim Eintritt validiert `ReportReader` einmal:

- Vertragsversion und alle Optionen;
- Quelle ist eine reguläre, lesbare Datei;
- Quelle und Ausgabeverzeichnis sind verschieden;
- das Ausgabeverzeichnis existiert, gehört ausschließlich zum Versuch und ist
  beschreibbar;
- Quellgröße und Quellhash entsprechen dem vom Host übergebenen Snapshot;
- kein Zielpfad verlässt das übergebene Ausgabeverzeichnis.

Eine Abweichung ist ein typisierter Vertragsfehler. Die Verarbeitung darf den
vom Host zugesicherten Hash nicht still neu interpretieren.

## Ergebnisvertrag

```python
class ReportAnonymizationResult(BaseModel):
    contract_version: Literal["report_anonymization_v2"]
    attempt_id: UUID
    source_sha256: str
    original_text: str
    anonymized_text: str
    extracted_metadata: SensitiveMeta
    artifact_path: Path
    artifact_sha256: str
    artifact_size_bytes: PositiveInt
    provenance: ReportAnonymizationProvenance
    warnings: tuple[ReportAnonymizationWarning, ...]
```

Das Ergebnisobjekt ist unveränderlich. Alle Hashes sind kleingeschriebene
SHA-256-Hexwerte mit exakt 64 Zeichen. `artifact_path` liegt direkt oder
aufgelöst innerhalb des Versuchsverzeichnisses. Ein erfolgreicher Rückgabewert
garantiert:

- die Ergebnisdatei ist geschlossen und lesbar;
- sie besitzt eine gültige PDF-Struktur;
- ihre Größe und ihr Hash entsprechen dem Ergebnisobjekt;
- die Quelldatei wurde nicht verändert;
- keine weitere Hintergrundarbeit schreibt in das Artefakt;
- Provenienz und Warnungen sind vollständig validiert.

Die Veröffentlichung als kanonisches Endoreg-Artefakt ist ausdrücklich keine
Garantie von `lx-anonymizer`.

## Ausgabe- und Dateisystemregeln

Jeder Aufruf erhält ein eindeutiges Versuchsverzeichnis. Innerhalb dieses
Verzeichnisses:

1. wird in eine eindeutige temporäre Datei geschrieben;
2. werden PDF und Metadaten vollständig finalisiert;
3. wird die Datei geschlossen und synchronisiert;
4. wird sie atomar auf einen versuchslokalen Ergebnisnamen umbenannt;
5. wird erst danach das Ergebnisobjekt erzeugt.

Bestehende Dateien werden nicht überschrieben. Ein Konflikt ist ein
`ArtifactAlreadyExistsError`. Relative Pfade, symbolische Verknüpfungen und
Pfadauflösung außerhalb des Versuchsverzeichnisses werden abgelehnt.

Die Bibliothek entfernt bei Fehlern ausschließlich eigene unveröffentlichte
temporäre Dateien. Das Versuchsverzeichnis, die Quelle und bereits
zurückgegebene Artefakte gehören dem Host.

Persistente Debug-Ausgaben, Seitenbilder, OCR-Zwischendateien und
Modellantworten sind im normalen Profil verboten. Ein explizites
Diagnoseprofil darf sie nur im Versuchsverzeichnis ablegen und muss ihre
Sensitivität in der Provenienz kennzeichnen.

## Parallelitätsmodell

Ein `ReportReader`-Objekt ist einem Aufruf oder Worker zugeordnet und wird
nicht gleichzeitig aus mehreren Threads verwendet. Konfiguration wird beim
Aufruf in ein unveränderliches Modell normalisiert.

Verboten sind aufrufübergreifende veränderliche Modulzustände für:

- aktuelle Quell- oder Ausgabepfade;
- extrahierten Text und sensitive Metadaten;
- Crop-Regionen oder Seitennummern;
- aktive Provider- oder Request-Optionen;
- temporäre Dateinamen;
- Fehler- und Retry-Zähler.

Gemeinsame Caches dürfen nur unveränderliche, nicht patientenbezogene
Modellressourcen enthalten. Cache-Initialisierung ist threadsicher und
beeinflusst keine aufrufspezifischen Ergebnisse.

Rechenintensive reine Rust-Funktionen geben den Python Global Interpreter Lock
frei. Dateisystem-, Provider- und Modellaufrufe bleiben an klaren
Workflowgrenzen. Python-Threads sind kein Ersatz für eine begrenzte
Prozesswarteschlange.

## Ressourcen und Backpressure

- Der Host begrenzt die Zahl paralleler Berichte.
- `lx-anonymizer` startet keinen unbegrenzten Executor.
- Optische Zeichenerkennung, OpenMP-, BLAS-, ONNX- und
  Machine-Learning-Runtimes erhalten explizite Threadbudgets.
- Eine Aufruf-Deadline wird vor jeder teuren Phase geprüft.
- Unterprozesse werden in einer eigenen Prozessgruppe gestartet und bei
  Abbruch vollständig beendet und eingesammelt.
- Temporärer Speicherbedarf wird vor Seitenrendering geprüft; eine
  Überschreitung führt zu einem typisierten Ressourcenfehler.
- Kein Fallback darf unbemerkt eine größere Modell- oder Threadkonfiguration
  aktivieren.

## Determinismus und Retry

Bei identischer Quelle, identischer Konfiguration und identischen
Modellartefakten muss die Redaktionsentscheidung reproduzierbar sein. Wo ein
Provider keine vollständige Deterministik garantiert, enthält die Provenienz
mindestens Provider, Modell, Version, Parameter und Seed, und das Ergebnis wird
nicht als bitweise deterministisch bezeichnet.

Ein Retry:

- verwendet eine neue Versuchkennung und ein neues Verzeichnis;
- liest denselben validierten Snapshot oder bricht bei Hashabweichung ab;
- übernimmt keine veränderlichen Objekte des vorherigen Aufrufs;
- veröffentlicht nie selbst ein älteres Ergebnis;
- klassifiziert Fehler als retryfähig oder endgültig.

Empfohlene Fehlerhierarchie:

```text
ReportAnonymizationError
+-- ReportContractError
+-- SourceIdentityMismatchError
+-- UnsupportedDocumentError
+-- ResourceLimitError
+-- ProviderUnavailableError
+-- AnonymizationValidationError
+-- ArtifactAlreadyExistsError
+-- OperationCancelledError
```

Breite Ausnahmen dürfen nur an der äußeren Integrationsgrenze protokolliert
werden. Internal phases catch specific failures. At the public report boundary,
exceptions retain a bounded failure category but use content-free messages
and suppress external exception chains. Raw dependency error messages can
contain report text, filenames or provider payloads and must not reach worker
tracebacks.

## Native Funktionen in `lx-anonymizer`

Das Modul `lx_anonymizer._lx_anonymizer_native` ist für reine, rechenintensive
Transformationen zuständig, etwa Textnormalisierung, Kandidatenbewertung oder
Bounding-Box-Operationen. Quellanspruch, Import-Snapshot und kanonische
Publikation bleiben in `endoreg-db`.

Für jede native Funktion gelten:

- typisierte, validierte Eingaben;
- keine versteckten Dateisystem- oder Netzwerkzugriffe;
- Freigabe des Python Global Interpreter Lock bei ausreichend großer Arbeit;
- identische fachliche Ergebnisse oder explizit dokumentierte Unterschiede
  zum Python-Pfad;
- deterministische Fehlerabbildung;
- Capability- und Implementierungsversion.

Eine zentrale Funktion wie `native_capabilities()` ermöglicht dem Host, den
installierten Wheel-Vertrag zu prüfen. Die `.pyi`-Datei wird aus dem Rust-Code
generiert und in Continuous Integration auf einen leeren Diff geprüft.

## Kanonische API

`ReportReader.process_report(request)` ist der einzige öffentliche,
produktionsfähige Verarbeitungsvertrag. Lose optionale Parameter, Tupelrückgaben
und Laufzeit-Fallbacks auf ältere Reader-Verträge sind nicht unterstützt. Eine
Bibliothek ohne den kanonischen Vertrag ist nicht bereit für den Berichtsimport
und muss beim Integrationscheck laut scheitern.

Die unterstützte Matrix muss mindestens enthalten:

| Host | Bibliothek | Vertrag | Erwartung |
| --- | --- | --- | --- |
| aktuelle `endoreg-db`-Version | älteste unterstützte `lx-anonymizer`-Version | kanonisch | Vertragstest erfolgreich |
| aktuelle `endoreg-db`-Version | aktuelle `lx-anonymizer`-Version | kanonisch | alle Capability- und Parallelitätstests erfolgreich |
| Produktionsprofil mit fehlender Pflichtfähigkeit | beliebig | unvollständig | Readiness schlägt fehl |

Konkrete Versionsnummern werden in der Release-Konfiguration gepflegt, nicht
als vermeintlicher Status in diesem Dokument.

## Testvorgaben

### Unit- und Paritätstests

- Request- und Resultvalidierung einschließlich unbekannter Vertragsversion;
- Pfadflucht, Symlink und bereits bestehendes Ziel;
- Hash- und Größenabweichung;
- native/Python-Parität für jede veröffentlichte Capability;
- Abbruch und Deadline zwischen allen teuren Phasen;
- Cleanup nur eigener temporärer Dateien;
- unveränderliche Provenienz und Warnungen.

### Parallelitätstests

- mindestens acht parallele Aufrufe mit unterschiedlichen Berichten;
- parallele Aufrufe mit demselben Snapshot, aber getrennten
  Versuchsverzeichnissen;
- keine Vermischung von Text, Metadaten, Crop-Regionen oder Ausgabepfaden;
- absichtlicher Prozessabbruch während Seitenrendering, Erkennung,
  Anonymisierung und PDF-Schreiben;
- begrenzte Thread- und Speichernutzung unter Last;
- Wiederholung nach Provider- und Ressourcenfehler.

### Wheel- und Integrationsprüfung

Continuous Integration muss:

1. `uv sync --extra dev` ausführen.
2. Pyright vor pytest ausführen.
3. Rust-Tests ausführen.
4. Wheel mit Maturin aus sauberem Checkout bauen.
5. Wheel in einer frischen Umgebung installieren.
6. Native Capability und einen kleinen Bericht aus dem installierten Wheel
   ausführen.
7. Stub-Parität und unveränderten Git-Status prüfen.
8. Den Cross-Repository-Vertrag gegen `endoreg-db` ausführen.

Ein Import direkt aus dem Arbeitsverzeichnis neben einem veralteten Shared
Object ist kein Paketnachweis.

## Beobachtbarkeit und Datenschutz

Die Bibliothek protokolliert strukturierte Phasenereignisse mit Versuchkennung,
Vertragsversion, Dauer, Seitenzahl, Backend, Modellversion und Ergebnisstatus.
Sie protokolliert niemals:

- Original- oder anonymisierten Berichtstext;
- Patientennamen oder andere extrahierte Identifikatoren;
- Rohseiten oder vollständige Dateipfade;
- Provider-Token, Schlüssel oder vollständige Prompts mit Patientendaten.

Der Host entscheidet über dauerhafte Audit-Speicherung. Bibliothekslogs dienen
nicht als Ersatz für Endoreg-Persistenz oder Fencing.

## Reihenfolge der Implementierung

1. Gemeinsame Request-, Result-, Provenienz- und Fehlerverträge typisieren.
2. `ReportReader`-Zustand pro Aufruf isolieren.
3. Versuchsverzeichnis und atomare lokale Ausgabe einführen.
4. Hash-, Größen- und PDF-Nachbedingungen ergänzen.
5. Ressourcenbudgets, Abbruch und Deadline implementieren.
6. Native Capability- und Stub-Prüfung ergänzen.
7. Altadapter und Cross-Version-Vertragstests bereitstellen.
8. Erst nach erfolgreicher End-to-End-Prüfung den Host auf den neuen Vertrag
   umstellen.

Jede Stufe muss den bisherigen öffentlichen Import erhalten oder über einen
getesteten Adapter bereitstellen. Sicherheits- oder Integritätsfehler dürfen
nicht durch einen stillen Legacy-Fallback verdeckt werden.

## Content-free diagnostic verification

Report, cropper, OCR and metadata diagnostics report phases, counts and bounded
failure categories. They must not include extracted text, matched identifiers,
metadata, provider endpoints, document filenames or raw exception payloads at
any log level. Removing DEBUG output or setting `verbose=False` is insufficient:
workers may capture stdout/stderr and render exception chains.

The public `ReportReader.process_report` boundary sanitizes failures, including
model cleanup failures, before a host can render them. The PDF anonymizer also
raises content-free artifact failures. Callers must use these public boundaries;
internal dependency exceptions are not a safe diagnostic interface.

Run `tests/test_report_log_privacy.py`, `tests/test_ocr_log_privacy.py`,
`tests/test_provider_log_privacy.py` and `tests/test_report_reader_contract.py`
after strict Pyright. These tests exercise synthetic sentinel content in
success, provider failure, redaction failure, captured streams and rendered
tracebacks. The reusable `scripts/verify_packaged_report_privacy.py` harness additionally
executes installed wheels in an actual local Celery memory-broker solo worker.
It verifies package/native origins, successful redaction, failed provider and
redaction tasks, sanitized tracebacks, redirected streams and immutable inputs.
Only external provider/OCR/detector boundaries are injected. This local candidate
check does not attest to a deployed worker revision.

The v2 shared report contract requires `extraction_evidence`: the actual
`configured_llm` or `spacy_regex` backend, findings/empty/fallback outcome,
validated populated metadata-field count, provider-attempt flag and bounded
provider-failure category. Readiness does not prove provider success. A selected
provider that fails may produce fallback evidence only after the deterministic
extractor actually completes and validates its result. A valid empty provider
response still runs a configured later local extractor; this fallback records
`provider_attempted=True` with no provider failure. A terminal valid empty
provider response remains distinct from a failed invocation. Empty text and
absence of evidence cannot become successful imports. Evidence is observational; Rust
in endoreg-db remains the single authority for durable lifecycle transitions.

The producer, shared lx-dtypes contract and endoreg-db consumer must be released
together. The v1 contract cannot be interpreted as v2 success. Local coordinated
tests use the sibling lx-data-models checkout explicitly; published dependency
resolution and packaged-worker output require release verification.

Candidate `lx-anonymizer==1.0.9.dev1` requires `lx-dtypes>=0.5.4.dev1`. These
versions identify the unreleased v2 boundary; older published versions are not
claimed compatible. The final candidate pair passed installed-package and
Celery verification using an isolated interpreter with a read-only dependency
baseline. Candidate packages and the native extension were installed from wheels,
not imported from sibling source checkouts. This is not a clean dependency
resolver test. Canonical lock regeneration remains blocked until the shared
prerelease is available from the configured index; retain existing lock evidence
and never fabricate registry metadata or commit verification-only artifact paths.

To repeat the local worker check, build both candidate wheels first and supply
those paths as `anonymizer_wheel` and `dtypes_wheel`. Create a fresh verification
venv with Python 3.12, install Celery and pytest into it, then install both wheels
with `uv pip install --python "$verify_root/venv/bin/python" --no-deps
"$dtypes_wheel" "$anonymizer_wheel"`. Supply compatible third-party dependencies
through a normal install, or explicitly reuse a read-only development dependency
directory: write its absolute `site-packages` path as the only line in a `.pth`
file in the verification venv. Do not use `site.addsitedir`, source checkout
paths, `PYTHONPATH`, or copied editable `.pth` files. A plain path entry does not
process that dependency directory's nested editable `.pth` files.

Run from the verification directory, outside all checkouts:

```bash
"$verify_root/venv/bin/python" -I \
  "$anonymizer_checkout/scripts/verify_packaged_report_privacy.py" \
  --site-packages "$verify_root/venv/lib/python3.12/site-packages"
```

The script rejects candidate package or native-extension origins outside the
specified installation directory, requires the v2 contract and emits only a
safe JSON result. Record the exact wheels' SHA256 digests and the third-party
dependency inventory alongside the result. Reused dependencies are a candidate
integration test, never evidence of a clean resolver installation or deployment.
