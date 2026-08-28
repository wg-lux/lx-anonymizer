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
    contract_version: Literal["report_anonymization"]
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
    contract_version: Literal["report_anonymization"]
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
werden. Interne Phasen fangen konkrete Fehler und erhalten die ursprüngliche
Ursache.

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
