# next

Neuer, sauberer `chew`-Pfad.

Ziel:
- `llama.cpp` als Referenz fuer Verhalten, Schichten und Performance-Denke
- Rust
- genau eine Binary
- CUDA-Kernels als Quellen im Binary
- NVRTC kompiliert beim Start
- keine zweite Build-Hoelle mit vorab kompilierten CUDA-Dateien

Schichten:
- `chew-cuda`
  NVRTC, Modul-Cache, Streams, Events, Graphs, CUDA-Backend-Helfer
- `chew-runtime`
  `n_ctx`, `n_batch`, `n_ubatch`, KV-Cache, Request-/Decode-Scheduler
- `chew-model-llama`
  erste Referenzimplementierung, nahe an `llama.cpp`
- `chew-model-gemma4`
  Gemma-4-Semantik sauber getrennt vom Runtime-Kern
- `chew`
  die eine Server-/CLI-Binary

Regeln:
- keine Modell-Sonderfaelle im Runtime-Kern
- keine Server-Logik im CUDA-Backend
- keine CUDA-Details in den Modellmodulen
- lieber wenige, grosse, lesbare Module als Dateisalat

Aktueller Stand:
- Workspace baut offline ohne crates.io-Abhaengigkeiten
- `chew-cuda` hat eine zentrale Registry fuer eingebettete NVRTC-Module
- derselbe CUDA-Pfad hat einen kleinen Compile-Cache fuer NVRTC-Optionen und Prewarm
- `chew-cuda` hat jetzt echte eingebettete `kv_cache`-Kernelquellen und vorbereitete KV-Launch-Bundles
- `chew-runtime` validiert `n_ctx >= n_batch >= n_ubatch` und `n_batch >= n_seq_max`
- Prefill-Planung ist als explizite Batch-/UBatch-Struktur vorhanden
- Session-Slots und ein kleiner Round-Robin-Decode-Scheduler sind modellfrei vorhanden
- KV-/Attention-Semantik ist als generische Runtime-Struktur vorhanden
- Session-KV-Writes werden geplant und gegen stale Pläne committed
- Whole-Batch-KV-Writes laufen jetzt als echter Runtime-Pfad: Batch-Plan -> CUDA-Launch-Bundle -> Layer-Inputs -> Bundle-Ausfuehrung
- `llama` und `gemma4` beschreiben ihre Layer-/KV-Layouts getrennt voneinander
- `chew` bootet als eigenstaendiges Binary gegen den neuen Workspace

Naechste Schritte:
- echte Device-Buffer-/Kernel-Ausfuehrung statt Host-Emulation des KV-Schreibpfads
- `kq_mask`/Attention-Inputs auf denselben Runtime-Vertrag haengen
- zuerst `llama`, danach `gemma4` auf denselben Kern setzen

Dieser Workspace ist absichtlich parallel zum Altbestand angelegt.
Der bestehende Baum bleibt zunaechst Referenz und Fallback, nicht Zielstruktur.
