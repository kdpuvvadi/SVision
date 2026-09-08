# SVision

Windows IPC inspection software for OK / NG decisions from sample images, plus OCR and code reading.

The UI is a web page served by the IPC. Open it from this PC or from another PC on the same network to configure recipes and run checks. Images are uploaded for now. A GigE / GenICam camera can be added later without changing the inspection flow.

## What it does

1. Create a scene (recipe).
2. Add OK samples and NG samples.
3. Train a model from those images.
4. Upload one or more images to inspect.
5. Optional tools: OCR and QR / barcode read.
6. Overall judgment is OK only if every enabled tool passes. Uncertain ML results fail as NG.

Classification is a compact image-feature model trained on your samples. It is built for a CPU IPC: training is usually well under a second, and the ML step is a few milliseconds. Several images and tools run in parallel.

## Install

Requires Python 3.10+.

```bat
install.bat
run.bat
```

Then open:

- This PC: `http://127.0.0.1:8080`
- Another PC: the Network URL printed in the console

Allow port 8080 in Windows Firewall if the other PC cannot connect.

Optional, for text and 1D codes:

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for OCR
- `pip install pyzbar` plus the ZBar DLL for 1D barcode and DataMatrix

QR reading works with the base install.

## Processing

- Tool steps for one image (ML, OCR, code) run on separate threads.
- A batch of uploaded images is inspected on a worker pool (`SVISION_THREADS`, default CPU count, max 16).
- The server binds `0.0.0.0` so another computer can configure it.

Environment:

- `SVISION_PORT` default `8080`
- `SVISION_HOST` default `0.0.0.0`
- `SVISION_DATA` recipe and sample folder
- `SVISION_THREADS` worker count
