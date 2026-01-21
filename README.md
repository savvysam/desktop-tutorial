# 16-Season Color Analysis (SwiftUI)

This repo contains a SwiftUI starter implementation for a 16-season personal color analysis iPhone app. The code provides a photo input flow, on-device Vision face detection, palette previews, and a results grid with sharing/saving actions.

## Highlights

- **Photo input** via camera or library using `UIImagePickerController` and `PhotosPicker`.
- **Face detection + cropping** using Vision (`VNDetectFaceRectanglesRequest`).
- **16-season palette grid** with overlays and swatches to visualize each season.
- **Primary + secondary recommendation** handling when confidence scores are close.
- **Share and save** actions via `ShareLink` and `UIImageWriteToSavedPhotosAlbum`.

## Permissions

Add these keys to your app’s `Info.plist`:

- `NSCameraUsageDescription` — required for camera access.
- `NSPhotoLibraryUsageDescription` — required for photo library selection.
- `NSPhotoLibraryAddUsageDescription` — required to save results.

## Model Integration

The current code ships with a **mock prediction pipeline** so the UI flow can be exercised without a trained Core ML model. Replace `mockPredictions` in `ColorAnalyzer` with a `VNCoreMLRequest` or direct model inference once your `.mlmodel` is ready.

## Suggested Project Structure

```
ColorSeasonApp/
├── ColorSeasonApp.swift
├── Models/
│   ├── ColorAnalyzer.swift
│   └── SeasonPalette.swift
└── Views/
    ├── ContentView.swift
    ├── ImagePicker.swift
    └── ResultsView.swift
```
