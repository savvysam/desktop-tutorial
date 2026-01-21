import CoreImage
import SwiftUI
import Vision

@MainActor
final class ColorAnalyzer: ObservableObject {
    @Published var result: AnalysisResult?
    @Published var isLoading = false

    func analyze(image: UIImage?) {
        guard let image else { return }
        isLoading = true

        Task.detached(priority: .userInitiated) {
            let faceImage = self.cropFace(from: image) ?? image
            let predictions = self.mockPredictions(for: faceImage)

            let sorted = predictions.sorted { $0.confidence > $1.confidence }
            let primary = sorted.first!
            let secondary = self.secondaryCandidate(from: sorted)
            let result = AnalysisResult(predictions: sorted, primary: primary, secondary: secondary)

            await MainActor.run {
                self.result = result
                self.isLoading = false
            }
        }
    }

    private func cropFace(from image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: image.cgImageOrientation, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let observation = request.results?.first as? VNFaceObservation else { return nil }
        let boundingBox = observation.boundingBox
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)
        let rect = CGRect(
            x: boundingBox.minX * width,
            y: (1 - boundingBox.maxY) * height,
            width: boundingBox.width * width,
            height: boundingBox.height * height
        )

        guard let cropped = cgImage.cropping(to: rect) else { return nil }
        return UIImage(cgImage: cropped, scale: image.scale, orientation: image.imageOrientation)
    }

    private func mockPredictions(for image: UIImage) -> [SeasonPrediction] {
        let base = Double.random(in: 0.55...0.95)
        let shuffled = SeasonPalette.all.shuffled()
        return shuffled.enumerated().map { index, palette in
            let confidence = max(0.02, base - Double(index) * 0.04)
            return SeasonPrediction(palette: palette, confidence: confidence)
        }
    }

    private func secondaryCandidate(from sorted: [SeasonPrediction]) -> SeasonPrediction? {
        guard sorted.count > 1 else { return nil }
        let primary = sorted[0]
        let secondary = sorted[1]
        if primary.confidence - secondary.confidence < 0.08 {
            return secondary
        }
        return nil
    }
}

private extension UIImage {
    var cgImageOrientation: CGImagePropertyOrientation {
        switch imageOrientation {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }
}
