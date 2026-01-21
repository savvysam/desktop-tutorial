import SwiftUI

struct ResultsView: View {
    let baseImage: UIImage?
    let result: AnalysisResult

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 12), count: 4)

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                VStack(spacing: 8) {
                    Text("Best Match")
                        .font(.headline)
                    Text(bestMatchTitle)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .multilineTextAlignment(.center)
                }

                Text(result.explanation)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)

                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(result.predictions) { prediction in
                        PaletteResultView(
                            baseImage: baseImage,
                            palette: prediction.palette,
                            isHighlighted: isHighlighted(prediction)
                        )
                    }
                }

                actionButtons
            }
            .padding()
        }
        .navigationTitle("Your Palette")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var bestMatchTitle: String {
        if let secondary = result.secondary {
            return "\(result.primary.palette.name) / \(secondary.palette.name)"
        }
        return result.primary.palette.name
    }

    private func isHighlighted(_ prediction: SeasonPrediction) -> Bool {
        prediction.palette == result.primary.palette || prediction.palette == result.secondary?.palette
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            if let baseImage {
                ShareLink(item: baseImage, preview: SharePreview("My Season", image: Image(uiImage: baseImage))) {
                    Label("Share Results", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                Button {
                    UIImageWriteToSavedPhotosAlbum(baseImage, nil, nil, nil)
                } label: {
                    Label("Save Photo", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

private struct PaletteResultView: View {
    let baseImage: UIImage?
    let palette: SeasonPalette
    let isHighlighted: Bool

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                if let baseImage {
                    Image(uiImage: baseImage)
                        .resizable()
                        .scaledToFill()
                        .frame(height: 90)
                        .clipped()
                } else {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.1))
                        .frame(height: 90)
                }

                Rectangle()
                    .fill(palette.overlayColor)
                    .opacity(0.28)

                if isHighlighted {
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.accentColor, lineWidth: 4)
                        .padding(2)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))

            Text(palette.name)
                .font(.caption2)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)

            HStack(spacing: 4) {
                ForEach(palette.swatchColors, id: \.self) { color in
                    Circle()
                        .fill(color)
                        .frame(width: 10, height: 10)
                }
            }
        }
        .padding(6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.05))
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("User photo with \(palette.name) palette")
    }
}

#Preview {
    NavigationStack {
        ResultsView(baseImage: nil, result: AnalysisResult(
            predictions: SeasonPalette.all.map { SeasonPrediction(palette: $0, confidence: 0.5) },
            primary: SeasonPrediction(palette: SeasonPalette.all[0], confidence: 0.9),
            secondary: SeasonPrediction(palette: SeasonPalette.all[1], confidence: 0.86)
        ))
    }
}
