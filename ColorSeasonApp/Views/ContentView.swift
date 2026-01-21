import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var analyzer = ColorAnalyzer()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("Upload a photo to find your color season!")
                    .font(.title2)
                    .multilineTextAlignment(.center)

                ZStack {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.secondary.opacity(0.1))
                        .frame(height: 260)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                        )

                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding()
                            .accessibilityLabel("Selected photo")
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "person.crop.square")
                                .font(.system(size: 48))
                            Text("Choose a clear face photo")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                HStack(spacing: 16) {
                    Button {
                        showCamera = true
                    } label: {
                        Label("Take Photo", systemImage: "camera")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)

                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Choose from Library", systemImage: "photo")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }

                if selectedImage != nil {
                    Button {
                        analyzer.analyze(image: selectedImage)
                    } label: {
                        Label("Analyze", systemImage: "sparkles")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }

                if analyzer.isLoading {
                    ProgressView("Analyzing…")
                }

                Spacer()
            }
            .padding()
            .navigationTitle("16-Season Analysis")
            .sheet(isPresented: $showCamera) {
                ImagePicker(sourceType: .camera) { image in
                    selectedImage = image
                }
            }
            .onChange(of: selectedItem) { _, newItem in
                guard let newItem else { return }
                Task {
                    if let data = try? await newItem.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        await MainActor.run {
                            selectedImage = image
                        }
                    }
                }
            }
            .navigationDestination(item: $analyzer.result) { result in
                ResultsView(
                    baseImage: selectedImage,
                    result: result
                )
            }
        }
    }
}

#Preview {
    ContentView()
}
