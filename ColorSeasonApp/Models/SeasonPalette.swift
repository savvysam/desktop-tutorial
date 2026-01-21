import SwiftUI

struct SeasonPalette: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let overlayColor: Color
    let swatchColors: [Color]

    static let all: [SeasonPalette] = [
        SeasonPalette(name: "Light Spring", overlayColor: Color(hex: "#F7C8A2"), swatchColors: [
            Color(hex: "#F7C8A2"), Color(hex: "#FAD7B6"), Color(hex: "#FDE6C8"), Color(hex: "#FFE4D4")
        ]),
        SeasonPalette(name: "Warm Spring", overlayColor: Color(hex: "#F4B183"), swatchColors: [
            Color(hex: "#F4B183"), Color(hex: "#F7A45A"), Color(hex: "#F9C784"), Color(hex: "#FFE0A3")
        ]),
        SeasonPalette(name: "Bright Spring", overlayColor: Color(hex: "#FF9F7D"), swatchColors: [
            Color(hex: "#FF9F7D"), Color(hex: "#FF6F61"), Color(hex: "#FFD166"), Color(hex: "#FFB347")
        ]),
        SeasonPalette(name: "Light Summer", overlayColor: Color(hex: "#C8D8F0"), swatchColors: [
            Color(hex: "#C8D8F0"), Color(hex: "#D9E2F2"), Color(hex: "#E6EEF7"), Color(hex: "#B8C4E6")
        ]),
        SeasonPalette(name: "Cool Summer", overlayColor: Color(hex: "#A9C0E8"), swatchColors: [
            Color(hex: "#A9C0E8"), Color(hex: "#C2D3F2"), Color(hex: "#B0C4DE"), Color(hex: "#9FB7D9")
        ]),
        SeasonPalette(name: "Soft Summer", overlayColor: Color(hex: "#B6BFC8"), swatchColors: [
            Color(hex: "#B6BFC8"), Color(hex: "#C6CCD2"), Color(hex: "#AEB4BD"), Color(hex: "#D6DAE0")
        ]),
        SeasonPalette(name: "Soft Autumn", overlayColor: Color(hex: "#C2A48C"), swatchColors: [
            Color(hex: "#C2A48C"), Color(hex: "#D1B59C"), Color(hex: "#B99575"), Color(hex: "#E2C2A1")
        ]),
        SeasonPalette(name: "Warm Autumn", overlayColor: Color(hex: "#C9834C"), swatchColors: [
            Color(hex: "#C9834C"), Color(hex: "#D9935C"), Color(hex: "#B8723B"), Color(hex: "#E0A06A")
        ]),
        SeasonPalette(name: "Deep Autumn", overlayColor: Color(hex: "#8C5E3C"), swatchColors: [
            Color(hex: "#8C5E3C"), Color(hex: "#A4663F"), Color(hex: "#6D432C"), Color(hex: "#B57446")
        ]),
        SeasonPalette(name: "Bright Winter", overlayColor: Color(hex: "#5E6AD2"), swatchColors: [
            Color(hex: "#5E6AD2"), Color(hex: "#2D35A7"), Color(hex: "#9B5DE5"), Color(hex: "#4EA8DE")
        ]),
        SeasonPalette(name: "Cool Winter", overlayColor: Color(hex: "#4B6CB7"), swatchColors: [
            Color(hex: "#4B6CB7"), Color(hex: "#3A5BA0"), Color(hex: "#6B8DD6"), Color(hex: "#2F4B7C")
        ]),
        SeasonPalette(name: "Deep Winter", overlayColor: Color(hex: "#2B2D42"), swatchColors: [
            Color(hex: "#2B2D42"), Color(hex: "#3D405B"), Color(hex: "#1B1D2E"), Color(hex: "#4A4E69")
        ]),
        SeasonPalette(name: "Light Autumn", overlayColor: Color(hex: "#D9B08C"), swatchColors: [
            Color(hex: "#D9B08C"), Color(hex: "#E2C1A2"), Color(hex: "#C9A57E"), Color(hex: "#F0D5B5")
        ]),
        SeasonPalette(name: "Warm Summer", overlayColor: Color(hex: "#8BB1B1"), swatchColors: [
            Color(hex: "#8BB1B1"), Color(hex: "#9BBABA"), Color(hex: "#7EA3A3"), Color(hex: "#A8C5C5")
        ]),
        SeasonPalette(name: "Bright Autumn", overlayColor: Color(hex: "#E08D5D"), swatchColors: [
            Color(hex: "#E08D5D"), Color(hex: "#F09F6C"), Color(hex: "#CC7A4A"), Color(hex: "#F2B082")
        ]),
        SeasonPalette(name: "True Winter", overlayColor: Color(hex: "#2F3E9E"), swatchColors: [
            Color(hex: "#2F3E9E"), Color(hex: "#22307A"), Color(hex: "#1B2457"), Color(hex: "#4052C4")
        ])
    ]
}

struct SeasonPrediction: Identifiable, Hashable {
    let id = UUID()
    let palette: SeasonPalette
    let confidence: Double
}

struct AnalysisResult: Identifiable {
    let id = UUID()
    let predictions: [SeasonPrediction]
    let primary: SeasonPrediction
    let secondary: SeasonPrediction?

    var explanation: String {
        if let secondary {
            return "You are between \(primary.palette.name) and \(secondary.palette.name). Your features appear balanced between the two palettes, so we recommend leaning toward \(primary.palette.name) while also exploring \(secondary.palette.name) shades."
        }
        return "Your coloring aligns with \(primary.palette.name). This palette best matches your undertone, contrast, and overall depth based on the model's confidence scores."
    }
}

extension AnalysisResult: Equatable {
    static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        lhs.id == rhs.id
    }
}

extension Color {
    init(hex: String) {
        let hexValue = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int = UInt64()
        Scanner(string: hexValue).scanHexInt64(&int)
        let r, g, b: UInt64
        switch hexValue.count {
        case 3:
            (r, g, b) = ((int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        default:
            (r, g, b) = (int >> 16, int >> 8 & 0xFF, int & 0xFF)
        }
        self.init(red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255)
    }
}
