//
//  ThemeColors.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/9/25.
//

import SwiftUI



public struct ThemeColors {
    public static let mainAccent : Color = .init(#colorLiteral(red: 0.9716802207, green: 0.9460327575, blue: 0.8938176578, alpha: 1))
    public static let secondAccent : Color = .init(#colorLiteral(red: 0.8208606243, green: 0.7911450267, blue: 0.7266902328, alpha: 1))
    
    public static let rainbowPink : Color = .init(hex: "CC4B60")
    public static let rainbowRed : Color = .init(hex: "D17081")
    public static let rainbowYellow : Color = .init(hex: "ECB543")
    public static let rainbowCyan : Color = .init(hex: "83B3C0")
    public static let rainbowBlue : Color = .init(hex: "5D6EAA")

}


public extension Color {
    init(hex: Substring) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue:  Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}
