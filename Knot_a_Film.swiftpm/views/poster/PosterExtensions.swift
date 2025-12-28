//
//  PosterExtensions.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 6/7/25.
//

import SwiftUI

extension Image {
    init(packageResource name: String, ofType type: String) {
        #if canImport(UIKit)
        guard let path = Bundle.main.path(forResource: name, ofType: type),
              let image = UIImage(contentsOfFile: path) else {
            self.init(name)
            return
        }
        self.init(uiImage: image)
        #elseif canImport(AppKit)
        guard let path = Bundle.main.path(forResource: name, ofType: type),
              let image = NSImage(contentsOfFile: path) else {
            self.init(name)
            return
        }
        self.init(nsImage: image)
        #else
        self.init(name)
        #endif
    }
}

struct ShrinkingButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct StrokeModifier<F>: ViewModifier where F : ShapeStyle {
    private let id = UUID()
    var strokeSize : CGFloat
    var strokeColor : F
    
    @Environment(\.shader) var shader : (_ strength : CGFloat) -> Shader

    func body(content: Content) -> some View {
        if strokeSize > 0 {
            appliedStrokeBackground(content: content)
        } else {
            content
        }
    }

    private func appliedStrokeBackground(content: Content) -> some View {
         content
             .padding(strokeSize*2)
             .background(
                 Rectangle()
                    .fill(self.strokeColor)
                    .mask(alignment: .center) {
                        mask(content: content)
                    }
                    .blur(radius: 2)
                    .colorEffect(self.shader(0.15))
             )
        
    }

    func mask(content: Content) -> some View {
        Canvas { context, size in
            context.addFilter(.alphaThreshold(min: 0.01))
            if let resolvedView = context.resolveSymbol(id: id) {
                context.draw(resolvedView, at: .init(x: size.width/2, y: size.height/2))
            }
        } symbols: {
            content
                .tag(id)
                .blur(radius: strokeSize)
        }
    }
}



extension View {
    func customStroke<F : ShapeStyle>(color: F, width: CGFloat) -> some View {
        self.modifier(StrokeModifier(strokeSize: width, strokeColor: color))
    }
}

#Preview {
    Text("asdfasdf")
        .bold()
        .foregroundStyle(.brown)
        .customStroke(color: Material.ultraThin, width: 1)
    
}
