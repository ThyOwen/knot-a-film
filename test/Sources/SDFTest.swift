//
//  SDFTest.swift
//  MetalTest
//
//  Created by Owen O'Malley on 12/16/25.
//

import SwiftUI

struct SDFView: View {

    
    static let shaderFunction: ShaderFunction = .init(library: .bundle(.main), name: "smoothUnion")

    
    public var shader : (CGFloat) -> Shader = { strength in
        Shader(function: Self.shaderFunction, arguments: [
            .float2(CGPoint(x: 0.5, y: 0.5)),
            .float(strength)
        ])
    }

    var body: some View {
        Rectangle()
            .colorEffect(shader(0.0))
    }
}
