//
//  Shader.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/19/25.
//

import SwiftUICore

extension EnvironmentValues {
    private static let shaderFunction: ShaderFunction = ShaderFunction(library: .bundle(.main), name: "coloredNoise")
    
    @Entry public var shader : (CGFloat) -> Shader = { strength in
        Shader(function: Self.shaderFunction, arguments: [
            .float(strength)
        ])
    }
}


