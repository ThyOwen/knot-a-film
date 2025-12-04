//
//  MetalView.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/18/25.
//

import SwiftUI
import MetalKit

struct MetalView : UIViewRepresentable {
        
    @Environment(GraphManager.self) private var graphManager
    
    func makeCoordinator() -> GraphRenderer  {
        self.graphManager.renderer
    }
    
    func makeUIView(context: Context) -> MTKView {
        
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 120
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            mtkView.device = metalDevice
        }
        
        mtkView.framebufferOnly = false
        mtkView.drawableSize = mtkView.frame.size
        mtkView.isPaused = false            // don’t pause after a single draw
        mtkView.enableSetNeedsDisplay = false
        mtkView.depthStencilPixelFormat = .invalid

        
        mtkView.isOpaque = true
        
        
        return mtkView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        
    }
}

