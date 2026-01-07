//
//  MetalView.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/18/25.
//

import SwiftUI
import MetalKit


struct MetalView : NSViewRepresentable {
        
    func makeCoordinator() -> GraphRenderer  {
        GraphRenderer()
    }
    
    func makeNSView(context: Context) -> MTKView {
        
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 120
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            mtkView.device = metalDevice
        }
        
        mtkView.framebufferOnly = false
        mtkView.drawableSize = mtkView.frame.size
        mtkView.isPaused = false            // don’t pause after a single draw
        mtkView.enableSetNeedsDisplay = false // don’t wait for setNeedsDisplay()
        
        return mtkView
    }
    
    func updateNSView(_ uiView: MTKView, context: Context) {
        
    }
}

