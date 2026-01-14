//
//  NodeView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/28/25.
//

import SwiftUI
import SwiftData
import SharedWithMetal
import MetalKit

extension Bool {
    var intValue: Int {
        return self ? 1 : 0
    }
}

enum ZoomGestureState {
    case inactive
    case active(scale : CGFloat)
}

enum PanGestureState {
    case inactive
    case active(translation: CGSize)
}

struct NodeGraphView: View {
    
    @Environment(ViewModel.self) private var graph
    
    private let lineColor : Color = .init(white: 0.3).opacity(0.6)

    @State private var initialOffset : CGSize = .zero
    @State private var initialScale : CGFloat = 1
    @GestureState private var panGestureState : PanGestureState = .inactive
    @GestureState private var zoomGestureState : ZoomGestureState = .inactive
    
    @FocusState private var isFocused: Bool
    
    var gesture : some Gesture {
        MagnifyGesture()
            .updating(self.$zoomGestureState) { gesture, state, transaction in

                state = .active(scale: gesture.magnification)
            }
            .onEnded { gesture in
                self.initialScale *= gesture.magnification
            }
    }

    var overlay: some View {
        Canvas { context, size in
            guard let positions = self.graph.graph?.renderer.bodyPositions else {
                return
            }
            
            for (index, position) in positions {
                let x = (position.x + 1.0) * size.width / 2.0
                let y = (1.0 - position.y) * size.height / 2.0
                
                let point = CGPoint(x: x, y: y)
                
                context.fill(
                    Circle().path(in: CGRect(x: point.x - 12, y: point.y - 12, width: 24, height: 24)),
                    with: .color(.black.opacity(0.7))
                )
                
                let text = Text("\(index)")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white)
                
                context.draw(text, at: point)
            }
        }
    }
    
    var body: some View {
        MetalView()
            .overlay {
                self.overlay
            }
            .focusable()
            .focused($isFocused)
            .onAppear {
                self.isFocused = true
            }
            .onKeyPress(.downArrow) {
                self.graph.graph?.renderer.screenTransform.offset.y += 0.1
                return .handled
            }
            .onKeyPress(.upArrow) {
                self.graph.graph?.renderer.screenTransform.offset.y -= 0.1
                return .handled
            }
            .onKeyPress(.leftArrow) {
                self.graph.graph?.renderer.screenTransform.offset.x += 0.1
                return .handled
            }
            .onKeyPress(.rightArrow) {
                self.graph.graph?.renderer.screenTransform.offset.x -= 0.1
                return .handled
            }
            .onKeyPress("=") { // "+" (Shift + "=")
                self.graph.graph?.renderer.screenTransform.scale *= 1.1
                return .handled
            }
            .onKeyPress("-") {
                self.graph.graph?.renderer.screenTransform.scale /= 1.1
                return .handled
            }
            .gesture(self.gesture)
            .onChange(of: self.initialScale) { oldValue, newValue in
                let scale : Float32 = switch self.zoomGestureState {
                case .inactive:
                    Float32(self.initialScale)
                case .active(let scale):
                    Float32(self.initialScale * scale)
                }
                
                self.graph.graph?.renderer.screenTransform.scale = .init(scale, scale)

            }

    }
}

#if DEBUG

#Preview {
    @Previewable @State var viewModel : ViewModel = .init()
    ZStack {
        ThemeColors.mainAccent.ignoresSafeArea()
        
        if let graph = viewModel.graph {
            NodeGraphView()
                .environment(graph)
        }
        
    }
}

#endif
