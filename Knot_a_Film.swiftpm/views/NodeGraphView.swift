//
//  NodeView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/28/25.
//

import SwiftUI
import SwiftData
import SharedWithMetal

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

    var gesture : some Gesture {
        DragGesture()
            .updating(self.$panGestureState) { gesture, state, transaction in
                
                let proposedOffset = CGSize(
                    width: (initialOffset.width + gesture.translation.width) / 1000,
                    height: (initialOffset.height + gesture.translation.height) / 1000
                )

                state = .active(translation: proposedOffset)
            }
            .onEnded { gesture in
                
                let proposedOffset = CGSize(
                    width: (initialOffset.width + gesture.translation.width) / 1000,
                    height: (initialOffset.height + gesture.translation.height) / 1000
                )
                
                self.initialOffset = proposedOffset
            }
            .simultaneously(with: MagnifyGesture()
                .updating(self.$zoomGestureState) { gesture, state, transaction in

                    state = .active(scale: gesture.magnification)
                }
                .onEnded { gesture in
                    self.initialScale *= gesture.magnification
                }
            )
    }
    
    var nodeView : some View {
        Canvas(rendersAsynchronously: true) { context, size in
            
        }
    }
    
    var body: some View {
        MetalView()
            .gesture(self.gesture)
            .onChange(of: self.initialOffset) { old, new in
                let (x, y) : (Float32, Float32) = switch self.panGestureState {
                case .inactive:
                    (Float32(self.initialOffset.width), Float32(self.initialOffset.height))
                case .active(let translation):
                    (Float32(self.initialOffset.width + translation.width), Float32(self.initialOffset.height + translation.height))
                }
                
                self.graph.graph?.renderer.screenTransform.offset = .init(x, y)
            }
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
    
    private static func colorFromText(_ text: String, saturation : Double = 0.6, brightness : Double = 0.8) -> Color {
        // Step 1: Hash the string into a UInt32
        var hash = UInt32(5381)
        for char in text.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt32(char)
        }
        
        // Step 2: Map hash to a hue (0–1)
        let hue = Double(hash % 360) / 360.0
        
        // Step 3: Create color with fixed saturation & brightness
        return Color(hue: hue, saturation: saturation, brightness: brightness)
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
