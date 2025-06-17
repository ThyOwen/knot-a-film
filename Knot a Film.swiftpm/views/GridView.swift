//
//  GridView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/1/25.
//

import SwiftUI

struct ChartGrid: Shape {
    var columns : Int
    var offset : CGSize  // Use this directly instead of separate offsetX and offsetY
    
    var animatableData: AnimatablePair<CGFloat, CGFloat> {
        get { AnimatablePair(self.offset.width, self.offset.height) }
        set {
            self.offset = CGSize(width: newValue.first, height: newValue.second)
        }
    }
    
    func path(in rect: CGRect) -> Path {
        Path { path in
            
            let dim = rect.width / CGFloat(self.columns)
            
            let numRows = Int(floor(rect.height / dim))
            
            let offsetX = self.offset.width - (dim * floor(self.offset.width / dim))
            let offsetY = self.offset.height - (dim * floor(self.offset.height / dim))
            
            // Draw horizontal lines
            for row in 0...numRows {
                let y = (CGFloat(row) * dim) + offsetY
                if y <= rect.height {
                    path.move(to: CGPoint(x: rect.minX, y: y))
                    path.addLine(to: CGPoint(x: rect.maxX, y: y))
                }
            }
            
            // Draw vertical lines
            for column in 0...(self.columns - 1) {
                let x = (CGFloat(column) * dim) + offsetX
                if x <= rect.width {
                    path.move(to: CGPoint(x: x, y: rect.minY))
                    path.addLine(to: CGPoint(x: x, y: rect.maxY))
                }
            }
            
        }
    }
}

public enum MagnificationGestureState {
    case active(_ amount : CGFloat, _ zoomCenter : UnitPoint)
    case inactive
    
    public var isActive : Bool {
        switch self {
        case .active( _, _):
            return true
        case .inactive:
            return false
        }
    }
}

public enum DragGestureState {
    case active(_ amount : CGSize)
    case inactive
    
    public var isActive : Bool {
        switch self {
        case .active( _):
            return true
        case .inactive:
            return false
        }
    }
}


struct GridView<V : View> : View {
    
    @Environment(GraphManager.self) private var graph
    
    @State private var zoom : CGFloat = 1.0
    @State private var translate : UnitPoint = .center
    
    @GestureState private var magnificationGestureState : MagnificationGestureState = .inactive
    @GestureState private var dragGestureState : DragGestureState = .inactive
    
    @ViewBuilder public var view : V
    
    private var activeZoom : CGFloat {
        switch self.magnificationGestureState {
        case .active(let value, _):
            return value * self.zoom
        case .inactive:
            return self.zoom
        }
    }
    
    private var activeZoomCenter : UnitPoint {
        switch self.magnificationGestureState {
        case .active(_, let zoomCenter):
            return zoomCenter
        case .inactive:
            return UnitPoint.center
        }
    }
    
    private var activeTranslate : UnitPoint {
        switch self.dragGestureState {
        case .active(let value):
            let x = (value.width / self.graph.activeBounds.width) + self.translate.x
            let y = (value.height / self.graph.activeBounds.height) + self.translate.y

            return UnitPoint(x: x, y: y)
        case .inactive:
            return self.translate
        }
    }
    
    private var activeScaledTranslate : CGSize {
        switch self.dragGestureState {
        case .active(let value):
            let width = value.width + (self.translate.x * self.graph.activeBounds.width)
            let height = value.height + (self.translate.y * self.graph.activeBounds.height)

            return CGSize(width: width, height: height)
        case .inactive:
            
            let width = self.translate.x * self.graph.activeBounds.width
            let height = self.translate.y * self.graph.activeBounds.height
            
            return CGSize(width: width, height: height)
        }
    }
       
    var zoomGesture : some Gesture {
        MagnifyGesture(minimumScaleDelta: 0)
            .updating(self.$magnificationGestureState) { value, state, _ in
                state = .active(value.magnification, value.startAnchor)
            }
            .onChanged { value in
                self.graph.userZoomCenter = self.activeZoomCenter
                self.graph.userZoom = self.activeZoom
            }
            .onEnded { value in
                self.zoom *= value.magnification
            }
    }
    
    var dragGesture : some Gesture {
        DragGesture(minimumDistance: 0)
            .updating($dragGestureState) { value, state, _ in
                state = .active(value.translation)
            }
            .onChanged { _ in
                self.graph.userTranslate = self.activeTranslate
            }
            .onEnded { value in
                self.translate.x += (value.translation.width / self.graph.activeBounds.width)
                self.translate.y += (value.translation.height / self.graph.activeBounds.height)
            }
    }
    
    var gesture : some Gesture {
        SimultaneousGesture(self.dragGesture, self.zoomGesture)
    }
    
    var body: some View  {
        ZStack {
            Rectangle()
                .stroke(Color.init(white: 0.3), lineWidth: 0.25)
                .blur(radius: 0.25)
            ChartGrid(columns: 16, offset: self.activeScaledTranslate)
                .stroke(Color.init(white: 0.5), lineWidth: 0.125)
                .blur(radius: 0.25)

            self.view
            /*
            ChartGrid(columns: 8, offset: self.activeScaledTranslate, zoom: self.graph.userZoom)
                .stroke(Color.init(white: 0.4), lineWidth: 0.25)
                .blur(radius: 0.5)
            ChartGrid(columns: 4, offset: self.activeScaledTranslate, zoom: self.graph.userZoom)
                .stroke(Color.init(white: 0.3), lineWidth: 0.5)
                .blur(radius: 0.75)
            */
        }
        .background(ThemeColors.mainAccent)
        .gesture(self.gesture)
    }
}

#Preview {
    @Previewable @State var viewModel : ViewModel = .init()
    ZStack {
        if let graph = viewModel.graph {
            ThemeColors.mainAccent.ignoresSafeArea()
            GridView {
                Circle()
            }.environment(graph)
        }
    }
}
