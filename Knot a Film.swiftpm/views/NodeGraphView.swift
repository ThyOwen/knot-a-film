//
//  NodeView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/28/25.
//

import SwiftUI
import SwiftData

extension Bool {
    var intValue: Int {
        return self ? 1 : 0
    }
}

struct MovieConnection: Shape {
    var from: CGPoint
    var to: CGPoint
    
    // Define animatable data as a pair of points
    var animatableData: AnimatablePair<CGPoint.AnimatableData, CGPoint.AnimatableData> {
        get {
            AnimatablePair(
                self.from.animatableData,
                self.to.animatableData
            )
        }
        set {
            self.from = CGPoint(x: newValue.first.first, y: newValue.first.second)
            self.to = CGPoint(x: newValue.second.first, y: newValue.second.second)
        }
    }
    
    init(from: consuming CGPoint, to: consuming CGPoint) {
        self.from = from
        self.to = to
    }
    
    func path(in rect: CGRect) -> Path {
        Path { path in
            let fromX = self.from.x
            let fromY = self.from.y
            let toX = self.to.x
            let toY = self.to.y
            
            path.move(to: .init(x: fromX, y: fromY))
            path.addLine(to: .init(x: toX, y: toY))
        }
    }
}

struct NodeGraphView: View {
    
    @Environment(GraphManager.self) private var graph
    
    private let lineColor : Color = .init(white: 0.3).opacity(0.6)
    
    var edges : some View {
        Canvas { context, cgSize in
            
            self.graph.activeBounds = cgSize
            
            for edge in self.graph.edges {
                let aIdx = edge.aNodePositionIndex
                let bIdx = edge.bNodePositionIndex
                
                let from = CGPoint(x: self.graph.positionsX[aIdx], y: self.graph.positionsY[aIdx])
                let to = CGPoint(x: self.graph.positionsX[bIdx], y: self.graph.positionsY[bIdx])
                
                let lineWidth = CGFloat(edge.writers.intValue + edge.directors.intValue + edge.actors.intValue)
                
                let path = Path { path in
                    path.move(to: from)
                    path.addLine(to: to)
                }
                context.stroke(path, with: .color(self.lineColor), style: .init(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
            }
        }.blur(radius: 0.25)
    }
    
    var nodes : some View {
        ZStack {
            ForEach(self.graph.nodes) { node in
                if self.graph.positionsX.isEmpty || self.graph.positionsY.isEmpty {
                    EmptyView()
                } else {
                    
                    let idx = (node.positionIndex ?? 0)
                    
                    let positionX = self.graph.positionsX[idx]
                    let positionY = self.graph.positionsY[idx]

                    let size : CGFloat = sqrt(CGFloat(node.movieConnections.count + 1) * 10) + 2

                    ZStack {
                        Circle()
                            .stroke(Color.init(white: 0.2).opacity(0.7), lineWidth: 3)
                            .fill(Color.init(white: 0.4).opacity(0.8))
                            .frame(width: size, height: size)
                            .position(x: positionX, y: positionY)
                            .blur(radius: 1)
                        
                        Text(node.title)
                            .font(.custom("Comica", size: 16))
                            .foregroundStyle(Color.init(white: 0.1))
                            .position(x: positionX, y: positionY)
                         
                    }

                }
            }
        }
    }
    
    /*
    var edges : some View {
        ZStack {
            self.graph.unNormalizedPositionsX.withUnsafeMutablePointerToElements { positionsX in
                self.graph.unNormalizedPositionsY.withUnsafeMutablePointerToElements { positionsY in
                    ForEach(self.graph.edges) { edge in
                        let aIdx = edge.aNodePositionIndex
                        let bIdx = edge.bNodePositionIndex
                        
                        let fromX = positionsX[aIdx]
                        let fromY = positionsY[aIdx]
                        let toX = positionsX[bIdx]
                        let toY = positionsY[bIdx]
                        
                        let from = CGPoint(x: CGFloat(fromX), y: CGFloat(fromY))
                        let to = CGPoint(x: CGFloat(toX), y: CGFloat(toY))
                        
                        let lineWidth = CGFloat(edge.writers.intValue + edge.directors.intValue + edge.actors.intValue)

                        
                        return MovieConnection(from: from, to: to)
                            .stroke(self.lineColor, style: .init(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
                            .blur(radius: 1)

                    }
                }
                }
            }

    }
    
    var nodes : some View {
        ZStack {
            ForEach(self.graph.nodes) { node in
                if self.graph.positionsX.isEmpty || self.graph.positionsY.isEmpty {
                    EmptyView()
                } else {
                    
                    let idx = (node.positionIndex ?? 0)
                    
                    let positionX = CGFloat(self.graph.positionsX[idx])
                    let positionY = CGFloat(self.graph.positionsY[idx])
                    
                    let size : CGFloat = sqrt(CGFloat(node.movieConnections.count + 1) * 10) + 2
                    
                    if !(0.1..<self.graph.activeBounds.width).contains(positionX) || !(0.1..<self.graph.activeBounds.height).contains(positionY) {
                        EmptyView()
                    } else {
                        ZStack {
                            Circle()
                                .stroke(Color.init(white: 0.25).opacity(0.7), lineWidth: 3)
                                .fill(Color.init(white: 0.3).opacity(0.8))
                                .frame(width: size, height: size)
                                .position(x: positionX, y: positionY)
                                .blur(radius: 1)
                            Text(node.title)
                                .font(.custom("Comica", size: 16))
                                .foregroundStyle(Color.init(white: 0.1))
                                .position(x: positionX, y: positionY + 16)
                        }

                    }
                }
            }
        }
    }
     */
    var body: some View {
        GeometryReader { proxy in
            self.graph.activeBounds = proxy.size
            return ZStack(alignment: .center) {
                self.edges
                self.nodes
            }
        }.drawingGroup(opaque: true)
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
