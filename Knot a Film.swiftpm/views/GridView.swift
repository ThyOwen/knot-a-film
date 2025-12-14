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

