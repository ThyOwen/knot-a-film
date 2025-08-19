//
//  TestView.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/18/25.
//

import SwiftUI

struct TestView: View {
    
    @State private var graph : Graph
    
    init() {
        
        let x : [Float32] = [0,1,2,3,4,5,6,7]
        let y : [Float32] = [4,5,6,7,8,9,10,11]
        
        self.graph = .init(x: x, y: y)
    }
    
    var body: some View {
        ZStack {
            
            let maxX = self.graph.x.max() ?? 0
            let maxY = self.graph.y.max() ?? 0
            
            ForEach(0..<self.graph.numNodes.intValue) { idx in
                
                let x = (CGFloat(self.graph.x[idx]) / CGFloat(maxX)) * 200
                let y = (CGFloat(self.graph.y[idx]) / CGFloat(maxY)) * 200
                
                return Circle()
                    .frame(width: 10, height: 10)
                    .position(x: x, y: y)
            }
        }
        .onAppear {
            
            self.graph.buildGraph()
        }
        .task {
            while true {
                self.graph.run()
                try? await Task.sleep(for: .seconds(0.1))
            }
        }
    }
    
    
}
