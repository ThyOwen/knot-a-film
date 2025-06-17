//
//  ContentView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/17/25.
//

import SwiftUI
import SwiftData

struct ContentView: View {
    
    @Environment(ViewModel.self) private var viewModel: ViewModel
    
    @Environment(\.shader) var shader : (_ strength : CGFloat) -> Shader
    
    var searchBar : some View {
        VStack {
            Spacer()
            
            SearchBarView()
                .scenePadding()
        }
    }
    
    @ViewBuilder var graph : some View {
        if let graph = self.viewModel.graph {
            GridView {
                NodeGraphView()
                    .colorEffect(self.shader(0.2))
            }
            .environment(graph)
            .padding(20)
            .drawingGroup()
        }
    }
    
    var body: some View {
        ZStack {
            self.graph
                
            self.searchBar

        }
        
        .background(ThemeColors.mainAccent, ignoresSafeAreaEdges: .all)
        .onAppear {
            self.viewModel.setup()
        }
        
    }
}

#Preview {
    ContentView()
}
