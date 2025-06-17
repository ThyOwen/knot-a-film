//
//  ChartsView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/31/25.
//

import SwiftUI
import Charts

struct ChartsView: View {
    
    
    
    var chartGenres : some View {
        Chart {
            
            BarMark(x: .value("Type", "bird"),
                    y: .value("Population", 1))
            .foregroundStyle(.pink)
            

            BarMark(x: .value("Type", "dog"),
                    y: .value("Population", 2))
            .foregroundStyle(.green)

            BarMark(x: .value("Type", "cat"),
                    y: .value("Population", 3))
            .foregroundStyle(.blue)
        }
    }
    
    var body: some View {
        self.chartGenres
    }
}

#Preview {
    ChartsView()
}
