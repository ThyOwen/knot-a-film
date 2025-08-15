//
//  SwiftUIView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/19/25.
//

import SwiftUI

struct SearchBarView: View {
    
    @Environment(ViewModel.self) private var viewModel
    
    @State private var searchText: String = ""
    @FocusState private var isFocused : Bool
    
    @Environment(\.shader) var shader : (_ strength : CGFloat) -> Shader
    
    
    var searchResults: some View {
        RoundedRectangle(cornerRadius: 20)
            .fill(Material.ultraThin)
            .background {
                RoundedRectangle(cornerRadius: 50)
                    .fill(Color.black.opacity(0.5)) // Shadow inside
                    .blur(radius: 30)
                    .offset(x: 50, y: 50)
                
                
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.black.opacity(0.5), style: .init(lineWidth: 1))
                    .blur(radius: 0.5)
                    .colorEffect(self.shader(0.2))
            }
            .overlay {
                self.searchResultsList
                    
            }
        //.aspectRatio(self.aspectRatio, contentMode: .fit)
    }
    
    var searchResultsList : some View {
        ScrollView(showsIndicators: false) {
            LazyVStack(spacing: 0) {
                if let searchMovies = self.viewModel.searchEngine?.activeSearchMovies {
                    ForEach(searchMovies) { movie in
                        PosterView(movie: movie, isCollapsed: .constant(true))
                            .frame(height: 100)
                            .scrollTransition { content, phase in
                                content
                                    .opacity(phase.isIdentity ? 1 : 0)
                                    .scaleEffect(phase.isIdentity ? 1 : 0.75)
                                    .blur(radius: phase.isIdentity ? 0 : 10)
                            }
                            .padding(.init(top: 10, leading: 0, bottom: 0, trailing: 0))
                    }
                }
            }
            .scrollTargetLayout()
        }
        .scrollTargetBehavior(.viewAligned)
    }
    
    var searchBar : some View {
        TextField("What would you like to see?", text: self.$searchText)
            .font(.custom(ThemeFont.salisbury.rawValue, size: 16, relativeTo: .subheadline))
            .scenePadding(.horizontal)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background {
                let gradient = LinearGradient(colors: [.gray.opacity(0.5),
                                                       .black.opacity(0.5)],
                                              startPoint: .topLeading,
                                              endPoint: .bottomTrailing)
                
                Capsule()
                    .stroke(gradient, style: .init(lineWidth: 0.5))
                    .blur(radius: 0.25)
                    .colorEffect(self.shader(0.25))
                    .drawingGroup()
            }
            .onSubmit(of: .text) {
                self.viewModel.search(using: self.searchText.lowercased())
                self.searchText = ""
            }
            .onChange(of: self.searchText) { oldValue, newValue in
                self.viewModel.search(using: self.searchText.lowercased())
            }
    }

    
    var bar: some View {
        Capsule()
            .fill(Material.ultraThin)
            .frame(maxWidth: .infinity, maxHeight: 50)
            .background {
                Capsule()
                    .fill(Color.black.opacity(0.2))
                    .drawingGroup()
                    .blur(radius: 25)
                    .offset(x: 30, y: 30)
                    
                   
                 
            }
            .overlay {
                self.searchBar
            }

    }
    
    var body: some View {
        VStack {
            if !self.searchText.isEmpty {
                self.searchResults
            }
            self.bar
        }
    }
}

#Preview {
    
    @Previewable @State var viewModel : ViewModel = .init()
    
    ZStack {
        ThemeColors.mainAccent.ignoresSafeArea()
        SearchBarView()
            .scenePadding()
    }.environment(viewModel)
}
