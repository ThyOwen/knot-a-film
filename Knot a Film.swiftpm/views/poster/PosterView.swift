//
//  PosterView.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/7/25.
//

import SwiftUI


struct PosterView: View {
    
    @Environment(ViewModel.self) private var viewModel

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    
    @Environment(\.shader) var shader : (_ strength : CGFloat) -> Shader

    
    @Bindable public var movie : Movie
    @Binding public var isCollapsed : Bool

    private let cornerRadius : CGFloat = 20
    
    @State private var posterTitle : String = ""
    @State private var isAuteur : Bool = false
    @State private var font : ThemeFont = .salisbury
    @State private var posterColors : [Color] = []
    @State private var colors : [Color] = []
    @State private var keyStars : [String] = []

    public static let posterTitleReplacements : [(String, String)] = [ ("by ", "\nby\n"),
                                                                       ("and ", "\nand\n"),
                                                                       (":", ":\n"),
                                                                       ("of ", "\nof\n"),
                                                                       ("Of ", "\nof\n") ]

    private var isVertical : Bool {
        self.verticalSizeClass == .regular && self.horizontalSizeClass == .compact
    }
    
    private var isLargeScreen : Bool {
        self.verticalSizeClass == .regular && self.horizontalSizeClass == .regular
    }
    
    private var shouldBeMultilineText : Bool {
        self.isVertical || self.isCollapsed
    }
    
    private var actorNamesLayout : AnyLayout {
        self.determineActorNamesLayout()
    }

    private var posterLayout : AnyLayout {
        self.isCollapsed ? AnyLayout(HStackLayout()) : AnyLayout(VStackLayout())
    }

    var shaderFunction: ShaderFunction {
        ShaderFunction(library: .bundle(.main), name: "coloredNoise")
    }
    
    var background : some View {
        
        TimelineView(.animation) { timeline in
            let x = (Float(cos(timeline.date.timeIntervalSince1970) / 5) + 1) / 2
            let y = (Float(sin(timeline.date.timeIntervalSince1970) / 5) + 1) / 2
            
            MeshGradient(width: 3, height: 3, points: [
                [0, 0], [0.5, 0], [1, 0],
                [0, 0.5], [x,y], [1, 0.5],
                [0, 1], [0.5, 1], [1, 1]
            ], colors: self.colors)
        }
        
        .clipShape(UnevenRoundedRectangle(cornerRadii: .init(topLeading: 10,
                                                             bottomLeading: self.isCollapsed ? 10 : 0,
                                                             bottomTrailing: self.isCollapsed ? 10 : 0,
                                                             topTrailing: 10)))
        .blur(radius: 1)
    }
    
    var dateBlurb : some View {
        VStack {
            if let releaseDate = self.movie.originalReleaseDate {
                let tagLine : String = if let streamingDate = self.movie.streamingReleaseDate, Studio.streamers.contains(self.movie.studio) || streamingDate == releaseDate {
                    "Steaming on \(self.movie.studio.rawValue)"
                } else {
                    "In Theaters"
                }

                Text(tagLine)
                    .font(.custom(ThemeFont.salisbury.rawValue, size: self.isLargeScreen ? 12 : 8, relativeTo: .body))
                    .foregroundStyle(.black)

                Text(releaseDate.formatted(date: .long, time: .omitted))
                    .font(.custom(ThemeFont.salisbury.rawValue, size: self.isLargeScreen ? 26 : 14, relativeTo: .body))
                    .foregroundStyle(.black)
                
            }
        }
    }

    var title : some View {
        Text(self.posterTitle)
            .font(.custom(self.font.rawValue, size: self.isVertical ? 56 : 84, relativeTo: .largeTitle))
            .foregroundStyle(Color.black)
            .minimumScaleFactor(0.01)
            .multilineTextAlignment(.center)
            .scaledToFit()
            .blur(radius: 0.75)
            .colorEffect(self.shader(0.15))
            .drawingGroup()
            .customStroke(color: Material.ultraThin, width: 1)

        
        .scenePadding(.horizontal)
        .transition(.move(edge: .bottom).combined(with: .blurReplace))
    }
    
    var top : some View {
        self.posterLayout {
            
            Spacer()
            
            if !self.isCollapsed && (self.isAuteur || !self.movie.directors.isEmpty) {
                Text("A Film by \(self.movie.directors[0].name)")
                    .font(.custom(self.font.rawValue, size: self.isVertical ? 14 : 24, relativeTo: .largeTitle))
                    .foregroundStyle(Color.black)
                    .blur(radius: 0.25)
                    .transition(.move(edge: .bottom).combined(with: .blurReplace))
                    .colorEffect(self.shader(0.15))
            }

            self.title
                
            
            if !self.isCollapsed {
                self.actorNamesLayout {
                    ForEach(self.keyStars, id: \.self) { actor in
                        Text(actor)
                            .font(.custom(ThemeFont.superDream.rawValue, size: self.isLargeScreen ? 24 : 18))
                            .foregroundStyle(Color.black)
                            .minimumScaleFactor(0.01)
                            .multilineTextAlignment(.center)
                            .lineLimit(2)
                            .blur(radius: 0.25)
                    }
                    
                }
                .scenePadding(.horizontal)
                .transition(.move(edge: .top).combined(with: .blurReplace).animation(.easeInOut(duration: 0.1)))
                
                Spacer()
                
            }
            
            if self.isCollapsed {
                Spacer()
                
                Button {
                    
                } label: {
                    ZStack {
                        Circle()
                            .fill(Material.ultraThin)
                        
                        Circle()
                            .stroke(Color(white: 0.7), lineWidth: 0.5)
                            .blur(radius: 0.5)
                            .colorEffect(self.shader(0.15))
                        
                        Image(systemName: "plus")
                            .foregroundStyle(Color.white.opacity(0.75))
                    }
                    .frame(minWidth: 50, minHeight: 50)

                }
                .padding(10)
                .transition(.blurReplace.combined(with: .move(edge: .bottom).animation(.easeInOut(duration: 0.1))))
                .buttonStyle(ShrinkingButton())
            }

        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            self.background
                .colorEffect(self.shader(0.15))
        }
    }
    
    var bottom : some View {
        VStack(spacing: 10) {
            
            let directorBlurb = !self.movie.directors.isEmpty ? "\(self.movie.directors.map { "\($0.name)'s" }.joined(separator: " and "))" : ""
            let actorBlurb = !self.movie.actors.isEmpty ? "starring \(self.movie.actors.map { $0.name }.joined(separator: "  "))" : ""
            let writerBlurb = !self.movie.writers.isEmpty ? "screenplay by \(self.movie.writers.map { $0.name }.joined(separator: " and "))" : ""
            
            
            Text("\(self.movie.productionCompany) presents \(directorBlurb) production of \(self.movie.title) \(actorBlurb) \(writerBlurb)")
            
                .multilineTextAlignment(.center)
                .font(.custom(ThemeFont.dailyNews.rawValue, size: self.isLargeScreen ? 26 : 18, relativeTo: .body))
                .foregroundStyle(.black)
                .blur(radius: 0.5)
                .fixedSize(horizontal: false, vertical: true)
            
            ZStack {
                
                self.dateBlurb
                
                HStack {
                    
                    Image(packageResource: "\(self.movie.contentRating.rawValue)", ofType: "png")
                        .resizable()
                        .scaledToFit()
                    
                    Spacer()
                    
                    Image(packageResource: "\(self.movie.studio.rawValue)", ofType: "png")
                        .resizable()
                        .scaledToFit()

                }
            }
            .blur(radius: 0.4)
            .frame(maxHeight: self.isLargeScreen ? 32 : 24)

        }
        .frame(maxWidth: 700)
        .colorEffect(self.shader(0.15))
        .zIndex(-1)
        .transition(.asymmetric(insertion: .move(edge: .bottom).combined(with: .opacity).animation(.easeInOut(duration: 0.1)),
                                removal: .move(edge: .top).combined(with: .opacity).animation(.easeInOut(duration: 0.1))))
        
    }
    
    var body: some View {
        VStack(spacing: 0) {
            self.top
            
            if !self.isCollapsed {
                self.bottom
                    .padding(10)
            }
        }
        .padding(.horizontal, 10)
        .onAppear {
            Task(priority : .userInitiated) {
                await self.loadPosterParameters()
                await self.filterKeyStars(limit: self.isVertical ? 8 : 4)
                await self.formatTitle(shouldExpandTwoWordTitles: self.isVertical)
            }
        }
        .onChange(of: self.shouldBeMultilineText) { oldValue, newValue in
            Task {
                await self.filterKeyStars(limit: newValue ? 8 : 4)
                await self.formatTitle(shouldExpandTwoWordTitles: newValue)
            }
        }
    
        //.aspectRatio(self.aspectRatio, contentMode: .fit)
    }
    
    private func determineActorNamesLayout() -> AnyLayout {
        if self.horizontalSizeClass == .compact, self.verticalSizeClass == .regular {
            return AnyLayout(VStackLayout(alignment : .center, spacing: 10))
        } else {
            return AnyLayout(HStackLayout(alignment : .center, spacing: 30))
        }
    }
    
    private func loadPosterParameters() async {
        let functionList : [@MainActor () async -> ()] = [
            self.checkAuteurism,
            self.setFont,
            self.loadColors
        ]
        
        await withDiscardingTaskGroup { group in
            for function in functionList {
                await function()
            }
        }
    }
    
    private func checkAuteurism() async {
        let writers = Set(self.movie.writers.map { $0.name })
        let directors = Set(self.movie.directors.map { $0.name })
        let actors = Set(self.movie.actors.map { $0.name })
        
        let isAuteur = (!writers.isDisjoint(with: directors) || !actors.isDisjoint(with: directors)) && !self.movie.genres.contains(.documentary)
        
        await MainActor.run {
            self.isAuteur = isAuteur
        }
    }
    
    private func setFont() async {
        
        var font : ThemeFont = .salisbury
        
        if self.movie.genres.contains(.horror) {
            font = .jasper
        } else if self.movie.genres.contains(.cult) {
            font = .robopixies
        } else if self.movie.genres.contains(.kidsFamily) || self.movie.genres.contains(.specialInterest) {
            font = .quadratum
        } else if self.movie.genres.contains(.scienceFictionFantasy) {
            font = .mainframe
        } else if self.movie.genres.contains(.musicalPerformingArts) {
            font = .superDream
        } else if self.movie.genres.contains(.western) {
            font = .worldsFinest
        } else if self.movie.genres.contains(.mysterySuspense) {
            font = .gunday
        } else if self.movie.genres.contains(.drama) || self.movie.genres.contains(.romance) {
            font = .garamond
        } else if self.movie.genres.contains(.actionAdventure) {
            font = .cipitillo
        }
        
        await MainActor.run { [font] in
            self.font = consume font
        }
    }
    
    private func loadColors() async {
        let colors = self.movie.colorsString.split(separator: ", ").map { Color(hex: $0) }
        
        await MainActor.run {
            self.colors = consume colors
        }
    }
    
    private func formatTitle(shouldExpandTwoWordTitles: Bool = false) async {

        let expandTwoWordTitles = (self.movie.title.filter { $0 == " " }.count == 1) || shouldExpandTwoWordTitles
        
        let updatedTitle : String
        if self.isCollapsed {
            updatedTitle = self.movie.title
        } else if expandTwoWordTitles {
            updatedTitle = self.movie.title.replacingOccurrences(of: " ", with: "\n")
        } else {
            var result = self.movie.title
            
            for (word, replacement) in Self.posterTitleReplacements {
                let pattern = "\\b\(word)\\b"
                if let regex = try? Regex(consume pattern) {
                    result = result.replacing(regex, with: replacement)
                }
            }
            
            updatedTitle = result
        }
        
        await MainActor.run { [updatedTitle] in
            self.posterTitle = consume updatedTitle
        }
    }
    
    private func filterKeyStars(limit : Int) async {
        let nameRegex : Regex = /\b[A-Z][a-z]+ [A-Z][a-zA-Z]+\b/
        var foundNames : Set<String> = []

        let mentionedNames = Set(self.movie.actors.map { $0.name })
        
        let matches = self.movie.info.matches(of: nameRegex)
        for match in matches {
            let name = String(match.output)
            if mentionedNames.contains(name) {
                foundNames.insert(name)
            }
        }
        
        if foundNames.count > limit {
            let rankedActors = self.movie.actors.filter {  foundNames.contains($0.name) }
                .sorted { $0.numMovies < $1.numMovies }
                .map { $0.name }[..<limit]

            let finalArray = Array(consume rankedActors)
            
            await MainActor.run {
                self.keyStars = consume finalArray
            }
        } else {
            let rankedActors = self.movie.actors
                .sorted { $0.numMovies > $1.numMovies }
                .map { $0.name}
            
            for actor in rankedActors {
                
                foundNames.insert(actor)
                
                if foundNames.count >= limit {
                    break
                }
            }
            
            let finalArray = Array(consume foundNames)
            
            await MainActor.run {
                self.keyStars = consume finalArray
            }
        }
    }
    

}

#if DEBUG
import SwiftData

struct TestView : View {
    
    @State private var collapsed : Bool = true

    @Environment(ViewModel.self) private var viewModel : ViewModel

    var body: some View {
        ZStack {
            
            ThemeColors.mainAccent.ignoresSafeArea()
            
            VStack {
                if let movies = self.viewModel.graph?.nodes, !movies.isEmpty {
                    PosterView(movie: movies[101], isCollapsed: self.$collapsed)
                        .frame(maxHeight: self.collapsed ? 100 : .infinity)
                        .padding(30)
                } else {
                    Text("There are no movies")
                }
                Button("collapse") {
                    withAnimation(.easeInOut(duration: 1)) {
                        self.collapsed.toggle()
                    }
                }
            }
        }
        .onAppear {
            self.viewModel.setup()
        }
    }
}

#Preview {
    
    @Previewable @State var viewModel : ViewModel = .init()
    
    TestView()
        .environment(viewModel)
}
#endif
