import SwiftUI

@main
struct KnotAFilm : App {
    
    @State private var viewModel : ViewModel = .init()
    
    init() {
        FontManager.registerFonts()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
            //TestView()
                .environment(self.viewModel)
        }
    }
}
