import SwiftUI

@main struct TestApp : App {
    
    var body: some Scene {
        WindowGroup {
            TestView()
                .frame(width: 200, height: 200)
        }
    }
}
